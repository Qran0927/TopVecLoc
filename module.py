# Copyright (c) Meta Platforms, Inc. and affiliates.

from pathlib import Path

import pytorch_lightning as pl
import torch
from omegaconf import DictConfig, OmegaConf, open_dict
from torchmetrics import MeanMetric, MetricCollection

from logger import logger
from models import get_model


GRAPH_PARAMETER_MARKERS = (
    "map_encoder.graph_encoder.",
    "map_encoder.graph_proj.",
    "map_encoder.fusion_gate",
)

BASELINE_BATCH_NORM_PREFIXES = (
    "image_encoder.",
    "map_encoder.cnn_encoder.",
)


def freeze_baseline_batch_normalization(model):
    """Keep pretrained CNN normalization buffers fixed while training the GNN."""
    for name, child in model.named_modules():
        if (
            isinstance(child, torch.nn.modules.batchnorm._BatchNorm)
            and name.startswith(BASELINE_BATCH_NORM_PREFIXES)
        ):
            child.eval()


def build_staged_optimizer(
    named_parameters,
    lr,
    baseline_lr_scale=0.1,
    baseline_warmup_steps=0,
):
    """Build disjoint baseline/GNN parameter groups with a baseline warm-up."""
    if not 0 < baseline_lr_scale <= 1:
        raise ValueError("baseline_lr_scale must be in (0, 1]")
    if baseline_warmup_steps < 0:
        raise ValueError("baseline_warmup_steps must be non-negative")

    baseline_parameters = []
    graph_parameters = []
    for name, parameter in named_parameters:
        if not parameter.requires_grad:
            continue
        if any(marker in name for marker in GRAPH_PARAMETER_MARKERS):
            graph_parameters.append(parameter)
        else:
            baseline_parameters.append(parameter)

    if not baseline_parameters or not graph_parameters:
        raise ValueError(
            "staged optimizer requires non-empty baseline and graph parameter groups"
        )

    optimizer = torch.optim.Adam(
        [
            {
                "params": baseline_parameters,
                "lr": lr * baseline_lr_scale,
                "name": "baseline",
            },
            {"params": graph_parameters, "lr": lr, "name": "graph"},
        ]
    )

    def baseline_schedule(step):
        return 0.0 if step < baseline_warmup_steps else 1.0

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=(baseline_schedule, lambda step: 1.0),
    )
    return optimizer, scheduler


class AverageKeyMeter(MeanMetric):
    def __init__(self, key, *args, **kwargs):
        self.key = key
        super().__init__(*args, **kwargs)

    def update(self, dict):
        value = dict[self.key]
        value = value[torch.isfinite(value)]
        return super().update(value)


class GenericModule(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        name = cfg.model.get("name")
        name = "removed_baseline" if name in ("localizer_bev_depth", None) else name
        self.model = get_model(name)(cfg.model)
        self.cfg = cfg
        self.save_hyperparameters(cfg)



        self.metrics_val = MetricCollection(self.model.metrics(), prefix="val/")
        self.losses_val = None  # we do not know the loss keys in advance

        # self.citys = self.cfg.data.val_citys
        # for i  in range(len(self.citys)):
        #     city=self.citys[i]
        #     setattr(self, "metric_vals_{}".format(i), MetricCollection(self.model.metrics(), prefix="val_{}/".format(city)))
        # self.losse_vals = [None for city in self.cfg.data.val_citys]


    def forward(self, batch):
        return self.model(batch)

    def training_step(self, batch):
        # Lightning may restore train mode after BaseModel.train() has frozen BN.
        # Enforce the baseline invariant immediately before every forward pass.
        freeze_baseline_batch_normalization(self.model)
        pred = self(batch)
        losses = self.model.loss(pred, batch)
        self.log_dict(
            {f"loss/{k}/train": v.mean() for k, v in losses.items()},
            prog_bar=True,
            rank_zero_only=True,
        )
        return losses["total"].mean()

    # def validation_step(self, batch, batch_idx,dataloader_idx):
    #     city=self.citys[dataloader_idx]
    #
    #     pred = self(batch)
    #     losses = self.model.loss(pred, batch)
    #
    #     if hasattr(self,"losse_val_{}".format(dataloader_idx)) is False:
    #         setattr(self,"losse_val_{}".format(dataloader_idx),MetricCollection(
    #             {k: AverageKeyMeter(k).to(self.device) for k in losses},
    #             prefix="loss_{}/".format(city),
    #             postfix="/val_{}".format(city),
    #         ))
    #
    #     # print(pred, batch)
    #     getattr(self,"metric_vals_{}".format(dataloader_idx))(pred, batch)
    #     self.log_dict(getattr(self,"metric_vals_{}".format(dataloader_idx))(pred, batch), sync_dist=True)
    #
    #     getattr(self,"losse_val_{}".format(dataloader_idx)).update(losses)
    #     # print(getattr(self,"losse_val_{}".format(dataloader_idx)))
    #     self.log_dict(getattr(self,"losse_val_{}".format(dataloader_idx)).compute(), sync_dist=True)
    def validation_step(self, batch, batch_idx):
        pred = self(batch)
        losses = self.model.loss(pred, batch)
        if self.losses_val is None:
            self.losses_val = MetricCollection(
                {k: AverageKeyMeter(k).to(self.device) for k in losses},
                prefix="loss/",
                postfix="/val",
            )
        self.metrics_val(pred, batch)
        self.log_dict(self.metrics_val, sync_dist=True)
        self.losses_val.update(losses)
        self.log_dict(self.losses_val, sync_dist=True)

    def validation_epoch_start(self, batch):
        self.losses_val = None
        # self.losse_val = [None for city in self.cfg.data.val_citys]

    def configure_optimizers(self):
        optimizer_conf = self.cfg.training.get("optimizer")
        if optimizer_conf is not None:
            optimizer, scheduler = build_staged_optimizer(
                self.named_parameters(),
                lr=self.cfg.training.lr,
                baseline_lr_scale=float(optimizer_conf.baseline_lr_scale),
                baseline_warmup_steps=int(optimizer_conf.baseline_warmup_steps),
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step",
                    "frequency": 1,
                    "name": "learning_rate",
                },
            }

        optimizer = torch.optim.Adam(self.parameters(), lr=self.cfg.training.lr)
        ret = {"optimizer": optimizer}
        cfg_scheduler = self.cfg.training.get("lr_scheduler")
        if cfg_scheduler is not None:
            scheduler = getattr(torch.optim.lr_scheduler, cfg_scheduler.name)(
                optimizer=optimizer, **cfg_scheduler.get("args", {})
            )
            ret["lr_scheduler"] = {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
                "monitor": "loss/total/val",
                "strict": True,
                "name": "learning_rate",
            }
        return ret

    @classmethod
    def load_from_checkpoint(
        cls,
        checkpoint_path,
        map_location=None,
        hparams_file=None,
        strict=True,
        cfg=None,
        find_best=False,
    ):
        assert hparams_file is None, "hparams are not supported."

        try:
            checkpoint = torch.load(
                checkpoint_path,
                map_location=map_location or (lambda storage, loc: storage),
                weights_only=False,
            )
        except TypeError:
            checkpoint = torch.load(
                checkpoint_path, map_location=map_location or (lambda storage, loc: storage)
            )
        if find_best:
            best_score, best_name = None, None
            modes = {"min": torch.lt, "max": torch.gt}
            for key, state in checkpoint["callbacks"].items():
                if not key.startswith("ModelCheckpoint"):
                    continue
                mode = eval(key.replace("ModelCheckpoint", ""))["mode"]
                if best_score is None or modes[mode](
                    state["best_model_score"], best_score
                ):
                    best_score = state["best_model_score"]
                    best_name = Path(state["best_model_path"]).name
            logger.info("Loading best checkpoint %s", best_name)
            if best_name != checkpoint_path:
                return cls.load_from_checkpoint(
                    Path(checkpoint_path).parent / best_name,
                    map_location,
                    hparams_file,
                    strict,
                    cfg,
                    find_best=False,
                )

        logger.info(
            "Using checkpoint %s from epoch %d and step %d.",
            Path(checkpoint_path).name,
            checkpoint["epoch"],
            checkpoint["global_step"],
        )
        cfg_ckpt = checkpoint[cls.CHECKPOINT_HYPER_PARAMS_KEY]
        if list(cfg_ckpt.keys()) == ["cfg"]:  # backward compatibility
            cfg_ckpt = cfg_ckpt["cfg"]
        cfg_ckpt = OmegaConf.create(cfg_ckpt)

        if cfg is None:
            cfg = {}
        if not isinstance(cfg, DictConfig):
            cfg = OmegaConf.create(cfg)
        with open_dict(cfg_ckpt):
            cfg = OmegaConf.merge(cfg_ckpt, cfg)
        OmegaConf.resolve(cfg)  #  ${xxx} 

        return pl.core.saving._load_state(cls, checkpoint, strict=strict, cfg=cfg)
