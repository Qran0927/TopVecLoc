"""
MapLocNet 

 PyTorch Lightning + Hydra 。

：
  ① Hydra  conf/maplocnet.yaml 
  ②  (MapLocNet + GenericModule) 
  ③  Checkpoint、Logger、Callbacks
  ④ trainer.fit() （）

：
  : python train.py
  Hydra  conf/maplocnet.yaml，，
"""

import os.path as osp
import warnings
from pathlib import Path
from typing import Optional

import hydra
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.utilities import rank_zero_only

from dataset import UavMapDatasetModule
from logger import EXPERIMENTS_PATH, logger, pl_logger
from models.hybrid_checkpoint import (
    load_cnn_baseline_into_hybrid,
    load_topveloc_into_hybrid,
)
from models.maplocnet import MapLocNet
from module import GenericModule

warnings.filterwarnings("ignore")

RECOVERY_CHECKPOINT_STEM = "last-step-checkpointing"


# ============================================================
# 、 Callbacks（PyTorch Lightning ）
# ============================================================

class CleanProgressBar(pl.callbacks.TQDMProgressBar):
    """
     ——  v_num  loss 。

     PyTorch Lightning  TQDMProgressBar，
     get_metrics ，。
    """

    def get_metrics(self, trainer, model):
        items = super().get_metrics(trainer, model)
        items.pop("v_num", None)   #  Lightning 
        items.pop("loss", None)    #  loss（ loss/total/train）
        return items


class SeedingCallback(pl.callbacks.Callback):
    """
     epoch 。

    ？
       epoch ，""。
       epoch  →  epoch  shuffle  → 。

    : module.cfg.experiment.seed ( 0)
     +  overfit: seed =  +  epoch 
    """

    def on_epoch_start_(self, trainer, module):
        seed = module.cfg.experiment.seed
        # overfit_batches > 0 （），
        is_overfit = module.cfg.training.trainer.get("overfit_batches", 0) > 0
        if trainer.training and not is_overfit:
            seed = seed + trainer.current_epoch

        # ， seed_everything 
        pl_logger.disabled = True
        try:
            pl.seed_everything(seed, workers=True)
        finally:
            pl_logger.disabled = False

    def on_train_epoch_start(self, *args, **kwargs):
        self.on_epoch_start_(*args, **kwargs)

    def on_validation_epoch_start(self, *args, **kwargs):
        self.on_epoch_start_(*args, **kwargs)

    def on_test_epoch_start(self, *args, **kwargs):
        self.on_epoch_start_(*args, **kwargs)


class ConsoleLogger(pl.callbacks.Callback):
    """
     epoch 。

     rank==0 （ GPU ）。
    """

    @rank_zero_only
    def on_train_epoch_start(self, trainer, module):
        logger.info(
            "New training epoch %d for experiment '%s'.",
            module.current_epoch,
            module.cfg.experiment.name,
        )


# ============================================================
# 、
# ============================================================

def find_last_checkpoint_path(experiment_dir):
    """
     checkpoint。

    :
      （）， None。

    Lightning  "last" checkpoint ：
      modelcheckpoint.CHECKPOINT_NAME_LAST + ".ckpt"
    """
    directory = Path(experiment_dir)
    recovery_candidates = list(
        directory.glob(f"{RECOVERY_CHECKPOINT_STEM}*.ckpt")
    )
    if recovery_candidates:
        return str(max(recovery_candidates, key=lambda path: path.stat().st_mtime))

    cls = pl.callbacks.ModelCheckpoint
    path = directory / f"{cls.CHECKPOINT_NAME_LAST}{cls.FILE_EXTENSION}"
    return str(path) if path.exists() else None


def prepare_experiment_dir(experiment_dir, cfg, rank):
    """
     + 。

    :
      ①  last.ckpt → 
         →  config，（）
      ②  → ， config.yaml 
      ③  last_checkpoint_path（， None）
    """
    config_path = osp.join(experiment_dir, "config.yaml")
    last_checkpoint_path = find_last_checkpoint_path(experiment_dir)

    if last_checkpoint_path is not None:
        # ──  ──
        if rank == 0:
            logger.info(
                "Resuming the training from checkpoint %s", last_checkpoint_path
            )
        # 
        if osp.exists(config_path):
            with open(config_path, "r") as fp:
                cfg_prev = OmegaConf.create(fp.read())
            compare_keys = ["experiment", "data", "model", "training"]
            if OmegaConf.masked_copy(cfg, compare_keys) != OmegaConf.masked_copy(
                cfg_prev, compare_keys
            ):
                raise ValueError(
                    "Attempting to resume training with a different config: "
                    f"{OmegaConf.masked_copy(cfg, compare_keys)} vs "
                    f"{OmegaConf.masked_copy(cfg_prev, compare_keys)}"
                )

    if rank == 0:
        #  + （）
        Path(experiment_dir).mkdir(exist_ok=True, parents=True)
        with open(config_path, "w") as fp:
            OmegaConf.save(cfg, fp)

    return last_checkpoint_path


# ============================================================
# 、
# ============================================================

def train(cfg: DictConfig) -> None:
    """
    MapLocNet 。

    :
      ①  →  →  GPU/CPU
      ② （ or  checkpoint ）
      ③ （）
      ④  Checkpoint 
      ⑤  DataModule + TensorBoard Logger
      ⑥  Callbacks
      ⑦ trainer.fit() 
    """

    # ── ①  ──
    #  float32 （ GPU Tensor Cores，）
    torch.set_float32_matmul_precision("medium")

    #  OmegaConf （ ${data.image_size}）
    OmegaConf.resolve(cfg)

    #  rank（， 0）
    rank = rank_zero_only.rank

    if rank == 0:
        logger.info("Starting training with config:\n%s", OmegaConf.to_yaml(cfg))

    # GPU/CPU 
    if cfg.experiment.gpus in (None, 0):
        logger.warning("Will train on CPU...")
        cfg.experiment.gpus = 0
    elif not torch.cuda.is_available():
        raise ValueError("Requested GPU but no NVIDIA drivers found.")

    # （）
    pl.seed_everything(cfg.experiment.seed, workers=True)

    # ── ②  ──
    init_checkpoint_path = cfg.training.get("finetune_from_checkpoint")
    if init_checkpoint_path is not None:
        #  checkpoint 
        logger.info(
            "Initializing the model from checkpoint %s.", init_checkpoint_path
        )
        model = GenericModule.load_from_checkpoint(
            init_checkpoint_path, strict=True, find_best=False, cfg=cfg
        )
    else:
        # 
        model = GenericModule(cfg)

    if rank == 0:
        logger.info("Network:\n%s", model.model)

    # ── ③  ──
    experiment_dir = osp.join(EXPERIMENTS_PATH, cfg.experiment.name)
    last_checkpoint_path = prepare_experiment_dir(experiment_dir, cfg, rank)

    baseline_checkpoint = cfg.training.get("baseline_checkpoint")
    topveloc_checkpoint = cfg.training.get("topveloc_checkpoint")
    if baseline_checkpoint is not None and topveloc_checkpoint is not None:
        raise ValueError("configure only one initialization checkpoint")
    if (
        topveloc_checkpoint is not None
        and init_checkpoint_path is None
        and last_checkpoint_path is None
    ):
        report = load_topveloc_into_hybrid(
            model,
            topveloc_checkpoint,
            fusion_gate_value=float(cfg.model.map_encoder.fusion.gate_init),
        )
        if rank == 0:
            logger.info(
                "Initialized from TopVeLoc %s (loaded=%d, new=%d).",
                topveloc_checkpoint,
                report.loaded_keys,
                len(report.missing_target_keys),
            )
    if (
        baseline_checkpoint is not None
        and init_checkpoint_path is None
        and last_checkpoint_path is None
    ):
        report = load_cnn_baseline_into_hybrid(model, baseline_checkpoint)
        if rank == 0:
            logger.info(
                "Initialized hybrid model from CNN baseline %s "
                "(image=%d, map=%d, shared=%d, skipped=%d).",
                baseline_checkpoint,
                report.loaded_image_keys,
                report.loaded_map_keys,
                report.loaded_shared_keys,
                len(report.skipped_source_keys),
            )

    # ── ④ Checkpoint  ──
    #  A: ， val loss 
    checkpointing_best = pl.callbacks.ModelCheckpoint(
        dirpath=experiment_dir,
        filename="checkpoint-best-step-{step}",
        auto_insert_metric_name=False,
        save_last=False,
        save_on_train_epoch_end=False,
        verbose=True,
        **cfg.training.checkpointing,    #  monitor, save_top_k, mode
    )

    #  B:  1000 ，
    checkpointing_recovery = pl.callbacks.ModelCheckpoint(
        dirpath=experiment_dir,
        filename="checkpoint-step-{step}",
        auto_insert_metric_name=False,
        monitor=None,
        save_top_k=1,
        save_last=True,
        every_n_train_steps=1000,
        verbose=True,
    )
    checkpointing_recovery.CHECKPOINT_NAME_LAST = RECOVERY_CHECKPOINT_STEM

    # （）
    # early_stopping_callback = EarlyStopping(
    #     monitor=cfg.training.checkpointing.monitor, patience=5
    # )

    # ── ⑤  ──
    strategy = "auto"
    if cfg.experiment.gpus > 1:
        #  GPU: DistributedDataParallel（，）
        strategy = pl.strategies.DDPStrategy(find_unused_parameters=False)
        #  batch_size  num_workers  GPU
        for split in ["train", "val"]:
            cfg.data[split].batch_size = (
                cfg.data[split].batch_size // cfg.experiment.gpus
            )
            cfg.data[split].num_workers = int(
                (cfg.data[split].num_workers + cfg.experiment.gpus - 1)
                / cfg.experiment.gpus
            )

    # ── ⑥  + Logger ──
    datamodule = UavMapDatasetModule(cfg.data)

    tb = pl.loggers.TensorBoardLogger(
        save_dir=osp.join(EXPERIMENTS_PATH, cfg.experiment.name),
        name="tb",
    )

    # ── ⑦  Callbacks ──
    callbacks = [
        checkpointing_best,
        checkpointing_recovery,
        pl.callbacks.LearningRateMonitor(),   # TensorBoard 
        SeedingCallback(),                    #  epoch 
        CleanProgressBar(),                   # 
        ConsoleLogger(),                      #  epoch 
    ]
    if cfg.experiment.gpus > 0:
        callbacks.append(pl.callbacks.DeviceStatsMonitor())  # GPU 

    # ── ⑧ Trainer  +  ──
    accelerator = "gpu" if cfg.experiment.gpus > 0 else "cpu"
    trainer_kwargs = OmegaConf.to_container(cfg.training.trainer, resolve=True)
    if accelerator == "cpu":
        # CPU  devices 
        trainer_kwargs.pop("devices", None)

    trainer = pl.Trainer(
        default_root_dir=experiment_dir,      # 
        detect_anomaly=False,                 # （）
        enable_model_summary=True,            # 
        sync_batchnorm=True,                  #  BN 
        enable_checkpointing=True,
        logger=tb,
        callbacks=callbacks,
        strategy=strategy,
        check_val_every_n_epoch=None,         #  step  epoch 
        accelerator=accelerator,
        num_nodes=1,
        **trainer_kwargs,                     # 
    )

    # ！
    # ckpt_path:  last checkpoint，
    trainer.fit(
        model=model,
        datamodule=datamodule,
        ckpt_path=last_checkpoint_path,
    )


# ============================================================
# 、
# ============================================================

@hydra.main(
    config_path=osp.join(osp.dirname(__file__), "conf"),
    config_name="maplocnet.yaml",
)
def main(cfg: DictConfig) -> None:
    """
    Hydra :
      -  conf/maplocnet.yaml
      - : python train.py model.hidden_dim=128
      -  outputs/ 
    """
    train(cfg)


if __name__ == "__main__":
    main()
