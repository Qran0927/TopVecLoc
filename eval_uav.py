"""UAV 。

 checkpoint ， evaluation/run.py ：
，MetricCollection ， R@。
"""

import csv
from pathlib import Path
import warnings

import cv2
import numpy as np
import torch
import torchvision.transforms as tvf
from omegaconf import OmegaConf, open_dict
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset
from torchmetrics import MetricCollection

from additional_trajectory_candidates import ADDITIONAL_CANDIDATE_GROUPS
from eval_uav_helpers import (
    build_sample_row,
    compute_paper_position_errors,
    resolve_selected_positions,
)
from module import GenericModule
from models.metrics import AngleError, LateralLongitudinalError


# 、；。
DATA_ROOT = Path("${UAV2MAP_ROOT}")
CHECKPOINT_PATH = "./experiments/maplocnet__CNN+GNN_1/checkpoint-best-step-150000.ckpt"
CONFIG_PATH = "conf/maplocnet.yaml"
BATCH_SIZE = 1
NUM_WORKERS = 0
THRESHOLDS = [1, 3, 5, 10]

# 。 None 。
#  20 ；，。
SELECTED_SAMPLE_INDICES = [
    sample_id
    for group in ADDITIONAL_CANDIDATE_GROUPS
    for sample_id in group
]


class SafeUavDataset(Dataset):
    """UAV ： info.csv，。"""

    def __init__(self, root: Path, city: str, transform):
        self.root = root
        self.city = city
        self.transform = transform
        self.uav_path = root / city / "uav"
        self.map_path = root / city / "map"
        self.graph_path = root / city / "graph"

        info_path = root / city / "info.csv"
        self.info = np.loadtxt(str(info_path), dtype=str, delimiter=",", skiprows=1)

        self.valid_indices = []
        for i in range(len(self.info)):
            uav_name = self.info[i][1]
            map_name = self.info[i][2]
            image_file = self.uav_path / uav_name
            map_file = self.map_path / map_name
            graph_file = self.graph_path / map_name.replace(".npy", ".pt")
            if (
                image_file.exists()
                and image_file.stat().st_size > 0
                and map_file.exists()
                and map_file.stat().st_size > 0
                and graph_file.exists()
            ):
                self.valid_indices.append(i)
        print(
            f"  {city}: {len(self.valid_indices)}/{len(self.info)} valid samples",
            flush=True,
        )

    def __getitem__(self, idx):
        """， info.csv 。"""
        real_idx = self.valid_indices[idx]
        row = self.info[real_idx]
        (
            sample_index,
            uav_name,
            map_name,
            uav_long,
            uav_lat,
            map_long,
            map_lat,
            tile_size_meters,
            pixel_per_meter,
            u,
            v,
            yaw,
            dis,
        ) = row

        uav_image = cv2.imread(str(self.uav_path / uav_name))
        if uav_image is None:
            raise FileNotFoundError(self.uav_path / uav_name)
        uav_image = cv2.cvtColor(uav_image, cv2.COLOR_BGR2RGB)
        if self.transform:
            uav_image = self.transform(uav_image)

        map_data = np.load(str(self.map_path / map_name))

        graph = None
        graph_file = self.graph_path / map_name.replace(".npy", ".pt")
        if graph_file.exists():
            graph = torch.load(graph_file, map_location="cpu", weights_only=False)

        result = {
            "map": torch.from_numpy(np.ascontiguousarray(map_data)).long(),
            "image": torch.as_tensor(uav_image),
            "roll_pitch_yaw": torch.tensor((0, 0, float(yaw))).float(),
            "pixels_per_meter": torch.tensor(float(pixel_per_meter)).float(),
            "uv": torch.tensor([float(u), float(v)]).float(),
        }
        result["sample_index"] = int(sample_index)
        if graph is not None:
            result["graph"] = graph
        return result

    def __len__(self):
        return len(self.valid_indices)

    def available_sample_indices(self):
        """ info.csv 。"""
        return [int(self.info[real_idx][0]) for real_idx in self.valid_indices]


def move_batch_to_device(batch, device):
    """ batch  Tensor ， Tensor 。"""
    return {
        key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def recall_percent(errors, threshold):
    """。"""
    return ((errors < threshold).float().mean() * 100).item()


def mean_position_error(total_error, sample_count):
    """（）。"""
    if sample_count <= 0:
        raise ValueError("mean position error requires at least one position error")
    return total_error / sample_count


def build_directional_error(pixel_per_meter):
    """、。"""
    return LateralLongitudinalError(pixel_per_meter)


def format_recalls(errors, thresholds):
    """ recall 。"""
    return " ".join(
        f"{recall_percent(errors, threshold):>8.2f}"
        for threshold in thresholds
    )


def print_paper_metrics(metrics, thresholds):
    """、、。"""
    directional_errors = (
        metrics["directional_error"]
        .get_errors()
        .detach()
        .float()
        .cpu()
        .reshape(-1, 2)
    )
    lateral_errors = directional_errors[:, 0]
    longitudinal_errors = directional_errors[:, 1]
    position_errors = (
        metrics["xy_max_error"].get_errors().detach().flatten().float().cpu()
    )
    yaw_errors = (
        metrics["yaw_max_error"].get_errors().detach().flatten().float().cpu()
    )

    print("\n========== Paper-style Metrics ==========")
    recall_header = " ".join(f"R@{threshold}".rjust(8) for threshold in thresholds)
    print(f"{'Metric':<13} {recall_header}")
    print(f"{'Longitudinal':<13} {format_recalls(longitudinal_errors, thresholds)}")
    print(f"{'Lateral':<13} {format_recalls(lateral_errors, thresholds)}")
    print(f"{'Position':<13} {format_recalls(position_errors, thresholds)}")
    print(f"{'Yaw':<13} {format_recalls(yaw_errors, thresholds)}")


def main(output_path=None):
    warnings.filterwarnings("ignore")

    cfg = OmegaConf.load(CONFIG_PATH)
    with open_dict(cfg):
        cfg.data.root = str(DATA_ROOT)
        cfg.experiment.gpus = 1
        cfg.training.trainer.devices = 1
        cfg.model.image_encoder.backbone.pretrained = False

    val_tfs = tvf.Compose(
        [
            tvf.ToTensor(),
            tvf.Resize(cfg.data.image_size),
        ]
    )

    datasets = [
        SafeUavDataset(root=Path(cfg.data.root), city=city, transform=val_tfs)
        for city in cfg.data.val_citys
    ]
    selected_mode = SELECTED_SAMPLE_INDICES is not None
    if selected_mode:
        if len(datasets) != 1:
            raise ValueError(
                "SELECTED_SAMPLE_INDICES requires exactly one validation city so that "
                "sample IDs are unambiguous"
            )
        source_dataset = datasets[0]
        selected_positions = resolve_selected_positions(
            source_dataset.available_sample_indices(),
            SELECTED_SAMPLE_INDICES,
        )
        val_dataset = Subset(source_dataset, selected_positions)
        print(
            f"Selected original sample indices: {SELECTED_SAMPLE_INDICES}",
            flush=True,
        )
    else:
        val_dataset = ConcatDataset(datasets)

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )
    print(f"Validation valid samples: {len(val_dataset)}", flush=True)

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available, but this evaluation script is configured for GPU."
        )
    device = torch.device("cuda")
    print(f"Using device: {device}", flush=True)
    print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)

    model = GenericModule.load_from_checkpoint(
        CHECKPOINT_PATH,
        cfg=cfg,
        strict=True,
        find_best=False,
        map_location=device,
    )
    model = model.eval().to(device)
    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Model loaded, parameters: {total_params:.1f}M", flush=True)

    model_metrics = model.model.metrics()
    metric_dict = {
        "xy_max_error": model_metrics["xy_max_error"],
        "yaw_max_error": model_metrics["yaw_max_error"],
        "xy_expectation_error": model_metrics["xy_expectation_error"],
        "yaw_expectation_error": AngleError("yaw_expectation"),
        "directional_error": build_directional_error(
            model.model.conf.pixel_per_meter,
        ),
    }
    metrics = MetricCollection(metric_dict).to(device)
    print(f"Starting evaluation on {len(val_dataset)} samples...", flush=True)

    ckpt_path = Path(CHECKPOINT_PATH)
    if not ckpt_path.is_absolute():
        ckpt_path = Path.cwd() / ckpt_path
    csv_suffix = "_additional_selected_per_sample.csv" if selected_mode else "_per_sample.csv"
    csv_path = ckpt_path.with_name(ckpt_path.stem + csv_suffix)
    if output_path is not None:
        csv_path = Path(output_path)
        if csv_path.exists():
            raise FileExistsError(f"Refusing to overwrite evaluation: {csv_path}")
        csv_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Results will be saved to: {csv_path}", flush=True)
    position_error_sum = 0.0
    position_error_count = 0

    with open(csv_path, "w", newline="") as csv_file:
        fieldnames = [
            "index",
            "pos_error_m",
            "yaw_error_deg",
            "signed_yaw_error_deg",
            "pred_yaw",
            "gt_yaw",
            "pred_u",
            "pred_v",
            "gt_u",
            "gt_v",
            "position_max_error_m",
            "lateral_max_error_m",
            "longitudinal_max_error_m",
            "pred_yaw_max",
            "yaw_max_error_deg",
        ]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        with torch.no_grad():
            for i, batch in enumerate(val_loader):
                batch = move_batch_to_device(batch, device)
                pred = model(batch)
                metrics(pred, batch)

                pos_err = float(
                    (
                        (pred["uv_expectation"] - batch["uv"]).norm(dim=-1)
                        / cfg.model.pixel_per_meter
                    )
                    .cpu()
                    .item()
                )
                position_error_sum += pos_err
                position_error_count += 1

                pred_yaw = float(
                    pred["yaw_expectation"].detach().cpu().reshape(-1)[0].item()
                )
                gt_yaw = float(
                    batch["roll_pitch_yaw"][..., -1]
                    .detach()
                    .cpu()
                    .reshape(-1)[0]
                    .item()
                )
                sample_index = int(
                    batch["sample_index"].detach().cpu().reshape(-1)[0].item()
                )
                uv_pred = pred["uv_expectation"].squeeze(0).cpu().numpy()
                uv_max = pred["uv_max"].squeeze(0).cpu().numpy()
                uv_gt = batch["uv"].squeeze(0).cpu().numpy()
                row = build_sample_row(
                        sample_index=sample_index,
                        position_error_m=pos_err,
                        pred_yaw=pred_yaw,
                        gt_yaw=gt_yaw,
                        uv_pred=uv_pred,
                        uv_gt=uv_gt,
                        pred_yaw_max=float(pred["yaw_max"].detach().cpu().reshape(-1)[0].item()),
                )
                row.update(
                    compute_paper_position_errors(
                        uv_max=uv_max,
                        uv_gt=uv_gt,
                        gt_yaw_deg=gt_yaw,
                        pixel_per_meter=float(batch["pixels_per_meter"].item()),
                    )
                )
                writer.writerow(row)

                if (i + 1) % 50 == 0:
                    csv_file.flush()
                    print(
                        f"  processed {i + 1}/{len(val_loader)} batches",
                        flush=True,
                    )

    print_paper_metrics(metrics, THRESHOLDS)
    avg_position_error = mean_position_error(
        position_error_sum,
        position_error_count,
    )
    print(f"Mean position error (expectation): {avg_position_error:.4f} m")
    print(f"\nPer-sample results saved to: {csv_path}")


if __name__ == "__main__":
    main()
