"""Pure helpers for selected-sample UAV evaluation and CSV export."""

from __future__ import annotations

from collections.abc import Sequence
import math


def resolve_selected_positions(
    available_sample_indices: Sequence[int], selected_sample_indices: Sequence[int]
) -> list[int]:
    """Map original sample IDs to dataset positions while preserving request order."""
    available = [int(index) for index in available_sample_indices]
    selected = [int(index) for index in selected_sample_indices]
    if len(set(available)) != len(available):
        raise ValueError("available sample indices contain duplicate values")
    if len(set(selected)) != len(selected):
        raise ValueError("selected sample indices contain duplicate values")

    position_by_index = {
        sample_index: position for position, sample_index in enumerate(available)
    }
    missing = [index for index in selected if index not in position_by_index]
    if missing:
        raise ValueError(f"selected sample indices are not available: {missing}")
    return [position_by_index[index] for index in selected]


def signed_angle_error_deg(prediction: float, target: float) -> float:
    """Return prediction minus target wrapped to the interval [-180, 180)."""
    return (float(prediction) - float(target) + 180.0) % 360.0 - 180.0


def compute_paper_position_errors(
    *, uv_max, uv_gt, gt_yaw_deg: float, pixel_per_meter: float
) -> dict[str, float]:
    """Compute the max-pose errors used by the paper's R@d table."""
    ppm = float(pixel_per_meter)
    if ppm <= 0:
        raise ValueError("pixel_per_meter must be positive")
    du = (float(uv_max[0]) - float(uv_gt[0])) * -1.0
    dv = float(uv_max[1]) - float(uv_gt[1])
    yaw = math.radians(float(gt_yaw_deg))
    lateral = math.cos(yaw) * du - math.sin(yaw) * dv
    longitudinal = math.sin(yaw) * du + math.cos(yaw) * dv
    return {
        "position_max_error_m": math.hypot(du, dv) / ppm,
        "lateral_max_error_m": abs(lateral) / ppm,
        "longitudinal_max_error_m": abs(longitudinal) / ppm,
    }


def build_sample_row(
    *,
    sample_index: int,
    position_error_m: float,
    pred_yaw: float,
    gt_yaw: float,
    uv_pred,
    uv_gt,
    pred_yaw_max: float | None = None,
) -> dict[str, float | int]:
    """Build one stable, plotting-ready per-sample CSV row."""
    signed_yaw_error = signed_angle_error_deg(pred_yaw, gt_yaw)
    row = {
        "index": int(sample_index),
        "pos_error_m": float(position_error_m),
        "yaw_error_deg": abs(signed_yaw_error),
        "signed_yaw_error_deg": signed_yaw_error,
        "pred_yaw": float(pred_yaw),
        "gt_yaw": float(gt_yaw),
        "pred_u": float(uv_pred[0]),
        "pred_v": float(uv_pred[1]),
        "gt_u": float(uv_gt[0]),
        "gt_v": float(uv_gt[1]),
    }
    if pred_yaw_max is not None:
        row["pred_yaw_max"] = float(pred_yaw_max)
        row["yaw_max_error_deg"] = abs(signed_angle_error_deg(pred_yaw_max, gt_yaw))
    return row
