from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


CANDIDATE_IDS = [
    7420,
    118,
    539,
    8408,
    7064,
    2269,
    6983,
    2848,
    3315,
    8572,
    5336,
    1661,
    7190,
    3719,
    8127,
    7683,
    8363,
    8192,
    7066,
    7587,
]

ARROW_POSITIONS = np.arange(20)


def uv_predictions_to_local_xy(
    gt_xy: np.ndarray,
    pred_u: np.ndarray,
    pred_v: np.ndarray,
    gt_u: np.ndarray,
    gt_v: np.ndarray,
    pixels_per_meter: np.ndarray,
) -> np.ndarray:
    """Convert map-pixel predictions to the ground-truth local metric frame."""
    east_offset = (np.asarray(pred_u) - np.asarray(gt_u)) / pixels_per_meter
    north_offset = -(np.asarray(pred_v) - np.asarray(gt_v)) / pixels_per_meter
    return np.asarray(gt_xy) + np.column_stack((east_offset, north_offset))


def yaw_to_east_north(yaw_deg: np.ndarray) -> np.ndarray:
    """Convert project yaw (clockwise from north) to east/north unit vectors."""
    yaw_rad = np.deg2rad(np.asarray(yaw_deg, dtype=float))
    return np.column_stack((np.sin(yaw_rad), np.cos(yaw_rad)))


def longitude_latitude_to_local_xy(
    longitude: np.ndarray, latitude: np.ndarray
) -> np.ndarray:
    """Project a small geographic region to a local east/north metric frame."""
    longitude = np.asarray(longitude, dtype=float)
    latitude = np.asarray(latitude, dtype=float)
    longitude_origin = longitude[0]
    latitude_origin = latitude[0]
    east = (
        (longitude - longitude_origin)
        * 111_320.0
        * np.cos(np.deg2rad(latitude_origin))
    )
    north = (latitude - latitude_origin) * 110_540.0
    return np.column_stack((east, north))


def load_candidate(root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ours = pd.read_csv(
        root
        / "experiments/maplocnet__CNN+GNN_1/"
        "checkpoint-best-step-150000_selected_per_sample.csv"
    ).set_index("index")
    baseline = pd.read_csv(
        root / "experiments/last-step-checkpointing-v1_selected_per_sample.csv"
    ).set_index("index")
    info = pd.read_csv(
        root / "UAV2MAP_v1.0/UAV2MAP/SanFrancisco/info.csv"
    ).set_index("id")

    ours = ours.loc[CANDIDATE_IDS]
    baseline = baseline.loc[CANDIDATE_IDS]
    info = info.loc[CANDIDATE_IDS]
    np.testing.assert_allclose(ours["gt_yaw"], baseline["gt_yaw"], atol=1e-6)
    np.testing.assert_allclose(
        ours[["gt_u", "gt_v"]], baseline[["gt_u", "gt_v"]], atol=1e-6
    )
    np.testing.assert_allclose(
        ours["yaw_error_deg"], ours["signed_yaw_error_deg"].abs(), atol=1e-6
    )
    np.testing.assert_allclose(
        baseline["yaw_error_deg"],
        baseline["signed_yaw_error_deg"].abs(),
        atol=1e-6,
    )
    return ours, baseline, info


def add_yaw_arrows(
    ax,
    xy: np.ndarray,
    yaw_deg: np.ndarray,
    color: str,
    *,
    arrow_length: float = 18.0,
    zorder: int = 6,
) -> None:
    positions = ARROW_POSITIONS
    vectors = yaw_to_east_north(np.asarray(yaw_deg)[positions]) * arrow_length
    ax.quiver(
        xy[positions, 0],
        xy[positions, 1],
        vectors[:, 0],
        vectors[:, 1],
        angles="xy",
        scale_units="xy",
        scale=1,
        color=color,
        width=0.0045,
        headwidth=4.2,
        headlength=5.0,
        headaxislength=4.3,
        alpha=0.95,
        zorder=zorder,
    )


def plot_candidate(root: Path, output_stem: Path) -> None:
    ours, baseline, info = load_candidate(root)
    gt_xy = longitude_latitude_to_local_xy(
        info["uav_long"].to_numpy(), info["uav_lat"].to_numpy()
    )
    pixels_per_meter = info["pixel_per_meter"].to_numpy()
    baseline_xy = uv_predictions_to_local_xy(
        gt_xy,
        baseline["pred_u"].to_numpy(),
        baseline["pred_v"].to_numpy(),
        baseline["gt_u"].to_numpy(),
        baseline["gt_v"].to_numpy(),
        pixels_per_meter,
    )
    ours_xy = uv_predictions_to_local_xy(
        gt_xy,
        ours["pred_u"].to_numpy(),
        ours["pred_v"].to_numpy(),
        ours["gt_u"].to_numpy(),
        ours["gt_v"].to_numpy(),
        pixels_per_meter,
    )

    np.testing.assert_allclose(
        np.linalg.norm(baseline_xy - gt_xy, axis=1),
        baseline["pos_error_m"].to_numpy(),
        rtol=2e-4,
        atol=2e-3,
    )
    np.testing.assert_allclose(
        np.linalg.norm(ours_xy - gt_xy, axis=1),
        ours["pos_error_m"].to_numpy(),
        rtol=2e-4,
        atol=2e-3,
    )

    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 8,
            "axes.linewidth": 0.8,
            "legend.frameon": False,
        }
    )
    fig, ax = plt.subplots(figsize=(7.2, 4.7), constrained_layout=True)
    colors = {"gt": "#222222", "baseline": "#D55E00", "ours": "#0072B2"}

    ax.plot(
        gt_xy[:, 0],
        gt_xy[:, 1],
        color=colors["gt"],
        linewidth=2.2,
        marker="o",
        markersize=3.0,
        label="Ground truth",
        zorder=4,
    )
    ax.plot(
        baseline_xy[:, 0],
        baseline_xy[:, 1],
        color=colors["baseline"],
        linewidth=1.3,
        linestyle="--",
        marker="o",
        markersize=2.7,
        alpha=0.88,
        label="Baseline",
        zorder=2,
    )
    ax.plot(
        ours_xy[:, 0],
        ours_xy[:, 1],
        color=colors["ours"],
        linewidth=1.8,
        marker="o",
        markersize=2.9,
        label="Ours",
        zorder=3,
    )

    add_yaw_arrows(ax, gt_xy, ours["gt_yaw"].to_numpy(), colors["gt"], zorder=7)
    add_yaw_arrows(
        ax,
        baseline_xy,
        baseline["pred_yaw"].to_numpy(),
        colors["baseline"],
        zorder=5,
    )
    add_yaw_arrows(
        ax,
        ours_xy,
        ours["pred_yaw"].to_numpy(),
        colors["ours"],
        zorder=6,
    )

    ax.scatter(
        gt_xy[0, 0], gt_xy[0, 1], s=54, marker="s", color="#009E73", zorder=9
    )
    ax.scatter(
        gt_xy[-1, 0], gt_xy[-1, 1], s=74, marker="*", color="#CC79A7", zorder=9
    )
    ax.annotate("Start", gt_xy[0], xytext=(5, 7), textcoords="offset points")
    ax.annotate("End", gt_xy[-1], xytext=(5, 7), textcoords="offset points")

    summary = (
        f"Median position error: {baseline.pos_error_m.median():.2f} → "
        f"{ours.pos_error_m.median():.2f} m\n"
        f"Median yaw error: {baseline.yaw_error_deg.median():.2f} → "
        f"{ours.yaw_error_deg.median():.2f}°\n"
        "Arrows show yaw at 7 representative samples"
    )
    ax.text(
        0.015,
        0.015,
        summary,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=7.2,
        bbox={
            "boxstyle": "round,pad=0.35",
            "facecolor": "white",
            "alpha": 0.92,
            "edgecolor": "#BBBBBB",
        },
        zorder=10,
    )
    ax.set_title("Representative spatial sequence (n = 20; non-temporal)", pad=8)
    ax.set_xlabel("East (m)")
    ax.set_ylabel("North (m)")
    ax.set_aspect("equal", adjustable="datalim")
    ax.margins(0.06)
    ax.grid(color="#D9D9D9", linewidth=0.5, alpha=0.7)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="upper right", ncols=3, handlelength=2.5, columnspacing=1.2)

    output_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(output_stem.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output_stem.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent
    plot_candidate(
        project_root,
        project_root / "outputs/trajectory_candidate_20_samples_with_yaw",
    )
