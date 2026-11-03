"""Render the three additional OSM trajectory comparisons with exact yaw arrows."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from additional_trajectory_candidates import ADDITIONAL_CANDIDATE_GROUPS
from osm_trajectory_map import plot_osm_trajectory


REQUIRED_RESULT_COLUMNS = {
    "pos_error_m",
    "yaw_error_deg",
    "signed_yaw_error_deg",
    "pred_yaw",
    "gt_yaw",
    "pred_u",
    "pred_v",
    "gt_u",
    "gt_v",
}


def load_additional_candidates(
    root: Path,
) -> dict[int, tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]]:
    """Load and validate candidates 2-4 in their fixed spatial path order."""
    baseline_all = pd.read_csv(
        root
        / "experiments/last-step-checkpointing-v1_additional_selected_per_sample.csv"
    ).set_index("index")
    ours_all = pd.read_csv(
        root
        / "experiments/maplocnet__CNN+GNN_1/"
        "checkpoint-best-step-150000_additional_selected_per_sample.csv"
    ).set_index("index")
    info_all = pd.read_csv(
        root / "UAV2MAP_v1.0/UAV2MAP/SanFrancisco/info.csv"
    ).set_index("id")

    for method, frame in (("TopVecLoc", baseline_all), ("TopVecLoc", ours_all)):
        missing = REQUIRED_RESULT_COLUMNS.difference(frame.columns)
        if missing:
            raise ValueError(f"{method} result is missing columns: {sorted(missing)}")

    candidates = {}
    for candidate_number, sample_ids in enumerate(ADDITIONAL_CANDIDATE_GROUPS, start=2):
        missing_baseline = set(sample_ids).difference(baseline_all.index)
        missing_ours = set(sample_ids).difference(ours_all.index)
        missing_info = set(sample_ids).difference(info_all.index)
        if missing_baseline or missing_ours or missing_info:
            raise ValueError(
                f"Candidate {candidate_number} has missing IDs: "
                f"baseline={sorted(missing_baseline)}, ours={sorted(missing_ours)}, "
                f"info={sorted(missing_info)}"
            )

        baseline = baseline_all.loc[sample_ids].copy()
        ours = ours_all.loc[sample_ids].copy()
        info = info_all.loc[sample_ids].copy()
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
        candidates[candidate_number] = (ours, baseline, info)
    return candidates


def main() -> None:
    root = Path(__file__).resolve().parent
    for candidate_number, candidate_data in load_additional_candidates(root).items():
        output_stem = (
            root
            / "outputs"
            / f"trajectory_candidate_{candidate_number}_20_samples_osm"
        )
        plot_osm_trajectory(root, output_stem, candidate_data=candidate_data)
        print(f"Candidate {candidate_number} exported to {output_stem}.*")


if __name__ == "__main__":
    main()
