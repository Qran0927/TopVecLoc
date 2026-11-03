from pathlib import Path

import numpy as np

from additional_trajectory_candidates import ADDITIONAL_CANDIDATE_GROUPS
from osm_additional_trajectories import load_additional_candidates


ROOT = Path(__file__).resolve().parents[1]


def test_additional_loader_returns_three_ordered_twenty_sample_candidates():
    candidates = load_additional_candidates(ROOT)

    assert list(candidates) == [2, 3, 4]
    for candidate, expected_ids in zip(
        candidates.values(), ADDITIONAL_CANDIDATE_GROUPS, strict=True
    ):
        ours, baseline, info = candidate
        assert ours.index.tolist() == expected_ids
        assert baseline.index.tolist() == expected_ids
        assert info.index.tolist() == expected_ids
        assert len(ours) == len(baseline) == len(info) == 20
        np.testing.assert_allclose(ours["gt_yaw"], baseline["gt_yaw"], atol=1e-6)
        np.testing.assert_allclose(
            ours["yaw_error_deg"], ours["signed_yaw_error_deg"].abs(), atol=1e-6
        )


def test_osm_plotter_accepts_explicit_candidate_data():
    source = (ROOT / "osm_trajectory_map.py").read_text(encoding="utf-8")

    assert "candidate_data=None" in source
    assert "candidate_data if candidate_data is not None else load_candidate(root)" in source
