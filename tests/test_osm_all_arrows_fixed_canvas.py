from pathlib import Path

import numpy as np

from trajectory_comparison import ARROW_POSITIONS


ROOT = Path(__file__).resolve().parents[1]


def test_all_twenty_samples_receive_yaw_arrows():
    np.testing.assert_array_equal(ARROW_POSITIONS, np.arange(20))


def test_osm_exports_use_the_same_fixed_canvas_without_tight_cropping():
    source = (ROOT / "osm_trajectory_map.py").read_text(encoding="utf-8")

    assert 'figsize=(8.0, 6.4)' in source
    assert 'bbox_inches="tight"' not in source
