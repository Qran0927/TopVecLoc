from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]


def test_ground_truth_path_is_thinner_than_ours():
    source = (ROOT / "osm_trajectory_map.py").read_text(encoding="utf-8")
    ground_truth_block = re.search(
        r'label="Ground truth",(?P<body>.*?)zorder=6,', source, flags=re.DOTALL
    ).group("body")
    ours_block = re.search(
        r'label="TopVecLoc \(ours\)",(?P<body>.*?)zorder=5,',
        source,
        flags=re.DOTALL,
    ).group("body")

    ground_truth_width = float(
        re.search(r"linewidth=([0-9.]+)", ground_truth_block).group(1)
    )
    ours_width = float(re.search(r"linewidth=([0-9.]+)", ours_block).group(1))

    assert ground_truth_width == 1.15
    assert ground_truth_width < ours_width
