from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_start_marker_is_smaller_and_end_symbol_is_removed():
    source = (ROOT / "osm_trajectory_map.py").read_text(encoding="utf-8")

    assert 's=28, marker="s"' in source
    assert 'marker="*"' not in source
    assert 'ax.annotate("Start"' in source
    assert 'ax.annotate("End"' in source
