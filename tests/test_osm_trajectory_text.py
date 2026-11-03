from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_osm_figure_uses_requested_mean_metrics_and_method_names():
    source = (ROOT / "osm_trajectory_map.py").read_text(encoding="utf-8")

    assert "Mean position error:" in source
    assert "Mean yaw error:" in source
    assert "baseline.pos_error_m.mean()" in source
    assert "ours.pos_error_m.mean()" in source
    assert "baseline.yaw_error_deg.mean()" in source
    assert "ours.yaw_error_deg.mean()" in source
    assert 'label="VecMapLocNet"' in source
    assert 'label="TopVecLoc (ours)"' in source
    assert "Median position error:" not in source
    assert "Median yaw error:" not in source


def test_osm_figure_has_no_title():
    source = (ROOT / "osm_trajectory_map.py").read_text(encoding="utf-8")

    assert "Representative spatial sequence" not in source
