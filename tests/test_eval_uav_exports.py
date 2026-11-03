from pathlib import Path

import pytest

from eval_uav_helpers import (
    build_sample_row,
    compute_paper_position_errors,
    resolve_selected_positions,
    signed_angle_error_deg,
)


ROOT = Path(__file__).resolve().parents[1]


def test_resolve_selected_positions_preserves_requested_order():
    assert resolve_selected_positions([10, 20, 30, 40], [30, 10]) == [2, 0]


@pytest.mark.parametrize(
    ("available", "selected", "message"),
    [
        ([10, 20], [10, 10], "duplicate"),
        ([10, 20], [30], "not available"),
    ],
)
def test_resolve_selected_positions_rejects_ambiguous_indices(
    available, selected, message
):
    with pytest.raises(ValueError, match=message):
        resolve_selected_positions(available, selected)


@pytest.mark.parametrize(
    ("prediction", "target", "expected"),
    [
        (1.0, 359.0, 2.0),
        (359.0, 1.0, -2.0),
        (180.0, 0.0, -180.0),
        (42.5, 40.0, 2.5),
    ],
)
def test_signed_angle_error_deg_wraps_to_half_open_interval(
    prediction, target, expected
):
    assert signed_angle_error_deg(prediction, target) == expected


def test_build_sample_row_keeps_original_index_and_complete_yaw_fields():
    row = build_sample_row(
        sample_index=7420,
        position_error_m=2.5,
        pred_yaw=1.0,
        gt_yaw=359.0,
        uv_pred=(10.0, 20.0),
        uv_gt=(11.0, 19.0),
    )

    assert row == {
        "index": 7420,
        "pos_error_m": 2.5,
        "yaw_error_deg": 2.0,
        "signed_yaw_error_deg": 2.0,
        "pred_yaw": 1.0,
        "gt_yaw": 359.0,
        "pred_u": 10.0,
        "pred_v": 20.0,
        "gt_u": 11.0,
        "gt_v": 19.0,
    }


def test_compute_paper_position_errors_uses_uv_max_and_ground_truth_yaw():
    errors = compute_paper_position_errors(
        uv_max=(13.0, 24.0),
        uv_gt=(10.0, 20.0),
        gt_yaw_deg=0.0,
        pixel_per_meter=1.0,
    )

    assert errors == {
        "position_max_error_m": pytest.approx(5.0),
        "lateral_max_error_m": pytest.approx(3.0),
        "longitudinal_max_error_m": pytest.approx(4.0),
    }


def test_eval_script_integrates_selected_samples_and_yaw_export():
    source = (ROOT / "eval_uav.py").read_text(encoding="utf-8")

    assert "SELECTED_SAMPLE_INDICES" in source
    assert "resolve_selected_positions" in source
    assert 'result["sample_index"]' in source
    assert "Subset" in source
    assert "_selected_per_sample.csv" in source
    for field in (
        "pred_yaw",
        "gt_yaw",
        "signed_yaw_error_deg",
        "position_max_error_m",
        "lateral_max_error_m",
        "longitudinal_max_error_m",
    ):
        assert field in source


def test_max_yaw_export_is_separate_from_expectation_and_wraps_angles():
    from eval_uav_helpers import build_sample_row
    row = build_sample_row(sample_index=0, position_error_m=1,
                           pred_yaw=30, gt_yaw=359, uv_pred=(0, 0), uv_gt=(0, 0),
                           pred_yaw_max=1)
    assert row['yaw_error_deg'] == 31
    assert row['pred_yaw_max'] == 1
    assert row['yaw_max_error_deg'] == 2

