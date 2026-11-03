import numpy as np

from trajectory_comparison import CANDIDATE_IDS, uv_predictions_to_local_xy


def test_uv_predictions_to_local_xy_respects_map_axis_directions():
    gt_xy = np.array([[100.0, 200.0], [10.0, 20.0]])
    pred_u = np.array([14.0, 8.0])
    pred_v = np.array([18.0, 24.0])
    gt_u = np.array([10.0, 10.0])
    gt_v = np.array([20.0, 20.0])
    pixels_per_meter = np.array([2.0, 2.0])

    actual = uv_predictions_to_local_xy(
        gt_xy, pred_u, pred_v, gt_u, gt_v, pixels_per_meter
    )

    expected = np.array([[102.0, 201.0], [9.0, 18.0]])
    np.testing.assert_allclose(actual, expected)


def test_representative_sequence_contains_the_selected_twenty_samples():
    assert CANDIDATE_IDS == [
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
