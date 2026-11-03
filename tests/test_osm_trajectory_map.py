import numpy as np

from osm_trajectory_map import (
    latlon_to_world_pixels,
    predictions_to_latlon,
    yaw_to_image_vectors,
)


def test_latlon_to_world_pixels_places_origin_at_world_center():
    actual = latlon_to_world_pixels(
        np.array([0.0]), np.array([0.0]), zoom=1
    )

    np.testing.assert_allclose(actual, np.array([[256.0, 256.0]]), atol=1e-12)


def test_predictions_to_latlon_respects_east_and_north_map_axes():
    actual_lat, actual_lon = predictions_to_latlon(
        gt_lat=np.array([0.0]),
        gt_lon=np.array([0.0]),
        pred_u=np.array([10.0]),
        pred_v=np.array([-10.0]),
        gt_u=np.array([0.0]),
        gt_v=np.array([0.0]),
        pixels_per_meter=np.array([1.0]),
    )

    np.testing.assert_allclose(actual_lat, np.array([10.0 / 110_540.0]))
    np.testing.assert_allclose(actual_lon, np.array([10.0 / 111_320.0]))


def test_yaw_to_image_vectors_matches_project_plotting_convention():
    actual = yaw_to_image_vectors(np.array([0.0, 90.0, 180.0, 270.0]))

    expected = np.array(
        [
            [0.0, -1.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [-1.0, 0.0],
        ]
    )
    np.testing.assert_allclose(actual, expected, atol=1e-12)
