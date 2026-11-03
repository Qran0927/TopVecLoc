import numpy as np

from osm_trajectory_map import fixed_aspect_viewport


def test_fixed_aspect_viewport_contains_points_and_padding_at_five_to_four():
    points = np.array(
        [
            [100.0, 100.0],
            [300.0, 1100.0],
            [250.0, 700.0],
        ]
    )

    x_min, x_max, y_min, y_max = fixed_aspect_viewport(
        points, padding_pixels=72, aspect_width=5, aspect_height=4
    )

    assert (x_max - x_min) / (y_max - y_min) == 5 / 4
    assert x_min <= points[:, 0].min() - 72
    assert x_max >= points[:, 0].max() + 72
    assert y_min <= points[:, 1].min() - 72
    assert y_max >= points[:, 1].max() + 72
    assert all(float(value).is_integer() for value in (x_min, x_max, y_min, y_max))


def test_fixed_aspect_viewport_is_identical_in_shape_for_landscape_and_portrait():
    landscape = np.array([[0.0, 0.0], [1000.0, 300.0]])
    portrait = np.array([[0.0, 0.0], [300.0, 1000.0]])

    landscape_bounds = fixed_aspect_viewport(landscape, padding_pixels=50)
    portrait_bounds = fixed_aspect_viewport(portrait, padding_pixels=50)

    for bounds in (landscape_bounds, portrait_bounds):
        x_min, x_max, y_min, y_max = bounds
        assert (x_max - x_min) / (y_max - y_min) == 5 / 4
