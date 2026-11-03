from pathlib import Path
import math

import numpy as np

from osm_viewport import fixed_aspect_viewport


ROOT = Path(__file__).resolve().parents[1]


def test_fixed_viewport_reserves_right_side_map_area_for_legend():
    points = np.array([[100.0, 200.0], [900.0, 700.0], [500.0, 450.0]])

    x_min, x_max, y_min, y_max = fixed_aspect_viewport(
        points,
        padding_pixels=72,
        right_reserved_fraction=0.30,
    )

    width = x_max - x_min
    assert width / (y_max - y_min) == 5 / 4
    assert x_min <= points[:, 0].min() - 72
    assert points[:, 0].max() + 72 <= x_max - math.ceil(0.30 * width)


def test_osm_uses_compact_canvas_and_internal_upper_right_legend():
    source = (ROOT / "osm_trajectory_map.py").read_text(encoding="utf-8")

    assert "right_reserved_fraction=0.30" in source
    assert "plt.figure(figsize=(8.0, 6.4))" in source
    assert "fig.add_axes([0.025, 0.025, 0.95, 0.95])" in source
    legend_block = source[source.index("    ax.legend(") : source.index("\n\n    output_stem")]
    assert 'loc="upper right"' in legend_block
    assert "bbox_to_anchor" not in legend_block

