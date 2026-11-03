from pathlib import Path

import numpy as np
from PIL import Image

import osm_trajectory_map


def test_mosaic_accepts_42_tiles_for_reserved_legend_viewport(
    monkeypatch, tmp_path: Path
):
    monkeypatch.setattr(
        osm_trajectory_map,
        "_fetch_tile",
        lambda cache_root, zoom, x, y: Image.new("RGB", (256, 256)),
    )
    world_points = np.array([[0.0, 0.0], [6 * 256 - 1, 7 * 256 - 1]])

    mosaic, _extent = osm_trajectory_map.build_osm_mosaic(
        world_points,
        tmp_path,
        padding_pixels=0,
    )

    assert mosaic.size == (6 * 256, 7 * 256)

