"""Geometry helpers for undistorted, consistently sized OSM figure viewports."""

from __future__ import annotations

import math

import numpy as np
from PIL import Image


def fixed_aspect_viewport(
    world_points: np.ndarray,
    *,
    padding_pixels: int = 72,
    aspect_width: int = 5,
    aspect_height: int = 4,
    right_reserved_fraction: float = 0.0,
) -> tuple[int, int, int, int]:
    """Return an integer viewport containing all points at an exact aspect ratio."""
    if not 0.0 <= right_reserved_fraction < 1.0:
        raise ValueError("right_reserved_fraction must be in [0, 1)")

    points = np.asarray(world_points, dtype=float)
    raw_x_min = float(points[:, 0].min() - padding_pixels)
    raw_x_max = float(points[:, 0].max() + padding_pixels)
    raw_y_min = float(points[:, 1].min() - padding_pixels)
    raw_y_max = float(points[:, 1].max() + padding_pixels)

    content_width = raw_x_max - raw_x_min
    required_width = content_width / (1.0 - right_reserved_fraction)
    required_height = raw_y_max - raw_y_min
    unit = math.ceil(
        max(required_width / aspect_width, required_height / aspect_height)
    )
    width = aspect_width * unit
    height = aspect_height * unit
    reserved_width = math.ceil(width * right_reserved_fraction)
    usable_width = width - reserved_width
    while usable_width < content_width:
        unit += 1
        width = aspect_width * unit
        height = aspect_height * unit
        reserved_width = math.ceil(width * right_reserved_fraction)
        usable_width = width - reserved_width

    x_min = math.floor((raw_x_min + raw_x_max - usable_width) / 2.0)
    y_min = math.floor((raw_y_min + raw_y_max - height) / 2.0)
    x_max = x_min + width
    y_max = y_min + height
    if x_min + usable_width < raw_x_max:
        x_min = math.floor(raw_x_max - usable_width)
        x_max = x_min + width
    if y_max < raw_y_max:
        y_max = math.ceil(raw_y_max)
        y_min = y_max - height
    return x_min, x_max, y_min, y_max


def crop_mosaic_to_viewport(
    mosaic: Image.Image,
    mosaic_extent: tuple[float, float, float, float],
    viewport: tuple[int, int, int, int],
) -> Image.Image:
    """Crop a tile-aligned mosaic to an exact world-pixel viewport."""
    mosaic_x_min, _mosaic_x_max, mosaic_y_min, _mosaic_y_max = mosaic_extent
    x_min, x_max, y_min, y_max = viewport
    crop_box = (
        int(x_min - mosaic_x_min),
        int(y_min - mosaic_y_min),
        int(x_max - mosaic_x_min),
        int(y_max - mosaic_y_min),
    )
    return mosaic.crop(crop_box)
