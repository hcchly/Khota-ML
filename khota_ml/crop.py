"""Shared full-resolution box cropping used by severity training and inference."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
from PIL import Image

from .config import CROP_EXPAND


def crop_box(image: Image.Image | np.ndarray, box, expand: float = CROP_EXPAND):
    """Crop a YOLO-normalised ``(x_center, y_center, width, height)`` box.

    Expansion is a fraction of box width/height on each side. Coordinates are
    clipped to the stored image bounds; no padding or resizing is performed.
    PIL images return PIL crops, and NumPy arrays return array crops.
    """
    if isinstance(image, Image.Image):
        width, height = image.size
        is_pil = True
    elif isinstance(image, np.ndarray) and image.ndim >= 2:
        height, width = image.shape[:2]
        is_pil = False
    else:
        raise TypeError("image must be a PIL image or a NumPy array")
    if width <= 0 or height <= 0:
        raise ValueError("image must have positive width and height")
    if expand < 0:
        raise ValueError("expand must be non-negative")

    if isinstance(box, Mapping):
        values = (box["x_center"], box["y_center"], box["width"], box["height"])
    else:
        try:
            if len(box) != 4:
                raise ValueError
        except (TypeError, ValueError):
            raise ValueError("box must contain normalized x_center, y_center, width, height")
        values = box
    cx, cy, bw, bh = (float(value) for value in values)
    if not np.isfinite([cx, cy, bw, bh]).all() or bw <= 0 or bh <= 0:
        raise ValueError("box coordinates must be finite and width/height positive")

    x0 = int(np.floor((cx - bw / 2 - expand * bw) * width))
    y0 = int(np.floor((cy - bh / 2 - expand * bh) * height))
    x1 = int(np.ceil((cx + bw / 2 + expand * bw) * width))
    y1 = int(np.ceil((cy + bh / 2 + expand * bh) * height))
    x0, y0 = max(0, min(width, x0)), max(0, min(height, y0))
    x1, y1 = max(0, min(width, x1)), max(0, min(height, y1))
    if x1 <= x0 or y1 <= y0:
        raise ValueError("box does not intersect the image")
    if is_pil:
        return image.crop((x0, y0, x1, y1))
    return image[y0:y1, x0:x1].copy()
