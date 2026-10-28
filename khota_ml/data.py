"""Load shared labels/splits and resolve Roboflow image files under raw/."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]


def load_annotations(path: str | Path | None = None) -> pd.DataFrame:
    """Load the canonical annotations.csv."""
    source = Path(path) if path is not None else REPO_ROOT / "data" / "annotations.csv"
    return pd.read_csv(source)


def load_split(path: str | Path | None = None) -> pd.DataFrame:
    """Load split.csv (image_id, location_id, split)."""
    source = Path(path) if path is not None else REPO_ROOT / "data" / "split.csv"
    frame = pd.read_csv(source)
    required = {"image_id", "location_id", "split"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"split file missing required columns: {sorted(missing)}")
    return frame


@lru_cache(maxsize=8)
def _image_index(raw_dir: str) -> dict[str, Path]:
    root = Path(raw_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"raw image directory does not exist: {root}")
    index: dict[str, Path] = {}
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".heic", ".heif"}:
            if path.name in index:
                raise ValueError(f"duplicate filename under {root}: {path.name}")
            index[path.name] = path
    return index


def find_image(image_file: str, raw_dir: str | Path | None = None) -> Path:
    """Find an image by its annotation filename anywhere beneath raw/."""
    root = Path(raw_dir) if raw_dir is not None else REPO_ROOT / "raw"
    key = str(root.resolve())
    try:
        return _image_index(key)[image_file]
    except KeyError as exc:
        raise FileNotFoundError(f"image_file not found under {root}: {image_file}") from exc
