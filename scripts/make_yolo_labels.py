"""Generate train/val YOLOv8 image and label folders from canonical CSVs."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from khota_ml.config import DETECTION_CLASSES, NO_DEFECT  # noqa: E402
from khota_ml.data import find_image  # noqa: E402

ANNOTATIONS = ROOT / "data" / "annotations.csv"
SPLIT = ROOT / "data" / "split.csv"
OUTPUT = ROOT / "datasets" / "yolo"


def main() -> None:
    annotations = pd.read_csv(ANNOTATIONS, dtype={"image_id": str})
    split_frame = pd.read_csv(SPLIT, dtype={"image_id": str})
    joined = annotations.merge(split_frame[["image_id", "split"]], on="image_id", how="left", validate="many_to_one")
    if joined.split.isna().any():
        raise ValueError("some annotation images are missing from split.csv")
    unknown = set(joined.split) - {"train", "val", "test"}
    if unknown:
        raise ValueError(f"unknown split values: {sorted(unknown)}")

    counts = {}
    for split in ("train", "val"):
        part = joined[joined.split == split]
        images_dir = OUTPUT / split / "images"
        labels_dir = OUTPUT / split / "labels"
        images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)
        by_image = part.groupby(["image_id", "image_file"], sort=True, dropna=False)
        image_count = box_count = empty_count = 0
        for (_, image_file), rows in by_image:
            source = find_image(str(image_file), ROOT / "raw")
            target_image = images_dir / Path(str(image_file)).name
            target_label = labels_dir / (Path(str(image_file)).stem + ".txt")
            if target_image.exists() or target_image.is_symlink():
                target_image.unlink()
            target_image.symlink_to(source.resolve())
            lines = []
            for row in rows.itertuples(index=False):
                if row.defect_class == NO_DEFECT:
                    continue
                class_id = DETECTION_CLASSES.index(row.defect_class)
                lines.append(f"{class_id} {row.x_center:.10g} {row.y_center:.10g} {row.width:.10g} {row.height:.10g}")
            target_label.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
            image_count += 1
            box_count += len(lines)
            empty_count += int(not lines)
        counts[split] = (image_count, box_count, empty_count)

    data_yaml = {
        "path": str(OUTPUT.resolve()),
        "train": "train/images",
        "val": "val/images",
        "names": list(DETECTION_CLASSES),
    }
    (OUTPUT / "data.yaml").write_text(yaml.safe_dump(data_yaml, sort_keys=False), encoding="utf-8")
    for split, (images, boxes, empty) in counts.items():
        print(f"{split}: {images} images, {boxes} boxes, {empty} empty-label background images")
    print(f"Wrote {OUTPUT / 'data.yaml'} (no test folders created).")


if __name__ == "__main__":
    main()
