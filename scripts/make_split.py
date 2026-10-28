"""Create the single location-grouped train/val/test split (AGENTS.md section 4)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from khota_ml.config import DETECTION_CLASSES, NO_DEFECT, SEED, SEVERITY_CLASSES  # noqa: E402

ANNOTATIONS = ROOT / "data" / "annotations.csv"
SPLIT_OUT = ROOT / "data" / "split.csv"
STATS_OUT = ROOT / "reports" / "split_stats.csv"
SPLIT_NAMES = ("train", "val", "test")
SPLIT_FRACTIONS = {"train": 0.70, "val": 0.15, "test": 0.15}


def _image_table(annotations: pd.DataFrame) -> pd.DataFrame:
    rows = annotations[["image_id", "location_id", "defect_class", "severity"]].copy()
    rows["location_id"] = rows["location_id"].fillna("").astype(str)
    if (rows.location_id == "").any():
        raise ValueError("annotations.csv contains images without a location_id")
    consistency = rows.groupby("image_id").location_id.nunique()
    if (consistency != 1).any():
        raise ValueError("an image_id maps to multiple location_id values")

    # Stratification uses a single, deterministic image-level label; acceptance
    # is checked against the full box-level distributions below.
    image_rows = []
    for image_id, group in rows.groupby("image_id", sort=False):
        defects = group.loc[group.defect_class != NO_DEFECT, "defect_class"]
        if defects.empty:
            stratum = NO_DEFECT
        else:
            defect = defects.value_counts().sort_values(ascending=False).index[0]
            severities = group.loc[group.defect_class != NO_DEFECT, "severity"]
            severity = severities.value_counts().sort_values(ascending=False).index[0]
            stratum = f"{defect}|{severity}"
        image_rows.append((image_id, group.location_id.iloc[0], stratum))
    return pd.DataFrame(image_rows, columns=["image_id", "location_id", "stratum"])


def _box_counts(annotations: pd.DataFrame, image_ids: set[str]) -> tuple[int, dict[str, int], dict[str, int]]:
    boxes = annotations[(annotations.image_id.isin(image_ids)) & (annotations.defect_class != NO_DEFECT)]
    severity = boxes.severity.value_counts().to_dict()
    defect = boxes.defect_class.value_counts().to_dict()
    return len(boxes), severity, defect


def _passes(annotations: pd.DataFrame, assignments: dict[str, str]) -> bool:
    whole_ids = set(assignments)
    total_boxes, total_severity, total_defect = _box_counts(annotations, whole_ids)
    if total_boxes == 0:
        return False
    for split in ("val", "test"):
        ids = {image_id for image_id, name in assignments.items() if name == split}
        n_boxes, severity, defect = _box_counts(annotations, ids)
        if severity.get(SEVERITY_CLASSES[2], 0) < 10 or defect.get(DETECTION_CLASSES[0], 0) < 15:
            return False
        if n_boxes == 0:
            return False
        for label in SEVERITY_CLASSES:
            if abs(severity.get(label, 0) / n_boxes - total_severity.get(label, 0) / total_boxes) > 0.10:
                return False
        for label in DETECTION_CLASSES:
            if abs(defect.get(label, 0) / n_boxes - total_defect.get(label, 0) / total_boxes) > 0.10:
                return False
    return True


def _stats(annotations: pd.DataFrame, split_frame: pd.DataFrame) -> pd.DataFrame:
    merged = annotations.merge(split_frame[["image_id", "split"]], on="image_id", how="left", validate="many_to_one")
    image_counts = {}
    image_table = merged[["image_id", "split", "defect_class"]].drop_duplicates()
    for split in SPLIT_NAMES:
        image_part = image_table[image_table.split == split]
        image_counts[split] = {
            "images_total": int(image_part.image_id.nunique()),
            "images_with_boxes": int(image_part.loc[image_part.defect_class != NO_DEFECT, "image_id"].nunique()),
            "images_without_boxes": int(image_part.loc[image_part.defect_class == NO_DEFECT, "image_id"].nunique()),
        }

    box_rows = merged[merged.defect_class != NO_DEFECT]
    rows = []
    for split in SPLIT_NAMES:
        for defect in DETECTION_CLASSES:
            for severity in SEVERITY_CLASSES:
                count = int(((box_rows.split == split) & (box_rows.defect_class == defect) & (box_rows.severity == severity)).sum())
                rows.append({"split": split, "defect_class": defect, "severity": severity,
                             "box_count": count, **image_counts[split]})
    return pd.DataFrame(rows)


def main() -> None:
    if SPLIT_OUT.exists():
        raise FileExistsError(f"{SPLIT_OUT.relative_to(ROOT)} already exists; the shared split must not be regenerated")
    annotations = pd.read_csv(ANNOTATIONS, dtype={"image_id": str})
    images = _image_table(annotations)
    min_stratum = images.stratum.value_counts().min()
    n_splits = min(7, int(min_stratum))
    if n_splits < 3:
        raise RuntimeError("not enough image-level strata for grouped split generation")

    accepted = None
    accepted_seed = None
    for seed in range(SEED, 1042):
        splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        folds = [set(images.iloc[indices].image_id) for _, indices in splitter.split(images.image_id, images.stratum, groups=images.location_id)]
        best_pair = None
        best_cost = float("inf")
        for val_fold in range(n_splits):
            for test_fold in range(val_fold + 1, n_splits):
                val_ids, test_ids = folds[val_fold], folds[test_fold]
                train_ids = set(images.image_id) - val_ids - test_ids
                sizes = (len(train_ids), len(val_ids), len(test_ids))
                targets = (0.70 * len(images), 0.15 * len(images), 0.15 * len(images))
                cost = sum(((size - target) / max(target, 1)) ** 2 for size, target in zip(sizes, targets))
                if cost < best_cost:
                    best_cost, best_pair = cost, (train_ids, val_ids, test_ids)
        if best_pair is None:
            continue
        assignment = {image_id: split for split, ids in zip(SPLIT_NAMES, best_pair) for image_id in ids}
        if len(assignment) == len(images) and _passes(annotations, assignment):
            accepted, accepted_seed = assignment, seed
            break

    if accepted is None:
        raise RuntimeError("no valid grouped split found for seeds 42 through 1041")

    split_frame = images[["image_id", "location_id"]].copy()
    split_frame["split"] = split_frame.image_id.map(accepted)
    split_frame = split_frame[["image_id", "location_id", "split"]].sort_values("image_id")
    if (split_frame.groupby("location_id").split.nunique() != 1).any():
        raise RuntimeError("internal error: a location was assigned to more than one split")
    SPLIT_OUT.parent.mkdir(parents=True, exist_ok=True)
    split_frame.to_csv(SPLIT_OUT, index=False)
    stats = _stats(annotations, split_frame)
    STATS_OUT.parent.mkdir(parents=True, exist_ok=True)
    stats.to_csv(STATS_OUT, index=False)
    print(f"Accepted seed: {accepted_seed}")
    print("Image counts:")
    print(stats[["split", "images_total", "images_with_boxes", "images_without_boxes"]].drop_duplicates().to_string(index=False))
    print(f"Wrote {SPLIT_OUT.relative_to(ROOT)} and {STATS_OUT.relative_to(ROOT)}")
    print("Batch 3 no-GPS groups were kept intact (nogps_01 and nogps_02).")


if __name__ == "__main__":
    main()
