"""Evaluate the train-majority severity baseline on val only (AGENTS.md section 7)."""

from __future__ import annotations

import csv
import sys
from datetime import date
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from khota_ml.config import NO_DEFECT, SEVERITY_CLASSES  # noqa: E402
from khota_ml.data import load_annotations, load_split  # noqa: E402
from khota_ml.metrics import save_confusion_matrix, severity_metrics  # noqa: E402

EXPERIMENTS = ROOT / "experiments" / "experiments.csv"
FIELDS = ["run_id", "date", "owner", "model", "change_from_previous", "key_settings",
          "val_main_metric", "val_other_metrics", "notes"]


def main() -> None:
    annotations = load_annotations()
    splits = load_split()
    # Restrict the joined input before calculating anything: test rows are not
    # inspected for baseline labels or class counts.
    split_rows = splits[splits.split.isin(["train", "val"])][["image_id", "split"]]
    frame = annotations.merge(split_rows, on="image_id", how="inner", validate="many_to_one")
    frame = frame[frame.defect_class != NO_DEFECT].copy()
    train = frame[frame.split == "train"]
    val = frame[frame.split == "val"]
    if train.empty or val.empty:
        raise ValueError("train and val must both contain severity boxes")
    counts = train.severity.value_counts()
    majority = max(SEVERITY_CLASSES, key=lambda name: int(counts.get(name, 0)))
    result = severity_metrics(val.severity.tolist(), [majority] * len(val))

    print(f"Train majority class: {majority}")
    print(f"Val accuracy: {result['accuracy']:.3f}")
    print(f"Val macro_f1: {result['macro_f1']:.3f}")
    print("Val F1 per class:")
    for name, score in result["f1_per_class"].items():
        print(f"  {name}: {score:.3f}")

    row = {
        "run_id": "severity_majority_baseline",
        "date": date.today().isoformat(),
        "owner": "Haifa",
        "model": "severity_baseline",
        "change_from_previous": "Initial majority-class baseline",
        "key_settings": f"train majority={majority}; seed=42; val boxes={len(val)}",
        "val_main_metric": f"{result['macro_f1']:.6f}",
        "val_other_metrics": f"accuracy={result['accuracy']:.6f}; f1_per_class={result['f1_per_class']}",
        "notes": "Predicted the train majority class for every val defect box; test data not used.",
    }
    EXPERIMENTS.parent.mkdir(parents=True, exist_ok=True)
    already_logged = False
    if EXPERIMENTS.exists():
        existing = pd.read_csv(EXPERIMENTS, dtype=str)
        if "run_id" in existing and (existing.run_id == row["run_id"]).any():
            old = existing.loc[existing.run_id == row["run_id"]].iloc[-1]
            if old.val_main_metric != row["val_main_metric"]:
                raise ValueError(f"experiment {row['run_id']} is already logged with a different result")
            already_logged = True
    run_dir = ROOT / "experiments" / row["run_id"]
    csv_path, png_path = save_confusion_matrix(result, run_dir)
    if not already_logged:
        with EXPERIMENTS.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDS, lineterminator="\n")
            if handle.tell() == 0:
                writer.writeheader()
            writer.writerow(row)
        print(f"Appended {row['run_id']} to {EXPERIMENTS.relative_to(ROOT)}")
    else:
        print(f"Experiment row already present; did not append a duplicate.")
    print(f"Saved confusion matrix: {csv_path.relative_to(ROOT)}, {png_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
