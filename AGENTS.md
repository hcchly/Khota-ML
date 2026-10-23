# Khota ML — shared rules (read before any change)

Khota is a KSU graduation project that detects sidewalk problems in street photos and rates their severity.
Two people work on two models in this repository. These rules keep their work compatible. **Do not break any rule
below. If a task seems to need breaking one, stop and ask.**

## 1. The two models

| Model | Owner | Input | Output |
|---|---|---|---|
| Detection (YOLOv8) | Friend | full photo | boxes with a type |
| Severity (classifier) | Haifa | one cropped box | Low / Medium / High |

At the end they run as one pipeline: photo → detection → crop (rule 5) → severity.

## 2. Class IDs (never change, never reorder)

Detection classes:
- `0` = `curb_ramp_issue`
- `1` = `surface_damage`

Severity classes:
- `0` = `low`
- `1` = `medium`
- `2` = `high`

Also used in the data, never as a training class:
- `not_assessable` — a defect that cannot be rated. It stays in detection data and is **excluded** from severity data.
- `no_defect` — a photo with no target defect. It stays in detection data as a background image with an empty
  label file and is **excluded** from severity data.

All class names live only in `khota_ml/config.py`. Every script imports them from there.

## 3. Single source of truth for labels

- `data/annotations.csv` is the only label file both models read. One row per box:
  `image_id, box_id, location_id, defect_class, severity, x_center, y_center, width, height`
  - box coordinates are YOLO-normalised (0–1);
  - `severity` is one of `low`, `medium`, `high`, `not_assessable`;
  - a `no_defect` image has one row with `defect_class = no_defect` and empty box and severity.
- YOLO `.txt` label files are **generated** from `annotations.csv` by `scripts/make_yolo_labels.py`.
  Never edit them by hand.
- Labels are fixed after the team's consistency review. A label change after that is made in `annotations.csv`
  only, recorded in `data/CHANGELOG.md` (date, image_id, old → new, reason), and **never** made after test
  results have been seen.

## 4. One split for both models

- `data/split.csv`: `image_id, location_id, split` with split in `train`, `val`, `test`.
- Created once by `scripts/make_split.py`, then committed and never regenerated.
- Grouped by `location_id`: all photos of one location are in one split.
- Target 70 / 15 / 15 by image count, `seed = 42`.
- Accept the split only if **val and test each** contain at least 10 `high` boxes and at least 15
  `curb_ramp_issue` boxes. Otherwise try the next seed (43, 44, …) and print the seed that was accepted.
- The script prints and saves `reports/split_stats.csv`: box counts per split × defect class × severity.

## 5. Crop rule (shared by training and the pipeline)

- Implemented once in `khota_ml/crop.py` as `crop_box(image, box, expand=0.20)`.
- Expand the box by 20% of its width on the left and right, and 20% of its height on the top and bottom.
  Clip to the image edges. Keep the aspect ratio. No padding.
- The severity model is trained on crops of the **human** boxes made with this function. The pipeline crops the
  **detected** boxes with the same function. Never write a second crop function.

## 6. The test split is locked

- No training, tuning, threshold choice, model choice or early stopping uses `test`.
- All comparisons use `val`.
- Only `scripts/final_eval.py` reads `test`, run once at the end on the chosen models.

## 7. Metrics (same names everywhere)

Detection, on val, per class and overall: `mAP50`, `mAP50-95`, `precision`, `recall`.

Severity, on val: `accuracy`, `macro_f1`, F1 per class, confusion matrix (saved as PNG and CSV).

Severity baseline: the majority class of **train** predicted for every val (and, at the end, test) crop.
Report its `accuracy` and `macro_f1` next to every severity result.

Pipeline (final only): severity `macro_f1` on the boxes detection found, plus the detection metrics.

## 8. Training rules

- `seed = 42` for Python, NumPy and PyTorch.
- Detection: start from pretrained `yolov8n.pt` or `yolov8s.pt`, `imgsz = 640`, early stopping on val.
  Use YOLO's built-in augmentation; add no other.
- Severity: pretrained backbone, class weights from train counts (inverse frequency).
- Severity augmentation on **train only**, and only: horizontal flip, rotation within ±10°,
  brightness/contrast within ±20%. Never strong zoom or crop, blur, or strong colour changes, because they change
  how severe a defect looks.
- Run the first severity experiment without augmentation and one with it, to show the difference.

## 9. Experiment log

Every run appends one row to `experiments/experiments.csv`:
`run_id, date, owner, model, change_from_previous, key_settings, val_main_metric, val_other_metrics, notes`.
`val_main_metric` is `mAP50` for detection and `macro_f1` for severity. Change one thing per run.
Outputs go in `experiments/<run_id>/`.

## 10. Repository layout

```
data/            annotations.csv, split.csv, CHANGELOG.md, images/ (not committed if too large)
khota_ml/        config.py, crop.py, metrics.py, data.py (shared code)
scripts/         make_yolo_labels.py, make_split.py, baseline.py, agreement.py, final_eval.py
detection/       YOLO training code
severity/        classifier training code
experiments/     experiments.csv, one folder per run
reports/         tables and figures for the report
```

Shared code in `khota_ml/` and `scripts/` changes only when both owners agree. Each owner edits only their own
folder (`detection/` or `severity/`).
