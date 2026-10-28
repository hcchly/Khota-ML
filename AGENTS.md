# Khota ML — shared rules (read before any change)

Khota is a KSU graduation project that detects sidewalk problems in street photos and rates their severity.
Two people work on two models in this repository. These rules keep their work compatible. **Do not break any rule
below. If a task seems to need breaking one, stop and ask.**

## 1. The two models

| Model | Owner | Input | Output |
|---|---|---|---|
| Model 1 — Detection (YOLOv8) | Friend | full photo | boxes with a type: `curb_ramp_issue` or `surface_damage` |
| Model 2 — Severity (classifier) | Haifa | one cropped box | `low` / `medium` / `high` |

At the end they run as one pipeline: photo → detection → crop (rule 5) → severity.
Both models use **the same boxes**: the boxes the team drew in Roboflow. There is one annotation, not two.

## 2. Class IDs (never change, never reorder)

Detection classes (same as `Class_ID` in the Excel and in the Roboflow export):
- `0` = `curb_ramp_issue`
- `1` = `surface_damage`

Severity classes:
- `0` = `low`
- `1` = `medium`
- `2` = `high`

Also used in the data, never as a training class:
- `no_defect` — a photo with no target defect. It stays in **detection** data as a background image with an empty
  label file and is **excluded** from severity data.

Not Assessable and duplicate photos are **not** in the data at all: they are listed in the Excel sheet
`Excluded_Images` and are removed from both models. There is no `not_assessable` severity.

All class names live only in `khota_ml/config.py`. Every script imports them from there.

## 3. Data sources and the single label file

### 3.1 Sources (do not edit by hand)

- `data/source/Khota_Model_Final.xlsx` — the team's final label table:
  - `Batch_1`, `Batch_2`, `Batch_3`: one row per box —
    `Image_ID, Box_Index, Class_ID, Class_Name, x_center, y_center, width, height, Severity, Reason_Codes`.
    `Box_Index` is the 0-based line order inside the Roboflow YOLO label file, so every box has its own severity.
  - `Image_Index`: every image the models use, with its Roboflow image filename, label filename and `Box_Count`.
  - `Empty_Images`: the `no_defect` images (0 boxes).
  - `Excluded_Images`: Not Assessable and Duplicate images. Never use them.
- `raw/` — the three Roboflow **YOLOv8** exports (Batch 1 v3, Batch 2 v3, Batch 3 v2), downloaded from the shared
  Google Drive folder `all_batches_boxes`. Not committed (too large). Scripts find files **by filename**
  anywhere under `raw/`, so the folder layout inside `raw/` does not matter.
- `data/source/gps_batch1_2.csv` — GPS of batch 1 and batch 2 photos, from the batch Excel files.

### 3.2 `data/annotations.csv` — the only label file both models read

Built by `scripts/convert_annotations.py` (already written; do not rewrite it). One row per box:

`image_id, batch, image_file, box_id, location_id, defect_class, severity, x_center, y_center, width, height, reason_codes`

- `image_file` is the Roboflow image filename under `raw/` (e.g. `RYD-0001_jpg.rf.c868….jpg`).
- `box_id` = `Box_Index`. Box coordinates are YOLO-normalised (0–1).
- `severity` is one of `low`, `medium`, `high`.
- A `no_defect` image has one row with `defect_class = no_defect` and empty box, box_id and severity.
- `reason_codes` is kept for the report only. No model trains on it.

Run `python scripts/convert_annotations.py --raw raw/` once after downloading the exports. It checks every box in
the Excel against the Roboflow label files (file exists, line count, class, coordinates) and stops on any mismatch.

YOLO `.txt` label files for training are **generated** from `annotations.csv` by `scripts/make_yolo_labels.py`.
Never edit them by hand.

### 3.3 Label changes

Labels are fixed. A change is made in Roboflow **and** the Excel, then `annotations.csv` is rebuilt, and the change
is recorded in `data/CHANGELOG.md` (date, image_id, box_id, old → new, reason). If boxes or their order change in a
label file, `Box_Index` in the Excel must be regenerated. No label change is ever made after test results have been
seen.

### 3.4 Dataset at the time of writing (check against the script output)

| | low | medium | high | total |
|---|---|---|---|---|
| curb_ramp_issue | 85 | 45 | 28 | 158 |
| surface_damage | 143 | 192 | 88 | 423 |
| total | 228 | 237 | 116 | **581 boxes** |

694 images (427 with boxes, 267 `no_defect`), 37 excluded. Batch 1: 389 images / 291 boxes, batch 2: 255 / 232,
batch 3: 50 / 58. Images have 1–4 boxes.

## 4. One split for both models

- `data/locations.csv` (`image_id, batch, latitude, longitude, gps_source, location_id`) is built by
  `scripts/make_locations.py` (already written). Photos within 50 m of each other (chained) share a `location_id`.
  Photos without GPS are grouped by runs of consecutive Image_IDs (`nogps_xx`).
- **Batch 3 has no GPS yet** (`RYD-0683 … RYD-0733`, 50 images, 43 of the 116 `high` boxes). Until GPS is added
  it is one big group and lands entirely in one split. If the original batch 3 photos are available, run
  `python scripts/make_locations.py --originals <folder of the original batch 3 photos>` before making the split
  to read their EXIF GPS. If not, the balance check below still finds a valid split; note in the report that
  batch 3 was kept together.
- `data/split.csv`: `image_id, location_id, split` with split in `train`, `val`, `test`.
- Created once by `scripts/make_split.py`, then committed and never regenerated.
- Grouped by `location_id`: all photos of one location are in one split.
- Target 70 / 15 / 15 by image count, starting at `seed = 42`.
- Accept a seed only if **val and test each** contain at least 10 `high` boxes and at least 15 `curb_ramp_issue`
  boxes, **and** in each split the share of every severity class and of every defect class is within
  10 percentage points of its share in the whole dataset. Otherwise try the next seed (43, 44, … up to 1041).
  Print the accepted seed; stop if none passes.
- The script prints and saves `reports/split_stats.csv`: box counts per split × defect class × severity, and
  image counts per split (with and without boxes).

## 5. Crop rule (shared by training and the pipeline)

- Implemented once in `khota_ml/crop.py` as `crop_box(image, box, expand=0.20)`.
- Expand the box by 20% of its width on the left and right, and 20% of its height on the top and bottom.
  Clip to the image edges. Keep the aspect ratio. No padding.
- The severity model is trained on crops of the **human** boxes made with this function. The pipeline crops the
  **detected** boxes with the same function. Never write a second crop function.
- Crops are cut from the image at its **full stored resolution**, before any resize.

## 6. The test split is locked

- No training, tuning, threshold choice, model choice or early stopping uses `test`.
- All comparisons use `val`.
- Only `scripts/final_eval.py` reads `test`, run once at the end on the chosen models.

## 7. Metrics (same names everywhere)

**Main metric — the one used to compare runs and choose models:**
- Detection: `mAP50` on val.
- Severity: `macro_f1` on val.

Also reported:
- Detection, per class and overall: `mAP50-95`, `precision`, `recall`.
- Severity: `accuracy`, F1 per class, confusion matrix (saved as PNG and CSV).

Accuracy is never the main severity metric: the classes are imbalanced (`high` is 20% of boxes).

Severity baseline: the majority class of **train** predicted for every val (and, at the end, test) crop. Report its
`accuracy` and `macro_f1` next to every severity result. On the whole dataset this is `medium`:
accuracy 0.408, macro_f1 0.193 (the split's train set may give slightly different numbers).

Pipeline (final only): severity `macro_f1` on the boxes detection found, plus the detection metrics.

## 8. Training rules

- `seed = 42` for Python, NumPy and PyTorch.
- Images are the Roboflow JPGs in `raw/`, found by `image_file`.
- Detection: start from pretrained `yolov8n.pt` or `yolov8s.pt`, `imgsz = 640`, early stopping on val.
  Use YOLO's built-in augmentation; add no other. `no_defect` images go in as background images with empty labels.
  `curb_ramp_issue` has far fewer boxes than `surface_damage` (158 vs 423): always report per-class mAP50.
  `imgsz = 1024` may be tried as one separate run (small `low` defects).
- Severity: pretrained backbone, class weights from train counts (inverse frequency).
- Severity augmentation on **train only**, and only: horizontal flip, rotation within ±10°,
  brightness/contrast within ±20%. Never strong zoom or crop, blur, or strong colour changes, because they change
  how severe a defect looks.
- Run the first severity experiment without augmentation and one with it, to show the difference.
- Severity input: compare the 20% expanded crop with the full image as one experiment (rubric "image context rule");
  keep the input that scores better on val.

## 9. Experiment log

Every run appends one row to `experiments/experiments.csv`:
`run_id, date, owner, model, change_from_previous, key_settings, val_main_metric, val_other_metrics, notes`.
`val_main_metric` is `mAP50` for detection and `macro_f1` for severity. Change one thing per run.
Outputs go in `experiments/<run_id>/`.

## 10. Repository layout

```
data/            annotations.csv, locations.csv, split.csv, CHANGELOG.md
data/source/     Khota_Model_Final.xlsx, gps_batch1_2.csv (inputs; do not edit)
raw/             Roboflow YOLOv8 exports (not committed)
khota_ml/        config.py, crop.py, metrics.py, data.py (shared code)
scripts/         convert_annotations.py, make_locations.py, make_yolo_labels.py, make_split.py,
                 baseline.py, final_eval.py
detection/       YOLO training code
severity/        classifier training code
experiments/     experiments.csv, one folder per run
reports/         tables and figures for the report
```

Shared code in `khota_ml/` and `scripts/` changes only when both owners agree. Each owner edits only their own
folder (`detection/` or `severity/`).
