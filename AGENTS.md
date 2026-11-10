# Khota ML — shared rules, version 2 (read before any change)

Khota is a KSU graduation project. Citizens photograph sidewalk problems in Riyadh; this repository holds the two
models that find the problems and rate their severity, and the code that connects them to the Khota app.
**Do not break any rule below. If a task seems to need breaking one, stop and ask.**

> **Version 2** (10 Nov 2026) replaces version 1. Severity is now **binary** (`low_medium` / `high`), the labels
> come from `Khota_Model_Final_Binary.xlsx`, and the dataset has 1,010 boxes. Version 1 (low / medium / high,
> 581 boxes) is kept at git tag `v1` and in `experiments/v1/` for the report only. Never mix v1 and v2 results,
> splits or checkpoints.

## 1. The two models and the pipeline

| Model | Owner | Folder | Input | Output |
|---|---|---|---|---|
| Model 1 — Detection (YOLOv8) | Hala | `detection/` | full photo | boxes with a type: `curb_ramp_issue` or `surface_damage` |
| Model 2 — Severity (classifier) | Haifa | `severity/` | one cropped box | `low_medium` or `high` |

Pipeline: photo → detection → `crop_box` (rule 5) → severity → result written to the report in Firebase (rule 11).
Both models use **the same boxes**: the boxes the team drew in Roboflow. There is one annotation, not two.

## 2. Class IDs (never change, never reorder)

Detection classes (same as `Class_ID` in the Excel and the order in every Roboflow `data.yaml`):
- `0` = `curb_ramp_issue`
- `1` = `surface_damage`

Severity classes (binary rubric, `Khota_Severity_Rubric_Binary`):
- `0` = `low_medium` (old Low + Medium: visible damage, the walking path or curb transition stays continuous)
- `1` = `high` (the damage interrupts or blocks the walking path or the curb transition)

Also used in the data, never as a training class:
- `no_defect` — a photo with no target defect. It stays in **detection** data as a background image with an empty
  label file and is **excluded** from severity data.

Not Assessable and duplicate photos are **not** in the data at all: they are in the Excel sheet `Excluded_Images`
and are removed from both models. There is no `not_assessable` severity class.

All class names live only in `khota_ml/config.py`. Every script imports them from there. Refer to the high class
by name (`config.HIGH`), never by index.

## 3. Data

### 3.1 Sources (do not edit by hand)

- `data/source/Khota_Model_Final_Binary.xlsx` — the team's final, confirmed label table:
  - `Boxes`: **one row per box** — `Batch, Image_ID, Box_Index, Class_ID, Class_Name, x_center, y_center, width,
    height, Area_%, Size_Check, Severity`. `Box_Index` `B01` = line 1 of the Roboflow label file, `B02` = line 2 …
    (in `annotations.csv` this becomes `box_id` 0, 1, … ).
  - `Batch_1`, `Batch_2`, `Batch_3`: one row per image + class + severity, with its boxes (`B01, B03`) and
    `Reason_Codes`.
  - `Image_Index`: every image the models use, with its Roboflow image filename, label filename and `Box_Count`.
  - `Empty_Images`: the `no_defect` images (0 boxes). `Excluded_Images`: never use them.
- `raw/` — the latest Roboflow **YOLOv8** exports of the three batches (taken **after 10 Nov 2026**, no resize, no
  augmentation). Not committed. Scripts find files **by filename** anywhere under `raw/`.
- `data/source/gps.csv` — GPS of every photo from the batch Excel files (batch 3 is included in v2).

### 3.2 `data/annotations.csv` — the only label file the models read

Built by `scripts/convert_annotations.py` (already written and checked; do not rewrite it). One row per box:

`image_id, batch, image_file, box_id, box_label, location_id, defect_class, severity, x_center, y_center, width,
height, reason_codes`

- `image_file` is the Roboflow image filename under `raw/`. Coordinates are YOLO-normalised (0–1).
- `severity` is `low_medium` or `high`. A `no_defect` image has one row with empty box, box_id and severity.
- `reason_codes` is for the report only. No model trains on it.

`python scripts/convert_annotations.py --raw raw/` checks the Excel against the Roboflow label files (file exists,
line count, class, coordinates) and stops on any mismatch. YOLO `.txt` files for training are **generated** from
`annotations.csv` by `scripts/make_yolo_labels.py`. Never edit them by hand.

### 3.3 Label changes

Labels are fixed. A change is made in Roboflow **and** the Excel, then `annotations.csv` is rebuilt, and the change
is recorded in `data/CHANGELOG.md`. If boxes or their order change in a label file, `Box_Index` must be regenerated.
No label change is ever made after test results have been seen.

### 3.4 Dataset v2 (from `convert_annotations.py`)

| | low_medium | high | total |
|---|---|---|---|
| curb_ramp_issue | 134 | 26 | 160 |
| surface_damage | 738 | 112 | 850 |
| total | 872 (86%) | 138 (14%) | **1,010 boxes** |

689 images (495 with boxes, 194 `no_defect`), 42 excluded (24 Not Assessable, 18 duplicate).
Batch 1: 388 images / 566 boxes, batch 2: 251 / 379, batch 3: 50 / 65. Images have 1–11 boxes.

**The data is imbalanced on purpose** (it reflects real sidewalks): `high` is 14% of boxes and `curb_ramp_issue`
16%. `curb_ramp_issue` + `high` has only 26 boxes (3 in val, 4 in test): report it, but never draw conclusions
from it alone.

## 4. One split for both models (already made — do not regenerate)

- `data/locations.csv`: photos within 50 m of each other (chained) share a `location_id`. 272 locations. Only
  RYD-0610 … 0621 have no GPS; they are kept together as `nogps_00`.
- `data/split.csv` (`image_id, location_id, split`) was made by `scripts/make_split.py` with **seed 42**, grouped
  by location, ~70/15/15. Rule: val and test each have ≥ 10 `high` boxes and ≥ 15 `curb_ramp_issue` boxes, and
  every class share within 10 percentage points of the whole dataset. Never regenerate it.

| split | images | with boxes | boxes | high | curb_ramp_issue |
|---|---|---|---|---|---|
| train | 491 | 353 | 695 | 95 | 118 |
| val | 99 | 71 | 160 | 21 | 21 |
| test | 99 | 71 | 155 | 22 | 21 |

## 5. Crop rule (shared by training and the pipeline)

- `khota_ml/crop.py` → `crop_box(image, box, expand=0.20)`: expand the box by 20% of its width on the left and
  right and 20% of its height on the top and bottom, clip to the image, no padding.
- The severity model is trained on crops of the **human** boxes. The pipeline crops the **detected** boxes with the
  same function. Never write a second crop function.
- Crops are cut from the image at its **full stored resolution**, before any resize.

## 6. The test split is locked

- No training, tuning, threshold choice, model choice or early stopping uses `test`.
- All comparisons use `val`.
- Only `scripts/final_eval.py` reads `test`, run **once** per release on the chosen models. Its results are final,
  even if they are worse than val.

## 7. Metrics (same names everywhere)

Main metric — the one used to compare runs and choose models:
- Detection: `mAP50` on val (also report per-class `mAP50`; `curb_ramp_issue` is the small class).
- Severity: `macro_f1` on val.

Also reported:
- Detection, per class and overall: `mAP50-95`, `precision`, `recall`.
- Severity: `recall_high`, `precision_high`, F1 per class, `accuracy`, confusion matrix (PNG + CSV), and
  `macro_f1` separately for `curb_ramp_issue` and `surface_damage` boxes.

**Accuracy is never the main severity metric.** Predicting `low_medium` for every box already gives val accuracy
0.869 but macro_f1 0.465 and recall_high 0 (the majority baseline, `experiments/severity_majority_baseline`).
Report the baseline next to every severity result.

Pipeline (Release 2): see prompt 6.

## 8. Training rules

- `seed = 42` for Python, NumPy and PyTorch. To show stability, repeat the selected run with seeds 43 and 44 and
  report mean ± std.
- Images are the Roboflow JPGs in `raw/`, found by `image_file`.
- Detection: pretrained `yolov8n.pt` / `yolov8s.pt`, `imgsz = 640` first, early stopping on val, YOLO's built-in
  augmentation only. `no_defect` images are background images with empty labels. `imgsz = 1024` may be one
  separate run (small defects).
- Severity: pretrained backbone; handle the imbalance with **one** method per run — class weights (inverse train
  frequency) **or** a weighted sampler — never both. A decision threshold for `high` may be chosen on **val**
  (store it with the checkpoint); default 0.5.
- Severity augmentation on **train only**, and only: horizontal flip, rotation within ±10°, brightness/contrast
  within ±20%. Never strong zoom or crop, blur, or strong colour changes (they change how severe a defect looks).
- Approved exception (10 Nov 2026): mild **box jitter** on train only — box centre ±10% of box size, width/height
  ×0.85–1.15, applied before `crop_box` — to simulate detector box error. Jittered val is report-only, never used
  for selection.
- Never oversample or augment val or test.

## 9. Experiment log

Every run appends one row to `experiments/experiments.csv` (v2 only):
`run_id, date, owner, model, change_from_previous, key_settings, val_main_metric, val_other_metrics, notes`.
Change one thing per run. Outputs go in `experiments/<run_id>/`. The chosen model of each owner is written in
`<folder>/SELECTED_MODEL.md` (run_id, checkpoint path, threshold, val metrics). Checkpoints are not committed
except the selected one (`git add -f`), or are uploaded to the shared Drive with the link in `SELECTED_MODEL.md`.

## 10. Repository layout

```
AGENTS.md / CLAUDE.md   these rules (CLAUDE.md just imports AGENTS.md)
data/            annotations.csv, locations.csv, split.csv, CHANGELOG.md (v1_split.csv is history)
data/source/     Khota_Model_Final_Binary.xlsx, gps.csv (inputs; do not edit)
raw/             Roboflow YOLOv8 exports (not committed)
khota_ml/        config.py, crop.py, metrics.py, data.py (shared code)
scripts/         convert_annotations.py, make_locations.py, make_split.py, make_yolo_labels.py, baseline.py,
                 final_eval.py
detection/       YOLO training (Hala)
severity/        classifier training (Haifa); train_v1.py is the v1 script, for reference
pipeline/        detection → crop → severity, used by evaluation and the service
service/         the analysis worker that connects the pipeline to the app's Firebase
experiments/     experiments.csv (v2), one folder per run; experiments/v1/ is history
reports/         tables and figures for the report (reports/final/ = test results)
docs/            PROMPTS.md (the prompts, in order), APP_INTEGRATION.md (the Firebase contract)
```

Shared code (`khota_ml/`, `scripts/`, `pipeline/`, `service/`) changes only when both owners agree. Each owner
edits only their own model folder.

## 11. Connection to the app (summary — full contract in `docs/APP_INTEGRATION.md`)

- The app (repo `hcchly/Khota`, Flutter + Firebase project `khota-988e6`) saves a report in Firestore
  `reports/{id}` with `analysisStatus = "processing"`, `issueClass = null`, `severity = null` and an `imageUrl`.
- The **analysis worker** (`service/`) finds reports with `analysisStatus == "processing"`, runs the pipeline on
  the image and writes the result with the Firebase Admin SDK. The app never runs the models.
- The worker writes only the fields listed in `docs/APP_INTEGRATION.md`. It never changes `reportStatus`, the
  user's fields or the image. Service-account keys are **never committed** (`.gitignore`), and nobody logs or
  copies report images outside Firebase.
- PIQS is not computed here yet; the worker stores what PIQS will need (issue class, severity, confidence per
  issue).
