# Prompts for Codex

`AGENTS.md` is at the root of the repository. Codex reads it on its own, so the rules apply to you and to your
friend.

Before prompt 1, one person does this **by hand**:

1. Download the shared Google Drive folder `all_batches_boxes` (the three Roboflow YOLOv8 exports).
2. Put everything inside `raw/` in the repository (unzip any zip files there). The layout inside `raw/` does not
   matter.
3. Optional but recommended: put the original batch 3 photos (`RYD-0683` … `RYD-0733`, HEIC or JPG, named by
   Image_ID) in `raw/batch3_originals/` so their GPS can be read.

Run prompt 1 first, **once**, by one person. Commit its output. Then each person runs their own prompt.

---

## Prompt 1: shared setup (one person)

```
Read AGENTS.md first and follow it exactly.

The labels are already converted. Do not rewrite scripts/convert_annotations.py or scripts/make_locations.py.

1. Install what is needed: pandas, openpyxl, numpy, pillow, pillow-heif, ultralytics, torch, torchvision, scikit-learn, matplotlib.
2. If raw/batch3_originals/ exists, run: python scripts/make_locations.py --originals raw/batch3_originals
   Otherwise run: python scripts/make_locations.py
   Show me the output.
3. Run: python scripts/convert_annotations.py --raw raw/
   If it stops, show me every problem and stop. Do not change the Excel or the labels yourself.
4. Print the image sizes (width x height) found in raw/ for the images in data/annotations.csv, as a count per size.
   If the images were resized by Roboflow to 640 or smaller, warn me, because severity crops need full resolution.
5. Write khota_ml/crop.py (section 5), khota_ml/metrics.py (section 7) and khota_ml/data.py (load annotations.csv
   and split.csv, find an image under raw/ by image_file).
6. Write scripts/make_split.py exactly as in AGENTS.md section 4 and run it. Show me reports/split_stats.csv and the
   accepted seed.
7. Write scripts/make_yolo_labels.py: from annotations.csv and split.csv, build datasets/yolo/{train,val}/images and
   labels (copy or link the images from raw/, one label line per box: class_id x y w h; an empty label file for
   no_defect images) and datasets/yolo/data.yaml with names in the order of khota_ml/config.py. Do not build test
   folders. Run it and print the counts per split.
8. Write scripts/baseline.py: majority class of train predicted for every val box; print accuracy and macro_f1
   (section 7) and add its row to experiments/experiments.csv.

Do not train any model yet. Do not read the test split except to count boxes in the split stats.
```

---

## Prompt 2: Haifa, severity model

```
Read AGENTS.md first and follow it exactly. I own severity/ and must not change shared code without asking.

Build the severity classifier:
1. Use data/annotations.csv and data/split.csv. Use only rows with defect_class curb_ramp_issue or surface_damage (no no_defect).
2. Crop every box with khota_ml.crop.crop_box (expand=0.20) from the full-resolution image. Resize the crop to 224x224 for the model.
3. Use a pretrained backbone (start with EfficientNet-B0 or ResNet-18) and class weights from train counts.
4. Run 1: no augmentation. Run 2: the augmentation allowed in AGENTS.md section 8, train only.
   Run 3: the best of runs 1-2 with the full image instead of the crop (section 8, severity input).
5. For each run report on val: macro_f1 (main), accuracy, F1 per class, confusion matrix, and the baseline numbers next to them.
   Also report macro_f1 separately for curb_ramp_issue boxes and surface_damage boxes.
   Append one row per run to experiments/experiments.csv.

Never use the test split.
```

---

## Prompt 3: friend, detection model

```
Read AGENTS.md first and follow it exactly. I own detection/ and must not change shared code without asking.

Build the detection model:
1. Use datasets/yolo/data.yaml made by scripts/make_yolo_labels.py (train and val only; no_defect images are background with empty label files).
2. Train YOLOv8 from pretrained yolov8n.pt, imgsz 640, seed 42, early stopping on val, built-in augmentation only.
3. Report on val, per class and overall: mAP50 (main), mAP50-95, precision, recall. Append the run to experiments/experiments.csv.
4. Second run: same settings with yolov8s.pt. Third run (optional): the better model with imgsz 1024. One change per run.

Never use the test split.
```

---

## Prompt 4: final evaluation, Release 1 (once, at the end of Sprint 2)

Run it only when both models are final and nothing will change.

```
Read AGENTS.md first and follow it exactly.

Write scripts/final_eval.py and run it once on the chosen models:
1. Detection metrics on test: mAP50, mAP50-95, precision, recall, per class and overall.
2. Severity on test using the human boxes: macro_f1, accuracy, F1 per class, confusion matrix, with the majority baseline next to it.
Do not build the full pipeline yet; that is Release 2.
Save every table and confusion matrix in reports/final/.
```

---

## Prompt 5: full pipeline, Release 2 (Sprint 3, not now)

```
Read AGENTS.md first and follow it exactly.

Add the full pipeline to scripts/final_eval.py: detection -> khota_ml.crop.crop_box -> severity.
Report severity macro_f1 on the detected boxes that match a human box (IoU >= 0.5), next to the Release 1 results.
Save the results in reports/final/pipeline/.
```
