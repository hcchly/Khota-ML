# Prompts for Codex

Put `AGENTS.md` at the root of the shared repository before anything else. Codex reads it on its own,
so the rules apply to you and to your friend.

Run prompt 1 first, **once**, by one person. Then each person runs their own prompt.

---

## Prompt 1: shared setup (one person, after the consistency review)

```
Read AGENTS.md first and follow it exactly.

Our labels are in: <write where the export is and its format, e.g. "a Roboflow YOLO export in raw/" or "a CVAT XML file raw/annotations.xml" and where severity and location are stored>.

1. Inspect that export and write scripts/convert_annotations.py that turns it into data/annotations.csv in the exact format of AGENTS.md section 3. Print counts per defect_class x severity and stop if any box has no severity or no location_id.
2. Create khota_ml/config.py with the class IDs of AGENTS.md section 2.
3. Write scripts/make_yolo_labels.py (section 3).
4. Write scripts/make_split.py and run it (section 4). Show me reports/split_stats.csv and the accepted seed.
5. Write khota_ml/crop.py (section 5) and khota_ml/metrics.py (section 7).
6. Write scripts/baseline.py: majority-class severity baseline, accuracy and macro_f1 on val (section 7). Add its row to experiments/experiments.csv.
7. Write scripts/agreement.py: given a CSV with columns image_id, box_id, annotator, severity, print Fleiss' kappa and Cohen's kappa for each pair, and list the boxes the annotators disagree on.

Do not train any model yet. Do not read the test split except to count boxes in the split stats.
```

---

## Prompt 2: Haifa, severity model

```
Read AGENTS.md first and follow it exactly. I own severity/ and must not change shared code without asking.

Build the severity classifier:
1. Use data/annotations.csv and data/split.csv. Exclude not_assessable and no_defect.
2. Crop every box with khota_ml.crop.crop_box (expand=0.20). Resize to 224x224 for the model.
3. Use a pretrained backbone (start with EfficientNet-B0 or ResNet-18) and class weights from train counts.
4. Run 1: no augmentation. Run 2: the augmentation allowed in AGENTS.md section 8, train only.
5. For each run report on val: accuracy, macro_f1, F1 per class, confusion matrix, and the baseline numbers next to them. Append one row per run to experiments/experiments.csv.

Never use the test split.
```

---

## Prompt 3: friend, detection model

```
Read AGENTS.md first and follow it exactly. I own detection/ and must not change shared code without asking.

Build the detection model:
1. Generate YOLO labels with scripts/make_yolo_labels.py and build the YOLO dataset folders from data/split.csv (train and val only; no_defect images go in as background with empty label files).
2. Write the YOLO data.yaml with names in the order of AGENTS.md section 2.
3. Train YOLOv8 from pretrained yolov8n.pt, imgsz 640, seed 42, early stopping on val, built-in augmentation only.
4. Report on val, per class and overall: mAP50, mAP50-95, precision, recall. Append the run to experiments/experiments.csv.

Never use the test split.
```

---

## Prompt 4: final evaluation (once, at the very end)

```
Read AGENTS.md first and follow it exactly.

Write scripts/final_eval.py and run it once on the chosen models:
1. Detection metrics on test.
2. Severity on test using the human boxes, with the baseline next to it.
3. Full pipeline on test: detection -> khota_ml.crop.crop_box -> severity, and severity macro_f1 on the detected boxes that match a human box (IoU >= 0.5).
Save every table and confusion matrix in reports/final/.
```
