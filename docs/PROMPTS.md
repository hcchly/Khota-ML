# Prompts — version 2

Give these to Claude Code (or Codex) inside the repository, one at a time, in order. The agent reads
`AGENTS.md` by itself (Claude Code through `CLAUDE.md`). Paste the text inside the box exactly.

| # | Prompt | Who | When |
|---|---|---|---|
| 0 | Download the data (by hand) | one person | now |
| 1 | Shared setup check | one person | now, once |
| 2 | Severity model | Haifa | after 1 |
| 3 | Detection model | Hala | after 1 (in parallel with 2) |
| 4 | Final evaluation, Release 1 (test, once) | one person | when 2 and 3 are final |
| 5 | Pipeline: detection → crop → severity | both agree | Release 2 |
| 6 | Pipeline evaluation (test, once) | one person | after 5 |
| 7 | Analysis worker (Firebase) | one person | after 5 |
| 8 | App integration — run in the **app** repo `hcchly/Khota` | app team | after 7 |
| 9 | End-to-end system test and timing | one person | after 8 |

Already done in version 2 (do not redo): `data/annotations.csv`, `data/locations.csv`, `data/split.csv`
(seed 42, never regenerate) and the majority baseline (`experiments/experiments.csv`).

---

## Step 0 — by hand: download the data

1. In Roboflow, export each batch again **now** (after the last fixes, e.g. RYD-0591 B01 = curb-ramp-issue):
   *Versions → Generate* with **no resize** and **no augmentation** (only auto-orient) → *Download → YOLOv8*.
2. Unzip the three exports inside `raw/` (any folder names). Make sure the images are there, not only the labels
   (the zips sent on 9 Nov were missing most images).
3. Do not commit `raw/`.

---

## Prompt 1 — shared setup check (one person, once)

```
Read AGENTS.md first and follow it exactly. This is version 2 (binary severity).

The labels, locations and split are already made. Do not rewrite scripts/convert_annotations.py,
scripts/make_locations.py or scripts/make_split.py, and never regenerate data/split.csv.

1. Install what is needed: pandas, openpyxl, numpy, pillow, ultralytics, torch, torchvision, scikit-learn, matplotlib.
2. Run: python scripts/convert_annotations.py --raw raw/
   It must say 0 problems, 689 images, 1010 boxes. If it stops, show me every problem and stop. Do not change the
   Excel or the labels yourself.
3. Run git diff --stat data/annotations.csv. It must be unchanged except for image_file names if Roboflow renamed the
   files. If boxes, classes or severities changed, show me the diff and stop.
4. Print the image sizes (width x height) of the images in data/annotations.csv, as a count per size. Warn me if
   any are 640 px or smaller (severity crops need full resolution).
5. Run scripts/make_yolo_labels.py and print the image and box counts per split (train, val). Check that the counts
   match AGENTS.md section 4.
6. Show me a summary and commit with the message "v2: data check and YOLO dataset".
```

---

## Prompt 2 — Haifa, severity model (v2)

```
Read AGENTS.md and severity/AGENTS.md first and follow them exactly. I own severity/ and must not change shared
code without asking. This is version 2: two classes, low_medium and high (config.SEVERITY_CLASSES).

Write severity/train.py (reuse the structure of severity/train_v1.py) and run these experiments, one change per run,
all on train, compared on val:
1. Data: rows of data/annotations.csv with defect_class curb_ramp_issue or surface_damage, split from data/split.csv.
   Crop every box with khota_ml.crop.crop_box (expand=0.20) from the full-resolution image, resize to 224x224.
2. Run A: ResNet-18 ImageNet-pretrained, crop input, no augmentation, class weights from train counts
   (the v1 winner, now binary). batch 32, lr 1e-4, max 20 epochs, early stopping on val macro_f1 (patience 5).
3. Run B: Run A + the augmentation allowed in AGENTS.md section 8 (train only).
4. Run C: the better of A/B but with a weighted random sampler instead of class weights (never both).
5. Run D: the best so far with EfficientNet-B0 instead of ResNet-18.
6. For the best run, choose the decision threshold for high on val (the value of p_high that maximises val
   macro_f1, searched from 0.10 to 0.90 in steps of 0.05) and report val results at 0.5 and at that threshold.
7. Repeat the best run with seeds 43 and 44 and report mean ± std of val macro_f1 and recall_high.

For every run report on val: macro_f1 (main), recall_high, precision_high, F1 per class, accuracy, confusion
matrix (PNG + CSV), macro_f1 for curb_ramp_issue boxes and for surface_damage boxes, and the majority baseline
(macro_f1 0.465, accuracy 0.869) next to them. Append one row per run to experiments/experiments.csv.
Save each run in experiments/<run_id>/.

Then write severity/predict.py with load_severity_model(path) and predict_severity(model, crops) returning
(label, p_high) per crop using the stored threshold, and severity/SELECTED_MODEL.md (run_id, checkpoint, threshold,
val metrics, mean ± std over seeds). Show me the comparison table before committing, then commit with
"v2 severity: runs A-D, selected <run_id>".

Never use the test split.
```

---

## Prompt 3 — Hala, detection model (v2)

```
Read AGENTS.md and detection/AGENTS.md first and follow them exactly. I own detection/ and must not change shared
code without asking. This is version 2 (1,010 boxes).

Write detection/train.py and run, one change per run, on datasets/yolo/data.yaml (train and val only; no_defect
images are background images with empty label files):
1. Run A: YOLOv8n from yolov8n.pt, imgsz 640, seed 42, epochs 100, patience 20, built-in augmentation only.
2. Run B: same with yolov8s.pt.
3. Run C: the better of A/B with imgsz 1024 (small defects).
4. For each run report on val, overall and per class (curb_ramp_issue, surface_damage): mAP50 (main), mAP50-95,
   precision, recall. Append one row per run to experiments/experiments.csv and save the plots and the
   confusion matrix in experiments/<run_id>/.
5. For the best run, choose the confidence threshold on val that maximises F1 over both classes (also report the
   per-class F1 at that threshold), and look at 10 val images with missed curb_ramp_issue boxes; describe the
   pattern in 3-4 sentences.
6. Repeat the best run with seeds 43 and 44 and report mean ± std of val mAP50.

Write detection/SELECTED_MODEL.md (run_id, weights path or Drive link, confidence threshold, val metrics) and
detection/predict.py with load_detector(path) and detect(model, image) returning a list of
(defect_class, confidence, x_center, y_center, width, height) in normalised coordinates, using the stored
threshold. Show me the comparison table before committing, then commit with "v2 detection: runs A-C, selected <run_id>".

Never use the test split.
```

---

## Prompt 4 — final evaluation, Release 1 (once, when both models are final)

```
Read AGENTS.md first and follow it exactly. Both selected models are final; nothing will be retrained after this.

Write scripts/final_eval.py --release 1 and run it once:
1. Detection on test (images of split == test, including no_defect images as background): mAP50, mAP50-95,
   precision, recall, overall and per class, with the selected confidence threshold for precision/recall.
2. Severity on test using the human boxes (crop_box on the test boxes): macro_f1, recall_high, precision_high,
   F1 per class, accuracy, confusion matrix, macro_f1 per defect class, and the majority baseline (train majority
   class) next to it. Use the stored threshold.
3. Put val and test results side by side in reports/final/release1_summary.md, plus the confusion matrices
   (PNG + CSV), and 6 example test images with predictions drawn (3 correct, 3 wrong).
Do not change any model, threshold or label after seeing these numbers. Commit with "v2 Release 1 final evaluation".
```

---

## Prompt 5 — pipeline: detection → crop → severity (Release 2)

```
Read AGENTS.md and pipeline/AGENTS.md first and follow them exactly.

Write pipeline/analyze.py with:
  load_models() -> reads detection/SELECTED_MODEL.md and severity/SELECTED_MODEL.md, loads both once.
  analyze_image(image: PIL.Image, models) -> dict:
     {status: "completed" | "no_issue_detected" | "not_assessable",
      issues: [{issue_class, severity, detection_confidence, p_high, box: {x, y, w, h}}] sorted high first, then
              by detection confidence,
      timings_ms: {quality, detection, severity, total}}
Steps: image-quality check (below) -> detection with the stored threshold -> khota_ml.crop.crop_box(expand=0.20)
on each detected box at full resolution -> severity with the stored threshold.

Image-quality check (AGENTS.md / docs/APP_INTEGRATION.md section 4): brightness (mean grey value), blur (variance of
the Laplacian) and minimum size. Choose the thresholds on VAL images only so that at most 2% of val images fail;
save them in pipeline/quality.yaml with the val numbers. Never use test for this.

Add pipeline/cli.py: python -m pipeline.cli <image> prints the JSON result. Add tests in pipeline/tests/ with 3 val
images (one no_defect). Run it on 20 val images and print the timings (median and max). Commit with
"v2 pipeline: analyze_image".
```

---

## Prompt 6 — pipeline evaluation (once, Release 2)

```
Read AGENTS.md first and follow it exactly. The pipeline and the models are final.

Add --release 2 to scripts/final_eval.py and run it once on the test split:
1. Run pipeline.analyze_image on every test image.
2. Match detected boxes to human boxes of the same image with IoU >= 0.5 (greedy, highest confidence first).
   Report: detection precision/recall at the pipeline threshold (per class), and severity macro_f1, recall_high and
   confusion matrix on the matched boxes (the severity of a matched detected box vs the human severity).
3. Image level (what the citizen sees): for each test image compare the pipeline's report-level result (severity =
   high if any issue is high, else low_medium; or no issue) with the human labels: a 3x3 table
   (no issue / low_medium / high) and its accuracy and macro_f1.
4. How many test images were marked not_assessable by the quality check (should be close to 0).
5. Median and 95th-percentile analysis time per image.
Save everything in reports/final/pipeline/ with a summary next to the Release 1 results. Commit with
"v2 Release 2 pipeline evaluation".
```

---

## Prompt 7 — analysis worker (connects the models to Firebase)

```
Read AGENTS.md, service/AGENTS.md and docs/APP_INTEGRATION.md first and follow them exactly.

Write service/worker.py (Python, firebase-admin):
1. Credentials only from GOOGLE_APPLICATION_CREDENTIALS. Add service/*.json and .env to .gitignore before anything else.
2. Load the models once with pipeline.analyze.load_models().
3. Listen (on_snapshot) to Firestore collection "reports" where analysisStatus == "processing". For each report:
   claim it in a transaction (processing -> analyzing, analysisClaimedAt), download imageUrl into memory (no file
   kept), run pipeline.analyze_image, and write exactly the fields in docs/APP_INTEGRATION.md section 2
   (analysisStatus, issues, issueClass, severity, analysisReason, modelVersion, analyzedAt, updatedAt).
   On error retry 3 times, then write analysisStatus "failed" with an Arabic analysisReason.
   On start, also pick up reports stuck in "analyzing" for more than 10 minutes.
4. Arabic analysisReason texts:
   not_assessable: "تعذّر تقييم الصورة. الرجاء التقاط صورة أوضح تُظهر الرصيف كاملًا."
   no_issue_detected: "لم يتم العثور على ضرر في الرصيف أو منحدره في هذه الصورة."
   failed: "حدث خطأ أثناء تحليل الصورة. سيُعاد التحليل لاحقًا."
5. A tiny HTTP server in the same process with GET /health -> {"status": "ok", "modelVersion": ...} for uptime
   monitoring.
6. service/Dockerfile and service/README.md: how to run locally (python -m service.worker) and how to deploy on
   Cloud Run with min-instances 1 (or another always-on host), with the key passed as a secret, never baked in.
7. service/tests/: unit tests with a fake Firestore (no network) for: claim, completed, no_issue_detected,
   not_assessable, failed after 3 errors, and "never writes reportStatus".
Run the tests. Then, only with my confirmation, run it against ONE test report created with a test account and
show me the document before and after. Commit with "v2 service: analysis worker".
```

---

## Prompt 8 — app integration (run this in the app repository `hcchly/Khota`)

```
Read the project's code style and existing structure first (lib/services/report_service.dart,
lib/widgets/report_status_widgets.dart, lib/screens/reports/). Do not change login, registration, the
report submission flow or the iOS project files (ios/, Info.plist, swiftpm). Never commit API keys or tokens.

The analysis worker (repo Khota-ML, docs/APP_INTEGRATION.md) now writes these fields to reports/{id}:
analysisStatus ("processing" | "analyzing" | "completed" | "no_issue_detected" | "not_assessable" | "failed"),
issues (list of {issueClass, severity, detectionConfidence, pHigh, box{x,y,w,h}}), issueClass, severity,
analysisReason (Arabic), modelVersion, analyzedAt.

1. Add an AnalysisResult model that parses these fields safely (missing or null fields = still processing).
2. In the citizen's report list and report details, listen to the document (snapshots) and show:
   - processing / analyzing: "جارٍ تحليل الصورة" with a progress indicator,
   - completed: issue type and severity in Arabic
     (curb_ramp_issue = "مشكلة في منحدر الرصيف", surface_damage = "تلف في سطح الرصيف",
      low_medium = "منخفضة إلى متوسطة", high = "عالية"), the number of issues, and the photo with the boxes drawn
     (box is normalised x_center, y_center, width, height),
   - no_issue_detected / not_assessable / failed: analysisReason, and for not_assessable a button to submit a new
     photo.
   Severity must not rely on colour alone (text + icon), and all new widgets need Arabic Semantics labels and must
   work at 200% text size (Accessibility NFR).
3. firestore.rules: keep citizens unable to write any analysis field (create rule unchanged; no client update).
   Add rules tests if the project has them.
4. Widget tests with fake data for each status (follow test/helpers/report_fakes.dart).
5. Run flutter analyze and flutter test. Show me screenshots of each state (golden or emulator), then commit on a
   new branch "feature/analysis-results" and open a PR. Do not merge.
```

---

## Prompt 9 — end-to-end system test and timing

```
Read AGENTS.md and docs/APP_INTEGRATION.md first.

Write scripts/e2e_test.py that uses a TEST Firebase account (credentials from the environment, never committed):
1. Upload 20 val images (not test) as reports exactly like the app does (Storage upload + reports document with
   analysisStatus "processing"), tagged with description "E2E TEST".
2. Wait for the worker and record for each report: time from createdAt to analyzedAt, the status, and whether the
   result equals pipeline.analyze_image run locally on the same image.
3. Print: success rate, median and 95th-percentile analysis time, and any mismatch. Save reports/e2e/e2e_results.csv
   and a short summary for the report.
4. Delete the 20 test reports and their images at the end (ask me first).
Also check GET /health of the deployed worker every minute for 30 minutes and report the uptime (Availability NFR).
Commit the script and the summary (not the images) with "v2 end-to-end test".
```
