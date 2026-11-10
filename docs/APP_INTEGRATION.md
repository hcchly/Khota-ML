# App integration contract (version 2)

How the models reach the Khota app. The app repo is `hcchly/Khota` (Flutter, Firebase project `khota-988e6`).

## 1. Flow

1. The citizen submits a report. The app uploads the photo to Cloud Storage (`uploads/{uid}/...`) and creates
   `reports/{reportId}` with `analysisStatus = "processing"`, `issueClass = null`, `severity = null`, `imageUrl`
   (a download URL) — this is already implemented (`lib/services/report_service.dart`).
2. The **analysis worker** (`service/worker.py`, this repo) listens to Firestore for reports with
   `analysisStatus == "processing"`. For each one it downloads the image from `imageUrl`, runs
   `pipeline.analyze_image(...)` and writes the result back with the Firebase Admin SDK (which is not limited by
   the security rules, so the citizen still cannot write these fields).
3. The app listens to the report document and shows the result as soon as it changes.

No Cloud Functions are needed (they require the paid Blaze plan). The worker is a normal Python process: on a
laptop for development and the demo, and on a small always-on host (e.g. Cloud Run with min-instances 1, Render or
a VM) for the evaluation period. It also serves `GET /health` so availability can be monitored (Availability NFR).

## 2. Fields the worker writes (and nothing else)

| field | type | value |
|---|---|---|
| `analysisStatus` | string | `completed` (≥ 1 issue found), `no_issue_detected` (nothing above the threshold), `not_assessable` (image-quality check failed, see §4), `failed` (error; the worker retries 3 times first) |
| `issues` | array of maps | one map per detected issue, highest severity first: `{issueClass, severity, detectionConfidence, pHigh, box: {x, y, w, h}}` (`box` YOLO-normalised) |
| `issueClass` | string or null | `issueClass` of the first (most severe, then most confident) issue |
| `severity` | string or null | `high` if any issue is `high`, else `low_medium`; null when no issue |
| `analysisReason` | string or null | Arabic text for `not_assessable` / `no_issue_detected` / `failed`, shown to the citizen |
| `modelVersion` | string | e.g. `v2-det:<run_id>-sev:<run_id>` |
| `analyzedAt` | timestamp | server time |
| `updatedAt` | timestamp | server time |

Values use the names in `khota_ml/config.py` (`curb_ramp_issue`, `surface_damage`, `low_medium`, `high`). The app
translates them with the Arabic labels in `config.APP_LABELS_AR`.

Never written by the worker: `reportStatus` (the administrator's job), `userId`, `imageUrl`, `description`,
`location*`, `createdAt`.

## 3. Rules

- Idempotent: the worker claims a report with a transaction (`analysisStatus: processing → analyzing`, plus
  `analysisClaimedAt`) so two workers never analyse the same report; a claim older than 10 minutes may be retaken.
- One image at a time is fine (a few reports per minute). Log report id, timings and status only — never the image.
- `firestore.rules` in the app must still forbid citizens from writing the analysis fields
  (`allow update: if false` for citizens today); the admin SDK bypasses the rules.

## 4. Not Assessable (rule, not a model)

The models have no Not Assessable class. The worker marks a photo `not_assessable` only when a simple quality check
fails: too dark, too blurred (variance of the Laplacian) or too small. The thresholds are chosen on **val**
images (they must not reject more than 2% of val images) and stored in `service/config.yaml`. Then the app asks the
citizen for a clearer photo (backlog item "informed when my image cannot be assessed").

## 5. Timing (Performance NFR)

The Performance NFR (≤ 10 s until the report shows *Processing*) is the app's job and does not wait for the
model. The analysis time (from `createdAt` to `analyzedAt`) is measured separately by prompt 8 and reported as
median and 95th percentile.
