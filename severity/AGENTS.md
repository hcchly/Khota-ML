# severity/ — Haifa's folder (version 2)

Follow the root AGENTS.md first. Extra rules for this folder:

- Only the severity model lives here: `low_medium` / `high` for one cropped box.
- Edit only files inside `severity/`. Ask before changing `khota_ml/`, `scripts/`, `data/`, `pipeline/` or `service/`.
- Crops come only from `khota_ml.crop.crop_box`. Classes come only from `khota_ml/config.py`.
- Use `train` to train and `val` to compare and to choose the `high` threshold. Never read `test`.
- Every run appends one row to `experiments/experiments.csv` with `owner = haifa`, `model = severity`.
- `train_v1.py` is the version 1 script (3 classes). Reuse its structure, but v2 code goes in `train.py`.
- The selected model is written in `severity/SELECTED_MODEL.md`, and it must expose
  `load_severity_model(path)` and `predict_severity(model, crops) -> list of (label, p_high)` in
  `severity/predict.py` for the pipeline.
