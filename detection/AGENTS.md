# detection/ — Hala's folder (version 2)

Follow the root AGENTS.md first. Extra rules for this folder:

- Only the YOLOv8 detection model lives here: boxes with type `curb_ramp_issue` or `surface_damage`.
- Edit only files inside `detection/`. Ask before changing `khota_ml/`, `scripts/`, `data/`, `pipeline/` or `service/`.
- YOLO labels come only from `scripts/make_yolo_labels.py`. Class order comes only from `khota_ml/config.py`.
- Use `train` to train and `val` to compare and to choose the confidence threshold. Never read `test`.
- Every run appends one row to `experiments/experiments.csv` with `owner = hala`, `model = detection`.
- The selected model is written in `detection/SELECTED_MODEL.md` (run_id, weights path or Drive link,
  confidence threshold, val metrics). The pipeline loads it with `ultralytics.YOLO(path)`.
