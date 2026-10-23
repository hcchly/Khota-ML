# detection/ — detection model folder

Follow the root AGENTS.md first. Extra rules for this folder:

- Only the YOLOv8 detection model lives here: boxes with type `curb_ramp_issue` or `surface_damage`.
- Edit only files inside `detection/`. Ask before changing `khota_ml/`, `scripts/` or `data/`.
- YOLO labels come only from `scripts/make_yolo_labels.py`. Class order comes only from `khota_ml/config.py`.
- Use `train` to train and `val` to compare. Never read `test`.
- Every run appends one row to `experiments/experiments.csv` with `model = detection`.

This folder is empty on purpose. Code is added with prompt 3 in `docs/PROMPTS.md`.
