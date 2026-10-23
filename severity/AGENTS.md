# severity/ — Haifa's folder

Follow the root AGENTS.md first. Extra rules for this folder:

- Only the severity model lives here: Low / Medium / High for one cropped box.
- Edit only files inside `severity/`. Ask before changing `khota_ml/`, `scripts/` or `data/`.
- Crops come only from `khota_ml.crop.crop_box`. Classes come only from `khota_ml/config.py`.
- Use `train` to train and `val` to compare. Never read `test`.
- Every run appends one row to `experiments/experiments.csv` with `owner = haifa`, `model = severity`.

This folder is empty on purpose. Code is added with prompt 2 in `docs/PROMPTS.md`.
