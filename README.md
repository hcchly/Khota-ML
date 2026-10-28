# Khota ML

Sidewalk defect detection (YOLOv8) and severity rating (Low / Medium / High) for the Khota graduation project.

- Shared rules for people and Codex: [AGENTS.md](AGENTS.md)
- Prompts to give Codex, in order: [docs/PROMPTS.md](docs/PROMPTS.md)
- Labels: `data/source/Khota_Model_Final.xlsx` → `data/annotations.csv` (`scripts/convert_annotations.py`)
- Locations for the grouped split: `data/locations.csv` (`scripts/make_locations.py`)
- Images: the Roboflow YOLOv8 exports in the shared Google Drive folder `all_batches_boxes`.
  Download them into `raw/` (not committed). Model weights are not committed either.
