# Khota ML — version 2

Sidewalk defect detection (YOLOv8) and **binary** severity rating (`low_medium` / `high`) for the Khota graduation
project, and the worker that connects them to the Khota app.

- Rules for people and AI agents: [AGENTS.md](AGENTS.md) (Claude Code reads it through [CLAUDE.md](CLAUDE.md))
- Prompts, in order: [docs/PROMPTS.md](docs/PROMPTS.md)
- Firebase contract with the app: [docs/APP_INTEGRATION.md](docs/APP_INTEGRATION.md)
- Labels: `data/source/Khota_Model_Final_Binary.xlsx` → `data/annotations.csv` (`scripts/convert_annotations.py`)
- Split (seed 42, by location): `data/split.csv` — never regenerate
- Images: the latest Roboflow YOLOv8 exports, downloaded into `raw/` (not committed)

| | low_medium | high | total |
|---|---|---|---|
| curb_ramp_issue | 134 | 26 | 160 |
| surface_damage | 738 | 112 | 850 |
| total | 872 | 138 | 1,010 boxes |

689 images (495 with boxes, 194 no_defect), 42 excluded.

**Version 1** (low / medium / high, 581 boxes) is at git tag `v1`; its experiments are in `experiments/v1/`.
