# Dataset summary (from Khota_Model_Final.xlsx, 2026-10-28)

| | low | medium | high | total |
|---|---|---|---|---|
| curb_ramp_issue | 85 | 45 | 28 | 158 |
| surface_damage | 143 | 192 | 88 | 423 |
| total | 228 (39%) | 237 (41%) | 116 (20%) | 581 boxes |

| batch | images | with boxes | no_defect | boxes | curb | surface |
|---|---|---|---|---|---|---|
| 1 | 389 | 190 | 199 | 291 | 18 | 273 |
| 2 | 255 | 187 | 68 | 232 | 114 | 118 |
| 3 | 50 | 50 | 0 | 58 | 26 | 32 |
| total | 694 | 427 | 267 | 581 | 158 | 423 |

- Excluded: 37 images (23 Not Assessable, 14 Duplicate).
- Boxes per image: 1 box 296 images, 2 boxes 111, 3 boxes 17, 4 boxes 3.
- Locations: 237 location groups (50 m radius). No GPS: RYD-0610…0621 (batch 2) and all of batch 3.
- Severity majority baseline (whole dataset, always `medium`): accuracy 0.408, macro_f1 0.193.
- Main metrics: detection mAP50, severity macro_f1 (val for comparison, test once at the end).
