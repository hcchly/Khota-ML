# Dataset summary — version 2 (from Khota_Model_Final_Binary.xlsx, 10 Nov 2026)

| | low_medium | high | total |
|---|---|---|---|
| curb_ramp_issue | 134 | 26 | 160 |
| surface_damage | 738 | 112 | 850 |
| total | 872 (86%) | 138 (14%) | 1,010 boxes |

| batch | images | with boxes | no_defect | boxes | curb | surface |
|---|---|---|---|---|---|---|
| 1 | 388 | 231 | 157 | 566 | 19 | 547 |
| 2 | 251 | 214 | 37 | 379 | 121 | 258 |
| 3 | 50 | 50 | 0 | 65 | 20 | 45 |
| total | 689 | 495 | 194 | 1,010 | 160 | 850 |

- Excluded: 42 images (24 Not Assessable, 18 duplicate or near-duplicate).
- Boxes per image: 1 box 231 images, 2 → 140, 3 → 62, 4 → 36, 5 → 10, 6 → 8, 8 → 4, 9 → 2, 10 → 1, 11 → 1.
- Locations: 272 groups (50 m). Only RYD-0610…0621 have no GPS (one group).
- Split (seed 42): train 491 images / 695 boxes, val 99 / 160 (21 high, 21 curb), test 99 / 155 (22 high, 21 curb).
- Severity majority baseline on val (always `low_medium`): accuracy 0.869, macro_f1 0.465, recall_high 0.
