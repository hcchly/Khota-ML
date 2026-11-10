# Label changes after the consistency review

| date | image_id | box_id | old | new | reason |
| --- | --- | --- | --- | --- | --- |

## Version 2 — 10 Nov 2026

- Severity changed from three levels (low / medium / high) to two: `low_medium` (old low + medium) and `high`
  (unchanged), following the binary rubric. Reason codes changed to SD-LM / SD-H / CR-LM / CR-H.
- New label table `Khota_Model_Final_Binary.xlsx` (replaces `Khota_Model_Final.xlsx`): the team re-annotated and
  reviewed all boxes against the new Roboflow exports (batch 1, batch 2 v5, batch 3 v4): 1,010 boxes
  (v1: 581), 689 images, 42 excluded.
- Box_Index is now `B01`, `B02`, … (1-based line number); `annotations.csv` keeps a 0-based `box_id` and adds
  `box_label`.
- GPS for batch 3 added (`data/source/gps.csv`), so batch 3 is no longer one no-GPS group. New `locations.csv`
  (272 locations) and new `split.csv` (seed 42). The v1 split is kept as `data/v1_split.csv` for the record.
- RYD-0591 B01 is `curb_ramp_issue` (changed in Roboflow after the 9 Nov export). Prompt 1 re-checks this.
