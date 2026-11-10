# Selected severity model — version 2 (binary)

| | |
|---|---|
| run_id | `sev2_C_resnet18_aug_sampler` |
| checkpoint | `experiments/sev2_C_resnet18_aug_sampler/best_model.pt` (committed with `git add -f`) |
| model | ResNet-18 ImageNet-pretrained, 2-class head (`low_medium`, `high`) |
| input | `khota_ml.crop.crop_box(expand=0.20)` on the full-resolution photo → 224×224 bilinear → ImageNet normalisation |
| training | train-only augmentation (hflip, rotation ±10°, brightness/contrast ±20%), weighted random sampler (no class weights), AdamW lr 1e-4, batch 32, best epoch 4 of 9 (early stopping on val macro_f1, patience 5), seed 42 |
| threshold for `high` | **0.50** (stored in the checkpoint). The val scan 0.10–0.90 put 0.50 and 0.55 equal best; ties go to the value closest to 0.5. |
| use | `severity/predict.py`: `load_severity_model(path)`, `predict_severity(model, crops) -> [(label, p_high)]` |

## Val results (seed 42, threshold 0.50; test not used)

| metric | model | majority baseline |
|---|---|---|
| **macro_f1** | **0.812** | 0.465 |
| recall_high | 0.571 (12 / 21) | 0.000 |
| precision_high | 0.800 (12 / 15) | — |
| F1 low_medium / high | 0.958 / 0.667 | 0.930 / 0.000 |
| accuracy | 0.925 | 0.869 |
| macro_f1 curb_ramp_issue boxes (21, 3 high) | 0.846 | — |
| macro_f1 surface_damage boxes (139, 18 high) | 0.801 | — |

Confusion matrix (rows true, columns predicted; low_medium, high): `[[136, 3], [9, 12]]`.
The tuned threshold equals 0.5, so the "tuned" results are the same as above.

## Stability (same settings, seeds 42 / 43 / 44)

| | mean ± std | per seed |
|---|---|---|
| val macro_f1 | **0.808 ± 0.015** | 0.812 / 0.791 / 0.821 |
| val recall_high | 0.587 ± 0.027 | 0.571 / 0.571 / 0.619 |

Seeds 43 and 44 are in `experiments/sev2_C_resnet18_aug_sampler_seed43/` and `_seed44/`. Training ran on Apple MPS,
which is not bit-for-bit deterministic, so a re-run can differ slightly.
