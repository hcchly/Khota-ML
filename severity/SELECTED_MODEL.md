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

## Improvement round 2 (10 Nov 2026) — no measurable improvement; C stays selected

Each run changed one thing, 3 seeds each (42/43/44), val only, threshold 0.5. Replacement rule: mean val macro_f1
≥ best + 0.02, or within ±0.01 with a higher mean recall_high. The jittered-val column is a fixed jittered copy of the
val boxes (seed 2026, centre ±10%, w/h ×0.85–1.15), **report only**, never used for selection. Box jitter in training
(run E) is the approved exception in AGENTS.md section 8.

| run | change | macro_f1 | recall_high | precision_high | accuracy | mF1 curb | mF1 surface | jittered val macro_f1 | rule decision |
|---|---|---|---|---|---|---|---|---|---|
| majority baseline | — | 0.465 | 0.000 | — | 0.869 | — | — | — | |
| **C (selected)** | ResNet-18, aug, sampler | **0.808 ± 0.015** | 0.587 ± 0.027 | 0.757 ± 0.048 | 0.921 | 0.846 | 0.797 | 0.784 ± 0.063 | start |
| E | C + box jitter (train) | 0.807 ± 0.024 | 0.603 ± 0.055 | 0.735 ± 0.067 | 0.919 | 0.826 | 0.800 | 0.770 ± 0.038 | replaces C (Δ −0.001, +1 high box over 3 seeds) |
| F | E at 320×320 | 0.791 ± 0.021 | 0.587 ± 0.073 | 0.699 ± 0.096 | 0.910 | 0.771 | 0.796 | 0.770 ± 0.029 | no |
| G | E with ResNet-50 | 0.789 ± 0.008 | 0.667 ± 0.126 | 0.638 ± 0.129 | 0.900 | 0.758 | 0.793 | 0.722 ± 0.023 | no |
| H | E with ConvNeXt-Tiny | 0.808 ± 0.041 | 0.587 ± 0.073 | 0.761 ± 0.106 | 0.921 | 0.878 | 0.793 | 0.770 ± 0.027 | no |
| I | mean p_high of E's 3 seeds | 0.801 | 0.571 | 0.750 | 0.919 | 0.846 | 0.788 | 0.801 | no |

Threshold scan (0.10–0.90, step 0.05) on the round's rule-best run E (seed 42): best at **0.50** (macro_f1 0.812,
recall_high 0.571), the same as C.

**Decision (owner, 10 Nov 2026): keep C.** E won only on the recall tie-break by one `high` box across three seeds
(1 / 63), its mean macro_f1 is 0.001 lower, and its jittered-val score is lower than C's (0.770 vs 0.784), so box
jitter showed no robustness gain. All differences are within the seed spread (std 0.015–0.041). The selected
checkpoint, threshold (0.50) and val metrics above are unchanged.

Round 2 checkpoints are not committed (outputs, metrics, histories, confusion matrices and p_high arrays are, in
`experiments/sev2_{E,F,G,H,I}_*`; summary in `experiments/severity_v2_round2.json`). `sev2_I_*/ensemble.json`
points at E checkpoints that exist only locally.

## Round 3 (10 Nov 2026) — no improvement; C stays selected

Decisions before the round (owner): M (3-level auxiliary head) dropped — v2 has no 3-level labels and v1 is not
mixed in. L (crop expand 0.50, conflicts with AGENTS.md rule 5) and N (cross-validation) are report-only. K may
replace C only by the round rule. Threshold 0.5 throughout; test split never read.

| run | what | macro_f1 | recall_high | precision_high | accuracy | mF1 curb | mF1 surface | status |
|---|---|---|---|---|---|---|---|---|
| majority baseline | val | 0.465 | 0.000 | — | 0.869 | — | — | |
| **C** | val, 3 seeds | **0.808 ± 0.015** | **0.587 ± 0.027** | 0.757 | 0.921 | 0.846 | 0.797 | selected |
| K-DINOv2 ViT-S/14 + LR | val, single (frozen, deterministic) | 0.626 | 0.524 | 0.306 | 0.781 | 0.606 | 0.629 | does not replace C |
| K-CLIP ViT-B/32 + LR | val, single (frozen, deterministic) | 0.686 | 0.429 | 0.474 | 0.863 | 0.738 | 0.664 | does not replace C |
| L: C with expand 0.50 | val, 3 seeds | 0.740 ± 0.028 | 0.460 ± 0.027 | 0.663 | 0.896 | 0.732 | 0.741 | report only, not flagged |
| N: C | 5-fold CV on train+val | 0.723 ± 0.075 | 0.416 ± 0.151 | 0.693 | 0.895 | 0.561 | 0.758 | report only |
| N: L (best of K/L) | 5-fold CV on train+val | 0.677 ± 0.091 | 0.313 ± 0.146 | 0.671 | 0.888 | 0.595 | 0.699 | report only |

K: features of the same crops as C (expand 0.20 → 224×224), StandardScaler + LogisticRegression(class_weight=
"balanced", C=1.0). DINOv2 weights are the official ViT-S/14 loaded through `timm`
(`vit_small_patch14_dinov2.lvd142m`) because the `torch.hub` code needs Python ≥ 3.10; CLIP is OpenAI ViT-B/32 via
`open_clip`. N: StratifiedGroupKFold(5, seed 42) on `location_id` over the 855 train+val boxes (no location in two
folds, no test image; folds in `experiments/sev2_N_cv5_trainval/folds.csv`); each fold trains a fixed 8 epochs
(median best epoch of the val runs), no early stopping.

**Caveat for the report:** C's cross-validated macro_f1 (0.723 ± 0.075, recall_high 0.416, per fold 0.64–0.83) is
well below its val score (0.808). The val split appears to be one of the easier partitions, so the Release 1 test
result may be closer to the CV range. CV is report-only and does not change the selection or the threshold.
