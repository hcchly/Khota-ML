"""Severity v2, round 3 (reference: sev2_C_resnet18_aug_sampler). Val only; the test split is never loaded.

    python severity/round3.py k    # K: frozen DINOv2 ViT-S/14 and CLIP ViT-B/32 features + logistic regression
    python severity/round3.py l    # L: C with crop expand 0.50 (report only, cannot be selected)
    python severity/round3.py n    # N: 5-fold location-grouped CV on train+val for C and the best of K/L (report only)

Decisions for this round (owner, 10 Nov 2026): M dropped (no 3-level labels in v2; v1 not used). L and N are
report-only. K may replace C only if its val macro_f1 >= 0.808 + 0.02, or within ±0.01 with higher recall_high.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from khota_ml.config import CROP_EXPAND, DETECTION_CLASSES, SEED  # noqa: E402
from severity.predict import IMAGENET_MEAN, IMAGENET_STD, build_model  # noqa: E402
from severity.train import (  # noqa: E402
    BASELINE, BATCH_SIZE, LEARNING_RATE, SELECTION, SEEDS, CropDataset, DataStore, full_metrics, load_train_val,
    log_run, predict_p_high, public, run_id_for, save_outputs, seed_everything, train_run, write_log_row,
)

EXP = ROOT / "experiments"
ROUND3 = EXP / "severity_v2_round3.json"
C_REF = {"run_id": "sev2_C_resnet18_aug_sampler", "macro_f1": 0.808, "recall_high": 0.587}  # mean over 3 seeds
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
SIZE = (224, 224)
N_FOLDS = 5


def device():
    return torch.device(os.environ.get("KHOTA_SEVERITY_DEVICE") or ("mps" if torch.backends.mps.is_available() else "cpu"))


def box_meta(frame: pd.DataFrame) -> pd.DataFrame:
    """Box rows in the exact order prepare_crops produces them."""
    return pd.concat([rows for _, rows in frame.groupby("image_file", sort=False)]).reset_index(drop=True)


def summary() -> dict:
    return json.loads(ROUND3.read_text()) if ROUND3.exists() else {"reference": C_REF, "baseline": BASELINE,
                                                                     "test_used": False}


def save_summary(s: dict) -> None:
    ROUND3.write_text(json.dumps(s, indent=2))


# ------------------------------------------------------------------------------------------------- K

def load_backbone(name: str):
    if name == "dinov2_vits14":
        import timm  # official DINOv2 ViT-S/14 weights (timm/vit_small_patch14_dinov2.lvd142m); CLS token output
        net = timm.create_model("vit_small_patch14_dinov2.lvd142m", pretrained=True, num_classes=0, img_size=224)
        return net.eval(), IMAGENET_MEAN, IMAGENET_STD, net
    if name == "clip_vitb32":
        import open_clip
        model, _, _ = open_clip.create_model_and_transforms("ViT-B-32-quickgelu", pretrained="openai")
        return model.eval(), CLIP_MEAN, CLIP_STD, model.encode_image
    raise ValueError(name)


def extract(name: str, images, dev) -> np.ndarray:
    model, mean, std, fn = load_backbone(name)
    model.to(dev)
    m = torch.tensor(mean).view(1, 3, 1, 1).to(dev)
    s = torch.tensor(std).view(1, 3, 1, 1).to(dev)
    out = []
    with torch.inference_mode():
        for i in range(0, len(images), 64):
            x = torch.from_numpy(np.stack(images[i:i + 64])).permute(0, 3, 1, 2).float().div(255).to(dev)
            out.append(fn((x - m) / s).float().cpu().numpy())
    return np.concatenate(out)


def logreg():
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    return make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=5000,
                                                              random_state=SEED))


def features(name: str, store: DataStore, dev) -> dict:
    """Frozen features of the same crops as C (crop_box expand 0.20 -> 224x224), cached in the K run folder."""
    run_dir = EXP / f"sev2_K_{name}_logreg"
    cache = run_dir / "features.npz"
    if cache.exists():
        return dict(np.load(cache))
    run_dir.mkdir(parents=True, exist_ok=True)
    tr_x, tr_y, tr_d = store.plain("train", SIZE)
    va_x, va_y, va_d = store.plain("val", SIZE)
    print(f"  extracting {name} features…")
    f = {"train": extract(name, tr_x, dev), "val": extract(name, va_x, dev), "train_y": tr_y, "val_y": va_y,
         "train_d": tr_d, "val_d": va_d}
    np.savez(cache, **f)
    return f


def run_k(store: DataStore) -> None:
    s = summary()
    dev = device()
    for name, text in (("dinov2_vits14", "frozen DINOv2 ViT-S/14 (CLS token, 384-d)"),
                       ("clip_vitb32", "frozen CLIP ViT-B/32 OpenAI (image embedding, 512-d)")):
        run_id = f"sev2_K_{name}_logreg"
        run_dir = EXP / run_id
        f = features(name, store, dev)
        clf = logreg().fit(f["train"], f["train_y"])
        p = clf.predict_proba(f["val"])[:, list(clf.classes_).index(1)]
        np.save(run_dir / "val_p_high.npy", p)
        metrics = full_metrics(f["val_y"], p, f["val_d"], 0.5)
        save_outputs(run_dir, metrics)
        v = public(metrics)
        d_f1, d_rec = v["macro_f1"] - C_REF["macro_f1"], v["recall_high"] - C_REF["recall_high"]
        replaces = d_f1 >= 0.02 - 1e-9 or (abs(d_f1) <= 0.01 + 1e-9 and v["recall_high"] > C_REF["recall_high"])
        result = {"run_id": run_id, "backbone": text, "classifier": "StandardScaler + LogisticRegression("
                  "class_weight='balanced', C=1.0, max_iter=5000)", "val": v, "baseline": BASELINE,
                  "replaces_C": replaces, "d_macro_f1_vs_C_mean": d_f1, "d_recall_high_vs_C_mean": d_rec,
                  "test_used": False}
        (run_dir / "metrics.json").write_text(json.dumps(result, indent=2))
        other = {k: v[k] for k in ("recall_high", "precision_high", "f1_per_class", "accuracy",
                                   "macro_f1_by_defect", "confusion_matrix")}
        other.update({"baseline_macro_f1": BASELINE["macro_f1"], "baseline_accuracy": BASELINE["accuracy"]})
        write_log_row({
            "run_id": run_id, "date": date.today().isoformat(), "owner": "haifa", "model": "severity",
            "change_from_previous": f"Run K: C's crops, but {text} features + logistic regression instead of a "
                                    f"fine-tuned ResNet-18",
            "key_settings": f"{text}; no fine-tuning; crop_box expand={CROP_EXPAND} -> 224x224, no augmentation; "
                            f"StandardScaler + LogisticRegression(class_weight=balanced, C=1.0); threshold=0.5; "
                            f"deterministic (single value, no seed spread)",
            "val_main_metric": f"{v['macro_f1']:.6f}", "val_other_metrics": json.dumps(other, sort_keys=True),
            "notes": f"v2 round 3; can replace C only by the round rule; replaces C: {replaces}; test split not used."})
        s.setdefault("K", {})[name] = {"run_id": run_id, "val": v, "replaces_C": replaces}
        print(f"{run_id}: macro_f1 {v['macro_f1']:.3f} recall_high {v['recall_high']:.3f} "
              f"precision_high {v['precision_high']:.3f} -> replaces C: {replaces}")
    save_summary(s)


# ------------------------------------------------------------------------------------------------- L

def run_l(store: DataStore) -> None:
    s = summary()
    spec = json.loads(SELECTION.read_text())["spec"] | {
        "name": "L", "crop_expand": 0.50,
        "change": "Run L (report only): C with crop_box expand 0.50 instead of 0.20"}
    results = []
    for seed in SEEDS:
        change = spec["change"] if seed == SEED else f"Repeat of {run_id_for(spec, SEED)} with seed {seed}"
        r = train_run(spec, seed, store, device(), force=False, change=change)
        log_run(r, change, notes="v2 round 3; REPORT ONLY (crop expand differs from AGENTS.md rule 5), "
                                 "cannot be selected; test split not used.")
        results.append(r)
    f1 = np.array([r["val"]["macro_f1"] for r in results])
    rec = np.array([r["val"]["recall_high"] for r in results])
    stats = {"macro_f1": [float(f1.mean()), float(f1.std(ddof=1))], "recall_high": [float(rec.mean()), float(rec.std(ddof=1))],
             "precision_high": float(np.mean([r["val"]["precision_high"] for r in results])),
             "accuracy": float(np.mean([r["val"]["accuracy"] for r in results])),
             "macro_f1_curb": float(np.mean([r["val"]["macro_f1_by_defect"]["curb_ramp_issue"] for r in results])),
             "macro_f1_surface": float(np.mean([r["val"]["macro_f1_by_defect"]["surface_damage"] for r in results])),
             "per_seed": f1.round(4).tolist(), "best_epochs": [r["best_epoch"] for r in results]}
    flag = stats["macro_f1"][0] - C_REF["macro_f1"] >= 0.02
    s["L"] = {"run_id": run_id_for(spec, SEED), "stats": stats, "report_only": True, "beats_C_by_0.02": flag}
    save_summary(s)
    print(f"L: macro_f1 {f1.mean():.3f} ± {f1.std(ddof=1):.3f}, recall_high {rec.mean():.3f} ± {rec.std(ddof=1):.3f}"
          f"{'  ** FLAG: beats C by >= 0.02 **' if flag else ''}")


# ------------------------------------------------------------------------------------------------- N

def make_folds(meta: pd.DataFrame) -> np.ndarray:
    from sklearn.model_selection import StratifiedGroupKFold
    folds = np.full(len(meta), -1)
    sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    for k, (_, test_idx) in enumerate(sgkf.split(meta, (meta.severity == "high").astype(int), meta.location_id)):
        folds[test_idx] = k
    return folds


def cnn_fold(spec: dict, epochs: int, tr_x, tr_y, te_x, dev) -> np.ndarray:
    """C's recipe on one fold for a fixed number of epochs (no early stopping: the held-out fold is not peeked)."""
    seed_everything(SEED)
    counts = np.bincount(tr_y, minlength=2).astype(np.float64)
    gen = torch.Generator().manual_seed(SEED)
    sampler = WeightedRandomSampler(torch.as_tensor((1.0 / counts)[tr_y], dtype=torch.double), len(tr_y),
                                    replacement=True, generator=gen)
    loader = DataLoader(CropDataset(tr_x, tr_y, augment=spec["augment"]), batch_size=BATCH_SIZE, sampler=sampler)
    net = build_model(spec["architecture"], pretrained=True).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    crit = nn.CrossEntropyLoss()
    for _ in range(epochs):
        net.train()
        for x, y in loader:
            opt.zero_grad(set_to_none=True)
            crit(net(x.to(dev)), y.to(dev)).backward()
            opt.step()
    return predict_p_high(net, te_x, dev)


def run_n(store: DataStore) -> None:
    s = summary()
    dev = device()
    frame = pd.concat([store.train, store.val])
    meta = pd.concat([box_meta(store.train), box_meta(store.val)]).reset_index(drop=True)
    run_dir = EXP / "sev2_N_cv5_trainval"
    run_dir.mkdir(parents=True, exist_ok=True)
    folds = make_folds(meta)
    meta = meta.assign(fold=folds)
    meta[["image_id", "box_id", "location_id", "severity", "defect_class", "fold"]].to_csv(
        run_dir / "folds.csv", index=False)
    assert meta.groupby("location_id").fold.nunique().max() == 1, "a location spans two folds"
    y = (meta.severity == "high").astype(np.int64).to_numpy()
    d = meta.defect_class.map(DETECTION_CLASSES.index).to_numpy()

    # second model: best of K (single value) and L (seed mean) by val macro_f1
    cands = {r["run_id"]: r["val"]["macro_f1"] for r in s.get("K", {}).values()}
    if "L" in s:
        cands[s["L"]["run_id"]] = s["L"]["stats"]["macro_f1"][0]
    second = max(cands, key=cands.get)
    c_spec = json.loads(SELECTION.read_text())["spec"]
    c_epochs = int(np.median([json.loads((EXP / run_id_for(c_spec, sd) / "metrics.json").read_text())["best_epoch"]
                              for sd in SEEDS]))
    models = {"C": ("cnn", c_spec, c_epochs)}
    if second.startswith("sev2_K_"):
        models["second"] = ("logreg", second[len("sev2_K_"):-len("_logreg")], None)
    else:
        l_spec = c_spec | {"name": "L", "crop_expand": 0.50}
        models["second"] = ("cnn", l_spec, int(np.median(s["L"]["stats"]["best_epochs"])))

    out = {"folds": N_FOLDS, "fold_sizes": np.bincount(folds).tolist(),
           "fold_high": [int(y[folds == k].sum()) for k in range(N_FOLDS)], "second_model": second, "models": {}}
    for key, (kind, spec, epochs) in models.items():
        name = "sev2_C_resnet18_aug_sampler" if key == "C" else second
        if kind == "cnn":
            expand = float(spec.get("crop_expand", CROP_EXPAND))
            x = store.plain("train", SIZE, expand)[0] + store.plain("val", SIZE, expand)[0]
        else:
            f = features(spec, store, dev)
            x = np.concatenate([f["train"], f["val"]])
        per_fold = []
        for k in range(N_FOLDS):
            tr, te = np.where(folds != k)[0], np.where(folds == k)[0]
            if kind == "cnn":
                p = cnn_fold(spec, epochs, [x[i] for i in tr], y[tr], [x[i] for i in te], dev)
            else:
                clf = logreg().fit(x[tr], y[tr])
                p = clf.predict_proba(x[te])[:, list(clf.classes_).index(1)]
            m = public(full_metrics(y[te], p, d[te], 0.5))
            per_fold.append(m)
            print(f"  N {name} fold {k}: n={len(te)} high={int(y[te].sum())} macro_f1 {m['macro_f1']:.3f} "
                  f"recall_high {m['recall_high']:.3f}")
        f1 = np.array([m["macro_f1"] for m in per_fold])
        rec = np.array([m["recall_high"] for m in per_fold])
        res = {"model": name, "kind": kind, "epochs": epochs, "per_fold": per_fold,
               "macro_f1": [float(f1.mean()), float(f1.std(ddof=1))],
               "recall_high": [float(rec.mean()), float(rec.std(ddof=1))]}
        out["models"][key] = res
        write_log_row({
            "run_id": f"sev2_N_cv5_{name[len('sev2_'):]}", "date": date.today().isoformat(), "owner": "haifa",
            "model": "severity",
            "change_from_previous": f"Run N (report only): 5-fold location-grouped CV on train+val for {name}",
            "key_settings": (f"StratifiedGroupKFold(5, shuffle, seed {SEED}) on location_id over train+val boxes "
                             f"(folds in experiments/sev2_N_cv5_trainval/folds.csv); "
                             + (f"fixed {epochs} epochs (median best epoch of the val runs), no early stopping, seed {SEED}"
                                if kind == "cnn" else "frozen features + logistic regression")
                             + "; threshold 0.5"),
            "val_main_metric": f"{f1.mean():.6f}",
            "val_other_metrics": json.dumps({"cv_macro_f1_std": float(f1.std(ddof=1)),
                                             "cv_recall_high_mean": float(rec.mean()),
                                             "cv_recall_high_std": float(rec.std(ddof=1)),
                                             "per_fold_macro_f1": f1.round(4).tolist()}),
            "notes": "v2 round 3; REPORT ONLY: never used for selection or threshold; val_main_metric = CV mean "
                     "over train+val folds, not the val split; test split not used."})
        print(f"N {name}: macro_f1 {f1.mean():.3f} ± {f1.std(ddof=1):.3f}, recall_high {rec.mean():.3f} ± {rec.std(ddof=1):.3f}")
    (run_dir / "metrics.json").write_text(json.dumps(out | {"test_used": False, "report_only": True}, indent=2))
    s["N"] = {k: {"model": v["model"], "macro_f1": v["macro_f1"], "recall_high": v["recall_high"]}
              for k, v in out["models"].items()}
    save_summary(s)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("steps", nargs="+", choices=["k", "l", "n"])
    args = ap.parse_args()
    os.environ.setdefault("TORCH_HOME", "/private/tmp/khota_torch_cache")
    store = DataStore(load_train_val())
    for step in args.steps:
        {"k": run_k, "l": run_l, "n": run_n}[step](store)


if __name__ == "__main__":
    main()
