"""Severity model, version 2 (binary: low_medium / high).

    python severity/train.py                 # Prompt 2: runs A, B, C, D (skips finished runs), then the threshold
    python severity/train.py --seeds 43 44   # repeat the Prompt 2 selected run with other seeds
    python severity/train.py --round2        # improvement round: E (box jitter), F (320 px), G (ResNet-50),
                                             # H (ConvNeXt-Tiny), I (3-seed ensemble), each 3 seeds
    python severity/train.py --force         # retrain runs even if experiments/<run_id>/metrics.json exists

Reads only `train` and `val` rows of data/split.csv. The test split is never loaded.
Structure follows severity/train_v1.py; preprocessing and the model builder are shared with severity/predict.py.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
from datetime import date
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/khota_matplotlib")

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms
from torchvision.transforms import InterpolationMode

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from khota_ml.config import CROP_EXPAND, DETECTION_CLASSES, HIGH, SEED as CONFIG_SEED, SEVERITY_CLASSES  # noqa: E402
from khota_ml.crop import crop_box  # noqa: E402
from khota_ml.data import find_image  # noqa: E402
from khota_ml.metrics import save_confusion_matrix, severity_metrics  # noqa: E402
from severity.predict import HIGH_INDEX, build_model, p_high_from_logits, resize_crop, to_tensor  # noqa: E402

ANNOTATIONS = ROOT / "data" / "annotations.csv"
SPLITS = ROOT / "data" / "split.csv"
EXPERIMENTS = ROOT / "experiments" / "experiments.csv"
SELECTION = ROOT / "experiments" / "severity_v2_selection.json"
ROUND2 = ROOT / "experiments" / "severity_v2_round2.json"
BATCH_SIZE = 32
MAX_EPOCHS = 20
PATIENCE = 5
LEARNING_RATE = 1e-4
SEEDS = (42, 43, 44)
THRESHOLDS = [round(0.10 + 0.05 * i, 2) for i in range(17)]  # 0.10 … 0.90
BASELINE = {"macro_f1": 0.465, "accuracy": 0.869}  # experiments/severity_majority_baseline (AGENTS.md section 7)
LOG_FIELDS = ["run_id", "date", "owner", "model", "change_from_previous", "key_settings",
              "val_main_metric", "val_other_metrics", "notes"]
SEVERITY_TO_ID = {name: i for i, name in enumerate(SEVERITY_CLASSES)}
AUGMENT_TEXT = "hflip+rotation_10deg+brightness_contrast_20pct (train only)"
# Box jitter (AGENTS.md section 8, approved exception 10 Nov 2026): train only, before crop_box.
JITTER_SHIFT = 0.10          # centre moves up to ±10% of box width / height
JITTER_SCALE = (0.85, 1.15)  # width and height scaled independently
JITTER_VARIANTS = 10         # pre-cut jittered crops per train box; one is drawn at random each time a box is used
JITTER_VAL_SEED = 2026       # fixed jittered copy of val: report-only, never used for selection


# ----------------------------------------------------------------------------------------------- data

def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_train_val() -> pd.DataFrame:
    """Defect boxes of train and val only (no_defect rows have no severity and are dropped)."""
    split = pd.read_csv(SPLITS, dtype={"image_id": str})
    split = split[split.split.isin(["train", "val"])][["image_id", "split"]]
    ann = pd.read_csv(ANNOTATIONS, dtype={"image_id": str})
    data = ann.merge(split, on="image_id", how="inner", validate="many_to_one")
    data = data[data.defect_class.isin(DETECTION_CLASSES) & data.severity.isin(SEVERITY_CLASSES)].copy()
    assert set(data.split) == {"train", "val"}
    return data.reset_index(drop=True)


def jitter_box(box, rng: np.random.Generator):
    cx, cy, w, h = box
    cx += rng.uniform(-JITTER_SHIFT, JITTER_SHIFT) * w
    cy += rng.uniform(-JITTER_SHIFT, JITTER_SHIFT) * h
    w *= rng.uniform(*JITTER_SCALE)
    h *= rng.uniform(*JITTER_SCALE)
    return (min(max(cx, 0.0), 1.0), min(max(cy, 0.0), 1.0), w, h)


def prepare_crops(frame: pd.DataFrame, size, jitter_seed: int | None = None, variants: int = 1,
                  expand: float = CROP_EXPAND):
    """crop_box (expand 0.20) on the full-resolution image, then resize (shared resize_crop).

    With `jitter_seed`, each box is jittered before crop_box; `variants` > 1 gives a list of crops per box.
    Rows keep the order of frame.groupby("image_file", sort=False), the same for every call.
    """
    rng = np.random.default_rng(jitter_seed) if jitter_seed is not None else None
    images, labels, defects = [], [], []
    for image_file, rows in frame.groupby("image_file", sort=False):
        with Image.open(find_image(str(image_file), ROOT / "raw")) as source:
            image = source.convert("RGB")
            for row in rows.itertuples(index=False):
                box = (row.x_center, row.y_center, row.width, row.height)
                if rng is None:
                    images.append(resize_crop(crop_box(image, box, expand=expand), size))
                else:
                    cut = [resize_crop(crop_box(image, jitter_box(box, rng), expand=CROP_EXPAND), size)
                           for _ in range(variants)]
                    images.append(cut if variants > 1 else cut[0])
                labels.append(SEVERITY_TO_ID[row.severity])
                defects.append(DETECTION_CLASSES.index(row.defect_class))
    return images, np.asarray(labels, dtype=np.int64), np.asarray(defects, dtype=np.int64)


class DataStore:
    """Caches crops per input size; train jitter variants are kept for one (size, seed) at a time."""

    def __init__(self, frame: pd.DataFrame):
        self.train = frame[frame.split == "train"]
        self.val = frame[frame.split == "val"]
        self._plain, self._jit_val, self._jit_train = {}, {}, (None, None)

    def plain(self, split: str, size, expand: float = CROP_EXPAND):
        key = (split, tuple(size), expand)
        if key not in self._plain:
            print(f"  cutting {split} crops at {size[0]}x{size[1]}, expand {expand}…")
            self._plain[key] = prepare_crops(self.train if split == "train" else self.val, size, expand=expand)
        return self._plain[key]

    def jittered_val(self, size):
        key = tuple(size)
        if key not in self._jit_val:
            print(f"  cutting jittered val crops at {size[0]}x{size[1]} (seed {JITTER_VAL_SEED}, report only)…")
            self._jit_val[key] = prepare_crops(self.val, size, jitter_seed=JITTER_VAL_SEED)
        return self._jit_val[key]

    def jittered_train(self, size, seed: int):
        key = (tuple(size), seed)
        if self._jit_train[0] != key:
            self._jit_train = (None, None)  # free the previous variants first
            print(f"  cutting {JITTER_VARIANTS} jittered train crops per box at {size[0]}x{size[1]} (seed {seed})…")
            self._jit_train = (key, prepare_crops(self.train, size, jitter_seed=seed, variants=JITTER_VARIANTS))
        return self._jit_train[1]


class CropDataset(Dataset):
    def __init__(self, images, labels, augment: bool):
        self.images, self.labels = images, labels
        self.augment = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(10, interpolation=InterpolationMode.BILINEAR, fill=0),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
        ]) if augment else None

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        image = self.images[i]
        if isinstance(image, list):  # jittered variants of one box
            image = image[random.randrange(len(image))]
        if self.augment is not None:
            image = np.array(self.augment(Image.fromarray(image)), dtype=np.uint8)
        return to_tensor(image), int(self.labels[i])


# -------------------------------------------------------------------------------------------- metrics

def full_metrics(y_true: np.ndarray, p_high: np.ndarray, defects: np.ndarray, threshold: float) -> dict:
    pred = np.where(p_high >= threshold, HIGH_INDEX, 1 - HIGH_INDEX)
    m = severity_metrics(y_true.tolist(), pred.tolist())
    cm = np.asarray(m["confusion_matrix"])
    tp = cm[HIGH_INDEX, HIGH_INDEX]
    out = {
        "threshold": threshold,
        "macro_f1": m["macro_f1"],
        "recall_high": float(tp / cm[HIGH_INDEX].sum()) if cm[HIGH_INDEX].sum() else 0.0,
        "precision_high": float(tp / cm[:, HIGH_INDEX].sum()) if cm[:, HIGH_INDEX].sum() else 0.0,
        "f1_per_class": m["f1_per_class"],
        "accuracy": m["accuracy"],
        "confusion_matrix": cm.tolist(),
        "macro_f1_by_defect": {},
        "_raw": {"overall": m, "by_defect": {}},
    }
    for d, name in enumerate(DETECTION_CLASSES):
        mask = defects == d
        md = severity_metrics(y_true[mask].tolist(), pred[mask].tolist())
        out["macro_f1_by_defect"][name] = md["macro_f1"]
        out["_raw"]["by_defect"][name] = md
    return out


def public(metrics: dict) -> dict:
    return {k: v for k, v in metrics.items() if not k.startswith("_")}


def predict_p_high(net: nn.Module, images, device) -> np.ndarray:
    net.eval()
    loader = DataLoader(CropDataset(images, np.zeros(len(images), dtype=np.int64), augment=False),
                        batch_size=BATCH_SIZE, shuffle=False)
    out = []
    with torch.inference_mode():
        for x, _ in loader:
            out.append(p_high_from_logits(net(x.to(device))).cpu().numpy())
    return np.concatenate(out)


# ----------------------------------------------------------------------------------------------- runs

def size_of(spec: dict) -> tuple:
    return tuple(spec.get("input_size", (224, 224)))


def expand_of(spec: dict) -> float:
    return float(spec.get("crop_expand", CROP_EXPAND))


def run_id_for(spec: dict, seed: int) -> str:
    rid = f"sev2_{spec['name']}_{spec['architecture'].replace('_', '')}_{'aug' if spec['augment'] else 'noaug'}_{spec['imbalance']}"
    if spec.get("jitter"):
        rid += "_jitter"
    if size_of(spec) != (224, 224):
        rid += f"_{size_of(spec)[0]}"
    if expand_of(spec) != CROP_EXPAND:
        rid += f"_expand{round(expand_of(spec) * 100):03d}"
    return rid if seed == CONFIG_SEED else f"{rid}_seed{seed}"


def train_run(spec: dict, seed: int, store: DataStore, device, force: bool, change: str | None = None) -> dict:
    run_id = run_id_for(spec, seed)
    run_dir = ROOT / "experiments" / run_id
    if (run_dir / "metrics.json").exists() and not force:
        print(f"\n{run_id}: already finished, using saved metrics")
        return json.loads((run_dir / "metrics.json").read_text())
    run_dir.mkdir(parents=True, exist_ok=True)

    size, expand = size_of(spec), expand_of(spec)
    if spec.get("jitter") and expand != CROP_EXPAND:
        raise ValueError("box jitter is only implemented for the default crop expand")
    _, tr_y, _ = store.plain("train", size, expand)
    tr_x = store.jittered_train(size, seed)[0] if spec.get("jitter") else store.plain("train", size, expand)[0]
    va_x, va_y, va_d = store.plain("val", size, expand)
    seed_everything(seed)
    counts = np.bincount(tr_y, minlength=len(SEVERITY_CLASSES)).astype(np.float64)
    generator = torch.Generator().manual_seed(seed)
    train_set = CropDataset(tr_x, tr_y, augment=spec["augment"])
    if spec["imbalance"] == "classweights":
        weights = torch.tensor(len(tr_y) / (len(counts) * counts), dtype=torch.float32, device=device)
        criterion = nn.CrossEntropyLoss(weight=weights)
        loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, generator=generator)
    elif spec["imbalance"] == "sampler":  # never together with class weights (AGENTS.md section 8)
        criterion = nn.CrossEntropyLoss()
        sample_w = torch.as_tensor((1.0 / counts)[tr_y], dtype=torch.double)
        sampler = WeightedRandomSampler(sample_w, num_samples=len(tr_y), replacement=True, generator=generator)
        loader = DataLoader(train_set, batch_size=BATCH_SIZE, sampler=sampler)
    else:
        raise ValueError(spec["imbalance"])

    net = build_model(spec["architecture"], pretrained=True).to(device)
    optimizer = torch.optim.AdamW(net.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    best_f1, best_epoch, stale, best_state, history = -1.0, 0, 0, None, []
    print(f"\n{run_id}: {len(tr_y)} train / {len(va_y)} val boxes, size={size[0]}, device={device}, seed={seed}")
    for epoch in range(1, MAX_EPOCHS + 1):
        net.train()
        loss_sum, seen = 0.0, 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(net(x), y)
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.item()) * len(y)
            seen += len(y)
        p = predict_p_high(net, va_x, device)
        m = full_metrics(va_y, p, va_d, 0.5)
        history.append({"epoch": epoch, "train_loss": loss_sum / seen, "val_macro_f1": m["macro_f1"],
                        "val_recall_high": m["recall_high"], "val_accuracy": m["accuracy"]})
        print(f"  epoch {epoch:02d}: train_loss={loss_sum / seen:.4f} val_macro_f1={m['macro_f1']:.4f} "
              f"recall_high={m['recall_high']:.3f}")
        if m["macro_f1"] > best_f1 + 1e-6:
            best_f1, best_epoch, stale = m["macro_f1"], epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
        else:
            stale += 1
            if stale >= PATIENCE:
                print(f"  early stop at epoch {epoch}; best epoch {best_epoch}")
                break

    net.load_state_dict(best_state)
    p_val = predict_p_high(net, va_x, device)
    metrics = full_metrics(va_y, p_val, va_d, 0.5)
    torch.save({"model_state_dict": best_state, "architecture": spec["architecture"],
                "severity_classes": SEVERITY_CLASSES, "detection_classes": DETECTION_CLASSES,
                "input_size": size, "crop_expand": expand, "threshold": 0.5,
                "run_id": run_id, "seed": seed, "best_epoch": best_epoch}, run_dir / "best_model.pt")
    pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)
    np.save(run_dir / "val_p_high.npy", p_val)
    save_outputs(run_dir, metrics)
    result = {"run_id": run_id, "spec": spec, "seed": seed, "best_epoch": best_epoch,
              "epochs_run": len(history), "device": device.type, "val": public(metrics),
              "baseline": BASELINE, "test_used": False}
    (run_dir / "metrics.json").write_text(json.dumps(result, indent=2))
    if change is None:
        change = spec["change"] if seed == CONFIG_SEED else f"Repeat of {run_id_for(spec, CONFIG_SEED)} with seed {seed}"
    log_run(result, change)
    return result


def jitter_eval(result: dict, store: DataStore, device) -> dict:
    """Add report-only metrics on the fixed jittered copy of val (never used for selection)."""
    run_dir = ROOT / "experiments" / result["run_id"]
    if "val_jitter" in result:
        return result
    ckpt = torch.load(run_dir / "best_model.pt", map_location="cpu", weights_only=False)
    net = build_model(ckpt["architecture"]).to(device)
    net.load_state_dict(ckpt["model_state_dict"])
    x, y, d = store.jittered_val(tuple(ckpt.get("input_size", (224, 224))))
    p = predict_p_high(net, x, device)
    np.save(run_dir / "val_jitter_p_high.npy", p)
    result["val_jitter"] = public(full_metrics(y, p, d, 0.5))
    (run_dir / "metrics.json").write_text(json.dumps(result, indent=2))
    return result


def save_outputs(run_dir: Path, metrics: dict, suffix: str = "") -> None:
    save_confusion_matrix(metrics["_raw"]["overall"], run_dir, stem=f"confusion_matrix{suffix}")
    for name, md in metrics["_raw"]["by_defect"].items():
        save_confusion_matrix(md, run_dir, stem=f"confusion_matrix_{name}{suffix}")


def settings_text(spec: dict, extra: str) -> str:
    size = size_of(spec)
    return (f"{spec['architecture']} ImageNet-pretrained; crop_box expand={expand_of(spec)} -> {size[0]}x{size[1]}; "
            f"augment={AUGMENT_TEXT if spec['augment'] else 'none'}; "
            f"box_jitter={'centre ±10%, w/h x0.85-1.15, train only, ' + str(JITTER_VARIANTS) + ' variants/box' if spec.get('jitter') else 'none'}; "
            f"imbalance={spec['imbalance']}; batch={BATCH_SIZE}; lr={LEARNING_RATE}; AdamW wd=1e-4; "
            f"max_epochs={MAX_EPOCHS}; early_stop=val macro_f1 patience {PATIENCE}; {extra}")


def write_log_row(row: dict) -> None:
    with EXPERIMENTS.open(encoding="utf-8", newline="") as fh:
        rows = [r for r in csv.DictReader(fh) if r["run_id"] != row["run_id"]]
    rows.append(row)
    with EXPERIMENTS.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=LOG_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def log_run(result: dict, change: str, notes: str | None = None) -> None:
    spec, v = result["spec"], result["val"]
    other = {k: v[k] for k in ("recall_high", "precision_high", "f1_per_class", "accuracy",
                               "macro_f1_by_defect", "confusion_matrix")}
    other.update({"baseline_macro_f1": BASELINE["macro_f1"], "baseline_accuracy": BASELINE["accuracy"]})
    note = "v2 binary; best epoch by val macro_f1 at threshold 0.5; test split not used."
    if spec.get("jitter"):
        note += " Box jitter = approved exception to AGENTS.md section 8 (10 Nov 2026)."
    write_log_row({
        "run_id": result["run_id"], "date": date.today().isoformat(), "owner": "haifa", "model": "severity",
        "change_from_previous": change,
        "key_settings": settings_text(spec, f"best_epoch={result['best_epoch']}; threshold=0.5; "
                                            f"seed={result['seed']}; device={result['device']}"),
        "val_main_metric": f"{v['macro_f1']:.6f}", "val_other_metrics": json.dumps(other, sort_keys=True),
        "notes": notes or note})


def scan_thresholds(p: np.ndarray, store: DataStore, size, out_dir: Path, stem: str) -> dict:
    """Threshold for `high` (0.10–0.90, step 0.05) maximising val macro_f1; ties -> closest to 0.5."""
    _, y, d = store.plain("val", size)
    scan = [full_metrics(y, p, d, t) for t in THRESHOLDS]
    best = max(scan, key=lambda m: (round(m["macro_f1"], 9), -abs(m["threshold"] - 0.5)))
    pd.DataFrame([{"threshold": m["threshold"], "macro_f1": m["macro_f1"], "recall_high": m["recall_high"],
                   "precision_high": m["precision_high"], "accuracy": m["accuracy"]} for m in scan]
                 ).to_csv(out_dir / f"{stem}.csv", index=False)
    save_outputs(out_dir, best, suffix="_tuned")
    return best


def choose_threshold(result: dict, store: DataStore) -> dict:
    run_dir = ROOT / "experiments" / result["run_id"]
    best = scan_thresholds(np.load(run_dir / "val_p_high.npy"), store, size_of(result["spec"]), run_dir,
                           "threshold_scan_val")
    ckpt_path = run_dir / "best_model.pt"
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    ckpt["threshold"] = best["threshold"]
    ckpt["threshold_note"] = "chosen on val macro_f1, grid 0.10-0.90 step 0.05"
    torch.save(ckpt, ckpt_path)
    result["val_tuned"] = public(best)
    (run_dir / "metrics.json").write_text(json.dumps(result, indent=2))
    return result


# --------------------------------------------------------------------------------------------- round 2

def group_stats(results: list[dict]) -> dict:
    def ms(key, block="val"):
        values = np.array([r[block][key] for r in results])
        return {"mean": float(values.mean()), "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                "per_seed": values.round(4).tolist()}
    return {"macro_f1": ms("macro_f1"), "recall_high": ms("recall_high"), "precision_high": ms("precision_high"),
            "accuracy": ms("accuracy"), "jitter_macro_f1": ms("macro_f1", "val_jitter"),
            "jitter_recall_high": ms("recall_high", "val_jitter"),
            "macro_f1_curb": {"mean": float(np.mean([r["val"]["macro_f1_by_defect"]["curb_ramp_issue"] for r in results]))},
            "macro_f1_surface": {"mean": float(np.mean([r["val"]["macro_f1_by_defect"]["surface_damage"] for r in results]))}}


def replaces(new: dict, best: dict) -> bool:
    """Round-2 rule: mean val macro_f1 >= best + 0.02, or within ±0.01 with a higher mean recall_high."""
    d = new["macro_f1"]["mean"] - best["macro_f1"]["mean"]
    return d >= 0.02 - 1e-9 or (abs(d) <= 0.01 + 1e-9 and new["recall_high"]["mean"] > best["recall_high"]["mean"])


def run_group(spec: dict, store: DataStore, device, force: bool) -> dict:
    results = []
    for seed in SEEDS:
        change = spec["change"] if seed == CONFIG_SEED else f"Repeat of {run_id_for(spec, CONFIG_SEED)} with seed {seed}"
        results.append(jitter_eval(train_run(spec, seed, store, device, force, change), store, device))
    stats = group_stats(results)
    print(f"  => {run_id_for(spec, CONFIG_SEED)}: macro_f1 {stats['macro_f1']['mean']:.3f} ± {stats['macro_f1']['std']:.3f}, "
          f"recall_high {stats['recall_high']['mean']:.3f} ± {stats['recall_high']['std']:.3f}, "
          f"jittered-val macro_f1 {stats['jitter_macro_f1']['mean']:.3f}")
    return {"run_id": run_id_for(spec, CONFIG_SEED), "spec": spec, "kind": "seeds", "stats": stats,
            "members": [r["run_id"] for r in results]}


def run_ensemble(group: dict, store: DataStore) -> dict:
    """Run I: mean p_high of the three seed models of `group`, at threshold 0.5."""
    spec = group["spec"]
    run_id = f"sev2_I_ensemble3_{group['run_id'][len('sev2_'):]}"
    run_dir = ROOT / "experiments" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    size = size_of(spec)
    _, y, d = store.plain("val", size)
    _, yj, dj = store.jittered_val(size)
    members = [ROOT / "experiments" / m for m in group["members"]]
    p = np.mean([np.load(m / "val_p_high.npy") for m in members], axis=0)
    pj = np.mean([np.load(m / "val_jitter_p_high.npy") for m in members], axis=0)
    np.save(run_dir / "val_p_high.npy", p)
    np.save(run_dir / "val_jitter_p_high.npy", pj)
    metrics, jmetrics = full_metrics(y, p, d, 0.5), full_metrics(yj, pj, dj, 0.5)
    save_outputs(run_dir, metrics)
    ensemble = {"run_id": run_id, "checkpoints": [str((m / "best_model.pt").relative_to(ROOT)) for m in members],
                "threshold": 0.5, "input_size": list(size), "combine": "mean p_high"}
    (run_dir / "ensemble.json").write_text(json.dumps(ensemble, indent=2))
    result = {"run_id": run_id, "spec": spec | {"name": "I"}, "seed": "42+43+44", "best_epoch": "-",
              "device": "mps", "val": public(metrics), "val_jitter": public(jmetrics), "baseline": BASELINE,
              "members": group["members"], "test_used": False}
    (run_dir / "metrics.json").write_text(json.dumps(result, indent=2))
    other = {k: result["val"][k] for k in ("recall_high", "precision_high", "f1_per_class", "accuracy",
                                           "macro_f1_by_defect", "confusion_matrix")}
    other.update({"baseline_macro_f1": BASELINE["macro_f1"], "baseline_accuracy": BASELINE["accuracy"],
                  "val_jitter_macro_f1_report_only": jmetrics["macro_f1"]})
    write_log_row({"run_id": run_id, "date": date.today().isoformat(), "owner": "haifa", "model": "severity",
                   "change_from_previous": f"Run I: ensemble (mean p_high) of the 3 seed models of {group['run_id']}",
                   "key_settings": settings_text(spec, f"members={','.join(group['members'])}; threshold=0.5"),
                   "val_main_metric": f"{metrics['macro_f1']:.6f}", "val_other_metrics": json.dumps(other, sort_keys=True),
                   "notes": "v2 binary; no training (combines existing checkpoints); single value, no seed std; test split not used."})
    one = lambda v: {"mean": float(v), "std": 0.0, "per_seed": [round(float(v), 4)]}  # noqa: E731
    stats = {"macro_f1": one(metrics["macro_f1"]), "recall_high": one(metrics["recall_high"]),
             "precision_high": one(metrics["precision_high"]), "accuracy": one(metrics["accuracy"]),
             "jitter_macro_f1": one(jmetrics["macro_f1"]), "jitter_recall_high": one(jmetrics["recall_high"]),
             "macro_f1_curb": {"mean": metrics["macro_f1_by_defect"]["curb_ramp_issue"]},
             "macro_f1_surface": {"mean": metrics["macro_f1_by_defect"]["surface_damage"]}}
    return {"run_id": run_id, "spec": result["spec"], "kind": "ensemble", "stats": stats, "members": group["members"]}


def round2(store: DataStore, device, force: bool) -> None:
    base_spec = json.loads(SELECTION.read_text())["spec"]  # run C
    c = run_group(base_spec, store, device, force=False)  # C seeds already trained; adds jittered-val only
    groups, decisions = [c], []
    best = c

    def consider(new):
        nonlocal best
        better = replaces(new["stats"], best["stats"])
        decisions.append({"run": new["run_id"], "vs": best["run_id"], "replaces": better,
                          "d_macro_f1": new["stats"]["macro_f1"]["mean"] - best["stats"]["macro_f1"]["mean"],
                          "d_recall_high": new["stats"]["recall_high"]["mean"] - best["stats"]["recall_high"]["mean"]})
        print(f"  {new['run_id']} {'REPLACES' if better else 'does not replace'} {best['run_id']}")
        if better:
            best = new

    def step(name, change, **updates):
        spec = best["spec"] | updates | {"name": name, "change": f"Run {name}: {best['run_id']} {change}"}
        group = run_group(spec, store, device, force)
        groups.append(group)
        consider(group)

    step("E", "+ box jitter on train (centre ±10%, w/h x0.85-1.15; approved exception)", jitter=True)
    step("F", "with input 320x320 instead of 224x224", input_size=[320, 320])
    step("G", "with ResNet-50 instead of ResNet-18", architecture="resnet50")
    step("H", "with ConvNeXt-Tiny instead of the ResNet backbone", architecture="convnext_tiny")
    ens = run_ensemble(best, store)
    groups.append(ens)
    consider(ens)

    # threshold scan for the best run on val: the deployable model (seed-42 checkpoint, or the ensemble)
    size = size_of(best["spec"])
    best_dir = ROOT / "experiments" / best["run_id"]
    p = np.load(best_dir / "val_p_high.npy")
    tuned = scan_thresholds(p, store, size, best_dir, "threshold_scan_val_round2")
    summary = {"baseline": BASELINE, "groups": groups, "decisions": decisions, "best": best["run_id"],
               "threshold": public(tuned), "jitter_val_seed": JITTER_VAL_SEED,
               "rule": "replace if mean val macro_f1 >= best + 0.02, or within ±0.01 and higher mean recall_high",
               "test_used": False}
    ROUND2.write_text(json.dumps(summary, indent=2))
    print(f"\nbest after round 2: {best['run_id']}; threshold {tuned['threshold']} -> macro_f1 {tuned['macro_f1']:.4f}, "
          f"recall_high {tuned['recall_high']:.3f}. Test split not used.")


# ----------------------------------------------------------------------------------------------- main

RUN_A = {"name": "A", "architecture": "resnet18", "augment": False, "imbalance": "classweights",
         "change": "Run A: v1 winner (ResNet-18, crop, no aug, class weights), now binary"}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, nargs="*", help="repeat the Prompt 2 selected run with these seeds")
    ap.add_argument("--round2", action="store_true", help="improvement round E-I")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    os.environ.setdefault("TORCH_HOME", "/private/tmp/khota_torch_cache")
    device = torch.device(os.environ.get("KHOTA_SEVERITY_DEVICE") or ("mps" if torch.backends.mps.is_available() else "cpu"))
    frame = load_train_val()
    store = DataStore(frame)
    print(f"train boxes={len(store.train)} (high={int((store.train.severity == HIGH).sum())}), "
          f"val boxes={len(store.val)} (high={int((store.val.severity == HIGH).sum())})")

    if args.round2:
        round2(store, device, args.force)
        return
    if args.seeds:
        selection = json.loads(SELECTION.read_text())
        for seed in args.seeds:
            train_run(selection["spec"], seed, store, device, args.force)
        return

    a = train_run(RUN_A, CONFIG_SEED, store, device, args.force)
    spec_b = RUN_A | {"name": "B", "augment": True, "change": "Run B: Run A + allowed train-only augmentation"}
    b = train_run(spec_b, CONFIG_SEED, store, device, args.force)
    ab = b if b["val"]["macro_f1"] > a["val"]["macro_f1"] else a
    spec_c = ab["spec"] | {"name": "C", "imbalance": "sampler",
                           "change": f"Run C: {ab['run_id']} with weighted random sampler instead of class weights"}
    c = train_run(spec_c, CONFIG_SEED, store, device, args.force)
    best = max((a, b, c), key=lambda r: r["val"]["macro_f1"])  # ties keep the earlier run
    spec_d = best["spec"] | {"name": "D", "architecture": "efficientnet_b0",
                             "change": f"Run D: {best['run_id']} with EfficientNet-B0 instead of ResNet-18"}
    d = train_run(spec_d, CONFIG_SEED, store, device, args.force)
    best = max((a, b, c, d), key=lambda r: r["val"]["macro_f1"])
    best = choose_threshold(best, store)
    SELECTION.write_text(json.dumps({"run_id": best["run_id"], "spec": best["spec"],
                                     "threshold": best["val_tuned"]["threshold"]}, indent=2))
    print("\nrun | macro_f1 | recall_high | precision_high | accuracy")
    for r in (a, b, c, d):
        v = r["val"]
        print(f"{r['run_id']} | {v['macro_f1']:.4f} | {v['recall_high']:.3f} | {v['precision_high']:.3f} | {v['accuracy']:.3f}")
    t = best["val_tuned"]
    print(f"selected {best['run_id']}; threshold {t['threshold']}: macro_f1={t['macro_f1']:.4f} "
          f"recall_high={t['recall_high']:.3f}. Test split not used.")


if __name__ == "__main__":
    main()
