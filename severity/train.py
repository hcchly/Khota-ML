"""Severity model, version 2 (binary: low_medium / high) — Prompt 2, runs A–D.

    python severity/train.py                 # runs A, B, C, D (skips runs already finished), then the threshold
    python severity/train.py --seeds 43 44   # repeat the selected run with other seeds
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
from severity.predict import (  # noqa: E402
    HIGH_INDEX, IMAGE_SIZE, build_model, p_high_from_logits, resize_crop, to_tensor,
)

ANNOTATIONS = ROOT / "data" / "annotations.csv"
SPLITS = ROOT / "data" / "split.csv"
EXPERIMENTS = ROOT / "experiments" / "experiments.csv"
SELECTION = ROOT / "experiments" / "severity_v2_selection.json"
BATCH_SIZE = 32
MAX_EPOCHS = 20
PATIENCE = 5
LEARNING_RATE = 1e-4
THRESHOLDS = [round(0.10 + 0.05 * i, 2) for i in range(17)]  # 0.10 … 0.90
BASELINE = {"macro_f1": 0.465, "accuracy": 0.869}  # experiments/severity_majority_baseline (AGENTS.md section 7)
LOG_FIELDS = ["run_id", "date", "owner", "model", "change_from_previous", "key_settings",
              "val_main_metric", "val_other_metrics", "notes"]
SEVERITY_TO_ID = {name: i for i, name in enumerate(SEVERITY_CLASSES)}
AUGMENT_TEXT = "hflip+rotation_10deg+brightness_contrast_20pct (train only)"


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


def prepare_crops(frame: pd.DataFrame):
    """crop_box (expand 0.20) on the full-resolution image, then resize to 224x224 (shared resize_crop)."""
    images, labels, defects = [], [], []
    for image_file, rows in frame.groupby("image_file", sort=False):
        with Image.open(find_image(str(image_file), ROOT / "raw")) as source:
            image = source.convert("RGB")
            for row in rows.itertuples(index=False):
                crop = crop_box(image, (row.x_center, row.y_center, row.width, row.height), expand=CROP_EXPAND)
                images.append(resize_crop(crop))
                labels.append(SEVERITY_TO_ID[row.severity])
                defects.append(DETECTION_CLASSES.index(row.defect_class))
    return images, np.asarray(labels, dtype=np.int64), np.asarray(defects, dtype=np.int64)


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
        if self.augment is not None:
            image = np.array(self.augment(Image.fromarray(image)), dtype=np.uint8)  # writable copy
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

def run_id_for(spec: dict, seed: int) -> str:
    rid = f"sev2_{spec['name']}_{spec['architecture'].replace('_', '')}_{'aug' if spec['augment'] else 'noaug'}_{spec['imbalance']}"
    return rid if seed == CONFIG_SEED else f"{rid}_seed{seed}"


def train_run(spec: dict, seed: int, data: dict, device, force: bool) -> dict:
    run_id = run_id_for(spec, seed)
    run_dir = ROOT / "experiments" / run_id
    if (run_dir / "metrics.json").exists() and not force:
        print(f"\n{run_id}: already finished, using saved metrics")
        return json.loads((run_dir / "metrics.json").read_text())
    run_dir.mkdir(parents=True, exist_ok=True)

    seed_everything(seed)
    tr_x, tr_y = data["train_images"], data["train_labels"]
    va_x, va_y, va_d = data["val_images"], data["val_labels"], data["val_defects"]
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
    print(f"\n{run_id}: {len(tr_y)} train / {len(va_y)} val boxes, device={device}, seed={seed}")
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
                "input_size": IMAGE_SIZE, "crop_expand": CROP_EXPAND, "threshold": 0.5,
                "run_id": run_id, "seed": seed, "best_epoch": best_epoch}, run_dir / "best_model.pt")
    pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)
    np.save(run_dir / "val_p_high.npy", p_val)
    save_outputs(run_dir, metrics)
    result = {"run_id": run_id, "spec": spec, "seed": seed, "best_epoch": best_epoch,
              "epochs_run": len(history), "device": device.type, "val": public(metrics),
              "baseline": BASELINE, "test_used": False}
    (run_dir / "metrics.json").write_text(json.dumps(result, indent=2))
    log_run(result, spec["change"] if seed == CONFIG_SEED else f"Repeat of {run_id_for(spec, CONFIG_SEED)} with seed {seed}")
    return result


def save_outputs(run_dir: Path, metrics: dict, suffix: str = "") -> None:
    save_confusion_matrix(metrics["_raw"]["overall"], run_dir, stem=f"confusion_matrix{suffix}")
    for name, md in metrics["_raw"]["by_defect"].items():
        save_confusion_matrix(md, run_dir, stem=f"confusion_matrix_{name}{suffix}")


def log_run(result: dict, change: str) -> None:
    spec, v = result["spec"], result["val"]
    settings = (f"{spec['architecture']} ImageNet-pretrained; crop_box expand={CROP_EXPAND} -> {IMAGE_SIZE[0]}x{IMAGE_SIZE[1]}; "
                f"augment={AUGMENT_TEXT if spec['augment'] else 'none'}; imbalance={spec['imbalance']}; "
                f"batch={BATCH_SIZE}; lr={LEARNING_RATE}; AdamW wd=1e-4; max_epochs={MAX_EPOCHS}; "
                f"early_stop=val macro_f1 patience {PATIENCE}; best_epoch={result['best_epoch']}; "
                f"threshold=0.5; seed={result['seed']}; device={result['device']}")
    other = {k: v[k] for k in ("recall_high", "precision_high", "f1_per_class", "accuracy",
                               "macro_f1_by_defect", "confusion_matrix")}
    other.update({"baseline_macro_f1": BASELINE["macro_f1"], "baseline_accuracy": BASELINE["accuracy"]})
    row = {"run_id": result["run_id"], "date": date.today().isoformat(), "owner": "haifa", "model": "severity",
           "change_from_previous": change, "key_settings": settings, "val_main_metric": f"{v['macro_f1']:.6f}",
           "val_other_metrics": json.dumps(other, sort_keys=True),
           "notes": "v2 binary; best epoch by val macro_f1 at threshold 0.5; test split not used."}
    with EXPERIMENTS.open(encoding="utf-8", newline="") as fh:
        rows = [r for r in csv.DictReader(fh) if r["run_id"] != row["run_id"]]
    rows.append(row)
    with EXPERIMENTS.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=LOG_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def choose_threshold(result: dict, data: dict) -> dict:
    """Pick the p_high threshold (0.10–0.90, step 0.05) that maximises val macro_f1; ties -> closest to 0.5."""
    run_dir = ROOT / "experiments" / result["run_id"]
    p = np.load(run_dir / "val_p_high.npy")
    scan = [full_metrics(data["val_labels"], p, data["val_defects"], t) for t in THRESHOLDS]
    best = max(scan, key=lambda m: (round(m["macro_f1"], 9), -abs(m["threshold"] - 0.5)))
    pd.DataFrame([{"threshold": m["threshold"], "macro_f1": m["macro_f1"], "recall_high": m["recall_high"],
                   "precision_high": m["precision_high"], "accuracy": m["accuracy"]} for m in scan]
                 ).to_csv(run_dir / "threshold_scan_val.csv", index=False)
    save_outputs(run_dir, best, suffix="_tuned")
    ckpt_path = run_dir / "best_model.pt"
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    ckpt["threshold"] = best["threshold"]
    ckpt["threshold_note"] = "chosen on val macro_f1, grid 0.10-0.90 step 0.05"
    torch.save(ckpt, ckpt_path)
    result["val_tuned"] = public(best)
    (run_dir / "metrics.json").write_text(json.dumps(result, indent=2))
    return result


# ----------------------------------------------------------------------------------------------- main

RUN_A = {"name": "A", "architecture": "resnet18", "augment": False, "imbalance": "classweights",
         "change": "Run A: v1 winner (ResNet-18, crop, no aug, class weights), now binary"}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, nargs="*", help="repeat the selected run with these seeds")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    os.environ.setdefault("TORCH_HOME", "/private/tmp/khota_torch_cache")
    device = torch.device(os.environ.get("KHOTA_SEVERITY_DEVICE") or ("mps" if torch.backends.mps.is_available() else "cpu"))
    frame = load_train_val()
    tr, va = frame[frame.split == "train"], frame[frame.split == "val"]
    print(f"train boxes={len(tr)} (high={int((tr.severity == HIGH).sum())}), "
          f"val boxes={len(va)} (high={int((va.severity == HIGH).sum())}); cropping…")
    data = {}
    data["train_images"], data["train_labels"], _ = prepare_crops(tr)
    data["val_images"], data["val_labels"], data["val_defects"] = prepare_crops(va)

    if args.seeds:
        selection = json.loads(SELECTION.read_text())
        for seed in args.seeds:
            train_run(selection["spec"], seed, data, device, args.force)
        return

    a = train_run(RUN_A, CONFIG_SEED, data, device, args.force)
    spec_b = RUN_A | {"name": "B", "augment": True, "change": "Run B: Run A + allowed train-only augmentation"}
    b = train_run(spec_b, CONFIG_SEED, data, device, args.force)
    ab = b if b["val"]["macro_f1"] > a["val"]["macro_f1"] else a
    spec_c = ab["spec"] | {"name": "C", "imbalance": "sampler",
                           "change": f"Run C: {ab['run_id']} with weighted random sampler instead of class weights"}
    c = train_run(spec_c, CONFIG_SEED, data, device, args.force)
    best = max((a, b, c), key=lambda r: r["val"]["macro_f1"])  # ties keep the earlier run
    spec_d = best["spec"] | {"name": "D", "architecture": "efficientnet_b0",
                             "change": f"Run D: {best['run_id']} with EfficientNet-B0 instead of ResNet-18"}
    d = train_run(spec_d, CONFIG_SEED, data, device, args.force)
    best = max((a, b, c, d), key=lambda r: r["val"]["macro_f1"])
    best = choose_threshold(best, data)
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
