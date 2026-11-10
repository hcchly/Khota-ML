"""Train and compare the three severity experiments required by PROMPT 2.

This script reads only train and val examples. The locked test split is never
merged into annotations or used to select a model.
"""

from __future__ import annotations

import csv
import argparse
import json
import os
import random
import sys
from datetime import date
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/khota_matplotlib")

import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from torchvision.models import EfficientNet_B0_Weights, ResNet18_Weights
from torchvision.transforms import InterpolationMode

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from khota_ml.config import DETECTION_CLASSES, SEED as CONFIG_SEED, SEVERITY_CLASSES  # noqa: E402
from khota_ml.crop import crop_box  # noqa: E402
from khota_ml.data import find_image  # noqa: E402
from khota_ml.metrics import save_confusion_matrix, severity_metrics  # noqa: E402

ANNOTATIONS = ROOT / "data" / "annotations.csv"
SPLITS = ROOT / "data" / "split.csv"
EXPERIMENTS = ROOT / "experiments" / "experiments.csv"
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
MAX_EPOCHS = 20
PATIENCE = 5
LEARNING_RATE = 1e-4
LOG_FIELDS = [
    "run_id", "date", "owner", "model", "change_from_previous", "key_settings",
    "val_main_metric", "val_other_metrics", "notes",
]
FORCE_RUN_IDS = {name.strip() for name in os.environ.get("KHOTA_SEVERITY_FORCE_RUNS", "").split(",") if name.strip()}
SEVERITY_TO_ID = {name: index for index, name in enumerate(SEVERITY_CLASSES)}
SEED = CONFIG_SEED


def seed_everything(seed=None) -> None:
    seed = SEED if seed is None else seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_train_val() -> pd.DataFrame:
    """Join only train/val image IDs and remove no_defect rows."""
    split = pd.read_csv(SPLITS, dtype={"image_id": str})
    split = split[split["split"].isin(["train", "val"])][["image_id", "split"]]
    annotations = pd.read_csv(ANNOTATIONS, dtype={"image_id": str})
    annotations = annotations[annotations["image_id"].isin(set(split.image_id))]
    data = annotations.merge(split, on="image_id", how="inner", validate="many_to_one")
    data = data[data.defect_class.isin(DETECTION_CLASSES) & data.severity.isin(SEVERITY_CLASSES)].copy()
    if data.empty or set(data.split) != {"train", "val"}:
        raise ValueError("expected defect boxes in both train and val")
    return data.reset_index(drop=True)


def prepare_examples(frame: pd.DataFrame, input_mode: str) -> tuple[list[np.ndarray], np.ndarray, np.ndarray]:
    """Crop at original resolution (or take full image), then resize to IMAGE_SIZE."""
    if input_mode not in {"crop", "full_image"}:
        raise ValueError(f"unknown input mode: {input_mode}")
    images: list[np.ndarray] = []
    labels: list[int] = []
    defects: list[int] = []
    for image_file, rows in frame.groupby("image_file", sort=False):
        path = find_image(str(image_file), ROOT / "raw")
        with Image.open(path) as source:
            image = source.convert("RGB")
            for row in rows.itertuples(index=False):
                if input_mode == "crop":
                    sample = crop_box(
                        image,
                        (row.x_center, row.y_center, row.width, row.height),
                        expand=0.20,
                    )
                else:
                    sample = image
                sample = sample.resize(IMAGE_SIZE, Image.Resampling.BILINEAR)
                images.append(np.asarray(sample, dtype=np.uint8).copy())
                labels.append(SEVERITY_TO_ID[row.severity])
                defects.append(DETECTION_CLASSES.index(row.defect_class))
    return images, np.asarray(labels, dtype=np.int64), np.asarray(defects, dtype=np.int64)


def prepare_geometry(frame: pd.DataFrame) -> np.ndarray:
    """Return the requested normalized box geometry in the same row order as crops."""
    geometry = []
    for _, rows in frame.groupby("image_file", sort=False):
        for row in rows.itertuples(index=False):
            width, height = float(row.width), float(row.height)
            geometry.append([width, height, width * height, float(row.x_center), float(row.y_center)])
    return np.asarray(geometry, dtype=np.float32)


class SeverityDataset(Dataset):
    def __init__(self, images: list[np.ndarray], labels: np.ndarray, training: bool, augment: bool):
        self.images = images
        self.labels = labels
        self.transform = transforms.Compose(
            ([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(10, interpolation=InterpolationMode.BILINEAR, fill=0),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
            ] if training and augment else [])
            + [
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int):
        image = Image.fromarray(self.images[index])
        return self.transform(image), int(self.labels[index])


class GeometrySeverityDataset(SeverityDataset):
    def __init__(self, images, labels, geometry, training: bool):
        super().__init__(images, labels, training=training, augment=False)
        self.geometry = geometry

    def __getitem__(self, index: int):
        image = Image.fromarray(self.images[index])
        return self.transform(image), torch.from_numpy(self.geometry[index]), int(self.labels[index])


def make_loaders(train_arrays, train_labels, val_arrays, val_labels, augment: bool):
    train_set = SeverityDataset(train_arrays, train_labels, training=True, augment=augment)
    val_set = SeverityDataset(val_arrays, val_labels, training=False, augment=False)
    generator = torch.Generator().manual_seed(SEED)
    return (
        DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, generator=generator),
        DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=0),
    )


def class_weights(labels: np.ndarray, device: torch.device) -> torch.Tensor:
    counts = np.bincount(labels, minlength=len(SEVERITY_CLASSES)).astype(np.float32)
    if np.any(counts == 0):
        raise ValueError(f"training data must contain all severity classes; counts={counts.tolist()}")
    weights = len(labels) / (len(SEVERITY_CLASSES) * counts)
    return torch.tensor(weights, dtype=torch.float32, device=device)


def make_model(device: torch.device, architecture: str = "resnet18") -> nn.Module:
    if architecture == "resnet18":
        model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        model.fc = nn.Linear(model.fc.in_features, len(SEVERITY_CLASSES))
    elif architecture == "efficientnet_b0":
        model = models.efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(SEVERITY_CLASSES))
    else:
        raise ValueError(f"unsupported architecture: {architecture}")
    return model.to(device)


class ResNet18WithGeometry(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        self.classifier = nn.Linear(features + 5, len(SEVERITY_CLASSES))

    def forward(self, image: torch.Tensor, geometry: torch.Tensor) -> torch.Tensor:
        visual = self.backbone(image)
        return self.classifier(torch.cat((visual, geometry), dim=1))


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[dict, float, list[int]]:
    model.eval()
    truths: list[int] = []
    predictions: list[int] = []
    total_loss = 0.0
    total = 0
    with torch.inference_mode():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            logits = model(inputs)
            loss = nn.functional.cross_entropy(logits, labels)
            total_loss += float(loss.item()) * labels.size(0)
            total += labels.size(0)
            truths.extend(labels.cpu().tolist())
            predictions.extend(logits.argmax(dim=1).cpu().tolist())
    metrics = severity_metrics(truths, predictions)
    return metrics, total_loss / max(total, 1), predictions


def model_metrics(y_true: np.ndarray, predictions: list[int], defect_ids: np.ndarray) -> dict:
    overall = severity_metrics(y_true.tolist(), predictions)
    by_defect = {}
    for defect_id, defect_name in enumerate(DETECTION_CLASSES):
        mask = defect_ids == defect_id
        by_defect[defect_name] = severity_metrics(y_true[mask].tolist(), np.asarray(predictions)[mask].tolist())
    return {"overall": overall, "by_defect": by_defect}


def as_json(metrics: dict, baseline: dict) -> dict:
    def plain(m):
        return {
            "accuracy": m["accuracy"], "macro_f1": m["macro_f1"],
            "f1_per_class": m["f1_per_class"],
            "confusion_matrix": np.asarray(m["confusion_matrix"]).tolist(),
            "class_names": m["class_names"],
        }

    return {
        "overall": plain(metrics["overall"]),
        "by_defect": {name: plain(value) for name, value in metrics["by_defect"].items()},
        "baseline": {
            "majority_class": baseline["majority_class"],
            "accuracy": baseline["overall"]["accuracy"],
            "macro_f1": baseline["overall"]["macro_f1"],
            "by_defect": {
                name: {"accuracy": value["accuracy"], "macro_f1": value["macro_f1"]}
                for name, value in baseline["by_defect"].items()
            },
        },
    }


def read_logged_run(run_id: str):
    metrics_path = ROOT / "experiments" / run_id / "metrics.json"
    if not metrics_path.exists():
        return None
    with metrics_path.open(encoding="utf-8") as handle:
        return json.load(handle)


def append_experiment(run_id: str, change: str, augment: bool, input_mode: str,
                      result: dict, epochs: int, device: torch.device, geometry: bool = False,
                      architecture: str = "resnet18"):
    metrics = result["overall"]
    baseline = result["baseline"]
    baseline_overall = baseline["overall"]
    settings = (
        f"{architecture.replace('_', '-').title()} ImageNet pretrained; input={input_mode}; size={IMAGE_SIZE[0]}x{IMAGE_SIZE[1]}; "
        f"augment={'horizontal_flip+rotation_10deg+brightness_contrast_20pct' if augment else 'none'}; "
        f"class_weight=inverse_train_frequency; batch_size={BATCH_SIZE}; lr={LEARNING_RATE}; "
        f"best_epoch={epochs}; early_stop_patience={PATIENCE}; max_epochs={MAX_EPOCHS}; "
        f"device={device.type}; seed={SEED}"
    )
    if geometry:
        settings += "; geometry=[width,height,width*height,x_center,y_center] concatenated to 512 CNN features"
    other = {
        "accuracy": metrics["accuracy"],
        "f1_per_class": metrics["f1_per_class"],
        "macro_f1_by_defect": {name: values["macro_f1"] for name, values in result["by_defect"].items()},
        "accuracy_by_defect": {name: values["accuracy"] for name, values in result["by_defect"].items()},
        "baseline_accuracy": baseline_overall["accuracy"],
        "baseline_macro_f1": baseline_overall["macro_f1"],
        "baseline_macro_f1_by_defect": {name: values["macro_f1"] for name, values in baseline["by_defect"].items()},
    }
    row = {
        "run_id": run_id,
        "date": date.today().isoformat(),
        "owner": "haifa",
        "model": "severity",
        "change_from_previous": change,
        "key_settings": settings,
        "val_main_metric": f"{metrics['macro_f1']:.6f}",
        "val_other_metrics": json.dumps(other, sort_keys=True),
        "notes": "Best checkpoint selected by val macro_f1; test split not used.",
    }
    EXPERIMENTS.parent.mkdir(parents=True, exist_ok=True)
    existing = []
    if EXPERIMENTS.exists():
        with EXPERIMENTS.open(encoding="utf-8", newline="") as handle:
            existing = list(csv.DictReader(handle))
    matching = [item for item in existing if item.get("run_id") == run_id]
    if matching:
        if matching[-1] == row:
            return
        updated = [row if item.get("run_id") == run_id else item for item in existing]
        with EXPERIMENTS.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=LOG_FIELDS, lineterminator="\n")
            writer.writeheader()
            writer.writerows(updated)
        return
    with EXPERIMENTS.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=LOG_FIELDS, lineterminator="\n")
        if not existing:
            writer.writeheader()
        writer.writerow(row)


def run_experiment(
    run_id: str,
    change: str,
    input_mode: str,
    augment: bool,
    train_arrays: list[np.ndarray],
    train_labels: np.ndarray,
    train_defects: np.ndarray,
    val_arrays: list[np.ndarray],
    val_labels: np.ndarray,
    val_defects: np.ndarray,
    baseline: dict,
    device: torch.device,
    architecture: str = "resnet18",
) -> dict:
    run_dir = ROOT / "experiments" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    logged = read_logged_run(run_id)
    needs_refresh = logged is not None and (
        logged.get("input_mode") != input_mode or logged.get("augment") != augment
    )
    if logged is not None and run_id not in FORCE_RUN_IDS and not needs_refresh:
        print(f"\nSkipping completed run {run_id}; using its saved val metrics.")
        append_experiment(run_id, change, augment, input_mode,
                          logged | {"baseline": baseline}, int(logged["best_epoch"]), device,
                          architecture=architecture)
        print_result(logged)
        return logged

    seed_everything()
    train_loader, val_loader = make_loaders(train_arrays, train_labels, val_arrays, val_labels, augment)
    model = make_model(device, architecture)
    weights = class_weights(train_labels, device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    best_f1 = -1.0
    best_epoch = 0
    stale_epochs = 0
    best_state = None
    history = []

    print(f"\nStarting {run_id}: {len(train_labels)} train boxes, {len(val_labels)} val boxes, device={device}")
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        running_loss = 0.0
        seen = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(inputs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            running_loss += float(loss.item()) * labels.size(0)
            seen += labels.size(0)

        val_result, val_loss, _ = evaluate(model, val_loader, device)
        epoch_row = {
            "epoch": epoch,
            "train_loss": running_loss / max(seen, 1),
            "val_loss": val_loss,
            "val_accuracy": val_result["accuracy"],
            "val_macro_f1": val_result["macro_f1"],
        }
        history.append(epoch_row)
        print(
            f"  epoch {epoch:02d}/{MAX_EPOCHS}: train_loss={epoch_row['train_loss']:.4f} "
            f"val_loss={val_loss:.4f} val_macro_f1={val_result['macro_f1']:.4f}"
        )
        if val_result["macro_f1"] > best_f1 + 1e-6:
            best_f1 = val_result["macro_f1"]
            best_epoch = epoch
            stale_epochs = 0
            best_state = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}
        else:
            stale_epochs += 1
        if stale_epochs >= PATIENCE:
            print(f"  early stopping after {epoch} epochs; best epoch was {best_epoch}")
            break

    if best_state is None:
        raise RuntimeError(f"no best checkpoint produced for {run_id}")
    checkpoint_path = run_dir / "best_model.pt"
    torch.save({
        "model_state_dict": best_state,
        "severity_classes": SEVERITY_CLASSES,
        "detection_classes": DETECTION_CLASSES,
        "input_mode": input_mode,
        "input_size": IMAGE_SIZE,
        "architecture": architecture,
        "best_epoch": best_epoch,
        "seed": SEED,
    }, checkpoint_path)
    pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)

    model.load_state_dict(best_state)
    val_loader = DataLoader(
        SeverityDataset(val_arrays, val_labels, training=False, augment=False),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=0,
    )
    _, _, predictions = evaluate(model, val_loader, device)
    detailed = model_metrics(val_labels, predictions, val_defects)
    serialized = as_json(detailed, baseline)
    serialized["run_id"] = run_id
    serialized["input_mode"] = input_mode
    serialized["augment"] = augment
    serialized["best_epoch"] = best_epoch
    serialized["device"] = device.type
    serialized["checkpoint"] = checkpoint_path.name
    serialized["architecture"] = architecture
    with (run_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(serialized, handle, indent=2)
    save_confusion_matrix(detailed["overall"], run_dir)
    for defect_name, defect_metrics in detailed["by_defect"].items():
        save_confusion_matrix(defect_metrics, run_dir, stem=f"confusion_matrix_{defect_name}")
    append_experiment(run_id, change, augment, input_mode,
                      detailed | {"baseline": baseline}, best_epoch, device,
                      architecture=architecture)
    print_result(serialized)
    return serialized


def print_result(result: dict) -> None:
    overall = result["overall"]
    baseline = result["baseline"]
    print(f"Val macro_f1: {overall['macro_f1']:.4f} (baseline {baseline['macro_f1']:.4f})")
    print(f"Val accuracy: {overall['accuracy']:.4f} (baseline {baseline['accuracy']:.4f})")
    print(f"F1 per class: {overall['f1_per_class']}")
    for name, metrics in result["by_defect"].items():
        base = baseline["by_defect"][name]
        print(f"  {name}: macro_f1={metrics['macro_f1']:.4f} (baseline {base['macro_f1']:.4f})")


def run_geometry_experiment(train_arrays, train_labels, train_defects, train_geometry,
                            val_arrays, val_labels, val_defects, val_geometry,
                            baseline, device):
    run_id = "severity_resnet18_crop_noaug_geometry"
    run_dir = ROOT / "experiments" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    seed_everything()
    train_set = GeometrySeverityDataset(train_arrays, train_labels, train_geometry, training=True)
    val_set = GeometrySeverityDataset(val_arrays, val_labels, val_geometry, training=False)
    generator = torch.Generator().manual_seed(SEED)
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, generator=generator)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    model = ResNet18WithGeometry().to(device)
    weights = class_weights(train_labels, device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    best_f1, best_epoch, stale_epochs, best_state = -1.0, 0, 0, None
    history = []
    print(f"\nStarting {run_id}: {len(train_labels)} train boxes, {len(val_labels)} val boxes, device={device}")
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        running_loss = 0.0
        seen = 0
        for inputs, geometry, labels in train_loader:
            inputs, geometry, labels = inputs.to(device), geometry.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(inputs, geometry)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            running_loss += float(loss.item()) * labels.size(0)
            seen += labels.size(0)
        model.eval()
        truths, preds = [], []
        val_loss_sum = 0.0
        with torch.inference_mode():
            for inputs, geometry, labels in val_loader:
                inputs, geometry, labels = inputs.to(device), geometry.to(device), labels.to(device)
                logits = model(inputs, geometry)
                val_loss_sum += float(criterion(logits, labels).item()) * labels.size(0)
                truths.extend(labels.cpu().tolist())
                preds.extend(logits.argmax(1).cpu().tolist())
        val_result = severity_metrics(truths, preds)
        val_loss = val_loss_sum / max(len(val_labels), 1)
        history.append({"epoch": epoch, "train_loss": running_loss / max(seen, 1),
                        "val_loss": val_loss, "val_accuracy": val_result["accuracy"],
                        "val_macro_f1": val_result["macro_f1"]})
        print(f"  epoch {epoch:02d}/{MAX_EPOCHS}: train_loss={history[-1]['train_loss']:.4f} "
              f"val_loss={val_loss:.4f} val_macro_f1={val_result['macro_f1']:.4f}")
        if val_result["macro_f1"] > best_f1 + 1e-6:
            best_f1, best_epoch, stale_epochs = val_result["macro_f1"], epoch, 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            stale_epochs += 1
        if stale_epochs >= PATIENCE:
            print(f"  early stopping after {epoch} epochs; best epoch was {best_epoch}")
            break
    if best_state is None:
        raise RuntimeError("no best checkpoint produced for geometry run")
    checkpoint_path = run_dir / "best_model.pt"
    torch.save({"model_state_dict": best_state, "severity_classes": SEVERITY_CLASSES,
                "detection_classes": DETECTION_CLASSES, "input_mode": "crop", "input_size": IMAGE_SIZE,
                "geometry_features": ["width", "height", "width*height", "x_center", "y_center"],
                "best_epoch": best_epoch, "seed": SEED}, checkpoint_path)
    pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)
    model.load_state_dict(best_state)
    model.eval()
    predictions = []
    with torch.inference_mode():
        for inputs, geometry, _labels in val_loader:
            predictions.extend(model(inputs.to(device), geometry.to(device)).argmax(1).cpu().tolist())
    detailed = model_metrics(val_labels, predictions, val_defects)
    errors = (np.asarray(predictions) != val_labels)
    areas = val_geometry[:, 0] * val_geometry[:, 1]
    order = np.argsort(areas, kind="stable")
    groups = np.empty(len(areas), dtype=object)
    group_names = ("small", "medium", "large")
    for name, indices in zip(group_names, np.array_split(order, 3)):
        groups[indices] = name
    size_errors = {}
    rows = []
    for group_name in group_names:
        mask = groups == group_name
        group_detail = {"count": int(mask.sum()), "error_rate": float(errors[mask].mean()) if mask.any() else None}
        for defect_id, defect_name in enumerate(DETECTION_CLASSES):
            defect_mask = mask & (val_defects == defect_id)
            group_detail[defect_name] = {
                "count": int(defect_mask.sum()),
                "error_rate": float(errors[defect_mask].mean()) if defect_mask.any() else None,
            }
            rows.append({"size_group": group_name, "defect_class": defect_name,
                         "count": int(defect_mask.sum()),
                         "error_rate": group_detail[defect_name]["error_rate"]})
        size_errors[group_name] = group_detail
        rows.append({"size_group": group_name, "defect_class": "all",
                     "count": int(mask.sum()), "error_rate": group_detail["error_rate"]})
    pd.DataFrame(rows).to_csv(run_dir / "size_group_error_rates.csv", index=False)
    serialized = as_json(detailed, baseline)
    serialized.update({"run_id": run_id, "input_mode": "crop", "augment": False,
                       "best_epoch": best_epoch, "device": device.type,
                       "checkpoint": checkpoint_path.name, "size_group_error_rates": size_errors})
    with (run_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(serialized, handle, indent=2)
    save_confusion_matrix(detailed["overall"], run_dir)
    for defect_name, defect_metrics in detailed["by_defect"].items():
        save_confusion_matrix(defect_metrics, run_dir, stem=f"confusion_matrix_{defect_name}")
    append_experiment(run_id, "Added five normalized box geometry values to Run 1 CNN features",
                      False, "crop", detailed | {"baseline": baseline}, best_epoch, device, geometry=True)
    print_result(serialized)
    print("Size-group val error rates:")
    for name, values in size_errors.items():
        print(f"  {name}: {values['error_rate']:.4f} ({values['count']} boxes); "
              f"curb_ramp_issue={values['curb_ramp_issue']['error_rate']} "
              f"({values['curb_ramp_issue']['count']}), "
              f"surface_damage={values['surface_damage']['error_rate']} "
              f"({values['surface_damage']['count']})")
    print("Confusion matrix (rows=true, columns=predicted; low, medium, high):")
    print(np.asarray(detailed["overall"]["confusion_matrix"]))
    print("Test split was not used.")
    return serialized


def main() -> None:
    global IMAGE_SIZE
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run4", action="store_true", help="run only Run 4: no-augmentation crops resized to 384x384")
    parser.add_argument("--run5", action="store_true", help="run only Run 5: crop model with box geometry")
    parser.add_argument("--run6", action="store_true", help="run only Run 6: EfficientNet-B0 no-augmentation crops")
    parser.add_argument("--repeat-run1", action="store_true", help="repeat Run 1 settings under the selected seed")
    parser.add_argument("--seed", type=int, default=CONFIG_SEED, help="Python, NumPy, and PyTorch random seed")
    args = parser.parse_args()
    global SEED
    SEED = args.seed
    if args.run4:
        IMAGE_SIZE = (384, 384)

    os.environ.setdefault("TORCH_HOME", "/private/tmp/khota_torch_cache")
    torch.hub.set_dir(str(Path(os.environ["TORCH_HOME"]) / "hub"))
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    seed_everything()
    requested_device = os.environ.get("KHOTA_SEVERITY_DEVICE")
    default_device = "mps" if torch.backends.mps.is_available() else "cpu"
    device = torch.device(requested_device or default_device)
    if device.type not in {"cpu", "cuda", "mps"}:
        raise ValueError("KHOTA_SEVERITY_DEVICE must be cpu, cuda, or mps")
    print(f"Using {device}; seed={SEED}; max_epochs={MAX_EPOCHS}; patience={PATIENCE}")

    data = load_train_val()
    train_frame = data[data.split == "train"]
    val_frame = data[data.split == "val"]
    print(f"Defect boxes only: train={len(train_frame)}, val={len(val_frame)}")

    train_counts = train_frame.severity.value_counts()
    majority = max(SEVERITY_CLASSES, key=lambda name: int(train_counts.get(name, 0)))
    val_truth = [SEVERITY_TO_ID[name] for name in val_frame.severity]
    baseline_overall = severity_metrics(val_truth, [SEVERITY_TO_ID[majority]] * len(val_truth))
    baseline_by_defect = {}
    for defect_name in DETECTION_CLASSES:
        subset = val_frame[val_frame.defect_class == defect_name]
        truths = [SEVERITY_TO_ID[name] for name in subset.severity]
        baseline_by_defect[defect_name] = severity_metrics(truths, [SEVERITY_TO_ID[majority]] * len(truths))
    baseline = {
        "majority_class": majority,
        "overall": baseline_overall,
        "by_defect": baseline_by_defect,
    }
    print(f"Train majority baseline: {majority}; accuracy={baseline_overall['accuracy']:.4f}; macro_f1={baseline_overall['macro_f1']:.4f}")

    train_crop, train_labels, train_defects = prepare_examples(train_frame, "crop")
    val_crop, val_labels, val_defects = prepare_examples(val_frame, "crop")
    if args.repeat_run1:
        if SEED == CONFIG_SEED:
            raise ValueError("--repeat-run1 requires a seed other than the existing seed 42")
        run_experiment(
            f"severity_resnet18_crop_noaug_seed{SEED}",
            f"Repeat of Run 1 with random seed {SEED}",
            "crop", False,
            train_crop, train_labels, train_defects,
            val_crop, val_labels, val_defects, baseline, device,
            architecture="resnet18",
        )
        return
    if args.run5:
        train_geometry = prepare_geometry(train_frame)
        val_geometry = prepare_geometry(val_frame)
        if len(train_geometry) != len(train_labels) or len(val_geometry) != len(val_labels):
            raise RuntimeError("geometry rows are not aligned with crop examples")
        run_geometry_experiment(train_crop, train_labels, train_defects, train_geometry,
                                val_crop, val_labels, val_defects, val_geometry, baseline, device)
        return
    if args.run6:
        run_experiment(
            "severity_efficientnet_b0_crop_noaug",
            "Changed Run 1 backbone from ResNet-18 to EfficientNet-B0",
            "crop", False,
            train_crop, train_labels, train_defects,
            val_crop, val_labels, val_defects, baseline, device,
            architecture="efficientnet_b0",
        )
        return
    if args.run4:
        result = run_experiment(
            "severity_resnet18_crop_384_noaug",
            "Same as Run 1; resized the expanded crop to 384x384",
            "crop", False,
            train_crop, train_labels, train_defects,
            val_crop, val_labels, val_defects, baseline, device,
        )
        print("Test split was not used.")
        return

    run1 = run_experiment(
        "severity_resnet18_crop_noaug", "Initial pretrained ResNet-18 crop baseline", "crop", False,
        train_crop, train_labels, train_defects, val_crop, val_labels, val_defects, baseline, device,
    )
    run2 = run_experiment(
        "severity_resnet18_crop_aug", "Added allowed train-only augmentation", "crop", True,
        train_crop, train_labels, train_defects, val_crop, val_labels, val_defects, baseline, device,
    )

    selected = run2 if run2["overall"]["macro_f1"] > run1["overall"]["macro_f1"] else run1
    selected_augment = bool(selected["augment"])
    print(f"\nRun 3 inherits augmentation={selected_augment} from {selected['run_id']}; replacing crop input with full image.")
    train_full, full_train_labels, full_train_defects = prepare_examples(train_frame, "full_image")
    val_full, full_val_labels, full_val_defects = prepare_examples(val_frame, "full_image")
    run3 = run_experiment(
        "severity_resnet18_fullimage_best",
        f"Changed input to full image using best crop settings from {selected['run_id']}",
        "full_image", selected_augment,
        train_full, full_train_labels, full_train_defects,
        val_full, full_val_labels, full_val_defects, baseline, device,
    )

    print("\nRun comparison (val macro_f1; higher is better):")
    for result in (run1, run2, run3):
        print(f"  {result['run_id']}: {result['overall']['macro_f1']:.4f}")
    print(f"  baseline: {baseline_overall['macro_f1']:.4f}")
    print("Test split was not used.")


if __name__ == "__main__":
    main()
