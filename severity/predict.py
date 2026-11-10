"""Severity inference for the pipeline (AGENTS.md section 1, severity/AGENTS.md).

    model = load_severity_model("experiments/<run_id>/best_model.pt")
    results = predict_severity(model, crops)   # [(label, p_high), ...]

`crops` are the outputs of `khota_ml.crop.crop_box` (PIL images or RGB uint8 arrays) cut from the full-resolution
photo. Preprocessing here is the same function training uses, so training and the pipeline cannot drift apart.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torchvision import models

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from khota_ml.config import HIGH, SEVERITY_CLASSES  # noqa: E402

IMAGE_SIZE = (224, 224)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
DEFAULT_THRESHOLD = 0.5
HIGH_INDEX = SEVERITY_CLASSES.index(HIGH)
LOW_INDEX = 1 - HIGH_INDEX
assert len(SEVERITY_CLASSES) == 2, "predict.py assumes binary severity (version 2)"


def build_model(architecture: str, pretrained: bool = False) -> nn.Module:
    """ResNet-18 or EfficientNet-B0 with a two-class head in config.SEVERITY_CLASSES order."""
    n = len(SEVERITY_CLASSES)
    if architecture == "resnet18":
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
        model.fc = nn.Linear(model.fc.in_features, n)
    elif architecture == "efficientnet_b0":
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT if pretrained else None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, n)
    else:
        raise ValueError(f"unsupported architecture: {architecture}")
    return model


def resize_crop(crop) -> np.ndarray:
    """A crop (PIL or array) -> RGB uint8 array of IMAGE_SIZE (bilinear), as used in training."""
    image = crop if isinstance(crop, Image.Image) else Image.fromarray(np.asarray(crop))
    image = image.convert("RGB").resize(IMAGE_SIZE, Image.Resampling.BILINEAR)
    return np.asarray(image, dtype=np.uint8).copy()


def to_tensor(array: np.ndarray) -> torch.Tensor:
    """Resized RGB uint8 array -> normalised CHW float tensor."""
    tensor = torch.from_numpy(array).permute(2, 0, 1).float().div(255.0)
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    return (tensor - mean) / std


@dataclass
class SeverityModel:
    net: nn.Module
    threshold: float
    device: torch.device
    run_id: str


def load_severity_model(path, device: str | None = None) -> SeverityModel:
    """Load a checkpoint written by severity/train.py, with its stored `high` threshold (default 0.5)."""
    dev = torch.device(device or "cpu")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if list(checkpoint["severity_classes"]) != list(SEVERITY_CLASSES):
        raise ValueError(f"checkpoint classes {checkpoint['severity_classes']} != config {SEVERITY_CLASSES}")
    net = build_model(checkpoint["architecture"], pretrained=False)
    net.load_state_dict(checkpoint["model_state_dict"])
    net.to(dev).eval()
    threshold = float(checkpoint.get("threshold", DEFAULT_THRESHOLD))
    return SeverityModel(net=net, threshold=threshold, device=dev, run_id=str(checkpoint.get("run_id", "")))


def p_high_from_logits(logits: torch.Tensor) -> torch.Tensor:
    return torch.softmax(logits.float(), dim=1)[:, HIGH_INDEX]


def predict_severity(model: SeverityModel, crops, batch_size: int = 32) -> list[tuple[str, float]]:
    """Return (label, p_high) per crop; label is `high` when p_high >= the stored threshold."""
    results: list[tuple[str, float]] = []
    crops = list(crops)
    with torch.inference_mode():
        for start in range(0, len(crops), batch_size):
            batch = torch.stack([to_tensor(resize_crop(c)) for c in crops[start:start + batch_size]])
            p_high = p_high_from_logits(model.net(batch.to(model.device))).cpu().tolist()
            for p in p_high:
                label = SEVERITY_CLASSES[HIGH_INDEX] if p >= model.threshold else SEVERITY_CLASSES[LOW_INDEX]
                results.append((label, float(p)))
    return results
