"""Severity inference for the pipeline (AGENTS.md section 1, severity/AGENTS.md).

    model = load_severity_model("experiments/<run_id>/best_model.pt")
    results = predict_severity(model, crops)   # [(label, p_high), ...]

`path` may also be a list of checkpoints, or an `ensemble.json` file listing them ({"checkpoints": [...],
"threshold": t}); p_high is then the mean over the models.

`crops` are the outputs of `khota_ml.crop.crop_box` (PIL images or RGB uint8 arrays) cut from the full-resolution
photo. Preprocessing here is the same function training uses, so training and the pipeline cannot drift apart.
"""

from __future__ import annotations

import json
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

IMAGE_SIZE = (224, 224)  # default; each checkpoint stores its own input_size
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
DEFAULT_THRESHOLD = 0.5
HIGH_INDEX = SEVERITY_CLASSES.index(HIGH)
LOW_INDEX = 1 - HIGH_INDEX
assert len(SEVERITY_CLASSES) == 2, "predict.py assumes binary severity (version 2)"


def build_model(architecture: str, pretrained: bool = False) -> nn.Module:
    """Backbone with a two-class head in config.SEVERITY_CLASSES order."""
    n = len(SEVERITY_CLASSES)
    if architecture == "resnet18":
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
        model.fc = nn.Linear(model.fc.in_features, n)
    elif architecture == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT if pretrained else None)
        model.fc = nn.Linear(model.fc.in_features, n)
    elif architecture == "efficientnet_b0":
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT if pretrained else None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, n)
    elif architecture == "convnext_tiny":
        model = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None)
        model.classifier[2] = nn.Linear(model.classifier[2].in_features, n)
    else:
        raise ValueError(f"unsupported architecture: {architecture}")
    return model


def resize_crop(crop, size=IMAGE_SIZE) -> np.ndarray:
    """A crop (PIL or array) -> RGB uint8 array of `size` (bilinear), as used in training."""
    image = crop if isinstance(crop, Image.Image) else Image.fromarray(np.asarray(crop))
    image = image.convert("RGB").resize(tuple(size), Image.Resampling.BILINEAR)
    return np.array(image, dtype=np.uint8)


def to_tensor(array: np.ndarray) -> torch.Tensor:
    """Resized RGB uint8 array -> normalised CHW float tensor."""
    tensor = torch.from_numpy(np.ascontiguousarray(array)).permute(2, 0, 1).float().div(255.0)
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    return (tensor - mean) / std


def p_high_from_logits(logits: torch.Tensor) -> torch.Tensor:
    return torch.softmax(logits.float(), dim=1)[:, HIGH_INDEX]


@dataclass
class _Member:
    net: nn.Module
    input_size: tuple


@dataclass
class SeverityModel:
    members: list
    threshold: float
    device: torch.device
    run_id: str


def _load_member(path, dev) -> tuple[_Member, dict]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if list(checkpoint["severity_classes"]) != list(SEVERITY_CLASSES):
        raise ValueError(f"checkpoint classes {checkpoint['severity_classes']} != config {SEVERITY_CLASSES}")
    net = build_model(checkpoint["architecture"], pretrained=False)
    net.load_state_dict(checkpoint["model_state_dict"])
    net.to(dev).eval()
    return _Member(net, tuple(checkpoint.get("input_size", IMAGE_SIZE))), checkpoint


def load_severity_model(path, device: str | None = None) -> SeverityModel:
    """Load one checkpoint, a list of checkpoints, or an ensemble.json, with the stored `high` threshold."""
    dev = torch.device(device or "cpu")
    if isinstance(path, (list, tuple)):
        paths, threshold, run_id = [Path(p) for p in path], DEFAULT_THRESHOLD, "ensemble"
    elif str(path).endswith(".json"):
        spec = json.loads(Path(path).read_text())
        paths = [ROOT / p for p in spec["checkpoints"]]
        threshold, run_id = float(spec.get("threshold", DEFAULT_THRESHOLD)), str(spec.get("run_id", "ensemble"))
    else:
        member, checkpoint = _load_member(path, dev)
        return SeverityModel([member], float(checkpoint.get("threshold", DEFAULT_THRESHOLD)), dev,
                             str(checkpoint.get("run_id", "")))
    members = [_load_member(p, dev)[0] for p in paths]
    return SeverityModel(members, threshold, dev, run_id)


def predict_p_high(model: SeverityModel, crops, batch_size: int = 32) -> np.ndarray:
    crops = list(crops)
    total = np.zeros(len(crops), dtype=np.float64)
    with torch.inference_mode():
        for member in model.members:
            for start in range(0, len(crops), batch_size):
                batch = torch.stack([to_tensor(resize_crop(c, member.input_size)) for c in crops[start:start + batch_size]])
                total[start:start + len(batch)] += p_high_from_logits(member.net(batch.to(model.device))).cpu().numpy()
    return total / len(model.members)


def predict_severity(model: SeverityModel, crops, batch_size: int = 32) -> list[tuple[str, float]]:
    """Return (label, p_high) per crop; label is `high` when p_high >= the stored threshold."""
    p_high = predict_p_high(model, crops, batch_size)
    return [(SEVERITY_CLASSES[HIGH_INDEX] if p >= model.threshold else SEVERITY_CLASSES[LOW_INDEX], float(p))
            for p in p_high]
