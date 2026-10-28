"""Shared evaluation metrics for severity classification and detection."""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from .config import SEVERITY_CLASSES


def severity_metrics(y_true, y_pred) -> dict:
    """Return accuracy, macro_f1, per-class F1, and confusion matrix.

    Inputs may be severity names or integer class IDs in config order.
    The confusion matrix uses rows=true and columns=predicted, in config order.
    """
    names = SEVERITY_CLASSES
    to_id = {name: index for index, name in enumerate(names)}

    def encode(values):
        encoded = []
        for value in values:
            if isinstance(value, str):
                if value not in to_id:
                    raise ValueError(f"unknown severity class: {value}")
                encoded.append(to_id[value])
            else:
                index = int(value)
                if index < 0 or index >= len(names):
                    raise ValueError(f"severity class id out of range: {value}")
                encoded.append(index)
        return np.asarray(encoded, dtype=int)

    truth, pred = encode(y_true), encode(y_pred)
    if truth.size != pred.size:
        raise ValueError("y_true and y_pred must have equal length")
    if not truth.size:
        raise ValueError("metrics require at least one example")
    matrix = confusion_matrix(truth, pred, labels=range(len(names)))
    per_class = f1_score(truth, pred, labels=range(len(names)), average=None, zero_division=0)
    return {
        "accuracy": float(accuracy_score(truth, pred)),
        "macro_f1": float(f1_score(truth, pred, labels=range(len(names)), average="macro", zero_division=0)),
        "f1_per_class": {name: float(value) for name, value in zip(names, per_class)},
        "confusion_matrix": matrix,
        "class_names": list(names),
    }


def save_confusion_matrix(metrics: dict, output_dir, stem: str = "confusion_matrix") -> tuple:
    """Save a ``severity_metrics`` result's confusion matrix as CSV and PNG."""
    from pathlib import Path

    names = list(metrics["class_names"])
    matrix = np.asarray(metrics["confusion_matrix"], dtype=int)
    if matrix.shape != (len(names), len(names)):
        raise ValueError("confusion matrix shape does not match class_names")
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    csv_path = destination / f"{stem}.csv"
    png_path = destination / f"{stem}.png"
    pd.DataFrame(matrix, index=names, columns=names).rename_axis("true\\predicted").to_csv(csv_path)

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.imshow(matrix, cmap="Blues")
    ax.set_xticks(range(len(names)), labels=names, rotation=30, ha="right")
    ax.set_yticks(range(len(names)), labels=names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion matrix")
    threshold = matrix.max() / 2 if matrix.size else 0
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            ax.text(col, row, str(matrix[row, col]), ha="center", va="center",
                    color="white" if matrix[row, col] > threshold else "black")
    fig.tight_layout()
    fig.savefig(png_path, dpi=160)
    plt.close(fig)
    return csv_path, png_path
