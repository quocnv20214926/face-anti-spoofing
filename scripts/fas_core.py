"""Shared model, preprocessing, checkpoint and metric utilities."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from sklearn.metrics import confusion_matrix
from torch import nn
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small
from torchvision.transforms import v2

IMAGE_SIZE = 224
MODEL_NAME = "mobilenet_v3_small"
DEFAULT_CLASSES = ["real", "spoof"]


def build_model(num_classes: int = 2, pretrained: bool = True) -> nn.Module:
    weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
    model = mobilenet_v3_small(weights=weights)
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
    return model


def transforms(train: bool):
    ops: list = [v2.ToImage()]
    if train:
        ops += [
            v2.RandomResizedCrop(IMAGE_SIZE, scale=(0.72, 1.0), ratio=(0.85, 1.15)),
            v2.RandomHorizontalFlip(),
            v2.RandomApply([v2.ColorJitter(0.25, 0.25, 0.2, 0.06)], p=0.7),
            v2.RandomGrayscale(p=0.05),
            v2.RandomApply([v2.GaussianBlur(3, sigma=(0.1, 1.2))], p=0.2),
        ]
    else:
        ops += [v2.Resize((IMAGE_SIZE, IMAGE_SIZE), antialias=True)]
    ops += [
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ]
    return v2.Compose(ops)


def save_checkpoint(path: Path, model: nn.Module, class_to_idx: dict[str, int], threshold: float, epoch: int, metrics: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "format_version": 2,
        "architecture": MODEL_NAME,
        "image_size": IMAGE_SIZE,
        "class_to_idx": class_to_idx,
        "real_threshold": float(threshold),
        "epoch": epoch,
        "metrics": metrics,
        "state_dict": model.state_dict(),
    }, path)


def load_checkpoint(path: Path, device: torch.device):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    classes = checkpoint["class_to_idx"]
    model = build_model(len(classes), pretrained=False)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device).eval()
    return model, checkpoint


def find_best_threshold(labels: Iterable[int], real_probs: Iterable[float], real_idx: int) -> tuple[float, dict]:
    y = np.asarray(list(labels))
    p = np.asarray(list(real_probs))
    candidates = np.linspace(0.05, 0.95, 181)
    best = (float("inf"), 0.5, {})
    for threshold in candidates:
        pred_real = p >= threshold
        true_real = y == real_idx
        apcer = float(np.mean(pred_real[~true_real])) if np.any(~true_real) else 0.0
        bpcer = float(np.mean(~pred_real[true_real])) if np.any(true_real) else 0.0
        acer = (apcer + bpcer) / 2
        if acer < best[0]:
            best = (acer, float(threshold), {"acer": acer, "apcer": apcer, "bpcer": bpcer})
    return best[1], best[2]


def binary_metrics(labels: Iterable[int], real_probs: Iterable[float], real_idx: int, threshold: float) -> dict:
    y = np.asarray(list(labels)) == real_idx
    pred = np.asarray(list(real_probs)) >= threshold
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[False, True]).ravel()
    apcer = fp / max(1, fp + tn)  # attack accepted as real
    bpcer = fn / max(1, fn + tp)  # bona fide rejected as spoof
    return {"accuracy": float((pred == y).mean()), "apcer": float(apcer), "bpcer": float(bpcer), "acer": float((apcer + bpcer) / 2)}
