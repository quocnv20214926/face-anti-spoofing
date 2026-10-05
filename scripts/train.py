"""Fast transfer-learning pipeline for binary face anti-spoofing."""
from __future__ import annotations

import argparse
import json
import random
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision.datasets import ImageFolder
from tqdm import tqdm

from fas_core import binary_metrics, build_model, find_best_threshold, save_checkpoint, transforms


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, default=Path(__file__).parents[1] / "Data")
    p.add_argument("--output", type=Path, default=Path(__file__).parents[1] / "artifacts" / "fas_mobilenetv3.pt")
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--freeze-epochs", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--workers", type=int, default=0 if __import__("os").name == "nt" else 4)
    p.add_argument("--patience", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


@torch.inference_mode()
def infer(model, loader, device, real_idx):
    labels, probs, loss_sum = [], [], 0.0
    criterion = nn.CrossEntropyLoss()
    model.eval()
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss_sum += criterion(logits, y).item() * y.numel()
        labels.extend(y.cpu().tolist())
        probs.extend(logits.softmax(1)[:, real_idx].cpu().tolist())
    return labels, probs, loss_sum / len(loader.dataset)


def main():
    args = parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    # Existing project convention: development is validation, evaluation is held-out test.
    dirs = {"train": args.data / "training", "val": args.data / "development", "test": args.data / "evaluation"}
    for name, path in dirs.items():
        if not path.is_dir():
            raise FileNotFoundError(f"Missing {name} split: {path}")
    datasets = {
        "train": ImageFolder(dirs["train"], transforms(True)),
        "val": ImageFolder(dirs["val"], transforms(False)),
        "test": ImageFolder(dirs["test"], transforms(False)),
    }
    if datasets["train"].classes != datasets["val"].classes or datasets["train"].classes != datasets["test"].classes:
        raise ValueError("All splits must have identical class folders")
    if "real" not in datasets["train"].class_to_idx:
        raise ValueError("A class folder named 'real' is required")
    pin = device.type == "cuda"
    train_targets = np.asarray(datasets["train"].targets)
    class_counts = np.bincount(train_targets, minlength=len(datasets["train"].classes))
    sample_weights = (1.0 / np.maximum(class_counts, 1))[train_targets]
    sampler = WeightedRandomSampler(torch.as_tensor(sample_weights, dtype=torch.double), len(sample_weights), replacement=True)
    loaders = {
        name: DataLoader(
            ds,
            batch_size=args.batch_size,
            shuffle=False,
            sampler=sampler if name == "train" else None,
            num_workers=args.workers,
            pin_memory=pin,
            persistent_workers=args.workers > 0,
        )
        for name, ds in datasets.items()
    }

    model = build_model(len(datasets["train"].classes), pretrained=True).to(device)
    real_idx = datasets["train"].class_to_idx["real"]
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    scaler = torch.amp.GradScaler("cuda", enabled=pin)
    best_acer, stale = float("inf"), 0
    optimizer = None
    print(f"device={device} classes={datasets['train'].class_to_idx} sizes=" + str({k: len(v) for k, v in datasets.items()}))

    for epoch in range(args.epochs):
        frozen = epoch < args.freeze_epochs
        for parameter in model.features.parameters():
            parameter.requires_grad = not frozen
        trainable = [p for p in model.parameters() if p.requires_grad]
        # Recreate only when the backbone is unfrozen; otherwise preserve AdamW state.
        if optimizer is None or epoch == args.freeze_epochs:
            optimizer = torch.optim.AdamW(trainable, lr=args.lr if not frozen else args.lr * 2, weight_decay=1e-4)
        model.train()
        total_loss = 0.0
        for x, y in tqdm(loaders["train"], desc=f"epoch {epoch + 1}/{args.epochs}"):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            amp = torch.amp.autocast("cuda") if pin else nullcontext()
            with amp:
                loss = criterion(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(optimizer); scaler.update()
            total_loss += loss.item() * y.numel()

        labels, probs, val_loss = infer(model, loaders["val"], device, real_idx)
        threshold, metrics = find_best_threshold(labels, probs, real_idx)
        print(json.dumps({"epoch": epoch + 1, "train_loss": total_loss / len(datasets["train"]), "val_loss": val_loss,
                          "threshold": threshold, **metrics}, indent=2))
        if metrics["acer"] < best_acer:
            best_acer, stale = metrics["acer"], 0
            save_checkpoint(args.output, model, datasets["train"].class_to_idx, threshold, epoch + 1, metrics)
        else:
            stale += 1
            if stale >= args.patience:
                print("Early stopping")
                break

    from fas_core import load_checkpoint
    model, checkpoint = load_checkpoint(args.output, device)
    labels, probs, _ = infer(model, loaders["test"], device, real_idx)
    result = binary_metrics(labels, probs, real_idx, checkpoint["real_threshold"])
    print("held-out test:", json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
