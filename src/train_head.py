#!/usr/bin/env python3
"""
SCALE x ODYSSEY -- src/train_head.py
Memory-safe head-only training for low-RAM machines (e.g. 8 GB Apple Silicon).

Full backbone fine-tuning of EfficientNet-B3 needs >12 GB and thrashes swap on
8 GB machines. This script keeps the pretrained backbone FROZEN and runs it under
torch.no_grad() (no autograd graph retained), training only the classifier head.
Result: ~2 GB peak RAM, ~100 s/epoch on MPS, no swap death.

Saves the best checkpoint (by val accuracy) in the same format load_checkpoint()
expects, so src/evaluate.py can load it directly.
"""

import time
import sys
from pathlib import Path

_SRC_DIR = Path(__file__).resolve().parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from dataset import get_loaders  # noqa: E402
from model import AstroClassifier  # noqa: E402
from utils import load_config, setup_logger, set_seed, compute_metrics, get_device  # noqa: E402


def run_epoch(model, loader, criterion, device, optimizer=None):
    train = optimizer is not None
    model.classifier.train(train)
    model.backbone.eval()  # keep pretrained BatchNorm stats frozen

    total_loss, total = 0.0, 0
    all_preds, all_labels = [], []

    for images, labels in tqdm(loader, desc="Train" if train else "Val  ", leave=False):
        images, labels = images.to(device), labels.to(device)

        # Frozen backbone: no grad, no graph -> minimal memory
        with torch.no_grad():
            feats = model.backbone.forward_features(images).detach()

        with torch.set_grad_enabled(train):
            outputs = model.classifier(feats)
            loss = criterion(outputs, labels)

        if train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.classifier.parameters(), 1.0)
            optimizer.step()

        total_loss += loss.item() * images.size(0)
        total += labels.size(0)
        all_preds.extend(outputs.argmax(1).cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    metrics = compute_metrics(all_labels, all_preds)
    metrics["loss"] = total_loss / total
    return metrics


def main():
    config = load_config()
    set_seed(config.get("seed", 42))

    data_dir = config["data"]["processed_dir"]
    image_size = config["data"]["image_size"]
    label_smoothing = config["training"].get("label_smoothing", 0.1)
    weight_decay = config["training"].get("weight_decay", 1e-4)
    backbone = config["model"]["backbone"]
    num_classes = config["model"]["num_classes"]
    dropout = config["model"]["dropout"]

    # Memory-safe overrides for 8 GB machines
    epochs = 25
    batch_size = 16
    num_workers = 0
    head_lr = 1e-3

    checkpoint_dir = Path(config["paths"]["checkpoints"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    device = get_device()
    logger = setup_logger(config["paths"]["logs"], f"head_{backbone}_{int(time.time())}")

    logger.info("=" * 70)
    logger.info("SCALE x ODYSSEY -- Head-Only Training (memory-safe)")
    logger.info(f"Device: {device} | Backbone: {backbone} (FROZEN) | Epochs: {epochs}")
    logger.info(f"Batch: {batch_size} | Head LR: {head_lr} | Workers: {num_workers}")
    logger.info("=" * 70)

    train_loader, val_loader, _ = get_loaders(data_dir, batch_size, num_workers, image_size)

    model = AstroClassifier(num_classes, backbone, pretrained=True, dropout=dropout).to(device)
    model.freeze_backbone()

    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    optimizer = optim.AdamW(model.classifier.parameters(), lr=head_lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_acc = 0.0
    for epoch in range(epochs):
        t0 = time.time()
        train_m = run_epoch(model, train_loader, criterion, device, optimizer)
        val_m = run_epoch(model, val_loader, criterion, device)
        scheduler.step()

        logger.info(
            f"[Epoch {epoch+1:02d}/{epochs}] {time.time()-t0:.1f}s | "
            f"LR: {optimizer.param_groups[0]['lr']:.2e} | "
            f"Train Loss: {train_m['loss']:.4f} Acc: {train_m['accuracy']:.4f} | "
            f"Val Loss: {val_m['loss']:.4f} Acc: {val_m['accuracy']:.4f} F1: {val_m['macro_f1']:.4f}"
        )

        if val_m["accuracy"] > best_val_acc:
            best_val_acc = val_m["accuracy"]
            torch.save(
                {"model_state_dict": model.state_dict(), "best_val_acc": best_val_acc,
                 "epoch": epoch, "config": config},
                str(checkpoint_dir / "best_head.pth"),
            )
            logger.info(f"  *** New best head saved! (val_acc={best_val_acc:.4f}) ***")

    logger.info("=" * 70)
    logger.info(f"Head training complete! Best Val Accuracy: {best_val_acc:.4f}")
    logger.info(f"Saved to {checkpoint_dir / 'best_head.pth'}")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
