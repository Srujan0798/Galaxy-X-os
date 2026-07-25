#!/usr/bin/env python3
"""
SCALE x ODYSSEY -- src/train.py
Complete Training Pipeline

Combines official starter guide (page 11-12) with our best practices:
- Progressive unfreezing (Phase 1: frozen backbone, Phase 2: full fine-tune)
- OneCycleLR scheduler
- Mixed precision (torch.amp)
- Weighted CrossEntropyLoss with label smoothing
- Early stopping + best checkpoint saving
- TensorBoard logging
"""

import sys
import time
from pathlib import Path

# Ensure sibling modules resolve when this file is loaded as ``src.train``
# (e.g. by the test runner) -- prepend this file's dir to sys.path.
_SRC_DIR = Path(__file__).resolve().parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR
from torch.amp import GradScaler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dataset import get_loaders  # noqa: E402
from model import AstroClassifier  # noqa: E402
from utils import load_config, load_class_weights, save_checkpoint, setup_logger, set_seed, compute_metrics, get_device, check_data_exists, get_autocast_context  # noqa: E402


def train_one_epoch(model, loader, optimizer, criterion, scaler, device, use_amp, scheduler=None):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []

    for images, labels in tqdm(loader, desc="Train", leave=False):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)

        if use_amp and scaler is not None and device.type in ("cuda", "mps"):
            with get_autocast_context(device.type):
                outputs = model(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    metrics = compute_metrics(all_labels, all_preds)
    metrics["loss"] = total_loss / total
    return metrics


def validate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Val  ", leave=False):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    metrics = compute_metrics(all_labels, all_preds)
    metrics["loss"] = total_loss / total
    return metrics


def setup_phase1_optimizer(model, lr, weight_decay):
    """Phase 1: Train only classifier head at 10x LR."""
    params = model.module.classifier.parameters() if hasattr(model, "module") else model.classifier.parameters()
    return optim.AdamW(params, lr=lr * 10, weight_decay=weight_decay)


def setup_phase2_optimizer(model, lr, weight_decay):
    """Phase 2: Full model with discriminative LR."""
    backbone_params = model.module.backbone.parameters() if hasattr(model, "module") else model.backbone.parameters()
    head_params = model.module.classifier.parameters() if hasattr(model, "module") else model.classifier.parameters()
    return optim.AdamW([
        {"params": backbone_params, "lr": lr / 10},
        {"params": head_params, "lr": lr},
    ], weight_decay=weight_decay)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="SCALE x ODYSSEY Training")
    parser.add_argument("--config", default="configs/config.yaml", help="Path to YAML config")
    parser.add_argument("--backbone", type=str, default=None, help="Override backbone (convnext_base, swin_base, efficientnet_b3)")
    parser.add_argument("--checkpoint", type=str, default=None, help="Override checkpoint save path")
    parser.add_argument("--epochs", type=int, default=None, help="Override number of epochs")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    parser.add_argument("--label-smoothing", type=float, default=None, help="Override label smoothing")
    parser.add_argument("--focal-gamma", type=float, default=None, help="Override focal loss gamma")
    parser.add_argument("--device", default=None, choices=["cuda", "mps", "cpu"], help="Force device")
    parser.add_argument("--seed", type=int, default=None, help="Override random seed")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.backbone is not None:
        config["model"]["backbone"] = args.backbone
    if args.checkpoint is not None:
        config["eval_checkpoint"] = args.checkpoint
    if args.epochs is not None:
        config["training"]["num_epochs"] = args.epochs
    if args.lr is not None:
        config["training"]["lr"] = args.lr
    if args.batch_size is not None:
        config["training"]["batch_size"] = args.batch_size
    if args.label_smoothing is not None:
        config["training"]["label_smoothing"] = args.label_smoothing
    if args.focal_gamma is not None:
        config["training"]["focal_gamma"] = args.focal_gamma
    if args.seed is not None:
        config["seed"] = args.seed

    set_seed(config.get("seed", 42))
    device = torch.device(args.device) if args.device else get_device()

    data_dir = config["data"]["processed_dir"]
    check_data_exists(data_dir)
    batch_size = config["training"].get("batch_size", 32)
    num_epochs = config["training"]["num_epochs"]
    lr = config["training"]["lr"]
    weight_decay = config["training"]["weight_decay"]
    label_smoothing = config["training"]["label_smoothing"]
    patience = config["training"]["patience"]
    use_amp = config["training"].get("mixed_precision", True)
    backbone = config["model"]["backbone"]
    num_classes = config["model"]["num_classes"]
    dropout = config["model"]["dropout"]
    image_size = config["data"]["image_size"]
    num_workers = config["data"].get("num_workers", 4)
    freeze_epochs = config["training"].get("freeze_backbone_epochs", 3)
    checkpoint_dir = Path(config["paths"]["checkpoints"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logger(config["paths"]["logs"], f"{backbone}_{int(time.time())}")

    logger.info("=" * 70)
    logger.info("SCALE x ODYSSEY -- Training Started")
    logger.info(f"Device: {device} | Backbone: {backbone} | Epochs: {num_epochs}")
    logger.info(f"Batch: {batch_size} | LR: {lr} | AMP: {use_amp}")
    logger.info("=" * 70)

    # Data
    train_loader, val_loader, test_loader = get_loaders(
        data_dir, batch_size, num_workers, image_size
    )

    # Model
    model = AstroClassifier(num_classes, backbone, pretrained=True, dropout=dropout).to(device)
    if torch.cuda.device_count() > 1:
        logger.info(f"Using {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)

    # Loss + weights
    class_weights = load_class_weights(data_dir, device)
    # Loss: Focal Loss + Label Smoothing
    focal_gamma = config["training"].get("focal_gamma", 2.0)
    from utils import FocalLoss, LabelSmoothingCrossEntropy
    if focal_gamma > 0:
        criterion = FocalLoss(gamma=focal_gamma, alpha=0.25)
        logger.info(f"Using Focal Loss (γ={focal_gamma})")
    else:
        criterion = LabelSmoothingCrossEntropy(smoothing=label_smoothing)
        logger.info(f"Using Label Smoothing CrossEntropy (ε={label_smoothing})")

    # Optimizer (will be recreated during unfreezing)
    optimizer = setup_phase1_optimizer(model, lr, weight_decay)

    # Scheduler
    steps_per_epoch = len(train_loader)
    scheduler = OneCycleLR(
        optimizer, max_lr=lr * 10, epochs=freeze_epochs + 1,
        steps_per_epoch=steps_per_epoch, pct_start=0.3,
        anneal_strategy="cos", div_factor=25.0, final_div_factor=1e4,
    )

    scaler = GradScaler(device.type) if (use_amp and device.type in ("cuda", "mps")) else None
    tb_writer = SummaryWriter(log_dir=config["paths"]["logs"])

    best_val_acc, patience_ctr = 0.0, 0
    current_phase = 1

    for epoch in range(num_epochs):
        epoch_start = time.time()

        # Progressive unfreezing
        if epoch == 0:
            if hasattr(model, "module"):
                model.module.freeze_backbone()
            else:
                model.freeze_backbone()
            logger.info(f"[Epoch {epoch+1}] Phase 1: Backbone FROZEN | Head LR = {lr*10:.2e}")

        elif epoch == freeze_epochs and current_phase == 1:
            current_phase = 2
            if hasattr(model, "module"):
                model.module.unfreeze_backbone()
            else:
                model.unfreeze_backbone()
            optimizer = setup_phase2_optimizer(model, lr, weight_decay)
            remaining_epochs = num_epochs - freeze_epochs
            scheduler = OneCycleLR(
                optimizer, max_lr=[lr / 10, lr], epochs=remaining_epochs,
                steps_per_epoch=steps_per_epoch, pct_start=0.2,
                anneal_strategy="cos", div_factor=25.0, final_div_factor=1e4,
            )
            logger.info(f"[Epoch {epoch+1}] Phase 2: Full fine-tuning | Backbone LR = {lr/10:.2e}, Head LR = {lr:.2e}")

        # Train & validate
        train_m = train_one_epoch(model, train_loader, optimizer, criterion, scaler, device, use_amp, scheduler)
        val_m = validate(model, val_loader, criterion, device)

        epoch_time = time.time() - epoch_start
        lr_now = optimizer.param_groups[0]["lr"]

        logger.info(
            f"[Epoch {epoch+1:02d}/{num_epochs}] {epoch_time:.1f}s | LR: {lr_now:.2e} | "
            f"Train Loss: {train_m['loss']:.4f} Acc: {train_m['accuracy']:.4f} | "
            f"Val Loss: {val_m['loss']:.4f} Acc: {val_m['accuracy']:.4f} F1: {val_m['macro_f1']:.4f}"
        )

        # TensorBoard
        tb_writer.add_scalars("Loss", {"train": train_m["loss"], "val": val_m["loss"]}, epoch)
        tb_writer.add_scalars("Accuracy", {"train": train_m["accuracy"], "val": val_m["accuracy"]}, epoch)
        tb_writer.add_scalar("F1/val_macro", val_m["macro_f1"], epoch)
        tb_writer.add_scalar("LR", lr_now, epoch)

        # Checkpoint
        if val_m["accuracy"] > best_val_acc:
            best_val_acc = val_m["accuracy"]
            patience_ctr = 0
            save_checkpoint(model, optimizer, scheduler, epoch, best_val_acc, config,
                           str(checkpoint_dir / "best_model.pth"))
            logger.info(f"  *** New best model saved! (val_acc={best_val_acc:.4f}) ***")
        else:
            patience_ctr += 1
            if (epoch + 1) % 5 == 0:
                save_checkpoint(model, optimizer, scheduler, epoch, best_val_acc, config,
                               str(checkpoint_dir / f"epoch_{epoch+1:03d}.pth"))

        if patience_ctr >= patience:
            logger.info(f"Early stopping triggered! No improvement for {patience} epochs.")
            break

    tb_writer.close()
    logger.info("=" * 70)
    logger.info(f"Training Complete! Best Val Accuracy: {best_val_acc:.4f}")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
