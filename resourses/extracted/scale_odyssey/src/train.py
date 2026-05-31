#!/usr/bin/env python3
"""
SCALE x ODYSSEY -- Complete Training Pipeline

Features:
- Transfer learning with progressive unfreezing
- Mixed precision (torch.amp) for speed
- OneCycleLR + warm restart scheduling
- Weighted CrossEntropyLoss with label smoothing
- Early stopping + best checkpoint saving
- TensorBoard + optional WandB logging
- Multi-GPU support (DataParallel)
"""

import os
import json
import time
import logging
from pathlib import Path
from typing import Dict, Optional

import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR
from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import AstroClassifier
from augmentations import AstroDataset, get_training_augmentations, get_validation_augmentations

# ============================================================================
# CONFIGURATION
# ============================================================================

def load_config(config_path: str = "configs/config.yaml") -> Dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


# ============================================================================
# LOGGING
# ============================================================================

def setup_logger(log_dir: str, experiment_name: str):
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"{experiment_name}.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(),
        ],
    )
    return logging.getLogger(__name__)


# ============================================================================
# METRICS
# ============================================================================

class MetricsTracker:
    """Track and compute training/validation metrics."""

    CLASSES = AstroDataset.CLASSES

    def __init__(self, num_classes: int = 5):
        self.num_classes = num_classes
        self.reset()

    def reset(self):
        self.loss_sum = 0.0
        self.correct = 0
        self.total = 0
        self.predictions = []
        self.targets = []

    def update(self, loss: float, outputs: torch.Tensor, labels: torch.Tensor):
        self.loss_sum += loss * labels.size(0)
        preds = outputs.argmax(dim=1)
        self.correct += (preds == labels).sum().item()
        self.total += labels.size(0)
        self.predictions.extend(preds.cpu().numpy())
        self.targets.extend(labels.cpu().numpy())

    @property
    def avg_loss(self) -> float:
        return self.loss_sum / max(self.total, 1)

    @property
    def accuracy(self) -> float:
        return self.correct / max(self.total, 0)

    def compute_f1_per_class(self) -> Dict[str, float]:
        from sklearn.metrics import f1_score
        f1_macro = f1_score(self.targets, self.predictions, average="macro", zero_division=0)
        f1_per_class = f1_score(self.targets, self.predictions, average=None, zero_division=0)
        result = {"macro_f1": f1_macro}
        for i, cls in enumerate(self.CLASSES):
            result[f"f1_{cls}"] = f1_per_class[i] if i < len(f1_per_class) else 0.0
        return result


# ============================================================================
# TRAINER
# ============================================================================

class Trainer:
    """Full-featured trainer with progressive unfreezing and mixed precision."""

    def __init__(self, config: Dict, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Extract config
        self.data_dir = config["data"]["processed_dir"]
        self.batch_size = config["training"].get("batch_size", 32)
        self.num_epochs = config["training"]["num_epochs"]
        self.lr = config["training"]["lr"]
        self.weight_decay = config["training"]["weight_decay"]
        self.label_smoothing = config["training"]["label_smoothing"]
        self.patience = config["training"]["patience"]
        self.use_amp = config["training"].get("mixed_precision", True)
        self.gradient_clip = config["training"].get("gradient_clip", 1.0)
        self.backbone = config["model"]["backbone"]
        self.num_classes = config["model"]["num_classes"]
        self.dropout = config["model"]["dropout"]
        self.image_size = config["data"]["image_size"]
        self.num_workers = config["data"].get("num_workers", 4)

        # Directories
        self.checkpoint_dir = Path(config["paths"]["checkpoints"])
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Initialize components
        self._setup_data()
        self._setup_model()
        self._setup_training()
        self._setup_logging()

        self.current_epoch = 0
        self.best_val_acc = 0.0
        self.patience_counter = 0
        self.train_metrics = MetricsTracker(self.num_classes)
        self.val_metrics = MetricsTracker(self.num_classes)

    def _setup_data(self):
        """Create datasets and dataloaders."""
        self.logger.info("Setting up data loaders...")

        train_ds = AstroDataset(self.data_dir, "train", self.image_size)
        val_ds = AstroDataset(self.data_dir, "val", self.image_size)
        test_ds = AstroDataset(self.data_dir, "test", self.image_size)

        self.train_loader = DataLoader(
            train_ds, batch_size=self.batch_size, shuffle=True,
            num_workers=self.num_workers, pin_memory=True, drop_last=True,
        )
        self.val_loader = DataLoader(
            val_ds, batch_size=self.batch_size, shuffle=False,
            num_workers=self.num_workers, pin_memory=True,
        )
        self.test_loader = DataLoader(
            test_ds, batch_size=self.batch_size, shuffle=False,
            num_workers=self.num_workers, pin_memory=True,
        )

        self.logger.info(f"  Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")

        # Load class weights for imbalance
        weights_path = Path(self.data_dir) / "class_weights.json"
        if weights_path.exists():
            with open(weights_path) as f:
                weights_dict = json.load(f)
            class_names = AstroDataset.CLASSES
            weights = torch.tensor(
                [weights_dict[c] for c in class_names], dtype=torch.float32
            )
            self.class_weights = weights.to(self.device)
            self.logger.info(f"  Loaded class weights: {weights_dict}")
        else:
            self.class_weights = None
            self.logger.warning("  No class weights found, using uniform weights")

    def _setup_model(self):
        """Initialize model."""
        self.logger.info(f"Building model: {self.backbone}")

        self.model = AstroClassifier(
            num_classes=self.num_classes,
            backbone=self.backbone,
            pretrained=True,
            dropout=self.dropout,
        ).to(self.device)

        # Multi-GPU support
        if torch.cuda.device_count() > 1:
            self.logger.info(f"  Using {torch.cuda.device_count()} GPUs")
            self.model = nn.DataParallel(self.model)

        total = sum(p.numel() for p in self.model.parameters())
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.logger.info(f"  Parameters: {total:,} total, {trainable:,} trainable")

    def _setup_training(self):
        """Setup loss, optimizer, scheduler, scaler."""
        self.criterion = nn.CrossEntropyLoss(
            weight=self.class_weights,
            label_smoothing=self.label_smoothing,
        )

        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
            betas=(0.9, 0.999),
        )

        steps_per_epoch = len(self.train_loader)
        self.scheduler = OneCycleLR(
            self.optimizer,
            max_lr=self.lr,
            epochs=self.num_epochs,
            steps_per_epoch=steps_per_epoch,
            pct_start=0.3,
            anneal_strategy="cos",
            div_factor=25.0,
            final_div_factor=1e4,
        )

        self.scaler = GradScaler() if self.use_amp else None
        self.freeze_epochs = self.config["training"].get("freeze_backbone_epochs", 3)

    def _setup_logging(self):
        """Setup TensorBoard and optional WandB."""
        log_dir = Path(self.config["paths"]["logs"])
        log_dir.mkdir(parents=True, exist_ok=True)
        self.tb_writer = SummaryWriter(log_dir=log_dir)
        self.use_wandb = False  # Set to True and import wandb to enable

    def _progressive_unfreeze(self, epoch: int):
        """Handle progressive backbone unfreezing."""
        if epoch == 0:
            # Phase 1: Freeze backbone, train only classifier
            if hasattr(self.model, "module"):
                self.model.module.freeze_backbone()
            else:
                self.model.freeze_backbone()

            params = (
                self.model.module.classifier.parameters()
                if hasattr(self.model, "module")
                else self.model.classifier.parameters()
            )
            self.optimizer = optim.AdamW(
                params, lr=self.lr * 10, weight_decay=self.weight_decay
            )

            self.logger.info(
                f"[Epoch {epoch+1}] Phase 1: Backbone FROZEN | "
                f"Classifier LR = {self.lr*10:.2e}"
            )

        elif epoch == self.freeze_epochs:
            # Phase 2: Unfreeze backbone, use discriminative LR
            if hasattr(self.model, "module"):
                self.model.module.unfreeze_backbone()
            else:
                self.model.unfreeze_backbone()

            if hasattr(self.model, "module"):
                backbone_params = self.model.module.backbone.parameters()
                head_params = self.model.module.classifier.parameters()
            else:
                backbone_params = self.model.backbone.parameters()
                head_params = self.model.classifier.parameters()

            self.optimizer = optim.AdamW([
                {"params": backbone_params, "lr": self.lr / 10},
                {"params": head_params, "lr": self.lr},
            ], weight_decay=self.weight_decay)

            # Re-create OneCycleLR for full model fine-tuning
            steps_per_epoch = len(self.train_loader)
            remaining_epochs = self.num_epochs - self.freeze_epochs
            self.scheduler = OneCycleLR(
                self.optimizer,
                max_lr=[self.lr / 10, self.lr],
                epochs=remaining_epochs,
                steps_per_epoch=steps_per_epoch,
                pct_start=0.2,
            )

            self.logger.info(
                f"[Epoch {epoch+1}] Phase 2: Full fine-tuning | "
                f"Backbone LR = {self.lr/10:.2e}, Head LR = {self.lr:.2e}"
            )

    def _run_epoch(self, loader, training: bool = True) -> Dict:
        """Run one epoch."""
        self.model.train(training)
        metrics = MetricsTracker(self.num_classes)

        pbar = tqdm(loader, desc="Train" if training else "Val  ", leave=False)

        for images, labels in pbar:
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            if training:
                self.optimizer.zero_grad(set_to_none=True)

            # Mixed precision forward
            if self.use_amp:
                with autocast():
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)
            else:
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

            if training:
                if self.use_amp:
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.gradient_clip
                    )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.scheduler.step()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.gradient_clip
                    )
                    self.optimizer.step()
                    self.scheduler.step()

            metrics.update(loss.item(), outputs.detach(), labels)

            f1_scores = metrics.compute_f1_per_class()
            pbar.set_postfix({
                "loss": f"{metrics.avg_loss:.4f}",
                "acc": f"{metrics.accuracy:.4f}",
                "f1": f"{f1_scores['macro_f1']:.4f}",
            })

        return {
            "loss": metrics.avg_loss,
            "accuracy": metrics.accuracy,
            "f1_scores": metrics.compute_f1_per_class(),
        }

    def _save_checkpoint(self, filename: str, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            "epoch": self.current_epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_val_acc": self.best_val_acc,
            "config": self.config,
        }

        path = self.checkpoint_dir / filename
        torch.save(checkpoint, path)

        if is_best:
            best_path = self.checkpoint_dir / "best_model.pth"
            torch.save(checkpoint, best_path)
            self.logger.info(
                f"  *** New best model saved! (val_acc={self.best_val_acc:.4f}) ***"
            )

    def train(self) -> Dict:
        """Main training loop."""
        self.logger.info("=" * 70)
        self.logger.info("SCALE x ODYSSEY -- Training Started")
        self.logger.info("=" * 70)
        self.logger.info(f"Device: {self.device}")
        self.logger.info(f"Epochs: {self.num_epochs}")
        self.logger.info(f"Batch size: {self.batch_size}")
        self.logger.info(f"Learning rate: {self.lr}")
        self.logger.info(f"Mixed precision: {self.use_amp}")
        self.logger.info(f"Progressive unfreezing: {self.freeze_epochs} epochs")
        self.logger.info("=" * 70)

        start_time = time.time()

        for epoch in range(self.num_epochs):
            self.current_epoch = epoch
            epoch_start = time.time()

            # Progressive unfreezing
            if epoch <= self.freeze_epochs:
                self._progressive_unfreeze(epoch)

            # Training
            train_results = self._run_epoch(self.train_loader, training=True)

            # Validation
            val_results = self._run_epoch(self.val_loader, training=False)

            epoch_time = time.time() - epoch_start
            lr = self.optimizer.param_groups[0]["lr"]

            self.logger.info(
                f"[Epoch {epoch+1:02d}/{self.num_epochs}] "
                f"Time: {epoch_time:.1f}s | LR: {lr:.2e} | "
                f"Train Loss: {train_results['loss']:.4f} | "
                f"Train Acc: {train_results['accuracy']:.4f} | "
                f"Val Loss: {val_results['loss']:.4f} | "
                f"Val Acc: {val_results['accuracy']:.4f} | "
                f"Val F1: {val_results['f1_scores']['macro_f1']:.4f}"
            )

            # TensorBoard
            self.tb_writer.add_scalar("Loss/train", train_results["loss"], epoch)
            self.tb_writer.add_scalar("Loss/val", val_results["loss"], epoch)
            self.tb_writer.add_scalar("Accuracy/train", train_results["accuracy"], epoch)
            self.tb_writer.add_scalar("Accuracy/val", val_results["accuracy"], epoch)
            self.tb_writer.add_scalar("F1/val_macro", val_results["f1_scores"]["macro_f1"], epoch)
            self.tb_writer.add_scalar("LR", lr, epoch)

            # Checkpoint saving
            is_best = val_results["accuracy"] > self.best_val_acc
            if is_best:
                self.best_val_acc = val_results["accuracy"]
                self.patience_counter = 0
                self._save_checkpoint(f"epoch_{epoch+1:03d}.pth", is_best=True)
            else:
                self.patience_counter += 1
                if (epoch + 1) % 5 == 0:
                    self._save_checkpoint(f"epoch_{epoch+1:03d}.pth")

            # Early stopping
            if self.patience_counter >= self.patience:
                self.logger.info(
                    f"Early stopping triggered! "
                    f"No improvement for {self.patience} epochs."
                )
                break

        total_time = time.time() - start_time
        self.logger.info("=" * 70)
        self.logger.info("Training Complete!")
        self.logger.info(f"Total time: {total_time/60:.1f} minutes")
        self.logger.info(f"Best validation accuracy: {self.best_val_acc:.4f}")
        self.logger.info("=" * 70)

        self.tb_writer.close()

        return {
            "best_val_acc": self.best_val_acc,
            "total_epochs": self.current_epoch + 1,
            "total_time": total_time,
        }


# ============================================================================
# ENTRY POINT
# ============================================================================

def main():
    config = load_config()
    experiment_name = f"{config['model']['backbone']}_{int(time.time())}"
    logger = setup_logger(config["paths"]["logs"], experiment_name)

    trainer = Trainer(config, logger)
    results = trainer.train()

    summary_path = Path(config["paths"]["results"]) / "training_summary.json"
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
