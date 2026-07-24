#!/usr/bin/env python3
"""
SCALE x ODYSSEY -- src/utils.py
Helper functions for training, evaluation, and inference.
"""

import os
import json
import random
import logging
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import yaml
from sklearn.metrics import f1_score

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """Pick the best available accelerator: CUDA > Apple MPS > CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def load_config(config_path: str = None) -> Dict:
    """Load YAML configuration (checks configs/ then config/)."""
    if config_path is None:
        for p in ["configs/config.yaml", "config/config.yaml"]:
            if os.path.exists(p):
                config_path = p
                break
        if config_path is None:
            raise FileNotFoundError("No config.yaml found in configs/ or config/")
    with open(config_path) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Class Weights
# ---------------------------------------------------------------------------

def load_class_weights(data_dir: str, device: torch.device):
    """Load pre-computed class weights for imbalanced datasets."""
    weights_path = Path(data_dir) / "class_weights.json"
    if not weights_path.exists():
        logger.warning("No class weights found, using uniform weights")
        return None

    from dataset import CLASSES
    with open(weights_path) as f:
        weights_dict = json.load(f)
    weights = torch.tensor([weights_dict[c] for c in CLASSES], dtype=torch.float32)
    logger.info(f"Loaded class weights: {weights_dict}")
    return weights.to(device)


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------

def save_checkpoint(
    model, optimizer, scheduler, epoch: int, best_val_acc: float,
    config: Dict, filepath: str,
):
    """Save training checkpoint."""
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "best_val_acc": best_val_acc,
        "config": config,
    }
    torch.save(checkpoint, filepath)


def load_checkpoint(filepath: str, model, device: torch.device):
    """Load checkpoint and return state."""
    checkpoint = torch.load(filepath, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return checkpoint


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(labels, predictions, num_classes: int = 5) -> Dict:
    """Compute accuracy, macro F1, and per-class F1.

    Always returns a per_class_f1 dict of length ``num_classes`` (padded with
    0.0 for classes that don't appear in the labels/predictions) so downstream
    code that iterates over all class names never crashes.
    """
    labels = np.array(labels)
    predictions = np.array(predictions)

    accuracy = float((predictions == labels).mean())
    macro_f1 = float(f1_score(labels, predictions, average="macro", zero_division=0))
    per_class_f1_arr = f1_score(
        labels, predictions, average=None, zero_division=0,
        labels=list(range(num_classes)),
    )

    from dataset import CLASS_NAMES_DISPLAY
    per_class_f1 = {
        CLASS_NAMES_DISPLAY[i]: float(per_class_f1_arr[i])
        for i in range(num_classes)
    }
    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "per_class_f1": per_class_f1,
    }


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logger(log_dir: str, experiment_name: str) -> logging.Logger:
    """Setup file + console logging."""
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
