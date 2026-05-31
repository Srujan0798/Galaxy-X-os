#!/usr/bin/env python3
"""
SCALE x ODYSSEY -- src/evaluate.py
Full Evaluation + Test-Time Augmentation

Combines official guide (page 13) with our batched TTA,
confusion matrix, per-class metrics, and confidence analysis.
"""

import os
import json
import logging
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from dataset import AstroDataset, CLASS_NAMES_DISPLAY, get_val_transforms
from model import AstroClassifier
from utils import load_config, load_checkpoint, compute_metrics
from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Standard Evaluation
# ---------------------------------------------------------------------------

def evaluate_standard(model, loader, device):
    """Standard evaluation without TTA."""
    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Evaluating"):
            images = images.to(device, non_blocking=True)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    metrics = compute_metrics(all_labels, all_preds)
    report = classification_report(all_labels, all_preds,
                                    target_names=CLASS_NAMES_DISPLAY, digits=4, zero_division=0)
    cm = confusion_matrix(all_labels, all_preds)

    return {
        **metrics,
        "confusion_matrix": cm,
        "classification_report": report,
        "predictions": all_preds,
        "labels": all_labels,
        "probabilities": all_probs,
    }


# ---------------------------------------------------------------------------
# Test-Time Augmentation (TTA) -- Batched
# ---------------------------------------------------------------------------

class TTADataset(Dataset):
    """Apply one TTA transform to all images for batched execution."""

    def __init__(self, base_dataset, tta_transform):
        self.base_dataset = base_dataset
        self.tta_transform = tta_transform

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        img = np.array(Image.open(self.base_dataset.samples[idx]).convert("RGB"))
        return self.tta_transform(image=img)["image"], self.base_dataset.labels[idx]


def evaluate_tta(model, test_dataset, device, batch_size=32, num_workers=4):
    """Batched Test-Time Augmentation with 6 variants."""
    import albumentations as A
    from albumentations.pytorch import ToTensorV2

    base = [A.Resize(224, 224), A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), ToTensorV2()]

    tta_transforms = [
        A.Compose(base),
        A.Compose([A.HorizontalFlip(p=1.0)] + base),
        A.Compose([A.VerticalFlip(p=1.0)] + base),
        A.Compose([A.RandomRotate90(p=1.0)] + base),
        A.Compose([A.RandomBrightnessContrast(brightness_limit=(0.05, 0.1), contrast_limit=0, p=1.0)] + base),
        A.Compose([A.RandomBrightnessContrast(brightness_limit=(-0.1, -0.05), contrast_limit=0, p=1.0)] + base),
    ]

    logger.info(f"Running TTA with {len(tta_transforms)} augmentations (batched)...")

    all_tta_probs = []
    with torch.no_grad():
        for aug_idx, transform in enumerate(tta_transforms):
            tta_ds = TTADataset(test_dataset, transform)
            tta_loader = DataLoader(tta_ds, batch_size=batch_size, shuffle=False,
                                    num_workers=num_workers, pin_memory=True)
            probs_aug = []
            for images, _ in tqdm(tta_loader, desc=f"TTA {aug_idx+1}/{len(tta_transforms)}", leave=False):
                images = images.to(device, non_blocking=True)
                outputs = model(images)
                probs_aug.extend(torch.softmax(outputs, dim=1).cpu().numpy())
            all_tta_probs.append(np.array(probs_aug))

    mean_probs = np.mean(all_tta_probs, axis=0)
    tta_preds = mean_probs.argmax(axis=1)
    all_labels = np.array(test_dataset.labels)

    accuracy = accuracy_score(all_labels, tta_preds)
    macro_f1 = f1_score(all_labels, tta_preds, average="macro", zero_division=0)

    logger.info(f"TTA Accuracy: {accuracy:.4f} | TTA Macro F1: {macro_f1:.4f}")
    return {"accuracy": accuracy, "macro_f1": macro_f1, "predictions": tta_preds,
            "labels": all_labels, "probabilities": mean_probs}


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def plot_confusion_matrix(cm, save_path):
    plt.figure(figsize=(10, 8))
    cm_norm = cm.astype("float") / (cm.sum(axis=1)[:, np.newaxis] + 1e-8)
    sns.heatmap(cm_norm, annot=True, fmt=".2%", cmap="Blues",
                xticklabels=CLASS_NAMES_DISPLAY, yticklabels=CLASS_NAMES_DISPLAY)
    plt.title("Confusion Matrix (Normalized)", fontsize=14, fontweight="bold")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    logger.info(f"Confusion matrix saved to {save_path}")


def plot_per_class_metrics(per_class_f1, save_path):
    classes = list(per_class_f1.keys())
    values = list(per_class_f1.values())
    plt.figure(figsize=(10, 6))
    colors = ["#3498db" if v > 0.8 else "#f39c12" if v > 0.6 else "#e74c3c" for v in values]
    plt.barh(classes, values, color=colors, edgecolor="white")
    plt.xlim(0, 1)
    plt.xlabel("F1 Score", fontsize=12)
    plt.title("Per-Class F1 Score", fontsize=14, fontweight="bold")
    plt.axvline(x=0.8, color="green", linestyle="--", alpha=0.5, label="Target (0.8)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    logger.info(f"Per-class metrics saved to {save_path}")


def plot_confidence_distribution(probs, save_path):
    max_conf = probs.max(axis=1)
    plt.figure(figsize=(10, 6))
    plt.hist(max_conf, bins=50, color="#9b59b6", edgecolor="white", alpha=0.8)
    plt.axvline(x=max_conf.mean(), color="red", linestyle="--", label=f"Mean: {max_conf.mean():.3f}")
    plt.xlabel("Maximum Confidence")
    plt.ylabel("Number of Samples")
    plt.title("Prediction Confidence Distribution", fontsize=14, fontweight="bold")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    logger.info(f"Confidence distribution saved to {save_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    config = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results_dir = Path(config["paths"]["results"])
    results_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    checkpoint_path = config.get("eval_checkpoint", "checkpoints/best_model.pth")
    logger.info(f"Loading checkpoint: {checkpoint_path}")

    model_cfg = config["model"]
    model = AstroClassifier(model_cfg["num_classes"], model_cfg["backbone"],
                            pretrained=False, dropout=model_cfg["dropout"]).to(device)
    load_checkpoint(checkpoint_path, model, device)

    # Load test data
    test_dataset = AstroDataset(config["data"]["processed_dir"], "test", config["data"]["image_size"])
    test_loader = DataLoader(test_dataset, batch_size=config["training"].get("batch_size", 32),
                             shuffle=False, num_workers=config["data"].get("num_workers", 4), pin_memory=True)

    logger.info(f"Test samples: {len(test_dataset)}")

    # Standard evaluation
    logger.info("=" * 60)
    logger.info("Standard Evaluation")
    logger.info("=" * 60)
    results = evaluate_standard(model, test_loader, device)

    logger.info(f"Test Accuracy:  {results['accuracy']:.4f}")
    logger.info(f"Macro F1:       {results['macro_f1']:.4f}")
    logger.info("\nPer-Class F1:")
    for cls, f1 in results["per_class_f1"].items():
        logger.info(f"  {cls:25s}: {f1:.4f}")
    logger.info(f"\nClassification Report:\n{results['classification_report']}")

    plot_confusion_matrix(results["confusion_matrix"], str(results_dir / "confusion_matrix.png"))
    plot_per_class_metrics(results["per_class_f1"], str(results_dir / "per_class_metrics.png"))
    plot_confidence_distribution(results["probabilities"], str(results_dir / "confidence_distribution.png"))

    # TTA
    logger.info("=" * 60)
    logger.info("Test-Time Augmentation")
    logger.info("=" * 60)
    tta_results = evaluate_tta(model, test_dataset, device,
                                config["training"].get("batch_size", 32),
                                config["data"].get("num_workers", 4))

    # Save
    output = {
        "standard": {"accuracy": float(results["accuracy"]), "macro_f1": float(results["macro_f1"]),
                     "per_class_f1": {k: float(v) for k, v in results["per_class_f1"].items()}},
        "tta": {"accuracy": float(tta_results["accuracy"]), "macro_f1": float(tta_results["macro_f1"])},
    }
    with open(results_dir / "evaluation_results.json", "w") as f:
        json.dump(output, f, indent=2)

    logger.info(f"\nResults saved to {results_dir / 'evaluation_results.json'}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
