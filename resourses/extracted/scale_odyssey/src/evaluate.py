#!/usr/bin/env python3
"""
SCALE x ODYSSEY -- Evaluation & Test-Time Augmentation

Computes:
- Overall accuracy, macro F1, per-class precision/recall/F1
- Confusion matrix (saved as image)
- Full classification report
- Test-Time Augmentation (TTA) with batched execution
- Per-class Grad-CAM sample generation
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List

import yaml
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, classification_report,
)
import albumentations as A
from albumentations.pytorch import ToTensorV2

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from model import AstroClassifier
from augmentations import AstroDataset, CLASSES, CLASS_NAMES_DISPLAY, get_validation_augmentations

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)


# ============================================================================
# CONFIG & MODEL LOADING
# ============================================================================

def load_config(config_path: str = "configs/config.yaml") -> Dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_model(checkpoint_path: str, config: Dict, device: torch.device):
    """Load model from checkpoint."""
    model = AstroClassifier(
        num_classes=config["model"]["num_classes"],
        backbone=config["model"]["backbone"],
        pretrained=False,
        dropout=config["model"]["dropout"],
    )

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    return model


# ============================================================================
# STANDARD EVALUATION
# ============================================================================

def evaluate_standard(model, dataloader, device) -> Dict:
    """Standard evaluation without TTA."""
    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Evaluating"):
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

    accuracy = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    weighted_f1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0)

    per_class_f1 = f1_score(all_labels, all_preds, average=None, zero_division=0)
    per_class_precision = precision_score(all_labels, all_preds, average=None, zero_division=0)
    per_class_recall = recall_score(all_labels, all_preds, average=None, zero_division=0)

    report = classification_report(
        all_labels, all_preds, target_names=CLASS_NAMES_DISPLAY, digits=4, zero_division=0
    )
    cm = confusion_matrix(all_labels, all_preds)

    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "per_class_f1": {CLASS_NAMES_DISPLAY[i]: per_class_f1[i] for i in range(len(CLASSES))},
        "per_class_precision": {CLASS_NAMES_DISPLAY[i]: per_class_precision[i] for i in range(len(CLASSES))},
        "per_class_recall": {CLASS_NAMES_DISPLAY[i]: per_class_recall[i] for i in range(len(CLASSES))},
        "confusion_matrix": cm,
        "classification_report": report,
        "predictions": all_preds,
        "labels": all_labels,
        "probabilities": all_probs,
    }


# ============================================================================
# BATCHED TEST-TIME AUGMENTATION (TTA)
# ============================================================================

class TTADataset(Dataset):
    """
    Dataset that applies one specific TTA transform to all images.
    Used for batched TTA execution (much faster than one-by-one).
    """

    def __init__(self, base_dataset: AstroDataset, tta_transform: A.Compose):
        self.base_dataset = base_dataset
        self.tta_transform = tta_transform

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        img = np.array(Image.open(self.base_dataset.samples[idx]).convert("RGB"))
        augmented = self.tta_transform(image=img)["image"]
        return augmented, self.base_dataset.labels[idx]


def evaluate_tta(model, test_dataset: AstroDataset, device, batch_size: int = 32, num_workers: int = 4) -> Dict:
    """
    Batched Test-Time Augmentation.

    Applies 6 augmentation variants, averages predictions.
    Uses DataLoader for GPU batching -- much faster than one-by-one.
    """
    from PIL import Image

    # Define 6 TTA transforms
    base = [
        A.Resize(224, 224),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ]

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
            # Create batched dataset + dataloader for this TTA variant
            tta_ds = TTADataset(test_dataset, transform)
            tta_loader = DataLoader(
                tta_ds,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True,
            )

            probs_aug = []
            for images, _ in tqdm(
                tta_loader,
                desc=f"TTA {aug_idx + 1}/{len(tta_transforms)}",
                leave=False,
            ):
                images = images.to(device, non_blocking=True)
                outputs = model(images)
                probs = torch.softmax(outputs, dim=1)
                probs_aug.extend(probs.cpu().numpy())

            all_tta_probs.append(np.array(probs_aug))

    # Average probabilities across augmentations
    mean_probs = np.mean(all_tta_probs, axis=0)
    tta_preds = mean_probs.argmax(axis=1)
    all_labels = np.array(test_dataset.labels)

    accuracy = accuracy_score(all_labels, tta_preds)
    macro_f1 = f1_score(all_labels, tta_preds, average="macro", zero_division=0)

    logger.info(f"TTA Accuracy:   {accuracy:.4f}")
    logger.info(f"TTA Macro F1:   {macro_f1:.4f}")

    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "predictions": tta_preds,
        "labels": all_labels,
        "probabilities": mean_probs,
    }


# ============================================================================
# GRAD-CAM GENERATION
# ============================================================================

def generate_gradcam_samples(
    model: AstroClassifier,
    test_dataset: AstroDataset,
    device: torch.device,
    output_dir: str,
    samples_per_class: int = 2,
):
    """
    Generate Grad-CAM heatmaps for sample images from each class.
    Saves overlaid visualizations to output_dir.
    """
    try:
        from pytorch_grad_cam import GradCAM
        from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
        from pytorch_grad_cam.utils.image import show_cam_on_image
    except ImportError:
        logger.warning("grad-cam not installed. Skipping Grad-CAM generation.")
        logger.warning("Install with: pip install grad-cam")
        return

    from PIL import Image

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get target layer from model
    if hasattr(model, "module"):
        target_layer = model.module.get_gradcam_target_layer()
    else:
        target_layer = model.get_gradcam_target_layer()

    # Wrap model for Grad-CAM (unwrap DataParallel if needed)
    cam_model = model.module if hasattr(model, "module") else model

    cam = GradCAM(model=cam_model, target_layers=target_layer)

    # Validation transform (no random augmentations)
    val_transform = get_validation_augmentations(224)

    logger.info("Generating Grad-CAM samples...")

    for class_idx, class_name in enumerate(CLASS_NAMES_DISPLAY):
        # Find test images belonging to this class
        class_indices = [
            i for i, label in enumerate(test_dataset.labels) if label == class_idx
        ]

        if not class_indices:
            continue

        # Pick random samples
        selected = np.random.choice(
            class_indices,
            size=min(samples_per_class, len(class_indices)),
            replace=False,
        )

        for sample_idx in selected:
            img_path = test_dataset.samples[sample_idx]
            img_pil = Image.open(img_path).convert("RGB")
            img_np = np.array(img_pil)

            # Prepare input tensor
            input_tensor = val_transform(image=img_np)["image"].unsqueeze(0).to(device)

            # Prepare visualization image (normalized to [0, 1])
            vis_img = cv2.resize(img_np, (224, 224)).astype(np.float32) / 255.0

            # Get prediction
            with torch.no_grad():
                output = cam_model(input_tensor)
                pred_idx = output.argmax(1).item()
                confidence = torch.softmax(output, 1)[0, pred_idx].item()

            # Generate Grad-CAM
            targets = [ClassifierOutputTarget(pred_idx)]
            grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0]

            # Overlay heatmap
            visualization = show_cam_on_image(
                vis_img, grayscale_cam, use_rgb=True, colormap=cv2.COLORMAP_JET
            )

            # Save
            true_label = CLASS_NAMES_DISPLAY[class_idx]
            pred_label = CLASS_NAMES_DISPLAY[pred_idx]
            fname = (
                f"{class_name.lower().replace(' ', '_')}_"
                f"pred{pred_label.lower().replace(' ', '_')}_"
                f"{confidence:.2f}_{Path(img_path).stem}.png"
            )
            save_path = output_dir / fname
            Image.fromarray(visualization).save(save_path)

            logger.info(f"  Saved: {fname}")

    logger.info(f"Grad-CAM samples saved to {output_dir}")


# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_confusion_matrix(cm: np.ndarray, save_path: str):
    """Plot and save confusion matrix."""
    plt.figure(figsize=(10, 8))

    cm_norm = cm.astype("float") / (cm.sum(axis=1)[:, np.newaxis] + 1e-8)

    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2%",
        cmap="Blues",
        xticklabels=CLASS_NAMES_DISPLAY,
        yticklabels=CLASS_NAMES_DISPLAY,
        annot_kws={"size": 12},
        cbar_kws={"label": "Proportion"},
    )

    plt.title("Confusion Matrix (Normalized)", fontsize=14, fontweight="bold")
    plt.ylabel("True Label", fontsize=12)
    plt.xlabel("Predicted Label", fontsize=12)
    plt.xticks(rotation=30, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()

    logger.info(f"Confusion matrix saved to {save_path}")


def plot_per_class_metrics(metrics: Dict, save_path: str):
    """Plot per-class precision, recall, F1."""
    classes = list(metrics["per_class_f1"].keys())
    f1_vals = list(metrics["per_class_f1"].values())
    precision_vals = list(metrics["per_class_precision"].values())
    recall_vals = list(metrics["per_class_recall"].values())

    x = np.arange(len(classes))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - width, precision_vals, width, label="Precision", color="#3498db", edgecolor="white")
    ax.bar(x, recall_vals, width, label="Recall", color="#2ecc71", edgecolor="white")
    ax.bar(x + width, f1_vals, width, label="F1 Score", color="#e74c3c", edgecolor="white")

    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Per-Class Metrics", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=30, ha="right")
    ax.legend(loc="lower right")
    ax.set_ylim([0, 1.05])
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()

    logger.info(f"Per-class metrics plot saved to {save_path}")


def plot_prediction_confidence_distribution(probs: np.ndarray, save_path: str):
    """Plot distribution of maximum prediction confidence."""
    max_conf = probs.max(axis=1)

    plt.figure(figsize=(10, 6))
    plt.hist(max_conf, bins=50, color="#9b59b6", edgecolor="white", alpha=0.8)
    plt.axvline(
        x=max_conf.mean(),
        color="red",
        linestyle="--",
        label=f"Mean: {max_conf.mean():.3f}",
    )
    plt.xlabel("Maximum Confidence", fontsize=12)
    plt.ylabel("Number of Samples", fontsize=12)
    plt.title("Prediction Confidence Distribution", fontsize=14, fontweight="bold")
    plt.legend()
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()

    logger.info(f"Confidence distribution saved to {save_path}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    config = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    results_dir = Path(config["paths"]["results"])
    results_dir.mkdir(parents=True, exist_ok=True)
    gradcam_dir = Path(config["paths"].get("gradcam_samples", "results/gradcam_samples"))
    gradcam_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    checkpoint_path = config.get("eval_checkpoint", "checkpoints/best_model.pth")
    logger.info(f"Loading model from: {checkpoint_path}")
    model = load_model(checkpoint_path, config, device)

    # Load test data
    test_dataset = AstroDataset(config["data"]["processed_dir"], "test", 224)
    test_loader = DataLoader(
        test_dataset,
        batch_size=config["training"].get("batch_size", 32),
        shuffle=False,
        num_workers=config["data"].get("num_workers", 4),
        pin_memory=True,
    )

    logger.info(f"Test samples: {len(test_dataset)}")

    # Standard evaluation
    logger.info("=" * 60)
    logger.info("Standard Evaluation (No TTA)")
    logger.info("=" * 60)
    results = evaluate_standard(model, test_loader, device)

    logger.info(f"Test Accuracy:  {results['accuracy']:.4f}")
    logger.info(f"Macro F1:       {results['macro_f1']:.4f}")
    logger.info(f"Weighted F1:    {results['weighted_f1']:.4f}")
    logger.info("\nPer-Class F1:")
    for cls, f1 in results["per_class_f1"].items():
        logger.info(f"  {cls:25s}: {f1:.4f}")
    logger.info("\nClassification Report:\n" + results["classification_report"])

    # Save confusion matrix
    plot_confusion_matrix(
        results["confusion_matrix"],
        str(results_dir / "confusion_matrix.png"),
    )

    # Save per-class metrics
    plot_per_class_metrics(
        results,
        str(results_dir / "per_class_metrics.png"),
    )

    # Save confidence distribution
    plot_prediction_confidence_distribution(
        results["probabilities"],
        str(results_dir / "confidence_distribution.png"),
    )

    # Test-Time Augmentation
    logger.info("=" * 60)
    logger.info("Test-Time Augmentation (TTA)")
    logger.info("=" * 60)
    tta_results = evaluate_tta(
        model,
        test_dataset,
        device,
        batch_size=config["training"].get("batch_size", 32),
        num_workers=config["data"].get("num_workers", 4),
    )

    # Grad-CAM samples
    logger.info("=" * 60)
    logger.info("Grad-CAM Explanation Generation")
    logger.info("=" * 60)
    generate_gradcam_samples(
        model,
        test_dataset,
        device,
        output_dir=gradcam_dir,
        samples_per_class=2,
    )

    # Save comprehensive results
    output = {
        "standard": {
            "accuracy": float(results["accuracy"]),
            "macro_f1": float(results["macro_f1"]),
            "weighted_f1": float(results["weighted_f1"]),
            "per_class_f1": {k: float(v) for k, v in results["per_class_f1"].items()},
        },
        "tta": {
            "accuracy": float(tta_results["accuracy"]),
            "macro_f1": float(tta_results["macro_f1"]),
        },
    }

    results_path = results_dir / "evaluation_results.json"
    with open(results_path, "w") as f:
        json.dump(output, f, indent=2)

    logger.info(f"\nResults saved to {results_path}")
    logger.info("=" * 60)
    logger.info("Evaluation Complete!")
    logger.info("=" * 60)

    return output


if __name__ == "__main__":
    main()
