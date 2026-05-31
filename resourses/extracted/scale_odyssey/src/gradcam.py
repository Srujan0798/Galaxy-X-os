#!/usr/bin/env python3
"""
SCALE x ODYSSEY -- Grad-CAM Explainability Module

Generates high-quality Grad-CAM visualizations for model interpretability.
Uses the existing model.get_gradcam_target_layer() for backbone-agnostic targeting.

Features:
- Single-image Grad-CAM with heatmap overlay
- visualize_predictions(): generates 10-15 test examples showing
  original image + true-class CAM + predicted-class CAM side-by-side
- Batch generation for full test-set coverage
- Saves high-res PNGs to results/gradcam/
"""

import os
import random
import logging
from pathlib import Path
from typing import Tuple, Optional, List, Dict
from dataclasses import dataclass

import cv2
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from model import AstroClassifier
from augmentations import AstroDataset, CLASS_NAMES_DISPLAY, get_validation_augmentations

try:
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    from pytorch_grad_cam.utils.image import show_cam_on_image
    GRADCAM_AVAILABLE = True
except ImportError:
    GRADCAM_AVAILABLE = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)

# Reproducibility
random.seed(42)
np.random.seed(42)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class GradCAMConfig:
    checkpoint_path: str = "checkpoints/best_model.pth"
    data_dir: str = "data/processed"
    output_dir: str = "results/gradcam"
    image_size: int = 224
    num_samples: int = 15              # Number of test examples to visualize
    samples_per_class: int = 3         # Min samples per class
    colormap: int = cv2.COLORMAP_JET
    alpha: float = 0.5                 # Heatmap overlay opacity
    fig_dpi: int = 200


# ---------------------------------------------------------------------------
# Core Grad-CAM Functions
# ---------------------------------------------------------------------------

def load_model_for_gradcam(
    checkpoint_path: str,
    device: torch.device,
) -> nn.Module:
    """Load model from checkpoint and prepare for Grad-CAM."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}\n"
            f"Train a model first: python src/train.py"
        )

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint.get("config", {})

    model_cfg = config.get("model", {})
    backbone = model_cfg.get("backbone", "efficientnet_b3")
    num_classes = model_cfg.get("num_classes", 5)
    dropout = model_cfg.get("dropout", 0.4)

    model = AstroClassifier(
        num_classes=num_classes,
        backbone=backbone,
        pretrained=False,
        dropout=dropout,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    logger.info(f"Loaded model: {backbone} | Classes: {num_classes}")
    return model


def generate_cam(
    model: nn.Module,
    input_tensor: torch.Tensor,
    target_class: int,
    device: torch.device,
) -> np.ndarray:
    """
    Generate Grad-CAM heatmap for a specific target class.

    Returns:
        grayscale_cam: 2D numpy array [H, W] with values in [0, 1]
    """
    # Unwrap DataParallel if present
    cam_model = model.module if hasattr(model, "module") else model
    target_layers = cam_model.get_gradcam_target_layer()

    with GradCAM(model=cam_model, target_layers=target_layers) as cam:
        targets = [ClassifierOutputTarget(target_class)]
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0]

    return grayscale_cam


def create_overlay(
    rgb_image: np.ndarray,
    cam_heatmap: np.ndarray,
    colormap: int = cv2.COLORMAP_JET,
    alpha: float = 0.5,
) -> np.ndarray:
    """
    Overlay Grad-CAM heatmap on RGB image.

    Args:
        rgb_image: [H, W, 3] float32 in [0, 1]
        cam_heatmap: [H, W] float32 in [0, 1]
    Returns:
        overlay: [H, W, 3] uint8 RGB
    """
    # Ensure correct shape and type
    if rgb_image.max() > 1.0:
        rgb_image = rgb_image.astype(np.float32) / 255.0

    # Resize CAM to match image if needed
    if cam_heatmap.shape[:2] != rgb_image.shape[:2]:
        cam_heatmap = cv2.resize(
            cam_heatmap,
            (rgb_image.shape[1], rgb_image.shape[0]),
            interpolation=cv2.INTER_LINEAR,
        )

    visualization = show_cam_on_image(
        rgb_image,
        cam_heatmap,
        use_rgb=True,
        colormap=colormap,
        image_weight=(1 - alpha),
    )
    return visualization


def preprocess_for_cam(image_path: str, image_size: int = 224) -> Tuple[torch.Tensor, np.ndarray]:
    """
    Preprocess image for Grad-CAM.

    Returns:
        input_tensor: [1, 3, H, W] on CPU (caller moves to device)
        vis_image: [H, W, 3] float32 in [0, 1] for overlay
    """
    transform = get_validation_augmentations(image_size)

    img_pil = Image.open(image_path).convert("RGB")
    img_np = np.array(img_pil)

    # Visualization image (resized, normalized to [0, 1])
    vis_image = cv2.resize(img_np, (image_size, image_size)).astype(np.float32) / 255.0

    # Model input tensor
    augmented = transform(image=img_np)["image"]
    input_tensor = augmented.unsqueeze(0)

    return input_tensor, vis_image


def predict_single(model: nn.Module, input_tensor: torch.Tensor, device: torch.device) -> Tuple[int, float, np.ndarray]:
    """Run inference and return prediction."""
    input_tensor = input_tensor.to(device)
    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.softmax(output, dim=1)

    pred_idx = probs.argmax(1).item()
    confidence = probs[0, pred_idx].item()
    all_probs = probs[0].cpu().numpy()

    return pred_idx, confidence, all_probs


# ---------------------------------------------------------------------------
# Single Image Pipeline
# ---------------------------------------------------------------------------

def explain_image(
    model: nn.Module,
    image_path: str,
    device: torch.device,
    true_label: Optional[int] = None,
    image_size: int = 224,
    output_path: Optional[str] = None,
    colormap: int = cv2.COLORMAP_JET,
    alpha: float = 0.5,
) -> Dict:
    """
    Full Grad-CAM pipeline for a single image.

    Generates:
    - Original image
    - Predicted-class CAM overlay
    - True-class CAM overlay (if true_label provided)

    Returns:
        dict with pred_class, confidence, all_probs, and paths to saved images
    """
    if not GRADCAM_AVAILABLE:
        raise ImportError("Install grad-cam: pip install grad-cam")

    # Preprocess
    input_tensor, vis_image = preprocess_for_cam(image_path, image_size)
    input_tensor = input_tensor.to(device)

    # Predict
    pred_idx, confidence, all_probs = predict_single(model, input_tensor, device)
    pred_class = CLASS_NAMES_DISPLAY[pred_idx]

    # Generate CAM for predicted class
    pred_cam = generate_cam(model, input_tensor, pred_idx, device)
    pred_overlay = create_overlay(vis_image, pred_cam, colormap, alpha)

    result = {
        "image_path": image_path,
        "pred_class": pred_class,
        "pred_idx": pred_idx,
        "confidence": confidence,
        "all_probabilities": {CLASS_NAMES_DISPLAY[i]: float(all_probs[i]) for i in range(len(CLASS_NAMES_DISPLAY))},
        "predicted_cam": pred_overlay,
    }

    # Generate CAM for true class if different from prediction
    if true_label is not None and true_label != pred_idx:
        true_cam = generate_cam(model, input_tensor, true_label, device)
        true_overlay = create_overlay(vis_image, true_cam, colormap, alpha)
        result["true_class"] = CLASS_NAMES_DISPLAY[true_label]
        result["true_idx"] = true_label
        result["true_cam"] = true_overlay
    else:
        result["true_class"] = pred_class
        result["true_idx"] = pred_idx
        result["true_cam"] = pred_overlay

    # Save if output path provided
    if output_path:
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
        Image.fromarray(result["predicted_cam"]).save(output_path)
        logger.info(f"Saved predicted-class CAM to {output_path}")

    return result


# ---------------------------------------------------------------------------
# Multi-Example Visualization
# ---------------------------------------------------------------------------

def select_diverse_samples(
    dataset: AstroDataset,
    num_total: int = 15,
    min_per_class: int = 3,
) -> List[Tuple[int, int]]:
    """
    Select diverse samples from test set for visualization.

    Ensures at least min_per_class samples per class,
    fills remaining with random picks.

    Returns:
        List of (dataset_index, label) tuples
    """
    samples_by_class = {}
    for idx, label in enumerate(dataset.labels):
        samples_by_class.setdefault(label, []).append(idx)

    selected = []
    remaining_budget = num_total

    # First pass: ensure minimum per class
    for class_idx in range(len(CLASS_NAMES_DISPLAY)):
        if class_idx not in samples_by_class:
            continue
        available = samples_by_class[class_idx]
        n_pick = min(min_per_class, len(available), remaining_budget)
        picks = random.sample(available, n_pick)
        selected.extend([(p, class_idx) for p in picks])
        remaining_budget -= n_pick

    # Second pass: fill remaining budget randomly
    all_remaining = [
        (idx, label)
        for idx, label in zip(range(len(dataset)), dataset.labels)
        if (idx, label) not in selected
    ]

    if all_remaining and remaining_budget > 0:
        extra = random.sample(all_remaining, min(remaining_budget, len(all_remaining)))
        selected.extend(extra)

    random.shuffle(selected)
    return selected[:num_total]


def visualize_predictions(
    model: nn.Module,
    dataset: AstroDataset,
    device: torch.device,
    output_dir: str = "results/gradcam",
    num_samples: int = 15,
    image_size: int = 224,
    fig_dpi: int = 200,
) -> List[Dict]:
    """
    Generate a grid of Grad-CAM visualizations for test-set samples.

    For each sample, creates a figure with 3 panels:
    - Original image
    - CAM for TRUE class
    - CAM for PREDICTED class

    Also saves a combined summary grid of all samples.

    Returns:
        List of result dicts from explain_image()
    """
    if not GRADCAM_AVAILABLE:
        logger.error("pytorch-grad-cam not installed. Run: pip install grad-cam")
        return []

    os.makedirs(output_dir, exist_ok=True)

    # Select diverse samples
    samples = select_diverse_samples(dataset, num_total=num_samples, min_per_class=3)
    logger.info(f"Generating Grad-CAM for {len(samples)} test samples...")

    all_results = []

    for i, (sample_idx, true_label) in enumerate(samples):
        img_path = dataset.samples[sample_idx]
        true_class_name = CLASS_NAMES_DISPLAY[true_label]

        # Generate explanation
        result = explain_image(
            model=model,
            image_path=img_path,
            device=device,
            true_label=true_label,
            image_size=image_size,
        )

        pred_class_name = result["pred_class"]
        confidence = result["confidence"]
        is_correct = result["pred_idx"] == true_label

        # Create 3-panel figure
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Panel 1: Original
        original_img = Image.open(img_path).convert("RGB")
        original_resized = np.array(original_img.resize((image_size, image_size)))
        axes[0].imshow(original_resized)
        axes[0].set_title(f"Original\n{true_class_name}", fontsize=11, fontweight="bold")
        axes[0].axis("off")

        # Panel 2: True-class CAM
        axes[1].imshow(result["true_cam"])
        title_color = "#2ecc71" if is_correct else "#e74c3c"
        axes[1].set_title(
            f"True-Class CAM\n{result['true_class']}",
            fontsize=11, fontweight="bold", color=title_color
        )
        axes[1].axis("off")

        # Panel 3: Predicted-class CAM
        pred_title = f"Predicted-Class CAM\n{pred_class_name} ({confidence:.1%})"
        axes[2].imshow(result["predicted_cam"])
        axes[2].set_title(pred_title, fontsize=11, fontweight="bold",
                          color="#3498db" if is_correct else "#e67e22")
        axes[2].axis("off")

        # Overall title
        status = "CORRECT" if is_correct else "MISCLASSIFIED"
        fig.suptitle(
            f"Sample {i+1}/{len(samples)} | True: {true_class_name} | "
            f"Pred: {pred_class_name} | {status}",
            fontsize=13, fontweight="bold", y=0.98
        )

        plt.tight_layout(rect=[0, 0, 1, 0.93])

        # Save individual figure
        fname = (
            f"sample_{i+1:02d}_{true_class_name.lower().replace(' ', '_')}_"
            f"pred_{pred_class_name.lower().replace(' ', '_')}_"
            f"{confidence:.2f}.png"
        )
        save_path = os.path.join(output_dir, fname)
        plt.savefig(save_path, dpi=fig_dpi, bbox_inches="tight", facecolor="white")
        plt.close(fig)

        logger.info(
            f"  [{i+1}/{len(samples)}] {fname} | "
            f"True: {true_class_name} | Pred: {pred_class_name} | {status}"
        )

        result["sample_index"] = i + 1
        result["true_class"] = true_class_name
        result["is_correct"] = is_correct
        all_results.append(result)

    # Create summary grid
    _create_summary_grid(all_results, output_dir, fig_dpi)

    # Log statistics
    correct = sum(1 for r in all_results if r["is_correct"])
    logger.info(f"\nGrad-CAM generation complete!")
    logger.info(f"  Total: {len(all_results)} | Correct: {correct} | Incorrect: {len(all_results) - correct}")
    logger.info(f"  Saved to: {output_dir}")

    return all_results


def _create_summary_grid(
    results: List[Dict],
    output_dir: str,
    fig_dpi: int = 200,
):
    """Create a compact grid of all predicted-class CAMs for quick review."""
    n = len(results)
    cols = 5
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3.2))
    axes = np.array(axes).reshape(-1)

    for i, result in enumerate(results):
        ax = axes[i]
        ax.imshow(result["predicted_cam"])

        color = "#2ecc71" if result["is_correct"] else "#e74c3c"
        ax.set_title(
            f"True: {result['true_class']}\n"
            f"Pred: {result['pred_class']}\n"
            f"Conf: {result['confidence']:.1%}",
            fontsize=8, color=color, fontweight="bold"
        )
        ax.axis("off")

    # Hide unused subplots
    for j in range(n, len(axes)):
        axes[j].axis("off")

    fig.suptitle(
        "SCALE x ODYSSEY -- Grad-CAM Summary (Predicted Class)",
        fontsize=14, fontweight="bold", y=1.02
    )
    plt.tight_layout()

    save_path = os.path.join(output_dir, "_summary_grid.png")
    plt.savefig(save_path, dpi=fig_dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info(f"  Summary grid saved to {save_path}")


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------

def main():
    """Generate Grad-CAM visualizations for test set."""
    cfg = GradCAMConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info("=" * 60)
    logger.info("SCALE x ODYSSEY -- Grad-CAM Generation")
    logger.info("=" * 60)

    # Load model
    model = load_model_for_gradcam(cfg.checkpoint_path, device)

    # Load test dataset
    test_dataset = AstroDataset(cfg.data_dir, "test", cfg.image_size)
    logger.info(f"Test samples available: {len(test_dataset)}")

    # Generate visualizations
    results = visualize_predictions(
        model=model,
        dataset=test_dataset,
        device=device,
        output_dir=cfg.output_dir,
        num_samples=cfg.num_samples,
        image_size=cfg.image_size,
        fig_dpi=cfg.fig_dpi,
    )

    logger.info("=" * 60)
    logger.info("Grad-CAM generation complete!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
