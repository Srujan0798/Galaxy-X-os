#!/usr/bin/env python3
"""
SCALE x ODYSSEY -- src/gradcam.py
Grad-CAM Explainability Module

Improved version of official guide (page 15-16):
- explain_image(): single-image Grad-CAM
- visualize_predictions(): 15 test samples (3 per class), 3-panel figures
- Summary grid for quick review
- Uses model.get_gradcam_target_layer() for backbone-agnostic targeting
"""

import os
import sys
import random
import logging
from pathlib import Path
from typing import List, Dict, Optional

_SRC_DIR = Path(__file__).resolve().parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

import cv2
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from dataset import AstroDataset, CLASS_NAMES_DISPLAY, get_val_transforms  # noqa: E402
from model import AstroClassifier  # noqa: E402

try:
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    from pytorch_grad_cam.utils.image import show_cam_on_image
    GRADCAM_AVAILABLE = True
except ImportError:
    GRADCAM_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)
random.seed(42)
np.random.seed(42)


def load_model_for_gradcam(checkpoint_path: str, device: torch.device) -> nn.Module:
    """Load model from checkpoint for Grad-CAM."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = checkpoint.get("config", {}).get("model", {})
    model = AstroClassifier(cfg.get("num_classes", 5), cfg.get("backbone", "efficientnet_b3"),
                            pretrained=False, dropout=cfg.get("dropout", 0.4))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    logger.info(f"Loaded model: {cfg.get('backbone', 'efficientnet_b3')}")
    return model


def generate_cam(model, input_tensor, target_class, device):
    """Generate Grad-CAM heatmap for target class."""
    cam_model = model.module if hasattr(model, "module") else model
    target_layers = cam_model.get_gradcam_target_layer()

    with GradCAM(model=cam_model, target_layers=target_layers) as cam:
        targets = [ClassifierOutputTarget(target_class)]
        return cam(input_tensor=input_tensor, targets=targets)[0]


def create_overlay(rgb_image, cam_heatmap, alpha=0.5):
    """Overlay Grad-CAM heatmap on RGB image."""
    if rgb_image.max() > 1.0:
        rgb_image = rgb_image.astype(np.float32) / 255.0
    if cam_heatmap.shape[:2] != rgb_image.shape[:2]:
        cam_heatmap = cv2.resize(cam_heatmap, (rgb_image.shape[1], rgb_image.shape[0]), interpolation=cv2.INTER_LINEAR)
    return show_cam_on_image(rgb_image, cam_heatmap, use_rgb=True, colormap=cv2.COLORMAP_JET, image_weight=(1 - alpha))


def preprocess_for_gradcam(image_path: str, image_size: int = 224):
    """Preprocess image: return input tensor + visualization image."""
    transform = get_val_transforms(image_size)
    img = np.array(Image.open(image_path).convert("RGB"))
    vis_image = cv2.resize(img, (image_size, image_size)).astype(np.float32) / 255.0
    input_tensor = transform(image=img)["image"].unsqueeze(0)
    return input_tensor, vis_image


def predict_single(model, input_tensor, device):
    """Run inference on single image."""
    input_tensor = input_tensor.to(device)
    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.softmax(output, dim=1)
    pred_idx = probs.argmax(1).item()
    return pred_idx, probs[0, pred_idx].item(), probs[0].cpu().numpy()


# ---------------------------------------------------------------------------
# explain_image(): Full Grad-CAM for single image
# ---------------------------------------------------------------------------

def explain_image(
    model: nn.Module,
    image_path: str,
    device: torch.device,
    true_label: Optional[int] = None,
    image_size: int = 224,
) -> Dict:
    """
    Full Grad-CAM pipeline for a single image.

    Returns dict with:
    - pred_class, confidence, all_probabilities
    - predicted_cam (overlay for predicted class)
    - true_cam (overlay for true class, if different)
    """
    if not GRADCAM_AVAILABLE:
        raise ImportError("pip install grad-cam")

    input_tensor, vis_image = preprocess_for_gradcam(image_path, image_size)
    pred_idx, confidence, all_probs = predict_single(model, input_tensor, device)
    pred_class = CLASS_NAMES_DISPLAY[pred_idx]

    # Predicted class CAM
    pred_cam = generate_cam(model, input_tensor.to(device), pred_idx, device)
    pred_overlay = create_overlay(vis_image, pred_cam)

    result = {
        "image_path": image_path,
        "pred_class": pred_class,
        "pred_idx": pred_idx,
        "confidence": confidence,
        "all_probabilities": {CLASS_NAMES_DISPLAY[i]: float(all_probs[i]) for i in range(len(CLASS_NAMES_DISPLAY))},
        "predicted_cam": pred_overlay,
        "true_class": pred_class,
        "true_idx": pred_idx,
        "true_cam": pred_overlay,
    }

    # True class CAM if different
    if true_label is not None and true_label != pred_idx:
        true_cam = generate_cam(model, input_tensor.to(device), true_label, device)
        result["true_class"] = CLASS_NAMES_DISPLAY[true_label]
        result["true_idx"] = true_label
        result["true_cam"] = create_overlay(vis_image, true_cam)

    return result


# ---------------------------------------------------------------------------
# visualize_predictions(): 15 diverse test samples
# ---------------------------------------------------------------------------

def select_diverse_samples(dataset: AstroDataset, num_total: int = 15, min_per_class: int = 3):
    """Select diverse samples ensuring minimum per class."""
    samples_by_class = {}
    for idx, label in enumerate(dataset.labels):
        samples_by_class.setdefault(label, []).append(idx)

    selected = []
    budget = num_total
    for class_idx in range(len(CLASS_NAMES_DISPLAY)):
        if class_idx not in samples_by_class:
            continue
        picks = random.sample(samples_by_class[class_idx], min(min_per_class, len(samples_by_class[class_idx]), budget))
        selected.extend([(p, class_idx) for p in picks])
        budget -= len(picks)

    selected_set = set(selected)
    remaining = [(i, lbl) for i, lbl in enumerate(dataset.labels) if (i, lbl) not in selected_set]
    if remaining and budget > 0:
        selected.extend(random.sample(remaining, min(budget, len(remaining))))
    random.shuffle(selected)
    return selected[:num_total]


def visualize_predictions(
    model: nn.Module,
    dataset: AstroDataset,
    device: torch.device,
    output_dir: str = "results/gradcam",
    num_samples: int = 15,
    image_size: int = 224,
):
    """Generate 3-panel Grad-CAM figures for diverse test samples."""
    if not GRADCAM_AVAILABLE:
        logger.error("pip install grad-cam")
        return []

    os.makedirs(output_dir, exist_ok=True)
    samples = select_diverse_samples(dataset, num_total=num_samples, min_per_class=3)
    logger.info(f"Generating Grad-CAM for {len(samples)} samples...")

    all_results = []
    for i, (sample_idx, true_label) in enumerate(samples):
        img_path = dataset.samples[sample_idx]
        result = explain_image(model, img_path, device, true_label=true_label, image_size=image_size)

        is_correct = result["pred_idx"] == true_label
        status = "CORRECT" if is_correct else "MISCLASSIFIED"

        # 3-panel figure
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Panel 1: Original
        orig = np.array(Image.open(img_path).convert("RGB").resize((image_size, image_size)))
        axes[0].imshow(orig)
        axes[0].set_title(f"Original\n{CLASS_NAMES_DISPLAY[true_label]}", fontsize=11, fontweight="bold")
        axes[0].axis("off")

        # Panel 2: True-class CAM
        axes[1].imshow(result["true_cam"])
        color = "#2ecc71" if is_correct else "#e74c3c"
        axes[1].set_title(f"True-Class CAM\n{result['true_class']}", fontsize=11, fontweight="bold", color=color)
        axes[1].axis("off")

        # Panel 3: Predicted-class CAM
        axes[2].imshow(result["predicted_cam"])
        pcolor = "#3498db" if is_correct else "#e67e22"
        axes[2].set_title(f"Predicted-Class CAM\n{result['pred_class']} ({result['confidence']:.1%})",
                          fontsize=11, fontweight="bold", color=pcolor)
        axes[2].axis("off")

        fig.suptitle(f"Sample {i+1}/{len(samples)} | True: {CLASS_NAMES_DISPLAY[true_label]} | "
                     f"Pred: {result['pred_class']} | {status}", fontsize=13, fontweight="bold", y=0.98)
        plt.tight_layout(rect=[0, 0, 1, 0.93])

        fname = f"sample_{i+1:02d}_{CLASS_NAMES_DISPLAY[true_label].replace(' ', '_').lower()}_" \
                f"pred_{result['pred_class'].replace(' ', '_').lower()}_{result['confidence']:.2f}.png"
        plt.savefig(os.path.join(output_dir, fname), dpi=200, bbox_inches="tight", facecolor="white")
        plt.close(fig)

        result["sample_index"] = i + 1
        result["true_class"] = CLASS_NAMES_DISPLAY[true_label]
        result["is_correct"] = is_correct
        all_results.append(result)

        logger.info(f"  [{i+1}/{len(samples)}] {fname} | {status}")

    # Summary grid
    _create_summary_grid(all_results, output_dir)

    correct = sum(1 for r in all_results if r["is_correct"])
    logger.info(f"\nDone! Total: {len(all_results)} | Correct: {correct} | Incorrect: {len(all_results)-correct}")
    return all_results


def _create_summary_grid(results: List[Dict], output_dir: str):
    """Create compact grid of all predicted-class CAMs."""
    n = len(results)
    cols, rows = 5, (n + 4) // 5
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3.2))
    axes = np.array(axes).reshape(-1)

    for i, result in enumerate(results):
        axes[i].imshow(result["predicted_cam"])
        color = "#2ecc71" if result["is_correct"] else "#e74c3c"
        axes[i].set_title(f"T: {result['true_class']}\nP: {result['pred_class']}\nC: {result['confidence']:.1%}",
                          fontsize=8, color=color, fontweight="bold")
        axes[i].axis("off")
    for j in range(n, len(axes)):
        axes[j].axis("off")

    fig.suptitle("SCALE x ODYSSEY -- Grad-CAM Summary (Predicted Class)", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "_summary_grid.png"), dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    logger.info(f"  Summary grid saved to {output_dir}/_summary_grid.png")


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------

def main():
    from utils import load_config, get_device
    config = load_config()
    device = get_device()
    model = load_model_for_gradcam(config.get("eval_checkpoint", "checkpoints/best_model.pth"), device)
    test_dataset = AstroDataset(config["data"]["processed_dir"], "test", config["data"]["image_size"])

    logger.info("=" * 60)
    logger.info("SCALE x ODYSSEY -- Grad-CAM Generation")
    logger.info("=" * 60)

    visualize_predictions(model, test_dataset, device,
                          output_dir=config["paths"].get("gradcam_dir", "results/gradcam"),
                          num_samples=15,
                          image_size=config["data"]["image_size"])

    logger.info("=" * 60)
    logger.info("Grad-CAM generation complete!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
