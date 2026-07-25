#!/usr/bin/env python3
"""
SCALE x ODYSSEY -- src/evaluate.py
Full Evaluation: Ensemble + TTA + Uncertainty Quantification

Combines official guide (page 13) with our:
- Ensemble evaluation (3 backbones averaged)
- Advanced TTA (10-crop + multi-scale + rotation)
- Uncertainty quantification (epistemic + aleatoric)
- Confusion matrix, per-class metrics, and confidence analysis
"""

import sys
import json
import time
import logging
from pathlib import Path
from typing import Dict, Optional
from dataclasses import dataclass

_SRC_DIR = Path(__file__).resolve().parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

from dataset import AstroDataset, CLASS_NAMES_DISPLAY  # noqa: E402
from model import AstroClassifier, AstroEnsemble, DEFAULT_ENSEMBLE_CONFIG  # noqa: E402
from utils import load_config, compute_metrics, get_device, check_data_exists  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class EvaluationResults:
    accuracy: float
    macro_precision: float
    macro_recall: float
    macro_f1: float
    per_class_f1: Dict[str, float]
    confusion_matrix: np.ndarray
    classification_report: str
    predicted_classes: np.ndarray
    true_classes: np.ndarray
    probabilities: np.ndarray
    epistemic_uncertainty: Optional[np.ndarray] = None
    aleatoric_uncertainty: Optional[np.ndarray] = None
    total_uncertainty: Optional[np.ndarray] = None


# ---------------------------------------------------------------------------
# Standard Evaluation
# ---------------------------------------------------------------------------

def evaluate_standard(model, loader, device) -> EvaluationResults:
    """Standard evaluation without TTA. Supports ensemble models."""
    all_preds, all_labels, all_probs = [], [], []
    all_epistemic, all_aleatoric = [], []

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Evaluating"):
            images = images.to(device, non_blocking=True)
            
            if isinstance(model, AstroEnsemble):
                mean_logits, epistemic, aleatoric, total = model.predict_with_uncertainty(images)
                probs = F.softmax(mean_logits, dim=1)
                preds = probs.argmax(dim=1)
                all_epistemic.extend(epistemic.cpu().numpy())
                all_aleatoric.extend(aleatoric.cpu().numpy())
            else:
                outputs = model(images)
                probs = F.softmax(outputs, dim=1)
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
    
    from sklearn.metrics import precision_score, recall_score
    macro_precision = precision_score(all_labels, all_preds, average="macro", zero_division=0)
    macro_recall = recall_score(all_labels, all_preds, average="macro", zero_division=0)

    return EvaluationResults(
        accuracy=float(metrics["accuracy"]),
        macro_precision=float(macro_precision),
        macro_recall=float(macro_recall),
        macro_f1=float(metrics["macro_f1"]),
        per_class_f1=metrics["per_class_f1"],
        confusion_matrix=cm,
        classification_report=report,
        predicted_classes=all_preds,
        true_classes=all_labels,
        probabilities=all_probs,
        epistemic_uncertainty=np.array(all_epistemic) if all_epistemic else None,
        aleatoric_uncertainty=np.array(all_aleatoric) if all_aleatoric else None,
        total_uncertainty=(np.array(all_epistemic) + np.array(all_aleatoric)) if (all_epistemic and all_aleatoric) else None,
    )


# ---------------------------------------------------------------------------
# TTA Evaluation
# ---------------------------------------------------------------------------

def evaluate_tta(model, test_dataset, device, batch_size=32, num_workers=4, image_size=224,
                 tta_type: str = "standard") -> Dict:
    """Test-Time Augmentation evaluation.
    
    tta_type options:
    - 'standard': 6 augmentations (original + flip + brightness variants)
    - 'advanced': 10-crop + multi-scale + rotation (30 augs)
    - 'heavy': 10-crop x 3 scales x 4 rotations (120 augs)
    """
    if tta_type == "advanced" or tta_type == "heavy":
        tta_transforms = get_tta_transforms(image_size) if tta_type == "advanced" else get_tta_transforms_heavy(image_size)
        logger.info(f"Running advanced TTA with {len(tta_transforms)} transforms...")
        
        all_logits_sum = None
        count = 0
        
        for transform in tqdm(tta_transforms, desc="TTA transforms"):
            loader = DataLoader(
                TTADataset(test_dataset, transform),
                batch_size=batch_size, shuffle=False,
                num_workers=num_workers, pin_memory=True
            )
            
            batch_logits = []
            for images, _ in loader:
                images = images.to(device, non_blocking=True)
                if isinstance(model, AstroEnsemble):
                    mean_logits, _, _, _ = model.predict_with_uncertainty(images)
                else:
                    with torch.no_grad():
                        mean_logits = model(images)
                batch_logits.append(mean_logits.cpu())
            
            all_logits = torch.cat(batch_logits, dim=0)
            
            if all_logits_sum is None:
                all_logits_sum = all_logits
            else:
                all_logits_sum += all_logits
            count += 1
        
        mean_logits = all_logits_sum / count
        mean_probs = F.softmax(mean_logits, dim=1).numpy()
        tta_preds = mean_probs.argmax(axis=1)
        all_labels = np.array(test_dataset.labels)
        
        accuracy = accuracy_score(all_labels, tta_preds)
        macro_f1_val = f1_score(all_labels, tta_preds, average="macro", zero_division=0)
        
        logger.info(f"Advanced TTA ({tta_type}): Acc={accuracy:.4f} | Macro F1={macro_f1_val:.4f}")
        return {"accuracy": accuracy, "macro_f1": macro_f1_val, "predictions": tta_preds,
                "labels": all_labels, "probabilities": mean_probs}
    
    else:
        # Standard 6x TTA
        import albumentations as A
        from albumentations.pytorch import ToTensorV2
        
        base = [A.Resize(image_size, image_size), A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), ToTensorV2()]
        
        tta_transforms = [
            A.Compose(base),
            A.Compose([A.HorizontalFlip(p=1.0)] + base),
            A.Compose([A.VerticalFlip(p=1.0)] + base),
            A.Compose([A.RandomRotate90(p=1.0)] + base),
            A.Compose([A.RandomBrightnessContrast(brightness_limit=(0.05, 0.1), contrast_limit=0, p=1.0)] + base),
            A.Compose([A.RandomBrightnessContrast(brightness_limit=(-0.1, -0.05), contrast_limit=0, p=1.0)] + base),
        ]
        
        logger.info(f"Running standard TTA with {len(tta_transforms)} augmentations...")
        
        all_tta_probs = []
        with torch.no_grad():
            for aug_idx, transform in enumerate(tta_transforms):
                tta_ds = TTADataset(test_dataset, transform)
                tta_loader = DataLoader(tta_ds, batch_size=batch_size, shuffle=False,
                                        num_workers=num_workers, pin_memory=True)
                probs_aug = []
                for images, _ in tqdm(tta_loader, desc=f"TTA {aug_idx+1}/{len(tta_transforms)}", leave=False):
                    images = images.to(device, non_blocking=True)
                    if isinstance(model, AstroEnsemble):
                        mean_logits, _, _, _ = model.predict_with_uncertainty(images)
                        probs_aug.extend(F.softmax(mean_logits, dim=1).cpu().numpy())
                    else:
                        outputs = model(images)
                        probs_aug.extend(F.softmax(outputs, dim=1).cpu().numpy())
                all_tta_probs.append(np.array(probs_aug))
        
        mean_probs = np.mean(all_tta_probs, axis=0)
        tta_preds = mean_probs.argmax(axis=1)
        all_labels = np.array(test_dataset.labels)
        
        accuracy = accuracy_score(all_labels, tta_preds)
        macro_f1_val = f1_score(all_labels, tta_preds, average="macro", zero_division=0)
        
        logger.info(f"Standard TTA: Acc={accuracy:.4f} | Macro F1={macro_f1_val:.4f}")
        return {"accuracy": accuracy, "macro_f1": macro_f1_val, "predictions": tta_preds,
                "labels": all_labels, "probabilities": mean_probs}


class TTADataset(torch.utils.data.Dataset):
    """Apply one TTA transform to all images for batched execution."""
    def __init__(self, base_dataset, tta_transform):
        self.base_dataset = base_dataset
        self.tta_transform = tta_transform
    
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx):
        img = np.array(Image.open(self.base_dataset.samples[idx]).convert("RGB"))
        return self.tta_transform(image=img)["image"], self.base_dataset.labels[idx]


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


def plot_uncertainty_distribution(epistemic, aleatoric, total, save_path):
    plt.figure(figsize=(14, 4))
    plt.subplot(1, 3, 1)
    plt.hist(epistemic, bins=50, color="#3498db", edgecolor="white", alpha=0.8)
    plt.title("Epistemic Uncertainty")
    plt.xlabel("Variance")
    
    plt.subplot(1, 3, 2)
    plt.hist(aleatoric, bins=50, color="#e74c3c", edgecolor="white", alpha=0.8)
    plt.title("Aleatoric Uncertainty")
    plt.xlabel("Entropy")
    
    plt.subplot(1, 3, 3)
    plt.hist(total, bins=50, color="#9b59b6", edgecolor="white", alpha=0.8)
    plt.title("Total Uncertainty")
    plt.xlabel("Combined")
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()


def plot_calibration_curve(probs, labels, save_path, n_bins=10):
    """Reliability diagram for confidence calibration."""
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    accuracies = (predictions == labels)
    
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    bin_accuracy = np.zeros(n_bins)
    bin_confidence = np.zeros(n_bins)
    bin_counts = np.zeros(n_bins)
    
    for i in range(n_bins):
        mask = (confidences >= bin_edges[i]) & (confidences < bin_edges[i + 1])
        bin_counts[i] = mask.sum()
        if bin_counts[i] > 0:
            bin_accuracy[i] = accuracies[mask].mean()
            bin_confidence[i] = confidences[mask].mean()
    
    # ECE (Expected Calibration Error)
    ece = np.sum(bin_counts * np.abs(bin_accuracy - bin_confidence)) / bin_counts.sum()
    
    plt.figure(figsize=(8, 8))
    plt.plot([0, 1], [0, 1], "k--", alpha=0.3, label="Perfectly Calibrated")
    plt.plot(bin_centers, bin_accuracy, "o-", color="#3498db", linewidth=2, label=f"Model (ECE={ece:.4f})")
    plt.bar(bin_centers, bin_accuracy - bin_confidence, width=1/n_bins, alpha=0.3, color="#e74c3c")
    plt.xlabel("Confidence", fontsize=12)
    plt.ylabel("Accuracy", fontsize=12)
    plt.title(f"Calibration Curve (ECE={ece:.4f})", fontsize=14, fontweight="bold")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    
    return ece


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def load_model_for_eval(config, device, ensemble: bool = False):
    """Load single model or ensemble for evaluation."""
    if ensemble:
        logger.info("Loading ensemble of 3 models...")
        models = []
        ensemble_config = config.get("ensemble", DEFAULT_ENSEMBLE_CONFIG)
        
        for i, mc in enumerate(ensemble_config.get("ensemble", [])):
            model = AstroClassifier(
                num_classes=mc.get("num_classes", 5),
                backbone=mc.get("backbone", "convnext_base"),
                pretrained=False,
                dropout=mc.get("dropout", 0.4),
            ).to(device)
            
            checkpoint_path = mc.get("checkpoint", f"checkpoints/{mc['backbone']}.pth")
            if Path(checkpoint_path).exists():
                ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
                model.load_state_dict(ckpt["model_state_dict"])
                logger.info(f"  [{i}] {mc['backbone']} loaded from {checkpoint_path}")
            else:
                logger.warning(f"  [{i}] Checkpoint not found: {checkpoint_path}, using random weights")
            
            models.append(model)
        
        ensemble_model = AstroEnsemble(models, mc_dropout=ensemble_config.get("mc_dropout", 10))
        ensemble_model.to(device)
        ensemble_model.eval()
        return ensemble_model
    
    else:
        # Single model
        checkpoint_path = config.get("eval_checkpoint", "checkpoints/best_model.pth")
        model_cfg = config["model"]
        model = AstroClassifier(model_cfg["num_classes"], model_cfg["backbone"],
                                pretrained=False, dropout=model_cfg["dropout"]).to(device)
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        logger.info(f"Loaded single model: {model_cfg['backbone']}")
        return model


def main():
    import argparse
    parser = argparse.ArgumentParser(description="SCALE x ODYSSEY Evaluation (Ensemble + TTA + Uncertainty)")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--device", default=None, choices=["cuda", "mps", "cpu"])
    parser.add_argument("--ensemble", action="store_true", help="Use ensemble (3 backbones)")
    parser.add_argument("--tta", type=str, default="standard", choices=["none", "standard", "advanced", "heavy"],
                       help="TTA type: none, standard (6x), advanced (30x), heavy (120x)")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    check_data_exists(config["data"]["processed_dir"])
    device = torch.device(args.device) if args.device else get_device()
    
    logger.info(f"Device: {device}")
    if args.ensemble:
        logger.info("Mode: ENSEMBLE (3 backbones + MC Dropout uncertainty)")
    else:
        logger.info("Mode: SINGLE MODEL")

    results_dir = Path(config["paths"]["results"])
    if args.overwrite:
        results_dir.mkdir(parents=True, exist_ok=True)
    else:
        tag = time.strftime("%Y%m%d_%H%M%S")
        results_dir = results_dir / f"eval_{tag}"
        results_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    model = load_model_for_eval(config, device, ensemble=args.ensemble)
    
    # Load test data
    test_dataset = AstroDataset(config["data"]["processed_dir"], "test", config["data"]["image_size"])
    test_loader = DataLoader(test_dataset, batch_size=config["training"].get("batch_size", 32),
                             shuffle=False, num_workers=config["data"].get("num_workers", 4), pin_memory=True)
    logger.info(f"Test samples: {len(test_dataset)}")

    # Standard evaluation
    logger.info("=" * 60)
    logger.info("STANDARD EVALUATION")
    logger.info("=" * 60)
    results = evaluate_standard(model, test_loader, device)

    logger.info(f"Accuracy:  {results.accuracy:.4f}")
    logger.info(f"Macro F1:  {results.macro_f1:.4f}")
    logger.info(f"Macro Precision: {results.macro_precision:.4f}")
    logger.info(f"Macro Recall:    {results.macro_recall:.4f}")
    logger.info("\nPer-Class F1:")
    for cls, f1 in results.per_class_f1.items():
        logger.info(f"  {cls:25s}: {f1:.4f}")

    if results.epistemic_uncertainty is not None:
        logger.info(f"\nEpistemic uncertainty (mean): {results.epistemic_uncertainty.mean():.6f}")
        logger.info(f"Aleatoric uncertainty (mean): {results.aleatoric_uncertainty.mean():.6f}")

    # Plots
    plot_confusion_matrix(results.confusion_matrix, str(results_dir / "confusion_matrix.png"))
    plot_per_class_metrics(results.per_class_f1, str(results_dir / "per_class_metrics.png"))
    plot_confidence_distribution(results.probabilities, str(results_dir / "confidence_distribution.png"))
    
    if results.epistemic_uncertainty is not None:
        plot_uncertainty_distribution(
            results.epistemic_uncertainty, results.aleatoric_uncertainty, results.total_uncertainty,
            str(results_dir / "uncertainty_distribution.png")
        )
    
    ece = plot_calibration_curve(results.probabilities, results.true_classes, str(results_dir / "calibration_curve.png"))

    # TTA
    tta_results = None
    if args.tta != "none":
        logger.info("=" * 60)
        logger.info(f"TEST-TIME AUGMENTATION ({args.tta.upper()})")
        logger.info("=" * 60)
        tta_results = evaluate_tta(model, test_dataset, device,
                                    config["training"].get("batch_size", 32),
                                    config["data"].get("num_workers", 4),
                                    config["data"]["image_size"],
                                    tta_type=args.tta)

    # Save results
    output = {
        "standard": {
            "accuracy": results.accuracy,
            "macro_precision": results.macro_precision,
            "macro_recall": results.macro_recall,
            "macro_f1": results.macro_f1,
            "per_class_f1": {k: float(v) for k, v in results.per_class_f1.items()},
            "ece": float(ece),
        },
        "tta": None if not tta_results else {
            "accuracy": float(tta_results["accuracy"]),
            "macro_f1": float(tta_results["macro_f1"]),
        },
        "uncertainty": None if results.epistemic_uncertainty is None else {
            "mean_epistemic": float(results.epistemic_uncertainty.mean()),
            "mean_aleatoric": float(results.aleatoric_uncertainty.mean()),
            "mean_total": float(results.total_uncertainty.mean()),
        },
        "ensemble": args.ensemble,
        "tta_type": args.tta,
        "model": config["model"],
        "classification_report": results.classification_report,
    }
    
    with open(results_dir / "evaluation_results.json", "w") as f:
        json.dump(output, f, indent=2)

    logger.info(f"\nResults saved to {results_dir}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()