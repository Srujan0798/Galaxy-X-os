#!/usr/bin/env python3
"""
SCALE x ODYSSEY -- src/pseudo_label.py
Self-training with pseudo-labeling on unlabeled NASA images.

Uses confident predictions on unlabeled data to generate additional training labels.
This effectively expands the dataset beyond the initial 500/class.
"""

import os
import sys
import json
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dataset import CLASSES, get_val_transforms  # noqa: E402
from download_archives import fetch_archive_class  # noqa: E402


class PseudoLabelDataset(Dataset):
    """Dataset of unlabeled images with pseudo-labels."""
    
    def __init__(self, images: List[np.ndarray], pseudo_labels: List[int],
                 confidences: List[float], image_size: int = 224):
        self.images = images
        self.pseudo_labels = pseudo_labels
        self.confidences = confidences
        self.image_size = image_size
        self.transform = get_val_transforms(image_size)
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img = self.transform(image=self.images[idx])["image"]
        return img, self.pseudo_labels[idx], self.confidences[idx]


def generate_pseudo_labels(model: torch.nn.Module, unlabeled_images: List[np.ndarray],
                           device: torch.device, confidence_threshold: float = 0.95,
                           image_size: int = 224) -> Tuple[List[np.ndarray], List[int], List[float]]:
    """
    Generate pseudo-labels for unlabeled images using confident model predictions.
    
    Returns: (filtered_images, pseudo_labels, confidences)
    """
    model.eval()
    transform = get_val_transforms(image_size)
    
    filtered_images = []
    pseudo_labels = []
    confidences = []
    
    for img in unlabeled_images:
        img_tensor = transform(image=img)["image"].unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(img_tensor)
            probs = F.softmax(logits, dim=1)[0]
            max_conf, pred = probs.max(0)
        
        if max_conf.item() >= confidence_threshold:
            filtered_images.append(img)
            pseudo_labels.append(pred.item())
            confidences.append(max_conf.item())
    
    return filtered_images, pseudo_labels, confidences


def fetch_unlabeled_nasa_images(per_class: int = 200, image_size: int = 224) -> List[np.ndarray]:
    """
    Fetch additional unlabeled images from NASA Image Library for pseudo-labeling.
    These are images that don't fit our strict class filters but are still astronomical.
    """
    # Broader queries that might capture additional real astronomical images
    broad_queries = [
        "deep field galaxy", "hubble deep field", "ultra deep field",
        "star forming region", "interstellar medium", "galactic nebula",
        "planetary nebula", "globular cluster", "open cluster",
        "spiral arm", "galaxy morphology", "cosmic web",
    ]
    
    all_images = []
    for query in broad_queries:
        try:
            images, _ = fetch_archive_class(
                "nebula",  # Use nebula queries as base, but accept any
                per_class // len(broad_queries),
                image_size,
                timeout=30
            )
            all_images.extend(images)
        except Exception:
            continue
    
    return all_images


def self_train_step(model: torch.nn.Module, pseudo_dataset: PseudoLabelDataset,
                    optimizer: torch.optim.Optimizer, device: torch.device,
                    batch_size: int = 32, num_workers: int = 4) -> float:
    """
    Single self-training step on pseudo-labeled data.
    Returns average loss.
    """
    loader = DataLoader(pseudo_dataset, batch_size=batch_size, shuffle=True,
                       num_workers=num_workers, drop_last=False)
    
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    for images, labels, _ in loader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        logits = model(images)
        loss = F.cross_entropy(logits, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
    
    return total_loss / max(num_batches, 1)


def pseudo_labeling_pipeline(model: torch.nn.Module, device: torch.device,
                            output_dir: str = "data/pseudo",
                            confidence_threshold: float = 0.95,
                            per_class: int = 200,
                            image_size: int = 224,
                            num_rounds: int = 3) -> Dict:
    """
    Full pseudo-labeling pipeline:
    1. Fetch unlabeled NASA images
    2. Generate pseudo-labels with current model
    3. Train on pseudo-labeled data
    4. Repeat for num_rounds
    
    Returns summary dict with statistics.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    summary = {
        "rounds": [],
        "total_pseudo_labeled": 0,
        "confidence_threshold": confidence_threshold,
    }
    
    for round_idx in range(num_rounds):
        print(f"\n[Round {round_idx + 1}/{num_rounds}] Pseudo-labeling...")
        
        # Fetch unlabeled images
        unlabeled = fetch_unlabeled_nasa_images(per_class, image_size)
        print(f"  Fetched {len(unlabeled)} unlabeled NASA images")
        
        # Generate pseudo-labels
        filtered_imgs, pseudo_labels, confidences = generate_pseudo_labels(
            model, unlabeled, device, confidence_threshold, image_size
        )
        
        round_stats = {
            "round": round_idx + 1,
            "unlabeled_fetched": len(unlabeled),
            "pseudo_labeled": len(filtered_imgs),
            "avg_confidence": float(np.mean(confidences)) if confidences else 0.0,
            "min_confidence": float(np.min(confidences)) if confidences else 0.0,
        }
        summary["rounds"].append(round_stats)
        summary["total_pseudo_labeled"] += len(filtered_imgs)
        
        print(f"  Pseudo-labeled: {len(filtered_imgs)}/{len(unlabeled)} "
              f"(avg conf: {round_stats['avg_confidence']:.3f})")
        
        if len(filtered_imgs) == 0:
            print("  No confident predictions, stopping.")
            break
        
        # Save pseudo-labeled data
        pseudo_dataset = PseudoLabelDataset(filtered_imgs, pseudo_labels, confidences, image_size)
        
        # Save to disk for reproducibility
        round_dir = os.path.join(output_dir, f"round_{round_idx + 1}")
        os.makedirs(round_dir, exist_ok=True)
        
        for i, (img, label, conf) in enumerate(zip(filtered_imgs, pseudo_labels, confidences)):
            class_name = CLASSES[label]
            class_dir = os.path.join(round_dir, class_name)
            os.makedirs(class_dir, exist_ok=True)
            from PIL import Image
            Image.fromarray(img).save(os.path.join(class_dir, f"pseudo_{i:04d}.png"))
        
        # Self-train
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=0.01)
        
        avg_loss = self_train_step(model, pseudo_dataset, optimizer, device)
        print(f"  Self-train loss: {avg_loss:.4f}")
        
        # Save model checkpoint
        torch.save({
            "model_state_dict": model.state_dict(),
            "round": round_idx + 1,
            "pseudo_labeled": len(filtered_imgs),
            "avg_confidence": round_stats["avg_confidence"],
        }, os.path.join(round_dir, "pseudo_checkpoint.pth"))
    
    # Save summary
    with open(os.path.join(output_dir, "pseudo_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    
    return summary