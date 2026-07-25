#!/usr/bin/env python3
"""
SCALE x ODYSSEY -- src/tta.py
Advanced Test-Time Augmentation (TTA) for maximum accuracy.

Supports:
- 10-crop (center + 4 corners + 5 flips)
- Multi-scale (0.8x, 1.0x, 1.2x)
- Rotation ensemble (0, 90, 180, 270)
- Configurable combinations
"""

import torch
import torch.nn.functional as F
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
from typing import List
import cv2


def get_tta_transforms(image_size: int = 224) -> List[A.Compose]:
    """
    Returns list of TTA transforms for ensemble prediction.
    Default: 10-crop + 3 scales = 30 augmentations per image.
    """
    base_size = int(image_size * 1.15)  # slightly larger for cropping
    
    transforms = []
    
    # 1. 10-crop at base scale
    crops = [
        A.Compose([
            A.Resize(height=base_size, width=base_size),
            A.CenterCrop(height=image_size, width=image_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ]),
        A.Compose([
            A.Resize(height=base_size, width=base_size),
            A.Crop(x_min=0, y_min=0, x_max=image_size, y_max=image_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ]),
        A.Compose([
            A.Resize(height=base_size, width=base_size),
            A.Crop(x_min=base_size-image_size, y_min=0, x_max=base_size, y_max=image_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ]),
        A.Compose([
            A.Resize(height=base_size, width=base_size),
            A.Crop(x_min=0, y_min=base_size-image_size, x_max=image_size, y_max=base_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ]),
        A.Compose([
            A.Resize(height=base_size, width=base_size),
            A.Crop(x_min=base_size-image_size, y_min=base_size-image_size, x_max=base_size, y_max=base_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ]),
    ]
    
    # Add flipped versions
    flipped_crops = []
    for t in crops:
        # Create a new transform that flips after the crop
        flipped = A.Compose([
            A.Resize(height=base_size, width=base_size),
            A.HorizontalFlip(p=1.0),
            t.transforms[1],  # crop
            t.transforms[2],  # normalize
            t.transforms[3],  # ToTensor
        ])
        flipped_crops.append(flipped)
    
    transforms = crops + flipped_crops
    
    # 2. Multi-scale (0.8x, 1.0x, 1.2x)
    for scale in [0.85, 1.0, 1.15]:
        scaled_size = int(image_size * scale)
        t = A.Compose([
            A.Resize(height=scaled_size, width=scaled_size),
            A.CenterCrop(height=image_size, width=image_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])
        transforms.append(t)
        
        # Flipped
        t_flip = A.Compose([
            A.Resize(height=scaled_size, width=scaled_size),
            A.HorizontalFlip(p=1.0),
            A.CenterCrop(height=image_size, width=image_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])
        transforms.append(t_flip)
    
    return transforms


def get_tta_transforms_heavy(image_size: int = 224) -> List[A.Compose]:
    """
    Heavy TTA: 10-crop x 3 scales x 4 rotations = 120 augmentations.
    Use for final submission only.
    """
    transforms = []
    
    rotations = [0, 90, 180, 270]
    scales = [0.85, 1.0, 1.15]
    
    for scale in scales:
        scaled_size = int(image_size * scale)
        for rot in rotations:
            if rot == 0:
                t = A.Compose([
                    A.Resize(height=scaled_size, width=scaled_size),
                    A.CenterCrop(height=image_size, width=image_size),
                    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ToTensorV2(),
                ])
            else:
                t = A.Compose([
                    A.Resize(height=scaled_size, width=scaled_size),
                    A.Rotate(limit=(rot, rot), p=1.0, border_mode=cv2.BORDER_CONSTANT, value=0),
                    A.CenterCrop(height=image_size, width=image_size),
                    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ToTensorV2(),
                ])
            transforms.append(t)
            
            # Flipped
            t_flip = A.Compose([
                A.Resize(height=scaled_size, width=scaled_size),
                A.HorizontalFlip(p=1.0),
            ] + t.transforms[1:])
            transforms.append(t_flip)
    
    return transforms


class TTADataset(torch.utils.data.Dataset):
    """Dataset that applies multiple TTA transforms to each image."""
    
    def __init__(self, base_dataset, tta_transforms: List[A.Compose]):
        self.base_dataset = base_dataset
        self.tta_transforms = tta_transforms
    
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx):
        img, label = self.base_dataset[idx]
        # img is already a tensor from base dataset
        # We need to convert back to numpy for TTA transforms
        if isinstance(img, torch.Tensor):
            img_np = img.permute(1, 2, 0).numpy()
            # Denormalize
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            img_np = (img_np * std + mean) * 255
            img_np = np.clip(img_np, 0, 255).astype(np.uint8)
        else:
            img_np = img
        
        # Apply all TTA transforms
        tta_imgs = []
        for t in self.tta_transforms:
            tta_imgs.append(t(image=img_np)["image"])
        
        return torch.stack(tta_imgs), label


def tta_predict(model: torch.nn.Module, images: torch.Tensor, device: torch.device,
                aggregation: str = "mean_logits") -> torch.Tensor:
    """
    Run TTA prediction on batch of images.
    
    Args:
        model: model in eval mode
        images: [B, TTA, C, H, W] or [TTA, C, H, W]
        device: torch device
        aggregation: "mean_logits", "mean_probs", "gmean_probs", "max_probs"
    
    Returns:
        [B, C] logits or probabilities
    """
    if images.dim() == 4:
        images = images.unsqueeze(0)  # [1, TTA, C, H, W]
    
    B, TTA, C, H, W = images.shape
    images = images.view(B * TTA, C, H, W).to(device, non_blocking=True)
    
    with torch.no_grad():
        logits = model(images)  # [B*TTA, C]
    
    logits = logits.view(B, TTA, -1)
    
    if aggregation == "mean_logits":
        return logits.mean(1)
    elif aggregation == "mean_probs":
        probs = F.softmax(logits, dim=-1)
        return probs.mean(1)
    elif aggregation == "gmean_probs":
        probs = F.softmax(logits, dim=-1)
        return probs.log().mean(1).exp()
    elif aggregation == "max_probs":
        probs = F.softmax(logits, dim=-1)
        return probs.max(1).values
    else:
        raise ValueError(f"Unknown aggregation: {aggregation}")


def tta_predict_batch(model: torch.nn.Module, loader: torch.utils.data.DataLoader,
                      device: torch.device, tta_transforms: List[A.Compose],
                      aggregation: str = "mean_logits") -> tuple:
    """
    Run TTA on entire dataset.
    Returns: (all_logits, all_labels, all_probs)
    """
    model.eval()
    all_logits = []
    all_labels = []
    
    base_ds = loader.dataset
    
    for batch_idx in range(len(base_ds)):
        if batch_idx % 50 == 0:
            print(f"  TTA progress: {batch_idx}/{len(base_ds)}")
        
        # Get TTA images for this sample
        img, label = base_ds[batch_idx]
        
        # Convert tensor back to numpy for TTA
        if isinstance(img, torch.Tensor):
            img_np = img.permute(1, 2, 0).numpy()
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            img_np = (img_np * std + mean) * 255
            img_np = np.clip(img_np, 0, 255).astype(np.uint8)
        else:
            img_np = img
        
        # Apply all TTA transforms
        tta_imgs = []
        for t in tta_transforms:
            tta_imgs.append(t(image=img_np)["image"])
        
        tta_batch = torch.stack(tta_imgs).to(device, non_blocking=True)
        
        with torch.no_grad():
            logits = model(tta_batch)
        
        if aggregation == "mean_logits":
            agg_logits = logits.mean(0)
        elif aggregation == "mean_probs":
            probs = F.softmax(logits, dim=-1)
            agg_logits = probs.mean(0).log()
        elif aggregation == "gmean_probs":
            probs = F.softmax(logits, dim=-1)
            agg_logits = probs.log().mean(0).exp().log()
        else:
            agg_logits = logits.mean(0)
        
        all_logits.append(agg_logits.cpu())
        all_labels.append(label)
    
    all_logits = torch.stack(all_logits)
    all_labels = torch.tensor(all_labels)
    all_probs = F.softmax(all_logits, dim=-1)
    
    return all_logits, all_labels, all_probs