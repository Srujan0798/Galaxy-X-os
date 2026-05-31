#!/usr/bin/env python3
"""
SCALE x ODYSSEY -- src/dataset.py
PyTorch Dataset + Albumentations Augmentation Pipeline

Matches official starter guide structure (page 8-9) with our
astronomy-specific transforms and strong Albumentations pipeline.
"""

import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

# ---------------------------------------------------------------------------
# Class Definitions
# ---------------------------------------------------------------------------

CLASSES = [
    "spiral_galaxy",
    "elliptical_galaxy",
    "nebula",
    "star_cluster",
    "planetary",
]

CLASS_NAMES_DISPLAY = [
    "Spiral Galaxy",
    "Elliptical Galaxy",
    "Nebula",
    "Star Cluster",
    "Planetary Object",
]

NUM_CLASSES = len(CLASSES)


# ---------------------------------------------------------------------------
# Astronomical-Specific Augmentations
# ---------------------------------------------------------------------------

class AddAstroNoise(A.ImageOnlyTransform):
    """Add realistic astronomical noise: Gaussian + Poisson."""

    def __init__(self, gaussian_sigma_limit=(5, 25),
                 poisson_scale_limit=(10, 50), always_apply=False, p=0.4):
        super().__init__(p=p)
        self.gaussian_sigma_limit = gaussian_sigma_limit
        self.poisson_scale_limit = poisson_scale_limit

    def apply(self, img, **params):
        import random
        gaussian_sigma = random.uniform(*self.gaussian_sigma_limit)
        poisson_scale = random.uniform(*self.poisson_scale_limit)
        gaussian_noise = np.random.normal(0, gaussian_sigma, img.shape)
        poisson_noise = np.random.poisson(img / 255.0 * poisson_scale) * (255.0 / poisson_scale) - img
        noisy = img.astype(np.float32) + gaussian_noise + poisson_noise * 0.3
        return np.clip(noisy, 0, 255).astype(np.uint8)

    def get_transform_init_args_names(self):
        return ("gaussian_sigma_limit", "poisson_scale_limit")


class SimulateCosmicRay(A.ImageOnlyTransform):
    """Simulate cosmic ray hits -- bright streaks across the image."""

    def __init__(self, num_streaks_limit=(1, 4), brightness_limit=(200, 255),
                 width_limit=(1, 3), always_apply=False, p=0.15):
        super().__init__(p=p)
        self.num_streaks_limit = num_streaks_limit
        self.brightness_limit = brightness_limit
        self.width_limit = width_limit

    def apply(self, img, **params):
        import random
        import cv2
        result = img.copy()
        h, w = img.shape[:2]
        for _ in range(random.randint(*self.num_streaks_limit)):
            x1, y1 = random.randint(0, w-1), random.randint(0, h-1)
            angle = random.uniform(0, np.pi)
            length = random.randint(h//8, h//2)
            x2, y2 = int(x1 + length * np.cos(angle)), int(y1 + length * np.sin(angle))
            brightness = random.randint(*self.brightness_limit)
            thickness = random.randint(*self.width_limit)
            color = (brightness,) * 3 if img.ndim == 3 else brightness
            cv2.line(result, (x1, y1), (x2, y2), color, thickness)
        return result

    def get_transform_init_args_names(self):
        return ("num_streaks_limit", "brightness_limit", "width_limit")


class SimulateVignetting(A.ImageOnlyTransform):
    """Simulate telescope vignetting -- darker edges."""

    def __init__(self, strength_limit=(0.1, 0.4), always_apply=False, p=0.2):
        super().__init__(p=p)
        self.strength_limit = strength_limit

    def apply(self, img, **params):
        import random
        h, w = img.shape[:2]
        cx, cy = w / 2, h / 2
        Y, X = np.ogrid[:h, :w]
        dist = np.sqrt((X - cx)**2 + (Y - cy)**2)
        max_dist = np.sqrt(cx**2 + cy**2)
        strength = random.uniform(*self.strength_limit)
        vignette = np.clip(1 - strength * (dist / max_dist), 0.3, 1.0)
        if img.ndim == 3:
            vignette = vignette[:, :, np.newaxis]
        return np.clip(img.astype(np.float32) * vignette, 0, 255).astype(np.uint8)

    def get_transform_init_args_names(self):
        return ("strength_limit",)


# ---------------------------------------------------------------------------
# Augmentation Pipelines
# ---------------------------------------------------------------------------

def get_train_transforms(image_size: int = 224) -> A.Compose:
    """Heavy augmentation pipeline for training."""
    import cv2
    return A.Compose([
        A.RandomRotate90(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        A.RandomResizedCrop(size=(image_size, image_size),
                            scale=(0.5, 1.0), ratio=(0.75, 1.33),
                            interpolation=cv2.INTER_LANCZOS4, p=0.6),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.15, rotate_limit=45,
                           border_mode=cv2.BORDER_CONSTANT, fill=0, p=0.5),
        A.RandomBrightnessContrast(brightness_limit=(-0.3, 0.3),
                                   contrast_limit=(-0.3, 0.3), p=0.6),
        A.RandomGamma(gamma_limit=(70, 130), p=0.3),
        A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=30,
                             val_shift_limit=20, p=0.3),
        AddAstroNoise(p=0.4),
        SimulateCosmicRay(p=0.15),
        SimulateVignetting(p=0.2),
        A.CoarseDropout(num_holes_range=(2, 12),
                        hole_height_range=(4, 4),
                        hole_width_range=(4, 4),
                        fill=0, p=0.3),
        A.MotionBlur(blur_limit=5, p=0.15),
        A.GaussianBlur(blur_limit=3, sigma_limit=1.0, p=0.15),
        A.Resize(height=image_size, width=image_size, p=1.0),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ], p=1.0)


def get_val_transforms(image_size: int = 224) -> A.Compose:
    """Minimal augmentation for validation/testing."""
    return A.Compose([
        A.Resize(height=image_size, width=image_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class AstroDataset(Dataset):
    """PyTorch Dataset for astronomical images."""

    def __init__(self, root_dir: str, split: str = "train", image_size: int = 224):
        self.root_dir = root_dir
        self.split = split
        self.image_size = image_size
        self.transform = get_train_transforms(image_size) if split == "train" else get_val_transforms(image_size)

        self.samples = []
        self.labels = []
        split_dir = os.path.join(root_dir, split)

        for class_idx, class_name in enumerate(CLASSES):
            class_dir = os.path.join(split_dir, class_name)
            if not os.path.exists(class_dir):
                continue
            for fname in sorted(os.listdir(class_dir)):
                if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                    self.samples.append(os.path.join(class_dir, fname))
                    self.labels.append(class_idx)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img = np.array(Image.open(self.samples[idx]).convert("RGB"))
        augmented = self.transform(image=img)["image"]
        return augmented, self.labels[idx]


# ---------------------------------------------------------------------------
# DataLoaders
# ---------------------------------------------------------------------------

def get_loaders(data_dir: str, batch_size: int = 32, num_workers: int = 4, image_size: int = 224):
    """Create train/val/test DataLoaders."""
    train_ds = AstroDataset(data_dir, "train", image_size)
    val_ds = AstroDataset(data_dir, "val", image_size)
    test_ds = AstroDataset(data_dir, "test", image_size)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)

    print(f"Dataset sizes -- Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")
    return train_loader, val_loader, test_loader
