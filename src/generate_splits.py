#!/usr/bin/env python3
"""
SCALE x ODYSSEY -- Generate Stratified Train/Val/Test Splits

Creates reproducible 80/10/10 stratified splits preserving class distribution.

Usage:
    python src/generate_splits.py
"""

import os
import json
import random
from pathlib import Path
from collections import defaultdict
from typing import Dict, List

import numpy as np
from PIL import Image
from sklearn.model_selection import StratifiedShuffleSplit

PROCESSED_DIR = Path("data/processed")
CLASS_NAMES = ["spiral_galaxy", "elliptical_galaxy", "nebula", "star_cluster", "planetary"]
SPLITS = {"train": 0.80, "val": 0.10, "test": 0.10}
SEED = 42


def collect_all_images() -> Dict[str, List[Path]]:
    """Collect all images from _train_source directory (before split)."""
    images_by_class = defaultdict(list)
    source_dir = PROCESSED_DIR / "_train_source"

    for class_name in CLASS_NAMES:
        class_dir = source_dir / class_name
        if not class_dir.exists():
            continue
        for img_path in class_dir.glob("*.jpg"):
            images_by_class[class_name].append(img_path)
        for img_path in class_dir.glob("*.png"):
            images_by_class[class_name].append(img_path)

    return dict(images_by_class)


def validate_image(img_path: Path) -> bool:
    """Check if image is valid."""
    try:
        with Image.open(img_path) as img:
            img.verify()
        return True
    except:
        return False


def create_stratified_splits(images_by_class: Dict[str, List[Path]]) -> Dict[str, Dict[str, List]]:
    """Create stratified splits using StratifiedShuffleSplit."""
    all_images = []
    all_labels = []

    for class_idx, (class_name, images) in enumerate(images_by_class.items()):
        for img_path in images:
            all_images.append(img_path)
            all_labels.append(class_idx)

    X = np.array([str(p) for p in all_images])
    y = np.array(all_labels)

    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.20, random_state=SEED)
    train_idx, temp_idx = next(splitter.split(X, y))

    temp_X = X[temp_idx]
    temp_y = y[temp_idx]

    val_size = 0.5
    splitter2 = StratifiedShuffleSplit(n_splits=1, test_size=val_size, random_state=SEED)
    val_idx, test_idx = next(splitter2.split(temp_X, temp_y))

    result = {
        "train": [Path(p) for p in X[train_idx]],
        "val": [Path(p) for p in X[temp_idx][val_idx]],
        "test": [Path(p) for p in X[temp_idx][test_idx]],
    }

    return result


def copy_to_splits(split_data: Dict[str, List[Path]]):
    """Copy images to proper split directories."""
    import shutil

    for split_name, images in split_data.items():
        split_dir = PROCESSED_DIR / split_name
        if split_dir.exists():
            shutil.rmtree(split_dir)
        split_dir.mkdir(parents=True, exist_ok=True)

        for img_path in images:
            class_name = img_path.parent.name
            class_dir = split_dir / class_name
            class_dir.mkdir(parents=True, exist_ok=True)
            dest = class_dir / img_path.name
            shutil.copy2(img_path, dest)


def generate_statistics(split_data: Dict[str, List[Path]]) -> Dict:
    """Generate split statistics."""
    stats = {}

    for split_name, images in split_data.items():
        class_counts = defaultdict(int)
        for img_path in images:
            class_counts[img_path.parent.name] += 1

        stats[split_name] = dict(class_counts)

    return stats


def main():
    import shutil
    print("=" * 60)
    print("SCALE x ODYSSEY -- Stratified Splits Generator")
    print("=" * 60)

    print("\nStep 0: Backing up source data...")
    source_dir = PROCESSED_DIR / "train"
    backup_dir = PROCESSED_DIR / "_train_source"
    if backup_dir.exists():
        shutil.rmtree(backup_dir)
    shutil.copytree(source_dir, backup_dir)

    print("\nStep 1: Collecting images...")
    images_by_class = collect_all_images()

    total = sum(len(imgs) for imgs in images_by_class.values())
    print(f"  Total images: {total}")
    for class_name, images in images_by_class.items():
        print(f"    {class_name}: {len(images)}")

    print("\nStep 2: Validating images...")
    valid_images = {}
    for class_name, images in images_by_class.items():
        valid = [img for img in images if validate_image(img)]
        removed = len(images) - len(valid)
        if removed > 0:
            print(f"  {class_name}: {len(valid)}/{len(images)} valid ({removed} removed)")
        valid_images[class_name] = valid
    images_by_class = valid_images

    print("\nStep 3: Creating stratified splits (80/10/10)...")
    split_data = create_stratified_splits(images_by_class)

    for split_name, images in split_data.items():
        print(f"  {split_name}: {len(images)} images")

    print("\nStep 4: Copying to split directories...")
    copy_to_splits(split_data)

    print("\nStep 5: Cleaning up backup...")
    if backup_dir.exists():
        shutil.rmtree(backup_dir)

    print("\nStep 6: Generating statistics...")
    stats = generate_statistics(split_data)

    stats_path = PROCESSED_DIR / "split_statistics.json"
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"  Saved to {stats_path}")

    print("\n" + json.dumps(stats, indent=4))
    print("\nStratified splits complete!")


if __name__ == "__main__":
    main()
