#!/usr/bin/env python3
"""
SCALE x ODYSSEY -- Class Weights Computation + Dataset Validation

Computes inverse-frequency class weights and validates the entire dataset.

Usage:
    python src/validate_dataset.py
"""

import json
import time
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
from PIL import Image

PROCESSED_DIR = Path("data/processed")
CLASS_NAMES = ["spiral_galaxy", "elliptical_galaxy", "nebula", "star_cluster", "planetary"]
SPLITS = ["train", "val", "test"]


def validate_image(img_path: Path) -> Tuple[bool, str]:
    """Validate a single image. Returns (is_valid, error_message)."""
    try:
        with Image.open(img_path) as img:
            img.verify()

        with Image.open(img_path) as img:
            img.load()

        img_array = np.array(img)
        if img_array.size == 0:
            return False, "Empty image"
        if img_array.ndim not in [2, 3]:
            return False, f"Invalid dimensions: {img_array.ndim}"

        if img_array.max() == img_array.min():
            return False, "Image is constant"

        return True, ""
    except Exception as e:
        return False, str(e)


def count_images_in_split(split: str) -> Dict[str, Dict]:
    """Count images per class in a split."""
    split_dir = PROCESSED_DIR / split
    counts = {}

    for class_name in CLASS_NAMES:
        class_dir = split_dir / class_name
        if not class_dir.exists():
            counts[class_name] = {"total": 0, "valid": 0, "corrupted": 0, "size": None}
            continue

        images = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.png"))
        valid_count = 0
        corrupted = 0
        size = None

        for img_path in images:
            is_valid, _ = validate_image(img_path)
            if is_valid:
                valid_count += 1
                if size is None:
                    try:
                        with Image.open(img_path) as img:
                            size = img.size
                    except:
                        pass
            else:
                corrupted += 1

        counts[class_name] = {
            "total": len(images),
            "valid": valid_count,
            "corrupted": corrupted,
            "size": size,
        }

    return counts


def compute_class_weights(counts: Dict[str, Dict]) -> Dict[str, float]:
    """Compute inverse-frequency class weights."""
    total_valid = sum(c["valid"] for c in counts.values())
    num_classes = len(counts)

    weights = {}
    for class_name, data in counts.items():
        if data["valid"] > 0:
            weights[class_name] = total_valid / (num_classes * data["valid"])
        else:
            weights[class_name] = 1.0

    max_weight = max(weights.values()) if weights else 1.0
    if max_weight > 0:
        weights = {k: v / max_weight * 5.0 for k, v in weights.items()}

    weights = {k: min(v, 5.0) for k, v in weights.items()}

    return weights


def validate_with_dataset():
    """Validate that AstroDataset can load the data."""
    from dataset import AstroDataset

    errors = []
    for split in SPLITS:
        try:
            ds = AstroDataset(str(PROCESSED_DIR), split=split, image_size=224)
            if len(ds) == 0:
                errors.append(f"{split}: AstroDataset returned 0 samples")
            else:
                sample, label = ds[0]
                print(f"  {split}: {len(ds)} samples, sample shape={sample.shape}")
        except Exception as e:
            errors.append(f"{split}: {str(e)}")

    return errors


def main():
    print("=" * 60)
    print("SCALE x ODYSSEY -- Dataset Validation + Class Weights")
    print("=" * 60)

    print("\nStep 1: Counting images per split/class...")
    all_counts = {}
    for split in SPLITS:
        print(f"\n  {split}:")
        counts = count_images_in_split(split)
        all_counts[split] = counts
        for class_name, data in counts.items():
            if data["total"] > 0:
                print(f"    {class_name}: {data['valid']}/{data['total']} valid", end="")
                if data["size"]:
                    print(f" @ {data['size']}", end="")
                if data["corrupted"] > 0:
                    print(f" [{data['corrupted']} corrupted]", end="")
                print()

    print("\nStep 2: Computing class weights...")
    train_counts = all_counts.get("train", {})
    weights = compute_class_weights(train_counts)

    weights_path = PROCESSED_DIR / "class_weights.json"
    with open(weights_path, 'w') as f:
        json.dump(weights, f, indent=2)
    print(f"  Saved to {weights_path}")
    print(f"\n  Weights: {json.dumps(weights, indent=4)}")

    print("\nStep 3: Validating AstroDataset loading...")
    dataset_errors = validate_with_dataset()
    if dataset_errors:
        for err in dataset_errors:
            print(f"  ERROR: {err}")
    else:
        print("  All splits load successfully!")

    print("\nStep 4: Generating validation report...")
    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "class_names": CLASS_NAMES,
        "splits": all_counts,
        "class_weights": weights,
        "dataset_errors": dataset_errors,
    }

    report_path = PROCESSED_DIR / "validation_report.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"  Report saved to {report_path}")

    total_valid = sum(
        counts["valid"]
        for split_counts in all_counts.values()
        for counts in split_counts.values()
    )
    total_corrupted = sum(
        counts["corrupted"]
        for split_counts in all_counts.values()
        for counts in split_counts.values()
    )

    print("\n" + "=" * 60)
    print("Validation Summary")
    print("=" * 60)
    print(f"  Total valid images: {total_valid}")
    print(f"  Total corrupted: {total_corrupted}")
    print("  Class weights: computed")
    print(f"  Dataset loading: {'PASS' if not dataset_errors else 'FAIL'}")


if __name__ == "__main__":
    main()
