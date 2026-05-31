#!/usr/bin/env python3
"""
SCALE x ODYSSEY -- Preprocessing Runner

Applies AstroPreprocessor pipeline to all images in data/processed.
Creates before/after comparison samples.

Usage:
    python src/run_preprocessing.py
"""

import os
import json
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List

import numpy as np
from PIL import Image
from tqdm import tqdm

from preprocess import AstroPreprocessor

PROCESSED_DIR = Path("data/processed")
OUTPUT_DIR = PROCESSED_DIR / "preprocessed"
SAMPLES_DIR = PROCESSED_DIR / "preprocess_samples"
CLASS_NAMES = ["spiral_galaxy", "elliptical_galaxy", "nebula", "star_cluster", "planetary"]
SPLITS = ["train", "val", "test"]


def preprocess_image(preprocessor: AstroPreprocessor, img_path: Path, output_path: Path) -> bool:
    """Preprocess a single image."""
    try:
        img = np.array(Image.open(img_path).convert("RGB"))
        processed = preprocessor.preprocess(img)
        Image.fromarray(processed).save(output_path)
        return True
    except Exception as e:
        return False


def preprocess_split(split: str, preprocessor: AstroPreprocessor, max_workers: int = 8) -> Dict:
    """Preprocess all images in a split."""
    split_input = PROCESSED_DIR / split
    split_output = OUTPUT_DIR / split

    stats = {"total": 0, "success": 0, "failed": 0, "by_class": {}}

    for class_name in CLASS_NAMES:
        class_input = split_input / class_name
        class_output = split_output / class_name
        class_output.mkdir(parents=True, exist_ok=True)

        if not class_input.exists():
            continue

        images = list(class_input.glob("*.jpg")) + list(class_input.glob("*.png"))
        stats["by_class"][class_name] = {"total": len(images), "success": 0, "failed": 0}
        stats["total"] += len(images)

        tasks = []
        for img_path in images:
            output_path = class_output / img_path.name
            tasks.append((img_path, output_path))

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(preprocess_image, preprocessor, inp, out): (inp, out)
                for inp, out in tasks
            }
            for future in tqdm(as_completed(futures), total=len(tasks), desc=f"  {split}/{class_name}", leave=False):
                success = future.result()
                class_name_for_stats = img_path.parent.name
                if success:
                    stats["success"] += 1
                    stats["by_class"][class_name]["success"] += 1
                else:
                    stats["failed"] += 1
                    stats["by_class"][class_name]["failed"] += 1

    return stats


def create_comparison_samples(preprocessor: AstroPreprocessor, num_samples: int = 5):
    """Create before/after comparison samples."""
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)

    for class_name in CLASS_NAMES[:num_samples]:
        class_dir = PROCESSED_DIR / "train" / class_name
        if not class_dir.exists():
            continue

        images = list(class_dir.glob("*.jpg"))[:1]
        if not images:
            continue

        img_path = images[0]
        img = np.array(Image.open(img_path).convert("RGB"))
        processed = preprocessor.preprocess(img)

        before_path = SAMPLES_DIR / f"{class_name}_before.jpg"
        after_path = SAMPLES_DIR / f"{class_name}_after.jpg"

        Image.fromarray(img).save(before_path)
        Image.fromarray(processed).save(after_path)


def main():
    print("=" * 60)
    print("SCALE x ODYSSEY -- Preprocessing Pipeline")
    print("=" * 60)

    preprocessor = AstroPreprocessor(target_size=(224, 224))

    print("\nStep 1: Creating before/after comparison samples...")
    create_comparison_samples(preprocessor, num_samples=5)
    print(f"  Samples saved to {SAMPLES_DIR}")

    start_time = time.time()

    print("\nStep 2: Preprocessing all splits...")
    all_stats = {}
    for split in SPLITS:
        print(f"\n  Processing {split}...")
        stats = preprocess_split(split, preprocessor)
        all_stats[split] = stats
        print(f"    {stats['success']}/{stats['total']} images preprocessed successfully")

    elapsed = time.time() - start_time

    print("\n" + "=" * 60)
    print("Preprocessing Complete!")
    print("=" * 60)
    print(f"  Total time: {elapsed:.1f}s")
    print(f"  Output directory: {OUTPUT_DIR}")
    print(f"  Sample comparisons: {SAMPLES_DIR}")

    stats_path = PROCESSED_DIR / "preprocessing_stats.json"
    with open(stats_path, 'w') as f:
        json.dump(all_stats, f, indent=2)
    print(f"  Statistics: {stats_path}")


if __name__ == "__main__":
    main()
