#!/usr/bin/env python3
"""
SCALE x ODYSSEY -- Dataset Download & Merge Script

Downloads astronomical image datasets from Kaggle, merges them into
a unified 5-class taxonomy, and creates 80/10/10 train/val/test splits.

Usage:
    cd scale_odyssey
    python src/download_datasets.py

Requirements:
    - Kaggle API credentials in ~/.kaggle/kaggle.json
    - conda activate scale_odyssey
    - ~5GB free disk space
"""

import os
import shutil
import zipfile
import random
import json
from pathlib import Path
from collections import defaultdict
from typing import Dict, List
import logging

import numpy as np
from PIL import Image
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = [
    "spiral_galaxy",
    "elliptical_galaxy",
    "nebula",
    "star_cluster",
    "planetary",
]

DATASETS = {
    "galaxy-zoo": {
        "kaggle_ref": "jaimetrickz/galaxy-classification",
        "subdirs": {
            "spiral_galaxy": ["spiral", "spiral_galaxy", "Spiral"],
            "elliptical_galaxy": ["elliptical", "elliptical_galaxy", "Elliptical"],
        },
    },
    "astronomy-classification": {
        "kaggle_ref": "fedesoriano/astronomy-dataset",
        "subdirs": {
            "nebula": ["nebula", "Nebula", "nebulae"],
            "star_cluster": ["star_cluster", "cluster", "StarCluster"],
        },
    },
    "deepsky": {
        "kaggle_ref": "zhangchHang/deepsky-dataset",
        "subdirs": {
            "nebula": ["nebula", "nebulae"],
            "star_cluster": ["cluster", "star_cluster", "globular_cluster"],
        },
    },
    "planetary": {
        "kaggle_ref": "brsdincer/planetary-solar-system-objects",
        "subdirs": {
            "planetary": ["planet", "planetary", "moon", "satellite"],
        },
    },
}

SPLITS = {"train": 0.80, "val": 0.10, "test": 0.10}


def download_dataset(name: str, kaggle_ref: str, force: bool = False) -> Path:
    """Download and extract a Kaggle dataset."""
    dataset_dir = RAW_DIR / name

    if dataset_dir.exists() and any(dataset_dir.iterdir()) and not force:
        logger.info(f"Dataset '{name}' already downloaded. Skipping.")
        return dataset_dir

    logger.info(f"Downloading '{name}' from Kaggle...")
    dataset_dir.mkdir(parents=True, exist_ok=True)

    cmd = f"kaggle datasets download -d {kaggle_ref} -p {RAW_DIR} -q"
    ret = os.system(cmd)
    if ret != 0:
        logger.warning(f"Failed to download {name}. You may need to download it manually.")
        logger.warning(f"  Dataset: {kaggle_ref}")
        logger.warning(f"  Save to: {dataset_dir}")
        return dataset_dir

    zip_files = list(RAW_DIR.glob("*.zip"))
    for zf in zip_files:
        try:
            logger.info(f"Extracting {zf.name}...")
            with zipfile.ZipFile(zf, 'r') as z:
                z.extractall(dataset_dir)
            zf.unlink()
        except (zipfile.BadZipFile, Exception) as e:
            logger.warning(f"Could not extract {zf.name}: {e}")
            continue

    logger.info(f"Dataset '{name}' ready at {dataset_dir}")
    return dataset_dir


def discover_images(dataset_dir: Path, class_patterns: Dict[str, List[str]]) -> Dict[str, List[Path]]:
    """Discover images and map them to unified class names."""
    images_by_class = defaultdict(list)
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
    all_images = []

    for ext in image_extensions:
        all_images.extend(dataset_dir.rglob(f"*{ext}"))
        all_images.extend(dataset_dir.rglob(f"*{ext.upper()}"))

    logger.info(f"  Found {len(all_images)} image files in {dataset_dir.name}")

    for img_path in all_images:
        path_parts = [p.lower() for p in img_path.parts]
        file_stem = img_path.stem.lower()
        matched = False

        for class_name, patterns in class_patterns.items():
            patterns_lower = [p.lower() for p in patterns]

            if any(p in path_parts for p in patterns_lower):
                images_by_class[class_name].append(img_path)
                matched = True
                break

            if any(p in file_stem for p in patterns_lower):
                images_by_class[class_name].append(img_path)
                matched = True
                break

        if not matched:
            if any(kw in file_stem for kw in ['spiral', 'arm', 'disk']):
                images_by_class['spiral_galaxy'].append(img_path)
            elif any(kw in file_stem for kw in ['elliptical', 'ellipse', 'oval', 'eso']):
                images_by_class['elliptical_galaxy'].append(img_path)
            elif any(kw in file_stem for kw in ['nebula', 'nebulae', 'gas', 'cloud']):
                images_by_class['nebula'].append(img_path)
            elif any(kw in file_stem for kw in ['cluster', 'globular', 'open_cluster']):
                images_by_class['star_cluster'].append(img_path)
            elif any(kw in file_stem for kw in ['planet', 'mars', 'jupiter', 'saturn', 'lunar', 'moon']):
                images_by_class['planetary'].append(img_path)

    return dict(images_by_class)


def merge_all_datasets() -> Dict[str, List[Path]]:
    """Discover and merge images from all downloaded datasets."""
    logger.info("=" * 60)
    logger.info("STEP 2: Discovering and classifying images")
    logger.info("=" * 60)

    merged = defaultdict(list)

    for name, config in DATASETS.items():
        dataset_dir = RAW_DIR / name
        if not dataset_dir.exists():
            logger.warning(f"Dataset dir {dataset_dir} not found, skipping.")
            continue

        discovered = discover_images(dataset_dir, config["subdirs"])
        for class_name, images in discovered.items():
            merged[class_name].extend(images)
            logger.info(f"  {name}/{class_name}: {len(images)} images")

    logger.info("\n--- Merged Dataset Summary ---")
    total = 0
    for class_name in CLASS_NAMES:
        count = len(merged.get(class_name, []))
        total += count
        logger.info(f"  {class_name:25s}: {count:6d} images")
    logger.info(f"  {'TOTAL':25s}: {total:6d} images\n")

    return dict(merged)


def validate_image(img_path: Path, min_size: int = 64) -> bool:
    """Validate that an image is loadable and meets minimum quality."""
    try:
        with Image.open(img_path) as img:
            if img.width < min_size or img.height < min_size:
                return False
            img_array = np.array(img)
            if img_array.max() == img_array.min():
                return False
            return True
    except Exception:
        return False


def filter_dataset(images_by_class: Dict[str, List[Path]]) -> Dict[str, List[Path]]:
    """Filter out corrupted and low-quality images."""
    logger.info("=" * 60)
    logger.info("STEP 3: Validating image quality")
    logger.info("=" * 60)

    filtered = {}
    for class_name, images in images_by_class.items():
        valid = []
        for img_path in tqdm(images, desc=f"  Validating {class_name}", leave=False):
            if validate_image(img_path):
                valid.append(img_path)

        removed = len(images) - len(valid)
        logger.info(f"  {class_name}: {len(valid)}/{len(images)} valid ({removed} removed)")
        filtered[class_name] = valid

    return filtered


def handle_imbalance(images_by_class: Dict[str, List[Path]]) -> Dict[str, List[Path]]:
    """Address class imbalance through oversampling."""
    logger.info("=" * 60)
    logger.info("STEP 4: Handling class imbalance")
    logger.info("=" * 60)

    counts = {cls: len(imgs) for cls, imgs in images_by_class.items()}
    median_count = int(np.median(list(counts.values())))

    logger.info(f"  Median class size: {median_count}")

    balanced = {}
    for class_name, images in images_by_class.items():
        target = max(median_count, int(counts[class_name] * 1.5))
        target = min(target, median_count * 2)

        if len(images) < target:
            oversampled = images.copy()
            while len(oversampled) < target:
                oversampled.extend(
                    random.sample(images, min(target - len(oversampled), len(images)))
                )
            balanced[class_name] = oversampled
            logger.info(f"    {class_name}: {len(images)} -> {len(oversampled)} (oversampled)")
        else:
            balanced[class_name] = images
            logger.info(f"    {class_name}: {len(images)} (no change)")

    weights = compute_class_weights(balanced)
    weights_path = PROCESSED_DIR / "class_weights.json"
    with open(weights_path, 'w') as f:
        json.dump(weights, f, indent=2)
    logger.info(f"\n  Class weights saved to {weights_path}")

    return balanced


def compute_class_weights(images_by_class: Dict[str, List[Path]]) -> Dict[str, float]:
    """Compute inverse-frequency weights for class imbalance."""
    counts = {cls: len(imgs) for cls, imgs in images_by_class.items()}
    total = sum(counts.values())

    weights = {
        cls: total / (len(counts) * count) if count > 0 else 0
        for cls, count in counts.items()
    }

    max_w = max(weights.values()) if weights else 1.0
    weights = {cls: min(w / max_w * 5.0, 5.0) for cls, w in weights.items()}

    return weights


def create_splits(images_by_class: Dict[str, List[Path]], splits: Dict[str, float]):
    """Create stratified train/val/test splits."""
    logger.info("=" * 60)
    logger.info("STEP 5: Creating train/val/test splits (80/10/10)")
    logger.info("=" * 60)

    result = {split: {cls: [] for cls in CLASS_NAMES} for split in splits.keys()}

    for class_name, images in images_by_class.items():
        shuffled = images.copy()
        random.shuffle(shuffled)

        n = len(shuffled)
        n_train = int(n * splits["train"])
        n_val = int(n * splits["val"])

        result["train"][class_name] = shuffled[:n_train]
        result["val"][class_name] = shuffled[n_train:n_train + n_val]
        result["test"][class_name] = shuffled[n_train + n_val:]

        logger.info(f"  {class_name}:")
        logger.info(f"    Train: {len(result['train'][class_name])}")
        logger.info(f"    Val:   {len(result['val'][class_name])}")
        logger.info(f"    Test:  {len(result['test'][class_name])}")

    logger.info()
    return result


def copy_to_processed(split_data):
    """Copy images to the processed directory structure."""
    logger.info("=" * 60)
    logger.info("STEP 6: Copying to processed directory")
    logger.info("=" * 60)

    if PROCESSED_DIR.exists():
        shutil.rmtree(PROCESSED_DIR)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    for split_name, classes in split_data.items():
        for class_name, images in classes.items():
            dest_dir = PROCESSED_DIR / split_name / class_name
            dest_dir.mkdir(parents=True, exist_ok=True)

            for img_path in tqdm(images, desc=f"  {split_name}/{class_name}", leave=False):
                dest_name = f"{img_path.stem}_{img_path.stat().st_ino}{img_path.suffix}"
                shutil.copy2(img_path, dest_dir / dest_name)

    stats = {}
    for split_name in ["train", "val", "test"]:
        split_dir = PROCESSED_DIR / split_name
        if split_dir.exists():
            stats[split_name] = {}
            for class_dir in split_dir.iterdir():
                if class_dir.is_dir():
                    count = len(list(class_dir.glob("*")))
                    stats[split_name][class_dir.name] = count

    stats_path = PROCESSED_DIR / "split_statistics.json"
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)

    logger.info(f"  Statistics saved to {stats_path}")
    logger.info(f"\n  {json.dumps(stats, indent=4)}")
    logger.info("\nDataset preparation COMPLETE!")
    logger.info(f"Processed data location: {PROCESSED_DIR.resolve()}")


def main():
    logger.info("=" * 60)
    logger.info("SCALE x ODYSSEY -- Dataset Preparation Pipeline")
    logger.info("=" * 60)

    logger.info("Step 1: Downloading datasets from Kaggle")
    logger.info("(If downloads fail, manually download and extract to data/raw/)")

    for name, config in DATASETS.items():
        try:
            download_dataset(name, config["kaggle_ref"], force=False)
        except Exception as e:
            logger.warning(f"Download issue with {name}: {e}")

    merged = merge_all_datasets()
    filtered = filter_dataset(merged)
    balanced = handle_imbalance(filtered)
    split_data = create_splits(balanced, SPLITS)
    copy_to_processed(split_data)

    logger.info("\nNext step: Run training!")
    logger.info("  python src/train.py")


if __name__ == "__main__":
    main()
