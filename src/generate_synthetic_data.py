#!/usr/bin/env python3
"""Generate synthetic astronomical images for quick testing."""
import os
import random
import numpy as np
from PIL import Image

CLASSES = ["spiral_galaxy", "elliptical_galaxy", "nebula", "star_cluster", "planetary"]

def generate_image(size=224, class_idx=0):
    """Generate a synthetic image with class-specific characteristics."""
    img = np.zeros((size, size, 3), dtype=np.uint8)
    
    if class_idx == 0:  # spiral galaxy - spiral pattern
        cx, cy = size // 2, size // 2
        for i in range(size):
            for j in range(size):
                dx, dy = j - cx, i - cy
                r = np.sqrt(dx*dx + dy*dy) + 1e-5
                theta = np.arctan2(dy, dx)
                spiral = np.sin(theta * 3 + r * 0.1)
                val = int(128 + 100 * spiral * np.exp(-r / (size * 0.4)))
                img[i, j] = [val, val, val + 20]
    elif class_idx == 1:  # elliptical galaxy - smooth oval
        cx, cy = size // 2, size // 2
        for i in range(size):
            for j in range(size):
                dx, dy = (j - cx) / (size * 0.4), (i - cy) / (size * 0.3)
                r = np.sqrt(dx*dx + dy*dy)
                val = int(180 * np.exp(-r*r))
                img[i, j] = [val, val, val]
    elif class_idx == 2:  # nebula - colorful clouds
        noise = np.random.randint(0, 80, (size, size, 3), dtype=np.uint8)
        for i in range(size):
            for j in range(size):
                val = int(100 + 80 * np.sin(i * 0.05) * np.cos(j * 0.05))
                img[i, j] = [val, val // 2, 255 - val]
        img = np.clip(img.astype(int) + noise, 0, 255).astype(np.uint8)
    elif class_idx == 3:  # star cluster - bright dots
        img = np.random.randint(0, 30, (size, size, 3), dtype=np.uint8)
        for _ in range(50):
            x, y = random.randint(0, size-1), random.randint(0, size-1)
            brightness = random.randint(150, 255)
            cv = min(size-1, max(0, x+2))
            rv = min(size-1, max(0, y+2))
            img[y, x] = [brightness, brightness, brightness]
            if y+1 < size: img[y+1, x] = [brightness//2, brightness//2, brightness//2]
            if x+1 < size: img[y, x+1] = [brightness//2, brightness//2, brightness//2]
    else:  # planetary - planet-like circle
        cx, cy = size // 2, size // 2
        radius = size // 3
        for i in range(size):
            for j in range(size):
                dx, dy = j - cx, i - cy
                r = np.sqrt(dx*dx + dy*dy)
                if r < radius:
                    val = int(200 - 100 * (r / radius))
                    img[i, j] = [val, val//2, val//3]
                else:
                    img[i, j] = [10, 10, 30]
    
    # Add universal noise
    noise = np.random.randint(-10, 10, img.shape, dtype=np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(img)


def generate_dataset(output_dir="data/processed", samples_per_class=50):
    """Generate synthetic dataset with train/val/test splits."""
    splits = {"train": 0.8, "val": 0.1, "test": 0.1}
    
    for split, ratio in splits.items():
        split_dir = os.path.join(output_dir, split)
        os.makedirs(split_dir, exist_ok=True)
        n = int(samples_per_class * ratio)
        
        for class_idx, class_name in enumerate(CLASSES):
            class_dir = os.path.join(split_dir, class_name)
            os.makedirs(class_dir, exist_ok=True)
            
            for i in range(n):
                img = generate_image(class_idx=class_idx)
                img.save(os.path.join(class_dir, f"{class_name}_{i:04d}.jpg"))
    
    # Create class weights
    import json
    weights = {c: 1.0 for c in CLASSES}
    with open(os.path.join(output_dir, "class_weights.json"), "w") as f:
        json.dump(weights, f)
    
    print(f"Synthetic dataset generated at {output_dir}")
    print(f"  Classes: {CLASSES}")
    print(f"  Samples per class: {samples_per_class}")


if __name__ == "__main__":
    generate_dataset()
