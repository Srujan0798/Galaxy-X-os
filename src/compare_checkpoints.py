#!/usr/bin/env python3
"""
SCALE x ODYSSEY -- Checkpoint Comparison

Evaluates all checkpoints and selects the best one.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import json
from src.model import AstroClassifier
from src.utils import load_config, load_checkpoint
from src.dataset import CLASSES

CHECKPOINT_DIR = Path("checkpoints")
DEVICE = torch.device("cpu")


def compare_checkpoints():
    config = load_config()
    checkpoints = sorted(CHECKPOINT_DIR.glob("*.pth"))

    results = []

    for ckpt_path in checkpoints:
        try:
            checkpoint = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
            val_acc = checkpoint.get("best_val_acc", 0.0)
            epoch = checkpoint.get("epoch", -1) + 1
            results.append({
                "file": ckpt_path.name,
                "epoch": epoch,
                "val_accuracy": val_acc,
                "path": str(ckpt_path)
            })
        except Exception as e:
            print(f"Error loading {ckpt_path.name}: {e}")

    results.sort(key=lambda x: x["val_accuracy"], reverse=True)

    print("=" * 60)
    print("Checkpoint Comparison")
    print("=" * 60)
    for r in results:
        print(f"  {r['file']:30s} | Epoch {r['epoch']:2d} | Val Acc: {r['val_accuracy']:.4f}")

    best = results[0] if results else None
    if best:
        print(f"\nBest checkpoint: {best['file']} (Epoch {best['epoch']}, Val Acc: {best['val_accuracy']:.4f})")

        best_ckpt = CHECKPOINT_DIR / "best_model.pth"
        if best["path"] != str(best_ckpt):
            import shutil
            shutil.copy2(best["path"], best_ckpt)
            print(f"Copied to {best_ckpt}")

        print(f"\nVerifying checkpoint loads correctly...")
        model = AstroClassifier(
            num_classes=config["model"]["num_classes"],
            backbone=config["model"]["backbone"],
            pretrained=False,
            dropout=config["model"]["dropout"]
        ).to(DEVICE)
        load_checkpoint(str(best_ckpt), model, DEVICE)
        print("  Checkpoint verified successfully!")

    return results


def main():
    results = compare_checkpoints()

    output_path = Path("results/checkpoint_comparison.json")
    output_path.parent.mkdir(exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved comparison to {output_path}")


if __name__ == "__main__":
    main()
