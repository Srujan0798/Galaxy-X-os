#!/usr/bin/env python3
"""
SCALE x ODYSSEY -- Training Visualization

Generates training curves from log files.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import re
import numpy as np
import matplotlib.pyplot as plt

LOGS_DIR = Path("results/logs")
OUTPUT_DIR = Path("results")


def extract_from_log_file(log_path: Path):
    """Extract training data from log file."""
    data = {
        "epochs": [],
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "val_f1": [],
        "lr": [],
    }

    pattern = r"\[Epoch (\d+)/50\] .* LR: ([\d.e-]+) \| Train Loss: ([\d.]+) Acc: ([\d.]+) \| Val Loss: ([\d.]+) Acc: ([\d.]+) F1: ([\d.]+)"

    with open(log_path) as f:
        for line in f:
            match = re.search(pattern, line)
            if match:
                epoch, lr, train_loss, train_acc, val_loss, val_acc, val_f1 = match.groups()
                data["epochs"].append(int(epoch))
                data["lr"].append(float(lr))
                data["train_loss"].append(float(train_loss))
                data["train_acc"].append(float(train_acc))
                data["val_loss"].append(float(val_loss))
                data["val_acc"].append(float(val_acc))
                data["val_f1"].append(float(val_f1))

    return data


def plot_training_curves(data, output_path: Path):
    """Generate and save training curves."""
    if not data["epochs"]:
        print("No training data found in logs")
        return

    epochs = data["epochs"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(epochs, data["train_loss"], label="Train Loss", linewidth=2, marker="o")
    axes[0].plot(epochs, data["val_loss"], label="Val Loss", linewidth=2, marker="s")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training and Validation Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, data["train_acc"], label="Train Acc", linewidth=2, marker="o")
    axes[1].plot(epochs, data["val_acc"], label="Val Acc", linewidth=2, marker="s")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Training and Validation Accuracy")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved training curves to {output_path}")
    plt.close()


def plot_lr_schedule(data, output_path: Path):
    """Generate and save learning rate schedule."""
    if not data["lr"]:
        print("No LR data found in logs")
        return

    epochs = data["epochs"]

    plt.figure(figsize=(10, 5))
    plt.plot(epochs, data["lr"], linewidth=2, marker="o", color="green")
    plt.xlabel("Epoch")
    plt.ylabel("Learning Rate")
    plt.title("Learning Rate Schedule")
    plt.grid(True, alpha=0.3)
    plt.yscale("log")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved LR schedule to {output_path}")
    plt.close()


def main():
    print("=" * 60)
    print("SCALE x ODYSSEY -- Training Visualization")
    print("=" * 60)

    log_files = sorted(LOGS_DIR.glob("*.log"), key=lambda x: x.stat().st_mtime, reverse=True)
    if not log_files:
        print("No log files found")
        return

    log_file = log_files[0]
    print(f"Using log file: {log_file.name}")

    data = extract_from_log_file(log_file)

    if not data["epochs"]:
        print("No epoch data found in log file")
        return

    print(f"Found data for {len(data['epochs'])} epochs")

    curves_path = OUTPUT_DIR / "training_curves.png"
    plot_training_curves(data, curves_path)

    lr_path = OUTPUT_DIR / "lr_schedule.png"
    plot_lr_schedule(data, lr_path)

    metrics_summary = {
        "epochs_trained": data["epochs"][-1] if data["epochs"] else 0,
        "final_train_loss": data["train_loss"][-1] if data["train_loss"] else None,
        "final_val_loss": data["val_loss"][-1] if data["val_loss"] else None,
        "final_train_acc": data["train_acc"][-1] if data["train_acc"] else None,
        "final_val_acc": data["val_acc"][-1] if data["val_acc"] else None,
        "best_val_acc": max(data["val_acc"]) if data["val_acc"] else None,
        "best_epoch": data["epochs"][data["val_acc"].index(max(data["val_acc"]))] if data["val_acc"] else None,
    }

    summary_path = OUTPUT_DIR / "training_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(metrics_summary, f, indent=2)
    print(f"\nTraining summary: {json.dumps(metrics_summary, indent=2)}")
    print(f"Saved summary to {summary_path}")


if __name__ == "__main__":
    main()
