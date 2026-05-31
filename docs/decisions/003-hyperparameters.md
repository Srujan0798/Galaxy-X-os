# ADR-003: Hyperparameter Configuration

**Date:** 2026-05-31
**Status:** Accepted

## Context

We need to optimize hyperparameters for training on 160 training images (real/synthetic astronomical data) across 5 classes with the goal of achieving >88% accuracy.

## Decision

We will use the following hyperparameter configuration:

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `num_epochs` | 50 | Sufficient for convergence with early stopping |
| `lr` | 3e-4 | Standard learning rate for fine-tuning pretrained models |
| `weight_decay` | 1e-4 | Light regularization to prevent overfitting on small dataset |
| `batch_size` | 32 | Fits in GPU memory, good gradient estimates |
| `label_smoothing` | 0.1 | Reduces overconfidence, improves calibration |
| `dropout` | 0.4 | Moderate dropout for regularization |
| `freeze_backbone_epochs` | 3 | Phase 1: Train classifier head first |
| `patience` | 12 | Allow sufficient time for learning rate annealing |
| `mixed_precision` | true | Faster training with minimal memory |

## Learning Rate Schedule

- **Phase 1 (Epochs 0-2):** Backbone frozen, head LR = 3e-3 (10x base)
- **Phase 2 (Epochs 3-49):** Full fine-tuning with discriminative LR
  - Backbone LR: 3e-5 (base/10)
  - Head LR: 3e-4 (base)

This schedule follows the principle of progressive unfreezing and discriminative learning rates from ULMFiT.

## Data Augmentation

The `get_train_transforms()` in `dataset.py` applies:
- Random rotation (90°)
- Horizontal/Vertical flips
- RandomResizedCrop (0.5-1.0 scale)
- ShiftScaleRotate
- Brightness/Contrast/Gamma adjustments
- Custom astronomical augmentations (AddAstroNoise, SimulateCosmicRay, SimulateVignetting)
- CoarseDropout
- Blur transforms

## Consequences

- Training should complete in ~25 min on RTX 4090
- Early stopping will trigger if no improvement for 12 epochs
- Mixed precision reduces memory usage by ~40%
