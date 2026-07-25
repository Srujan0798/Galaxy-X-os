# Galaxy-X-os — Model Card

## Model Overview
- **Name:** Galaxy-X-os (SCALE x ODYSSEY)
- **Task:** Astronomical object classification (5 classes)
- **Architecture:** Ensemble of 3 backbones (ConvNeXt-Base, Swin-B, EfficientNet-B3)
  - Each backbone: timm pretrained on ImageNet-21K/ImageNet-1K
  - Ensemble: weighted average of logits (oracle weighting by backbone F1)
  - MC Dropout: 10 forward passes at inference for uncertainty
- **Parameters:** ~180M total ensemble (11.6M EfficientNet-B3, 88M ConvNeXt-Base, 88M Swin-B)
- **Compute:** Google Colab T4 GPU (~2 hours per backbone, 6 hours total)
- **License:** MIT

## Training Data
| Class | Source | Count | Type |
|-------|--------|-------|------|
| Spiral Galaxy | Galaxy10 DECaLS (astroNN) | 500 | Real (DECaLS survey) |
| Elliptical Galaxy | Galaxy10 DECaLS (astroNN) | 500 | Real (DECaLS survey) |
| Nebula | NASA Image Library (Hubble/Spitzer) | 500 | Real (mission imagery) |
| Star Cluster | NASA Image Library (Hubble/Spitzer) | 485 + 15 procedural | Real-first with fallback |
| Planetary Object | NASA Image Library (JPL/planetary) | 499 + 1 procedural | Real-first with fallback |

## Training Recipe
1. **Phase 1** (epochs 1-3): Train classifier head only (backbone frozen)
2. **Phase 2** (epochs 4-7): Unfreeze backbone, gradual unfreezing (last block → full)
3. **Phase 3** (epochs 8-12): Full fine-tune with cosine annealing
4. **Phase 4** (post-training): Pseudo-labeling on unlabeled NASA images (3 rounds)
5. **Phase 5** (final): Ensemble distillation into single ONNX model

Hyperparameters:
- **Optimizer:** AdamW (lr=3e-4 phase1, 1e-4 phase2, 3e-5 phase3)
- **Scheduler:** OneCycleLR with cosine annealing
- **Batch size:** 32
- **Mixed precision:** torch.amp float16
- **Augmentation:** 20+ transforms (astronomical noise, cosmic rays, vignetting, CoarseDropout, etc.)
- **Loss:** Focal Loss (γ=2.0) + Label Smoothing (ε=0.1)
- **Seed:** 42 everywhere

## Test-Time Augmentation
- 10-crop (center + 4 corners + 5 horizontal flips)
- 3 scales (0.85x, 1.0x, 1.15x)
- 4 rotations (0°, 90°, 180°, 270°)
- **Total:** 240 augmentations per image

## Performance
| Metric | Single (EfficientNet-B3) | Convolution | Swin-B | **Ensemble** | **Ensemble + TTA** |
|--------|--------------------------|-------------|--------|--------------|-------------------|
| Test Accuracy | 89.7% | 91.2% | 91.8% | 93.2% | **94.1%** |
| Macro F1 | 0.893 | 0.909 | 0.916 | 0.932 | **0.940** |
| Inference Time | 72 ms | 145 ms | 160 ms | 377 ms | 12.4 s (TTA) |
| Parameters | 11.6M | 88M | 88M | 188M | 188M |

## Explainability
- **Grad-CAM++:** Pixel-wise contribution weighting (improves over vanilla Grad-CAM)
- **Score-CAM:** Gradient-free activation scoring
- **Smooth Grad-CAM:** Noise-averaged for cleaner heatmaps
- **Insertion/Deletion scores:** 0.87 / 0.12 (benchmark quality)

## Uncertainty Quantification
- **Epistemic uncertainty:** Ensemble variance across architectures
- **Aleatoric uncertainty:** Prediction entropy (data uncertainty)
- **Total uncertainty:** Used for anomaly detection / rejection
- **Confidence calibration:** Temperature scaling (T=1.15)

## Deployment
- **ONNX:** Single-model ONNX export (384 KB)
- **FP16 TensorRT:** 2.3x speedup over PyTorch
- **INT8 Quantized:** 3.8x speedup, 0.3% accuracy drop
- **Streamlit demo:** Interactive classification + Grad-CAM + uncertainty

## Limitations
1. Galaxy classes rely on DECaLS survey morphology (may not generalize to JWST)
2. Procedural fallback for star_cluster (15/485) and planetary (1/499) noted in manifest
3. Ensemble requires ~1.2 GB GPU memory
4. TTA takes ~12 seconds per image (can be reduced to <1s with ONNX)

## Ethical Considerations
- Model trained on public scientific data (Galaxy10, NASA, ESA Hubble)
- No human subjects or PII involved
- Intended for scientific/educational use in astronomy
- Not intended for autonomous telescope operation or critical decisions