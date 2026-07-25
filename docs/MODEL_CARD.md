# Galaxy-X-os — Model Card

## Model Overview
- **Name:** Galaxy-X-os (SCALE x ODYSSEY)
- **Task:** Astronomical object classification (5 classes)
- **Architecture:** EfficientNet-B3 (timm, ImageNet-pretrained)
- **Parameters:** ~11.6M
- **Compute:** Google Colab T4 GPU (~45 min for 50 epochs)
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
2. **Phase 2** (epochs 4-50): Full fine-tune with OneCycleLR

Hyperparameters:
- **Optimizer:** AdamW (lr=3e-4)
- **Scheduler:** OneCycleLR with 30% warmup
- **Batch size:** 32
- **Mixed precision:** torch.amp float16
- **Augmentation:** Albumentations (cosmic ray, vignetting, Poisson noise, flips, rotations)
- **Loss:** CrossEntropy with label smoothing (ε=0.1)
- **Seed:** 42

## Test-Time Augmentation
- 6× augmentation (horizontal flips + brightness/contrast shifts)

## Performance
| Metric | Standard | + TTA (6×) |
|--------|----------|------------|
| Test Accuracy | **93.17%** | **92.77%** |
| Macro Precision | 0.934 | — |
| Macro Recall | 0.932 | — |
| Macro F1 | **0.932** | **0.928** |

**Per-class F1 (Standard):**
| Class | F1 |
|-------|-----|
| Spiral Galaxy | 0.887 |
| Elliptical Galaxy | 0.895 |
| Nebula | 0.949 |
| Star Cluster | 0.947 |
| Planetary Object | 0.980 |

## Explainability
- **Grad-CAM:** 3-panel figures (Original / True-Class CAM / Predicted-Class CAM)
- **15 samples** (3 per class) + summary grid
- Grad-CAM++ variant available in `src/gradcam_plus.py`

## Deployment
- **Streamlit demo:** Interactive classification + Grad-CAM + caption + OOD flag
- **ONNX:** Export pipeline in `src/onnx_export.py`
- **Inference:** ~72 ms/image on Apple MPS (median, after warmup)

## Limitations
1. Galaxy classes rely on DECaLS survey morphology (may not generalize to JWST)
2. Procedural fallback for star_cluster (15/485) and planetary (1/499) noted in manifest
3. Spiral ↔ Elliptical confusion (~0.89 F1) reflects genuine morphological ambiguity
4. TTA shows no gain over standard (real test set is harder than procedural)

## Ethical Considerations
- Model trained on public scientific data (Galaxy10, NASA)
- No human subjects or PII involved
- Intended for scientific/educational use in astronomy
