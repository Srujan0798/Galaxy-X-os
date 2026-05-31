# HANDOFF.md

> Switching sessions? Read this first.

## Current State

- **Project:** SCALE x ODYSSEY - COMPLETE ✅
- **Model:** EfficientNet-B3, 15 epochs trained
- **Performance:** 95% standard accuracy, **100% with TTA**

## Quick Start

```bash
# Launch web demo
streamlit run app/app.py

# Re-train (needs GPU for speed)
python src/train.py

# Evaluate
python src/evaluate.py
```

## Results

| Metric | Value |
|--------|-------|
| Test Accuracy | 95% (standard) / **100%** (TTA) |
| Macro F1 | 0.95 (standard) / **1.0** (TTA) |
| Per-Class F1 | All >0.85 |
| Checkpoint | 52MB trained model |

## Key Files

| File | Purpose |
|------|---------|
| `src/model.py` | EfficientNet-B3 classifier |
| `src/train.py` | Training with progressive unfreezing |
| `src/evaluate.py` | Evaluation with TTA |
| `src/gradcam.py` | Explainability visualizations |
| `src/inference.py` | Fast inference |
| `app/app.py` | Streamlit web demo |
| `config/config.yaml` | All hyperparameters |

## Data

- Train: 160 images (32/class)
- Val: 20 images (4/class)
- Test: 20 images (4/class)
- 5 classes: spiral_galaxy, elliptical_galaxy, nebula, star_cluster, planetary
