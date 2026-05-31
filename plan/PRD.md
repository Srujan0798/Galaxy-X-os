# PRD — Product Requirements Document

## Galaxy-X-os / SCALE x ODYSSEY

### Problem Statement

Classify raw telescope images into 5 celestial categories using only pixel data — no handcrafted features.

### Solution

EfficientNet-B3 + transfer learning + astro-specific augmentations + progressive unfreezing + Grad-CAM explainability + Streamlit demo.

### Classes

1. Spiral Galaxy
2. Elliptical Galaxy
3. Nebula
4. Star Cluster
5. Planetary Object

### Success Metrics

- Classification accuracy ≥ 88% (≥ 94% with TTA)
- Inference time < 15ms per image on GPU
- Complete Grad-CAM visualizations for 15 samples
- Working Streamlit demo

### MVP Definition

Ingest one telescope image, output one of 5 class predictions with confidence score and Grad-CAM overlay.

### Waves

| Wave | Name | Deliverable |
|------|------|-------------|
| 1 | Foundation | Project structure, config, data pipeline skeleton |
| 2 | Data Pipeline | Working dataset, augmentations, loaders |
| 3 | Training + Evaluation | Trained model, evaluation metrics, TTA |
| 4 | Explainability + Demo | Grad-CAM, Streamlit app, inference CLI |
| 5 | Polish + Submission | Notebooks, documentation, demo video |

### Entities

- `AstroImage`: input image (RGB, 224x224)
- `AstroClass`: one of 5 categorical labels
- `ModelCheckpoint`: saved weights + config + optimizer state
- `GradCAMOutput`: heatmap overlay + confidence

### Risks

- Dataset quality (corrupted images)
- Class imbalance
- OOM with large batch sizes
- Grad-CAM dependency availability
