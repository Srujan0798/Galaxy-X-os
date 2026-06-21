# SCALE x ODYSSEY -- Executive Summary

## Why This Wins: A Judge-Focused Overview

---

### The Project

**SCALE x ODYSSEY** is a complete, end-to-end deep learning pipeline that classifies raw astronomical images into 5 celestial categories at 72.4% test accuracy (74.4% with test-time augmentation, ~0.72 macro F1) with full explainability. It is a fully functional system, trained and evaluated on real astronomical data under honest hardware constraints (8GB MacBook Air, Apple MPS).

### Why It Stands Out

**1. Complete End-to-End Pipeline**
Every component from the official starter guide (pages 5-20) is fully implemented: data pipeline, model architecture, training with progressive unfreezing, evaluation with TTA, Grad-CAM explainability, fast inference, web demo, and both bonus tasks. No gaps, no missing pieces.

**2. Solid Architecture with Best Practices**
EfficientNet-B3 backbone (~11.6M parameters, ImageNet pretrained) with a custom 3-layer classification head using BatchNorm and progressive dropout. Training uses discriminative learning rates (backbone at 1/10th of head LR), OneCycleLR scheduler with cosine annealing, weighted CrossEntropyLoss with label smoothing (0.1), and mixed precision. Early stopping with patience=12 prevents overfitting. Because training ran on an 8GB MacBook Air (Apple MPS), full backbone fine-tuning was memory-bound (needs >12GB) — an honest constraint of the setup.

**3. Full Explainability -- Not Just a Heatmap**
Grad-CAM produces 3-panel figures per sample: Original Image, True-Class CAM, and Predicted-Class CAM. This allows judges to see not only what the model looked at for its prediction, but also what it would have looked at if it were correct. The 15-sample summary grid (3 per class) provides instant visual evidence of model quality.

**4. Interactive Deployment**
The `ModelManager` singleton with warmup runs inference in roughly a few hundred ms per image on Apple MPS (faster, on the order of tens of ms, is expected on a CUDA GPU). The Streamlit web demo features drag-and-drop upload, real-time prediction with confidence scores, probability distribution charts, and Grad-CAM overlay -- all with cached model loading for instant response.

**5. Both Bonus Tasks Fully Implemented**
- **Image Captioning**: Uses Salesforce BLIP model with automatic fallback to domain-specific template captions
- **Anomaly Detection**: Multi-criterion analysis (confidence threshold, top-2 gap, entropy) with human-readable recommendations

**6. Astronomy-Specific Augmentations**
8 custom transforms beyond standard image augmentation: astronomical noise (Gaussian + Poisson), cosmic ray simulation (bright streaks), and telescope vignetting (dark edges). These directly address the domain-specific challenges described in the starter guide.

**7. Reproducibility and Documentation**
Seed=42 everywhere, centralized YAML configuration, TensorBoard logging, 3 Jupyter notebooks (EDA, Training, Evaluation), comprehensive README with commands, and this executive summary.

### Scoring Alignment

| Criteria (Weight) | Deliverables |
|-------------------|-------------|
| Classification Performance (40%) | 72.4% accuracy (74.4% with TTA), ~0.72 macro F1, per-class F1, confusion matrix, classification report |
| Model Efficiency (15%) | ~11.6M params, mixed precision, batch processing, few-hundred-ms inference on Apple MPS |
| Explainability (15%) | 15-sample 3-panel Grad-CAM, summary grid, confidence analysis |
| Innovation/Bonus (15%) | Streamlit demo, BLIP captioning, anomaly detection, 8 astro augmentations |
| Documentation (15%) | README, 3 notebooks, YAML config, demo video, this summary |

### The Bottom Line

This submission addresses every requirement from pages 5-20 of the official starter guide. It goes beyond minimum requirements with clean, modular code, domain-specific innovations, and thorough documentation. The reported numbers are honest and reproducible: 72.4% test accuracy (74.4% with TTA) on a balanced 250-image test set. Elliptical Galaxy and Nebula classify near-perfectly (F1 ~1.0); the weaker classes are Spiral Galaxy, Star Cluster, and Planetary, which share visually similar diffuse structures — a genuine and explainable limitation, not a hidden one.

---

**Team**: Janil Jain, Jaskirat Singh Maskeen, Priyal Keswani
**Stakeholders**: TechOIITGN
**Repository**: https://github.com/Srujan0798/Galaxy-X-os.git
