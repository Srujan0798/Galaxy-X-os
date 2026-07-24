# SCALE x ODYSSEY -- Executive Summary

## Why This Wins: A Judge-Focused Overview

---

### The Project

**SCALE x ODYSSEY** is a complete, end-to-end deep learning pipeline that classifies raw astronomical images into 5 celestial categories at **95.6% test accuracy** (96.4% with test-time augmentation, **0.96 macro F1**) with full explainability. It is a fully functional system, trained and evaluated on a 250-image held-out disjoint test set on real astronomical data (Galaxy10 DECaLS for the two galaxy classes, Kaggle deep-space sets + labelled procedural fallback for the rest — see `data/processed/DATA_MANIFEST.json`).

### Why It Stands Out

**1. Complete End-to-End Pipeline**
Every component from the official starter guide (pages 5-20) is fully implemented: data pipeline, model architecture, training with progressive unfreezing, evaluation with TTA, Grad-CAM explainability, fast inference, web demo, and both bonus tasks. No gaps, no missing pieces.

**2. Solid Architecture with Best Practices**
EfficientNet-B3 backbone (~11.6M parameters, ImageNet pretrained) with a custom 3-layer classification head using BatchNorm and progressive dropout. Training uses OneCycleLR scheduler with cosine annealing, weighted CrossEntropyLoss with label smoothing (0.1), mixed precision, and progressive unfreezing (backbone frozen for the first 3 epochs, full fine-tune after). Early stopping with patience=12 prevents overfitting. The final model was trained on a Google Colab CUDA GPU (T4) via `notebooks/Galaxy_X_Colab.ipynb`.

**3. Full Explainability -- Not Just a Heatmap**
Grad-CAM produces 3-panel figures per sample: Original Image, True-Class CAM, and Predicted-Class CAM. This allows judges to see not only what the model looked at for its prediction, but also what it would have looked at if it were correct. The 15-sample summary grid (3 per class) provides instant visual evidence of model quality.

**4. Interactive Deployment**
The `ModelManager` singleton with warmup runs inference at **~72 ms median (118 ms mean) per image on Apple MPS** — well under the 5 s requirement, and faster on a CUDA GPU. The Streamlit web demo features drag-and-drop upload, real-time prediction with confidence scores, probability distribution charts, and Grad-CAM overlay -- all with cached model loading for instant response.

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
| Classification Performance (40%) | **95.6% accuracy (96.4% with TTA)**, **0.956 macro F1 (0.964 TTA)**, per-class F1, confusion matrix, classification report |
| Model Efficiency (15%) | ~11.6M params, mixed precision, batch processing, **~72 ms/image on Apple MPS** (faster on CUDA) |
| Explainability (15%) | 15-sample 3-panel Grad-CAM, summary grid, confidence analysis |
| Innovation/Bonus (15%) | Streamlit demo, BLIP captioning, anomaly detection, 8 astro augmentations |
| Documentation (15%) | README, 3 notebooks, YAML config, demo video, this summary |

### The Bottom Line

This submission addresses every requirement from pages 5-20 of the official starter guide. It goes beyond minimum requirements with clean, modular code, domain-specific innovations, and thorough documentation. The reported numbers are honest and reproducible: **95.6% test accuracy (96.4% with TTA)** on a balanced 250-image held-out disjoint test set. Nebula, Star Cluster, and Planetary classify perfectly (F1 = 1.000); the residual confusion is between Spiral and Elliptical Galaxy (~0.88–0.90 F1), which share genuinely similar morphologies — a real, explainable limitation of the domain, not a hidden one.

---

**Team**: Janil Jain, Jaskirat Singh Maskeen, Priyal Keswani
**Stakeholders**: TechOIITGN
**Repository**: https://github.com/Srujan0798/Galaxy-X-os.git
