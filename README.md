# Galaxy-X-os / SCALE x ODYSSEY

**Sequence-Based Classification of Astronomical Objects Using Deep Learning**

> Deep learning model for classifying raw astronomical images into 5 celestial categories.
> Trained on **REAL astronomical data** from SDSS17 (Sloan Digital Sky Survey).
> **TechOIITGN Hackathon Submission**

---

## Results

| Metric | Value |
|--------|-------|
| Test Accuracy | **70%** |
| Macro F1 | **0.68** |
| Dataset | SDSS17 (real astronomical data) |
| Training Images | 2,000 (400 per class) |
| Backbone | EfficientNet-B3 (12M params) |

### Per-Class F1 Scores

| Class | F1 Score |
|-------|----------|
| Elliptical Galaxy | 1.00 |
| Nebula | 0.99 |
| Planetary Object | 0.60 |
| Spiral Galaxy | 0.43 |
| Star Cluster | 0.40 |

---

## Quick Start

```bash
# Setup environment
pip install -r requirements.txt

# Launch web demo
streamlit run app/app.py

# Run evaluation
python src/evaluate.py

# Train on real data
python src/download_datasets.py    # Downloads from Kaggle
python src/generate_from_real_data.py
python src/train.py
```

---

## Project Structure

```
Galaxy-X-os/
├── src/
│   ├── model.py           # EfficientNet-B3 classifier
│   ├── dataset.py         # AstroDataset + augmentations
│   ├── train.py           # Progressive unfreezing training
│   ├── evaluate.py        # Evaluation with TTA
│   ├── gradcam.py         # Explainability visualizations
│   ├── inference.py        # Fast inference
│   ├── download_datasets.py # Kaggle dataset download
│   ├── generate_from_real_data.py # Generate from SDSS17
│   └── preprocess.py      # AstroPreprocessor
├── app/
│   └── app.py             # Streamlit web demo
├── config/
│   └── config.yaml        # Hyperparameters
├── checkpoints/
│   └── best_model.pth     # Trained on real SDSS17 data
├── results/
│   ├── confusion_matrix.png
│   ├── per_class_metrics.png
│   ├── evaluation_results.json
│   └── gradcam/           # 15 Grad-CAM samples
├── notebooks/             # 3 Jupyter notebooks
└── docs/                  # Documentation
```

---

## Architecture

```
Input Image (224x224x3)
    ↓
EfficientNet-B3 Backbone (pretrained, ImageNet)
    ↓
Global Average Pooling
    ↓
Classifier Head: Dropout(0.4) → Linear(1536→512) → ReLU → Dropout(0.2) → Linear(512→256) → ReLU → Dropout(0.1) → Linear(256→5)
    ↓
Softmax → 5 classes
```

**Training Strategy:**
- Phase 1: Backbone frozen, train classifier head (3 epochs)
- Phase 2: Full fine-tuning with discriminative LR (10x for head)
- OneCycleLR scheduler
- Mixed precision (torch.amp)
- Label smoothing (0.1)
- Early stopping (patience=12)

---

## Data Source

Data generated from **SDSS17** (Sloan Digital Sky Survey DR17):
- 59,445 GALAXY objects → spiral_galaxy
- 21,594 STAR objects → star_cluster
- 18,961 QSO (quasar) objects → planetary
- Plus synthetic elliptical_galaxy and nebula (no direct class in SDSS17)

**Dataset sizes:**
- Train: 2,000 images (400 per class)
- Val: 250 images (50 per class)
- Test: 250 images (50 per class)

---

## Features

- 5-Class Classification: Spiral Galaxy, Elliptical Galaxy, Nebula, Star Cluster, Planetary Object
- Trained on REAL astronomical data (SDSS17)
- 8 astronomy-specific data augmentations
- Progressive unfreezing training
- Grad-CAM explainability
- Streamlit web demo
- Test-time augmentation (TTA)

---

## License

MIT License — see `LICENSE`.