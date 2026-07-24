# SCALE x ODYSSEY

**Sequence-Based Classification of Astronomical Objects Using Deep Learning**

> Deep learning model for classifying raw astronomical images into 5 celestial categories.
> **TechOIITGN Hackathon Submission** | **95.6% test accuracy (96.4% with TTA, 0.96 macro F1)** | **Grad-CAM explainability + Web Demo**

---

## Project Overview

| | |
|---|---|
| **Problem** | Classify raw telescope images (Spiral Galaxy, Elliptical Galaxy, Nebula, Star Cluster, Planetary Object) using **only pixel data** — no handcrafted features |
| **Solution** | EfficientNet-B3 + transfer learning + astro-specific augmentations + progressive unfreezing + Grad-CAM explainability + Streamlit demo |
| **Key Results** | 95.6% test accuracy (96.4% with TTA) / 0.96 macro F1 on a held-out disjoint test set (250 images, MD5 leak-checked). Real Galaxy10 DECaLS survey imagery for the two galaxy classes + Kaggle deep-space sets (with labelled procedural fallback) for the rest — see `data/processed/DATA_MANIFEST.json`. Professional Grad-CAM visualizations + interactive web application |

### 5-Class Taxonomy

| # | Class | Description |
|---|-------|-------------|
| 0 | **Spiral Galaxy** | Milky Way-type structures with spiral arms |
| 1 | **Elliptical Galaxy** | Smooth, featureless oval blobs |
| 2 | **Nebula** | Gas/dust clouds, often colorful |
| 3 | **Star Cluster** | Dense groupings of stars |
| 4 | **Planetary Object** | Planets, moons, rings |

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Prepare data (downloads real data + writes DATA_MANIFEST.json)
python src/prepare_data.py

# 3. Train (full EfficientNet-B3 fine-tune)
python src/train.py

# 4. Evaluate (standard + TTA)
python src/evaluate.py

# 5. Generate Grad-CAM report (15 samples)
python src/gradcam.py

# 6. Fast inference test
python src/inference.py data/processed/test/ --batch-size 16

# 7. Bonus: caption + anomaly detection
python src/bonus.py data/processed/test/spiral_galaxy/sample.jpg

# 8. Launch web demo (record for submission!)
streamlit run app/app.py
```

> **One-click Colab:** open [`notebooks/Galaxy_X_Colab.ipynb`](notebooks/Galaxy_X_Colab.ipynb) to run the whole pipeline on a free GPU (this is how the reported model was trained).
>
> **Trained weights:** the trained `best_model.pth` (~141 MB) exceeds GitHub's 100 MB file limit, so it is published as a GitHub Release asset — download it from https://github.com/Srujan0798/Galaxy-X-os/releases/latest and drop it in `checkpoints/`. You can also reproduce it via steps 2–3 above.

---

## Repository Structure

```
Galaxy-X-os/
├── README.md                        # This file — overview + how to run
├── REPORT.md / REPORT.pdf           # One-page submission report (metrics, etc.)
├── PROBLEM_STATEMENT.md             # Official problem statement (transcribed)
├── BONUS_FEATURES.md                # Captioning + anomaly detection write-up
├── SUBMISSION.md                    # Submission checklist (criteria → files)
├── Makefile  requirements.txt  Dockerfile  LICENSE
├── configs/
│   └── config.yaml                  # Central configuration
├── src/
│   ├── dataset.py                   # PyTorch Dataset + Albumentations pipeline
│   ├── model.py                     # EfficientNet-B3 + custom head + Grad-CAM hooks
│   ├── utils.py                     # Helpers (device, seed, config, metrics, logging)
│   ├── generate_splits.py           # Stratified DISJOINT train/val/test split
│   ├── prepare_data.py              # Download real data + build splits + DATA_MANIFEST.json
│   ├── train.py                     # Full pipeline: progressive unfreezing + OneCycleLR
│   ├── evaluate.py                  # Metrics (Acc/Prec/Rec/F1) + TTA + confusion matrix
│   ├── gradcam.py                   # explain_image() + 15-sample 3-panel visualization
│   ├── inference.py                 # Fast single/batch inference + ModelManager
│   └── bonus.py                     # Image captioning (BLIP) + anomaly detection
├── app/
│   └── app.py                       # Streamlit interactive web demo
├── notebooks/                       # 01_EDA, 02_Training, 03_Evaluation
├── data/
│   ├── raw/                         # Source datasets (Galaxy10 DECaLS + Kaggle deep-space)
│   └── processed/                   # Disjoint 5-class split + DATA_MANIFEST.json
├── checkpoints/                     # Trained weights (best_model.pth, gitignored)
├── results/                         # confusion_matrix.png, gradcam/, evaluation_results.json
├── docs/                            # ADRs, runbooks, presentation/
├── tests/                           # Test suite
└── attic/                           # Archived orchestration scaffolding (not part of deliverable)
```

---

## Evaluation Criteria Coverage

| Component | Weight | Score | Evidence |
|-----------|--------|-------|----------|
| **Classification Performance** | 40% | — | 95.6% accuracy (96.4% with TTA), 0.956 macro F1 (0.964 TTA), confusion matrix, per-class F1, classification report |
| **Model Efficiency** | 15% | — | ~11.6M parameters, mixed precision (torch.amp), measured ~72 ms/image (median) on Apple MPS |
| **Explainability & Visualization** | 15% | — | Grad-CAM 3-panel figures (Original / True CAM / Predict CAM), 15-sample summary grid, confidence analysis |
| **Innovation / Bonus Features** | 15% | — | Streamlit web app + BLIP captioning + anomaly detection + astro-specific augmentations |
| **Documentation & Presentation** | 15% | — | Full README, modular code, 3 Jupyter notebooks, reproducible configs, demo video |

---

## Phase 1: Data Pipeline

```bash
python src/prepare_data.py
```

- Multi-source data: real Galaxy10 DECaLS survey imagery (Spiral + Elliptical) + Kaggle deep-space sets for Nebula / Star Cluster / Planetary, with a clearly-labelled procedural fallback. See `data/processed/DATA_MANIFEST.json` for the authoritative per-class real-vs-fallback record.
- Balanced **disjoint** splits (verified by MD5 — no image in >1 split)
- Class imbalance handling (oversampling + weighted loss)
- Quality filtering (corrupted image removal)
- 8 astronomy-specific augmentation transforms (cosmic ray simulation, vignetting, Poisson noise)

**Key features in `src/dataset.py`:**
- `AstroDataset`: PyTorch Dataset with `.samples` and `.labels` attributes
- `get_train_transforms()`: Heavy augmentation pipeline for training
- `get_val_transforms()`: Minimal transforms for validation/testing
- `get_loaders()`: Factory for train/val/test DataLoaders

---

## Phase 2: Training

```bash
python src/train.py
```

### Progressive Unfreezing Strategy

| Phase | Epochs | Backbone | Learning Rate | Notes |
|-------|--------|----------|---------------|-------|
| **Phase 1** | 1-3 | Frozen | 3e-3 (10x base) | Train only classifier head |
| **Phase 2** | 4+ | Unfrozen | 3e-4 (head), 3e-5 (backbone) | Discriminative fine-tuning |

### Training Features

- **Backbone**: EfficientNet-B3 (~11.6M parameters, pretrained on ImageNet)
- **Optimizer**: AdamW with weight decay
- **Scheduler**: OneCycleLR with 30% warmup
- **Loss**: Weighted CrossEntropyLoss + label smoothing (0.1)
- **Mixed Precision**: torch.amp (float16 on GPU)
- **Early Stopping**: Patience = 12 epochs
- **Checkpointing**: Best model + periodic saves

### Configuration (`configs/config.yaml`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model.backbone` | `efficientnet_b3` | Options: `efficientnet_b3`, `efficientnet_b4`, `resnet50` |
| `training.num_epochs` | 50 | Increase to 100 for final run |
| `training.lr` | 3e-4 | OneCycleLR max learning rate |
| `training.batch_size` | 32 | Reduce to 16 if you hit GPU OOM on a smaller card |
| `training.patience` | 12 | Early stopping patience |
| `training.mixed_precision` | true | Disable if instability |
| `training.freeze_backbone_epochs` | 3 | Progressive unfreezing duration |

---

## Phase 3: Explainability, Optimization & Demo

### Grad-CAM Explainability

```bash
python src/gradcam.py
```

- `explain_image()`: Full Grad-CAM pipeline for single images (predicted-class + true-class CAM)
- `visualize_predictions()`: 15 diverse test examples (3 per class minimum)
- **3-panel figure per sample**: Original Image | True-Class CAM | Predicted-Class CAM
- Summary grid (`_summary_grid.png`) for quick review
- Uses `model.get_gradcam_target_layer()` for backbone-agnostic targeting

**Output**: `results/gradcam/sample_XX_*.png` + `results/gradcam/_summary_grid.png`

### Fast Inference

```bash
# Single image
python src/inference.py path/to/image.jpg

# Batch (directory)
python src/inference.py path/to/images/ --batch-size 16
```

- `ModelManager` with **singleton caching** for web apps
- Single + batch prediction with **automatic mixed precision**
- Returns: class name, confidence, full probability distribution, inference time

**Measured**: ~72 ms per image (median; 118 ms mean) on Apple MPS after warmup — well under the 5 s requirement. Faster on a CUDA GPU.

### Interactive Web Demo

```bash
streamlit run app/app.py
# Opens at: http://localhost:8501
```

- Drag-and-drop image upload
- Real-time prediction with confidence score
- Full probability distribution bar chart
- **Grad-CAM heatmap overlay** with explanation
- **Offline template caption** for the predicted class (BLIP with automatic template fallback)
- **Softmax-entropy anomaly / OOD flag** (low-confidence + top-2 gap + entropy)
- GPU/CPU info sidebar
- Cached model loading for instant response

---

## Phase 4: Bonus Features (Page 19)

### Bonus 1: Image Captioning (BLIP)

```python
from src.bonus import generate_caption_with_fallback

result = generate_caption_with_fallback("image.jpg", "Spiral Galaxy")
print(result["caption"])
# "A magnificent spiral galaxy with distinct swirling arms and a bright central bulge."
```

- Uses `Salesforce/blip-image-captioning-base` model
- **Automatic fallback** to template captions if BLIP unavailable
- Templates tailored per astronomical class

### Bonus 2: Anomaly / OOD Detection (softmax entropy)

```python
from src.bonus import AnomalyDetector
from src.inference import predict_image

detector = AnomalyDetector(confidence_threshold=0.5, gap_threshold=0.15)
result = predict_image("image.jpg")
analysis = detector.analyze(result)
print(analysis["recommendation"])
# "Normal: Spiral Galaxy (94.2% confidence)" or "ANOMALY: ..."
```

- Flags **low-confidence** predictions
- Detects **ambiguous classifications** (small top-2 gap)
- Measures **entropy** of probability distribution

### Run Both Bonuses

```bash
python src/bonus.py path/to/image.jpg
```

---

## Jupyter Notebooks

| Notebook | Purpose |
|----------|---------|
| `01_EDA.ipynb` | Dataset distribution, sample images, size statistics |
| `02_Training.ipynb` | Interactive training with live loss/accuracy plots |
| `03_Evaluation.ipynb` | Confusion matrix, Grad-CAM generation, result viewing |

```bash
jupyter lab notebooks/
```

---

## Tech Stack

| Component | Choice |
|-----------|--------|
| Framework | PyTorch 2.4+ |
| Backbone | EfficientNet-B3 (~11.6M params) via timm |
| Augmentation | Albumentations + 8 custom astro transforms |
| Scheduler | OneCycleLR |
| Optimizer | AdamW |
| Mixed Precision | torch.amp |
| Explainability | pytorch-grad-cam |
| Captioning | BLIP (transformers) |
| Web Framework | Streamlit |
| Logging | TensorBoard |

---

## Reproducibility

```python
# Set in train.py
import torch
import numpy as np
import random

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
torch.backends.cudnn.deterministic = True
```

All training configurations are centralized in `configs/config.yaml`.

---

## Learning Outcomes Demonstrated

1. **Computer vision pipelines** for scientific imagery
2. **Transfer learning** with progressive unfreezing
3. **Explainable AI** (Grad-CAM visualizations)
4. **Interactive deployment** (Streamlit web app)
5. **Real-world domain-specific ML** (astronomical imaging challenges)

---

## Team

| Name | Role |
|------|------|
| **Janil Jain** | Team Lead & ML Engineering |
| **Jaskirat Singh Maskeen** | Data Pipeline & Augmentation |
| **Priyal Keswani** | Evaluation & Deployment |

**Stakeholders**: TechOIITGN

---

## License

MIT License -- see `LICENSE` file.

Pretrained weights from TIMM (Apache 2.0) and BLIP (BSD-3-Clause) used with proper attribution.

---

**Ready for submission!**
