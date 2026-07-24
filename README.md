# SCALE x ODYSSEY

**Sequence-Based Classification of Astronomical Objects Using Deep Learning**

> Deep learning model for classifying raw astronomical images into 5 celestial categories.
> **TechOIITGN Hackathon Submission** | **93.17% test accuracy (92.77% with TTA, 0.93 macro F1)** on fully-real imagery | **Grad-CAM explainability + Web Demo**

---

## Project Overview

| | |
|---|---|
| **Problem** | Classify raw telescope images (Spiral Galaxy, Elliptical Galaxy, Nebula, Star Cluster, Planetary Object) using **only pixel data** — no handcrafted features |
| **Solution** | EfficientNet-B3 + transfer learning + astro-specific augmentations + progressive unfreezing + Grad-CAM explainability + Streamlit demo |
| **Key Results** | 93.17% test accuracy (92.77% with TTA) / 0.932 macro F1 on a held-out disjoint test set (249 images, MD5 leak-checked), all five classes trained on **real** telescope imagery: Galaxy10 DECaLS survey images for Spiral + Elliptical, NASA Image Library (images.nasa.gov) Hubble/Spitzer/JPL imagery for Nebula / Star Cluster / Planetary (no API key) — see `data/processed/DATA_MANIFEST.json`. Professional Grad-CAM visualizations + interactive web application |

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
> **Trained weights:** `best_model.pth` (~141 MB) exceeds GitHub's 100 MB file limit, so it is **not** committed. Two ways to get it: (1) **reproduce** it exactly via the one-click Colab notebook or steps 2–3 above (~30 min on a free GPU) — this is the guaranteed path; (2) if published, download it from the [**v1.0 Release**](https://github.com/Srujan0798/Galaxy-X-os/releases/tag/v1.0) and drop it in `checkpoints/`.
>
> **Verify the checkpoint** after download:
> ```bash
> shasum -a 256 checkpoints/best_model.pth   # macOS
> sha256sum checkpoints/best_model.pth       # Linux
> ```
> Record the hash in [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md) so graders can confirm integrity.

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
│   ├── prepare_data.py              # Download real data (Galaxy10 + NASA Image Library + Kaggle) + splits + manifest
│   ├── download_archives.py         # NASA Image Library fetch + purity filters (real nebula/cluster/planetary)
│   ├── train.py                     # Full pipeline: progressive unfreezing + OneCycleLR
│   ├── evaluate.py                  # Metrics (Acc/Prec/Rec/F1) + TTA + confusion matrix
│   ├── gradcam.py                   # explain_image() + 15-sample 3-panel visualization
│   ├── inference.py                 # Fast single/batch inference + ModelManager
│   └── bonus.py                     # Captioning (template+BLIP) + localization + anomaly/OOD
├── app/
│   └── app.py                       # Streamlit interactive web demo
├── notebooks/                       # 01_EDA, 02_Training, 03_Evaluation + Galaxy_X_Colab
├── data/
│   ├── raw/                         # Source datasets (Galaxy10 + NASA Image Library + Kaggle deep-space)
│   └── processed/                   # Disjoint 5-class split + DATA_MANIFEST.json
├── checkpoints/                     # Trained weights (best_model.pth — GitHub Release, gitignored)
├── results/                         # confusion_matrix.png, gradcam/, evaluation_results.json
├── tests/                           # pytest suites (unit / integration / e2e)
├── attic/                           # Archived orchestration scaffolding + superseded src files
└── docs/                            # ADRs, runbooks, presentation/ (incl. demo.mp4)
```

---

## Evaluation Criteria Coverage

| Component | Weight | Score | Evidence |
|-----------|--------|-------|----------|
| **Classification Performance** | 40% | — | 93.17% accuracy (92.77% with TTA), 0.932 macro F1 (0.928 TTA), confusion matrix, per-class F1, classification report |
| **Model Efficiency** | 15% | — | ~11.6M parameters, mixed precision (torch.amp), measured ~72 ms/image (median) on Apple MPS |
| **Explainability & Visualization** | 15% | — | Grad-CAM 3-panel figures (Original / True CAM / Predict CAM), 15-sample summary grid, confidence analysis |
| **Innovation / Bonus Features** | 15% | — | Streamlit web app + BLIP captioning + anomaly detection + astro-specific augmentations |
| **Documentation & Presentation** | 15% | — | Full README, modular code, 3 Jupyter notebooks, reproducible configs, demo video |

---

## Phase 1: Data Pipeline

```bash
python src/prepare_data.py
```

- Multi-source real data (real-first, safe-fallback):
  - **Galaxy10 DECaLS / SDSS** via `astroNN` for Spiral + Elliptical galaxy imagery (no API key).
  - **NASA Image Library** ([images.nasa.gov](https://images.nasa.gov), no API key) for Nebula / Star Cluster / Planetary — real Hubble / Spitzer / JPL mission imagery retrieved by keyword with per-class purity filtering (see `src/download_archives.py`).
  - **Kaggle** deep-space sets (`fedesoriano/deep-space-images`, `brsdincer/planetary-solar-system-objects`) and a labelled procedural generator remain as fallbacks only.
  - See `data/processed/DATA_MANIFEST.json` for the authoritative per-class real-vs-fallback record.
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

### Bonus 1: Image Captioning (template + optional BLIP)

```python
from src.bonus import generate_caption_with_fallback

result = generate_caption_with_fallback(
    "image.jpg", class_name="Spiral Galaxy", confidence=0.87, use_blip=False
)
print(result["caption"], result["method"])
# "A spiral galaxy showing a bright central core and sweeping arm structure (confidence 0.87)." "template"
```

- **Default path (offline, deterministic, no downloads):** `generate_template_caption` composes a short natural-language description from the predicted class + its structural cue + image brightness/contrast + confidence. Always available.
- **Optional neural path:** set `use_blip=True` to use `Salesforce/blip-image-captioning-base` via `transformers` (downloads ~1 GB on first use). Falls back to the template automatically if transformers is missing or BLIP errors.
- Per-class structural descriptors in `CLASS_DESCRIPTORS`.

### Bonus 2: Object Localization (Grad-CAM pseudo-bbox)

```python
from src.bonus import localize_object, render_localization_overlay
from src.gradcam import generate_cam  # raw 2-D heatmap
import numpy as np

cam = generate_cam(model, input_tensor, target_class=0, device=device)  # 2-D heatmap
loc = localize_object(cam, threshold=0.30)
print(loc["bbox"], loc["area_frac"])
overlay = render_localization_overlay(image_rgb, cam, loc["bbox"])
```

- **Honest scope:** this is a saliency-threshold localizer, NOT a learned detector (no YOLO / Mask R-CNN). It re-uses the classifier's Grad-CAM — the same mechanism the problem statement permits under "attention or Grad-CAM visualizations" — to produce a tight bounding box of the region that most influenced the decision.
- Returns `[x_min, y_min, x_max, y_max]` in heatmap-pixel coords, plus `bbox_frac` (0–1), `area_frac`, and `mask_area_px`.

### Bonus 3: Anomaly / OOD Detection (softmax entropy + max-prob)

```python
from src.bonus import AnomalyDetector, detect_anomaly
from src.inference import predict_image

result = predict_image("image.jpg")
verdict = detect_anomaly(result.all_probabilities)
print(verdict["is_anomaly"], verdict["reason"])
# False "In-distribution: confident prediction (max prob 0.94, entropy 0.21 bits)"

det = AnomalyDetector(max_prob_threshold=0.45, entropy_threshold=1.50)
print(det.analyze(result)["is_anomaly"])
```

- Flags a prediction as a possible anomaly / OOD when **max-prob < 0.45 OR entropy > 1.50 bits** (5-class max entropy ≈ 2.322 bits). Purely post-hoc on the softmax — no retraining, no external service, fully explainable.

### Run all bonuses together

```bash
python src/bonus.py path/to/image.jpg            # caption + localization + OOD
python src/bonus.py path/to/image.jpg --no-localize  # skip localization
python src/bonus.py path/to/image.jpg --use-blip    # opt into BLIP
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
