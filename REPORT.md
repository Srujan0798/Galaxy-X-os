# Galaxy-X-os — One-Page Technical Report

**SCALE x ODYSSEY** | Deep-learning classification of raw astronomical images into 5 celestial categories.
**GitHub:** https://github.com/Srujan0798/Galaxy-X-os

---

## Dataset Sources

- **SDSS DR17** (Sloan Digital Sky Survey, Data Release 17) — real astronomical imaging used for the final trained model.
- Merged Kaggle galaxy-morphology datasets (Galaxy Zoo / DeepSky / Planetary imagery) for class balancing.
- **5 classes:** Spiral Galaxy, Elliptical Galaxy, Nebula, Star Cluster, Planetary Object.
- **Split:** 80 / 10 / 10 (train / val / test). 2000 training images (400/class), 250 val, 250 test.

## Model Architecture

- **Backbone:** EfficientNet-B3 (ImageNet pretrained, transfer learning) — ~11.6M parameters.
- **Head:** Replaced classifier → 5-class output.
- **Training strategy:** Progressive unfreezing (backbone frozen first 3 epochs), mixed-precision (`torch.amp`), AdamW (lr `3e-4`), batch size 32, OneCycleLR scheduler.
- **Augmentations:** Albumentations astro-specific pipeline (cosmic-ray simulation, vignetting, Poisson noise, flips/rotations, label smoothing 0.1).
- **Explainability:** Grad-CAM heatmaps over predictions.

## Final Test Metrics

| Metric | Value |
|---|---|
| **Test Accuracy (Standard)** | **72.4%** |
| **Test Accuracy (TTA, 6× aug)** | **74.4%** |
| **Macro F1 (Standard)** | **0.720** |
| **Macro F1 (TTA)** | **0.742** |
| Best Val Accuracy | 95.0% (epoch 5, CPU run) |

**Per-class F1 (Standard):**

| Class | F1 |
|---|---|
| Elliptical Galaxy | 1.000 |
| Nebula | 1.000 |
| Planetary Object | 0.618 |
| Star Cluster | 0.535 |
| Spiral Galaxy | 0.449 |

*Strongest on Elliptical/Nebula; Spiral vs Star Cluster confusion drives the lower macro score (visually similar diffuse structures). TTA adds +2% accuracy across the board.*

## Confusion Matrix

See [`results/confusion_matrix.png`](results/confusion_matrix.png). Additional plots: `results/per_class_metrics.png`, `results/training_curves.png`, `results/confidence_distribution.png`.

## Inference Time

- **~410 ms per image on Apple MPS** (EfficientNet-B3, mixed precision, batch=1, after warmup).
- **Target <15 ms on CUDA GPU** — code supports `torch.autocast("cuda")` for GPU deployment.
- Streamlit demo runs in real time.

## Setup Instructions

```bash
conda create -n scale_odyssey python=3.10 -y
conda activate scale_odyssey
pip install -r requirements.txt

python src/download_datasets.py   # prepare data
python src/train.py               # train
python src/evaluate.py            # evaluate (standard + TTA)
python src/gradcam.py            # Grad-CAM report
streamlit run app/app.py          # web demo
```

## Training Script

Main entry point: [`src/train.py`](src/train.py). Config: [`config/config.yaml`](config/config.yaml). Best checkpoint: `checkpoints/epoch_005.pth` (95% val acc, CPU run) / `checkpoints/best_model.pth` (72.4% test acc, MPS run).