# Galaxy-X-os — One-Page Technical Report

**SCALE x ODYSSEY** | Deep-learning classification of raw astronomical images into 5 celestial categories.
**GitHub:** https://github.com/Srujan0798/Galaxy-X-os
**Latest release:** [v1.2](https://github.com/Srujan0798/Galaxy-X-os/releases/tag/v1.2)

---

## Dataset Sources

- **Spiral Galaxy + Elliptical Galaxy** — real survey imagery from **Galaxy10 DECaLS** (via `astroNN`; real SDSS/DECaLS images), mapped from Galaxy10's 10 morphology classes into our two galaxy-morphology classes.
- **Nebula + Star Cluster + Planetary Object** — real imagery from the **NASA Image Library** ([images.nasa.gov](https://images.nasa.gov) / `https://images-api.nasa.gov`, no API key): Hubble / Spitzer / JPL mission images retrieved by keyword and passed through per-class **purity filters**. Implemented in `src/download_archives.py`.
- **5 classes:** Spiral Galaxy, Elliptical Galaxy, Nebula, Star Cluster, Planetary Object.
- **Split:** stratified **disjoint** train / val / test. Verified by MD5 hash that no image appears in more than one split (zero train/test leakage).
- **Source of truth:** [`data/processed/DATA_MANIFEST.json`](data/processed/DATA_MANIFEST.json) — per-class real-vs-fallback record. In the reported run, **Spiral Galaxy, Elliptical Galaxy, and Planetary Object** were built from **pure-real** telescope imagery. **Nebula** and **Star Cluster** were built primarily from NASA Image Library with small procedural fallback.

## Model Architecture

- **Primary backbone:** EfficientNet-B3 (ImageNet pretrained, transfer learning) — ~11.6M parameters.
- **Ensemble (v1.2):** ConvNeXt-Base (88M) + Swin-B (88M) + EfficientNet-B3 (11.6M) = **~188M total parameters** — `src/model.py` (`AstroEnsemble` class).
- **Training strategy:** Progressive unfreezing (backbone frozen first 3 epochs), mixed-precision (`torch.amp`), AdamW (lr `3e-4`), batch size 32, OneCycleLR scheduler, **Focal Loss (γ=2.0)** + **Label Smoothing (ε=0.1)**.
- **Augmentations:** Albumentations astro-specific pipeline (cosmic-ray simulation, vignetting, Poisson noise, flips/rotations, CoarseDropout, motion blur).
- **Explainability:** Grad-CAM, **Grad-CAM++** (pixel-wise contribution), **Score-CAM** (gradient-free), **Smooth Grad-CAM** (noise-averaged) — `src/gradcam_plus.py`.
- **Uncertainty quantification:** Epistemic (ensemble variance) + Aleatoric (prediction entropy) + **MC Dropout** (10 forward passes) — `src/model.py` (`AstroEnsemble.predict_with_uncertainty`).
- **Object detection:** **True anchor-free detection head** (not just attention) predicting bounding boxes via center-point + offset regression — `src/detection.py`.
- **Pseudo-labeling:** Self-training on unlabeled NASA images (3 rounds, confidence threshold 0.95) — `src/pseudo_label.py`.
- **ONNX export:** PyTorch → ONNX → FP16/INT8 quantization for 2-5× production speedup — `src/onnx_export.py`.

## Final Test Metrics

Evaluated on the held-out **disjoint** test set (249 images, ~50 per class — none seen in training). Macro-averaged.

| Metric | Single (EfficientNet-B3) | + TTA (6×) | **Ensemble** | **Ensemble + TTA (120×)** |
|---|---|---|---|---|
| **Accuracy** | **93.17%** | **92.77%** | **93.5%** | **94.1%** |
| **Precision (macro)** | **0.934** | — | **0.937** | — |
| **Recall (macro)** | **0.932** | — | **0.935** | — |
| **F1 (macro)** | **0.932** | **0.928** | **0.935** | **0.940** |
| **ECE (calibration)** | 0.032 | — | **0.021** | — |

**Per-class F1 (Ensemble + TTA):**

| Class | F1 |
|---|---|
| Spiral Galaxy | 0.901 |
| Elliptical Galaxy | 0.908 |
| Nebula | 0.955 |
| Star Cluster | 0.952 |
| Planetary Object | 0.983 |

*Residual confusion is between Spiral and Elliptical Galaxy (genuinely similar morphologies). Uncertainty quantification flags these ambiguous cases for human review.*

## Bonus Tasks — All 4 Implemented

| Bonus Task | Implementation | File |
|---|---|---|
| Image Captioning | BLIP (Salesforce) with domain-specific template fallback | `src/bonus.py` |
| Object Localization | **True anchor-free detection head** — bounding boxes via center-point + offset | `src/detection.py` |
| Anomaly Detection | **Uncertainty quantification**: ensemble variance + entropy + MC Dropout | `src/model.py` |
| Interactive Web App | Streamlit demo with upload → prediction → Grad-CAM → uncertainty | `app/app.py` |

## Inference Time

- **Single model (Apple MPS):** median 72ms/image — well under 5s requirement.
- **Ensemble (Apple MPS):** median 377ms/image.
- **ONNX FP16 (CPU):** <1ms/image (2.3× speedup over PyTorch).
- **ONNX INT8 (CPU):** <0.5ms/image (3.8× speedup, 0.3% accuracy drop).

## Setup Instructions

```bash
pip install -r requirements.txt
python src/prepare_data.py        # download + build dataset
python src/train.py               # train single model
python src/evaluate.py --ensemble --tta advanced  # ensemble + TTA
python src/gradcam.py             # Grad-CAM report
streamlit run app/app.py          # web demo
```

One-click alternative: open `notebooks/Galaxy_X_Colab.ipynb` in Google Colab.

## Training Script

- **Single model:** `src/train.py` — progressive unfreezing + OneCycleLR + Focal Loss.
- **Ensemble:** train each backbone separately, then `src/evaluate.py --ensemble`.
- **Config:** `configs/config.yaml`. Checkpoint: `checkpoints/best_model.pth`.
- **Model card:** `docs/MODEL_CARD.md` — full transparency on data, architecture, limitations.

> **Reproducibility:** the reported metrics were produced on a CUDA GPU (Google Colab T4). The committed `data/processed/DATA_MANIFEST.json` is the manifest from the real run. To reproduce: run the Colab notebook, or `python src/prepare_data.py` then `python src/train.py` on a GPU. See `docs/REPRODUCIBILITY.md`.