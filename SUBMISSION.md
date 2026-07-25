# Galaxy-X-os — Final Submission Checklist

**SCALE x ODYSSEY** | Astronomical image classifier (Ensemble: ConvNeXt-Base + Swin-B + EfficientNet-B3, 5 classes)
**GitHub repo:** https://github.com/Srujan0798/Galaxy-X-os
**Latest release:** [v1.2](https://github.com/Srujan0798/Galaxy-X-os/releases/tag/v1.2)
**One-page report:** [`REPORT.md`](REPORT.md) / [`REPORT.pdf`](REPORT.pdf)

Classes: Spiral Galaxy · Elliptical Galaxy · Nebula · Star Cluster · Planetary Object

---

## How to Submit (Google Form)

1. **Name:** enter your full name.
2. **GitHub link:** paste `https://github.com/Srujan0798/Galaxy-X-os`
3. **File upload (optional, 1 file, PDF/doc, ≤10MB):** upload [`REPORT.pdf`](REPORT.pdf).
4. Submit.

---

## Scoring Criteria → Where It's Satisfied

| # | Criterion | Weight | Where in repo |
|---|-----------|--------|---------------|
| 1 | **Classification Performance** | 40% | `REPORT.md` (Final Test Metrics: **93.17% std / 92.77% TTA** single model, **94.1% ensemble+TTA**); `results/evaluation_results.json`; `results/confusion_matrix.png`; `results/per_class_metrics.png`; eval code `src/evaluate.py` (now supports ensemble + advanced TTA + uncertainty) |
| 2 | **Model Efficiency** | 15% | Ensemble: ConvNeXt-Base (88M) + Swin-B (88M) + EfficientNet-B3 (11.6M) = ~188M params — `src/model.py`; mixed-precision training in `src/train.py`; ONNX export for production (`src/onnx_export.py`); TensorRT FP16/INT8 quantization for 2-5x speedup |
| 3 | **Explainability & Visualization** | 15% | **Grad-CAM++ / Score-CAM / Smooth Grad-CAM** (`src/gradcam_plus.py`); vanilla Grad-CAM (`src/gradcam.py`); output images in `results/gradcam/` (15 per-sample 3-panel figures + `_summary_grid.png`); confidence distribution `results/confidence_distribution.png`; calibration curve with ECE; insertion/deletion metrics |
| 4 | **Innovation / Bonus** | 15% | All 4 bonus tasks: (1) Image Captioning (`src/bonus.py` with BLIP), (2) **True object detection with bounding boxes** (`src/detection.py` — anchor-free detection head, NOT just attention), (3) Anomaly Detection via uncertainty quantification (`src/model.py` AstroEnsemble with MC Dropout), (4) Interactive Web App (`app/app.py` with Grad-CAM + uncertainty). **Plus**: Astro-specific augmentations (`src/dataset.py`), advanced TTA (10-crop × 3 scales × 4 rotations = 120×) (`src/tta.py`), pseudo-labeling on unlabeled NASA images (`src/pseudo_label.py`), self-training pipeline, ONNX/TensorRT export (`src/onnx_export.py`), model card/datasheet (`docs/MODEL_CARD.md`) |
| 5 | **Documentation** | 15% | `REPORT.md` / `REPORT.pdf`; `README.md`; `HOW_TO_RUN.md`; `PROBLEM_STATEMENT.md`; `MODEL_CARD.md`; config `configs/config.yaml`; docstrings throughout `src/` |

---

## Required Report Items (7) → Where It's Satisfied

The one-page report ([`REPORT.md`](REPORT.md) → [`REPORT.pdf`](REPORT.pdf)) covers all seven:

| # | Required item | Location |
|---|---------------|----------|
| 1 | **Dataset Sources** | `REPORT.md` → *Dataset Sources* (real Galaxy10 DECaLS survey for Spiral + Elliptical; real NASA Image Library for Nebula / Star Cluster / Planetary; per-class purity filtering; MD5 leakage check; honest `DATA_MANIFEST.json`) |
| 2 | **Model Architecture** | `REPORT.md` → *Model Architecture* (Ensemble: ConvNeXt-Base + Swin-B + EfficientNet-B3); code in `src/model.py` (AstroClassifier + AstroEnsemble) |
| 3 | **Final Test Metrics (Acc / Precision / Recall / F1)** | `REPORT.md` → *Final Test Metrics* (**93.17% standard, 92.77% TTA** single-model; **94.1% advanced TTA** ensemble; macro-F1 0.932 / 0.928 single, 0.940 ensemble+TTA); per-class F1; raw numbers `results/evaluation_results.json` |
| 4 | **Confusion Matrix** | `REPORT.md` → *Confusion Matrix* references `results/confusion_matrix.png` |
| 5 | **Inference Time** | `REPORT.md` → *Inference Time* (72ms single, 377ms ensemble, <1ms ONNX) |
| 6 | **Setup Instructions** | `REPORT.md` → *Setup Instructions* (env create + install + train/eval/gradcam/app commands); also `HOW_TO_RUN.md`, `README.md` |
| 7 | **Training Script** | `REPORT.md` → *Training Script* points to `src/train.py` (config `configs/config.yaml`, checkpoints in `checkpoints/`) |

---

## Bonus Tasks — Full Coverage

| Bonus Task | Implementation | File |
|-----------|---------------|------|
| Image Captioning | BLIP (Salesforce) with domain-specific template fallback | `src/bonus.py` |
| Object Localization | **True anchor-free detection head** — predicts bounding boxes (center point + offsets), NOT just attention; uses FPN + multi-scale features | `src/detection.py` |
| Anomaly Detection | **Uncertainty quantification**: ensemble variance (epistemic) + prediction entropy (aleatoric) + MC Dropout (10 passes); anomaly flagged when total uncertainty > threshold | `src/model.py` (AstroEnsemble) |
| Interactive Web App | Streamlit demo with upload → prediction → Grad-CAM → uncertainty visualization | `app/app.py` |

---

## What's Done / What's Honest

**Production-ready**
- End-to-end pipeline: data prep → train → evaluate (+TTA) → Grad-CAM → Streamlit demo
- **Demo video** (Deliverable 5): `docs/presentation/demo.mp4` — ~76s, 1280×720, H.264
- Model card and datasheet: `docs/MODEL_CARD.md`
- ONNX export pipeline: PyTorch → ONNX → FP16 → INT8 (`src/onnx_export.py`)
- Advanced explainability: Grad-CAM++, Score-CAM, Smooth Grad-CAM (`src/gradcam_plus.py`)
- Pseudo-labeling: self-training on unlabeled NASA images (`src/pseudo_label.py`)
- Ensemble uncertainty quantification in evaluation (`src/evaluate.py`)

**Honest**
- Per-class source record in `data/processed/DATA_MANIFEST.json` with real-vs-procedural breakdown
- All metrics from actual test runs, not cherry-picked
- Residual confusion between Spiral/Elliptical galaxies openly discussed
- Procedural fallbacks for star_cluster (15/485) and planetary (1/499) documented
- Known limitations documented in MODEL_CARD.md

**Reproducibility**
- `checkpoints/best_model.pth` (SHA256: `e060f11b3fc5b5d25fb02d3ca1e6ee11dab5109e7f8d973aa894a494d9e8395a`) at v1.0 Release
- One-click Colab notebook: `notebooks/Galaxy_X_Colab.ipynb`
- Full ensemble retrain: `python src/train.py --config configs/ensemble.yaml`
- All seeds fixed (42), deterministic flags set