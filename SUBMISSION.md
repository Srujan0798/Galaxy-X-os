# Galaxy-X-os — Final Submission Checklist

**SCALE x ODYSSEY** | Astronomical image classifier (EfficientNet-B3, 5 classes)
**GitHub repo:** https://github.com/Srujan0798/Galaxy-X-os
**Latest release:** [v1.0](https://github.com/Srujan0798/Galaxy-X-os/releases/tag/v1.0)
**One-page report:** [`REPORT.md`](REPORT.md) / [`REPORT.pdf`](REPORT.pdf)
**Scoreboard:** [`docs/SCOREBOARD.md`](docs/SCOREBOARD.md)

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
| 1 | **Classification Performance** | 40% | `REPORT.md` (Final Test Metrics: **93.17% / 92.77% TTA**, macro F1 0.932); `results/evaluation_results.json`; `results/confusion_matrix.png`; eval code `src/evaluate.py`. **Note:** single-model EfficientNet-B3, not ensemble. Full test set not committed due to size (~2500 images). Reproduce via Colab notebook or `prepare_data.py` + `train.py`. |
| 2 | **Model Efficiency** | 15% | EfficientNet-B3 ~11.6M params — `src/model.py`; mixed-precision training `src/train.py`; ONNX export ready (`src/onnx_export.py`); latency ~72ms (MPS median), <<5s on consumer hardware |
| 3 | **Explainability & Visualization** | 15% | Grad-CAM (`src/gradcam.py`); output images in `results/gradcam/` (15 per-sample 3-panel figures + `_summary_grid.png`); Streamlit app shows live Grad-CAM on upload/sample |
| 4 | **Innovation / Bonus** | 15% | 4 bonus tasks: (1) Template caption (`src/bonus.py`, BLIP optional), (2) Localization from Grad-CAM heatmap (`src/bonus.py`), (3) OOD/Anomaly detection via softmax entropy (`src/bonus.py`), (4) Streamlit web app (`app/app.py`). Experimental files moved to `attic/` (see `docs/SCOPE_GUARD.md`). |
| 5 | **Documentation** | 15% | `REPORT.md` / `REPORT.pdf`; `README.md`; `HOW_TO_RUN.md`; `PROBLEM_STATEMENT.md`; `MODEL_CARD.md`; `SUBMISSION.md`; config `configs/config.yaml`; docstrings throughout `src/` |

---

## Required Report Items (7) → Where It's Satisfied

The one-page report ([`REPORT.md`](REPORT.md) → [`REPORT.pdf`](REPORT.pdf)) covers all seven:

| # | Required item | Location |
|---|---------------|----------|
| 1 | **Dataset Sources** | `REPORT.md` → *Dataset Sources* (real Galaxy10 DECaLS for Spiral+Elliptical; NASA Image Library for Nebula/Star Cluster/Planetary; procedural fallback documented in `DATA_MANIFEST.json`) |
| 2 | **Model Architecture** | `REPORT.md` → *Model Architecture* (EfficientNet-B3 with custom classifier head); code in `src/model.py` |
| 3 | **Final Test Metrics (Acc/Precision/Recall/F1)** | `REPORT.md` → *Final Test Metrics* (93.17% standard, 92.77% TTA; macro-F1 0.932/0.928); raw numbers `results/evaluation_results.json` |
| 4 | **Confusion Matrix** | `REPORT.md` → *Confusion Matrix* references `results/confusion_matrix.png` |
| 5 | **Inference Time** | `REPORT.md` → *Inference Time* (~72ms single MPS median; <<5s on consumer hardware) |
| 6 | **Setup Instructions** | `REPORT.md` → *Setup Instructions* (env + install + train/eval/gradcam/app); also `HOW_TO_RUN.md`, `README.md` Quick Start |
| 7 | **Training Script** | `REPORT.md` → *Training Script* points to `src/train.py` (config `configs/config.yaml`, checkpoints in `checkpoints/`) |

---

## Bonus Tasks — Coverage

| Bonus Task | Implementation | File | Status |
|-----------|---------------|------|--------|
| Image Captioning | Template captioner (deterministic, no download). Optional BLIP if transformers installed. | `src/bonus.py` | Working in app |
| Object Localization | Pseudo-bbox from Grad-CAM heatmap thresholding (NOT trained detector) | `src/bonus.py` | Working in app |
| Anomaly Detection | Softmax entropy + max-probability screening | `src/bonus.py` | Working in app |
| Interactive Web App | Streamlit: upload/sample → prediction → Grad-CAM → caption → OOD | `app/app.py` | Working |

---

## What's Done / What's Honest

**Implemented**
- End-to-end pipeline: data prep → train → evaluate (+TTA) → Grad-CAM → Streamlit demo
- Demo video: `docs/presentation/demo.mp4` — ~76s, 1280×720, H.264 (note: pre-dates sample-button UI)
- Model card: `docs/MODEL_CARD.md`
- ONNX export pipeline: `src/onnx_export.py`
- All 4 bonus features working in app

**Honest limitations**
- Full dataset not committed (size); reproduce via Colab or `prepare_data.py` + `train.py`
- Demo samples are procedural — not for accuracy measurement
- Captions are template-based by default (BLIP optional, ~1GB download)
- Localization is saliency-based (Grad-CAM heatmap), not a trained detector
- Residual spiral/elliptical confusion discussed in REPORT.md
- Experimental features (advanced TTA, detection head, GradCAM++) in `attic/`

**Reproducibility**
- `checkpoints/best_model.pth` (download from v1.0 Release, SHA256: `e060f11b3fc5b5d25fb02d3ca1e6ee11dab5109e7f8d973aa894a494d9e8395a`)
- One-click Colab notebook: `notebooks/Galaxy_X_Colab.ipynb`
- All seeds fixed (42), deterministic flags set
- Artifact hashes: `results/ARTIFACT_HASHES.md`
