# Galaxy-X-os — Final Submission Checklist

**SCALE x ODYSSEY** | Astronomical image classifier (EfficientNet-B3, 5 classes)
**GitHub repo:** https://github.com/Srujan0798/Galaxy-X-os
**One-page report:** [`REPORT.md`](REPORT.md) / [`REPORT.pdf`](REPORT.pdf) (upload this PDF)

Classes: Spiral Galaxy · Elliptical Galaxy · Nebula · Star Cluster · Planetary Object

---

## How to Submit (Google Form)

1. **Name:** enter your full name.
2. **GitHub link:** paste `https://github.com/Srujan0798/Galaxy-X-os`
3. **File upload (optional, 1 file, PDF/doc, ≤10MB):** upload [`REPORT.pdf`](REPORT.pdf).
4. Submit.

> Before submitting, confirm the repo is pushed to GitHub and public so graders can clone it.

---

## Scoring Criteria → Where It's Satisfied

| # | Criterion | Weight | Where in repo |
|---|-----------|--------|---------------|
| 1 | **Classification Performance** | 40% | `REPORT.md` (Final Test Metrics: **95.6% std / 96.4% TTA**, macro-F1 0.956 / 0.964); `results/evaluation_results.json`; `results/confusion_matrix.png`; `results/per_class_metrics.png`; eval code `src/evaluate.py` |
| 2 | **Model Efficiency** | 15% | EfficientNet-B3 (~11.6M params) — `src/model.py`; mixed-precision + progressive unfreezing in `src/train.py`; inference time in `REPORT.md`; TTA in `src/evaluate.py` |
| 3 | **Explainability & Visualization** | 15% | Grad-CAM module `src/gradcam.py`; output images in `results/gradcam/` (15 per-sample 3-panel figures + `_summary_grid.png`); confidence distribution `results/confidence_distribution.png` |
| 4 | **Innovation / Bonus** | 15% | Astro-specific augmentations (cosmic-ray sim, vignetting, Poisson noise) `src/augmentations.py`; TTA (6× aug) in `src/evaluate.py`; bonus features `src/bonus.py`; real-first data pipeline `src/prepare_data.py` with honest procedural fallback (`data/processed/DATA_MANIFEST.json`); Streamlit demo `app/app.py` |
| 5 | **Documentation** | 15% | `REPORT.md` / `REPORT.pdf`; `README.md`; `HOW_TO_RUN.md`; `PROBLEM_STATEMENT.md`; config `configs/config.yaml`; docstrings throughout `src/` |

---

## Required Report Items (7) → Where It's Satisfied

The one-page report ([`REPORT.md`](REPORT.md) → [`REPORT.pdf`](REPORT.pdf)) covers all seven:

| # | Required item | Location |
|---|---------------|----------|
| 1 | **Dataset Sources** | `REPORT.md` → *Dataset Sources* (real Galaxy10 DECaLS survey imagery for Spiral + Elliptical; real NASA Image Library imagery (images.nasa.gov, no API key) for Nebula / Star Cluster / Planetary with per-class purity filtering, Kaggle + labelled procedural as fallbacks only; disjoint stratified split, MD5 leakage check, honest `DATA_MANIFEST.json`) |
| 2 | **Model Architecture** | `REPORT.md` → *Model Architecture* (EfficientNet-B3, transfer learning, AdamW, OneCycleLR); code in `src/model.py` |
| 3 | **Final Test Metrics (Acc / Precision / Recall / F1)** | `REPORT.md` → *Final Test Metrics* (**95.6% std, 96.4% TTA**, macro-F1 0.956 / 0.964, per-class F1); raw numbers `results/evaluation_results.json` |
| 4 | **Confusion Matrix** | `REPORT.md` → *Confusion Matrix* references `results/confusion_matrix.png` (+ `results/per_class_metrics.png`) |
| 5 | **Inference Time** | `REPORT.md` → *Inference Time* |
| 6 | **Setup Instructions** | `REPORT.md` → *Setup Instructions* (env create + install + train/eval/gradcam/app commands); also `HOW_TO_RUN.md`, `README.md` |
| 7 | **Training Script** | `REPORT.md` → *Training Script* points to `src/train.py` (config `configs/config.yaml`, checkpoints in `checkpoints/`) |

---

## Grad-CAM Outputs Present (`results/gradcam/`)

15 per-sample 3-panel figures, `sample_01..15_*.png` (from the final 95.6%-accuracy run), plus `_summary_grid.png`. Includes both correct and incorrect predictions so reviewers can see where the model actually looks.

---

## What's Done / What's Honest About the Pipeline

**Done**
- End-to-end pipeline: data prep → train → evaluate (+TTA) → Grad-CAM → Streamlit demo.
- **Demo video** (Deliverable 5): [`docs/presentation/demo.mp4`](docs/presentation/demo.mp4) — 76 s, 1280×720, H.264. Walks through architecture → metrics → confusion matrix → per-class F1 → confidence distribution → all 15 Grad-CAM samples → summary grid. Script: [`docs/presentation/Demo_Video_Script.md`](docs/presentation/Demo_Video_Script.md).
- Real data sources: Galaxy10 DECaLS survey imagery (via astroNN) for Spiral + Elliptical; **NASA Image Library** (images.nasa.gov, no API key — Hubble/Spitzer/JPL) for Nebula / Star Cluster / Planetary, retrieved by keyword with per-class purity filtering (`src/download_archives.py`). Kaggle deep-space images (`fedesoriano/deep-space-images`, `brsdincer/planetary-solar-system-objects`) and a clearly-labelled procedural fallback remain as fallbacks only, so the pipeline never breaks. All five classes can now be built from real imagery with no API key.
- Honest per-class source record in `data/processed/DATA_MANIFEST.json` (rebuilt by `python src/prepare_data.py`).
- Disjoint stratified split, verified by MD5 hash that no image appears in >1 split.
- Full one-page report with all 7 required items, exported to PDF.
- 15 Grad-CAM figures + summary grid committed under `results/gradcam/`.
- Device selection is portable: `get_device()` in `src/utils.py` prefers **CUDA > MPS > CPU**, and `src/gradcam.py` uses it — so the same code runs on a grader's GPU box or this MacBook.

**Honest note on training**
- Final model was trained on a **CUDA GPU (Google Colab, T4)** via `notebooks/Galaxy_X_Colab.ipynb` — full EfficientNet-B3 fine-tune, 250-image held-out test set.
- Final honest test result: **95.6% accuracy (96.4% with TTA)**, macro-F1 **0.956 / 0.964**.
- Residual confusion is between Spiral and Elliptical Galaxy (visually similar morphologies), surfaced openly in the report and Grad-CAM figures.
- The trained weights (`checkpoints/best_model.pth`, ~141 MB) exceed GitHub's 100 MB file limit, so they are **not** committed. Guaranteed path: reproduce them via `python src/prepare_data.py` → `python src/train.py` (or the one-click `notebooks/Galaxy_X_Colab.ipynb`). If published, they are also attached to the [v1.0 Release](https://github.com/Srujan0798/Galaxy-X-os/releases/tag/v1.0).
