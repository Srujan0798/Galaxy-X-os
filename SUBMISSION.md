# Galaxy-X-os — Final Submission Checklist

**SCALE x ODYSSEY** | Astronomical image classifier (EfficientNet-B3, 5 classes)
**GitHub repo:** https://github.com/Srujan0798/Galaxy-X-os
**One-page report:** [`REPORT.md`](REPORT.md) / [`REPORT.pdf`](REPORT.pdf) (upload this PDF)

Classes: Spiral Galaxy · Elliptical Galaxy · Nebula · Star Cluster · Planetary Object

---

## How to Submit (Google Form)

1. **Name:** enter your full name.
2. **GitHub link:** paste `https://github.com/Srujan0798/Galaxy-X-os`
3. **File upload (optional, 1 file, PDF/doc, ≤10MB):** upload [`REPORT.pdf`](REPORT.pdf) (~95 KB — well under the limit).
4. Submit.

> Before submitting, confirm the repo is pushed to GitHub and public so graders can clone it.

---

## Scoring Criteria → Where It's Satisfied

| # | Criterion | Weight | Where in repo |
|---|-----------|--------|---------------|
| 1 | **Classification Performance** | 40% | `REPORT.md` (Final Test Metrics: 72.4% std / 74.4% TTA, macro-F1, per-class F1); `results/evaluation_results.json`; `results/confusion_matrix.png`; `results/per_class_metrics.png`; eval code `src/evaluate.py` |
| 2 | **Model Efficiency** | 15% | EfficientNet-B3 (~11.6M params) — `src/model.py`; mixed-precision + progressive unfreezing in `src/train.py`; inference time (~410 ms/img on MPS) in `REPORT.md`; `results/training_summary.json` |
| 3 | **Explainability & Visualization** | 15% | Grad-CAM module `src/gradcam.py`; output images in `results/gradcam/` (15 per-sample 3-panel figures + `_summary_grid.png`); training/confidence plots `results/training_curves.png`, `results/confidence_distribution.png`, `results/lr_schedule.png` |
| 4 | **Innovation / Bonus** | 15% | Astro-specific augmentations (cosmic-ray sim, vignetting, Poisson noise) `src/augmentations.py`; TTA (6× aug) in `src/evaluate.py`; bonus features `src/bonus.py`; real SDSS DR17 data pipeline `src/download_datasets.py`, `src/generate_from_real_data.py`; Streamlit demo `app/app.py` |
| 5 | **Documentation** | 15% | `REPORT.md` / `REPORT.pdf`; `README.md`; `HOW_TO_RUN.md`; `PROBLEM_STATEMENT.md`; config `config/config.yaml`; docstrings throughout `src/` |

---

## Required Report Items (7) → Where It's Satisfied

The one-page report ([`REPORT.md`](REPORT.md) → [`REPORT.pdf`](REPORT.pdf)) covers all seven:

| # | Required item | Location |
|---|---------------|----------|
| 1 | **Dataset Sources** | `REPORT.md` → *Dataset Sources* (SDSS DR17 + merged Kaggle galaxy-morphology sets; 80/10/10 split, 2000 train images) |
| 2 | **Model Architecture** | `REPORT.md` → *Model Architecture* (EfficientNet-B3, transfer learning, AdamW, OneCycleLR); code in `src/model.py` |
| 3 | **Final Test Metrics (Acc / Precision / Recall / F1)** | `REPORT.md` → *Final Test Metrics* (72.4% std, 74.4% TTA, macro-F1 0.720/0.742, per-class F1); raw numbers `results/evaluation_results.json` |
| 4 | **Confusion Matrix** | `REPORT.md` → *Confusion Matrix* references `results/confusion_matrix.png` (+ `results/per_class_metrics.png`) |
| 5 | **Inference Time** | `REPORT.md` → *Inference Time* (~410 ms/img on Apple MPS, batch=1, after warmup) |
| 6 | **Setup Instructions** | `REPORT.md` → *Setup Instructions* (env create + install + train/eval/gradcam/app commands); also `HOW_TO_RUN.md`, `README.md` |
| 7 | **Training Script** | `REPORT.md` → *Training Script* points to `src/train.py` (config `config/config.yaml`, checkpoints in `checkpoints/`) |

---

## Grad-CAM Outputs Present (`results/gradcam/`)

Already generated (do **not** regenerate — a training job is running):

- 15 latest per-sample 3-panel figures, `sample_01..15_*.png` (dated 2026-06-21), e.g.
  `sample_02_elliptical_galaxy_pred_elliptical_galaxy_1.00.png`,
  `sample_04_star_cluster_pred_star_cluster_1.00.png`,
  `sample_15_nebula_pred_nebula_1.00.png`
- `_summary_grid.png` — compact grid of all predicted-class CAMs
- (Older 2026-05-31 runs also remain in the folder; the 06-21 set is the current one.)

---

## What's Done / What's Hardware-Limited (honest note)

**Done**
- End-to-end pipeline: data prep → train → evaluate (+TTA) → Grad-CAM → Streamlit demo.
- Real SDSS DR17 astronomical data, not synthetic-only.
- Full one-page report with all 7 required items, exported to PDF.
- Explainability artifacts (15 Grad-CAM figures + summary grid) committed under `results/gradcam/`.
- Device selection is portable: `get_device()` in `src/utils.py` prefers **CUDA > MPS > CPU**, and `src/gradcam.py` uses it — so the same code runs on a grader's GPU box or this MacBook.

**Hardware-limited (be transparent)**
- Trained on an **8GB MacBook Air (Apple MPS, no CUDA)**. Full fine-tuning of EfficientNet-B3 was **memory-bound** at this RAM, so the final honest test result is **72.4% accuracy (74.4% with TTA)** rather than a larger-batch / longer-schedule GPU run.
- The 95% val-accuracy figure was a short CPU run on an earlier/smaller split and is **not** the headline number; the reported metric is the 72.4% MPS test result.
- Inference is ~410 ms/image on MPS; on a CUDA GPU (with `torch.autocast("cuda")` already supported in the code) this would drop to the low-millisecond range.
- Lower per-class F1 on Spiral vs Star Cluster reflects genuine visual similarity of diffuse structures, not a pipeline defect — surfaced openly in the report and Grad-CAM figures.
