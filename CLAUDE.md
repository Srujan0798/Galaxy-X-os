# Galaxy-X-os — Orchestrator Kernel (CLAUDE.md)

> Auto-loaded by Claude Code. Interchangeable with KIMI.md.

## Project Identity

- **Name:** Galaxy-X-os (SCALE x ODYSSEY)
- **Goal:** Classify raw astronomical images into 5 celestial categories using deep learning
- **Domain:** Computer Vision / ML / Astronomical Imaging
- **Tech Stack:** Python, PyTorch, EfficientNet-B3, Albumentations, Streamlit, Grad-CAM
- **Tier:** T1 (Standard)
- **Final model:** EfficientNet-B3, **95.6% test accuracy / 96.4% TTA / 0.96 macro F1**

## Quick Commands

| Command | Purpose |
|---------|---------|
| `pip install -r requirements.txt` | Install dependencies |
| `python src/prepare_data.py` | Build the 5-class dataset (real-first) |
| `python src/train.py` | Train EfficientNet-B3 (needs GPU for reasonable time) |
| `python src/evaluate.py` | Evaluate (standard + TTA) |
| `python src/gradcam.py` | Generate Grad-CAM visualizations |
| `streamlit run app/app.py` | Launch Streamlit demo |
| `pytest tests/ -v` | Run all tests |
| `ruff check src/` | Lint |

For a one-click Colab GPU run: `notebooks/Galaxy_X_Colab.ipynb` → `Runtime → Run all`.

## Architecture Overview

```
Raw Images → Preprocess → AstroDataset → EfficientNet-B3 → 5-Class Output
                ↓              ↓              ↓
          Augmentations   DataLoaders    Grad-CAM / Streamlit
```

## Directory Map

| Path | Purpose |
|------|---------|
| `src/` | Source code (prepare_data, train, evaluate, gradcam, inference, bonus, model, dataset, utils) |
| `app/` | Streamlit web demo |
| `notebooks/` | `Galaxy_X_Colab.ipynb` — one-click Colab pipeline |
| `configs/` | `config.yaml` — single source of truth |
| `data/` | Raw + processed datasets (gitignored, reproducible) |
| `checkpoints/` | Model weights (gitignored — 134 MB) |
| `results/` | Evaluation artifacts (committed: PNGs + JSON + gradcam/) |
| `tests/` | pytest suites (unit/, integration/, e2e/) |
| `docs/` | Runbooks, conventions, decision log, wave briefs |
| `attic/`, `docs/historical/` | Archived orchestration scaffolding |

## Risk Tier

This project is **T1 — Standard**.
- No customer PII
- No compliance requirements
- No production SLA
- Internal tool / hackathon submission

## Blast Radius

| Action | Radius | Gate |
|--------|--------|------|
| Read files, run tests | r0 | Auto |
| Write to src/, modify tests | r1 | Log + proceed |
| Add deps, change CI | r2 | Await approval |
| rm -rf, force-push | r3 | Block |

## Core Rules

1. All waves are complete — submission ready.
2. Never delete — move to `attic/` or `docs/historical/`.
3. Keep `REPORT.md`, `README.md`, `SUBMISSION.md` in sync with real numbers.
4. `configs/config.yaml` is the single source of truth.
5. Commit evaluation artifacts (PNGs + JSON), not the 134 MB `.pth`.
