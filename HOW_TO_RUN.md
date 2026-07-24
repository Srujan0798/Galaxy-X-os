# HOW_TO_RUN.md

## Quick Start (Most Readers)

```bash
pip install -r requirements.txt

# Prepare the 5-class dataset (real-first, procedural fallback for missing Kaggle creds)
python src/prepare_data.py --per-class 500

# Train
python src/train.py

# Evaluate (standard + TTA)
python src/evaluate.py

# Grad-CAM
python src/gradcam.py

# Web demo
streamlit run app/app.py

# Tests + lint
pytest tests/ -v
ruff check src/
```

The trained model + results in `results/` + `checkpoints/` were produced with `python src/train.py` on a GPU (see `notebooks/Galaxy_X_Colab.ipynb` for a reproducible Colab run).

## Makefile Shortcuts

```bash
make install      # pip install -r requirements.txt
make train        # python src/train.py
make evaluate     # python src/evaluate.py
make gradcam      # python src/gradcam.py
make app          # streamlit run app/app.py
make test         # pytest tests/ -v
make lint         # ruff check src/ && mypy src/
make split        # python src/generate_splits.py (legacy, prefer prepare_data.py)
make train-head   # python src/train_head.py (head-only, low-RAM machines)
```

## Colab One-Click Pipeline

Open `notebooks/Galaxy_X_Colab.ipynb` in Google Colab, set runtime to **GPU (T4)**,
paste your Kaggle token into Cell 2 (optional, unlocks real nebula/cluster/planetary
images), then `Runtime → Run all`. The final cell produces `results.zip` — download
it and unzip into the repo root.

## Project Structure (Orchestrator Reference)

- `src/` — main Python modules (prepare_data, train, evaluate, gradcam, inference, bonus, utils, dataset, augmentations, model, preprocess)
- `app/` — Streamlit web demo
- `configs/config.yaml` — single source of truth for hyperparameters + paths
- `notebooks/Galaxy_X_Colab.ipynb` — one-click GPU pipeline (recommended)
- `data/raw/`, `data/processed/` — datasets (gitignored, reproducible)
- `checkpoints/best_model.pth` — trained weights (gitignored, too large)
- `results/` — evaluation artifacts (committed: PNGs + JSON + gradcam/)
- `tests/` — pytest suites (unit/, integration/, e2e/)
- `docs/` — runbooks, conventions, decision log, wave briefs
- `attic/`, `docs/historical/` — archived orchestration scaffolding
