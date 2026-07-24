# Contributing

## Quick Loop

```bash
pip install -r requirements.txt
pytest tests/ -v            # unit tests
ruff check src/             # lint
```

## Code Style

- Python 3.10+, PEP 8, ruff
- Docstrings: Google style
- Type annotations on public functions

## Testing

- Unit tests: `tests/unit/`
- Integration tests: `tests/integration/` (currently empty)
- e2e tests: `tests/e2e/` (currently empty)
- Run all: `pytest tests/ -v`

## Project Layout

- `src/` — main pipeline (prepare_data, train, evaluate, gradcam, inference, bonus)
- `app/` — Streamlit web demo
- `configs/config.yaml` — single source of truth
- `notebooks/Galaxy_X_Colab.ipynb` — one-click Colab pipeline
- `data/`, `checkpoints/`, `results/` — gitignored (reproducible via `src/prepare_data.py` + `src/train.py`)
