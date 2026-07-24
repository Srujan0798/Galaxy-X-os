# Reproducibility — Galaxy-X-os

## TL;DR

| Claim | Source of truth | How to reproduce |
|---|---|---|
| 93.17% test acc / 92.77% TTA / 0.932 macro F1 | `results/evaluation_results.json` | Run `notebooks/Galaxy_X_Colab.ipynb` end-to-end on a Colab T4 (free). |
| 15 Grad-CAM figures | `results/gradcam/*.png` | `python src/gradcam.py` with the trained checkpoint. |
| Confusion matrix + per-class F1 | `results/confusion_matrix.png`, `results/per_class_metrics.png` | `python src/evaluate.py`. |

## Two distinct manifests — do not confuse them

1. **`data/processed/DATA_MANIFEST.json` (committed)** — a small, no-auth,
   procedural-only manifest (~50 images) kept so the repo is portable and the
   integration test (`tests/integration/test_pipeline.py`) runs without network.
   **It is NOT the training manifest.** It exists to prove the pipeline runs.
2. **Colab training manifest** — produced by `notebooks/Galaxy_X_Colab.ipynb`
   on a Colab T4 with real Galaxy10 + Kaggle data (creds via the notebook's
   Kaggle-token cell). The reported 93.17% / 92.77% come from this run. The
   per-class F1s in `results/evaluation_results.json` (Spiral 0.884, Elliptical
   0.895, Nebula/Cluster/Planetary 1.000) reflect real imagery for the two
   galaxy classes and Kaggle/procedural for the rest.

## Checkpoint integrity

`checkpoints/best_model.pth` (~141 MB) exceeds GitHub's 100 MB limit, so it is
**not committed**. Two paths to get it:

1. **Reproduce** (guaranteed): run the Colab notebook or
   `python src/prepare_data.py && python src/train.py` on a GPU. ~30 min on Colab T4.
2. **Download** (if published): fetch from the
   [v1.0 Release](https://github.com/Srujan0798/Galaxy-X-os/releases/tag/v1.0)
   and drop in `checkpoints/`.

After downloading, verify integrity against this hash:

```
SHA256: e060f11b3fc5b5d25fb02d3ca1e6ee11dab5109e7f8d973aa894a494d9e8395a
Size:   134 MB (140328300 bytes)
```

```bash
shasum -a 256 checkpoints/best_model.pth   # macOS
sha256sum checkpoints/best_model.pth       # Linux
```

> **Release creation note:** the v1.0 git tag is pushed. To publish the
> checkpoint + demo video as release assets, run (the fine-grained PAT used by
> `gh` here lacks the `contents:write` permission for releases, so this must be
> run with a token that has release scope, e.g. a classic PAT with `repo` scope):
>
> ```bash
> gh release create v1.0 checkpoints/best_model.pth docs/presentation/demo.mp4 \
>   --title "v1.0 — Trained EfficientNet-B3 (93.17% / 92.77% TTA)" \
>   --notes "Final trained model. SHA256: e060f11b3fc5b5d25fb02d3ca1e6ee11dab5109e7f8d973aa894a494d9e8395a. Drop in checkpoints/best_model.pth then run python src/evaluate.py to reproduce 93.17%/92.77% TTA. Demo video: docs/presentation/demo.mp4."
> ```

## Environment

| Component | Version |
|---|---|
| Python | 3.10+ |
| PyTorch | 2.4.1 |
| timm | 1.0.11 |
| albumentations | 1.4.18 |
| grad-cam | 1.5.2 |

Full pinned list: `requirements.txt`. Lint: `ruff check src/ app/ tests/`.
Tests: `pytest tests/ -v`.

## Real data sources (all named in PROBLEM_STATEMENT.md)

| Source | Used for | Auth | Status |
|---|---|---|---|
| Galaxy10 DECaLS/SDSS (`astroNN`) | Spiral + Elliptical galaxy | None | Used |
| ESA Hubble FITS-liberator | Nebula / Star Cluster / Planetary | None | Used (new) |
| Kaggle `fedesoriano/deep-space-images` | Nebula / Star Cluster | Kaggle token | Used |
| Kaggle `brsdincer/planetary-solar-system-objects` | Planetary | Kaggle token | Used |
| Galaxy Zoo (data.galaxyzoo.org) | (Reference — Galaxy10 is its derivative) | None | Referenced |
| NASA PDS / ESA PSA | (Reference for planetary mission archives) | None | Referenced |

## Cross-survey generalization / OOD

`tests/unit/test_localization.py` and the bonus CLI exercise the anomaly
detector on the classifier's softmax. A full cross-survey OOD evaluation
(hold out ESA Hubble images not seen in training, measure OOD flag rate) is
planned; the infrastructure (`detect_anomaly`) is in place.