# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2026-07-24

### Added
- **Real GPU training run (Colab T4)** — final model: **95.6% accuracy, 96.4% TTA, 0.96 macro F1**
- 15 Grad-CAM per-sample visualizations + summary grid (`results/gradcam/`)
- Evaluation artifacts committed: `evaluation_results.json`, confusion matrix, per-class metrics, confidence distribution
- `src/prepare_data.py` — real-first data pipeline with Kaggle credential support (KGAT_ token + legacy kaggle.json)
- One-click Colab pipeline (`notebooks/Galaxy_X_Colab.ipynb`)
- Honest per-class source tracking in `data/processed/DATA_MANIFEST.json`
- Disjoint stratified split with MD5 leakage check

### Fixed
- Critical import-when-loaded-as-package bug in `src/inference.py` (sys.path shim)
- Same bug in `src/train.py`, `src/train_head.py`, `src/evaluate.py`, `src/gradcam.py`
- Cryptic sklearn error when `--per-class` too small for 80/10/10 split (now clear message)
- `compute_metrics()` IndexError when not all classes present in labels
- Timeout in `test_model_forward` (switched to resnet18, pretrained=False)
- Hardcoded `config/config.yaml` path in `test_load_config`

### Changed
- Documentation synced to real 95.6% accuracy results (`README.md`, `REPORT.md`, `SUBMISSION.md`)
- Cell 2 of Colab notebook restructured with clear banner comments
- `.gitignore` updated to keep evaluation artifacts (PNGs + JSON + gradcam/)

## [0.1.0] - 2026-05-31

### Added
- Initial project setup per OS-Setup v1.3 (T1)
- SCALE x ODYSSEY astronomical image classifier
- EfficientNet-B3 backbone with custom head
- 8 astronomy-specific augmentations (cosmic ray, vignetting, Poisson noise)
- Progressive unfreezing training strategy
- Test-Time Augmentation (TTA)
- Grad-CAM explainability with 3-panel visualizations
- Streamlit interactive web demo
- BLIP image captioning bonus feature
- Anomaly detection (low-confidence flagging)
- Complete evaluation pipeline (confusion matrix, per-class F1, confidence distribution)
- 3 Jupyter notebooks (EDA, Training, Evaluation)
- Central YAML configuration
- Full OS-Setup orchestrator apparatus
