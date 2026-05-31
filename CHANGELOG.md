# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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

## [Unreleased]

- Wave-1: Foundation setup and validation
- Wave-2: Data pipeline hardening
- Wave-3: Training optimization
- Wave-4: Demo polish
- Wave-5: Submission packaging
