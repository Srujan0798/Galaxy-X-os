# Phase 5 — Brownies / Innovation — Evidence Report

## Orphan File Decisions

| File | Decision | Rationale |
|------|----------|-----------|
| `src/bonus.py` | **Keep** | Already solid: CLI works, template caption + anomaly detection + localization all functional. 20 unit tests pass. |
| `src/detection.py` | → `src/attic/detection.py` | Real detection head code (`DetectionHead`, `AstroDetector`, `decode_detections`) but completely unwired — no training loop or dataset. Moving to attic preserves code for later use. |
| `src/gradcam_plus.py` | → `src/attic/gradcam_plus.py` | Advanced CAM methods (`CAMComputer` with gradcam++/scorecam/layercam/smoothcam) but the existing `gradcam.py` uses the `pytorch_grad_cam` library. Wiring would require incompatible restructuring. Experimental `compute_explainability_metrics` is just `pass`. |
| `src/onnx_export.py` | **Keep + CLI added** | Working code with `export_to_onnx`, `check_onnx_model`, `quantize_onnx`, `benchmark_onnx`. Added `main()` with argparse; `--help` verified. 3 unit tests written. |
| `src/pseudo_label.py` | → `src/attic/pseudo_label.py` | Experimental self-training pipeline (`PseudoLabelDataset`, `generate_pseudo_labels`, `pseudo_labeling_pipeline`) depending on external NASA downloads. Movable; not wired into any production path. |

## Test Results

```
tests/unit/test_bonus.py .............. 20 passed
tests/unit/test_localization.py ..... 5 passed
tests/unit/test_ood.py .............. 6 passed
tests/unit/test_onnx_export.py ...... 3 passed
```

## Feature Status

- **OOD panel** — Working (tested via `test_ood.py` and `bonus.detect_anomaly`)
- **Template caption** — Working (20 tests in `test_bonus.py`)
- **Localization bbox overlay** — Working via Grad-CAM threshold (`localize_object` in `bonus.py`; 5 tests in `test_localization.py`)
- **ONNX export** — CLI functional (`python3 src/onnx_export.py --help`)
- **GradCAM++** — Moved to attic; existing `gradcam.py` uses `pytorch_grad_cam` library

## Orphan Module Audit

All 11 `src/*.py` modules (excluding `__init__.py` and `attic/`) are either imported by the training/evaluation pipeline, tested, or part of the CLI:
- `bonus`, `dataset`, `download_archives`, `evaluate`, `gradcam`, `inference`, `model`, `prepare_data`, `train`, `utils` — all imported/wired
- `onnx_export` — standalone CLI with tests
