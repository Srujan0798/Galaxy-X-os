# Task: Model Architecture

## Context
Dataset ready. Need EfficientNet-B3 model with custom head.

## Goal
Create `src/model.py` with AstroClassifier and Grad-CAM hooks.

## Acceptance Criteria
- [ ] `AstroClassifier` uses EfficientNet-B3 backbone
- [ ] Custom head with BN, Dropout, residual-style layers
- [ ] `freeze_backbone()` and `unfreeze_backbone()` methods
- [ ] `get_gradcam_target_layer()` backbone-agnostic
- [ ] Tests pass: `pytest tests/unit/test_model.py -v`

## Files
- Create: `src/model.py`
- Create: `tests/unit/test_model.py`
