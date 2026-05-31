# Task: Evaluation + Inference

## Context
Training pipeline ready. Need evaluation and fast inference.

## Goal
Create `src/evaluate.py`, `src/inference.py`, `src/gradcam.py`, `app/app.py`.

## Acceptance Criteria
- [ ] `evaluate.py`: metrics + TTA + confusion matrix + plots
- [ ] `inference.py`: ModelManager singleton + <15ms inference
- [ ] `gradcam.py`: 3-panel figures + summary grid
- [ ] `app/app.py`: Streamlit demo with upload + predict + Grad-CAM

## Files
- Create: `src/evaluate.py`, `src/inference.py`, `src/gradcam.py`, `app/app.py`
- Create: `tests/unit/test_inference.py`
