# Task: Generate 15 Grad-CAM Samples

## Context
Grad-CAM is key for explainability score.

## Goal
Generate 15 diverse test samples with 3-panel figures.

## Acceptance Criteria
- [ ] 15 samples: 3 per class minimum
- [ ] 3-panel per sample: Original | True-CAM | Predicted-CAM
- [ ] Summary grid (_summary_grid.png)
- [ ] At least 12/15 correct predictions
- [ ] Green labels for correct, red for incorrect

## Files to Modify
- `src/gradcam.py` (if fixes needed)

## Notes
Run: python src/gradcam.py
Output: results/gradcam/
