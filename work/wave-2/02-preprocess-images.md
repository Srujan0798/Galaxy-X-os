# Task: Preprocess All Images with AstroPreprocessor

## Context
Raw telescope images have noise, vignetting, cosmic rays, varying backgrounds.

## Goal
Apply the full AstroPreprocessor pipeline to all images.

## Pipeline Steps
1. Cosmic ray removal (cv2.inpaint)
2. Hot pixel correction (median blur)
3. Non-local means denoising
4. Background gradient removal
5. Vignetting correction
6. Dynamic range normalization (percentile clipping)
7. CLAHE contrast enhancement
8. Resize to 224x224

## Acceptance Criteria
- [ ] All images in data/processed/ are preprocessed
- [ ] Output size: 224x224 RGB
- [ ] Visual comparison: before/after for 5 sample images
- [ ] Preprocessing completes in <5 minutes for full dataset
- [ ] No crashes on edge cases (very dark, very bright, corrupted)

## Files to Modify
- `src/preprocess.py` (enhance if needed)
- Create preprocessing runner script

## Notes
The AstroPreprocessor class is already implemented in src/preprocess.py.
