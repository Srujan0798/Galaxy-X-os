# Task: Download Real Astronomical Datasets

## Context
Currently using synthetic data. Need real telescope images for meaningful training.

## Goal
Download and merge 3 Kaggle datasets into a unified 5-class structure.

## Datasets
1. Galaxy Zoo (spiral + elliptical galaxies)
2. DeepSky (nebulae + star clusters)
3. Planetary (planetary objects)

## Acceptance Criteria
- [ ] data/raw/ contains 3 downloaded datasets
- [ ] Merged into data/processed/ with 5 subdirectories
- [ ] No corrupted images (validate with PIL)
- [ ] At least 500 images per class
- [ ] Download script is idempotent (won't re-download if exists)

## Files to Modify
- `src/download_datasets.py` (enhance)
- `data/raw/.gitkeep` (remove, add real data)

## Notes
Use Kaggle API or direct download links.
Handle rate limiting and retries.
