# Task: Data Pipeline Skeleton

## Context
Config is ready. Need PyTorch Dataset and augmentations.

## Goal
Create `src/dataset.py` with AstroDataset and astronomy-specific transforms.

## Acceptance Criteria
- [ ] `AstroDataset` loads images from train/val/test splits
- [ ] 8 custom augmentations implemented (cosmic ray, vignetting, noise)
- [ ] `get_loaders()` returns train/val/test DataLoaders
- [ ] Tests pass: `pytest tests/unit/test_dataset.py -v`

## Files
- Create: `src/dataset.py`
- Create: `tests/unit/test_dataset.py`
