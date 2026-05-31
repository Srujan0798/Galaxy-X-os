# Task: Checkpoint Management + Best Model Selection

## Context
Training generates multiple checkpoints. Need to select best.

## Goal
Evaluate all checkpoints and select the best one.

## Acceptance Criteria
- [ ] Best checkpoint identified by val accuracy
- [ ] Copy best to checkpoints/best_model.pth
- [ ] Generate checkpoint comparison table
- [ ] Verify checkpoint loads correctly
- [ ] Document which epoch was best

## Files to Create
- `results/checkpoint_comparison.md`

## Notes
Use load_checkpoint() from utils.py to verify.
