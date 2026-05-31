# Task: Run Full 50-Epoch Training

## Context
Model architecture and data pipeline are ready. Need to train.

## Goal
Run complete training pipeline for 50 epochs on real data.

## Acceptance Criteria
- [ ] Training starts without errors
- [ ] Progressive unfreezing works correctly
- [ ] Mixed precision (torch.amp) active
- [ ] OneCycleLR scheduler updates each epoch
- [ ] Early stopping triggers if no improvement for 12 epochs
- [ ] Best checkpoint saved to checkpoints/best_model.pth
- [ ] Periodic checkpoints every 5 epochs
- [ ] Training log file generated

## Files to Modify
- `src/train.py` (minor fixes if needed)

## Notes
Expected time: ~25 min on RTX 4090, ~2-3 hours on CPU
Monitor GPU memory usage.
