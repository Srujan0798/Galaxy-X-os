# Task: Training Pipeline

## Context
Model ready. Need training loop with best practices.

## Goal
Create `src/train.py` with progressive unfreezing, mixed precision, checkpointing.

## Acceptance Criteria
- [ ] Progressive unfreezing (Phase 1 frozen, Phase 2 full)
- [ ] OneCycleLR scheduler
- [ ] Mixed precision with torch.amp
- [ ] Weighted CrossEntropyLoss + label smoothing
- [ ] Early stopping + best checkpoint saving
- [ ] TensorBoard logging

## Files
- Create: `src/train.py`
- Create: `tests/unit/test_train.py` (mocked)
