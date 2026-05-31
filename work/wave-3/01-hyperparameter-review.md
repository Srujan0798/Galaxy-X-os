# Task: Hyperparameter Review + Config Tuning

## Context
Current config has defaults. Need to optimize for real data.

## Goal
Review and tune hyperparameters based on real dataset characteristics.

## Parameters to Review
- learning_rate: 3e-4 (may need adjustment)
- batch_size: 32 (depends on GPU memory)
- freeze_backbone_epochs: 3 (may increase)
- patience: 12 (may need longer)
- label_smoothing: 0.1
- dropout: 0.4

## Acceptance Criteria
- [ ] Config optimized for real data size
- [ ] Batch size fits in GPU memory
- [ ] LR schedule appropriate for dataset size
- [ ] Document rationale for each choice in ADR

## Files to Modify
- `configs/config.yaml`
- `docs/decisions/003-hyperparameters.md`
