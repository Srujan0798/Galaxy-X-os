# Task: Compute Class Weights + Validate Data

## Context
Class imbalance is common in astronomical datasets.

## Goal
Compute inverse-frequency class weights and validate entire dataset.

## Acceptance Criteria
- [ ] class_weights.json generated with 5 weights
- [ ] Weights are inverse-frequency normalized
- [ ] No corrupted images in dataset
- [ ] Data validation report: counts, sizes, formats
- [ ] Dataset loads successfully via AstroDataset

## Files to Create/Modify
- `data/processed/class_weights.json`
- Create validation report

## Notes
class_weights.json format: {"spiral_galaxy": 1.0, ...}
