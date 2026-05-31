# Task: Generate Stratified Train/Val/Test Splits

## Context
Need reproducible, stratified splits for fair evaluation.

## Goal
Create 80/10/10 train/val/test splits with class balance preserved.

## Acceptance Criteria
- [ ] data/processed/{train,val,test}/ each has 5 class subdirectories
- [ ] Train: 80%, Val: 10%, Test: 10% (per class)
- [ ] Stratified (same class distribution in all splits)
- [ ] split_statistics.json generated with counts per split/class
- [ ] Reproducible (fixed random seed 42)
- [ ] Validation: no data leakage between splits

## Files to Create/Modify
- Create split generation script
- `data/processed/split_statistics.json`

## Notes
Use sklearn.model_selection.StratifiedShuffleSplit or manual stratification.
