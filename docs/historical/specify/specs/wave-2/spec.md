# Wave 2 Spec — Real Data Integration

## Goal
Replace synthetic data with real astronomical datasets from Kaggle.

## Deliverables
- Downloaded Galaxy Zoo, DeepSky, Planetary datasets
- Preprocessed 5-class dataset with AstroPreprocessor
- Stratified 80/10/10 splits
- Computed class weights for imbalance handling
- Data quality validation report

## Acceptance Criteria
- [ ] Real images in data/raw/
- [ ] Processed images in data/processed/{train,val,test}/
- [ ] class_weights.json generated
- [ ] split_statistics.json generated
- [ ] Data quality report (no corrupted images)
- [ ] Dataset loads without errors
