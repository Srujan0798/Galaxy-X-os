# ARCHITECTURE.md

## Galaxy-X-os — System Architecture

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│  Raw Images │────▶│  AstroDataset │────▶│ DataLoader  │
│  (5 classes)│     │  + Albumentations│    │  (train/val/test)│
└─────────────┘     └──────────────┘     └──────┬──────┘
                                                │
                                                ▼
┌──────────────────────────────────────────────────────────┐
│              EfficientNet-B3 Backbone                    │
│              (pretrained on ImageNet)                    │
└────────────────────────┬─────────────────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────────────┐
│              Custom Classification Head                  │
│  AdaptiveAvgPool → Flatten → BN → Dropout → Linear(512)│
│  → ReLU → BN → Dropout → Linear(256) → ReLU → BN →     │
│  Dropout → Linear(5)                                     │
└────────────────────────┬─────────────────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────────────┐
│                     Output Layer                         │
│              Softmax → 5-Class Probabilities             │
└────────────────────────┬─────────────────────────────────┘
                         │
         ┌───────────────┼───────────────┐
         ▼               ▼               ▼
    ┌─────────┐    ┌──────────┐   ┌──────────┐
    │Grad-CAM │    │ Streamlit│   │  JSON    │
    │Heatmap  │    │   Demo   │   │ Metrics  │
    └─────────┘    └──────────┘   └──────────┘
```

## Component Design

### src/dataset.py
- `AstroDataset`: PyTorch Dataset with astronomy-specific transforms
- `get_train_transforms()`: 8 custom augmentations
- `get_val_transforms()`: minimal validation transforms
- `get_loaders()`: DataLoader factory

### src/model.py
- `AstroClassifier`: EfficientNet-B3 + custom head
- `freeze_backbone()`: for progressive unfreezing
- `get_gradcam_target_layer()`: backbone-agnostic CAM targeting

### src/train.py
- Progressive unfreezing (Phase 1 frozen, Phase 2 full)
- OneCycleLR scheduler
- Mixed precision (torch.amp)
- Weighted CrossEntropyLoss + label smoothing
- Early stopping + checkpointing

### src/evaluate.py
- Standard evaluation metrics
- Batched Test-Time Augmentation (6 variants)
- Confusion matrix, per-class F1, confidence distribution

### src/gradcam.py
- `explain_image()`: single-image Grad-CAM
- `visualize_predictions()`: 15 diverse samples, 3-panel figures
- Summary grid

### src/inference.py
- `ModelManager`: singleton-cached model loading
- `predict()`: single image, <15ms target
- `predict_batch()`: batched inference

### src/bonus.py
- `generate_caption_with_fallback()`: BLIP or template captions
- `AnomalyDetector`: low-confidence + ambiguous + entropy flags

### app/app.py
- Streamlit web demo
- Upload → predict → Grad-CAM overlay
- Cached model loading
