# Steering

## North Star
> Classify raw telescope images into 5 celestial categories. Current model achieves **93.17% test accuracy (92.77% with TTA, 0.93 macro F1)**.

## Constraints
- No handcrafted features
- Inference under 5 seconds per image on consumer hardware (~72 ms median on Apple MPS)
- Must include Grad-CAM explainability
- Must include interactive demo
