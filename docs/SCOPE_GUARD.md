# SCOPE_GUARD — Galaxy-X-os

## In Scope

- 5-class astronomical image classification (Spiral, Elliptical, Nebula, Star Cluster, Planetary)
- Training, evaluation, inference
- Grad-CAM explainability (localization as optional bonus via heatmap inspection)
- Streamlit web demo
- BLIP captioning + anomaly detection (bonus)
- **EfficientNet-B3** is the primary backbone
- Reproducible pipeline (Colab + local)

## Out of Scope

- Real-time telescope feeds
- Object detection / segmentation (localization only via Grad-CAM heatmap)
- Production deployment / serving infrastructure
- Multi-language support
- Commercial API integrations (KaggleHub for free datasets is OK; no paid APIs)
- Distributed training (single-GPU focus)

## Backbone Policy

- `efficientnet_b3` from `timm` is the primary backbone
- Other backbones (convnext, resnet) may appear in ensemble evaluation only
