#!/usr/bin/env python3
"""
SCALE x ODYSSEY -- Model Architecture

EfficientNet-B3 backbone with custom classification head.
Designed for 5-class astronomical object classification.
Easily swappable to ResNet-50, EfficientNet-B4, or ViT-B/16.
"""

import torch
import torch.nn as nn
import timm


class AstroClassifier(nn.Module):
    """
    Astronomical Object Classifier with transfer learning backbone.

    Architecture:
    - Pretrained EfficientNet-B3 (features only)
    - Global Average Pooling
    - BatchNorm + Dropout block
    - Hidden layer (512 -> 256) with skip connections
    - Output layer (5 classes)
    """

    def __init__(
        self,
        num_classes: int = 5,
        backbone: str = "efficientnet_b3",
        pretrained: bool = True,
        dropout: float = 0.4,
    ):
        super().__init__()

        self.num_classes = num_classes
        self.backbone_name = backbone

        # Load pretrained backbone without classifier head
        self.backbone = timm.create_model(
            backbone,
            pretrained=pretrained,
            num_classes=0,          # Remove default head
            global_pool="",         # We'll do our own pooling
        )

        # Get feature dimension from backbone
        in_features = self.backbone.num_features

        # Custom classification head with regularization
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),

            # Block 1
            nn.BatchNorm1d(in_features),
            nn.Dropout(dropout),
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),

            # Block 2
            nn.BatchNorm1d(512),
            nn.Dropout(dropout / 2),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),

            # Output
            nn.BatchNorm1d(256),
            nn.Dropout(dropout / 4),
            nn.Linear(256, num_classes),
        )

        # Initialize classifier weights (backbone already pretrained)
        self._init_weights()

    def _init_weights(self):
        """Initialize new layers with Kaiming initialization."""
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone.forward_features(x)
        out = self.classifier(features)
        return out

    def freeze_backbone(self):
        """Freeze backbone parameters for warmup phase."""
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        """Unfreeze backbone for fine-tuning."""
        for param in self.backbone.parameters():
            param.requires_grad = True

    def get_gradcam_target_layer(self):
        """
        Return the target layer for Grad-CAM visualization.
        This is the last convolutional block before global pooling.
        """
        if "efficientnet" in self.backbone_name:
            # EfficientNet: last MBConv block
            return [self.backbone.blocks[-1][-1]]
        elif "resnet" in self.backbone_name:
            # ResNet: last bottleneck block
            return [self.backbone.layer4[-1]]
        elif "vit" in self.backbone_name:
            # ViT: last transformer block (Grad-CAM less effective here)
            return [self.backbone.blocks[-1].norm1]
        else:
            # Fallback: try to infer
            return [list(self.backbone.children())[-2]]


# Factory function
def build_model(
    num_classes: int = 5,
    backbone: str = "efficientnet_b3",
    pretrained: bool = True,
    dropout: float = 0.4,
    device: str = "cuda",
) -> AstroClassifier:
    """Factory function to create and move model to device."""
    model = AstroClassifier(
        num_classes=num_classes,
        backbone=backbone,
        pretrained=pretrained,
        dropout=dropout,
    )
    return model.to(device)


# Quick test
if __name__ == "__main__":
    print("Testing AstroClassifier...")

    for backbone in ["efficientnet_b3", "resnet50"]:
        model = build_model(num_classes=5, backbone=backbone, pretrained=False)
        x = torch.randn(2, 3, 224, 224)
        out = model(x)

        params = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

        # Verify Grad-CAM target layer exists
        target_layer = model.get_gradcam_target_layer()

        print(f"\n  Backbone: {backbone}")
        print(f"  Input:    {x.shape}")
        print(f"  Output:   {out.shape}")
        print(f"  Params:   {params:,} total, {trainable:,} trainable")
        print(f"  Grad-CAM target: {target_layer[0].__class__.__name__}")
        assert out.shape == (2, 5), "Output shape mismatch!"

    print("\nAll model tests passed!")
