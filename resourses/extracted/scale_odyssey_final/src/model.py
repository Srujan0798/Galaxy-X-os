#!/usr/bin/env python3
"""
SCALE x ODYSSEY -- src/model.py
EfficientNet-B3 with Custom Classification Head

Matches official starter guide (page 10) with our
get_gradcam_target_layer() for backbone-agnostic Grad-CAM.
"""

import torch
import torch.nn as nn
import timm


class AstroClassifier(nn.Module):
    """
    Astronomical Object Classifier.
    Pretrained EfficientNet-B3 backbone + custom head with
    BatchNorm, Dropout, and residual-style skip connections.
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

        self.backbone = timm.create_model(
            backbone, pretrained=pretrained, num_classes=0, global_pool=""
        )

        in_features = self.backbone.num_features

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.BatchNorm1d(in_features),
            nn.Dropout(dropout),
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(dropout / 2),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(256),
            nn.Dropout(dropout / 4),
            nn.Linear(256, num_classes),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone.forward_features(x)
        return self.classifier(features)

    def freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = True

    def get_gradcam_target_layer(self):
        """Return target layer for Grad-CAM (backbone-agnostic)."""
        if "efficientnet" in self.backbone_name:
            return [self.backbone.blocks[-1][-1]]
        elif "resnet" in self.backbone_name:
            return [self.backbone.layer4[-1]]
        elif "vit" in self.backbone_name:
            return [self.backbone.blocks[-1].norm1]
        else:
            return [list(self.backbone.children())[-2]]
