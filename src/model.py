#!/usr/bin/env python3
"""
SCALE x ODYSSEY -- src/model.py
Modern Backbone Support: ConvNeXt, Swin, EfficientNet, ViT
with unified classifier head and Grad-CAM support.
"""

import torch
import torch.nn as nn
import timm


BACKBONE_CONFIGS = {
    "convnext_base": {
        "features": 1024,
        "pool": "avg",
    },
    "convnext_large": {
        "features": 1536,
        "pool": "avg",
    },
    "swin_base_patch4_window7_224": {
        "features": 1024,
        "pool": "avg",
    },
    "swin_large_patch4_window7_224": {
        "features": 1536,
        "pool": "avg",
    },
    "efficientnet_b3": {
        "features": 1536,
        "pool": "avg",
    },
    "efficientnet_b4": {
        "features": 1792,
        "pool": "avg",
    },
    "vit_base_patch16_224": {
        "features": 768,
        "pool": "cls",
    },
    "vit_large_patch16_224": {
        "features": 1024,
        "pool": "cls",
    },
}


class AstroClassifier(nn.Module):
    """
    Astronomical Object Classifier.
    Supports modern backbones (ConvNeXt, Swin, ViT, EfficientNet)
    with unified classifier head and backbone-agnostic Grad-CAM.
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
        self.pretrained = pretrained

        # Create backbone
        self.backbone = timm.create_model(
            backbone, pretrained=pretrained, num_classes=0, global_pool=""
        )

        # Get feature dimension; always trust timm's actual num_features
        cfg = BACKBONE_CONFIGS.get(backbone, {"pool": "avg"})
        self.pool_type = cfg.get("pool", "avg")
        in_features = self.backbone.num_features

        # Unified classifier head
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1) if self.pool_type == "avg" else nn.Identity(),
            nn.Flatten(1) if self.pool_type == "avg" else nn.Identity(),
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
        
        # Handle different backbone output formats
        if self.pool_type == "cls" and features.ndim == 3:
            # ViT: [B, N, C] -> take CLS token
            features = features[:, 0]
        elif features.ndim == 4:
            # CNN: [B, C, H, W] -> handled by classifier's AdaptiveAvgPool
            pass
        elif features.ndim == 3:
            # Swin: [B, H*W, C] -> pool
            features = features.mean(dim=1)
        
        return self.classifier(features)

    def freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = True

    def get_gradcam_target_layer(self):
        """Return target layer for Grad-CAM (backbone-agnostic)."""
        name = self.backbone_name.lower()
        if "convnext" in name:
            return [self.backbone.stages[-1].blocks[-1].norm]
        elif "swin" in name:
            return [self.backbone.layers[-1].blocks[-1].norm1]
        elif "vit" in name:
            return [self.backbone.blocks[-1].norm1]
        elif "efficientnet" in name:
            return [self.backbone.blocks[-1][-1]]
        elif "resnet" in name:
            return [self.backbone.layer4[-1]]
        else:
            return [list(self.backbone.children())[-2]]


class AstroEnsemble(nn.Module):
    """
    Ensemble of multiple AstroClassifiers with uncertainty quantification.
    Supports MC Dropout and ensemble variance for anomaly detection.
    """

    def __init__(self, models: list[nn.Module], mc_dropout: int = 10):
        super().__init__()
        self.models = nn.ModuleList(models)
        self.mc_dropout = mc_dropout
        self.num_models = len(models)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Standard forward: average predictions."""
        preds = []
        for model in self.models:
            model.eval()
            with torch.no_grad():
                preds.append(model(x))
        return torch.stack(preds).mean(0)

    def predict_with_uncertainty(self, x: torch.Tensor, return_all: bool = False):
        """
        Returns: mean_logits, epistemic_uncertainty, aleatoric_uncertainty, all_logits
        """
        all_logits = []
        
        # Ensemble predictions
        for model in self.models:
            model.eval()
            with torch.no_grad():
                all_logits.append(model(x))
        
        # MC Dropout predictions (enable dropout at inference)
        for model in self.models:
            model.train()  # Enable dropout
            for _ in range(self.mc_dropout):
                with torch.no_grad():
                    all_logits.append(model(x))
            model.eval()
        
        all_logits = torch.stack(all_logits)  # [N, B, C]
        mean_logits = all_logits.mean(0)
        
        # Epistemic uncertainty: variance across ensemble + MC samples
        epistemic = all_logits.var(0).mean(-1)  # [B]
        
        # Aleatoric uncertainty: mean entropy of predictions
        probs = torch.softmax(all_logits, dim=-1)
        entropy = -(probs * probs.log()).sum(-1)
        aleatoric = entropy.mean(0)  # [B]
        
        total_uncertainty = epistemic + aleatoric
        
        if return_all:
            return mean_logits, epistemic, aleatoric, total_uncertainty, all_logits
        return mean_logits, epistemic, aleatoric, total_uncertainty


def create_model_from_config(config: dict) -> nn.Module:
    """Factory function to create model from config dict."""
    if config.get("ensemble"):
        models = [create_model_from_config(c) for c in config["ensemble"]]
        return AstroEnsemble(models, mc_dropout=config.get("mc_dropout", 10))
    return AstroClassifier(
        num_classes=config.get("num_classes", 5),
        backbone=config.get("backbone", "efficientnet_b3"),
        pretrained=config.get("pretrained", True),
        dropout=config.get("dropout", 0.4),
    )


# Default ensemble config for top-tier submission
DEFAULT_ENSEMBLE_CONFIG = {
    "ensemble": [
        {"backbone": "convnext_base", "pretrained": True, "dropout": 0.4},
        {"backbone": "swin_base_patch4_window7_224", "pretrained": True, "dropout": 0.4},
        {"backbone": "efficientnet_b3", "pretrained": True, "dropout": 0.4},
    ],
    "mc_dropout": 10,
}