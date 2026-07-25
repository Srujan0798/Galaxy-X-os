#!/usr/bin/env python3
"""
SCALE x ODYSSEY -- src/detection.py
True object detection head for the localization bonus task.

Implements a proper detection head (not just attention) that predicts:
- Bounding boxes (x, y, w, h)
- Objectness score
- Class label

Uses anchor-free center-point detection for simplicity and robustness.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict


class DetectionHead(nn.Module):
    """
    Anchor-free detection head for astronomical object localization.
    
    Predicts:
    - Center point heatmap (classification + center-ness)
    - Bounding box size (w, h)
    - Bounding box offset (x, y)
    
    Based on FCOS/CenterNet architecture.
    """
    
    def __init__(self, in_channels: int, num_classes: int = 5, 
                 feat_channels: int = 256, num_stages: int = 3):
        super().__init__()
        
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.num_stages = num_stages
        
        # Feature adaptation for each FPN stage
        self.stages = nn.ModuleList()
        for _ in range(num_stages):
            self.stages.append(nn.Sequential(
                nn.Conv2d(in_channels, feat_channels, 3, padding=1),
                nn.GroupNorm(32, feat_channels),
                nn.SiLU(inplace=True),
                nn.Conv2d(feat_channels, feat_channels, 3, padding=1),
                nn.GroupNorm(32, feat_channels),
                nn.SiLU(inplace=True),
            ))
        
        # Detection heads for each stage
        self.cls_heads = nn.ModuleList()
        self.box_heads = nn.ModuleList()
        self.centerness_heads = nn.ModuleList()
        
        for _ in range(num_stages):
            # Classification head
            self.cls_heads.append(nn.Sequential(
                nn.Conv2d(feat_channels, feat_channels, 3, padding=1),
                nn.GroupNorm(32, feat_channels),
                nn.SiLU(inplace=True),
                nn.Conv2d(feat_channels, num_classes, 3, padding=1),
            ))
            
            # Bounding box head (l, t, r, b offsets)
            self.box_heads.append(nn.Sequential(
                nn.Conv2d(feat_channels, feat_channels, 3, padding=1),
                nn.GroupNorm(32, feat_channels),
                nn.SiLU(inplace=True),
                nn.Conv2d(feat_channels, 4, 3, padding=1),
            ))
            
            # Centerness head
            self.centerness_heads.append(nn.Sequential(
                nn.Conv2d(feat_channels, feat_channels, 3, padding=1),
                nn.GroupNorm(32, feat_channels),
                nn.SiLU(inplace=True),
                nn.Conv2d(feat_channels, 1, 3, padding=1),
            ))
    
    def forward(self, features: List[torch.Tensor]) -> Dict:
        """
        Forward pass.
        
        Args:
            features: List of feature maps from backbone [stage1, stage2, stage3]
                      Each is [B, C, H, W]
        
        Returns:
            Dict with 'cls', 'box', 'centerness' for each stage
        """
        outputs = {
            'cls': [],
            'box': [],
            'centerness': [],
            'strides': [],
        }
        
        for i, feat in enumerate(features[:self.num_stages]):
            adapted = self.stages[i](feat)
            
            cls_out = self.cls_heads[i](adapted)
            box_out = self.box_heads[i](adapted)
            centerness_out = self.centerness_heads[i](adapted)
            
            outputs['cls'].append(cls_out)
            outputs['box'].append(box_out)
            outputs['centerness'].append(centerness_out)
            outputs['strides'].append(8 * (2 ** i))  # 8, 16, 32
        
        return outputs


class AstroDetector(nn.Module):
    """
    Full detection model: backbone + FPN + detection head.
    Can be used standalone or jointly with classification.
    """
    
    def __init__(self, backbone: nn.Module, num_classes: int = 5,
                 in_channels_list: List[int] = [256, 512, 1024]):
        super().__init__()
        
        self.backbone = backbone
        self.num_classes = num_classes
        
        # FPN
        self.fpn = nn.ModuleList()
        for in_ch in in_channels_list:
            self.fpn.append(nn.Sequential(
                nn.Conv2d(in_ch, 256, 1),
                nn.GroupNorm(32, 256),
                nn.SiLU(inplace=True),
            ))
        
        # FPN top-down
        self.fpn_topdown = nn.ModuleList()
        for _ in in_channels_list:
            self.fpn_topdown.append(nn.Sequential(
                nn.Conv2d(256, 256, 3, padding=1),
                nn.GroupNorm(32, 256),
                nn.SiLU(inplace=True),
            ))
        
        # Detection head
        self.det_head = DetectionHead(256, num_classes, 256, len(in_channels_list))
    
    def _get_backbone_features(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Extract multi-scale features from backbone."""
        if hasattr(self.backbone, 'forward_features'):
            features = self.backbone.forward_features(x)
        else:
            features = self.backbone(x)
        
        # Handle different backbone output formats
        if isinstance(features, torch.Tensor):
            return [features]
        elif isinstance(features, (list, tuple)):
            return list(features)
        elif isinstance(features, dict):
            return list(features.values())
        else:
            return [features]
    
    def forward(self, x: torch.Tensor) -> Dict:
        """Forward pass returning detection outputs."""
        features = self._get_backbone_features(x)
        
        # FPN
        fpn_features = []
        for i, (feat, fpn_layer) in enumerate(zip(features, self.fpn)):
            fpn_features.append(fpn_layer(feat))
        
        # Top-down
        for i in range(len(fpn_features) - 2, -1, -1):
            upsampled = F.interpolate(fpn_features[i + 1], scale_factor=2, mode='bilinear', align_corners=False)
            fpn_features[i] = fpn_features[i] + upsampled
        
        # Apply top-down conv
        for i in range(len(fpn_features)):
            fpn_features[i] = self.fpn_topdown[i](fpn_features[i])
        
        # Detection
        det_outputs = self.det_head(fpn_features)
        
        return det_outputs


def decode_detections(det_outputs: Dict, conf_threshold: float = 0.3,
                      nms_threshold: float = 0.5, max_detections: int = 100) -> List[Dict]:
    """
    Decode raw detection outputs to bounding boxes.
    
    Args:
        det_outputs: Dict from AstroDetector forward
        conf_threshold: minimum confidence
        nms_threshold: NMS IoU threshold
        max_detections: max detections per image
    
    Returns:
        List of dicts with 'boxes', 'scores', 'labels' per image
    """
    batch_size = det_outputs['cls'][0].shape[0]
    results = []
    
    for b in range(batch_size):
        all_boxes = []
        all_scores = []
        all_labels = []
        
        for stage_idx in range(len(det_outputs['cls'])):
            cls_out = det_outputs['cls'][stage_idx][b]  # [C, H, W]
            box_out = det_outputs['box'][stage_idx][b]  # [4, H, W]
            centerness_out = det_outputs['centerness'][stage_idx][b]  # [1, H, W]
            stride = det_outputs['strides'][stage_idx]
            
            H, W = cls_out.shape[1:]
            
            # Get class probabilities
            cls_prob = torch.sigmoid(cls_out)  # [C, H, W]
            
            # Find candidate points
            max_cls_prob, max_cls_idx = cls_prob.max(dim=0)  # [H, W]
            centerness = torch.sigmoid(centerness_out[0])  # [H, W]
            
            # Combined score
            combined_score = max_cls_prob * centerness  # [H, W]
            
            # Top-k candidates
            topk_scores, topk_indices = combined_score.flatten().topk(
                min(max_detections, H * W)
            )
            
            for score, idx in zip(topk_scores, topk_indices):
                if score < conf_threshold:
                    continue
                
                h_idx = idx // W
                w_idx = idx % W
                
                # Decode box
                box = box_out[:, h_idx, w_idx]  # [4]
                left, top, right, bottom = box.unbind()

                x_center = (w_idx + 0.5) * stride
                y_center = (h_idx + 0.5) * stride

                x1 = x_center - left * stride
                y1 = y_center - top * stride
                x2 = x_center + right * stride
                y2 = y_center + bottom * stride
                
                label = max_cls_idx[h_idx, w_idx].item()
                
                all_boxes.append([x1.item(), y1.item(), x2.item(), y2.item()])
                all_scores.append(score.item())
                all_labels.append(label)
        
        if all_boxes:
            # Apply NMS
            boxes = torch.tensor(all_boxes)
            scores = torch.tensor(all_scores)
            labels = torch.tensor(all_labels)
            
            keep = batched_nms(boxes, scores, labels, nms_threshold)
            
            results.append({
                'boxes': boxes[keep].tolist(),
                'scores': scores[keep].tolist(),
                'labels': labels[keep].tolist(),
            })
        else:
            results.append({
                'boxes': [],
                'scores': [],
                'labels': [],
            })
    
    return results


def batched_nms(boxes: torch.Tensor, scores: torch.Tensor, 
                labels: torch.Tensor, iou_threshold: float) -> torch.Tensor:
    """Simple NMS implementation."""
    if len(boxes) == 0:
        return torch.tensor([], dtype=torch.long)
    
    keep = []
    order = scores.sort(descending=True)[1]
    
    while len(order) > 0:
        i = order[0]
        keep.append(i)
        
        if len(order) == 1:
            break
        
        # Compute IoU
        ious = box_iou(boxes[i], boxes[order[1:]])
        order = order[1:][ious < iou_threshold]
    
    return torch.tensor(keep, dtype=torch.long)


def box_iou(box: torch.Tensor, boxes: torch.Tensor) -> torch.Tensor:
    """Compute IoU between a box and a set of boxes."""
    x1 = torch.max(box[0], boxes[:, 0])
    y1 = torch.max(box[1], boxes[:, 1])
    x2 = torch.min(box[2], boxes[:, 2])
    y2 = torch.min(box[3], boxes[:, 3])
    
    w = (x2 - x1).clamp(min=0)
    h = (y2 - y1).clamp(min=0)
    
    inter = w * h
    area1 = (box[2] - box[0]) * (box[3] - box[1])
    area2 = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    
    union = area1 + area2 - inter
    return inter / union.clamp(min=1e-8)