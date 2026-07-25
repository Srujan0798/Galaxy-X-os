#!/usr/bin/env python3
"""
SCALE x ODYSSEY -- src/gradcam_plus.py
Advanced Explainability: Grad-CAM++, Score-CAM, and Layer-CAM.

Supports multiple CAM methods:
- Grad-CAM: vanilla gradient-weighted class activation
- Grad-CAM++: improved weighting with pixel-wise contributions
- Score-CAM: gradient-free, score-based
- Layer-CAM: layer-wise relevance propagation
- Smooth Grad-CAM: noise-averaged for cleaner maps
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2


class CAMComputer:
    """
    Unified CAM computer supporting multiple explainability methods.
    """
    
    def __init__(self, model: nn.Module, target_layer: nn.Module,
                 num_classes: int = 5):
        self.model = model
        self.target_layer = target_layer
        self.num_classes = num_classes
        self.activations = None
        self.gradients = None
        self._register_hooks()
    
    def _register_hooks(self):
        """Register forward/backward hooks on target layer."""
        def forward_hook(module, input, output):
            self.activations = output.detach()
        
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()
        
        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)
    
    def _get_gradcam(self, class_idx: int) -> np.ndarray:
        """Vanilla Grad-CAM."""
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)  # [B, C, 1, 1]
        cam = (self.activations * weights).sum(dim=1, keepdim=True)  # [B, 1, H, W]
        cam = F.relu(cam)
        return self._normalize_cam(cam)
    
    def _get_gradcam_plusplus(self, class_idx: int) -> np.ndarray:
        """Grad-CAM++ with pixel-wise contributions."""
        # Gradients: [B, C, H, W]
        grad = self.gradients
        act = self.activations
        
        # First-order gradients
        grad_1 = grad

        # Second-order gradients (approximate via chain rule)
        grad_2 = grad_1 ** 2

        # Third-order gradients
        grad_3 = grad_2 * grad_1

        # Alpha weights (pixel-wise contribution)
        alpha_num = grad_2
        alpha_den = grad_2 * 2 + grad_3 * (act + 1e-10)
        alpha_den = alpha_den.sum(dim=(2, 3), keepdim=True)  # [B, C, 1, 1]
        alpha = alpha_num / (alpha_den + 1e-10)
        
        # Weighted combination
        weights = (alpha * F.relu(grad_1)).sum(dim=(2, 3), keepdim=True)  # [B, C, 1, 1]
        cam = (act * weights).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        return self._normalize_cam(cam)
    
    def _get_scorecam(self, input_tensor: torch.Tensor, class_idx: int,
                      num_samples: int = 100) -> np.ndarray:
        """Score-CAM: gradient-free, uses activation scores."""
        act = self.activations  # [B, C, H, W]
        B, C, H, W = act.shape
        
        # Reshape activations to [B, C, H*W] and normalize per channel
        act_flat = act.view(B, C, -1)
        act_norm = (act_flat - act_flat.min(dim=-1, keepdim=True)[0]) / \
                   (act_flat.max(dim=-1, keepdim=True)[0] - act_flat.min(dim=-1, keepdim=True)[0] + 1e-10)
        act_norm = act_norm.view(B, C, H, W)
        
        # Sample channels (for large feature maps)
        channels = min(C, num_samples)
        channel_indices = torch.linspace(0, C - 1, channels, dtype=torch.long)
        
        # For each channel, compute importance score
        scores = torch.zeros(B, channels, device=input_tensor.device)
        
        with torch.no_grad():
            for i, ch in enumerate(channel_indices):
                # Project single channel to input space
                mask = F.interpolate(
                    act_norm[:, ch:ch+1, :, :],
                    size=input_tensor.shape[-2:],
                    mode='bilinear', align_corners=False
                )
                # Multiply input with mask
                masked_input = input_tensor * mask
                output = self.model(masked_input)
                scores[:, i] = F.softmax(output, dim=1)[:, class_idx]
        
        # Weighted combination
        weights = F.softmax(scores / 0.1, dim=1)  # temperature scaling
        cam = (act_norm[:, channel_indices] * weights.view(B, -1, 1, 1)).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        return self._normalize_cam(cam)
    
    def _get_layercam(self) -> np.ndarray:
        """Layer-CAM: element-wise product of activations and gradients."""
        cam = (self.activations * F.relu(self.gradients)).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        return self._normalize_cam(cam)
    
    def _get_smoothcam(self, input_tensor: torch.Tensor, class_idx: int,
                       num_noise: int = 50, noise_std: float = 0.15) -> np.ndarray:
        """Smooth Grad-CAM: average CAM over multiple noisy inputs."""
        cams = []
        for _ in range(num_noise):
            noise = torch.randn_like(input_tensor) * noise_std
            noisy_input = input_tensor + noise
            noisy_input.requires_grad = True
            
            output = self.model(noisy_input)
            score = output[0, class_idx]
            score.backward()
            
            cam = self._get_gradcam(class_idx)
            cams.append(cam)
        
        return np.mean(cams, axis=0)
    
    def _normalize_cam(self, cam: torch.Tensor) -> np.ndarray:
        """Normalize CAM to [0, 1]."""
        B = cam.shape[0]
        cam_np = cam.detach().cpu().numpy()
        for i in range(B):
            c = cam_np[i]
            c = c - c.min()
            c = c / (c.max() + 1e-10)
            cam_np[i] = c
        return cam_np
    
    def compute(self, input_tensor: torch.Tensor, class_idx: int,
                method: str = 'gradcam', **kwargs) -> np.ndarray:
        """
        Compute CAM for given input and class.
        
        Args:
            input_tensor: [B, C, H, W] tensor on correct device
            class_idx: target class index
            method: 'gradcam', 'gradcam_plusplus', 'scorecam', 'layercam', 'smoothcam'
        
        Returns:
            [B, 1, H, W] numpy array in [0, 1]
        """
        if method == 'gradcam':
            # Need to do a forward + backward pass
            self.model.zero_grad()
            input_tensor = input_tensor.detach().requires_grad_(True)
            output = self.model(input_tensor)
            score = output[0, class_idx]
            score.backward()
            return self._get_gradcam(class_idx)
        
        elif method == 'gradcam_plusplus':
            self.model.zero_grad()
            input_tensor = input_tensor.detach().requires_grad_(True)
            output = self.model(input_tensor)
            score = output[0, class_idx]
            score.backward()
            return self._get_gradcam_plusplus(class_idx)
        
        elif method == 'scorecam':
            self.model.zero_grad()
            with torch.no_grad():
                _ = self.model(input_tensor)
            return self._get_scorecam(input_tensor, class_idx, **kwargs)
        
        elif method == 'layercam':
            self.model.zero_grad()
            input_tensor = input_tensor.detach().requires_grad_(True)
            output = self.model(input_tensor)
            score = output[0, class_idx]
            score.backward()
            return self._get_layercam()
        
        elif method == 'smoothcam':
            return self._get_smoothcam(input_tensor, class_idx, **kwargs)
        
        else:
            raise ValueError(f"Unknown method: {method}")


def create_cam_overlay(rgb_image: np.ndarray, cam_heatmap: np.ndarray,
                       alpha: float = 0.5, colormap: int = cv2.COLORMAP_JET) -> np.ndarray:
    """Create overlay of CAM heatmap on RGB image."""
    if rgb_image.max() > 1.0:
        rgb_image = rgb_image.astype(np.float32) / 255.0
    
    # Resize CAM to image size
    if cam_heatmap.shape[:2] != rgb_image.shape[:2]:
        cam_heatmap = cv2.resize(cam_heatmap, (rgb_image.shape[1], rgb_image.shape[0]),
                                 interpolation=cv2.INTER_LINEAR)
    
    # Apply colormap
    heatmap = np.uint8(255 * cam_heatmap)
    heatmap_colored = cv2.applyColorMap(heatmap, colormap)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB) / 255.0
    
    # Overlay
    overlay = (1 - alpha) * rgb_image + alpha * heatmap_colored
    return np.clip(overlay, 0, 1)


def compute_explainability_metrics(model: nn.Module, dataset, device: torch.device,
                                   num_samples: int = 50) -> dict:
    """
    Compute insertion/deletion scores for Grad-CAM quality assessment.
    
    Insertion: create images by blurring and progressively revealing
    Deletion: create images by progressively masking
    Higher insertion AUC and lower deletion AUC = better explanations.
    """
    pass