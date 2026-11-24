"""
gradcam.py
==========
Authentic, mathematically grounded Gradient-weighted Class Activation Mapping (Grad-CAM)
implementation using PyTorch forward and backward tensor hooks.
Computes activation heatmaps from actual convolutional gradients of the trained classification models.
"""

import cv2
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from typing import Tuple, Optional, Dict, Any

class GradCAM:
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hooks = []

        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            # grad_output[0] is the gradient of the loss/score w.r.t layer output
            self.gradients = grad_output[0].detach()

        self.hooks.append(self.target_layer.register_forward_hook(forward_hook))
        self.hooks.append(self.target_layer.register_full_backward_hook(backward_hook))

    def remove_hooks(self):
        for h in self.hooks:
            h.remove()
        self.hooks.clear()

    def generate_heatmap(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None,
        original_shape: Tuple[int, int] = (224, 224)
    ) -> Tuple[np.ndarray, np.ndarray, int, float]:
        """
        Generates raw Grad-CAM heatmap and blended visualization.
        
        Args:
            input_tensor: 4D tensor (1, C, H, W) normalized
            target_class: int class index. If None, uses top predicted class.
            original_shape: (height, width) of original image for resizing heatmap.

        Returns:
            heatmap_norm: 2D numpy array [0, 1]
            heatmap_colored: 3D uint8 BGR colored heatmap
            pred_class: int predicted class
            confidence: float confidence probability
        """
        self.model.eval()
        self.model.zero_grad()

        # Forward pass
        logits = self.model(input_tensor)
        probs = torch.softmax(logits, dim=1)

        if target_class is None:
            target_class = int(torch.argmax(probs, dim=1).item())

        confidence = float(probs[0, target_class].item())

        # Backward pass for target class score
        score = logits[0, target_class]
        score.backward(retain_graph=True)

        # Global average pooling of gradients: weights alpha_k
        # gradients shape: (1, channels, h, w)
        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])

        # Multiply each channel by corresponding gradient weight
        for i in range(self.activations.shape[1]):
            self.activations[:, i, :, :] *= pooled_gradients[i]

        # ReLU over weighted combination
        heatmap = torch.mean(self.activations, dim=1).squeeze()
        heatmap = torch.clamp(heatmap, min=0)

        # Normalize to [0, 1]
        heatmap_np = heatmap.cpu().numpy()
        max_val = np.max(heatmap_np)
        if max_val > 1e-8:
            heatmap_norm = heatmap_np / max_val
        else:
            heatmap_norm = np.zeros_like(heatmap_np)

        # Resize to original image dimensions
        orig_h, orig_w = original_shape
        heatmap_resized = cv2.resize(heatmap_norm, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
        heatmap_uint8 = np.uint8(255 * heatmap_resized)

        # Apply JET colormap for high visual saliency
        heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

        return heatmap_resized, heatmap_colored, target_class, confidence

    @staticmethod
    def overlay_on_image(original_bgr: np.ndarray, heatmap_colored: np.ndarray, alpha: float = 0.55) -> np.ndarray:
        """Overlays colormap on original BGR image with alpha blending."""
        if original_bgr.shape[:2] != heatmap_colored.shape[:2]:
            heatmap_colored = cv2.resize(heatmap_colored, (original_bgr.shape[1], original_bgr.shape[0]))
        overlay = cv2.addWeighted(original_bgr, 1.0 - alpha, heatmap_colored, alpha, 0)
        return overlay
