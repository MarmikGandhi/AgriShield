"""
models.py
=========
Defines the deep learning architectures used in the crop health monitoring pipeline:
1. EfficientNet-B0 (Transfer Learning backbone for 38 plant-stress categories)
2. MobileNetV3-Small (High-efficiency mobile architecture for 38 classes)
3. Dual-CNN Weighted Ensemble combining probability outputs
4. U-Net Architecture for pixel-level disease lesion segmentation (trained on PlantSeg)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from typing import Dict, Any, Tuple, Optional
from pathlib import Path

def validate_ensemble_weights(weight_eff: float, weight_mob: float) -> Tuple[float, float]:
    """Validates that ensemble weights are finite, non-negative, and sum to 1.0."""
    if not (isinstance(weight_eff, (int, float)) and isinstance(weight_mob, (int, float))):
        raise ValueError(f"Ensemble weights must be numeric. Got weight_eff={weight_eff}, weight_mob={weight_mob}")
    if math.isnan(weight_eff) or math.isinf(weight_eff) or math.isnan(weight_mob) or math.isinf(weight_mob):
        raise ValueError(f"Ensemble weights must be finite. Got weight_eff={weight_eff}, weight_mob={weight_mob}")
    if weight_eff < 0.0 or weight_mob < 0.0:
        raise ValueError(f"Ensemble weights must be non-negative. Got weight_eff={weight_eff}, weight_mob={weight_mob}")
    weight_sum = weight_eff + weight_mob
    if not math.isclose(weight_sum, 1.0, rel_tol=1e-3, abs_tol=1e-3):
        raise ValueError(f"Ensemble weights must sum to 1.0 (got {weight_sum:.4f}: weight_eff={weight_eff}, weight_mob={weight_mob})")
    return float(weight_eff), float(weight_mob)

# --- 1. EfficientNet-B0 Architecture ---
def get_efficientnet_b0(num_classes: int = 38, pretrained: bool = True) -> nn.Module:
    """Builds EfficientNet-B0 with a fine-tuning classification head."""
    weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
    model = models.efficientnet_b0(weights=weights)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3, inplace=True),
        nn.Linear(in_features, 512),
        nn.SiLU(),
        nn.Dropout(p=0.2, inplace=True),
        nn.Linear(512, num_classes)
    )
    return model

# --- 2. MobileNetV3-Small Architecture ---
def get_mobilenet_v3(num_classes: int = 38, pretrained: bool = True) -> nn.Module:
    """Builds MobileNetV3-Small with a fine-tuning classification head."""
    weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
    model = models.mobilenet_v3_small(weights=weights)
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, num_classes)
    return model

# --- 3. Dual-CNN Weighted Ensemble ---
class ClassificationEnsemble(nn.Module):
    def __init__(self, model_eff: nn.Module, model_mob: nn.Module, weight_eff: float = 0.55, weight_mob: float = 0.45):
        super().__init__()
        self.model_eff = model_eff
        self.model_mob = model_mob
        self.weight_eff, self.weight_mob = validate_ensemble_weights(weight_eff, weight_mob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out_eff = self.model_eff(x)
        out_mob = self.model_mob(x)
        prob_eff = F.softmax(out_eff, dim=1)
        prob_mob = F.softmax(out_mob, dim=1)
        ensemble_prob = (self.weight_eff * prob_eff) + (self.weight_mob * prob_mob)
        return ensemble_prob

# --- 4. U-Net Architecture for Disease Lesion Segmentation ---
class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)

class UNet(nn.Module):
    """
    Lightweight, high-resolution U-Net for leaf lesion segmentation.
    Trained on PlantSeg image-mask pairs to produce pixel-level affected tissue maps.
    """
    def __init__(self, in_channels: int = 3, out_channels: int = 1, features: Tuple[int, ...] = (32, 64, 128, 256)):
        super().__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Downsampling path
        in_ch = in_channels
        for feature in features:
            self.downs.append(DoubleConv(in_ch, feature))
            in_ch = feature

        # Bottleneck
        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)

        # Upsampling path
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2)
            )
            self.ups.append(DoubleConv(feature * 2, feature))

        # Final 1x1 projection
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip_connections = []
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx // 2]
            if x.shape != skip_connection.shape:
                x = F.interpolate(x, size=skip_connection.shape[2:], mode="bilinear", align_corners=True)
            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx + 1](concat_skip)

        return torch.sigmoid(self.final_conv(x))
