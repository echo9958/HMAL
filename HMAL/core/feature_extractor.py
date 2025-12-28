"""
Multi-Modal Feature Extraction Module for RGB-IR Pedestrian Detection.
Implements various backbone architectures and feature extraction strategies.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional
import torchvision.models as models


class MultiModalFeatureExtractor(nn.Module):
    """
    Extracts hierarchical features from RGB and thermal modalities.
    Supports multiple backbone architectures (ResNet, EfficientNet, etc.).
    """

    def __init__(self, backbone: str = 'resnet50', pretrained: bool = True,
                 feature_channels: List[int] = [256, 512, 1024, 2048]):
        """
        Args:
            backbone: Backbone architecture name
            pretrained: Whether to use ImageNet pretrained weights
            feature_channels: Expected output channels at each level
        """
        super().__init__()

        self.backbone_name = backbone
        self.feature_channels = feature_channels

        # Build RGB and thermal feature extractors
        self.rgb_extractor = self._build_extractor(backbone, pretrained, in_channels=3)
        self.thermal_extractor = self._build_extractor(backbone, pretrained, in_channels=3)

    def _build_extractor(self, backbone: str, pretrained: bool, in_channels: int) -> nn.Module:
        """Build feature extractor based on backbone type."""
        if 'resnet' in backbone:
            return ResNetExtractor(backbone, pretrained, in_channels)
        elif 'efficientnet' in backbone:
            return EfficientNetExtractor(backbone, pretrained, in_channels)
        elif 'mobilenet' in backbone:
            return MobileNetExtractor(backbone, pretrained, in_channels)
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

    def forward(self, rgb: torch.Tensor, thermal: torch.Tensor) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        Extract multi-scale features from both modalities.

        Args:
            rgb: RGB image tensor [B, 3, H, W]
            thermal: Thermal image tensor [B, 3, H, W]

        Returns:
            Tuple of (rgb_features, thermal_features), each a list of feature maps
        """
        rgb_features = self.rgb_extractor(rgb)
        thermal_features = self.thermal_extractor(thermal)

        return rgb_features, thermal_features


class ResNetExtractor(nn.Module):
    """
    ResNet-based feature extractor.
    Extracts features at multiple scales (C2, C3, C4, C5).
    """

    def __init__(self, backbone: str = 'resnet50', pretrained: bool = True, in_channels: int = 3):
        super().__init__()

        # Load pretrained ResNet
        if backbone == 'resnet18':
            base_model = models.resnet18(pretrained=pretrained)
            self.out_channels = [64, 128, 256, 512]
        elif backbone == 'resnet34':
            base_model = models.resnet34(pretrained=pretrained)
            self.out_channels = [64, 128, 256, 512]
        elif backbone == 'resnet50':
            base_model = models.resnet50(pretrained=pretrained)
            self.out_channels = [256, 512, 1024, 2048]
        elif backbone == 'resnet101':
            base_model = models.resnet101(pretrained=pretrained)
            self.out_channels = [256, 512, 1024, 2048]
        else:
            raise ValueError(f"Unsupported ResNet variant: {backbone}")

        # Modify first conv if needed
        if in_channels != 3:
            self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        else:
            self.conv1 = base_model.conv1

        self.bn1 = base_model.bn1
        self.relu = base_model.relu
        self.maxpool = base_model.maxpool

        # ResNet stages
        self.layer1 = base_model.layer1  # C2
        self.layer2 = base_model.layer2  # C3
        self.layer3 = base_model.layer3  # C4
        self.layer4 = base_model.layer4  # C5

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Extract multi-scale features.

        Args:
            x: Input tensor [B, C, H, W]

        Returns:
            List of feature maps [C2, C3, C4, C5]
        """
        # Stem
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # Multi-scale features
        c2 = self.layer1(x)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)

        return [c2, c3, c4, c5]


class EfficientNetExtractor(nn.Module):
    """
    EfficientNet-based feature extractor.
    Lightweight and efficient for real-time applications.
    """

    def __init__(self, backbone: str = 'efficientnet_b0', pretrained: bool = True, in_channels: int = 3):
        super().__init__()

        # Load pretrained EfficientNet
        if backbone == 'efficientnet_b0':
            base_model = models.efficientnet_b0(pretrained=pretrained)
            self.out_channels = [40, 80, 112, 320]
        elif backbone == 'efficientnet_b1':
            base_model = models.efficientnet_b1(pretrained=pretrained)
            self.out_channels = [40, 80, 112, 320]
        elif backbone == 'efficientnet_b3':
            base_model = models.efficientnet_b3(pretrained=pretrained)
            self.out_channels = [48, 96, 136, 384]
        else:
            raise ValueError(f"Unsupported EfficientNet variant: {backbone}")

        # Modify stem if needed
        if in_channels != 3:
            first_conv = base_model.features[0][0]
            base_model.features[0][0] = nn.Conv2d(
                in_channels, first_conv.out_channels,
                kernel_size=first_conv.kernel_size,
                stride=first_conv.stride,
                padding=first_conv.padding,
                bias=False
            )

        self.features = base_model.features

        # Identify stage indices for multi-scale extraction
        self.stage_indices = [2, 4, 6, 8]  # Approximate indices for C2-C5

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Extract multi-scale features from EfficientNet."""
        feature_maps = []

        for i, module in enumerate(self.features):
            x = module(x)
            if i in self.stage_indices:
                feature_maps.append(x)

        return feature_maps


class MobileNetExtractor(nn.Module):
    """
    MobileNet-based feature extractor.
    Extremely lightweight for embedded deployment.
    """

    def __init__(self, backbone: str = 'mobilenet_v2', pretrained: bool = True, in_channels: int = 3):
        super().__init__()

        # Load pretrained MobileNet
        if backbone == 'mobilenet_v2':
            base_model = models.mobilenet_v2(pretrained=pretrained)
            self.out_channels = [24, 32, 96, 320]
        elif backbone == 'mobilenet_v3_small':
            base_model = models.mobilenet_v3_small(pretrained=pretrained)
            self.out_channels = [24, 40, 96, 576]
        elif backbone == 'mobilenet_v3_large':
            base_model = models.mobilenet_v3_large(pretrained=pretrained)
            self.out_channels = [40, 80, 160, 960]
        else:
            raise ValueError(f"Unsupported MobileNet variant: {backbone}")

        # Modify first conv if needed
        if in_channels != 3:
            first_conv = base_model.features[0][0]
            base_model.features[0][0] = nn.Conv2d(
                in_channels, first_conv.out_channels,
                kernel_size=first_conv.kernel_size,
                stride=first_conv.stride,
                padding=first_conv.padding,
                bias=False
            )

        self.features = base_model.features

        # Stage indices for multi-scale features
        self.stage_indices = [3, 6, 13, 17]  # Approximate for MobileNetV2

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Extract multi-scale features from MobileNet."""
        feature_maps = []

        for i, module in enumerate(self.features):
            x = module(x)
            if i in self.stage_indices:
                feature_maps.append(x)

        return feature_maps


class CrossModalityFeatureFusion(nn.Module):
    """
    Fuses features from RGB and thermal modalities at multiple scales.
    Uses attention mechanisms for adaptive weighting.
    """

    def __init__(self, feature_channels: List[int]):
        super().__init__()

        self.num_levels = len(feature_channels)

        # Fusion modules for each level
        self.fusion_modules = nn.ModuleList([
            ModalityFusionBlock(ch) for ch in feature_channels
        ])

    def forward(self, rgb_features: List[torch.Tensor],
                thermal_features: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Fuse multi-scale RGB and thermal features.

        Args:
            rgb_features: List of RGB feature maps
            thermal_features: List of thermal feature maps

        Returns:
            List of fused feature maps
        """
        fused_features = []

        for rgb_feat, thermal_feat, fusion_module in zip(
            rgb_features, thermal_features, self.fusion_modules
        ):
            fused = fusion_module(rgb_feat, thermal_feat)
            fused_features.append(fused)

        return fused_features


class ModalityFusionBlock(nn.Module):
    """Single-level modality fusion with attention."""

    def __init__(self, channels: int):
        super().__init__()

        # Channel attention
        self.channel_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels * 2, channels // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 4, channels * 2, 1),
            nn.Sigmoid()
        )

        # Spatial attention
        self.spatial_attn = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )

        # Fusion convolution
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, rgb_feat: torch.Tensor, thermal_feat: torch.Tensor) -> torch.Tensor:
        """Fuse RGB and thermal features with attention."""
        # Concatenate features
        combined = torch.cat([rgb_feat, thermal_feat], dim=1)

        # Channel attention
        channel_weights = self.channel_attn(combined)
        combined_ca = combined * channel_weights

        # Spatial attention
        avg_out = torch.mean(combined_ca, dim=1, keepdim=True)
        max_out, _ = torch.max(combined_ca, dim=1, keepdim=True)
        spatial_input = torch.cat([avg_out, max_out], dim=1)
        spatial_weights = self.spatial_attn(spatial_input)
        combined_sa = combined_ca * spatial_weights

        # Fusion
        fused = self.fusion_conv(combined_sa)

        return fused


class AdaptiveFeatureSelector(nn.Module):
    """
    Adaptively selects relevant features based on input characteristics.
    Implements modality-aware feature selection for robust detection.
    """

    def __init__(self, in_channels: int, num_classes: int = 1):
        super().__init__()

        # Scene understanding branch
        self.scene_encoder = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // 8, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 8, in_channels, 1)
        )

        # Feature importance predictor
        self.importance_predictor = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, 1),
            nn.Sigmoid()
        )

    def forward(self, rgb_feat: torch.Tensor, thermal_feat: torch.Tensor) -> torch.Tensor:
        """
        Select and combine features adaptively.

        Args:
            rgb_feat: RGB features
            thermal_feat: Thermal features

        Returns:
            Selected and fused features
        """
        # Encode scene characteristics
        rgb_scene = self.scene_encoder(rgb_feat)
        thermal_scene = self.scene_encoder(thermal_feat)

        # Predict feature importance
        scene_combined = torch.cat([rgb_scene, thermal_scene], dim=1)
        importance_weights = self.importance_predictor(scene_combined)

        # Apply importance weighting
        rgb_weighted = rgb_feat * importance_weights
        thermal_weighted = thermal_feat * (1 - importance_weights)

        # Combine weighted features
        selected_features = rgb_weighted + thermal_weighted

        return selected_features
