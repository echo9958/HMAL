"""
Specialized Pedestrian Detection Module for RGB-IR Fusion.
Implements multi-scale detection with hierarchical modality advantage learning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Dict, Optional
import numpy as np


class SpatialAttentionGate(nn.Module):
    """
    Spatial attention mechanism for highlighting pedestrian-relevant regions.
    Uses both RGB and thermal modalities to compute attention weights.
    """

    def __init__(self, channels: int):
        super().__init__()
        self.conv_query = nn.Conv2d(channels, channels // 8, kernel_size=1)
        self.conv_key = nn.Conv2d(channels, channels // 8, kernel_size=1)
        self.conv_value = nn.Conv2d(channels, channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = x.size()

        query = self.conv_query(x).view(batch, -1, height * width).permute(0, 2, 1)
        key = self.conv_key(x).view(batch, -1, height * width)
        attention = torch.bmm(query, key)
        attention = F.softmax(attention, dim=-1)

        value = self.conv_value(x).view(batch, channels, -1)
        out = torch.bmm(value, attention.permute(0, 2, 1))
        out = out.view(batch, channels, height, width)

        return self.gamma * out + x


class TemperatureAwareModule(nn.Module):
    """
    Leverages thermal information for robust pedestrian detection in challenging conditions.
    Extracts temperature-specific features that complement RGB appearance.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.thermal_branch = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.thermal_pool = nn.AdaptiveAvgPool2d(1)
        self.temperature_fc = nn.Sequential(
            nn.Linear(out_channels, out_channels // 4),
            nn.ReLU(inplace=True),
            nn.Linear(out_channels // 4, out_channels),
            nn.Sigmoid()
        )

    def forward(self, thermal_feat: torch.Tensor) -> torch.Tensor:
        thermal = self.thermal_branch(thermal_feat)

        # Global temperature context
        batch, channels, _, _ = thermal.size()
        temp_context = self.thermal_pool(thermal).view(batch, channels)
        temp_weights = self.temperature_fc(temp_context).view(batch, channels, 1, 1)

        return thermal * temp_weights


class CrossModalityCalibration(nn.Module):
    """
    Calibrates feature representations between RGB and thermal modalities.
    Ensures consistent feature scale and semantic alignment.
    """

    def __init__(self, rgb_channels: int, thermal_channels: int, unified_channels: int):
        super().__init__()
        self.rgb_proj = nn.Sequential(
            nn.Conv2d(rgb_channels, unified_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(unified_channels)
        )
        self.thermal_proj = nn.Sequential(
            nn.Conv2d(thermal_channels, unified_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(unified_channels)
        )
        self.calibration = nn.Sequential(
            nn.Conv2d(unified_channels * 2, unified_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(unified_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(unified_channels, 2, kernel_size=1),
            nn.Softmax(dim=1)
        )

    def forward(self, rgb_feat: torch.Tensor, thermal_feat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        rgb_aligned = self.rgb_proj(rgb_feat)
        thermal_aligned = self.thermal_proj(thermal_feat)

        combined = torch.cat([rgb_aligned, thermal_aligned], dim=1)
        calibration_weights = self.calibration(combined)

        w_rgb = calibration_weights[:, 0:1, :, :]
        w_thermal = calibration_weights[:, 1:2, :, :]

        return rgb_aligned * w_rgb, thermal_aligned * w_thermal


class PedestrianFocusedDetectionHead(nn.Module):
    """
    Detection head optimized for pedestrian-specific characteristics.
    Handles multi-scale pedestrian instances with aspect ratio awareness.
    """

    def __init__(self, num_classes: int, in_channels: List[int],
                 anchors: List[List[int]], aspect_ratios: List[float] = [0.41, 0.45, 0.5]):
        super().__init__()
        self.num_classes = num_classes
        self.num_outputs = num_classes + 5  # x, y, w, h, obj, classes
        self.num_anchors = len(anchors[0]) // 2
        self.num_layers = len(in_channels)
        self.aspect_ratios = aspect_ratios

        # Pedestrian-specific anchor refinement
        self.anchor_refinement = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(ch, ch, kernel_size=3, padding=1, groups=ch, bias=False),
                nn.BatchNorm2d(ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(ch, self.num_anchors * 2, kernel_size=1)  # offset for x, y
            ) for ch in in_channels
        ])

        # Standard detection convolutions
        self.detection_conv = nn.ModuleList([
            nn.Conv2d(ch, self.num_anchors * self.num_outputs, kernel_size=1)
            for ch in in_channels
        ])

        self.stride = torch.tensor([8., 16., 32.])
        self.register_buffer('anchors', torch.tensor(anchors).float().view(len(anchors), -1, 2))

    def forward(self, features: List[torch.Tensor]) -> List[torch.Tensor]:
        outputs = []

        for i, feat in enumerate(features):
            # Refine anchor positions for pedestrians
            anchor_offset = self.anchor_refinement[i](feat)

            # Standard detection
            detection = self.detection_conv[i](feat)
            batch, _, height, width = detection.shape

            detection = detection.view(
                batch, self.num_anchors, self.num_outputs, height, width
            ).permute(0, 1, 3, 4, 2).contiguous()

            outputs.append(detection)

        return outputs


class PedestrianDetector(nn.Module):
    """
    Complete RGB-IR Pedestrian Detector with Hierarchical Modality Advantage Learning.
    Integrates temperature-aware processing, cross-modality calibration, and pedestrian-focused detection.
    """

    def __init__(self, num_classes: int = 1, backbone_channels: List[int] = [64, 128, 256, 512],
                 fusion_channels: int = 256, anchors: Optional[List[List[int]]] = None):
        super().__init__()

        self.num_classes = num_classes
        self.backbone_channels = backbone_channels

        if anchors is None:
            # Pedestrian-optimized anchors (tall aspect ratios)
            anchors = [
                [8, 18, 12, 28, 16, 38],      # Small pedestrians (stride 8)
                [20, 48, 28, 68, 36, 88],     # Medium pedestrians (stride 16)
                [48, 108, 64, 148, 80, 188],  # Large pedestrians (stride 32)
            ]

        # Modality-specific stem networks
        self.rgb_stem = self._build_stem(3, backbone_channels[0])
        self.thermal_stem = self._build_stem(3, backbone_channels[0])

        # Temperature-aware processing for thermal modality
        self.temperature_module = TemperatureAwareModule(backbone_channels[0], backbone_channels[0])

        # Cross-modality calibration layers
        self.calibration_layers = nn.ModuleList([
            CrossModalityCalibration(ch, ch, ch) for ch in backbone_channels
        ])

        # Spatial attention for pedestrian regions
        self.spatial_attention = nn.ModuleList([
            SpatialAttentionGate(ch) for ch in backbone_channels[1:]
        ])

        # Shared backbone stages
        self.stage1 = self._build_stage(backbone_channels[0], backbone_channels[1], num_blocks=3)
        self.stage2 = self._build_stage(backbone_channels[1], backbone_channels[2], num_blocks=4)
        self.stage3 = self._build_stage(backbone_channels[2], backbone_channels[3], num_blocks=6)

        # Feature pyramid network
        self.fpn = FeaturePyramidNetwork(backbone_channels[1:], fusion_channels)

        # Pedestrian-specific detection head
        self.detection_head = PedestrianFocusedDetectionHead(
            num_classes=num_classes,
            in_channels=[fusion_channels] * 3,
            anchors=anchors
        )

        self.stride = torch.tensor([8., 16., 32.])
        self.names = ['pedestrian'] if num_classes == 1 else [f'class_{i}' for i in range(num_classes)]

    def _build_stem(self, in_channels: int, out_channels: int) -> nn.Module:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 2, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels // 2, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def _build_stage(self, in_channels: int, out_channels: int, num_blocks: int) -> nn.Module:
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ]

        for _ in range(num_blocks):
            layers.extend([
                ResidualBlock(out_channels, out_channels)
            ])

        return nn.Sequential(*layers)

    def forward(self, rgb: torch.Tensor, thermal: torch.Tensor, augment: bool = False) -> Tuple:
        # Extract modality-specific stem features
        rgb_feat0 = self.rgb_stem(rgb)
        thermal_feat0 = self.thermal_stem(thermal)

        # Apply temperature-aware processing to thermal features
        thermal_feat0 = self.temperature_module(thermal_feat0)

        # Calibrate initial features
        rgb_feat0, thermal_feat0 = self.calibration_layers[0](rgb_feat0, thermal_feat0)

        # Stage 1: Multi-modal feature extraction
        rgb_feat1 = self.stage1(rgb_feat0)
        thermal_feat1 = self.stage1(thermal_feat0)
        rgb_feat1, thermal_feat1 = self.calibration_layers[1](rgb_feat1, thermal_feat1)
        rgb_feat1 = self.spatial_attention[0](rgb_feat1)
        fused_feat1 = rgb_feat1 + thermal_feat1

        # Stage 2
        rgb_feat2 = self.stage2(fused_feat1)
        thermal_feat2 = self.stage2(thermal_feat1)
        rgb_feat2, thermal_feat2 = self.calibration_layers[2](rgb_feat2, thermal_feat2)
        rgb_feat2 = self.spatial_attention[1](rgb_feat2)
        fused_feat2 = rgb_feat2 + thermal_feat2

        # Stage 3
        rgb_feat3 = self.stage3(fused_feat2)
        thermal_feat3 = self.stage3(thermal_feat2)
        rgb_feat3, thermal_feat3 = self.calibration_layers[3](rgb_feat3, thermal_feat3)
        rgb_feat3 = self.spatial_attention[2](rgb_feat3)
        fused_feat3 = rgb_feat3 + thermal_feat3

        # Feature pyramid
        pyramid_features = self.fpn([fused_feat1, fused_feat2, fused_feat3])

        # Detection
        detections = self.detection_head(pyramid_features)

        if self.training:
            return detections
        else:
            return self._postprocess_detections(detections), detections

    def _postprocess_detections(self, detections: List[torch.Tensor]) -> torch.Tensor:
        """Convert raw detections to inference format."""
        outputs = []

        for i, det in enumerate(detections):
            batch, num_anchors, height, width, num_outputs = det.shape
            det = det.sigmoid()

            # Create grid
            grid_y, grid_x = torch.meshgrid(
                torch.arange(height, device=det.device),
                torch.arange(width, device=det.device),
                indexing='ij'
            )
            grid = torch.stack([grid_x, grid_y], dim=-1).float()
            grid = grid.view(1, 1, height, width, 2)

            # Decode boxes
            det[..., 0:2] = (det[..., 0:2] * 2 - 0.5 + grid) * self.stride[i]
            det[..., 2:4] = (det[..., 2:4] * 2) ** 2 * self.stride[i] * 2

            outputs.append(det.view(batch, -1, num_outputs))

        return torch.cat(outputs, dim=1)

    def fuse(self):
        """Fuse Conv2d and BatchNorm2d layers for inference optimization."""
        for module in self.modules():
            if isinstance(module, nn.Conv2d) and hasattr(module, 'bn'):
                module.conv = fuse_conv_bn(module.conv, module.bn)
                delattr(module, 'bn')
                module.forward = module.forward_fuse
        return self


class ResidualBlock(nn.Module):
    """Standard residual block for backbone."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        return self.relu(out)


class FeaturePyramidNetwork(nn.Module):
    """Feature Pyramid Network for multi-scale pedestrian detection."""

    def __init__(self, in_channels_list: List[int], out_channels: int):
        super().__init__()
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(in_ch, out_channels, kernel_size=1) for in_ch in in_channels_list
        ])
        self.fpn_convs = nn.ModuleList([
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
            for _ in in_channels_list
        ])

    def forward(self, features: List[torch.Tensor]) -> List[torch.Tensor]:
        # Build top-down pathway
        laterals = [lateral_conv(feat) for feat, lateral_conv in zip(features, self.lateral_convs)]

        for i in range(len(laterals) - 1, 0, -1):
            upsampled = F.interpolate(laterals[i], size=laterals[i-1].shape[2:], mode='nearest')
            laterals[i-1] = laterals[i-1] + upsampled

        # Apply FPN convolutions
        outputs = [fpn_conv(lateral) for lateral, fpn_conv in zip(laterals, self.fpn_convs)]

        return outputs


def fuse_conv_bn(conv: nn.Conv2d, bn: nn.BatchNorm2d) -> nn.Conv2d:
    """Fuse Conv2d and BatchNorm2d layers."""
    fused_conv = nn.Conv2d(
        conv.in_channels, conv.out_channels, conv.kernel_size,
        stride=conv.stride, padding=conv.padding,
        groups=conv.groups, bias=True
    ).to(conv.weight.device)

    w_conv = conv.weight.clone()
    w_bn = torch.diag(bn.weight.div(torch.sqrt(bn.eps + bn.running_var)))
    fused_conv.weight.copy_(torch.mm(w_bn, w_conv.view(w_conv.size(0), -1)).view(w_conv.size()))

    b_conv = torch.zeros(conv.out_channels, device=conv.weight.device) if conv.bias is None else conv.bias
    b_bn = bn.bias - bn.weight.mul(bn.running_mean).div(torch.sqrt(bn.running_var + bn.eps))
    fused_conv.bias.copy_(torch.mm(w_bn, b_conv.reshape(-1, 1)).reshape(-1) + b_bn)

    return fused_conv
