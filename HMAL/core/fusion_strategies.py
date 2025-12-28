"""
Advanced Fusion Strategies for RGB-Thermal Pedestrian Detection.
Implements various cross-modality fusion approaches with learnable weighting.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Optional
import math


class AdaptiveFusionModule(nn.Module):
    """
    Adaptive fusion that learns optimal combination of RGB and thermal features.
    Uses attention mechanism to weight modality contributions dynamically.
    """

    def __init__(self, channels: int, reduction_ratio: int = 16):
        super().__init__()
        self.channels = channels

        # Global context extraction
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        # Modality importance predictor
        self.importance_net = nn.Sequential(
            nn.Linear(channels * 2, channels // reduction_ratio),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction_ratio, 2),
            nn.Softmax(dim=1)
        )

        # Local feature refinement
        self.local_refine = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels)
        )

    def forward(self, rgb_features: torch.Tensor, thermal_features: torch.Tensor) -> torch.Tensor:
        batch_size, channels, height, width = rgb_features.shape

        # Extract global context from both modalities
        rgb_global = self.global_pool(rgb_features).view(batch_size, channels)
        thermal_global = self.global_pool(thermal_features).view(batch_size, channels)

        # Predict modality importance weights
        combined_global = torch.cat([rgb_global, thermal_global], dim=1)
        importance_weights = self.importance_net(combined_global)

        w_rgb = importance_weights[:, 0].view(batch_size, 1, 1, 1)
        w_thermal = importance_weights[:, 1].view(batch_size, 1, 1, 1)

        # Weighted fusion at global level
        global_fused = w_rgb * rgb_features + w_thermal * thermal_features

        # Local refinement
        local_input = torch.cat([rgb_features, thermal_features], dim=1)
        local_refined = self.local_refine(local_input)

        # Combine global and local information
        output = global_fused + local_refined

        return output


class HierarchicalFusion(nn.Module):
    """
    Hierarchical fusion strategy that progressively combines multi-scale features.
    Maintains both low-level details and high-level semantic information.
    """

    def __init__(self, channel_pyramid: List[int]):
        super().__init__()
        self.num_levels = len(channel_pyramid)

        # Per-level fusion modules
        self.level_fusion = nn.ModuleList([
            LevelWiseFusion(ch) for ch in channel_pyramid
        ])

        # Cross-level interaction
        self.cross_level = nn.ModuleList([
            CrossLevelInteraction(channel_pyramid[i], channel_pyramid[i+1])
            for i in range(self.num_levels - 1)
        ])

        # Final integration
        self.integration = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(ch * 2, ch, kernel_size=1, bias=False),
                nn.BatchNorm2d(ch),
                nn.ReLU(inplace=True)
            ) for ch in channel_pyramid
        ])

    def forward(self, rgb_pyramid: List[torch.Tensor],
                thermal_pyramid: List[torch.Tensor]) -> List[torch.Tensor]:
        assert len(rgb_pyramid) == len(thermal_pyramid) == self.num_levels

        # Bottom-up fusion
        fused_features = []
        for i in range(self.num_levels):
            fused = self.level_fusion[i](rgb_pyramid[i], thermal_pyramid[i])
            fused_features.append(fused)

        # Top-down refinement with cross-level interaction
        refined_features = [fused_features[-1]]
        for i in range(self.num_levels - 2, -1, -1):
            high_level_info = refined_features[0]
            current_fused = fused_features[i]

            cross_level_feat = self.cross_level[i](current_fused, high_level_info)

            combined = torch.cat([current_fused, cross_level_feat], dim=1)
            refined = self.integration[i](combined)

            refined_features.insert(0, refined)

        return refined_features


class LevelWiseFusion(nn.Module):
    """Fuses features at a single pyramid level."""

    def __init__(self, channels: int):
        super().__init__()
        self.channel_attention = ChannelWiseAttention(channels)
        self.spatial_attention = SpatialWiseAttention(channels)

        self.fusion_conv = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, rgb: torch.Tensor, thermal: torch.Tensor) -> torch.Tensor:
        # Apply attention mechanisms
        rgb_ca = self.channel_attention(rgb)
        thermal_ca = self.channel_attention(thermal)

        rgb_sa = self.spatial_attention(rgb_ca)
        thermal_sa = self.spatial_attention(thermal_ca)

        # Concatenate and fuse
        concatenated = torch.cat([rgb_sa, thermal_sa], dim=1)
        fused = self.fusion_conv(concatenated)

        return fused


class CrossLevelInteraction(nn.Module):
    """Facilitates information flow between different pyramid levels."""

    def __init__(self, low_channels: int, high_channels: int):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.channel_align = nn.Conv2d(high_channels, low_channels, kernel_size=1)

        self.interaction = nn.Sequential(
            nn.Conv2d(low_channels * 2, low_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(low_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, low_level: torch.Tensor, high_level: torch.Tensor) -> torch.Tensor:
        high_upsampled = self.upsample(high_level)
        high_aligned = self.channel_align(high_upsampled)

        # Resize if dimensions don't match
        if high_aligned.shape[2:] != low_level.shape[2:]:
            high_aligned = F.interpolate(high_aligned, size=low_level.shape[2:],
                                        mode='bilinear', align_corners=False)

        combined = torch.cat([low_level, high_aligned], dim=1)
        output = self.interaction(combined)

        return output


class ChannelWiseAttention(nn.Module):
    """
    Channel-wise attention to emphasize informative feature channels.
    Uses both max and average pooling for robust statistics.
    """

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, kernel_size=1, bias=False)
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))

        attention = self.sigmoid(avg_out + max_out)

        return x * attention


class SpatialWiseAttention(nn.Module):
    """
    Spatial attention to focus on relevant image regions.
    Particularly useful for localizing pedestrians in complex scenes.
    """

    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)

        spatial_input = torch.cat([avg_out, max_out], dim=1)
        attention = self.conv(spatial_input)

        return x * attention


class ModalityRouter(nn.Module):
    """
    Intelligent routing mechanism that decides which modality to prioritize.
    Learns scene-dependent modality preferences for optimal pedestrian detection.
    """

    def __init__(self, channels: int, num_experts: int = 4):
        super().__init__()
        self.num_experts = num_experts

        # Expert networks for different fusion strategies
        self.experts = nn.ModuleList([
            ExpertNetwork(channels) for _ in range(num_experts)
        ])

        # Gating network to select experts
        self.gate = GatingNetwork(channels, num_experts)

        # Modality selection predictor
        self.modality_selector = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels * 2, channels // 4, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 4, 3, kernel_size=1),  # RGB, Thermal, Fused
            nn.Softmax(dim=1)
        )

    def forward(self, rgb_features: torch.Tensor, thermal_features: torch.Tensor) -> torch.Tensor:
        batch_size = rgb_features.shape[0]

        # Compute gating weights
        gate_input = torch.cat([rgb_features, thermal_features], dim=1)
        gate_weights = self.gate(gate_input)  # [batch, num_experts]

        # Apply experts
        expert_outputs = []
        for expert in self.experts:
            expert_out = expert(rgb_features, thermal_features)
            expert_outputs.append(expert_out)

        # Weighted combination of expert outputs
        expert_stack = torch.stack(expert_outputs, dim=1)  # [batch, num_experts, C, H, W]
        gate_weights = gate_weights.view(batch_size, self.num_experts, 1, 1, 1)

        routed_output = torch.sum(expert_stack * gate_weights, dim=1)

        # Modality selection
        modality_weights = self.modality_selector(gate_input)
        w_rgb = modality_weights[:, 0:1, :, :]
        w_thermal = modality_weights[:, 1:2, :, :]
        w_fused = modality_weights[:, 2:3, :, :]

        # Final output combining individual modalities and fused features
        final_output = (w_rgb * rgb_features +
                       w_thermal * thermal_features +
                       w_fused * routed_output)

        return final_output


class ExpertNetwork(nn.Module):
    """Individual expert network for specific fusion pattern."""

    def __init__(self, channels: int):
        super().__init__()
        self.rgb_branch = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels)
        )

        self.thermal_branch = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels)
        )

        self.fusion = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, rgb: torch.Tensor, thermal: torch.Tensor) -> torch.Tensor:
        rgb_processed = self.rgb_branch(rgb)
        thermal_processed = self.thermal_branch(thermal)

        combined = torch.cat([rgb_processed, thermal_processed], dim=1)
        fused = self.fusion(combined)

        return fused


class GatingNetwork(nn.Module):
    """Gating network for expert selection."""

    def __init__(self, channels: int, num_experts: int):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.gate_fc = nn.Sequential(
            nn.Linear(channels * 4, channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(channels, num_experts),
            nn.Softmax(dim=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, channels, _, _ = x.shape

        avg_pool = self.avg_pool(x).view(batch_size, channels * 2)
        max_pool = self.max_pool(x).view(batch_size, channels * 2)

        pooled = torch.cat([avg_pool, max_pool], dim=1)
        weights = self.gate_fc(pooled)

        return weights


class IlluminationInvariantFusion(nn.Module):
    """
    Fusion module robust to illumination changes.
    Particularly useful for day/night pedestrian detection scenarios.
    """

    def __init__(self, channels: int):
        super().__init__()

        # Illumination estimation branch
        self.illumination_est = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // 8, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 8, 1, kernel_size=1),
            nn.Sigmoid()
        )

        # RGB processing (illumination-dependent)
        self.rgb_proc = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )

        # Thermal processing (illumination-invariant)
        self.thermal_proc = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )

        # Adaptive fusion
        self.adaptive_fusion = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, rgb: torch.Tensor, thermal: torch.Tensor) -> torch.Tensor:
        # Estimate illumination level from RGB features
        illumination_score = self.illumination_est(rgb)

        # Process both modalities
        rgb_processed = self.rgb_proc(rgb)
        thermal_processed = self.thermal_proc(thermal)

        # Weight modalities based on illumination
        # Low illumination -> more weight on thermal
        thermal_weight = 1.0 - illumination_score
        rgb_weight = illumination_score

        weighted_rgb = rgb_processed * rgb_weight
        weighted_thermal = thermal_processed * thermal_weight

        # Fuse weighted features
        combined = torch.cat([weighted_rgb, weighted_thermal], dim=1)
        fused = self.adaptive_fusion(combined)

        return fused


class ComplementarityAwareFusion(nn.Module):
    """
    Exploits complementarity between RGB and thermal modalities.
    Identifies unique information from each modality for enhanced fusion.
    """

    def __init__(self, channels: int):
        super().__init__()

        # Extract modality-specific features
        self.rgb_specific = nn.Sequential(
            nn.Conv2d(channels, channels // 2, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels // 2),
            nn.ReLU(inplace=True)
        )

        self.thermal_specific = nn.Sequential(
            nn.Conv2d(channels, channels // 2, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels // 2),
            nn.ReLU(inplace=True)
        )

        # Extract shared features
        self.shared_extractor = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels)
        )

        # Complementarity gate
        self.complementarity_gate = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.Sigmoid()
        )

        # Final fusion
        self.final_fusion = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, rgb: torch.Tensor, thermal: torch.Tensor) -> torch.Tensor:
        # Extract modality-specific features
        rgb_spec = self.rgb_specific(rgb)
        thermal_spec = self.thermal_specific(thermal)

        # Extract shared features
        combined = torch.cat([rgb, thermal], dim=1)
        shared_feat = self.shared_extractor(combined)

        # Compute complementarity
        specific_concat = torch.cat([rgb_spec, thermal_spec], dim=1)
        complement_gate = self.complementarity_gate(specific_concat)

        # Apply complementarity-aware weighting
        complementary_feat = complement_gate * specific_concat

        # Final fusion
        final_combined = torch.cat([shared_feat, complementary_feat], dim=1)
        output = self.final_fusion(final_combined)

        return output
