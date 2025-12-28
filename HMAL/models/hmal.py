import math
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.common import C3, Conv


class ChannelAttention(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = torch.sigmoid(self.fc(self.pool(x)))
        return x * weights


class ModalityAwareFusion(nn.Module):
    """Auxiliary stream used in TCDE for early alignment."""

    def __init__(self, in_channels: int):
        super().__init__()
        fused_channels = in_channels * 2
        self.norm = nn.InstanceNorm2d(fused_channels, affine=True)
        self.attn = ChannelAttention(fused_channels)
        self.reduce = nn.Conv2d(fused_channels, in_channels, kernel_size=1, bias=False)
        self.mix = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, rgb: torch.Tensor, ir: torch.Tensor) -> torch.Tensor:
        fused = torch.cat([rgb, ir], dim=1)
        fused = self.norm(fused)
        fused = self.attn(fused)
        fused = self.reduce(fused)
        return self.mix(fused)


class AdvantageEstimator(nn.Module):
    """Estimate modality advantage weights."""

    def __init__(self, channels: int):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, 2, 1),
        )

    def forward(self, rgb: torch.Tensor, ir: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        pooled = torch.cat([self.pool(rgb), self.pool(ir)], dim=1)
        logits = self.head(pooled)
        weights = torch.softmax(logits, dim=1)
        return weights[:, :1], weights[:, 1:]


class CrossLevelFusion(nn.Module):
    """Cross-attention style fusion between high-level semantics and auxiliary details."""

    def __init__(self, high_channels: int, low_channels: int):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Conv2d(high_channels + low_channels, low_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(low_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(low_channels, low_channels, 1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, high: torch.Tensor, aux_low: torch.Tensor, fused_low: torch.Tensor) -> torch.Tensor:
        up = F.interpolate(high, size=fused_low.shape[2:], mode="nearest")
        attn = self.gate(torch.cat([up, aux_low], dim=1))
        return fused_low + attn * aux_low


class HMALDetector(nn.Module):
    """
    Hierarchical Modality Advantage Learning detector.
    Triple-stream encoder (RGB, IR, auxiliary) with advantage-guided fusion and YOLO-style head.
    """

    def __init__(self, nc: int = 1, anchors: List[List[int]] = None):
        super().__init__()
        self.nc = nc
        anchors = anchors or [
            [10, 13, 16, 30, 33, 23],  # P3/8
            [30, 61, 62, 45, 59, 119],  # P4/16
            [116, 90, 156, 198, 373, 326],  # P5/32
        ]

        # Stage 1
        self.rgb_stem = Conv(3, 32, k=3, s=2)
        self.ir_stem = Conv(3, 32, k=3, s=2)
        self.mafu = ModalityAwareFusion(32)

        # Shared backbone blocks
        self.stage2 = nn.Sequential(Conv(32, 64, k=3, s=2), C3(64, 64, n=2))
        self.stage3 = nn.Sequential(Conv(64, 128, k=3, s=2), C3(128, 128, n=4))
        self.stage4 = nn.Sequential(Conv(128, 256, k=3, s=2), C3(256, 256, n=4))
        self.stage5 = nn.Sequential(Conv(256, 256, k=3, s=2), C3(256, 256, n=2))

        # Advantage-guided fusion
        self.advantage = AdvantageEstimator(256)
        self.cross_level = CrossLevelFusion(high_channels=256, low_channels=128)

        # FPN / PAN head
        self.lat_p5 = Conv(256, 128, k=1, s=1)
        self.c3_p4 = C3(128 + 256, 192, n=2, shortcut=False)
        self.lat_p4 = Conv(192, 96, k=1, s=1)
        self.c3_p3 = C3(96 + 128, 160, n=2, shortcut=False)

        self.down_p3 = Conv(160, 96, k=3, s=2)
        self.c3_n4 = C3(96 + 192, 192, n=2, shortcut=False)
        self.down_p4 = Conv(192, 128, k=3, s=2)
        self.c3_n5 = C3(128 + 256, 256, n=2, shortcut=False)

        self.detect = Detect(nc=nc, anchors=anchors, ch=[160, 192, 256])
        self.stride = torch.tensor([8., 16., 32.])
        self.detect.stride = self.stride
        self.names = [str(i) for i in range(self.nc)]
        self.model = nn.ModuleList(
            [
                self.rgb_stem,
                self.ir_stem,
                self.mafu,
                self.stage2,
                self.stage3,
                self.stage4,
                self.stage5,
                self.advantage,
                self.cross_level,
                self.lat_p5,
                self.c3_p4,
                self.lat_p4,
                self.c3_p3,
                self.down_p3,
                self.c3_n4,
                self.down_p4,
                self.c3_n5,
                self.detect,
            ]
        )

    def forward(self, rgb: torch.Tensor, ir: torch.Tensor):
        # Stage 1
        rgb1 = self.rgb_stem(rgb)
        ir1 = self.ir_stem(ir)
        aux1 = self.mafu(rgb1, ir1)

        # Shared backbone
        rgb2, ir2, aux2 = self.stage2(rgb1), self.stage2(ir1), self.stage2(aux1)
        rgb3, ir3, aux3 = self.stage3(rgb2), self.stage3(ir2), self.stage3(aux2)
        rgb4, ir4 = self.stage4(rgb3), self.stage4(ir3)
        rgb5, ir5 = self.stage5(rgb4), self.stage5(ir4)

        # Advantage-aware fusion
        w_rgb, w_ir = self.advantage(rgb5, ir5)
        fused5 = w_rgb * rgb5 + w_ir * ir5
        fused4 = w_rgb * rgb4 + w_ir * ir4
        fused3 = w_rgb * rgb3 + w_ir * ir3
        fused3 = self.cross_level(fused5, aux3, fused3)

        # FPN top-down
        p5_lat = self.lat_p5(fused5)
        p5_up = F.interpolate(p5_lat, scale_factor=2, mode="nearest")
        p4 = self.c3_p4(torch.cat([p5_up, fused4], dim=1))

        p4_lat = self.lat_p4(p4)
        p4_up = F.interpolate(p4_lat, scale_factor=2, mode="nearest")
        p3 = self.c3_p3(torch.cat([p4_up, fused3], dim=1))

        # PAN bottom-up
        n4 = self.c3_n4(torch.cat([self.down_p3(p3), p4], dim=1))
        n5 = self.c3_n5(torch.cat([self.down_p4(n4), fused5], dim=1))

        return self.detect([p3, n4, n5])

    def fuse(self):
        # placeholder for compatibility
        return self
class Detect(nn.Module):
    """Lightweight Detect head (YOLO-style) with anchor grids."""
    stride = None
    export = False

    def __init__(self, nc=80, anchors=(), ch=()):
        super().__init__()
        self.nc = nc
        self.no = nc + 5
        self.nl = len(anchors)
        self.na = len(anchors[0]) // 2
        self.grid = [torch.zeros(1)] * self.nl
        a = torch.tensor(anchors).float().view(self.nl, -1, 2)
        self.register_buffer('anchors', a)
        self.register_buffer('anchor_grid', a.clone().view(self.nl, 1, -1, 1, 1, 2))
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)

    def forward(self, x):
        z = []
        self.training |= self.export
        for i in range(self.nl):
            x[i] = self.m[i](x[i])
            bs, _, ny, nx = x[i].shape
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()
            if not self.training:
                if self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i] = self._make_grid(nx, ny).to(x[i].device)
                y = x[i].sigmoid()
                y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grid[i]) * self.stride[i]
                y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]
                z.append(y.view(bs, -1, self.no))
        return x if self.training else (torch.cat(z, 1), x)

    @staticmethod
    def _make_grid(nx=20, ny=20):
        yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)], indexing='ij')
        return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()
