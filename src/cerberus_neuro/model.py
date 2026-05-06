"""Cerberus-inspired multi-task model.

Single ResNet34 encoder feeding three heterogeneous task heads:

- :class:`VirtualStainingHead` — U-Net-style decoder predicting 5
  fluorescence channels from brightfield input.
- :class:`ClassifierHead` (cell type) — 4-way softmax (stem / progen /
  neuron / astro).
- :class:`ClassifierHead` (line condition) — binary (control vs deletion).

ResNet34 follows the standard torchvision implementation but with a
single-channel input conv. Trained from scratch; no ImageNet weights
(per ``CLAUDE.md`` scope discipline for the public reproduction).

For a 256x256 input the encoder produces feature maps at strides
{2, 4, 8, 16, 32} with channel counts {64, 64, 128, 256, 512}; the
decoder upsamples from x4 back to the input resolution, concatenating
each encoder stage as a skip connection.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet34


@dataclass
class CerberusOutput:
    cell_type_logits: torch.Tensor       # (N, n_cell_types)
    line_condition_logits: torch.Tensor  # (N, n_line_conditions)
    fluorescence_pred: torch.Tensor      # (N, n_fluorescence_channels, H, W)


class ResNet34Encoder(nn.Module):
    """ResNet34 trunk producing feature maps at five stride levels."""

    def __init__(self, in_channels: int = 1):
        super().__init__()
        base = resnet34(weights=None)
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = base.bn1
        self.relu = base.relu
        self.maxpool = base.maxpool
        self.layer1 = base.layer1
        self.layer2 = base.layer2
        self.layer3 = base.layer3
        self.layer4 = base.layer4

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        x0 = self.relu(self.bn1(self.conv1(x)))  # (64, H/2,  W/2)
        x1 = self.layer1(self.maxpool(x0))       # (64, H/4,  W/4)
        x2 = self.layer2(x1)                     # (128, H/8,  W/8)
        x3 = self.layer3(x2)                     # (256, H/16, W/16)
        x4 = self.layer4(x3)                     # (512, H/32, W/32)
        return x0, x1, x2, x3, x4


class _UpBlock(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch + skip_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.conv(torch.cat([x, skip], dim=1))


class VirtualStainingHead(nn.Module):
    """U-Net-style decoder. Output activated with sigmoid to match [0, 1] targets."""

    def __init__(self, out_channels: int = 5):
        super().__init__()
        self.up3 = _UpBlock(512, 256, 256)  # H/32 -> H/16
        self.up2 = _UpBlock(256, 128, 128)  # H/16 -> H/8
        self.up1 = _UpBlock(128, 64, 64)    # H/8  -> H/4
        self.up0 = _UpBlock(64, 64, 32)     # H/4  -> H/2
        self.up_final = nn.Sequential(
            nn.Conv2d(32, 16, 3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
        )
        self.final = nn.Conv2d(16, out_channels, 1)

    def forward(
        self,
        x0: torch.Tensor, x1: torch.Tensor, x2: torch.Tensor,
        x3: torch.Tensor, x4: torch.Tensor,
        target_size: tuple[int, int],
    ) -> torch.Tensor:
        x = self.up3(x4, x3)
        x = self.up2(x, x2)
        x = self.up1(x, x1)
        x = self.up0(x, x0)
        x = F.interpolate(x, size=target_size, mode="bilinear", align_corners=False)
        x = self.up_final(x)
        return torch.sigmoid(self.final(x))


class ClassifierHead(nn.Module):
    def __init__(self, in_channels: int = 512, n_classes: int = 4):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(in_channels, n_classes)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.fc(self.pool(features).flatten(1))


class CerberusModel(nn.Module):
    """Shared ResNet34 encoder + three heterogeneous task heads.

    Default head sizes match the v0 task definitions: 4-way cell-type,
    binary line-condition, 5-channel virtual staining.
    """

    def __init__(
        self,
        in_channels: int = 1,
        n_cell_types: int = 4,
        n_line_conditions: int = 2,
        n_fluorescence_channels: int = 5,
    ):
        super().__init__()
        self.encoder = ResNet34Encoder(in_channels=in_channels)
        self.cell_type_head = ClassifierHead(512, n_cell_types)
        self.line_condition_head = ClassifierHead(512, n_line_conditions)
        self.fluorescence_head = VirtualStainingHead(n_fluorescence_channels)

    def forward(self, x: torch.Tensor) -> CerberusOutput:
        target_size = x.shape[-2:]
        x0, x1, x2, x3, x4 = self.encoder(x)
        return CerberusOutput(
            cell_type_logits=self.cell_type_head(x4),
            line_condition_logits=self.line_condition_head(x4),
            fluorescence_pred=self.fluorescence_head(x0, x1, x2, x3, x4, target_size),
        )

    def parameter_count(self) -> dict[str, int]:
        def count(m: nn.Module) -> int:
            return sum(p.numel() for p in m.parameters() if p.requires_grad)
        return {
            "encoder": count(self.encoder),
            "cell_type_head": count(self.cell_type_head),
            "line_condition_head": count(self.line_condition_head),
            "fluorescence_head": count(self.fluorescence_head),
            "total": count(self),
        }
