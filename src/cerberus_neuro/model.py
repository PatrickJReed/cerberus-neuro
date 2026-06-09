"""Cerberus-inspired multi-task model + all-channel baseline.

Two modules in this file:

1. :class:`CerberusModel` — single ResNet34 encoder feeding three heterogeneous
   task heads from a 1-channel brightfield input:

   - :class:`VirtualStainingHead` — U-Net-style decoder predicting 5
     fluorescence channels.
   - :class:`ClassifierHead` (cell type) — 4-way softmax
     (stem / progen / neuron / astro).
   - :class:`ClassifierHead` (line condition) — binary (control vs deletion).

   This is the headline v0 model. The novel claim is "disease state
   recoverable from brightfield alone when the encoder is forced to also
   predict fluorescence."

2. :class:`BaselineDiseaseClassifier` — same ResNet34 encoder but takes the
   full 6-channel stack (BF + 5 fluorescence) as input, with only the
   line-condition head. Establishes the all-channel disease-accuracy upper
   bound the Cerberus model is compared against in the paired-experiment
   v0 evaluation.

ResNet34 follows the standard torchvision implementation. The encoder is
initialized from torchvision's ImageNet1K_V1 weights by default
(``pretrained_encoder=True``); ``conv1`` is rebuilt to accept the model's
``in_channels`` while preserving expected activation magnitude (mean for
1-channel BF; tile-and-scale for the 6-channel baseline). Pass
``pretrained_encoder=False`` for the clean-from-scratch ablation.

For a 256x256 input the encoder produces feature maps at strides
{2, 4, 8, 16, 32} with channel counts {64, 64, 128, 256, 512}; the
decoder upsamples from x4 back to the input resolution, concatenating
each encoder stage as a skip connection.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812 — conventional PyTorch alias
from torchvision.models import ResNet34_Weights, resnet34


@dataclass
class CerberusOutput:
    cell_type_logits: torch.Tensor  # (N, n_cell_types)
    line_condition_logits: torch.Tensor  # (N, n_line_conditions)
    fluorescence_logits: (
        torch.Tensor
    )  # (N, n_fluorescence_channels, H, W) — raw logits; apply sigmoid for [0, 1] probability masks


def _adapt_conv1(pretrained_weight: torch.Tensor, in_channels: int) -> torch.Tensor:
    """Adapt a pretrained 3-channel conv1 to ``in_channels`` while preserving
    expected activation magnitude.

    - 1 channel: mean across input channels.
    - 3 channels: pass through unchanged.
    - >3 channels: tile the pretrained weights and scale by ``3 / in_channels``.
    """
    if in_channels == 3:
        return pretrained_weight.clone()
    if in_channels == 1:
        return pretrained_weight.mean(dim=1, keepdim=True).clone()
    reps = (in_channels + 2) // 3
    tiled = pretrained_weight.repeat(1, reps, 1, 1)[:, :in_channels]
    return (tiled * (3.0 / in_channels)).clone()


class ResNet34Encoder(nn.Module):
    """ResNet34 trunk producing feature maps at five stride levels.

    When ``pretrained=True`` (default), the trunk is initialized from
    torchvision's ImageNet1K_V1 weights and ``conv1`` is rebuilt to accept
    ``in_channels`` while preserving expected activation magnitude. This is
    the standard microscopy-CV transfer-learning recipe; ``pretrained=False``
    falls back to random init for the clean public-reproduction story.
    """

    def __init__(self, in_channels: int = 1, pretrained: bool = True):
        super().__init__()
        weights = ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
        base = resnet34(weights=weights)
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        if pretrained:
            with torch.no_grad():
                self.conv1.weight.copy_(_adapt_conv1(base.conv1.weight, in_channels))
        self.bn1 = base.bn1
        self.relu = base.relu
        self.maxpool = base.maxpool
        self.layer1 = base.layer1
        self.layer2 = base.layer2
        self.layer3 = base.layer3
        self.layer4 = base.layer4

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        x0 = self.relu(self.bn1(self.conv1(x)))  # (64, H/2,  W/2)
        x1 = self.layer1(self.maxpool(x0))  # (64, H/4,  W/4)
        x2 = self.layer2(x1)  # (128, H/8,  W/8)
        x3 = self.layer3(x2)  # (256, H/16, W/16)
        x4 = self.layer4(x3)  # (512, H/32, W/32)
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
    """U-Net-style decoder.

    Returns raw **logits** (no sigmoid). The downstream loss is
    ``F.binary_cross_entropy_with_logits``, which fuses sigmoid+BCE for
    numerical stability under AMP autocast (plain ``binary_cross_entropy``
    is rejected by autocast). For inference / visualization, callers should
    apply ``torch.sigmoid()`` to the head's output to get [0, 1] probability
    masks.
    """

    def __init__(self, out_channels: int = 5):
        super().__init__()
        self.up3 = _UpBlock(512, 256, 256)  # H/32 -> H/16
        self.up2 = _UpBlock(256, 128, 128)  # H/16 -> H/8
        self.up1 = _UpBlock(128, 64, 64)  # H/8  -> H/4
        self.up0 = _UpBlock(64, 64, 32)  # H/4  -> H/2
        self.up_final = nn.Sequential(
            nn.Conv2d(32, 16, 3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
        )
        self.final = nn.Conv2d(16, out_channels, 1)

    def forward(
        self,
        x0: torch.Tensor,
        x1: torch.Tensor,
        x2: torch.Tensor,
        x3: torch.Tensor,
        x4: torch.Tensor,
        target_size: tuple[int, int],
    ) -> torch.Tensor:
        x = self.up3(x4, x3)
        x = self.up2(x, x2)
        x = self.up1(x, x1)
        x = self.up0(x, x0)
        x = F.interpolate(x, size=target_size, mode="bilinear", align_corners=False)
        x = self.up_final(x)
        return self.final(x)


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

    # Stable identity for training-loop dispatch. Survives sys.modules purges
    # that would otherwise break isinstance checks (different class objects
    # with the same name after a re-import).
    model_kind = "cerberus"

    def __init__(
        self,
        in_channels: int = 1,
        n_cell_types: int = 4,
        n_line_conditions: int = 2,
        n_fluorescence_channels: int = 5,
        pretrained_encoder: bool = True,
    ):
        super().__init__()
        self.encoder = ResNet34Encoder(in_channels=in_channels, pretrained=pretrained_encoder)
        self.cell_type_head = ClassifierHead(512, n_cell_types)
        self.line_condition_head = ClassifierHead(512, n_line_conditions)
        self.fluorescence_head = VirtualStainingHead(n_fluorescence_channels)

    def forward(self, x: torch.Tensor) -> CerberusOutput:
        target_size = x.shape[-2:]
        x0, x1, x2, x3, x4 = self.encoder(x)
        return CerberusOutput(
            cell_type_logits=self.cell_type_head(x4),
            line_condition_logits=self.line_condition_head(x4),
            fluorescence_logits=self.fluorescence_head(x0, x1, x2, x3, x4, target_size),
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


class CellTypeOnlyModel(nn.Module):
    """Sanity-check model: encoder + cell-type head only.

    Used to isolate whether the pretrained encoder can learn cell-type
    discrimination on this dataset *without* multi-task interference. If
    cell-type accuracy doesn't reach a healthy value (~0.80+) with this
    stripped-down model in a few epochs, the issue is in the data pipeline,
    augmentation, or label assignment, not the multi-task setup.
    """

    model_kind = "cell_type_only"

    def __init__(self, in_channels: int = 1, n_classes: int = 4, pretrained_encoder: bool = True):
        super().__init__()
        self.encoder = ResNet34Encoder(in_channels=in_channels, pretrained=pretrained_encoder)
        self.head = ClassifierHead(512, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, _, _, _, x4 = self.encoder(x)
        return self.head(x4)


class VirtualStainingOnlyModel(nn.Module):
    """Sanity-check model: encoder + U-Net decoder, no classifier heads.

    Used to isolate whether the U-Net decoder can learn virtual staining
    *without* classifier-head gradient interference. If virtual-staining loss
    doesn't drop meaningfully below the random-init baseline with this model
    in a few epochs, the issue is in the decoder architecture or data, not
    the multi-task balance.
    """

    model_kind = "vs_only"

    def __init__(
        self, in_channels: int = 1, out_channels: int = 5, pretrained_encoder: bool = True
    ):
        super().__init__()
        self.encoder = ResNet34Encoder(in_channels=in_channels, pretrained=pretrained_encoder)
        self.decoder = VirtualStainingHead(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        target_size = x.shape[-2:]
        x0, x1, x2, x3, x4 = self.encoder(x)
        return self.decoder(x0, x1, x2, x3, x4, target_size)


class BaselineDiseaseClassifier(nn.Module):
    """All-channel single-task disease classifier (the v0 upper-bound baseline).

    Same ResNet34 encoder as :class:`CerberusModel`, but takes the full 6-channel
    stack (brightfield + 5 fluorescence) as input and exposes only the
    line-condition head. Establishes "what's the best disease accuracy you can
    get with all the data, no virtual-staining task to share gradient with?".
    The Cerberus model's disease number is meaningful in comparison to this
    upper bound: it answers "how much of that signal is recoverable from
    brightfield alone, when the encoder is forced to also predict fluorescence?".
    """

    model_kind = "baseline"

    def __init__(self, in_channels: int = 6, n_classes: int = 2, pretrained_encoder: bool = True):
        super().__init__()
        self.encoder = ResNet34Encoder(in_channels=in_channels, pretrained=pretrained_encoder)
        self.head = ClassifierHead(512, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, _, _, _, x4 = self.encoder(x)
        return self.head(x4)

    def extract_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """Return the 512-dim pre-classifier embedding for the donor probe.

        This is the global-average-pooled output of the encoder's final conv
        stage. Shape: ``[B, 512]``.
        """
        _, _, _, _, x4 = self.encoder(x)
        pooled = torch.nn.functional.adaptive_avg_pool2d(x4, output_size=1)
        return pooled.flatten(1)

    def parameter_count(self) -> dict[str, int]:
        def count(m: nn.Module) -> int:
            return sum(p.numel() for p in m.parameters() if p.requires_grad)

        return {
            "encoder": count(self.encoder),
            "head": count(self.head),
            "total": count(self),
        }
