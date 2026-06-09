"""Tests for attention-rollout attribution (Abnar & Zuidema, 2020).

Two layers of coverage:

- B1: rollout math on synthetic, known attention matrices via a stub model that
  exposes a hand-set ``attention_maps`` list and a no-op forward. Lets us assert
  the algebra directly (identity → uniform importance; a peaked attention →
  that token dominates) without depending on a trained network.
- B2: end-to-end smoke on a real :class:`ArgusCCT`, pinning the public shape
  contract the harness relies on.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from cerberus_neuro.attribution import compute_attention_rollout
from cerberus_neuro.attribution.base import AttributionResult


class _StubCCT(nn.Module):
    """Minimal model exposing the ArgusCCT rollout contract.

    Holds a fixed ``attention_maps`` list (``num_layers`` tensors of shape
    ``[B, num_heads, T, T]``) and re-populates it on every forward, exactly as
    ``ArgusCCT._encode`` does. ``forward`` is a no-op that returns dummy logits;
    rollout only reads ``attention_maps``.
    """

    def __init__(self, maps: list[torch.Tensor], n_classes: int = 2):
        super().__init__()
        self._maps_template = maps
        self.attention_maps: list[torch.Tensor] = []
        self.n_classes = n_classes

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        # Side-effect mirrors the real model: repopulate on every forward.
        self.attention_maps = [m.clone() for m in self._maps_template]
        batch = images.shape[0]
        return torch.zeros(batch, self.n_classes)


def _uniform_maps(batch: int, num_heads: int, n_tokens: int, num_layers: int) -> list[torch.Tensor]:
    """Identity-style attention: each token attends only to itself, every layer."""
    eye = torch.eye(n_tokens)
    one = eye.view(1, 1, n_tokens, n_tokens).expand(batch, num_heads, n_tokens, n_tokens)
    return [one.clone() for _ in range(num_layers)]


def _peaked_maps(
    batch: int, num_heads: int, n_tokens: int, num_layers: int, hot: int
) -> list[torch.Tensor]:
    """Every query token attends entirely to a single ``hot`` token, every layer."""
    attn = torch.zeros(batch, num_heads, n_tokens, n_tokens)
    attn[:, :, :, hot] = 1.0
    return [attn.clone() for _ in range(num_layers)]


def test_identity_attention_gives_uniform_importance():
    """B1a: identity attention across all layers → every token equally important.

    Rolling out identity (with residual mixing + row-normalisation) stays a
    uniform doubly-stochastic-ish map, so the mean-over-queries importance is
    constant across tokens. After per-sample min-max it collapses to all-equal,
    which the implementation maps to a flat (here, all-zero after min-max of a
    constant) but finite saliency. We assert the pre-normalisation uniformity by
    checking the saliency is spatially constant per sample.
    """
    batch, num_heads, num_layers = 2, 3, 4
    grid = 4
    n_tokens = grid * grid
    maps = _uniform_maps(batch, num_heads, n_tokens, num_layers)
    model = _StubCCT(maps)
    images = torch.randn(batch, 6, 16, 16)

    result = compute_attention_rollout(model, images, target_class=1)

    assert isinstance(result, AttributionResult)
    assert result.saliency.shape == (batch, 1, 16, 16)
    assert torch.isfinite(result.saliency).all()
    # Spatially constant per sample (uniform importance over tokens).
    for b in range(batch):
        flat = result.saliency[b, 0].flatten()
        assert torch.allclose(flat, flat[0].expand_as(flat), atol=1e-5)


def test_peaked_attention_makes_one_token_dominate():
    """B1b: attention concentrated on one token → that token has max importance.

    The hot token receives all attention from every query at every layer, so it
    is the most-attended token. After upsampling, its spatial region carries the
    saliency maximum and the rest is near the minimum.
    """
    batch, num_heads, num_layers = 1, 2, 3
    grid = 4
    n_tokens = grid * grid
    hot = 5  # token (row=1, col=1) in the 4x4 grid
    maps = _peaked_maps(batch, num_heads, n_tokens, num_layers, hot=hot)
    model = _StubCCT(maps)
    images = torch.randn(batch, 6, 16, 16)

    result = compute_attention_rollout(model, images, target_class=1)

    assert result.saliency.shape == (1, 1, 16, 16)
    assert torch.isfinite(result.saliency).all()
    # Recover the [B, T] token importance the spatial map was built from by
    # downsampling back to the token grid via average pooling.
    g = grid
    cell = 16 // g
    pooled = torch.nn.functional.avg_pool2d(result.saliency, kernel_size=cell)
    token_grid = pooled[0, 0]  # [g, g]
    hot_r, hot_c = divmod(hot, g)
    hot_val = token_grid[hot_r, hot_c]
    # The hot token's region is the maximum and clearly above the median.
    assert torch.isclose(hot_val, token_grid.max())
    assert hot_val > token_grid.median()


def test_returns_attribution_result_with_metadata():
    """B1c: the metadata dict carries method name and target_class."""
    batch, num_heads, num_layers, n_tokens = 2, 2, 2, 9
    maps = _uniform_maps(batch, num_heads, n_tokens, num_layers)
    model = _StubCCT(maps)
    images = torch.randn(batch, 6, 12, 12)

    result = compute_attention_rollout(model, images, target_class=0)

    assert result.metadata["method"] == "attention_rollout"
    assert result.metadata["target_class"] == 0
    # Channel-agnostic: per-sample channel scores are broadcast across C.
    assert result.channel_scores.shape == (batch, 6)
    for b in range(batch):
        row = result.channel_scores[b]
        assert torch.allclose(row, row[0].expand_as(row))


def test_rollout_on_argus_cct():
    """B2: end-to-end on a real ArgusCCT (shape contract for the harness)."""
    from cerberus_neuro.models.cct import ArgusCCT

    model = ArgusCCT(in_channels=6, n_classes=2, img_size=64).eval()
    images = torch.randn(2, 6, 64, 64)
    result = compute_attention_rollout(model, images, target_class=1)
    assert result.saliency.shape == (2, 1, 64, 64)
    assert torch.isfinite(result.saliency).all()
    assert result.channel_scores.shape[0] == 2
