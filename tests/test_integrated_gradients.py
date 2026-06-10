"""Tests for Integrated Gradients attribution."""

from __future__ import annotations

import torch

from argus_cells.attribution.integrated_gradients import compute_integrated_gradients


def test_ig_returns_per_input_saliency(tiny_model_6ch, tiny_batch_6ch):
    result = compute_integrated_gradients(
        model=tiny_model_6ch,
        images=tiny_batch_6ch["images"],
        target_class=1,
        n_steps=8,
    )
    # IG saliency lives at input resolution + input channel count.
    assert result.saliency.shape == (4, 6, 64, 64)
    assert result.metadata["method"] == "integrated_gradients"
    assert result.metadata["n_steps"] == 8


def test_ig_channel_scores_shape(tiny_model_6ch, tiny_batch_6ch):
    result = compute_integrated_gradients(
        model=tiny_model_6ch,
        images=tiny_batch_6ch["images"],
        target_class=1,
        n_steps=8,
    )
    assert result.channel_scores.shape == (4, 6)


def test_ig_zero_input_yields_near_zero_saliency(tiny_model_6ch):
    """If both baseline and input are zero, IG must be near-zero everywhere
    (the integrand path has zero length)."""
    images = torch.zeros(2, 6, 64, 64)
    result = compute_integrated_gradients(
        model=tiny_model_6ch,
        images=images,
        target_class=1,
        n_steps=8,
    )
    assert result.saliency.abs().max().item() < 1e-4


def test_ig_changes_with_target_class(tiny_model_6ch, tiny_batch_6ch):
    """Different target class → different attribution map."""
    images = tiny_batch_6ch["images"]
    r0 = compute_integrated_gradients(
        model=tiny_model_6ch, images=images, target_class=0, n_steps=8
    )
    r1 = compute_integrated_gradients(
        model=tiny_model_6ch, images=images, target_class=1, n_steps=8
    )
    assert not torch.allclose(r0.saliency, r1.saliency)
