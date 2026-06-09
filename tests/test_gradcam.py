"""Tests for the GradCAM attribution method."""

from __future__ import annotations

import torch

from cerberus_neuro.attribution.gradcam import compute_gradcam


def test_compute_gradcam_returns_saliency_at_input_resolution(tiny_model_6ch, tiny_batch_6ch):
    images = tiny_batch_6ch["images"]
    result = compute_gradcam(
        model=tiny_model_6ch,
        target_layer=tiny_model_6ch.encoder.layer4,
        images=images,
        target_class=1,
    )
    # Saliency upsampled to the input H, W. Shape is [B, 1, H, W] (single
    # channel-agnostic spatial map per sample).
    assert result.saliency.shape == (4, 1, 64, 64)
    # Saliency is non-negative after ReLU.
    assert (result.saliency >= 0).all()
    # Method metadata is populated.
    assert result.metadata["method"] == "gradcam"
    assert result.metadata["target_class"] == 1


def test_compute_gradcam_channel_scores_have_shape_B_by_C_with_input_channels(  # noqa: N802 — B/C name the tensor shape axes
    tiny_model_6ch,
    tiny_batch_6ch,
):
    """GradCAM saliency is channel-agnostic (single map per sample), so we
    broadcast its sum across the input-channel axis: channel_scores[b, c] =
    sum(saliency[b, 0]) for all c. This keeps the AttributionResult shape
    uniform with IG and channel ablation.
    """
    result = compute_gradcam(
        model=tiny_model_6ch,
        target_layer=tiny_model_6ch.encoder.layer4,
        images=tiny_batch_6ch["images"],
        target_class=1,
    )
    assert result.channel_scores.shape == (4, 6)
    # Per-sample, the 6 channel scores are equal (broadcast from one map).
    for b in range(4):
        first = result.channel_scores[b, 0]
        assert torch.allclose(result.channel_scores[b], first.expand(6))


def test_compute_gradcam_target_class_zero_gives_different_map(
    tiny_model_6ch,
    tiny_batch_6ch,
):
    """GradCAM for class 0 should differ from class 1 (the gradient targets
    different logits)."""
    images = tiny_batch_6ch["images"]
    r0 = compute_gradcam(
        model=tiny_model_6ch,
        target_layer=tiny_model_6ch.encoder.layer4,
        images=images,
        target_class=0,
    )
    r1 = compute_gradcam(
        model=tiny_model_6ch,
        target_layer=tiny_model_6ch.encoder.layer4,
        images=images,
        target_class=1,
    )
    assert not torch.allclose(r0.saliency, r1.saliency)
