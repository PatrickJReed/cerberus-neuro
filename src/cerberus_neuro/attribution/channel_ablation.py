"""Channel-ablation attribution.

For each input channel, zero it out, run the model, measure the change in
classification accuracy or per-sample target-class confidence. The score is
the *drop* (baseline - ablated): positive means the model relies on that
channel, near-zero means the channel is unused, negative means ablating the
channel *helped* (which is a useful red flag).

Designed for the 6-channel brightfield + Cell Painting input. No gradients;
just forward passes.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .base import AttributionResult


@torch.no_grad()
def compute_channel_ablation(
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
) -> AttributionResult:
    """Per-channel accuracy drop across the whole batch.

    Parameters
    ----------
    model
        Trained classifier returning ``[B, n_classes]`` logits.
    images
        Input tensor of shape ``[B, C, H, W]``.
    labels
        Ground-truth integer labels of shape ``[B]``.

    Returns
    -------
    :class:`AttributionResult` with ``saliency=None`` and ``channel_scores`` of
    shape ``[C]`` (one accuracy-drop score per input channel).
    """
    was_training = model.training
    model.eval()
    try:
        n_channels = images.shape[1]
        baseline_acc = _accuracy(model, images, labels)
        drops = torch.zeros(n_channels)
        for c in range(n_channels):
            ablated = images.clone()
            ablated[:, c, :, :] = 0.0
            ablated_acc = _accuracy(model, ablated, labels)
            drops[c] = baseline_acc - ablated_acc
        return AttributionResult(
            saliency=None,
            channel_scores=drops,
            metadata={
                "method": "channel_ablation",
                "baseline_accuracy": float(baseline_acc),
                "aggregation": "batch_accuracy_drop",
            },
        )
    finally:
        model.train(was_training)


@torch.no_grad()
def compute_channel_ablation_per_sample(
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
) -> AttributionResult:
    """Per-sample per-channel target-class confidence drop.

    Yields ``[B, C]`` channel_scores: how much zeroing channel ``c`` reduces the
    softmax probability assigned to the correct class for sample ``b``.

    Returns
    -------
    :class:`AttributionResult` with ``saliency=None`` and ``channel_scores`` of
    shape ``[B, C]``.
    """
    was_training = model.training
    model.eval()
    try:
        B, n_channels = images.shape[0], images.shape[1]  # noqa: N806 — B is the batch dimension
        baseline_conf = _target_class_confidence(model, images, labels)  # [B]
        drops = torch.zeros(B, n_channels)
        for c in range(n_channels):
            ablated = images.clone()
            ablated[:, c, :, :] = 0.0
            ablated_conf = _target_class_confidence(model, ablated, labels)  # [B]
            drops[:, c] = baseline_conf - ablated_conf
        return AttributionResult(
            saliency=None,
            channel_scores=drops,
            metadata={
                "method": "channel_ablation_per_sample",
                "aggregation": "per_sample_target_confidence_drop",
            },
        )
    finally:
        model.train(was_training)


def _accuracy(model: nn.Module, images: torch.Tensor, labels: torch.Tensor) -> float:
    logits = model(images)
    preds = logits.argmax(dim=-1)
    return (preds == labels).float().mean().item()


def _target_class_confidence(
    model: nn.Module, images: torch.Tensor, labels: torch.Tensor
) -> torch.Tensor:
    logits = model(images)
    probs = torch.softmax(logits, dim=-1)
    return probs.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
