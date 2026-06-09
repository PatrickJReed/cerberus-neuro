"""Tests for the AttributionResult dataclass and shared helpers."""

from __future__ import annotations

import torch

from cerberus_neuro.attribution.base import AttributionResult, channel_scores_from_saliency


def test_attribution_result_minimal():
    r = AttributionResult(
        saliency=None,
        channel_scores=torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6]),
        metadata={"method": "test"},
    )
    assert r.saliency is None
    assert r.channel_scores.shape == (6,)
    assert r.metadata == {"method": "test"}


def test_attribution_result_with_saliency():
    r = AttributionResult(
        saliency=torch.zeros(2, 6, 8, 8),
        channel_scores=torch.zeros(2, 6),
        metadata={"method": "test"},
    )
    assert r.saliency.shape == (2, 6, 8, 8)
    assert r.channel_scores.shape == (2, 6)


def test_channel_scores_from_saliency_per_sample():
    saliency = torch.tensor(
        [[[[1.0, 1.0], [1.0, 1.0]], [[2.0, 2.0], [2.0, 2.0]], [[0.0, 0.0], [0.0, 0.0]]]]
    )  # shape [1, 3, 2, 2]: channel 0 sums to 4, channel 1 sums to 8, channel 2 sums to 0
    scores = channel_scores_from_saliency(saliency.abs())
    assert scores.shape == (1, 3)
    assert torch.allclose(scores[0], torch.tensor([4.0, 8.0, 0.0]))


def test_channel_scores_handles_negative_values():
    # Negative attributions are kept signed for sum aggregation by default.
    saliency = torch.tensor([[[[1.0, -1.0]], [[0.5, 0.5]]]])  # [1, 2, 1, 2]
    scores = channel_scores_from_saliency(saliency)
    assert torch.allclose(scores[0], torch.tensor([0.0, 1.0]))
