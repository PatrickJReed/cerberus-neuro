"""Tests for cross-architecture saliency agreement (Spearman)."""

from __future__ import annotations

import pytest
import torch

from argus_cells.analysis import saliency_agreement
from argus_cells.attribution.base import AttributionResult


def _res(saliency):
    return AttributionResult(saliency=saliency, channel_scores=None, metadata={})


def test_identical_saliency_agrees_perfectly():
    s = torch.randn(4, 1, 8, 8)
    out = saliency_agreement(_res(s), _res(s.clone()))
    assert out["per_sample"].shape == (4,)
    assert torch.allclose(out["per_sample"], torch.ones(4), atol=1e-5)
    assert abs(out["mean"] - 1.0) < 1e-5


def test_rank_reversed_saliency_anticorrelates():
    s = torch.randn(3, 1, 8, 8)
    out = saliency_agreement(_res(s), _res(-s))  # monotonic-decreasing => Spearman -1
    assert torch.allclose(out["per_sample"], -torch.ones(3), atol=1e-5)


def test_none_saliency_raises():
    s = torch.randn(2, 1, 8, 8)
    with pytest.raises(ValueError):
        saliency_agreement(_res(s), _res(None))


def test_shape_mismatch_raises():
    with pytest.raises(ValueError):
        saliency_agreement(_res(torch.randn(2, 6, 8, 8)), _res(torch.randn(2, 1, 8, 8)))


def test_constant_sample_does_not_poison_mean():
    # Sample 0 has a constant saliency map in both results: Spearman is NaN for
    # that sample. Samples 1 and 2 are identical (Spearman 1.0). The NaN must not
    # crash the function nor poison the mean; the mean is over finite samples only.
    s_a = torch.randn(3, 1, 8, 8)
    s_b = s_a.clone()
    s_a[0] = 1.0  # constant => zero variance => Spearman NaN
    s_b[0] = 1.0

    out = saliency_agreement(_res(s_a), _res(s_b))

    assert out["per_sample"].shape == (3,)
    assert torch.isnan(out["per_sample"][0])
    assert torch.allclose(out["per_sample"][1:], torch.ones(2), atol=1e-5)
    # Mean over the two finite (1.0) samples, ignoring the NaN sample.
    assert abs(out["mean"] - 1.0) < 1e-5
