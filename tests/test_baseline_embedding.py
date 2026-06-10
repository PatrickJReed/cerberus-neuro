"""Test that BaselineDiseaseClassifier exposes a 512-dim embedding extractor."""

from __future__ import annotations

import torch

from argus_cells.model import BaselineDiseaseClassifier


def test_extract_embedding_returns_512d_vector():
    model = BaselineDiseaseClassifier(in_channels=6, n_classes=2, pretrained_encoder=False)
    model.eval()
    x = torch.randn(3, 6, 64, 64)
    with torch.no_grad():
        emb = model.extract_embedding(x)
    assert emb.shape == (3, 512)
    assert emb.dtype == torch.float32


def test_extract_embedding_no_grad_does_not_require_grad():
    """The extractor should work cleanly under no_grad; output should be
    a plain Tensor without requires_grad set when called under no_grad."""
    model = BaselineDiseaseClassifier(in_channels=6, n_classes=2, pretrained_encoder=False)
    model.eval()
    x = torch.randn(2, 6, 64, 64)
    with torch.no_grad():
        emb = model.extract_embedding(x)
    assert emb.requires_grad is False
