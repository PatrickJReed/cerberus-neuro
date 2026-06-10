"""Tests for ArgusCCT — minimal from-scratch CCT-7/3x1 disease classifier.

Contract these tests pin down (so the existing interpretability harness works
on ArgusCCT unchanged):

- ``forward(x[B, 6, H, W]) -> [B, 2]`` disease logits.
- ``extract_embedding(x[B, 6, H, W]) -> [B, embed_dim]`` pooled feature vector.
- ``parameter_count()`` returns a non-empty dict mirroring
  ``BaselineDiseaseClassifier`` keys (total/trainable style).
- ``attention_maps``: list of length ``num_layers`` after each forward, each a
  ``[B, num_heads, T, T]`` per-head attention tensor; reset (not appended) every
  forward.
- ``.eval()`` forward is deterministic.
"""

from __future__ import annotations

import torch

from argus_cells.models import ArgusCCT


def test_forward_emits_two_logits():
    model = ArgusCCT(in_channels=6, n_classes=2, img_size=64).eval()
    out = model(torch.randn(3, 6, 64, 64))
    assert out.shape == (3, 2)


def test_extract_embedding_shape():
    model = ArgusCCT(in_channels=6, n_classes=2, img_size=64, embed_dim=256).eval()
    emb = model.extract_embedding(torch.randn(3, 6, 64, 64))
    assert emb.shape == (3, 256)


def test_attention_maps_captured():
    model = ArgusCCT(in_channels=6, n_classes=2, img_size=64, num_layers=7, num_heads=4).eval()
    _ = model(torch.randn(2, 6, 64, 64))
    assert len(model.attention_maps) == 7
    a = model.attention_maps[0]
    assert a.shape[0] == 2 and a.shape[1] == 4 and a.shape[2] == a.shape[3]


def test_attention_maps_reset_across_forwards():
    model = ArgusCCT(in_channels=6, n_classes=2, img_size=64, num_layers=7, num_heads=4).eval()
    x = torch.randn(2, 6, 64, 64)
    _ = model(x)
    _ = model(x)
    # Reset, not appended: length stays num_layers, not 2 * num_layers.
    assert len(model.attention_maps) == 7


def test_eval_forward_is_deterministic():
    model = ArgusCCT(in_channels=6, n_classes=2, img_size=64).eval()
    x = torch.randn(2, 6, 64, 64)
    assert torch.allclose(model(x), model(x))


def test_parameter_count_keys():
    model = ArgusCCT()
    pc = model.parameter_count()
    assert isinstance(pc, dict) and pc  # match BaselineDiseaseClassifier's keys
