"""The Argus-CCT conv tokenizer must downsample enough that the token count is
manageable at the production 256x256 crop size.

Transformer self-attention is O(T^2). With only 4x downsampling, a 256x256 crop
yields 64x64 = 4096 tokens, whose per-layer attention is ~17 GB at batch 64 and
will not fit on an A100. The tokenizer downsamples 16x by default (4 conv+pool
stages), giving a 16x16 = 256-token sequence at 256px.
"""

from __future__ import annotations

import torch

from argus_cells.models.cct import ArgusCCT, ConvTokenizer


def test_token_count_at_256px_is_256():
    model = ArgusCCT(in_channels=6, n_classes=2, img_size=256)
    assert model.num_tokens == 256  # 16x16 grid via 16x downsampling


def test_forward_and_attention_at_256px():
    model = ArgusCCT(in_channels=6, n_classes=2, img_size=256, num_heads=4).eval()
    x = torch.randn(2, 6, 256, 256)
    assert model(x).shape == (2, 2)
    # 256 tokens, not 4096: attention is [B, num_heads, T, T].
    assert model.attention_maps[0].shape == (2, 4, 256, 256)


def test_tokenizer_n_conv_layers_is_configurable():
    # 3 conv stages -> 8x downsample -> 32x32 = 1024 tokens at 256px.
    tok = ConvTokenizer(in_channels=6, embed_dim=64, n_conv_layers=3)
    out = tok(torch.zeros(1, 6, 256, 256))
    assert out.shape == (1, 1024, 64)
