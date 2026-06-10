"""ArgusCCT — minimal from-scratch Compact Convolutional Transformer.

A CCT-7/3x1-style classifier for 6-channel Cell Painting disease prediction.
"Compact Convolutional Transformer" (Hassani et al., 2021): replace the patch
embedding of a ViT with a small conv tokenizer, drop the class token in favor
of sequence pooling, and you get a transformer that trains from scratch on
modest data without ImageNet pretraining. That from-scratch property is the
point here — it matches the clean public-reproduction story of the rest of the
repo and gives a transformer-family counterpart to the ResNet34
:class:`~argus_cells.model.BaselineDiseaseClassifier`.

The model satisfies the same interpretability contract as the baseline so the
existing harness runs on it unchanged:

- ``forward(x[B, in_channels, H, W]) -> [B, n_classes]`` disease logits, for
  Integrated Gradients and channel ablation.
- ``extract_embedding(x) -> [B, embed_dim]`` pooled feature vector, for the
  linear probes.
- ``parameter_count()`` mirroring the baseline's return shape.

Beyond the baseline contract, every forward records per-head attention weights
into ``self.attention_maps`` (a list of length ``num_layers``, each
``[B, num_heads, T, T]``). A later ``compute_attention_rollout`` consumes these,
so the per-head shape is part of the contract and the list is reset (not
appended) at the start of every forward.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812 — conventional PyTorch alias


class ConvTokenizer(nn.Module):
    """Conv stack mapping an image to a token sequence.

    ``n_conv_layers`` blocks of (Conv2d 3x3 s1 p1 -> ReLU -> MaxPool2d 3x3 s2 p1)
    take ``[B, in_channels, H, W]`` to ``[B, embed_dim, h', w']``; each block
    halves the spatial resolution, so the grid is downsampled by
    ``2 ** n_conv_layers`` overall. The default of 4 (16x downsampling) keeps the
    256x256 production crops at a 16x16 = 256-token sequence; transformer
    attention is O(T^2), so the token count must stay modest. The spatial grid is
    then flattened to a token sequence ``[B, h'*w', embed_dim]``.
    """

    def __init__(self, in_channels: int, embed_dim: int, n_conv_layers: int = 4):
        super().__init__()
        hidden = embed_dim // 2
        # in -> hidden for every block but the last, hidden -> embed_dim on the
        # last. Holding the high-resolution early blocks at `hidden` (not the
        # full embed_dim) keeps activation memory down.
        chans = [in_channels] + [hidden] * (n_conv_layers - 1) + [embed_dim]
        self.blocks = nn.Sequential(
            *(
                nn.Sequential(
                    nn.Conv2d(chans[i], chans[i + 1], kernel_size=3, stride=1, padding=1),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                )
                for i in range(n_conv_layers)
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.blocks(x)  # [B, embed_dim, h', w']
        return x.flatten(2).transpose(1, 2)  # [B, T, embed_dim]


class TransformerEncoderLayer(nn.Module):
    """Pre-norm transformer encoder layer with per-head attention capture.

    LayerNorm -> MultiheadAttention -> residual; LayerNorm -> MLP -> residual.
    ``forward`` returns ``(x, attn)`` where ``attn`` is the per-head attention
    tensor ``[B, num_heads, T, T]`` (``average_attn_weights=False``).
    """

    def __init__(self, embed_dim: int, num_heads: int, mlp_ratio: float, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        hidden = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.norm1(x)
        attn_out, attn_weights = self.attn(h, h, h, need_weights=True, average_attn_weights=False)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x, attn_weights  # attn_weights: [B, num_heads, T, T]


class SeqPool(nn.Module):
    """CCT sequence-pooling head.

    A ``Linear(embed_dim, 1)`` scores each token; a softmax over tokens turns
    the scores into attention weights; the weighted sum of token features yields
    a single ``[B, embed_dim]`` vector. Replaces the class token of a ViT.
    """

    def __init__(self, embed_dim: int):
        super().__init__()
        self.score = nn.Linear(embed_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, embed_dim]
        weights = F.softmax(self.score(x), dim=1)  # [B, T, 1]
        return torch.bmm(weights.transpose(1, 2), x).squeeze(1)  # [B, embed_dim]


class ArgusCCT(nn.Module):
    """Compact Convolutional Transformer (CCT-7/3x1 style) disease classifier.

    Conv tokenizer -> learnable positional embedding -> ``num_layers`` pre-norm
    transformer encoder layers -> sequence pooling -> linear classifier. No
    class token; from-scratch (no pretraining).
    """

    # Stable identity for harness/training-loop dispatch, mirroring the
    # ``model_kind`` convention in argus_cells.model.
    model_kind = "argus_cct"

    def __init__(
        self,
        in_channels: int = 6,
        n_classes: int = 2,
        img_size: int = 64,
        embed_dim: int = 256,
        n_conv_layers: int = 4,
        num_layers: int = 7,
        num_heads: int = 4,
        mlp_ratio: float = 2.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.num_heads = num_heads

        self.tokenizer = ConvTokenizer(in_channels, embed_dim, n_conv_layers)

        # Number of tokens: run a dummy image through the tokenizer once.
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, img_size, img_size)
            num_tokens = self.tokenizer(dummy).shape[1]
        self.num_tokens = num_tokens
        self.pos_embed = nn.Parameter(torch.zeros(1, num_tokens, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.dropout = nn.Dropout(dropout)

        self.layers = nn.ModuleList(
            TransformerEncoderLayer(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(num_layers)
        )
        self.seq_pool = SeqPool(embed_dim)
        self.head = nn.Linear(embed_dim, n_classes)

        # Per-head attention weights from the most recent forward; consumed by
        # the attention-rollout attribution method. Reset every forward.
        self.attention_maps: list[torch.Tensor] = []

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        """Tokenize, run the transformer, and sequence-pool to ``[B, embed_dim]``.

        Records per-head attention into ``self.attention_maps`` (reset first).
        """
        self.attention_maps = []
        tokens = self.tokenizer(x) + self.pos_embed
        tokens = self.dropout(tokens)
        for layer in self.layers:
            tokens, attn = layer(tokens)
            self.attention_maps.append(attn)
        return self.seq_pool(tokens)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self._encode(x))

    def extract_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """Return the pre-head pooled feature vector ``[B, embed_dim]``.

        This is the sequence-pooled token representation, the same vector the
        classifier head consumes. Note this still populates
        ``self.attention_maps`` as a side effect.
        """
        return self._encode(x)

    def parameter_count(self) -> dict[str, int]:
        def count(m: nn.Module) -> int:
            return sum(p.numel() for p in m.parameters() if p.requires_grad)

        return {
            "tokenizer": count(self.tokenizer),
            "transformer": count(self.layers),
            "seq_pool": count(self.seq_pool),
            "head": count(self.head),
            "total": count(self),
        }
