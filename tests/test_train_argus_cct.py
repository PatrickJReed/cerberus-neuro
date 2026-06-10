"""train() must accept ArgusCCT via the single-head disease path.

ArgusCCT is a single-head 6-channel disease classifier (model_kind="argus_cct"),
the same training/eval shape as BaselineDiseaseClassifier. This checks the
dispatch routes it through the baseline step (CE on disease logits over the
concatenated 6-channel input) and runs a step end-to-end on CPU.
"""

from __future__ import annotations

import torch

from argus_cells.models.cct import ArgusCCT
from argus_cells.training import TrainConfig, train


class _TinyLoader:
    """Re-iterable loader yielding (bf[B,1,H,W], fluo[B,5,H,W], ct[B], cond[B])."""

    def __iter__(self):
        for _ in range(2):
            yield (
                torch.randn(2, 1, 32, 32),
                torch.randn(2, 5, 32, 32),
                torch.zeros(2, dtype=torch.long),
                torch.randint(0, 2, (2,)),
            )


def test_train_accepts_argus_cct(tmp_path):
    model = ArgusCCT(
        in_channels=6, n_classes=2, img_size=32, embed_dim=64, num_layers=2, num_heads=2
    )
    cfg = TrainConfig(
        n_epochs=1,
        steps_per_epoch=1,
        amp=False,
        warmup_steps=0,
        ckpt_every_steps=999,
        log_every_steps=999,
    )
    summary = train(
        model, _TinyLoader(), cfg, checkpoint_dir=tmp_path, val_loader=None, device="cpu"
    )
    assert isinstance(summary, dict)
