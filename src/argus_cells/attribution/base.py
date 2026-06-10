"""Common interface for interpretability attribution methods.

Every method in :mod:`argus_cells.attribution` returns the same
:class:`AttributionResult` dataclass so downstream analysis treats them
uniformly.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch


@dataclass
class AttributionResult:
    """Uniform return type for all attribution methods.

    Attributes
    ----------
    saliency
        Per-pixel attribution map. Shape ``[B, C, H, W]`` where ``B`` is the
        batch size, ``C`` matches the input-channel count (typically 6 here:
        brightfield + 5 fluorescence). ``None`` for methods that do not produce
        per-pixel maps (e.g. channel ablation).
    channel_scores
        Per-channel importance scores. Shape ``[B, C]`` for per-sample scores,
        or ``[C]`` for aggregated-batch scores. Sign conventions vary by
        method; consult ``metadata["aggregation"]``.
    metadata
        Free-form dict describing the method, its hyperparameters, and any
        runtime info worth carrying through the analysis pipeline.
    """

    saliency: torch.Tensor | None
    channel_scores: torch.Tensor
    metadata: dict = field(default_factory=dict)


def channel_scores_from_saliency(saliency: torch.Tensor) -> torch.Tensor:
    """Aggregate per-pixel saliency to per-channel scores by summing over H, W.

    Parameters
    ----------
    saliency
        Tensor of shape ``[B, C, H, W]``. Caller controls sign (pass
        ``saliency.abs()`` if absolute attribution is desired).

    Returns
    -------
    Tensor of shape ``[B, C]``.
    """
    if saliency.ndim != 4:
        raise ValueError(f"expected [B,C,H,W], got shape {tuple(saliency.shape)}")
    return saliency.sum(dim=(2, 3))
