"""Cross-architecture saliency agreement.

Measures how much two attribution maps agree on the same crops via per-sample
Spearman rank correlation. The intended use is quantifying whether two
attribution maps attend to the same pixels: either the *same* method run on two
architectures (e.g. Integrated Gradients on Argus-RN34 vs Argus-CCT) or the same
rollout method across architectures. This is the core finding-shaped output of
the argus-cells Phase 2 cross-architecture-agreement analysis.

Only **like-for-like** comparisons are meaningful: IG-vs-IG, rollout-vs-rollout.
Comparing *different* methods (e.g. GradCAM vs attention rollout) is not a valid
agreement measure even when the saliency shapes happen to match (the methods
attribute fundamentally different quantities); enforcing that is the caller's
responsibility, not something this function can check.
"""

from __future__ import annotations

import warnings

import torch
from scipy.stats import ConstantInputWarning, spearmanr

from argus_cells.attribution.base import AttributionResult


def saliency_agreement(result_a: AttributionResult, result_b: AttributionResult) -> dict:
    """Per-sample Spearman rank correlation between two saliency maps.

    Parameters
    ----------
    result_a, result_b
        Two :class:`AttributionResult` objects whose ``saliency`` tensors share
        the same shape ``[B, ...]`` (e.g. ``[B, 6, H, W]`` for Integrated
        Gradients, ``[B, 1, H, W]`` for attention rollout). For each sample the
        full per-pixel map is flattened to 1-D before correlating, so any
        trailing dimensions are allowed as long as the two shapes match.

    Returns
    -------
    dict with
    - ``"per_sample"``: ``torch.Tensor`` of shape ``[B]`` (float), the Spearman
      correlation for each sample. A sample whose saliency is constant in either
      map has zero rank variance, so Spearman is undefined and the entry is
      ``NaN``.
    - ``"mean"``: ``float``, the mean of the per-sample correlations computed
      with NaN-aware handling (``nanmean``). A single constant (NaN) sample does
      not poison the mean; it is dropped and the average is taken over the
      finite samples. If *every* sample is NaN the mean is ``NaN``.

    Raises
    ------
    ValueError
        If either ``saliency`` is ``None`` (e.g. channel ablation produces no
        per-pixel map, so it is not comparable this way), or if the two saliency
        tensors have different shapes (e.g. IG's ``[B, 6, H, W]`` vs rollout's
        ``[B, 1, H, W]`` are not comparable — that mismatch is the intended
        "not comparable" signal).
    """
    saliency_a = result_a.saliency
    saliency_b = result_b.saliency
    if saliency_a is None or saliency_b is None:
        raise ValueError(
            "both results must have non-None saliency; a None saliency (e.g. "
            "channel ablation) has no per-pixel map and is not comparable this way."
        )
    if saliency_a.shape != saliency_b.shape:
        raise ValueError(
            f"saliency shapes differ: {tuple(saliency_a.shape)} vs "
            f"{tuple(saliency_b.shape)}; differently-shaped maps (e.g. IG "
            "[B,6,H,W] vs rollout [B,1,H,W]) are not comparable."
        )

    batch = saliency_a.shape[0]
    flat_a = saliency_a.reshape(batch, -1).detach().cpu().numpy()
    flat_b = saliency_b.reshape(batch, -1).detach().cpu().numpy()

    per_sample = torch.empty(batch, dtype=torch.float)
    # spearmanr returns NaN when either input has zero variance (a constant map):
    # ranks are undefined. We keep that NaN per-sample and drop it from the mean
    # below; scipy's ConstantInputWarning for it is non-actionable, so silence it.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ConstantInputWarning)
        for b in range(batch):
            rho, _ = spearmanr(flat_a[b], flat_b[b])
            per_sample[b] = float(rho)

    mean = float(torch.nanmean(per_sample).item())
    return {"per_sample": per_sample, "mean": mean}
