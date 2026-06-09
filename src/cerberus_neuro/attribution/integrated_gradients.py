"""Integrated Gradients (Sundararajan et al., 2017).

IG_i(x) = (x_i - x'_i) * (1/N) * sum_{k=1..N} d f_target / d x_i |_{x' + (k/N)(x-x')}

where ``x'`` is the baseline (zero image by default). Approximates the path
integral of gradients from ``x'`` to ``x`` over ``n_steps`` linear interpolations.

Cross-architecture: works on any differentiable classifier. The argus-cells
methods spine uses this on both Argus-RN34 and (later) Argus-CCT for the
cross-architecture-agreement analysis.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .base import AttributionResult, channel_scores_from_saliency


def compute_integrated_gradients(
    model: nn.Module,
    images: torch.Tensor,
    target_class: int,
    n_steps: int = 32,
    baseline: torch.Tensor | None = None,
) -> AttributionResult:
    """Compute IG attribution for ``images`` against ``target_class``.

    Parameters
    ----------
    model
        Classifier producing ``[B, n_classes]`` logits.
    images
        ``[B, C, H, W]`` input tensor.
    target_class
        Integer class index to attribute.
    n_steps
        Number of Riemann steps along the baseline → input path. Default 32.
    baseline
        Reference input for the path integral. ``None`` (default) uses a
        zero tensor matching ``images.shape``.

    Returns
    -------
    :class:`AttributionResult` with
    - ``saliency`` of shape ``[B, C, H, W]`` (signed attribution per input
      element),
    - ``channel_scores`` of shape ``[B, C]`` (sum of |saliency| over H, W).
    """
    was_training = model.training
    model.eval()
    try:
        if baseline is None:
            baseline = torch.zeros_like(images)
        if baseline.shape != images.shape:
            raise ValueError(
                f"baseline shape {tuple(baseline.shape)} != images shape {tuple(images.shape)}"
            )
        alphas = torch.linspace(1.0 / n_steps, 1.0, n_steps, device=images.device)
        # Average gradient over the path.
        grad_accum = torch.zeros_like(images)
        for alpha in alphas:
            interp = baseline + alpha * (images - baseline)
            interp = interp.detach().clone().requires_grad_(True)
            logits = model(interp)
            score = logits[:, target_class].sum()
            model.zero_grad(set_to_none=True)
            score.backward()
            grad_accum = grad_accum + interp.grad.detach()
        avg_grad = grad_accum / n_steps
        saliency = (images - baseline) * avg_grad
        channel_scores = channel_scores_from_saliency(saliency.abs())
        return AttributionResult(
            saliency=saliency.detach(),
            channel_scores=channel_scores.detach(),
            metadata={
                "method": "integrated_gradients",
                "target_class": int(target_class),
                "n_steps": int(n_steps),
                "aggregation": "abs_spatial_sum",
            },
        )
    finally:
        model.train(was_training)
