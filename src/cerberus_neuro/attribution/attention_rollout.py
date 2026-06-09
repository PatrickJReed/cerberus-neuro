"""Attention rollout (Abnar & Zuidema, 2020).

Quantifies how information flows from input tokens to the output of a
transformer by recursively multiplying the per-layer attention matrices. For
each layer the per-head weights are averaged, residual connections are modelled
by mixing in the identity (``A_hat = 0.5 * A + 0.5 * I``) and re-normalising
rows to sum to 1, then the layers are composed by matrix multiplication
(``R = A_hat[L-1] @ ... @ A_hat[0]``). The mean over the query dimension of the
rolled-out matrix gives, per token, the average attention that token receives;
that vector is reshaped to the token grid and upsampled to the input resolution.

Class-agnostic. Basic rollout uses only the attention weights and does not
condition on a target logit, so ``target_class`` is accepted purely for
interface symmetry with the gradient-based methods (Integrated Gradients,
GradCAM) and is recorded in ``metadata`` but does not affect the result.

Channel-agnostic. Rollout produces a single spatial map per sample; it does not
distinguish the input channels (unlike channel ablation or Integrated
Gradients, which attribute per channel). To keep the :class:`AttributionResult`
shape uniform we broadcast the per-sample spatial sum across the input-channel
axis, matching the convention used by GradCAM.

Consumes the per-head attention captured by :class:`~cerberus_neuro.models.cct.ArgusCCT`
into ``model.attention_maps`` (a list of length ``num_layers``, each
``[B, num_heads, T, T]``). The function runs its own ``no_grad`` forward and
reads ``attention_maps`` immediately after, because other entry points (notably
``extract_embedding``) also repopulate that attribute as a side effect.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812 — conventional PyTorch alias

from .base import AttributionResult


def compute_attention_rollout(
    model: nn.Module,
    images: torch.Tensor,
    target_class: int = 1,
) -> AttributionResult:
    """Compute attention-rollout saliency for ``images``.

    Parameters
    ----------
    model
        A transformer that records per-head attention into
        ``model.attention_maps`` on every forward (see
        :class:`~cerberus_neuro.models.cct.ArgusCCT`). Must be a list of length
        ``num_layers``, each entry ``[B, num_heads, T, T]``.
    images
        ``[B, C, H, W]`` input tensor.
    target_class
        Accepted for interface symmetry with the gradient-based methods.
        **Ignored** by basic rollout, which is class-agnostic. Recorded in
        ``metadata["target_class"]``.

    Returns
    -------
    :class:`AttributionResult` with
    - ``saliency`` of shape ``[B, 1, H, W]``: a single channel-agnostic spatial
      map per sample, min-max normalised per sample to ``[0, 1]`` at the token
      grid then bilinearly upsampled to the input resolution.
    - ``channel_scores`` of shape ``[B, C]``: the per-sample spatial sum of the
      saliency, broadcast across every input channel (rollout does not
      distinguish input channels).
    """
    was_training = model.training
    model.eval()
    try:
        # Own forward so we read fresh attention, not a map left by some other
        # call (e.g. extract_embedding repopulates attention_maps as well).
        with torch.no_grad():
            model(images)
            maps = model.attention_maps

        if not maps:
            raise ValueError(
                "model.attention_maps is empty after forward; the model must "
                "record per-head attention (e.g. ArgusCCT)."
            )

        device = maps[0].device
        n_tokens = maps[0].shape[-1]  # T  # noqa: N806 — T is the token count
        eye = torch.eye(n_tokens, device=device)  # I  # noqa: N806 — identity matrix

        rollout = None  # R, batched [B, T, T]  # noqa: N806 — R per the paper
        for layer_maps in maps:
            attn = layer_maps.mean(dim=1)  # A: average over heads → [B, T, T]
            # Residual mixing, then row-normalise so each row sums to 1.
            a_hat = 0.5 * attn + 0.5 * eye
            a_hat = a_hat / a_hat.sum(dim=-1, keepdim=True)
            rollout = a_hat if rollout is None else torch.bmm(a_hat, rollout)

        # Average attention each token *receives* → mean over the query dim.
        token_importance = rollout.mean(dim=1)  # [B, T]

        # Per-sample min-max normalise to [0, 1]; guard the constant case.
        lo = token_importance.amin(dim=1, keepdim=True)
        hi = token_importance.amax(dim=1, keepdim=True)
        token_importance = (token_importance - lo) / (hi - lo).clamp_min(1e-12)

        # Reshape [B, T] → token grid [B, 1, g, g]; crops are square so g*g == T.
        grid = int(round(n_tokens**0.5))
        if grid * grid != n_tokens:
            raise ValueError(
                f"token count {n_tokens} is not a perfect square; rollout assumes "
                "a square token grid (square input crops)."
            )
        batch = token_importance.shape[0]
        token_map = token_importance.view(batch, 1, grid, grid)

        # Upsample to the input resolution.
        saliency = F.interpolate(
            token_map, size=images.shape[-2:], mode="bilinear", align_corners=False
        )

        # Channel-agnostic: broadcast the per-sample spatial sum across channels
        # to keep the AttributionResult shape uniform with IG / channel ablation.
        per_sample_sum = saliency.sum(dim=(2, 3)).squeeze(-1)  # [B]
        n_channels = images.shape[1]
        channel_scores = per_sample_sum.unsqueeze(-1).expand(-1, n_channels).clone()

        return AttributionResult(
            saliency=saliency.detach(),
            channel_scores=channel_scores.detach(),
            metadata={
                "method": "attention_rollout",
                "target_class": int(target_class),
                "num_layers": len(maps),
                "class_agnostic": True,
                "aggregation": "spatial_sum_broadcast",
            },
        )
    finally:
        model.train(was_training)
