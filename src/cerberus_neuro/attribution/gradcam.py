"""GradCAM attribution for CNN classifiers.

Hooks a target conv layer for both the forward activation and the backward
gradient. GradCAM = ReLU(sum_c (avg_pool(grad_c) * act_c)) — a class-conditional
saliency map at the spatial resolution of the target layer, upsampled bilinearly
to the input resolution.

For the argus-cells `BaselineDiseaseClassifier`, the target layer is
``encoder.layer4`` (the deepest conv stage, where class-discriminative spatial
features live).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812 — conventional PyTorch alias

from .base import AttributionResult


def compute_gradcam(
    model: nn.Module,
    target_layer: nn.Module,
    images: torch.Tensor,
    target_class: int,
) -> AttributionResult:
    """Compute GradCAM saliency for ``images`` against ``target_class``.

    Parameters
    ----------
    model
        Classifier producing ``[B, n_classes]`` logits.
    target_layer
        The conv-style module whose activations + gradients drive the map.
        For BaselineDiseaseClassifier: ``model.encoder.layer4``.
    images
        ``[B, C_in, H, W]`` input tensor.
    target_class
        Integer class index to compute saliency for.

    Returns
    -------
    :class:`AttributionResult` with
    - ``saliency`` of shape ``[B, 1, H, W]`` (single channel-agnostic spatial
      map per sample, upsampled bilinearly to the input resolution),
    - ``channel_scores`` of shape ``[B, C_in]`` where each row is the per-sample
      saliency sum broadcast across input channels (GradCAM does not
      distinguish input channels).
    """
    was_training = model.training
    model.eval()

    activations: dict[str, torch.Tensor] = {}
    gradients: dict[str, torch.Tensor] = {}

    def fwd_hook(_module, _inputs, output):
        activations["act"] = output

    def bwd_hook(_module, _grad_input, grad_output):
        gradients["grad"] = grad_output[0]

    fh = target_layer.register_forward_hook(fwd_hook)
    bh = target_layer.register_full_backward_hook(bwd_hook)

    try:
        images = images.detach().clone().requires_grad_(False)
        logits = model(images)
        if logits.ndim != 2:
            raise ValueError(f"expected 2D logits, got shape {tuple(logits.shape)}")
        score = logits[:, target_class].sum()
        model.zero_grad(set_to_none=True)
        score.backward()

        act = activations["act"]  # [B, K, h, w]
        grad = gradients["grad"]  # [B, K, h, w]
        weights = grad.mean(dim=(2, 3), keepdim=True)  # [B, K, 1, 1]
        cam = (weights * act).sum(dim=1, keepdim=True)  # [B, 1, h, w]
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=images.shape[-2:], mode="bilinear", align_corners=False)

        # Channel-agnostic spatial map; broadcast a per-sample scalar across
        # the 6 input channels to keep the AttributionResult shape uniform.
        per_sample_sum = cam.sum(dim=(2, 3)).squeeze(-1)  # [B]
        n_channels = images.shape[1]
        channel_scores = per_sample_sum.unsqueeze(-1).expand(-1, n_channels).detach().clone()

        return AttributionResult(
            saliency=cam.detach(),
            channel_scores=channel_scores,
            metadata={
                "method": "gradcam",
                "target_class": int(target_class),
                "target_layer": type(target_layer).__name__,
                "aggregation": "spatial_sum_broadcast",
            },
        )
    finally:
        fh.remove()
        bh.remove()
        model.train(was_training)
