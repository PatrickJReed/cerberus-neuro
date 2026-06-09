"""Interpretability attribution methods."""

from .attention_rollout import compute_attention_rollout
from .base import AttributionResult, channel_scores_from_saliency
from .channel_ablation import (
    compute_channel_ablation,
    compute_channel_ablation_per_sample,
)
from .gradcam import compute_gradcam
from .integrated_gradients import compute_integrated_gradients

__all__ = [
    "AttributionResult",
    "channel_scores_from_saliency",
    "compute_attention_rollout",
    "compute_channel_ablation",
    "compute_channel_ablation_per_sample",
    "compute_gradcam",
    "compute_integrated_gradients",
]
