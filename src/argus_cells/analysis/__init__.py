"""Analysis utilities: agreement, stratification, figures."""

from .agreement import saliency_agreement
from .figures import plot_channel_ablation_heatmap, plot_probe_comparison
from .stratification import stratify_channel_scores_by_cell_type

__all__ = [
    "plot_channel_ablation_heatmap",
    "plot_probe_comparison",
    "saliency_agreement",
    "stratify_channel_scores_by_cell_type",
]
