"""Production figures for the Phase 1 interpretability harness output."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_channel_ablation_heatmap(
    df: pd.DataFrame,
    cell_type_order: list[str],
    channel_order: list[str],
    title: str = "Channel-ablation accuracy drop per (cell type, channel)",
) -> plt.Figure:
    """Heatmap of cell_type x channel mean attribution scores.

    Parameters
    ----------
    df
        Long-form DataFrame with at least ``cell_type``, ``channel``, and
        ``mean_score`` columns (output of
        :func:`stratify_channel_scores_by_cell_type`).
    cell_type_order, channel_order
        Lists giving the axis order. Rows (y-axis) = cell_type_order,
        columns (x-axis) = channel_order.
    title
        Figure title.

    Returns
    -------
    The matplotlib Figure.
    """
    pivot = df.pivot_table(
        index="cell_type", columns="channel", values="mean_score", fill_value=0.0
    )
    pivot = pivot.reindex(index=cell_type_order, columns=channel_order)
    fig, ax = plt.subplots(figsize=(8, 4))
    im = ax.imshow(
        pivot.values,
        aspect="auto",
        cmap="RdBu_r",
        vmin=-pivot.abs().max().max(),
        vmax=pivot.abs().max().max(),
    )
    ax.set_xticks(range(len(channel_order)))
    ax.set_xticklabels(channel_order, rotation=0)
    ax.set_yticks(range(len(cell_type_order)))
    ax.set_yticklabels(cell_type_order)
    for i in range(len(cell_type_order)):
        for j in range(len(channel_order)):
            ax.text(
                j,
                i,
                f"{pivot.values[i, j]:.3f}",
                ha="center",
                va="center",
                fontsize=9,
                color="black",
            )
    ax.set_xlabel("Channel")
    ax.set_ylabel("Cell type")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label="mean score")
    fig.tight_layout()
    return fig


def plot_probe_comparison(
    report: dict,
    title: str = "Donor probe vs disease probe accuracy",
) -> plt.Figure:
    """Bar chart of donor-probe vs disease-probe accuracy on shared embeddings.

    Adds dotted horizontal lines at each probe's random-baseline accuracy and
    annotates the ratio.
    """
    donor = report["donor"]
    disease = report["disease"]
    ratio = report["ratio"]

    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.array([0, 1])
    accuracies = [donor["val_accuracy"], disease["val_accuracy"]]
    baselines = [donor["random_baseline"], disease["random_baseline"]]
    bars = ax.bar(x, accuracies, color=["#d62728", "#1f77b4"], width=0.6)
    for xi, b in zip(x, baselines, strict=True):
        ax.hlines(
            b, xmin=xi - 0.3, xmax=xi + 0.3, colors="black", linestyles="dotted", linewidth=1.5
        )
    ax.set_xticks(x)
    ax.set_xticklabels([f"donor (N={donor['n_classes']})", "disease (N=2)"])
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Validation accuracy")
    ax.set_title(f"{title}\nratio = {ratio:.3f} (donor / disease)")
    for bar, val in zip(bars, accuracies, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2, val + 0.02, f"{val:.3f}", ha="center", fontsize=10
        )
    fig.tight_layout()
    return fig
