"""Tests for the analysis-figure generators.

Verify each function returns a matplotlib Figure and produces a non-empty file
when saved. No pixel-level assertions — visual correctness is verified by eye
in the Phase 1 notebook.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # headless backend for tests
import pandas as pd
import pytest

from cerberus_neuro.analysis.figures import (
    plot_channel_ablation_heatmap,
    plot_probe_comparison,
)


@pytest.fixture
def ablation_df():
    """4 cell types x 6 channels long-form table."""
    rows = []
    for ct in ["stem", "progen", "neuron", "astro"]:
        for ch in ["BF", "DNA", "Mito", "AGP", "ER", "RNA"]:
            rows.append({"cell_type": ct, "channel": ch, "mean_score": 0.1, "n_samples": 100})
    return pd.DataFrame(rows)


def test_plot_channel_ablation_heatmap_saves_figure(ablation_df, tmp_path):
    fig = plot_channel_ablation_heatmap(
        df=ablation_df,
        cell_type_order=["stem", "progen", "neuron", "astro"],
        channel_order=["BF", "DNA", "Mito", "AGP", "ER", "RNA"],
        title="Test Heatmap",
    )
    out = tmp_path / "heatmap.png"
    fig.savefig(out, dpi=80)
    assert out.exists()
    assert out.stat().st_size > 1000  # non-trivial PNG


def test_plot_probe_comparison_saves_figure(tmp_path):
    probe_report = {
        "donor": {"val_accuracy": 0.30, "random_baseline": 0.021, "n_classes": 48},
        "disease": {"val_accuracy": 0.73, "random_baseline": 0.5, "n_classes": 2},
        "ratio": 0.41,
    }
    fig = plot_probe_comparison(report=probe_report, title="Probe Comparison")
    out = tmp_path / "probe.png"
    fig.savefig(out, dpi=80)
    assert out.exists()
    assert out.stat().st_size > 1000
