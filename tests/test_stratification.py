"""Tests for cell-type stratification of attribution results."""

from __future__ import annotations

import pandas as pd
import torch

from cerberus_neuro.analysis.stratification import stratify_channel_scores_by_cell_type
from cerberus_neuro.attribution.base import AttributionResult


def test_stratify_groups_per_cell_type():
    # B=8 samples, C=6 channels, all samples in cell_type 0 have score 1.0 in
    # channel 0; samples in cell_type 1 have score 2.0 in channel 1.
    scores = torch.zeros(8, 6)
    scores[0:4, 0] = 1.0
    scores[4:8, 1] = 2.0
    cell_types = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
    cell_type_names = ["stem", "progen", "neuron", "astro"]

    result = AttributionResult(saliency=None, channel_scores=scores, metadata={"method": "test"})
    df = stratify_channel_scores_by_cell_type(
        result=result,
        cell_types=cell_types,
        cell_type_names=cell_type_names,
        channel_names=["BF", "DNA", "Mito", "AGP", "ER", "RNA"],
    )
    assert isinstance(df, pd.DataFrame)
    assert set(df.columns) == {"cell_type", "channel", "mean_score", "n_samples"}
    # cell_type=0 (stem) has 4 samples, channel BF mean=1.0.
    stem_bf = df[(df["cell_type"] == "stem") & (df["channel"] == "BF")].iloc[0]
    assert stem_bf["mean_score"] == 1.0
    assert stem_bf["n_samples"] == 4
    # cell_type=1 (progen) has 4 samples, channel DNA mean=2.0.
    progen_dna = df[(df["cell_type"] == "progen") & (df["channel"] == "DNA")].iloc[0]
    assert progen_dna["mean_score"] == 2.0
    assert progen_dna["n_samples"] == 4
    # Total rows: 4 cell types x 6 channels = 24. Cell types with no samples
    # still appear with n_samples=0.
    assert len(df) == 24


def test_stratify_ignores_unknown_cell_types():
    """A cell_type index outside the names list raises a clear error."""
    scores = torch.zeros(2, 6)
    cell_types = torch.tensor([0, 99])  # 99 is out-of-range
    result = AttributionResult(saliency=None, channel_scores=scores, metadata={})

    import pytest

    with pytest.raises(IndexError):
        stratify_channel_scores_by_cell_type(
            result=result,
            cell_types=cell_types,
            cell_type_names=["stem", "progen"],
            channel_names=["BF", "DNA", "Mito", "AGP", "ER", "RNA"],
        )
