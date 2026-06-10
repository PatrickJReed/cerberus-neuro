"""Tests for the donor probe."""

from __future__ import annotations

from argus_cells.probes.donor_probe import (
    fit_linear_probe,
    parallel_probe_report,
)


def test_fit_linear_probe_recovers_separable_donor(synthetic_embeddings):
    """Synthetic donors are linearly separable; probe accuracy should be > 0.8."""
    report = fit_linear_probe(
        train_emb=synthetic_embeddings["train_emb"],
        train_labels=synthetic_embeddings["train_donor"],
        val_emb=synthetic_embeddings["val_emb"],
        val_labels=synthetic_embeddings["val_donor"],
        n_classes=4,
    )
    assert report["val_accuracy"] > 0.8
    assert report["random_baseline"] == 0.25  # 1/4 for 4 donors


def test_fit_linear_probe_on_random_labels_returns_near_baseline(synthetic_embeddings):
    """Probing for random disease labels (no structure) → near-baseline accuracy."""
    import numpy as np

    rng = np.random.default_rng(0)
    n_train = synthetic_embeddings["train_emb"].shape[0]
    n_val = synthetic_embeddings["val_emb"].shape[0]
    random_train = rng.integers(0, 2, size=n_train)
    random_val = rng.integers(0, 2, size=n_val)
    report = fit_linear_probe(
        train_emb=synthetic_embeddings["train_emb"],
        train_labels=random_train,
        val_emb=synthetic_embeddings["val_emb"],
        val_labels=random_val,
        n_classes=2,
    )
    # No more than 20pp above random for noise-only labels.
    assert report["val_accuracy"] < report["random_baseline"] + 0.2


def test_parallel_probe_report_includes_ratio(synthetic_embeddings):
    """The parallel report runs two probes on the same embeddings and returns
    donor / disease accuracy as the confound-strength scalar."""
    report = parallel_probe_report(
        train_emb=synthetic_embeddings["train_emb"],
        train_donor=synthetic_embeddings["train_donor"],
        train_disease=synthetic_embeddings["train_disease"],
        val_emb=synthetic_embeddings["val_emb"],
        val_donor=synthetic_embeddings["val_donor"],
        val_disease=synthetic_embeddings["val_disease"],
        n_donors=4,
    )
    assert "donor" in report and "disease" in report
    assert "ratio" in report
    # In the synthetic data, donor is linearly separable, disease is not, so
    # the ratio (donor / disease) should be > 1.
    assert report["ratio"] > 1.0
    assert report["donor"]["random_baseline"] == 0.25
    assert report["disease"]["random_baseline"] == 0.5
