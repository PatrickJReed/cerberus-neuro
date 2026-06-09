"""Donor confound probe.

Fit a linear classifier on frozen encoder embeddings to predict donor identity
(``Metadata_line_ID``). High accuracy means the encoder has learned features
that linearly distinguish donor identity, which is a confound for any
disease-classification claim built on those embeddings.

The parallel report fits a second probe on the same embeddings against the
disease label, then reports the donor / disease ratio as a confound-strength
scalar. ratio ≪ 1 means the encoder retains less donor info than disease info
(good); ratio ≥ 1 means donor info is at least as linearly extractable as
disease info (red flag).
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression


def fit_linear_probe(
    train_emb: np.ndarray,
    train_labels: np.ndarray,
    val_emb: np.ndarray,
    val_labels: np.ndarray,
    n_classes: int,
    max_iter: int = 1000,
    C: float = 1.0,  # noqa: N803 — mirrors scikit-learn's LogisticRegression(C=...) parameter
) -> dict[str, float]:
    """Fit an L2-regularized multinomial logistic regression and report accuracy.

    Parameters
    ----------
    train_emb, train_labels
        Training embeddings ``[N_train, D]`` and integer class labels ``[N_train]``.
    val_emb, val_labels
        Validation embeddings ``[N_val, D]`` and labels ``[N_val]``.
    n_classes
        Number of distinct classes in the labels (used to compute the random
        baseline as ``1/n_classes``).
    max_iter
        Maximum optimizer iterations.
    C
        Inverse of regularization strength. Default 1.0 matches scikit-learn's
        default; useful to tune to balance under- vs over-fitting in small
        regimes.

    Returns
    -------
    Dict with ``train_accuracy``, ``val_accuracy``, ``random_baseline``,
    ``n_classes`` keys.
    """
    clf = LogisticRegression(
        max_iter=max_iter,
        C=C,
        solver="lbfgs",
    )
    clf.fit(train_emb, train_labels)
    train_acc = float(clf.score(train_emb, train_labels))
    val_acc = float(clf.score(val_emb, val_labels))
    return {
        "train_accuracy": train_acc,
        "val_accuracy": val_acc,
        "random_baseline": 1.0 / float(n_classes),
        "n_classes": int(n_classes),
    }


def parallel_probe_report(
    train_emb: np.ndarray,
    train_donor: np.ndarray,
    train_disease: np.ndarray,
    val_emb: np.ndarray,
    val_donor: np.ndarray,
    val_disease: np.ndarray,
    n_donors: int,
) -> dict[str, dict | float]:
    """Fit two probes (donor identity and disease) and report the ratio.

    Donor probe: ``n_donors``-way classification.
    Disease probe: binary classification.

    Returns
    -------
    Dict with keys ``donor`` (probe report), ``disease`` (probe report), and
    ``ratio`` (``donor.val_accuracy / disease.val_accuracy``). A ratio ≪ 1 is
    good (encoder retained little donor info); ratio ≥ 1 is a red flag.
    """
    donor_report = fit_linear_probe(
        train_emb=train_emb,
        train_labels=train_donor,
        val_emb=val_emb,
        val_labels=val_donor,
        n_classes=n_donors,
    )
    disease_report = fit_linear_probe(
        train_emb=train_emb,
        train_labels=train_disease,
        val_emb=val_emb,
        val_labels=val_disease,
        n_classes=2,
    )
    ratio = donor_report["val_accuracy"] / max(disease_report["val_accuracy"], 1e-6)
    return {
        "donor": donor_report,
        "disease": disease_report,
        "ratio": float(ratio),
    }
