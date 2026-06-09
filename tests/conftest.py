"""Shared pytest fixtures for the interpretability harness tests."""

from __future__ import annotations

import pytest
import torch


@pytest.fixture
def tiny_batch_6ch() -> dict[str, torch.Tensor]:
    """Synthetic batch matching the 6-channel disease-classifier input shape.

    Returns a dict with keys: ``images`` ([B=4, C=6, H=64, W=64]),
    ``labels`` ([B] binary disease labels in {0, 1}),
    ``cell_type`` ([B] integer cell-type labels in {0, 1, 2, 3}),
    ``line_ID`` ([B] integer donor IDs in {1, 2, 3, 4}).

    Deterministic via fixed seed.
    """
    g = torch.Generator().manual_seed(0)
    return {
        "images": torch.randn(4, 6, 64, 64, generator=g),
        "labels": torch.tensor([0, 1, 0, 1]),
        "cell_type": torch.tensor([0, 1, 2, 3]),
        "line_ID": torch.tensor([1, 2, 3, 4]),
    }


@pytest.fixture
def tiny_model_6ch():
    """Tiny 6-channel binary classifier mirroring BaselineDiseaseClassifier
    structure (encoder with a `layer4` attribute, head producing 2-logit output)
    but small enough to run on CPU in tests.

    Exposes the same attributes the attribution methods rely on:
    - ``encoder.layer4``: last conv stage (the GradCAM target)
    - ``encoder``: callable mapping [B, 6, H, W] -> 5-tuple feature stack
    - ``head``: classifier on the 4th feature stage
    """
    import torch.nn as nn

    class TinyEncoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer1 = nn.Sequential(nn.Conv2d(6, 8, 3, padding=1), nn.ReLU())
            self.layer2 = nn.Sequential(nn.Conv2d(8, 16, 3, stride=2, padding=1), nn.ReLU())
            self.layer3 = nn.Sequential(nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU())
            self.layer4 = nn.Sequential(nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU())

        def forward(self, x):
            x1 = self.layer1(x)
            x2 = self.layer2(x1)
            x3 = self.layer3(x2)
            x4 = self.layer4(x3)
            return x1, x2, x3, x4, x4

    class TinyHead(nn.Module):
        def __init__(self):
            super().__init__()
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.fc = nn.Linear(64, 2)

        def forward(self, features):
            return self.fc(self.pool(features).flatten(1))

    class TinyClassifier(nn.Module):
        model_kind = "baseline"

        def __init__(self):
            super().__init__()
            self.encoder = TinyEncoder()
            self.head = TinyHead()

        def forward(self, x):
            *_, x4 = self.encoder(x)
            return self.head(x4)

        def extract_embedding(self, x):
            *_, x4 = self.encoder(x)
            return self.head.pool(x4).flatten(1)

    m = TinyClassifier()
    m.eval()
    return m


@pytest.fixture
def synthetic_embeddings():
    """Synthetic 512-dim embeddings + donor + disease labels for the donor probe.

    Designed so donor identity is strongly linearly separable (each donor's
    embeddings cluster around a unique vector) but disease is not.

    Returns dict with keys: ``train_emb`` [120, 512], ``train_donor`` [120],
    ``train_disease`` [120], plus matching ``val_*`` arrays of shape [40].
    """
    import numpy as np

    rng = np.random.default_rng(0)
    n_donors = 4
    train_per_donor, val_per_donor = 30, 10
    centers = rng.normal(size=(n_donors, 512)).astype("float32") * 5.0

    def gen(n_per):
        emb = np.empty((n_donors * n_per, 512), dtype="float32")
        donor = np.empty(n_donors * n_per, dtype="int64")
        disease = np.empty(n_donors * n_per, dtype="int64")
        for d in range(n_donors):
            sl = slice(d * n_per, (d + 1) * n_per)
            emb[sl] = centers[d] + rng.normal(size=(n_per, 512)).astype("float32")
            donor[sl] = d
            disease[sl] = rng.integers(0, 2, size=n_per)
        return emb, donor, disease

    train_emb, train_donor, train_disease = gen(train_per_donor)
    val_emb, val_donor, val_disease = gen(val_per_donor)
    return {
        "train_emb": train_emb,
        "train_donor": train_donor,
        "train_disease": train_disease,
        "val_emb": val_emb,
        "val_donor": val_donor,
        "val_disease": val_disease,
    }
