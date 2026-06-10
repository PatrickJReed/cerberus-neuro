"""Tests for NeuroPaintingDataset donor-ID routing (yield_donor flag).

The dataset streams from S3, so we monkeypatch the three module-level I/O
helpers (`_s3_client`, `_load_image`, `load_cell_centroids`) and the tile
selector (`tile_top_cells`) to run __iter__ end-to-end without network access.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import argus_cells.data as data_mod
from argus_cells.data import CHANNEL_ORDER, NeuroPaintingDataset


def _fake_manifest() -> pd.DataFrame:
    """Two sites with distinct donors, cell types, and conditions (shuffle=False
    preserves this order)."""
    rows = []
    specs = [(7, "stem", "control"), (33, "astro", "deletion")]
    for i, (donor, ct, cond) in enumerate(specs):
        row = {
            "batch": "B",
            "Metadata_Plate": "P",
            "Metadata_Well": f"W{i}",
            "Metadata_Site": i,
            "Metadata_cell_type": ct,
            "Metadata_line_condition": cond,
            "Metadata_line_ID": donor,
        }
        for c in CHANNEL_ORDER:
            row[f"URL_{c}"] = f"s3://bucket/{c}_{i}.tiff"
        rows.append(row)
    return pd.DataFrame(rows)


@pytest.fixture
def patched_io(monkeypatch):
    """Replace the S3 / image / centroid / tiling helpers with synthetic ones so
    __iter__ yields one deterministic 8x8 crop per site."""
    monkeypatch.setattr(data_mod, "_s3_client", lambda: None)
    monkeypatch.setattr(
        data_mod, "_load_image", lambda s3, url, cache: np.full((8, 8), 30000, dtype=np.uint16)
    )
    monkeypatch.setattr(data_mod, "load_cell_centroids", lambda *a, **k: np.array([[4.0, 4.0]]))
    monkeypatch.setattr(data_mod, "tile_top_cells", lambda *a, **k: [(0, 0, 5)])


def _dataset(**overrides):
    kwargs = dict(
        manifest=_fake_manifest(),
        cache_dir="/tmp/does-not-matter",
        crop_size=8,
        crops_per_site=1,
        min_cells_per_crop=1,
        shuffle=False,
        augment=False,
    )
    kwargs.update(overrides)
    return NeuroPaintingDataset(**kwargs)


def test_default_yields_four_tuple(patched_io):
    """Backward compatibility: without yield_donor the dataset still yields
    (brightfield, fluorescence, cell_type, line_condition)."""
    items = list(_dataset())
    assert len(items) == 2
    assert all(len(sample) == 4 for sample in items)


def test_yield_donor_appends_correct_donor(patched_io):
    """With yield_donor=True the dataset yields a 5-tuple whose final element is
    the crop's true Metadata_line_ID, paired to the right site."""
    items = list(_dataset(yield_donor=True))
    assert len(items) == 2
    assert all(len(sample) == 5 for sample in items)

    bf, fluo, ct, cond, donor = items[0]
    assert bf.shape == (1, 8, 8)
    assert fluo.shape == (5, 8, 8)
    assert int(donor) == 7  # first manifest row's Metadata_line_ID

    assert [int(s[4]) for s in items] == [7, 33]  # donor follows manifest order
