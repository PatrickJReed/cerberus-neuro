"""Tests for cerberus_neuro.audit donor-structure utilities."""

from __future__ import annotations

import math

import pandas as pd
import pytest

from cerberus_neuro.audit import (
    crop_budget_estimate,
    donor_counts_by_condition,
    donor_well_table,
    imbalance_metric,
)


@pytest.fixture
def tiny_manifest() -> pd.DataFrame:
    """Synthetic manifest mirroring the real cpg0038 schema.

    3 donors per condition (D1-D3 control, D4-D6 deletion).
    4 cell types per donor (stem, progen, neuron, astro).
    2 wells per (donor, cell_type), 4 sites per well.
    Total rows: 6 donors x 4 cell types x 2 wells x 4 sites = 192.
    """
    rows = []
    donors = [
        ("D1", "control"),
        ("D2", "control"),
        ("D3", "control"),
        ("D4", "deletion"),
        ("D5", "deletion"),
        ("D6", "deletion"),
    ]
    for donor_id, condition in donors:
        for cell_type in ["stem", "progen", "neuron", "astro"]:
            for well_idx in range(2):
                well = f"A{well_idx + 1:02d}"
                for site in range(4):
                    rows.append(
                        {
                            "Metadata_Plate": f"plate_{donor_id}_{cell_type}",
                            "Metadata_Well": well,
                            "Metadata_Site": site + 1,
                            "Metadata_cell_type": cell_type,
                            "Metadata_line_ID": donor_id,
                            "Metadata_line_condition": condition,
                            "Metadata_line_source": "synthetic",
                            "batch": f"NCP_{cell_type.upper()}_1",
                        }
                    )
    return pd.DataFrame(rows)


def test_donor_counts_by_condition_returns_correct_counts(tiny_manifest):
    counts = donor_counts_by_condition(tiny_manifest)
    assert counts == {"control": 3, "deletion": 3}


def test_donor_counts_by_condition_empty_returns_empty():
    empty = pd.DataFrame(columns=["Metadata_line_ID", "Metadata_line_condition"])
    assert donor_counts_by_condition(empty) == {}


def test_donor_well_table_shape(tiny_manifest):
    table = donor_well_table(tiny_manifest)
    # 6 donors x 4 cell types per donor = 24 (donor, cell_type) groups.
    assert len(table) == 24
    assert set(table.columns) == {"cell_type", "line_condition", "line_ID", "n_wells"}


def test_donor_well_table_well_counts(tiny_manifest):
    table = donor_well_table(tiny_manifest)
    # Each (donor, cell_type) combo has 2 wells in the fixture.
    assert (table["n_wells"] == 2).all()


def test_donor_well_table_donor_coverage(tiny_manifest):
    table = donor_well_table(tiny_manifest)
    # Every donor appears in every cell type.
    assert set(table["line_ID"]) == {"D1", "D2", "D3", "D4", "D5", "D6"}
    assert set(table["cell_type"]) == {"stem", "progen", "neuron", "astro"}


def test_imbalance_metric_perfect_balance(tiny_manifest):
    """Equal wells per donor → CV = 0 for every (cell_type, line_condition)."""
    table = donor_well_table(tiny_manifest)
    imbalance = imbalance_metric(table)
    # 4 cell types x 2 conditions = 8 groups.
    assert len(imbalance) == 8
    for _key, val in imbalance.items():
        assert val["cv"] == 0
        assert val["n_donors"] == 3


def test_imbalance_metric_imbalanced_high_cv():
    """One donor has 10 wells, others have 1 — CV should be > 1.0."""
    table = pd.DataFrame(
        [
            {"cell_type": "stem", "line_condition": "control", "line_ID": "D1", "n_wells": 10},
            {"cell_type": "stem", "line_condition": "control", "line_ID": "D2", "n_wells": 1},
            {"cell_type": "stem", "line_condition": "control", "line_ID": "D3", "n_wells": 1},
        ]
    )
    imbalance = imbalance_metric(table)
    assert imbalance[("stem", "control")]["cv"] > 1.0
    assert imbalance[("stem", "control")]["n_donors"] == 3


def test_imbalance_metric_single_donor_returns_nan():
    """Single donor in a group: CV is undefined → NaN."""
    table = pd.DataFrame(
        [{"cell_type": "stem", "line_condition": "control", "line_ID": "D1", "n_wells": 5}]
    )
    imbalance = imbalance_metric(table)
    assert math.isnan(imbalance[("stem", "control")]["cv"])
    assert imbalance[("stem", "control")]["n_donors"] == 1


def test_crop_budget_estimate_returns_expected_shape(tiny_manifest):
    budget = crop_budget_estimate(tiny_manifest, crops_per_site=10)
    # 6 donors x 4 cell types x 2 wells x 4 sites = 192 rows
    # 6 donors x 4 cell types x 2 wells = 48 wells.
    assert budget["n_sites"] == 192
    assert budget["n_wells"] == 48
    assert budget["crops_per_site"] == 10
    assert budget["max_crops_upper_bound"] == 1920


def test_crop_budget_estimate_zero_crops_per_site(tiny_manifest):
    budget = crop_budget_estimate(tiny_manifest, crops_per_site=0)
    assert budget["max_crops_upper_bound"] == 0
