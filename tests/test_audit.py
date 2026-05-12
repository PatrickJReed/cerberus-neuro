"""Tests for cerberus_neuro.audit donor-structure utilities."""
from __future__ import annotations

import pandas as pd
import pytest

from cerberus_neuro.audit import donor_counts_by_condition


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
        ("D1", "control"), ("D2", "control"), ("D3", "control"),
        ("D4", "deletion"), ("D5", "deletion"), ("D6", "deletion"),
    ]
    for donor_id, condition in donors:
        for cell_type in ["stem", "progen", "neuron", "astro"]:
            for well_idx in range(2):
                well = f"A{well_idx + 1:02d}"
                for site in range(4):
                    rows.append({
                        "Metadata_Plate": f"plate_{donor_id}_{cell_type}",
                        "Metadata_Well": well,
                        "Metadata_Site": site + 1,
                        "Metadata_cell_type": cell_type,
                        "Metadata_line_ID": donor_id,
                        "Metadata_line_condition": condition,
                        "Metadata_line_source": "synthetic",
                        "batch": f"NCP_{cell_type.upper()}_1",
                    })
    return pd.DataFrame(rows)


def test_donor_counts_by_condition_returns_correct_counts(tiny_manifest):
    counts = donor_counts_by_condition(tiny_manifest)
    assert counts == {"control": 3, "deletion": 3}


def test_donor_counts_by_condition_empty_returns_empty():
    empty = pd.DataFrame(columns=["Metadata_line_ID", "Metadata_line_condition"])
    assert donor_counts_by_condition(empty) == {}


from cerberus_neuro.audit import donor_well_table


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
