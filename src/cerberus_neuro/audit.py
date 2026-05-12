"""Donor-structure audit utilities for Phase 0 of argus-cells.

Pure-pandas operations on the manifest DataFrame returned by
:func:`cerberus_neuro.data.build_manifest`. No S3, no torch, no PyTorch
dependencies — these utilities run on Colab Free.
"""
from __future__ import annotations

import math

import pandas as pd


def donor_counts_by_condition(manifest: pd.DataFrame) -> dict[str, int]:
    """Count unique donor lines (Metadata_line_ID) per Metadata_line_condition.

    Parameters
    ----------
    manifest
        DataFrame with at least ``Metadata_line_ID`` and
        ``Metadata_line_condition`` columns.

    Returns
    -------
    Dict mapping each condition value to the number of unique line_IDs
    observed under that condition. Empty manifest returns an empty dict.
    """
    if len(manifest) == 0:
        return {}
    return (
        manifest.groupby("Metadata_line_condition")["Metadata_line_ID"]
        .nunique()
        .to_dict()
    )


def donor_well_table(manifest: pd.DataFrame) -> pd.DataFrame:
    """Build a (cell_type, line_condition, line_ID) -> well-count table.

    One row per unique (cell_type, line_condition, line_ID) triple, with
    ``n_wells`` = count of distinct (Metadata_Plate, Metadata_Well) pairs
    falling under that triple.

    Returns
    -------
    DataFrame with columns: ``cell_type``, ``line_condition``, ``line_ID``,
    ``n_wells``. Sorted by (cell_type, line_condition, line_ID).
    """
    well_keys = (
        manifest[
            [
                "Metadata_cell_type",
                "Metadata_line_condition",
                "Metadata_line_ID",
                "Metadata_Plate",
                "Metadata_Well",
            ]
        ]
        .drop_duplicates()
    )
    counts = (
        well_keys.groupby(
            ["Metadata_cell_type", "Metadata_line_condition", "Metadata_line_ID"]
        )
        .size()
        .reset_index(name="n_wells")
    )
    counts = counts.rename(
        columns={
            "Metadata_cell_type": "cell_type",
            "Metadata_line_condition": "line_condition",
            "Metadata_line_ID": "line_ID",
        }
    )
    return counts.sort_values(["cell_type", "line_condition", "line_ID"]).reset_index(drop=True)


def imbalance_metric(table: pd.DataFrame) -> dict[tuple[str, str], dict[str, float]]:
    """Donor-balance coefficient of variation per (cell_type, line_condition).

    For each (cell_type, line_condition) group, computes the coefficient of
    variation (std/mean) of per-donor ``n_wells``. CV=0 means perfectly
    balanced donor representation; higher CV means one or two donors
    dominate the group.

    Single-donor groups return ``cv=NaN`` (CV is undefined with N=1).

    Parameters
    ----------
    table
        Output of :func:`donor_well_table`. Must have columns
        ``cell_type``, ``line_condition``, ``line_ID``, ``n_wells``.

    Returns
    -------
    Dict keyed by ``(cell_type, line_condition)`` tuples, with values
    ``{"cv": float, "n_donors": int}``.
    """
    out: dict[tuple[str, str], dict[str, float]] = {}
    for (cell_type, condition), group in table.groupby(["cell_type", "line_condition"]):
        wells = group["n_wells"].to_numpy()
        n_donors = int(len(wells))
        if n_donors <= 1:
            cv = math.nan
        else:
            mean = float(wells.mean())
            cv = float(wells.std() / mean) if mean > 0 else math.nan
        out[(cell_type, condition)] = {"cv": cv, "n_donors": n_donors}
    return out


def crop_budget_estimate(manifest: pd.DataFrame, crops_per_site: int) -> dict[str, int]:
    """Naive upper-bound estimate of total crops yielded by the pipeline.

    Real yield is typically 60-90% of this upper bound after the
    CellProfiler ``min_cells_per_crop`` filter is applied at dataset
    iteration time. Use this estimate as an *upper bound* for Phase 2
    crop-budget planning; multiply by the empirical yield ratio from
    Phase 0.5 / Phase 1 for a realistic number.

    Parameters
    ----------
    manifest
        DataFrame with columns ``Metadata_Plate``, ``Metadata_Well`` (one
        row per (plate, well, site) tuple).
    crops_per_site
        Configurable number of crops yielded per site by the
        CellProfiler-centroid tile selector.

    Returns
    -------
    Dict with ``n_sites``, ``n_wells``, ``crops_per_site``, and
    ``max_crops_upper_bound`` keys.
    """
    n_sites = int(len(manifest))
    n_wells = int(
        manifest.groupby(["Metadata_Plate", "Metadata_Well"]).ngroups
    )
    return {
        "n_sites": n_sites,
        "n_wells": n_wells,
        "crops_per_site": int(crops_per_site),
        "max_crops_upper_bound": n_sites * int(crops_per_site),
    }
