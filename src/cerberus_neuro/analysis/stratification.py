"""Cell-type stratification of attribution results.

Groups per-sample per-channel scores by cell type and reports the
per-(cell_type, channel) mean. This is the core finding-shaped output of
the argus-cells Phase 1 harness: a 4x6 table answering "for each cell type,
which of the 6 input channels does the model rely on most?".
"""

from __future__ import annotations

import pandas as pd
import torch

from cerberus_neuro.attribution.base import AttributionResult


def stratify_channel_scores_by_cell_type(
    result: AttributionResult,
    cell_types: torch.Tensor,
    cell_type_names: list[str],
    channel_names: list[str],
) -> pd.DataFrame:
    """Group channel scores by cell type and report per-group means.

    Parameters
    ----------
    result
        An :class:`AttributionResult` with ``channel_scores`` of shape
        ``[B, C]`` (per-sample, per-channel scores).
    cell_types
        Integer cell-type labels of shape ``[B]``. Each value indexes into
        ``cell_type_names``.
    cell_type_names
        Human-readable cell-type names. ``cell_types[i]`` must be a valid
        index into this list.
    channel_names
        Human-readable channel names. Length must match
        ``result.channel_scores.shape[1]``.

    Returns
    -------
    Long-format DataFrame with columns ``cell_type``, ``channel``,
    ``mean_score``, ``n_samples``. One row per (cell_type, channel) pair, for
    a total of ``len(cell_type_names) * len(channel_names)`` rows. Cell types
    with no samples are emitted with ``mean_score=0`` and ``n_samples=0``.
    """
    if result.channel_scores.ndim != 2:
        raise ValueError(
            f"expected channel_scores shape [B, C], got {tuple(result.channel_scores.shape)}"
        )
    n_channels = result.channel_scores.shape[1]
    if len(channel_names) != n_channels:
        raise ValueError(f"channel_names length {len(channel_names)} != C={n_channels}")
    # Validate that all cell-type indices are in range; raise IndexError
    # otherwise (matches list indexing semantics).
    cell_types_list = cell_types.tolist()
    for idx in cell_types_list:
        _ = cell_type_names[idx]

    rows = []
    for ct_idx, ct_name in enumerate(cell_type_names):
        mask = cell_types == ct_idx
        n = int(mask.sum().item())
        for ch_idx, ch_name in enumerate(channel_names):
            if n > 0:
                mean = float(result.channel_scores[mask, ch_idx].mean().item())
            else:
                mean = 0.0
            rows.append(
                {
                    "cell_type": ct_name,
                    "channel": ch_name,
                    "mean_score": mean,
                    "n_samples": n,
                }
            )
    return pd.DataFrame(rows)
