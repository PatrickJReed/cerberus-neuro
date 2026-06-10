"""Model zoo for cerberus-neuro / argus-cells.

Re-exports the disease-classifier models so callers can import them from one
place. :class:`BaselineDiseaseClassifier` (ResNet34, 6-channel) lives in
``argus_cells.model`` and is re-exported here for symmetry with the
transformer-family :class:`ArgusCCT`.
"""

from __future__ import annotations

from argus_cells.model import BaselineDiseaseClassifier

from .cct import ArgusCCT

__all__ = ["ArgusCCT", "BaselineDiseaseClassifier"]
