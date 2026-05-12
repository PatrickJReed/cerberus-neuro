"""Confound-detection probes on frozen encoder embeddings."""
from .donor_probe import fit_linear_probe, parallel_probe_report

__all__ = ["fit_linear_probe", "parallel_probe_report"]
