"""
Clinical Evidence Extraction package.

Converts refined movement segments and frame-level landmark features into
structured, objective numerical evidence for downstream VLM consumption.

This module does NOT diagnose. It only reports numerical evidence.
"""

from .config import EvidenceConfig
from .pipeline import extract_clinical_evidence

__all__ = ["EvidenceConfig", "extract_clinical_evidence"]
