"""
Facial palsy diagnosis package.

Combines VLM-extracted visual evidence with landmark-derived clinical evidence
to produce a House-Brackmann (HB) grade using a GPT reasoning model.
"""

from .config import DiagnosisConfig
from .pipeline import diagnose_facial_palsy

__all__ = ["DiagnosisConfig", "diagnose_facial_palsy"]
