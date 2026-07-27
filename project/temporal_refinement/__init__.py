"""
Temporal refinement package for facial action boundary detection.

Refines rough audio-based timestamps using geometric feature trajectories.
"""

from .pipeline import refine_all_tasks
from .config import RefinementConfig

__all__ = ["refine_all_tasks", "RefinementConfig"]
