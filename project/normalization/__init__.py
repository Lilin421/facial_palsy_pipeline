"""
Facial landmark normalization package.

Provides spatial normalization, temporal smoothing (One Euro Filter),
missing-frame interpolation, and CSV export for MediaPipe Face Landmarker outputs.
"""

from .config import NormalizationConfig
from .pipeline import normalize_video_landmarks, normalize_landmark_sequence

__all__ = [
    "NormalizationConfig",
    "normalize_video_landmarks",
    "normalize_landmark_sequence",
]
