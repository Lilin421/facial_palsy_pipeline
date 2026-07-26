"""
Feature extraction package for facial landmark time-series.

Extracts geometric features from normalized MediaPipe FaceMesh landmarks.
"""

from .feature_pipeline import extract_all_features

__all__ = ["extract_all_features"]
