"""
Eyebrow feature extraction.

Features:
    BrowHeight_L: Vertical distance between landmarks 52 and 33 (left brow to left eye)
    BrowHeight_R: Vertical distance between landmarks 282 and 263 (right brow to right eye)
    BrowSpeed_L: Temporal derivative of BrowHeight_L
    BrowSpeed_R: Temporal derivative of BrowHeight_R
"""

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from .geometry import vertical_distance, velocity


def extract_brow_features(
    landmarks: NDArray[np.float64],
    reference: NDArray[np.float64],
) -> pd.DataFrame:
    """Extract eyebrow-related features.

    Args:
        landmarks: Shape (T, N, 3).
        reference: Shape (T,) interocular distance.

    Returns:
        DataFrame with columns: BrowHeight_L, BrowHeight_R, BrowSpeed_L, BrowSpeed_R.
    """
    brow_l = vertical_distance(landmarks, 52, 33, reference)
    brow_r = vertical_distance(landmarks, 282, 263, reference)

    return pd.DataFrame({
        "BrowHeight_L": brow_l,
        "BrowHeight_R": brow_r,
        "BrowSpeed_L": velocity(brow_l),
        "BrowSpeed_R": velocity(brow_r),
    })
