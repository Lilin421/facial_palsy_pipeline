"""
Cheek feature extraction.

Features:
    CheekWidth_L: Distance 4 ↔ 205
    CheekWidth_R: Distance 4 ↔ 425
"""

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from .geometry import distance


def extract_cheek_features(
    landmarks: NDArray[np.float64],
    reference: NDArray[np.float64],
) -> pd.DataFrame:
    """Extract cheek-related features.

    Args:
        landmarks: Shape (T, N, 3).
        reference: Shape (T,) interocular distance.

    Returns:
        DataFrame with columns: CheekWidth_L, CheekWidth_R.
    """
    cheek_l = distance(landmarks, 4, 205, reference)
    cheek_r = distance(landmarks, 4, 425, reference)

    return pd.DataFrame({
        "CheekWidth_L": cheek_l,
        "CheekWidth_R": cheek_r,
    })
