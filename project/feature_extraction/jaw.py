"""
Jaw feature extraction.

Features:
    JawOpen: Distance 13 ↔ 152
"""

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from .geometry import distance


def extract_jaw_features(
    landmarks: NDArray[np.float64],
    reference: NDArray[np.float64],
) -> pd.DataFrame:
    """Extract jaw-related features.

    Args:
        landmarks: Shape (T, N, 3).
        reference: Shape (T,) interocular distance.

    Returns:
        DataFrame with column: JawOpen.
    """
    jaw_open = distance(landmarks, 13, 152, reference)

    return pd.DataFrame({
        "JawOpen": jaw_open,
    })
