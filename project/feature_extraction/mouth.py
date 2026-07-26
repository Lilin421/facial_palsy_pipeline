"""
Mouth feature extraction.

Features:
    MouthWidth: Distance 61 ↔ 291
    LipGap: Distance 13 ↔ 14
    MouthArea: MouthWidth * LipGap
    LipCornerLift_L: Vertical displacement of 61 relative to 13
    LipCornerLift_R: Vertical displacement of 291 relative to 13
    UpperLipHeight: Distance 2 ↔ 13
    LowerLipHeight: Distance 14 ↔ 152
"""

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from .geometry import distance, vertical_distance


def extract_mouth_features(
    landmarks: NDArray[np.float64],
    reference: NDArray[np.float64],
) -> pd.DataFrame:
    """Extract mouth-related features.

    Args:
        landmarks: Shape (T, N, 3).
        reference: Shape (T,) interocular distance.

    Returns:
        DataFrame with mouth feature columns.
    """
    mouth_width = distance(landmarks, 61, 291, reference)
    lip_gap = distance(landmarks, 13, 14, reference)
    mouth_area = mouth_width * lip_gap

    lip_corner_l = vertical_distance(landmarks, 61, 13, reference)
    lip_corner_r = vertical_distance(landmarks, 291, 13, reference)

    upper_lip_height = distance(landmarks, 2, 13, reference)
    lower_lip_height = distance(landmarks, 14, 152, reference)

    return pd.DataFrame({
        "MouthWidth": mouth_width,
        "LipGap": lip_gap,
        "MouthArea": mouth_area,
        "LipCornerLift_L": lip_corner_l,
        "LipCornerLift_R": lip_corner_r,
        "UpperLipHeight": upper_lip_height,
        "LowerLipHeight": lower_lip_height,
    })
