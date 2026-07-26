"""
Eye feature extraction.

Features:
    EAR_L: Eye Aspect Ratio (left eye)
    EAR_R: Eye Aspect Ratio (right eye)
    EARSpeed_L: Temporal derivative of EAR_L
    EARSpeed_R: Temporal derivative of EAR_R

Eye Aspect Ratio (EAR) formula:
    EAR = (|p2 - p6| + |p3 - p5|) / (2 * |p1 - p4|)

    Left eye landmarks (MediaPipe):
        p1=33, p2=160, p3=158, p4=133, p5=153, p6=144
    Right eye landmarks (MediaPipe):
        p1=362, p2=385, p3=387, p4=263, p5=380, p6=373

Reference:
    Soukupová & Čech, "Real-Time Eye Blink Detection using Facial Landmarks", 2016.
"""

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from .geometry import distance, velocity


# Left eye EAR landmarks
LEFT_EYE = {"p1": 33, "p2": 160, "p3": 158, "p4": 133, "p5": 153, "p6": 144}
# Right eye EAR landmarks
RIGHT_EYE = {"p1": 362, "p2": 385, "p3": 387, "p4": 263, "p5": 380, "p6": 373}


def _compute_ear(
    landmarks: NDArray[np.float64],
    eye: dict[str, int],
    reference: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Compute Eye Aspect Ratio for one eye.

    Args:
        landmarks: Shape (T, N, 3).
        eye: Dict with keys p1..p6 mapping to landmark indices.
        reference: Shape (T,) interocular distance.

    Returns:
        Shape (T,) EAR values (already scale-invariant by formula).
    """
    # Vertical distances (unnormalized — EAR is a ratio)
    v1 = np.linalg.norm(
        landmarks[:, eye["p2"], :] - landmarks[:, eye["p6"], :], axis=1
    )
    v2 = np.linalg.norm(
        landmarks[:, eye["p3"], :] - landmarks[:, eye["p5"], :], axis=1
    )
    # Horizontal distance
    h = np.linalg.norm(
        landmarks[:, eye["p1"], :] - landmarks[:, eye["p4"], :], axis=1
    )

    # Avoid division by zero
    h[h < 1e-8] = 1e-8

    ear = (v1 + v2) / (2.0 * h)
    return ear


def extract_eye_features(
    landmarks: NDArray[np.float64],
    reference: NDArray[np.float64],
) -> pd.DataFrame:
    """Extract eye-related features.

    Args:
        landmarks: Shape (T, N, 3).
        reference: Shape (T,) interocular distance.

    Returns:
        DataFrame with columns: EAR_L, EAR_R, EARSpeed_L, EARSpeed_R.
    """
    ear_l = _compute_ear(landmarks, LEFT_EYE, reference)
    ear_r = _compute_ear(landmarks, RIGHT_EYE, reference)

    return pd.DataFrame({
        "EAR_L": ear_l,
        "EAR_R": ear_r,
        "EARSpeed_L": velocity(ear_l),
        "EARSpeed_R": velocity(ear_r),
    })
