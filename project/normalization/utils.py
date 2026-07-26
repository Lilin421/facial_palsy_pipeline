"""
Utility functions for landmark normalization.
"""

import numpy as np
from numpy.typing import NDArray
from typing import Any


def mediapipe_to_numpy(face_landmarks: Any) -> NDArray[np.float64]:
    """Convert MediaPipe NormalizedLandmark list to numpy array.

    Args:
        face_landmarks: result.face_landmarks[0] from MediaPipe FaceLandmarker.
            Each element has .x, .y, .z attributes.

    Returns:
        Shape (N, 3) numpy array where N is 468 or 478.
    """
    return np.array(
        [[lm.x, lm.y, lm.z] for lm in face_landmarks],
        dtype=np.float64,
    )


def validate_landmarks(sequence: NDArray[np.float64]) -> None:
    """Validate landmark sequence for common issues.

    Args:
        sequence: Shape (T, N, 3) array.

    Raises:
        ValueError: If validation fails.
    """
    if sequence.ndim != 3:
        raise ValueError(f"Expected 3D array (T, N, 3), got shape {sequence.shape}")

    if sequence.shape[2] != 3:
        raise ValueError(f"Expected 3 coordinates, got {sequence.shape[2]}")

    if np.any(np.isnan(sequence)):
        nan_frames = np.any(np.isnan(sequence), axis=(1, 2))
        raise ValueError(
            f"NaN values found in {nan_frames.sum()} frames. "
            "Run interpolation before validation."
        )


def compute_inter_eye_distance(
    left_eye: NDArray[np.float64],
    right_eye: NDArray[np.float64],
) -> float:
    """Compute Euclidean distance between eye centers.

    Args:
        left_eye: Shape (3,) left eye center.
        right_eye: Shape (3,) right eye center.

    Returns:
        Inter-eye distance (scalar).
    """
    return float(np.linalg.norm(right_eye - left_eye))
