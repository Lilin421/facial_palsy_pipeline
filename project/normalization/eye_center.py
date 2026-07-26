"""
Eye center computation from MediaPipe face landmarks.

Mathematical rationale:
    Using ALL contour landmarks (not just corners) provides a stable geometric
    center that is robust to partial occlusion and expression changes.
    When iris landmarks (indices 468-477) are available (478-landmark model),
    the iris center is used as it directly represents pupil position.

MediaPipe landmark indices:
    Left eye contour:  [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
    Right eye contour: [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
    Left iris:  [468, 469, 470, 471, 472]  (center = 468)
    Right iris: [473, 474, 475, 476, 477]  (center = 473)
"""

import numpy as np
from numpy.typing import NDArray

# Eye contour landmark indices (MediaPipe Face Mesh canonical)
LEFT_EYE_CONTOUR = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
RIGHT_EYE_CONTOUR = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]

# Iris center indices (available in 478-landmark model)
LEFT_IRIS_CENTER = 468
RIGHT_IRIS_CENTER = 473

# Total landmark counts
LANDMARKS_468 = 468
LANDMARKS_478 = 478


def compute_eye_centers(
    landmarks: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Compute left and right eye centers from a single frame's landmarks.

    Automatically selects iris center (478 model) or contour centroid (468 model).

    Args:
        landmarks: Array of shape (N, 3) where N is 468 or 478.

    Returns:
        Tuple of (left_eye_center, right_eye_center), each shape (3,).

    Raises:
        ValueError: If landmark count is not 468 or 478.
    """
    n = landmarks.shape[0]

    if n not in (LANDMARKS_468, LANDMARKS_478):
        raise ValueError(
            f"Expected 468 or 478 landmarks, got {n}. "
            "Ensure input is from MediaPipe FaceLandmarker."
        )

    if n == LANDMARKS_478:
        # Use iris center landmarks directly — most stable
        left_center = landmarks[LEFT_IRIS_CENTER].copy()
        right_center = landmarks[RIGHT_IRIS_CENTER].copy()
    else:
        # Fall back to centroid of eye contour
        left_center = landmarks[LEFT_EYE_CONTOUR].mean(axis=0)
        right_center = landmarks[RIGHT_EYE_CONTOUR].mean(axis=0)

    return left_center, right_center
