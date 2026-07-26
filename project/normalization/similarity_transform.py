"""
Similarity transform for spatial normalization of facial landmarks.

Mathematical derivation:
    Given left_eye (L) and right_eye (R), the similarity transform:
    1. Translates origin to midpoint M = (L + R) / 2
    2. Scales so that ||L - R|| = 1 (inter-eye distance normalization)
    3. Rotates so that the eye-line becomes horizontal (removes head roll)

    The 2D rotation angle θ is computed as:
        θ = atan2(R_y - L_y, R_x - L_x)
    We rotate by -θ to make the eye line horizontal.

    For z-coordinate:
        - Translation: z is shifted by midpoint_z (same as x,y)
        - Scale: z is divided by inter-eye distance (same as x,y)
        - Rotation: z is NOT rotated because roll rotation is in the x-y plane.
          The z-axis represents depth. Rotating z would require knowledge of
          pitch/yaw which we intentionally preserve.

    This preserves facial geometry while removing position, scale, and roll.
"""

import numpy as np
from numpy.typing import NDArray


def compute_similarity_transform(
    landmarks: NDArray[np.float64],
    left_eye: NDArray[np.float64],
    right_eye: NDArray[np.float64],
    enable_rotation: bool = True,
) -> NDArray[np.float64]:
    """Apply similarity transform to normalize landmarks spatially.

    Steps applied in order:
        1. Translate so eye midpoint is at origin
        2. Scale so inter-eye distance = 1
        3. Rotate so eye line is horizontal (optional)

    Args:
        landmarks: Shape (N, 3) landmark coordinates.
        left_eye: Shape (3,) left eye center.
        right_eye: Shape (3,) right eye center.
        enable_rotation: If True, remove head roll.

    Returns:
        Normalized landmarks of shape (N, 3).

    Raises:
        ValueError: If inter-eye distance is zero (degenerate case).
    """
    # Step 1: Translation — move origin to eye midpoint
    midpoint = (left_eye + right_eye) / 2.0
    translated = landmarks - midpoint

    # Also translate eye centers for rotation computation
    left_translated = left_eye - midpoint
    right_translated = right_eye - midpoint

    # Step 2: Scale — normalize by inter-eye distance
    inter_eye_dist = np.linalg.norm(right_eye - left_eye)

    if inter_eye_dist < 1e-8:
        raise ValueError(
            "Inter-eye distance is approximately zero. "
            "Cannot normalize scale. Check landmark detection quality."
        )

    scaled = translated / inter_eye_dist

    # Step 3: Rotation — remove head roll (rotate in x-y plane)
    if enable_rotation:
        # Compute angle of eye line relative to horizontal
        dx = right_translated[0] - left_translated[0]
        dy = right_translated[1] - left_translated[1]
        angle = np.arctan2(dy, dx)

        # Rotation matrix for -angle (to make horizontal)
        cos_a = np.cos(-angle)
        sin_a = np.sin(-angle)

        # Apply 2D rotation to x, y only; z remains unchanged
        # This is correct because roll is rotation around the z-axis
        x_rot = scaled[:, 0] * cos_a - scaled[:, 1] * sin_a
        y_rot = scaled[:, 0] * sin_a + scaled[:, 1] * cos_a

        result = np.column_stack([x_rot, y_rot, scaled[:, 2]])
    else:
        result = scaled

    return result
