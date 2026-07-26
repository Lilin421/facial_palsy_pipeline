"""
Reusable geometric feature extraction primitives.

All distance functions operate on landmark arrays of shape (T, N, 3)
where T = frames, N = landmarks, 3 = x,y,z.

All distances are normalized by a reference distance (interocular)
to reduce head-scale variation.
"""

import numpy as np
from numpy.typing import NDArray


def distance(
    landmarks: NDArray[np.float64],
    idx_a: int,
    idx_b: int,
    reference: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Euclidean distance between two landmarks, normalized by reference.

    Args:
        landmarks: Shape (T, N, 3).
        idx_a: First landmark index.
        idx_b: Second landmark index.
        reference: Shape (T,) reference distance per frame.

    Returns:
        Shape (T,) normalized distance.
    """
    diff = landmarks[:, idx_a, :] - landmarks[:, idx_b, :]
    dist = np.linalg.norm(diff, axis=1)
    return dist / reference


def vertical_distance(
    landmarks: NDArray[np.float64],
    idx_a: int,
    idx_b: int,
    reference: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Signed vertical (y-axis) displacement from idx_a to idx_b, normalized.

    Positive = idx_a is above idx_b (smaller y in image coords).

    Args:
        landmarks: Shape (T, N, 3).
        idx_a: Landmark whose vertical position is measured.
        idx_b: Reference landmark.
        reference: Shape (T,) reference distance.

    Returns:
        Shape (T,) normalized vertical displacement.
    """
    dy = landmarks[:, idx_b, 1] - landmarks[:, idx_a, 1]
    return dy / reference


def horizontal_distance(
    landmarks: NDArray[np.float64],
    idx_a: int,
    idx_b: int,
    reference: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Horizontal (x-axis) distance between two landmarks, normalized.

    Args:
        landmarks: Shape (T, N, 3).
        idx_a: First landmark index.
        idx_b: Second landmark index.
        reference: Shape (T,) reference distance.

    Returns:
        Shape (T,) normalized horizontal distance (absolute).
    """
    dx = np.abs(landmarks[:, idx_a, 0] - landmarks[:, idx_b, 0])
    return dx / reference


def velocity(signal: NDArray[np.float64]) -> NDArray[np.float64]:
    """First-order temporal derivative (frame-to-frame difference).

    Args:
        signal: Shape (T,) time series.

    Returns:
        Shape (T,) velocity. First frame is 0.
    """
    v = np.zeros_like(signal)
    v[1:] = signal[1:] - signal[:-1]
    return v


def interocular_distance(
    landmarks: NDArray[np.float64],
    left_idx: int = 33,
    right_idx: int = 263,
) -> NDArray[np.float64]:
    """Compute interocular distance as the reference for normalization.

    Uses landmarks 33 (left eye inner corner) and 263 (right eye inner corner).

    Args:
        landmarks: Shape (T, N, 3).
        left_idx: Left reference landmark.
        right_idx: Right reference landmark.

    Returns:
        Shape (T,) interocular distance per frame.
    """
    diff = landmarks[:, left_idx, :] - landmarks[:, right_idx, :]
    dist = np.linalg.norm(diff, axis=1)
    # Avoid division by zero
    dist[dist < 1e-8] = 1e-8
    return dist
