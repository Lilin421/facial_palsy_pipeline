"""
Temporal smoothing for landmark sequences using One Euro Filter.

Applies independent One Euro Filters to each landmark coordinate (x, y, z).
This matches MediaPipe's approach in landmarks_smoothing_calculator.cc where
each landmark dimension is filtered independently.

Reference:
    mediapipe/calculators/util/landmarks_smoothing_calculator.cc
    - Creates one filter per landmark per dimension
    - Applies filter at each frame using frame timestamp
"""

import numpy as np
from numpy.typing import NDArray

from .one_euro_filter import OneEuroFilter
from .config import NormalizationConfig


def smooth_landmark_sequence(
    sequence: NDArray[np.float64],
    fps: float,
    config: NormalizationConfig,
    valid_mask: NDArray[np.bool_] | None = None,
) -> NDArray[np.float64]:
    """Apply One Euro Filter to a sequence of landmark frames.

    Filters each landmark's x, y, z independently across time.

    Args:
        sequence: Shape (T, N, 3) — T frames, N landmarks, 3 coordinates.
        fps: Frames per second of the video.
        config: Normalization configuration with filter parameters.
        valid_mask: Shape (T,) boolean array. True = valid frame.
            If None, all frames are assumed valid.

    Returns:
        Smoothed sequence of same shape (T, N, 3).
    """
    T, N, D = sequence.shape
    smoothed = sequence.copy()

    if valid_mask is None:
        valid_mask = np.ones(T, dtype=bool)

    # Create filters: one per landmark per dimension
    # filters[landmark_idx][dim] = OneEuroFilter instance
    filters: list[list[OneEuroFilter]] = [
        [
            OneEuroFilter(
                min_cutoff=config.one_euro_min_cutoff,
                beta=config.one_euro_beta,
                derivate_cutoff=config.one_euro_derivate_cutoff,
            )
            for _ in range(D)
        ]
        for _ in range(N)
    ]

    for t in range(T):
        if not valid_mask[t]:
            # Skip invalid frames — filters retain state from last valid frame
            continue

        timestamp_s = t / fps

        for n in range(N):
            for d in range(D):
                smoothed[t, n, d] = filters[n][d].apply(
                    sequence[t, n, d], timestamp_s
                )

    return smoothed
