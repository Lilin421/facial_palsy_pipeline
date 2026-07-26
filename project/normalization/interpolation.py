"""
Temporal interpolation for missing face detections.

Handles three cases:
    1. Missing beginning: Copies the first valid frame backward.
       Rationale: No future information at the start; holding constant is the
       safest assumption (extrapolation introduces drift).

    2. Missing ending: Copies the last valid frame forward.
       Same rationale as missing beginning.

    3. Missing interior frames: Linear interpolation between the nearest
       valid frames on both sides.
       Rationale: Linear interpolation minimizes perceptual jitter and is
       the standard approach for short gaps in motion capture data.

For multiple consecutive missing frames, all are interpolated between
the two bounding valid frames.
"""

import numpy as np
from numpy.typing import NDArray


def interpolate_missing_frames(
    sequence: NDArray[np.float64],
    valid_mask: NDArray[np.bool_],
) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
    """Fill missing frames using linear interpolation.

    Args:
        sequence: Shape (T, N, 3) landmark array. Invalid frames may contain
            zeros or NaN (they will be overwritten).
        valid_mask: Shape (T,) boolean array. True = valid detection.

    Returns:
        Tuple of:
            - Interpolated sequence (T, N, 3)
            - Updated valid_mask where all frames are now True
              (unless the entire sequence is invalid).

    Behavior:
        - If all frames are invalid, returns the input unchanged.
        - Missing start frames are filled with the first valid frame.
        - Missing end frames are filled with the last valid frame.
        - Interior gaps are linearly interpolated.
    """
    T = sequence.shape[0]
    result = sequence.copy()
    new_mask = valid_mask.copy()

    valid_indices = np.where(valid_mask)[0]

    if len(valid_indices) == 0:
        # Entire sequence is invalid — nothing to interpolate from
        return result, new_mask

    first_valid = valid_indices[0]
    last_valid = valid_indices[-1]

    # Case 1: Missing beginning — hold first valid frame
    if first_valid > 0:
        result[:first_valid] = result[first_valid]
        new_mask[:first_valid] = True

    # Case 2: Missing ending — hold last valid frame
    if last_valid < T - 1:
        result[last_valid + 1:] = result[last_valid]
        new_mask[last_valid + 1:] = True

    # Case 3: Interior gaps — linear interpolation
    for i in range(len(valid_indices) - 1):
        start = valid_indices[i]
        end = valid_indices[i + 1]

        if end - start <= 1:
            continue  # No gap

        # Interpolate between start and end
        gap_size = end - start
        for t in range(1, gap_size):
            alpha = t / gap_size
            result[start + t] = (1.0 - alpha) * result[start] + alpha * result[end]
            new_mask[start + t] = True

    return result, new_mask
