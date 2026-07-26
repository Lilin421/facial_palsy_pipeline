"""
Debug script — tests temporal interpolation on synthetic data with gaps.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from normalization.interpolation import interpolate_missing_frames


def main() -> None:
    # Create synthetic sequence: 20 frames, 3 landmarks, xyz
    T, N = 20, 3
    sequence = np.zeros((T, N, 3))

    # Linear motion: landmark 0 moves from (0,0,0) to (1,1,1)
    for t in range(T):
        sequence[t, 0] = [t / (T - 1), t / (T - 1), t / (T - 1)]
        sequence[t, 1] = [0.5, 0.5, 0.5]  # stationary
        sequence[t, 2] = [t / (T - 1), 0, 0]  # x-only motion

    # Create gaps
    valid_mask = np.ones(T, dtype=bool)
    valid_mask[0:3] = False    # Missing beginning
    valid_mask[8:12] = False   # Interior gap
    valid_mask[17:] = False    # Missing ending

    # Zero out invalid frames
    sequence_with_gaps = sequence.copy()
    sequence_with_gaps[~valid_mask] = 0

    # Interpolate
    result, new_mask = interpolate_missing_frames(sequence_with_gaps, valid_mask)

    print("=== Interpolation Debug ===")
    print(f"Total frames: {T}")
    print(f"Valid before: {valid_mask.sum()}")
    print(f"Valid after:  {new_mask.sum()}")
    print()

    # Check missing beginning (should hold first valid = frame 3)
    print("Missing beginning (frames 0-2, held from frame 3):")
    print(f"  Frame 0, LM 0: {result[0, 0]} (expected: {sequence[3, 0]})")
    print()

    # Check interior gap (should be linearly interpolated)
    print("Interior gap (frames 8-11, interpolated between 7 and 12):")
    for t in range(8, 12):
        alpha = (t - 7) / (12 - 7)
        expected = (1 - alpha) * sequence[7, 0] + alpha * sequence[12, 0]
        print(f"  Frame {t}, LM 0: {result[t, 0]} (expected: {expected})")
    print()

    # Check missing ending (should hold last valid = frame 16)
    print("Missing ending (frames 17-19, held from frame 16):")
    print(f"  Frame 19, LM 0: {result[19, 0]} (expected: {sequence[16, 0]})")
    print()

    print("All NaN check:", not np.any(np.isnan(result)))


if __name__ == "__main__":
    main()
