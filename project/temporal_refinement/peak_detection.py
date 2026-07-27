"""
Peak detection — for oscillatory movements (Template B).

Used for: Blink repeatedly, Blow kisses.
Detects multiple peaks (or valleys) in a windowed signal.
"""

import numpy as np
from numpy.typing import NDArray
from scipy.signal import find_peaks

from .config import RefinementConfig


def detect_oscillatory_peaks(
    signal: NDArray[np.float64],
    config: RefinementConfig,
    direction: str = "decrease",
) -> dict:
    """Detect repeated peaks/valleys in an oscillatory signal.

    Args:
        signal: Smoothed windowed signal, shape (W,).
        config: Refinement parameters.
        direction: "decrease" to find valleys (e.g., blinks),
                   "increase" to find peaks.

    Returns:
        Dict with: movement_start, movement_end, peak_times (frame indices),
        peak_count, average_peak_interval.
    """
    if len(signal) < 5:
        return _empty_oscillatory()

    # Invert signal to find valleys as peaks
    if direction == "decrease":
        search_signal = -signal
    else:
        search_signal = signal.copy()

    # Compute prominence threshold from signal range
    sig_range = np.ptp(search_signal)
    min_prominence = sig_range * config.min_peak_prominence_ratio

    peaks, properties = find_peaks(
        search_signal,
        distance=config.min_peak_distance_frames,
        prominence=min_prominence,
    )

    if len(peaks) == 0:
        return _empty_oscillatory()

    # Movement boundaries
    movement_start = int(peaks[0])
    movement_end = int(peaks[-1])

    # Average interval between peaks
    if len(peaks) > 1:
        intervals = np.diff(peaks).astype(float)
        avg_interval = float(np.mean(intervals))
    else:
        avg_interval = 0.0

    return {
        "movement_start": movement_start,
        "movement_end": movement_end,
        "peak_times": peaks.tolist(),
        "peak_count": len(peaks),
        "average_peak_interval": avg_interval,
    }


def _empty_oscillatory() -> dict:
    """Return empty oscillatory result."""
    return {
        "movement_start": None,
        "movement_end": None,
        "peak_times": [],
        "peak_count": 0,
        "average_peak_interval": 0.0,
    }
