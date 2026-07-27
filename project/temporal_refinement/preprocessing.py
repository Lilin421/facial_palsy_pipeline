"""
Feature preprocessing — smoothing and windowing.
"""

import numpy as np
from numpy.typing import NDArray
from scipy.signal import savgol_filter

from .config import RefinementConfig


def smooth_signal(
    signal: NDArray[np.float64],
    config: RefinementConfig,
) -> NDArray[np.float64]:
    """Apply Savitzky-Golay smoothing to a 1D signal.

    Args:
        signal: Shape (T,) raw feature values.
        config: Contains smooth_window and smooth_polyorder.

    Returns:
        Smoothed signal of same shape.
    """
    if len(signal) <= config.smooth_polyorder + 1:
        # Signal too short to smooth — return as-is
        return signal.copy()

    window = config.smooth_window
    # Ensure window doesn't exceed signal length and is odd
    if window > len(signal):
        window = len(signal) if len(signal) % 2 == 1 else len(signal) - 1
    if window <= config.smooth_polyorder:
        window = config.smooth_polyorder + 1
        if window % 2 == 0:
            window += 1

    return savgol_filter(signal, window, config.smooth_polyorder)


def extract_window(
    signal: NDArray[np.float64],
    movement_start_s: float,
    next_instruction_s: float | None,
    fps: float,
    window_pad_s: float,
    total_frames: int,
) -> tuple[NDArray[np.float64], int, int]:
    """Extract a local search window from the signal.

    Args:
        signal: Shape (T,) full feature time series.
        movement_start_s: Rough movement start in seconds.
        next_instruction_s: Next instruction time in seconds (or None for last task).
        fps: Frames per second.
        window_pad_s: Seconds to pad before/after.
        total_frames: Total number of frames in video.

    Returns:
        Tuple of (windowed_signal, start_frame, end_frame).
    """
    search_start_s = movement_start_s - window_pad_s
    if next_instruction_s is not None:
        search_end_s = next_instruction_s + window_pad_s
    else:
        search_end_s = total_frames / fps

    start_frame = max(0, int(search_start_s * fps))
    end_frame = min(total_frames, int(search_end_s * fps))

    return signal[start_frame:end_frame], start_frame, end_frame


def compute_baseline(
    signal: NDArray[np.float64],
    n_frames: int = 10,
) -> float:
    """Estimate baseline from the beginning of a windowed signal.

    Uses the mean of the first n_frames as baseline estimate.

    Args:
        signal: Windowed signal.
        n_frames: Number of initial frames to average.

    Returns:
        Baseline value.
    """
    n = min(n_frames, len(signal))
    return float(np.mean(signal[:n]))
