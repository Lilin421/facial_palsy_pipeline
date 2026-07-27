"""
Event detector for dynamic/oscillatory actions.

Strategy:
    Feature → Peak Detection → Peak Merge → Movement Window

This module detects events on EACH feature independently, then merges
temporally close peaks from different features into unified events.

Used for: Blink repeatedly, Blow kisses.
NOT used for plateau-type actions (those use boundary_detection + feature_fusion).
"""

import numpy as np
from numpy.typing import NDArray
from scipy.signal import find_peaks, savgol_filter

from .config import RefinementConfig


def detect_peaks_single_feature(
    signal: NDArray[np.float64],
    fps: float,
    direction: str = "decrease",
    min_peak_distance_frames: int = 5,
    min_prominence_ratio: float = 0.2,
    smooth_window: int = 5,
    smooth_polyorder: int = 2,
) -> NDArray[np.int64]:
    """Detect peaks (or valleys) on a single feature signal.

    Only applies light smoothing to preserve temporal precision.

    Args:
        signal: Shape (W,) windowed feature signal.
        fps: Frames per second.
        direction: "decrease" to find valleys, "increase" to find peaks.
        min_peak_distance_frames: Minimum frames between detected peaks.
        min_prominence_ratio: Minimum prominence as fraction of signal range.
        smooth_window: Light smoothing window (odd integer).
        smooth_polyorder: Polynomial order for smoothing.

    Returns:
        Array of peak frame indices (relative to window start).
    """
    if len(signal) < 5:
        return np.array([], dtype=np.int64)

    # Light smoothing — avoid over-smoothing to preserve peak timing
    win = min(smooth_window, len(signal))
    if win % 2 == 0:
        win -= 1
    if win > smooth_polyorder:
        smoothed = savgol_filter(signal, win, smooth_polyorder)
    else:
        smoothed = signal.copy()

    # Invert for valley detection
    if direction == "decrease":
        search_signal = -smoothed
    else:
        search_signal = smoothed

    # Compute prominence threshold
    sig_range = np.ptp(search_signal)
    if sig_range < 1e-8:
        return np.array([], dtype=np.int64)

    min_prominence = sig_range * min_prominence_ratio

    peaks, _ = find_peaks(
        search_signal,
        distance=min_peak_distance_frames,
        prominence=min_prominence,
    )

    return peaks


def merge_peaks_across_features(
    all_peaks: list[NDArray[np.int64]],
    tolerance_frames: int = 5,
) -> NDArray[np.int64]:
    """Merge peaks from multiple features into unified events.

    Peaks from different features that occur within tolerance_frames of each
    other are treated as the same event. The merged event time is the median
    of the clustered peaks.

    Args:
        all_peaks: List of peak arrays (one per feature).
        tolerance_frames: Max frame distance to merge peaks into one event.

    Returns:
        Sorted array of merged event frame indices.
    """
    # Collect all peaks into one sorted array
    non_empty = [p for p in all_peaks if len(p) > 0]
    if not non_empty:
        return np.array([], dtype=np.int64)

    combined = np.concatenate(non_empty)
    if len(combined) == 0:
        return np.array([], dtype=np.int64)

    combined = np.sort(combined)

    # Cluster peaks within tolerance
    clusters: list[list[int]] = []
    current_cluster: list[int] = [int(combined[0])]

    for i in range(1, len(combined)):
        if combined[i] - current_cluster[-1] <= tolerance_frames:
            current_cluster.append(int(combined[i]))
        else:
            clusters.append(current_cluster)
            current_cluster = [int(combined[i])]
    clusters.append(current_cluster)

    # Merged event = median of each cluster
    merged = np.array([int(np.median(c)) for c in clusters], dtype=np.int64)
    return merged


def detect_dynamic_events(
    signals: dict[str, NDArray[np.float64]],
    directions: dict[str, str],
    config: RefinementConfig,
) -> dict:
    """Full dynamic event detection pipeline.

    Detects peaks independently on each feature, then merges across features.

    Args:
        signals: Dict mapping feature_name → windowed signal array.
        directions: Dict mapping feature_name → "increase" or "decrease".
        config: Refinement configuration.

    Returns:
        Dict with: movement_start, movement_end, peak_times, peak_count,
        average_peak_interval. All in frame indices relative to window.
    """
    # Tolerance: ~150ms default
    tolerance_frames = max(1, int(0.15 * config.fps))

    # Detect peaks per feature independently
    all_peaks: list[NDArray[np.int64]] = []

    for feat_name, signal in signals.items():
        direction = directions.get(feat_name, "decrease")
        peaks = detect_peaks_single_feature(
            signal,
            fps=config.fps,
            direction=direction,
            min_peak_distance_frames=config.min_peak_distance_frames,
            min_prominence_ratio=config.min_peak_prominence_ratio,
            smooth_window=5,   # Light smoothing for dynamic actions
            smooth_polyorder=2,
        )
        all_peaks.append(peaks)

    # Merge peaks from all features
    merged = merge_peaks_across_features(all_peaks, tolerance_frames)

    if len(merged) == 0:
        return {
            "movement_start": None,
            "movement_end": None,
            "peak_times": [],
            "peak_count": 0,
            "average_peak_interval": 0.0,
        }

    movement_start = int(merged[0])
    movement_end = int(merged[-1])
    peak_count = len(merged)

    if peak_count > 1:
        avg_interval = float(np.mean(np.diff(merged)))
    else:
        avg_interval = 0.0

    return {
        "movement_start": movement_start,
        "movement_end": movement_end,
        "peak_times": merged.tolist(),
        "peak_count": peak_count,
        "average_peak_interval": avg_interval,
    }
