"""
Feature fusion — combine multiple feature signals for joint boundary detection.

For symmetric features (left/right), boundaries are fused:
    onset = min(left_onset, right_onset)
    offset = max(left_offset, right_offset)
    plateau covers both sides.
"""

import numpy as np
from numpy.typing import NDArray

from .boundary_detection import detect_boundaries_step
from .peak_detection import detect_oscillatory_peaks
from .config import RefinementConfig


def fuse_step_boundaries(
    boundaries_list: list[dict],
) -> dict:
    """Fuse multiple step-boundary results into a single boundary.

    Takes the earliest onset, latest offset, and union of plateau regions.
    Peak time is the one with largest amplitude (assumed first in list = primary).

    Args:
        boundaries_list: List of boundary dicts from detect_boundaries_step.

    Returns:
        Single fused boundary dict.
    """
    valid = [b for b in boundaries_list if b.get("onset") is not None]

    if not valid:
        return {
            "onset": None,
            "plateau_start": None,
            "peak_time": None,
            "plateau_end": None,
            "offset": None,
        }

    onset = min(b["onset"] for b in valid)
    offset = max(b["offset"] for b in valid)
    plateau_start = min(b["plateau_start"] for b in valid)
    plateau_end = max(b["plateau_end"] for b in valid)
    # Use primary feature's peak (first in list)
    peak_time = valid[0]["peak_time"]

    return {
        "onset": onset,
        "plateau_start": plateau_start,
        "peak_time": peak_time,
        "plateau_end": plateau_end,
        "offset": offset,
    }


def fuse_oscillatory_boundaries(
    results_list: list[dict],
) -> dict:
    """Fuse multiple oscillatory results.

    Movement covers the earliest start to latest end.
    Peak times are merged and deduplicated.

    Args:
        results_list: List of oscillatory detection dicts.

    Returns:
        Single fused oscillatory dict.
    """
    valid = [r for r in results_list if r.get("movement_start") is not None]

    if not valid:
        return {
            "movement_start": None,
            "movement_end": None,
            "peak_times": [],
            "peak_count": 0,
            "average_peak_interval": 0.0,
        }

    movement_start = min(r["movement_start"] for r in valid)
    movement_end = max(r["movement_end"] for r in valid)

    # Merge peaks from primary feature (first)
    peak_times = valid[0]["peak_times"]
    peak_count = len(peak_times)

    if peak_count > 1:
        avg_interval = float(np.mean(np.diff(peak_times)))
    else:
        avg_interval = 0.0

    return {
        "movement_start": movement_start,
        "movement_end": movement_end,
        "peak_times": peak_times,
        "peak_count": peak_count,
        "average_peak_interval": avg_interval,
    }
