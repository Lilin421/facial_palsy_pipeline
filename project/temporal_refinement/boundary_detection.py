"""
Boundary detection — onset, offset, plateau for step-like movements (Template A).

New strategy (candidate plateau selection):
    1. Compute baseline from early frames in the window.
    2. Identify ALL candidate plateau regions where the signal stays elevated
       (past a level threshold) — not just the first transition.
    3. Merge regions separated only by brief dips (noise / blinks / micro-moves)
       so a plateau remains continuous despite short fluctuations.
    4. Discard plateaus shorter than a minimum duration.
    5. Score each candidate by duration, stability, and transition amplitude.
    6. Select the highest-scoring plateau as the target action.
    7. Compute onset/offset/peak ONLY from the selected plateau.

This avoids locking onto the first transition (which may be a blink or noise)
and instead finds the strongest sustained movement.
"""

import numpy as np
from numpy.typing import NDArray

from .config import RefinementConfig


def detect_boundaries_step(
    signal: NDArray[np.float64],
    config: RefinementConfig,
    direction: str = "increase",
) -> dict:
    """Detect onset, plateau, peak, and offset for a step-like movement.

    Args:
        signal: Smoothed windowed signal, shape (W,).
        config: Refinement parameters.
        direction: "increase" if feature goes up during action,
                   "decrease" if feature goes down.

    Returns:
        Dict with keys: onset, plateau_start, peak_time, plateau_end, offset.
        Frame indices relative to window start. None values if detection fails.
    """
    if len(signal) < 5:
        return _empty_boundaries()

    baseline = float(np.mean(signal[:10])) if len(signal) >= 10 else float(signal[0])

    # Deviation from baseline in the "action" direction (always positive)
    if direction == "increase":
        deviation = signal - baseline
    else:
        deviation = baseline - signal

    peak_amplitude = float(np.max(deviation))
    if peak_amplitude < 1e-6:
        return _empty_boundaries()

    # 1. Find all candidate plateaus
    candidates = _find_candidate_plateaus(deviation, peak_amplitude, config)
    if not candidates:
        return _empty_boundaries()

    # 2. Score and select the best plateau
    best = _select_best_plateau(deviation, candidates, peak_amplitude, config)
    plateau_start, plateau_end = best

    # 3. Peak inside the selected plateau
    peak_time = int(plateau_start + np.argmax(deviation[plateau_start:plateau_end + 1]))

    # 4. Onset/offset computed only from the selected plateau
    onset_thresh = peak_amplitude * config.onset_threshold_ratio
    onset = _find_onset(deviation, onset_thresh, plateau_start)
    offset = _find_offset(deviation, onset_thresh, plateau_end)

    return {
        "onset": onset,
        "plateau_start": plateau_start,
        "peak_time": peak_time,
        "plateau_end": plateau_end,
        "offset": offset,
    }


def _find_candidate_plateaus(
    deviation: NDArray[np.float64],
    peak_amplitude: float,
    config: RefinementConfig,
) -> list[tuple[int, int]]:
    """Find all candidate plateau regions where the signal stays elevated.

    A sample is "elevated" if its deviation exceeds level_ratio * peak_amplitude.
    Regions separated only by brief dips (within gap tolerance) are merged so
    that noise / micro-movements do not fragment a plateau. Regions shorter
    than the minimum duration are discarded.

    Args:
        deviation: Deviation-from-baseline signal (positive in action direction).
        peak_amplitude: Maximum deviation.
        config: Refinement parameters.

    Returns:
        List of (start_idx, end_idx) tuples for surviving candidate plateaus.
    """
    level = peak_amplitude * config.plateau_level_ratio
    elevated = deviation >= level

    gap_tolerance = max(1, int(config.plateau_gap_tolerance_s * config.fps))
    min_duration = max(1, int(config.min_plateau_duration_s * config.fps))

    # Find contiguous elevated runs
    runs = _find_runs(elevated)

    if not runs:
        return []

    # Merge runs separated by short gaps (brief dips)
    merged: list[list[int]] = [list(runs[0])]
    for start, end in runs[1:]:
        prev_end = merged[-1][1]
        if start - prev_end <= gap_tolerance:
            merged[-1][1] = end  # Extend across the brief gap
        else:
            merged.append([start, end])

    # Discard short plateaus
    candidates = [
        (s, e) for s, e in merged
        if (e - s + 1) >= min_duration
    ]

    return candidates


def _find_runs(mask: NDArray[np.bool_]) -> list[tuple[int, int]]:
    """Find contiguous runs of True in a boolean mask.

    Args:
        mask: Boolean array.

    Returns:
        List of (start_idx, end_idx) inclusive ranges.
    """
    runs = []
    idx = np.where(mask)[0]
    if len(idx) == 0:
        return runs

    start = idx[0]
    prev = idx[0]
    for i in idx[1:]:
        if i == prev + 1:
            prev = i
        else:
            runs.append((int(start), int(prev)))
            start = i
            prev = i
    runs.append((int(start), int(prev)))
    return runs


def _select_best_plateau(
    deviation: NDArray[np.float64],
    candidates: list[tuple[int, int]],
    peak_amplitude: float,
    config: RefinementConfig,
) -> tuple[int, int]:
    """Score candidate plateaus and return the highest-scoring one.

    Score combines (normalized 0..1 each):
        - duration: longer plateaus score higher
        - stability: lower relative variance within the plateau scores higher
        - amplitude: higher mean deviation scores higher

    Args:
        deviation: Deviation-from-baseline signal.
        candidates: List of (start, end) plateau candidates.
        peak_amplitude: Maximum deviation (for amplitude normalization).
        config: Refinement parameters (scoring weights).

    Returns:
        (start, end) of the best-scoring plateau.
    """
    max_duration = max((e - s + 1) for s, e in candidates)

    best_score = -np.inf
    best_region = candidates[0]

    for s, e in candidates:
        segment = deviation[s:e + 1]
        duration = e - s + 1

        # Duration score (normalized to longest candidate)
        duration_score = duration / max_duration

        # Stability score: 1 - normalized std within plateau
        mean_val = float(np.mean(segment))
        std_val = float(np.std(segment))
        if mean_val > 1e-8:
            stability_score = max(0.0, 1.0 - std_val / mean_val)
        else:
            stability_score = 0.0

        # Amplitude score: mean deviation relative to peak
        amplitude_score = mean_val / peak_amplitude if peak_amplitude > 1e-8 else 0.0

        score = (
            config.score_weight_duration * duration_score
            + config.score_weight_stability * stability_score
            + config.score_weight_amplitude * amplitude_score
        )

        if score > best_score:
            best_score = score
            best_region = (s, e)

    return best_region


def _find_onset(
    deviation: NDArray[np.float64],
    threshold: float,
    plateau_start: int,
) -> int:
    """Find onset: last frame before plateau_start where deviation is below threshold.

    Walks backward from plateau_start until the signal drops below the onset
    threshold, marking the start of the transition into the selected plateau.

    Args:
        deviation: Deviation-from-baseline signal.
        threshold: Onset threshold.
        plateau_start: Start index of the selected plateau.

    Returns:
        Onset frame index.
    """
    onset = plateau_start
    for i in range(plateau_start, -1, -1):
        if deviation[i] < threshold:
            onset = i
            break
        onset = i
    return onset


def _find_offset(
    deviation: NDArray[np.float64],
    threshold: float,
    plateau_end: int,
) -> int:
    """Find offset: first frame after plateau_end where deviation drops below threshold.

    Walks forward from plateau_end until the signal returns below the onset
    threshold, marking the end of the recovery from the selected plateau.

    Args:
        deviation: Deviation-from-baseline signal.
        threshold: Offset threshold.
        plateau_end: End index of the selected plateau.

    Returns:
        Offset frame index.
    """
    n = len(deviation)
    offset = plateau_end
    for i in range(plateau_end, n):
        if deviation[i] < threshold:
            offset = i
            break
        offset = i
    return offset


def _empty_boundaries() -> dict:
    """Return empty boundary dict when detection fails."""
    return {
        "onset": None,
        "plateau_start": None,
        "peak_time": None,
        "plateau_end": None,
        "offset": None,
    }
