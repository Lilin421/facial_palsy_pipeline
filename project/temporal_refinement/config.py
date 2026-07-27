"""
Configuration for temporal refinement.
"""

from dataclasses import dataclass


@dataclass
class RefinementConfig:
    """Parameters for temporal boundary refinement.

    Attributes:
        fps: Video frame rate.
        window_pad_s: Seconds to extend search window before/after audio timestamps.
        smooth_window: Savitzky-Golay filter window length (must be odd).
        smooth_polyorder: Savitzky-Golay polynomial order.
        onset_threshold_ratio: Fraction of peak amplitude to detect onset/offset.
        plateau_threshold_ratio: Fraction of peak to define plateau boundaries.
        min_peak_prominence_ratio: Minimum prominence for peak detection (fraction of signal range).
        min_peak_distance_frames: Minimum distance between peaks in frames.
    """

    fps: float = 30.0
    window_pad_s: float = 2.0
    smooth_window: int = 7
    smooth_polyorder: int = 3
    onset_threshold_ratio: float = 0.15
    plateau_threshold_ratio: float = 0.70
    min_peak_prominence_ratio: float = 0.2
    min_peak_distance_frames: int = 5

    # --- Candidate plateau detection ---
    # Minimum plateau duration to be considered valid (seconds).
    # Discards short plateaus caused by blinks / noise / micro-movements.
    min_plateau_duration_s: float = 0.4

    # Level threshold: fraction of amplitude a sample must reach to be
    # considered "elevated" (part of a potential plateau).
    plateau_level_ratio: float = 0.5

    # Gap tolerance: brief dips below the level lasting up to this many
    # seconds do NOT terminate a plateau (handles noise / micro-movements).
    plateau_gap_tolerance_s: float = 0.15

    # Scoring weights for candidate plateau selection.
    score_weight_duration: float = 1.0
    score_weight_stability: float = 1.0
    score_weight_amplitude: float = 1.0
