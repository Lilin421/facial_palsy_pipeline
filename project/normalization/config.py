"""
Configuration dataclass for the normalization pipeline.
"""

from dataclasses import dataclass, field


@dataclass
class NormalizationConfig:
    """Configuration for facial landmark normalization pipeline.

    Attributes:
        enable_rotation: Whether to remove head roll via similarity transform.
        enable_temporal_smoothing: Whether to apply One Euro Filter.
        enable_interpolation: Whether to interpolate missing detections.

        one_euro_min_cutoff: Minimum cutoff frequency for One Euro Filter.
            Controls the amount of smoothing at low speeds.
            Lower = more smoothing. MediaPipe default: 1.0 Hz.
        one_euro_beta: Speed coefficient for One Euro Filter.
            Controls how much speed increases the cutoff.
            Higher = less lag during fast movements. MediaPipe default: 0.0.
        one_euro_derivate_cutoff: Cutoff frequency for the derivative filter.
            MediaPipe default: 1.0 Hz.

        csv_precision: Number of decimal places in CSV output.
        fps: Video frames per second (used for One Euro Filter timing).
    """

    # Spatial normalization
    enable_rotation: bool = True

    # Temporal smoothing — One Euro Filter
    # Defaults match MediaPipe landmarks_smoothing_calculator.cc
    # Reference: mediapipe/calculators/util/landmarks_smoothing_calculator.cc
    enable_temporal_smoothing: bool = True
    one_euro_min_cutoff: float = 1.0
    one_euro_beta: float = 0.0
    one_euro_derivate_cutoff: float = 1.0

    # Interpolation
    enable_interpolation: bool = True

    # Output
    csv_precision: int = 6

    # Video
    fps: float = 30.0
