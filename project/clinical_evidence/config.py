"""
Configuration for clinical evidence extraction.
"""

from dataclasses import dataclass


@dataclass
class EvidenceConfig:
    """Parameters for clinical evidence extraction.

    Attributes:
        fps: Video frame rate (frames <-> seconds conversion).
        min_associated_duration_s: Minimum duration for an associated movement
            to be reported (filters transient noise).
        unilateral_ratio_threshold: Minimum asymmetry (0..1) for a movement to be
            considered predominantly unilateral. Below this it is treated as
            bilateral/physiological and NOT reported as synkinesis evidence.
        associated_change_threshold: Minimum normalized feature change (relative
            to its baseline std) for a region to count as "moving".
        blink_match_tolerance_s: Tolerance for matching an eye minimum to a
            Blink Detector peak time. Eye movements matching a detected blink
            are excluded from synkinesis analysis.
    """

    fps: float = 30.0
    min_associated_duration_s: float = 0.3
    unilateral_ratio_threshold: float = 0.3
    associated_change_threshold: float = 2.0
    blink_match_tolerance_s: float = 0.2

    # A bilateral region is flagged as a possible asymmetry region when its
    # symmetry_score falls below this threshold (1.0 = perfectly symmetric).
    asymmetry_score_threshold: float = 0.6
    # Minimum amplitude (either side) for a region to be considered actively
    # moving; static regions are not flagged as asymmetric.
    asymmetry_min_amplitude: float = 0.1
