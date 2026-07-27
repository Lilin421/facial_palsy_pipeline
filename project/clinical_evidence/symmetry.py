"""
Step 1 — Movement Symmetry Analysis.

Evaluates ALL available bilateral features during the task duration
(onset → plateau_end), not only the primary feature. The primary feature is
only highlighted in the summary; it does not limit the analysis.

Reports numerical values only. No clinical conclusions.
"""

import numpy as np
import pandas as pd

from .config import EvidenceConfig
from .regions import RegionFeatures, REGIONS, BILATERAL_REGIONS


def _safe_ratio(a: float, b: float) -> float:
    """Ratio of the smaller to the larger value (0..1), symmetric."""
    a, b = abs(a), abs(b)
    hi = max(a, b)
    if hi < 1e-8:
        return 1.0
    return round(min(a, b) / hi, 4)


def _amplitude(segment: np.ndarray) -> float:
    """Peak absolute deviation from the segment's starting baseline.

    Direction-agnostic so all bilateral features can be analyzed uniformly.
    """
    if len(segment) == 0:
        return 0.0
    baseline = float(segment[0])
    return float(np.max(np.abs(segment - baseline)))


def _onset_frame(segment: np.ndarray, threshold_ratio: float = 0.15) -> int:
    """Relative frame index where the segment first crosses onset threshold."""
    if len(segment) < 2:
        return 0
    baseline = float(segment[0])
    amp = _amplitude(segment)
    if amp < 1e-8:
        return 0
    thresh = amp * threshold_ratio
    for i in range(len(segment)):
        if abs(segment[i] - baseline) >= thresh:
            return i
    return 0


def analyze_region_symmetry(
    features_df: pd.DataFrame,
    region: RegionFeatures,
    lo: int,
    hi: int,
    config: EvidenceConfig,
) -> dict:
    """Compute numerical symmetry evidence for one bilateral region.

    Args:
        features_df: Full feature DataFrame.
        region: Left/right feature definition.
        lo: Start frame (inclusive).
        hi: End frame (inclusive).
        config: Evidence configuration.

    Returns:
        Dict of numerical symmetry evidence for this region.
    """
    left = features_df[region.left].values[lo:hi + 1].astype(float)
    right = features_df[region.right].values[lo:hi + 1].astype(float)

    left_amp = _amplitude(left)
    right_amp = _amplitude(right)

    left_peak = float(left[np.argmax(np.abs(left - left[0]))])
    right_peak = float(right[np.argmax(np.abs(right - right[0]))])

    left_speed = float(np.max(np.abs(np.diff(left)))) * config.fps if len(left) > 1 else 0.0
    right_speed = float(np.max(np.abs(np.diff(right)))) * config.fps if len(right) > 1 else 0.0

    left_onset_rel = _onset_frame(left)
    right_onset_rel = _onset_frame(right)
    onset_delay_s = abs(left_onset_rel - right_onset_rel) / config.fps

    amplitude_ratio = _safe_ratio(left_amp, right_amp)
    speed_ratio = _safe_ratio(left_speed, right_speed)

    delay_penalty = max(0.0, 1.0 - onset_delay_s)
    symmetry_score = round(
        float(np.mean([amplitude_ratio, speed_ratio, delay_penalty])), 4
    )

    return {
        "left_feature": region.left,
        "right_feature": region.right,
        "left_amplitude": round(left_amp, 6),
        "right_amplitude": round(right_amp, 6),
        "amplitude_ratio": amplitude_ratio,
        "left_peak": round(left_peak, 6),
        "right_peak": round(right_peak, 6),
        "left_speed": round(left_speed, 6),
        "right_speed": round(right_speed, 6),
        "speed_ratio": speed_ratio,
        "left_onset_s": round((lo + left_onset_rel) / config.fps, 3),
        "right_onset_s": round((lo + right_onset_rel) / config.fps, 3),
        "onset_delay_s": round(onset_delay_s, 3),
        "symmetry_score": symmetry_score,
    }


def analyze_symmetry(
    features_df: pd.DataFrame,
    onset_frame: int,
    plateau_end_frame: int,
    config: EvidenceConfig,
) -> dict:
    """Compute symmetry evidence for ALL bilateral regions over the segment.

    Args:
        features_df: Full feature DataFrame.
        onset_frame: Start frame (inclusive).
        plateau_end_frame: End frame (inclusive).
        config: Evidence configuration.

    Returns:
        Dict mapping region_name → symmetry evidence, plus plateau_duration_s.
    """
    lo = max(0, onset_frame)
    hi = min(len(features_df) - 1, plateau_end_frame)
    if hi <= lo:
        return {}

    result: dict = {
        "plateau_duration_s": round((hi - lo) / config.fps, 3),
        "regions": {},
    }

    for region_name in BILATERAL_REGIONS:
        region = REGIONS.get(region_name)
        if region is None or region.bilateral:
            continue
        if region.left not in features_df.columns or region.right not in features_df.columns:
            continue
        result["regions"][region_name] = analyze_region_symmetry(
            features_df, region, lo, hi, config
        )

    return result


def find_possible_asymmetry_regions(
    symmetry: dict,
    config: EvidenceConfig,
) -> list[dict]:
    """Identify bilateral regions with notable left/right asymmetry.

    A region is flagged when it is actively moving (amplitude above a floor)
    AND its symmetry_score falls below the configured threshold.

    Args:
        symmetry: Output of analyze_symmetry (contains per-region evidence).
        config: Evidence configuration.

    Returns:
        List of dicts describing possible asymmetry regions, sorted by
        ascending symmetry_score (most asymmetric first).
    """
    regions = symmetry.get("regions", {})
    flagged: list[dict] = []

    for region_name, ev in regions.items():
        score = ev.get("symmetry_score", 1.0)
        left_amp = ev.get("left_amplitude", 0.0)
        right_amp = ev.get("right_amplitude", 0.0)
        max_amp = max(left_amp, right_amp)

        # Skip static regions — asymmetry is only meaningful during movement
        if max_amp < config.asymmetry_min_amplitude:
            continue

        if score < config.asymmetry_score_threshold:
            weaker_side = "L" if left_amp <= right_amp else "R"
            flagged.append({
                "region": region_name,
                "symmetry_score": score,
                "amplitude_ratio": ev.get("amplitude_ratio"),
                "speed_ratio": ev.get("speed_ratio"),
                "onset_delay_s": ev.get("onset_delay_s"),
                "weaker_side": weaker_side,
                "left_amplitude": left_amp,
                "right_amplitude": right_amp,
            })

    flagged.sort(key=lambda d: d["symmetry_score"])
    return flagged
