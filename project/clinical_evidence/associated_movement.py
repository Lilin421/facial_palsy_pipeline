"""
Step 2 & 3 — Associated Movement Monitoring and Possible Synkinesis Evidence.

Monitors non-primary facial regions during the primary movement plateau.

Distinguishes:
    - Physiological associated movement (bilateral, symmetric) → NOT reported.
    - Unexpected unilateral associated movement → reported as possible evidence.

Rules for possible synkinesis evidence:
    1. Occurs during the primary movement plateau.
    2. Occurs outside the primary region.
    3. Eye movements matching Blink Detector peaks are excluded.
    4. Movement is predominantly unilateral (bilateral physiological ignored).
    5. Duration exceeds a configurable minimum.

Reports evidence only. Does NOT diagnose synkinesis.
"""

import numpy as np
import pandas as pd

from .config import EvidenceConfig
from .regions import RegionFeatures, REGIONS


def _baseline_std(signal: np.ndarray, plateau_lo: int) -> tuple[float, float]:
    """Estimate baseline mean and std from frames before the plateau."""
    pre = signal[:plateau_lo] if plateau_lo > 3 else signal[:max(3, plateau_lo)]
    if len(pre) < 2:
        return float(signal[0]), 1e-6
    return float(np.mean(pre)), float(np.std(pre)) + 1e-8


def _movement_duration(
    signal: np.ndarray,
    lo: int,
    hi: int,
    baseline_mean: float,
    baseline_std: float,
    change_threshold: float,
    fps: float,
) -> float:
    """Duration (seconds) that the signal deviates beyond threshold within [lo, hi]."""
    segment = signal[lo:hi + 1]
    deviating = np.abs(segment - baseline_mean) > (change_threshold * baseline_std)
    return float(np.sum(deviating)) / fps


def _overlaps_detected_blink(
    lo: int,
    hi: int,
    blink_peak_frames: list[int],
    tolerance_frames: int,
) -> bool:
    """Check whether the plateau window overlaps any detected blink peak.

    Uses the Blink Detector output (peak frames) rather than a heuristic.

    Args:
        lo: Plateau start frame.
        hi: Plateau end frame.
        blink_peak_frames: Blink peak frame indices from the Blink Detector.
        tolerance_frames: Matching tolerance around each blink peak.

    Returns:
        True if a detected blink falls within the window (± tolerance).
    """
    for pf in blink_peak_frames:
        if (lo - tolerance_frames) <= pf <= (hi + tolerance_frames):
            return True
    return False


def monitor_associated_movements(
    features_df: pd.DataFrame,
    monitored_regions: list[str],
    plateau_start_frame: int,
    plateau_end_frame: int,
    config: EvidenceConfig,
    blink_peak_frames: list[int] | None = None,
) -> list[dict]:
    """Detect possible associated movements in monitored regions during plateau.

    Args:
        features_df: Full feature DataFrame.
        monitored_regions: Region names to monitor (non-primary).
        plateau_start_frame: Plateau start frame.
        plateau_end_frame: Plateau end frame.
        config: Evidence configuration.
        blink_peak_frames: Blink Detector peak frames used to exclude normal
            blinks from eye-region synkinesis analysis.

    Returns:
        List of evidence dicts: region, side, feature, change, duration, confidence,
        classification.
    """
    lo = max(0, plateau_start_frame)
    hi = min(len(features_df) - 1, plateau_end_frame)
    if hi <= lo:
        return []

    if blink_peak_frames is None:
        blink_peak_frames = []
    blink_tol = max(1, int(config.blink_match_tolerance_s * config.fps))

    evidence: list[dict] = []

    for region_name in monitored_regions:
        region = REGIONS.get(region_name)
        if region is None:
            continue

        # Bilateral/central regions can't be unilateral — skip synkinesis flag
        if region.bilateral:
            continue

        # Rule 3: exclude eye movements matching a detected blink
        if region_name == "eye" and _overlaps_detected_blink(lo, hi, blink_peak_frames, blink_tol):
            continue

        # Analyze each side independently
        side_changes = {}
        for side_label, feat in (("L", region.left), ("R", region.right)):
            if feat not in features_df.columns:
                continue
            signal = features_df[feat].values.astype(float)
            base_mean, base_std = _baseline_std(signal, lo)

            segment = signal[lo:hi + 1]
            change = float(np.max(np.abs(segment - base_mean)))
            normalized_change = change / base_std
            duration = _movement_duration(
                signal, lo, hi, base_mean, base_std,
                config.associated_change_threshold, config.fps
            )
            side_changes[side_label] = {
                "feature": feat,
                "change": round(change, 6),
                "normalized_change": round(normalized_change, 3),
                "duration": round(duration, 3),
            }

        if "L" not in side_changes or "R" not in side_changes:
            continue

        left_change = side_changes["L"]["normalized_change"]
        right_change = side_changes["R"]["normalized_change"]
        hi_change = max(left_change, right_change)

        # Not enough movement in either side → nothing to report
        if hi_change < config.associated_change_threshold:
            continue

        lo_change = min(left_change, right_change)
        asymmetry = 1.0 - (lo_change / hi_change if hi_change > 1e-8 else 1.0)

        # Rule 4: physiological (bilateral/symmetric) movement is NOT reported
        if asymmetry < config.unilateral_ratio_threshold:
            continue

        # Predominant (unexpected unilateral) side
        dominant_side = "L" if left_change >= right_change else "R"
        info = side_changes[dominant_side]

        # Rule 5: duration threshold
        if info["duration"] < config.min_associated_duration_s:
            continue

        confidence = round(min(1.0, (asymmetry + min(hi_change / 10.0, 1.0)) / 2.0), 3)

        evidence.append({
            "region": region_name,
            "side": dominant_side,
            "feature": info["feature"],
            "change": info["change"],
            "duration": info["duration"],
            "asymmetry": round(asymmetry, 3),
            "classification": "unexpected_unilateral",
            "confidence": confidence,
        })

    return evidence
