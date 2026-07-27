"""
Clinical evidence extraction pipeline — orchestrates Steps 1-4.

Produces clinical_evidence.json with per-task primary_summary, symmetry
(all bilateral features), and possible_associated_movements.
Objective numerical evidence only.
"""

import json
import logging
import pandas as pd
from pathlib import Path

from .config import EvidenceConfig
from .regions import get_task_spec, REGIONS, TaskRegionSpec
from .symmetry import analyze_symmetry, find_possible_asymmetry_regions
from .associated_movement import monitor_associated_movements

logger = logging.getLogger(__name__)


def _to_frame(seconds: float | None, fps: float) -> int | None:
    """Convert seconds to frame index."""
    if seconds is None:
        return None
    return int(round(seconds * fps))


def _resolve_segment(task: dict, fps: float) -> tuple[int | None, int | None, int | None]:
    """Resolve (onset, plateau_start, plateau_end) frames for step or dynamic tasks.

    Step tasks use onset/plateau_start/plateau_end.
    Dynamic tasks (blink, blow kisses) use movement_start/movement_end.

    Returns:
        (onset_frame, plateau_start_frame, plateau_end_frame).
    """
    if "onset" in task and task.get("onset") is not None:
        onset_f = _to_frame(task.get("onset"), fps)
        plateau_start_f = _to_frame(task.get("plateau_start"), fps)
        plateau_end_f = _to_frame(task.get("plateau_end"), fps)
        # Fall back if plateau missing
        if plateau_start_f is None:
            plateau_start_f = onset_f
        if plateau_end_f is None:
            plateau_end_f = _to_frame(task.get("offset"), fps)
        return onset_f, plateau_start_f, plateau_end_f

    # Dynamic task
    onset_f = _to_frame(task.get("movement_start"), fps)
    end_f = _to_frame(task.get("movement_end"), fps)
    return onset_f, onset_f, end_f


def _blink_peak_frames(task: dict, all_tasks: list[dict], fps: float) -> list[int]:
    """Collect blink peak frames from the Blink Detector output.

    Uses the "Blink repeatedly" task's peak_times as the Blink Detector result.

    Args:
        task: Current task.
        all_tasks: All tasks (to locate the blink task).
        fps: Frame rate.

    Returns:
        List of blink peak frame indices.
    """
    peaks: list[int] = []
    for t in all_tasks:
        name = t.get("task_name", "").lower()
        if "blink" in name and "peak_times" in t:
            peaks.extend(_to_frame(p, fps) for p in t["peak_times"])
    return peaks


def extract_task_evidence(
    task: dict,
    features_df: pd.DataFrame,
    config: EvidenceConfig,
    blink_peak_frames: list[int],
) -> dict:
    """Extract clinical evidence for a single task.

    Args:
        task: Refined task dict with boundaries.
        features_df: Full feature DataFrame.
        config: Evidence configuration.
        blink_peak_frames: Blink Detector peak frames (for blink exclusion).

    Returns:
        Evidence dict with primary_summary, symmetry, possible_associated_movements.
    """
    task_name = task["task_name"]
    spec = get_task_spec(task_name)

    result = {
        "task_id": task.get("task_id"),
        "task_name": task_name,
        "primary_summary": {},
        "symmetry": {},
        "possible_asymmetry_regions": [],
        "possible_associated_movements": [],
    }

    onset_f, plateau_start_f, plateau_end_f = _resolve_segment(task, config.fps)

    if onset_f is None or plateau_end_f is None:
        result["note"] = "Missing boundaries; evidence skipped."
        return result

    # Step 4: primary_summary
    if spec is not None:
        result["primary_summary"] = {
            "primary_region": spec.primary_region,
            "primary_feature": spec.primary_feature,
            "primary_direction": spec.primary_direction,
        }

    # Step 1: Symmetry over ALL bilateral features (onset -> plateau_end)
    result["symmetry"] = analyze_symmetry(
        features_df, onset_f, plateau_end_f, config
    )

    # Possible asymmetry regions derived from the symmetry evidence
    result["possible_asymmetry_regions"] = find_possible_asymmetry_regions(
        result["symmetry"], config
    )

    # Steps 2 & 3: Associated movement / synkinesis evidence during plateau
    if spec is not None:
        result["possible_associated_movements"] = monitor_associated_movements(
            features_df, spec.monitored_regions,
            plateau_start_f, plateau_end_f, config,
            blink_peak_frames=blink_peak_frames,
        )

    return result


def extract_clinical_evidence(
    refined_tasks_path: str,
    features_csv_path: str,
    output_path: str,
    config: EvidenceConfig | None = None,
) -> list[dict]:
    """Extract clinical evidence for all tasks and write clinical_evidence.json.

    Args:
        refined_tasks_path: Path to refined_tasks.json.
        features_csv_path: Path to features.csv.
        output_path: Path to write clinical_evidence.json.
        config: Optional configuration.

    Returns:
        List of per-task evidence dicts.
    """
    if config is None:
        config = EvidenceConfig()

    with open(refined_tasks_path, "r", encoding="utf-8") as f:
        tasks = json.load(f)
    if isinstance(tasks, dict):
        tasks = tasks.get("tasks", [])

    features_df = pd.read_csv(features_csv_path)
    logger.info(f"Loaded {len(tasks)} tasks, {len(features_df)} frames.")

    # Blink Detector output — used to exclude normal blinks from synkinesis
    blink_peaks = _blink_peak_frames({}, tasks, config.fps)

    evidence = [
        extract_task_evidence(t, features_df, config, blink_peaks)
        for t in tasks
    ]

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(evidence, f, indent=2, ensure_ascii=False)

    logger.info(f"Wrote clinical evidence: {output_path}")
    return evidence
