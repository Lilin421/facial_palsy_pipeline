"""
Temporal refinement pipeline — orchestrates all components.

Takes audio-based task annotations and feature DataFrame,
refines temporal boundaries for each task.
"""

import json
import logging
import numpy as np
import pandas as pd
from pathlib import Path

from .config import RefinementConfig
from .preprocessing import smooth_signal, extract_window
from .boundary_detection import detect_boundaries_step
from .peak_detection import detect_oscillatory_peaks
from .feature_fusion import fuse_step_boundaries, fuse_oscillatory_boundaries
from .event_detector import detect_dynamic_events
from .action_templates import get_template, ActionTemplate

logger = logging.getLogger(__name__)


def refine_single_task(
    task: dict,
    features_df: pd.DataFrame,
    config: RefinementConfig,
) -> dict:
    """Refine temporal boundaries for a single task.

    Args:
        task: Task annotation dict with movement_start, next_instruction, etc.
        features_df: Full features DataFrame.
        config: Refinement configuration.

    Returns:
        Refined annotation dict.
    """
    task_name = task["task_name"]
    template = get_template(task_name)

    if template is None:
        logger.warning(f"No template found for: {task_name}")
        return _fallback_result(task, config)

    total_frames = len(features_df)
    movement_start_s = task["movement_start"]
    next_instruction_s = task.get("next_instruction")

    # Process each primary feature
    if template.pattern == "step":
        result = _refine_step(
            task, template, features_df, total_frames,
            movement_start_s, next_instruction_s, config
        )
    else:
        result = _refine_oscillatory(
            task, template, features_df, total_frames,
            movement_start_s, next_instruction_s, config
        )

    return result


def _refine_step(
    task: dict,
    template: ActionTemplate,
    features_df: pd.DataFrame,
    total_frames: int,
    movement_start_s: float,
    next_instruction_s: float | None,
    config: RefinementConfig,
) -> dict:
    """Refine a step-type (Template A) movement."""
    boundaries_list = []

    for feat_name, direction in zip(template.primary_features, template.primary_directions):
        if feat_name not in features_df.columns:
            logger.warning(f"Feature {feat_name} not in DataFrame, skipping.")
            continue

        signal = features_df[feat_name].values.astype(np.float64)

        # Extract search window
        windowed, start_frame, end_frame = extract_window(
            signal, movement_start_s, next_instruction_s,
            config.fps, config.window_pad_s, total_frames
        )

        # Smooth
        smoothed = smooth_signal(windowed, config)

        # Detect boundaries
        boundaries = detect_boundaries_step(smoothed, config, direction)
        boundaries_list.append(boundaries)

    # Fuse boundaries from all primary features
    fused = fuse_step_boundaries(boundaries_list)

    # Convert frame indices to seconds
    # start_frame is the offset of the window within the full video
    _, start_frame, end_frame = extract_window(
        features_df[template.primary_features[0]].values,
        movement_start_s, next_instruction_s,
        config.fps, config.window_pad_s, total_frames
    )

    search_window = [start_frame / config.fps, end_frame / config.fps]

    def to_seconds(frame_idx):
        if frame_idx is None:
            return None
        return round((start_frame + frame_idx) / config.fps, 3)

    # Compute confidence based on whether all features agree
    valid_count = sum(1 for b in boundaries_list if b.get("onset") is not None)
    confidence = round(valid_count / max(len(template.primary_features), 1), 2)

    return {
        "task_id": task["task_id"],
        "task_name": task["task_name"],
        "search_window": [round(search_window[0], 2), round(search_window[1], 2)],
        "onset": to_seconds(fused["onset"]),
        "plateau_start": to_seconds(fused["plateau_start"]),
        "peak_time": to_seconds(fused["peak_time"]),
        "plateau_end": to_seconds(fused["plateau_end"]),
        "offset": to_seconds(fused["offset"]),
        "confidence": confidence,
        "used_features": template.primary_features,
    }


def _refine_oscillatory(
    task: dict,
    template: ActionTemplate,
    features_df: pd.DataFrame,
    total_frames: int,
    movement_start_s: float,
    next_instruction_s: float | None,
    config: RefinementConfig,
) -> dict:
    """Refine an oscillatory-type (Template B) movement.

    Uses the event_detector approach:
        Feature → Independent Peak Detection → Peak Merge → Movement Window

    Does NOT fuse features before analysis.
    """
    # Collect windowed signals for each primary feature independently
    signals: dict[str, np.ndarray] = {}
    directions: dict[str, str] = {}
    start_frame = 0

    for feat_name, direction in zip(template.primary_features, template.primary_directions):
        if feat_name not in features_df.columns:
            continue

        signal = features_df[feat_name].values.astype(np.float64)

        windowed, sf, end_frame = extract_window(
            signal, movement_start_s, next_instruction_s,
            config.fps, config.window_pad_s, total_frames
        )
        start_frame = sf  # Same window for all features
        signals[feat_name] = windowed
        directions[feat_name] = direction

    if not signals:
        return _fallback_result(task, config)

    # Run independent peak detection + merge
    result = detect_dynamic_events(signals, directions, config)

    # Convert frame indices to seconds (offset by window start)
    def to_seconds(frame_idx):
        if frame_idx is None:
            return None
        return round((start_frame + frame_idx) / config.fps, 3)

    peak_times_s = [to_seconds(p) for p in result["peak_times"]]

    # Average interval in seconds
    avg_interval_s = result["average_peak_interval"] / config.fps if result["average_peak_interval"] else 0.0

    # Confidence: based on whether peaks were detected
    confidence = 1.0 if result["peak_count"] > 0 else 0.0

    return {
        "task_id": task["task_id"],
        "task_name": task["task_name"],
        "movement_start": to_seconds(result["movement_start"]),
        "movement_end": to_seconds(result["movement_end"]),
        "peak_times": peak_times_s,
        "peak_count": result["peak_count"],
        "average_peak_interval": round(avg_interval_s, 3),
        "confidence": confidence,
        "used_features": list(signals.keys()),
    }


def _fallback_result(task: dict, config: RefinementConfig) -> dict:
    """Return a fallback result when no template matches."""
    return {
        "task_id": task["task_id"],
        "task_name": task["task_name"],
        "onset": task["movement_start"],
        "offset": task.get("next_instruction"),
        "confidence": 0.0,
        "used_features": [],
        "note": "No matching action template found.",
    }


def refine_all_tasks(
    tasks_json_path: str,
    features_csv_path: str,
    config: RefinementConfig | None = None,
) -> list[dict]:
    """Refine temporal boundaries for all tasks.

    Args:
        tasks_json_path: Path to audio-generated task.json.
        features_csv_path: Path to features.csv.
        config: Optional config. Uses defaults if None.

    Returns:
        List of refined annotation dicts.
    """
    if config is None:
        config = RefinementConfig()

    # Load inputs
    with open(tasks_json_path, "r", encoding="utf-8") as f:
        tasks_data = json.load(f)

    tasks = tasks_data.get("tasks", tasks_data) if isinstance(tasks_data, dict) else tasks_data
    features_df = pd.read_csv(features_csv_path)

    logger.info(f"Loaded {len(tasks)} tasks, {len(features_df)} frames of features.")

    results = []
    for task in tasks:
        refined = refine_single_task(task, features_df, config)
        results.append(refined)
        logger.info(f"Refined: {task['task_name']} (confidence={refined.get('confidence', 0)})")

    return results
