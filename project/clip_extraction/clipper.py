"""
Video clip extraction from refined task boundaries.

For each task, cuts a clip from the source video spanning:
    start = onset (or movement_start) - pre_pad
    end   = offset (or movement_end) + post_pad

Uses ffmpeg via subprocess. Output filenames include the task name.
"""

import json
import logging
import re
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)


def _sanitize_filename(name: str) -> str:
    """Convert a task name into a safe filename fragment.

    Args:
        name: Raw task name (e.g., "Raise eyebrow").

    Returns:
        Sanitized snake_case string (e.g., "raise_eyebrow").
    """
    cleaned = re.sub(r"[^\w\s-]", "", name).strip().lower()
    return re.sub(r"[\s]+", "_", cleaned)


def _get_task_bounds(task: dict) -> tuple[float, float] | None:
    """Resolve (start, end) seconds for a task.

    Step tasks use onset/offset; dynamic tasks use movement_start/movement_end.

    Args:
        task: Refined task dict.

    Returns:
        (start_s, end_s) or None if bounds cannot be resolved.
    """
    if task.get("onset") is not None and task.get("offset") is not None:
        return float(task["onset"]), float(task["offset"])

    if task.get("movement_start") is not None and task.get("movement_end") is not None:
        return float(task["movement_start"]), float(task["movement_end"])

    return None


def extract_clip(
    video_path: str,
    start_s: float,
    end_s: float,
    output_path: str,
    ffmpeg_path: str = "ffmpeg",
) -> None:
    """Extract a single clip from the video using ffmpeg.

    Re-encodes to ensure accurate cut points (stream copy can be imprecise).

    Args:
        video_path: Source video path.
        start_s: Clip start in seconds.
        end_s: Clip end in seconds.
        output_path: Destination clip path.
        ffmpeg_path: Path to the ffmpeg executable (default "ffmpeg" on PATH).

    Raises:
        FileNotFoundError: If ffmpeg executable cannot be found.
        subprocess.CalledProcessError: If ffmpeg fails.
    """
    duration = max(0.0, end_s - start_s)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        ffmpeg_path,
        "-y",
        "-ss", f"{start_s:.3f}",
        "-i", video_path,
        "-t", f"{duration:.3f}",
        "-c:v", "libx264",
        "-c:a", "aac",
        output_path,
    ]
    logger.info(f"Extracting clip [{start_s:.2f}s - {end_s:.2f}s] -> {output_path}")
    subprocess.run(cmd, check=True, capture_output=True)


def _resolve_ffmpeg(ffmpeg_path: str) -> str:
    """Locate the ffmpeg executable, falling back to imageio-ffmpeg if bundled.

    Args:
        ffmpeg_path: User-provided path or "ffmpeg".

    Returns:
        A usable ffmpeg executable path.

    Raises:
        FileNotFoundError: If no ffmpeg can be located.
    """
    import shutil

    # 1. Explicit path or PATH lookup
    found = shutil.which(ffmpeg_path)
    if found:
        return found

    # 2. Try imageio-ffmpeg's bundled binary if installed
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        pass

    raise FileNotFoundError(
        "ffmpeg not found. Install it with 'winget install ffmpeg' and restart "
        "your terminal, or 'pip install imageio-ffmpeg', or pass --ffmpeg with a full path."
    )


def extract_task_clips(
    video_path: str,
    refined_tasks_path: str,
    output_dir: str,
    pre_pad_s: float = 0.5,
    post_pad_s: float = 0.5,
    ffmpeg_path: str = "ffmpeg",
) -> list[dict]:
    """Extract a video clip for every task in refined_tasks.json.

    Clip window: [start - pre_pad, end + post_pad], clamped to >= 0.

    Args:
        video_path: Path to source video.
        refined_tasks_path: Path to refined_tasks.json.
        output_dir: Directory to write clips into.
        pre_pad_s: Seconds to include before the onset/movement_start.
        post_pad_s: Seconds to include after the offset/movement_end.
        ffmpeg_path: Path to ffmpeg executable (default "ffmpeg" on PATH).

    Returns:
        List of dicts describing each generated clip (task_id, task_name,
        clip_path, clip_start, clip_end).
    """
    if not Path(video_path).exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    ffmpeg_exe = _resolve_ffmpeg(ffmpeg_path)

    with open(refined_tasks_path, "r", encoding="utf-8") as f:
        tasks = json.load(f)
    if isinstance(tasks, dict):
        tasks = tasks.get("tasks", [])

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    manifest: list[dict] = []

    for task in tasks:
        bounds = _get_task_bounds(task)
        if bounds is None:
            logger.warning(f"Skipping task {task.get('task_id')} — no valid bounds.")
            continue

        start_s, end_s = bounds
        clip_start = max(0.0, start_s - pre_pad_s)
        clip_end = end_s + post_pad_s

        task_id = task.get("task_id", 0)
        task_name = task.get("task_name", "task")
        fname = f"{task_id:02d}_{_sanitize_filename(task_name)}.mp4"
        clip_path = str(Path(output_dir) / fname)

        try:
            extract_clip(video_path, clip_start, clip_end, clip_path, ffmpeg_exe)
        except subprocess.CalledProcessError as e:
            logger.error(f"ffmpeg failed for task {task_id}: {e.stderr.decode(errors='ignore')}")
            continue

        manifest.append({
            "task_id": task_id,
            "task_name": task_name,
            "clip_path": clip_path,
            "clip_start": round(clip_start, 3),
            "clip_end": round(clip_end, 3),
        })

    # Save manifest
    manifest_path = Path(output_dir) / "clips_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    logger.info(f"Extracted {len(manifest)} clips → {output_dir}")
    return manifest


def find_resting_frame(
    features_csv_path: str,
    window_frames: int = 15,
    search_end_frame: int | None = None,
) -> int:
    """Find the most stable ("resting") frame across ALL features.

    The resting state is the person at rest BEFORE any task begins, so the
    search is restricted to frames before the first task's onset when
    search_end_frame is provided.

    Slides a window over the feature time-series, computes the total normalized
    variance of every feature within each window, and returns the center frame
    of the window with the lowest combined variance.

    Args:
        features_csv_path: Path to features.csv.
        window_frames: Window length (frames) used to measure stability.
        search_end_frame: Only search frames [0, search_end_frame). If None,
            the whole recording is searched.

    Returns:
        Frame index of the most stable moment.
    """
    import numpy as np
    import pandas as pd

    df = pd.read_csv(features_csv_path)
    feature_cols = [c for c in df.columns if c != "frame"]
    data = df[feature_cols].values.astype(float)  # (T, F)

    T = data.shape[0]

    # Normalize each feature (over the full recording) to unit std
    std = data.std(axis=0)
    std[std < 1e-8] = 1e-8
    normed = data / std

    # Restrict search to the pre-task (resting) region
    limit = T if search_end_frame is None else min(T, max(window_frames, search_end_frame))

    if limit <= window_frames:
        return max(0, limit // 2)

    best_frame = window_frames // 2
    best_score = np.inf
    for start in range(0, limit - window_frames + 1):
        window = normed[start:start + window_frames]
        score = float(window.var(axis=0).sum())
        if score < best_score:
            best_score = score
            best_frame = start + window_frames // 2

    logger.info(
        f"Resting frame: {best_frame} (stability score={best_score:.4f}, "
        f"searched frames 0-{limit})"
    )
    return best_frame


def _first_task_onset_frame(refined_tasks_path: str, fps: float) -> int | None:
    """Return the frame index of the earliest task onset/movement_start.

    Used to restrict the resting search to the pre-task region.

    Args:
        refined_tasks_path: Path to refined_tasks.json.
        fps: Frame rate.

    Returns:
        Earliest onset frame, or None if unavailable.
    """
    with open(refined_tasks_path, "r", encoding="utf-8") as f:
        tasks = json.load(f)
    if isinstance(tasks, dict):
        tasks = tasks.get("tasks", [])

    starts = []
    for t in tasks:
        s = t.get("onset")
        if s is None:
            s = t.get("movement_start")
        if s is not None:
            starts.append(float(s))

    if not starts:
        return None
    return int(round(min(starts) * fps))


def extract_resting_image(
    video_path: str,
    features_csv_path: str,
    output_path: str,
    fps: float = 30.0,
    window_frames: int = 15,
    ffmpeg_path: str = "ffmpeg",
    refined_tasks_path: str | None = None,
) -> dict:
    """Extract a single 'resting' image at the most stable pre-task moment.

    The resting state is taken from the beginning of the recording, before any
    task begins. When refined_tasks_path is provided, the search is restricted
    to frames before the first task onset.

    Args:
        video_path: Source video path.
        features_csv_path: Path to features.csv (used to find stability).
        output_path: Destination image path (e.g. .../00_resting.png).
        fps: Video frame rate.
        window_frames: Stability measurement window length.
        ffmpeg_path: Path to ffmpeg executable.
        refined_tasks_path: Optional refined_tasks.json to bound the search
            to the pre-task region.

    Returns:
        Dict describing the resting image (frame, time, image_path).

    Raises:
        FileNotFoundError: If video or ffmpeg cannot be found.
        subprocess.CalledProcessError: If ffmpeg fails.
    """
    if not Path(video_path).exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    ffmpeg_exe = _resolve_ffmpeg(ffmpeg_path)

    search_end = None
    if refined_tasks_path is not None:
        search_end = _first_task_onset_frame(refined_tasks_path, fps)

    frame = find_resting_frame(features_csv_path, window_frames, search_end)
    timestamp_s = frame / fps

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        ffmpeg_exe,
        "-y",
        "-ss", f"{timestamp_s:.3f}",
        "-i", video_path,
        "-frames:v", "1",
        output_path,
    ]
    logger.info(f"Extracting resting image at frame {frame} ({timestamp_s:.2f}s) -> {output_path}")
    subprocess.run(cmd, check=True, capture_output=True)

    return {
        "task_name": "Resting",
        "frame": frame,
        "time": round(timestamp_s, 3),
        "image_path": output_path,
    }
