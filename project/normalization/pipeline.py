"""
Main normalization pipeline — orchestrates all steps.

Pipeline order:
    1. Extract landmarks from video using existing FaceLandmarker
    2. Interpolate missing detections
    3. Spatial normalization (per-frame):
        a. Compute eye centers
        b. Translate to eye midpoint
        c. Scale by inter-eye distance
        d. Rotate to remove head roll
    4. Temporal smoothing (One Euro Filter)
    5. Export to CSV
"""

import logging
import numpy as np
from numpy.typing import NDArray
from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks.python import vision

from .config import NormalizationConfig
from .eye_center import compute_eye_centers
from .similarity_transform import compute_similarity_transform
from .temporal_smoothing import smooth_landmark_sequence
from .interpolation import interpolate_missing_frames
from .io import save_landmarks_csv
from .utils import mediapipe_to_numpy, validate_landmarks

logger = logging.getLogger(__name__)


def extract_landmarks_from_video(
    video_path: str,
    detector: vision.FaceLandmarker,
) -> tuple[NDArray[np.float64], NDArray[np.bool_], float]:
    """Extract per-frame landmarks from a video using an existing detector.

    Args:
        video_path: Path to video file.
        detector: Pre-configured MediaPipe FaceLandmarker in VIDEO mode.

    Returns:
        Tuple of:
            - sequence: Shape (T, N, 3) — all frames' landmarks.
              Invalid frames have zeros.
            - valid_mask: Shape (T,) — True where face was detected.
            - fps: Video frame rate.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    logger.info(f"Video: {video_path} ({total_frames} frames @ {fps:.1f} fps)")

    # We don't know N until first detection; start with None
    all_landmarks: list[NDArray[np.float64] | None] = []
    n_landmarks: int | None = None

    frame_id = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        timestamp_ms = int(frame_id * 1000 / fps)

        result = detector.detect_for_video(mp_image, timestamp_ms)

        if result.face_landmarks and len(result.face_landmarks) > 0:
            lm_array = mediapipe_to_numpy(result.face_landmarks[0])
            if n_landmarks is None:
                n_landmarks = lm_array.shape[0]
            all_landmarks.append(lm_array)
        else:
            all_landmarks.append(None)

        frame_id += 1

    cap.release()

    if n_landmarks is None:
        # No faces detected in entire video
        raise RuntimeError("No face detected in any frame of the video.")

    T = len(all_landmarks)
    sequence = np.zeros((T, n_landmarks, 3), dtype=np.float64)
    valid_mask = np.zeros(T, dtype=bool)

    for t, lm in enumerate(all_landmarks):
        if lm is not None:
            sequence[t] = lm
            valid_mask[t] = True

    logger.info(f"Extracted {T} frames, {valid_mask.sum()} with detections.")
    return sequence, valid_mask, fps


def normalize_spatially(
    sequence: NDArray[np.float64],
    valid_mask: NDArray[np.bool_],
    enable_rotation: bool = True,
) -> NDArray[np.float64]:
    """Apply spatial normalization to each valid frame.

    Args:
        sequence: Shape (T, N, 3).
        valid_mask: Shape (T,).
        enable_rotation: Whether to remove head roll.

    Returns:
        Spatially normalized sequence (T, N, 3).
    """
    result = sequence.copy()
    T = sequence.shape[0]

    for t in range(T):
        if not valid_mask[t]:
            continue

        landmarks = sequence[t]
        left_eye, right_eye = compute_eye_centers(landmarks)
        result[t] = compute_similarity_transform(
            landmarks, left_eye, right_eye, enable_rotation=enable_rotation
        )

    return result


def normalize_landmark_sequence(
    sequence: NDArray[np.float64],
    valid_mask: NDArray[np.bool_],
    fps: float,
    config: NormalizationConfig | None = None,
) -> NDArray[np.float64]:
    """Full normalization pipeline on a pre-extracted landmark sequence.

    Args:
        sequence: Shape (T, N, 3).
        valid_mask: Shape (T,) — True for frames with detections.
        fps: Frame rate.
        config: Optional config. Uses defaults if None.

    Returns:
        Fully normalized sequence (T, N, 3).
    """
    if config is None:
        config = NormalizationConfig()
    config.fps = fps

    # Step 1: Interpolate missing frames
    if config.enable_interpolation:
        sequence, valid_mask = interpolate_missing_frames(sequence, valid_mask)
        logger.info("Interpolation complete.")

    # Step 2: Spatial normalization
    normalized = normalize_spatially(sequence, valid_mask, config.enable_rotation)
    logger.info("Spatial normalization complete.")

    # Step 3: Temporal smoothing
    if config.enable_temporal_smoothing:
        normalized = smooth_landmark_sequence(
            normalized, fps, config, valid_mask
        )
        logger.info("Temporal smoothing complete.")

    return normalized


def normalize_video_landmarks(
    video_path: str,
    detector: vision.FaceLandmarker,
    output_csv: str,
    config: NormalizationConfig | None = None,
) -> NDArray[np.float64]:
    """End-to-end normalization pipeline: video → normalized CSV.

    Args:
        video_path: Path to input video.
        detector: Pre-configured FaceLandmarker (VIDEO mode).
        output_csv: Path for output CSV file.
        config: Optional config.

    Returns:
        Normalized sequence (T, N, 3).
    """
    if config is None:
        config = NormalizationConfig()

    # Extract
    sequence, valid_mask, fps = extract_landmarks_from_video(video_path, detector)
    config.fps = fps

    # Store raw for visualization
    raw_sequence = sequence.copy()

    # Normalize
    normalized = normalize_landmark_sequence(sequence, valid_mask, fps, config)

    # Validate
    validate_landmarks(normalized)
    logger.info("Validation passed: no NaN, correct dimensions.")

    # Save
    save_landmarks_csv(normalized, output_csv, precision=config.csv_precision)
    logger.info(f"Saved CSV: {output_csv}")

    return normalized
