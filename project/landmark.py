"""
Landmark module — Face landmark detection functions.
"""

import cv2
import time
import logging
import mediapipe as mp
from pathlib import Path
from typing import Optional

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

logger = logging.getLogger(__name__)


def load_face_landmarker(
    model_path: str = "landmark/face_landmarker.task",
) -> vision.FaceLandmarker:
    """Load the MediaPipe FaceLandmarker model.

    Args:
        model_path: Path to the .task model file.

    Returns:
        FaceLandmarker instance.
    """
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    base_options = python.BaseOptions(model_asset_path=str(model_path))
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_faces=1,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
    )
    detector = vision.FaceLandmarker.create_from_options(options)
    logger.info(f"Loaded FaceLandmarker from: {model_path}")
    return detector


def process_video_landmarks(
    video_path: str,
    output_path: str,
    model_path: str = "landmark/face_landmarker.task",
) -> dict:
    """Run face landmark detection on a video and save annotated result.

    Args:
        video_path: Path to input video file.
        output_path: Path to output annotated video.
        model_path: Path to the .task model file.

    Returns:
        Dict with processing stats: frame_count, avg_ms_per_frame.
    """
    # Validate input
    if not Path(video_path).exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    # Ensure output directory exists
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # Load detector
    detector = load_face_landmarker(model_path)

    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        detector.close()
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    logger.info(f"Video: {video_path} ({width}x{height} @ {fps:.1f} fps)")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_id = 0
    times = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        timestamp_ms = int(frame_id * 1000 / fps)

        t0 = time.perf_counter()
        result = detector.detect_for_video(mp_image, timestamp_ms)
        t1 = time.perf_counter()

        times.append((t1 - t0) * 1000)

        if len(result.face_landmarks):
            h, w = frame.shape[:2]
            for lm in result.face_landmarks[0]:
                x = int(lm.x * w)
                y = int(lm.y * h)
                cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)

        writer.write(frame)
        frame_id += 1

    cap.release()
    writer.release()
    detector.close()

    avg_ms = sum(times) / len(times) if times else 0.0
    logger.info(f"Processed {len(times)} frames, avg {avg_ms:.1f} ms/frame")
    logger.info(f"Saved to: {output_path}")

    return {
        "frame_count": len(times),
        "avg_ms_per_frame": avg_ms,
        "output_path": output_path,
    }
