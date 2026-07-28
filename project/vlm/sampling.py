"""
Shared video sampling and image encoding utilities.

Frame sampling is backend-independent: clips are sampled at a configurable
fps (default 5 fps for open-source VLM experiments), capped at max_frames.
"""

import base64
import logging
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


def sample_video_fps(
    clip_path: str,
    sampling_fps: float,
    max_frames: int,
) -> list[NDArray[np.uint8]]:
    """Sample frames from a clip at a target fps.

    Args:
        clip_path: Path to the video clip.
        sampling_fps: Target sampling rate (frames per second).
        max_frames: Maximum number of frames to return.

    Returns:
        List of BGR frames in temporal order. Empty if the clip can't be read.
    """
    cap = cv2.VideoCapture(clip_path)
    if not cap.isOpened():
        logger.warning(f"Cannot open clip: {clip_path}")
        return []

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return []

    step = max(1, int(round(src_fps / max(sampling_fps, 1e-6))))
    indices = list(range(0, total, step))

    # Cap frame count (evenly subsample if too many)
    if len(indices) > max_frames:
        sel = np.linspace(0, len(indices) - 1, max_frames).astype(int)
        indices = [indices[i] for i in sel]

    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frames.append(frame)

    cap.release()
    logger.info(f"Sampled {len(frames)} frames from {Path(clip_path).name} "
                f"(src {src_fps:.1f}fps → {sampling_fps}fps)")
    return frames


def load_image(image_path: str) -> NDArray[np.uint8]:
    """Load an image file as a BGR array.

    Args:
        image_path: Path to the image.

    Returns:
        BGR image array.

    Raises:
        FileNotFoundError: If the image cannot be read.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    return img


def encode_bgr_to_data_url(frame: NDArray[np.uint8]) -> str:
    """Encode a BGR frame as a base64 JPEG data URL (for API backends).

    Args:
        frame: BGR image array.

    Returns:
        Base64 data URL string.
    """
    ok, buf = cv2.imencode(".jpg", frame)
    if not ok:
        raise ValueError("Failed to encode frame as JPEG.")
    b64 = base64.b64encode(buf).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"


def bgr_to_rgb(frame: NDArray[np.uint8]) -> NDArray[np.uint8]:
    """Convert a BGR frame to RGB (for local backends that expect RGB / PIL)."""
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
