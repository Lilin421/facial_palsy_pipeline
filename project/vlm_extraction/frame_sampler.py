"""
Frame sampling and image encoding utilities for VLM input.

GPT Vision consumes images, so clips are sampled into representative frames
which are base64-encoded for the API.
"""

import base64
import logging
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def sample_clip_frames(clip_path: str, n_frames: int = 4) -> list[np.ndarray]:
    """Sample evenly-spaced frames from a video clip.

    Args:
        clip_path: Path to the video clip.
        n_frames: Number of frames to sample.

    Returns:
        List of BGR frames (numpy arrays). Empty if the clip cannot be read.
    """
    cap = cv2.VideoCapture(clip_path)
    if not cap.isOpened():
        logger.warning(f"Cannot open clip: {clip_path}")
        return []

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return []

    # Evenly spaced indices across the clip
    indices = np.linspace(0, total - 1, min(n_frames, total)).astype(int)
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if ret:
            frames.append(frame)

    cap.release()
    return frames


def encode_image_bgr(frame: np.ndarray) -> str:
    """Encode a BGR frame as a base64 JPEG data URL.

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


def encode_image_file(image_path: str) -> str:
    """Encode an image file as a base64 data URL.

    Args:
        image_path: Path to the image (e.g., resting.png).

    Returns:
        Base64 data URL string.
    """
    p = Path(image_path)
    if not p.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    data = p.read_bytes()
    b64 = base64.b64encode(data).decode("utf-8")
    suffix = p.suffix.lstrip(".").lower() or "png"
    mime = "jpeg" if suffix in ("jpg", "jpeg") else suffix
    return f"data:image/{mime};base64,{b64}"
