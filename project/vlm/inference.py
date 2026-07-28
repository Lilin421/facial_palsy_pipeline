"""
Unified, backend-independent inference API.

The upper pipeline calls `extract_features(...)` regardless of backend.
Switching GPT ↔ Qwen only requires changing config.model.
"""

import json
import logging
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from .base import VLMClient
from .config import VLMConfig
from .factory import create_client
from .sampling import sample_video_fps, load_image

logger = logging.getLogger(__name__)


def _load_prompt(prompt_file: str) -> str:
    """Load the shared, backend-independent prompt.

    Args:
        prompt_file: Path to the prompt text file.

    Returns:
        Prompt string.

    Raises:
        FileNotFoundError: If the prompt file does not exist.
    """
    p = Path(prompt_file)
    if not p.exists():
        raise FileNotFoundError(f"Prompt file not found: {prompt_file}")
    return p.read_text(encoding="utf-8")


def _build_user_text(
    prompt: str,
    region: str,
    task_name: str,
    landmark_evidence: dict | list,
) -> str:
    """Assemble the user text: shared prompt + region/task + landmark evidence.

    The instructed-movement name is provided so findings can reference it, but
    the prompt does NOT ask the model to recognize which task is performed —
    task localization is already complete upstream.

    Args:
        prompt: Shared prompt text.
        region: Facial region label.
        task_name: Instructed movement name.
        landmark_evidence: Landmark-derived clinical evidence.

    Returns:
        Composed user text.
    """
    return (
        f"{prompt}\n\n"
        f"--- Context ---\n"
        f"Region to analyse: {region}\n"
        f"Instructed movement: {task_name}\n\n"
        f"--- Landmark-derived clinical evidence (guidance only) ---\n"
        f"{json.dumps(landmark_evidence, ensure_ascii=False, indent=2)}\n\n"
        f"Report every finding with feature, side, and the instructed movement "
        f"name '{task_name}'."
    )


def extract_features(
    region: str,
    task_name: str,
    video_path: str,
    resting_image: str,
    landmark_evidence: dict | list,
    prompt_file: str | None = None,
    config: VLMConfig | None = None,
    client: VLMClient | None = None,
) -> dict:
    """Extract structured visual clinical findings for one instructed movement.

    This is the single unified API. It is identical regardless of backend.

    Args:
        region: Facial region label (e.g. "buccal").
        task_name: Instructed movement name (e.g. "Smile").
        video_path: Path to the task clip.
        resting_image: Path to the resting baseline image.
        landmark_evidence: Landmark-derived clinical evidence (dict or list).
        prompt_file: Optional prompt path override (defaults to config).
        config: VLM configuration (backend selection, sampling fps, etc.).
        client: Optional pre-initialized client (avoids reloading the model
            across multiple calls). If None, one is created from config.

    Returns:
        Structured evidence dict.
    """
    if config is None:
        config = VLMConfig()

    prompt_path = prompt_file or config.prompt_file
    prompt = _load_prompt(prompt_path)

    # Shared sampling — backend-independent
    resting = load_image(resting_image)
    frames = sample_video_fps(video_path, config.sampling_fps, config.max_frames)

    # Reuse client if provided, else create one
    own_client = client is None
    if own_client:
        client = create_client(config)

    system_prompt = prompt
    user_text = _build_user_text(prompt, region, task_name, landmark_evidence)

    evidence = client.generate(system_prompt, user_text, resting, frames)

    # Shared post-processing: ensure schema keys and context are present
    evidence.setdefault("region", region)
    evidence.setdefault("instructed_movement", task_name)
    for key in ("primary_findings", "secondary_findings"):
        evidence.setdefault(key, {})
    for key in ("visual_observations", "possible_associated_movements", "uncertain_findings"):
        evidence.setdefault(key, [])

    return evidence
