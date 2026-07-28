"""
Configuration for the VLM feature extraction pipeline.
"""

from dataclasses import dataclass


@dataclass
class VLMConfig:
    """Parameters for VLM clinical evidence extraction.

    Attributes:
        model: OpenAI vision-capable model name.
        temperature: Sampling temperature (0 for deterministic extraction).
        frames_per_clip: Number of frames sampled per clip for the VLM.
        image_detail: Image detail level for the vision API ("low"/"high"/"auto").
        max_tokens: Max tokens in the response.
        fps: Frame rate of source clips (for frame sampling).
    """

    model: str = "gpt-4o"
    temperature: float = 0.0
    frames_per_clip: int = 4
    dynamic_frames_per_clip: int = 8
    image_detail: str = "high"
    max_tokens: int = 1500
    fps: float = 10.0
