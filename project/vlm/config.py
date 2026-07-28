"""
Configuration for the backend-independent VLM module.

The backend is selected purely by the `model` field ("gpt" or "qwen").
No other code needs to change to switch backends.
"""

from dataclasses import dataclass, field


@dataclass
class VLMConfig:
    """Configuration shared across all VLM backends.

    Attributes:
        model: Backend selector — "gpt" or "qwen".
        gpt_model_name: Model name for the GPT backend.
        qwen_model_path: Model path/name for the Qwen backend.
        temperature: Sampling temperature (0 for deterministic extraction).
        max_tokens: Max tokens in the response.
        sampling_fps: Frames-per-second used to sample the task video.
            Defaults to 5 fps for open-source VLM experiments.
        max_frames: Hard cap on frames sampled from a single clip.
        image_detail: Detail level for image-based backends ("low"/"high"/"auto").
        prompt_file: Path to the shared, backend-independent prompt file.
        device: Device for local backends (e.g. "cuda", "cpu").
    """

    model: str = "gpt"

    # Backend model identifiers
    gpt_model_name: str = "gpt-4o"
    qwen_model_path: str = "Qwen/Qwen2.5-VL-32B-Instruct"

    # Inference params (shared)
    temperature: float = 0.0
    max_tokens: int = 1500

    # Video sampling (shared, configurable)
    sampling_fps: float = 5.0
    max_frames: int = 32

    # Image backend options
    image_detail: str = "high"

    # Shared prompt
    prompt_file: str = "vlm/prompts/clinical_extraction.txt"

    # Local backend options
    device: str = "cuda"
