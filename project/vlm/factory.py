"""
Backend factory — selects the VLM client based on configuration.

Adding a new backend (InternVL, GLM, MiniCPM, LLaVA, ...) requires only:
    1. Implement a VLMClient subclass under clients/.
    2. Register it in the _BACKENDS mapping below.
No upper-level code changes.
"""

import logging

from .base import VLMClient
from .config import VLMConfig

logger = logging.getLogger(__name__)


def create_client(config: VLMConfig) -> VLMClient:
    """Instantiate the VLM client for the configured backend.

    Args:
        config: VLM configuration (config.model selects the backend).

    Returns:
        An initialized VLMClient subclass.

    Raises:
        ValueError: If the configured backend is unknown.
    """
    backend = config.model.lower().strip()

    # Lazy imports so a backend's heavy deps are only loaded when selected.
    if backend == "gpt":
        from .clients.gpt_client import GPTClient
        return GPTClient(config)

    if backend == "qwen":
        from .clients.qwen_client import QwenClient
        return QwenClient(config)

    raise ValueError(
        f"Unknown VLM backend: '{config.model}'. Supported: 'gpt', 'qwen'."
    )
