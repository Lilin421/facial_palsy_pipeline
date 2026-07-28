"""
Abstract VLM client interface.

Every backend (GPT, Qwen, future models) implements only:
    - model initialization
    - message construction
    - inference
    - response parsing

Everything else (prompts, sampling, pipeline, output schema, post-processing)
is shared and lives outside the backend clients.
"""

import json
import logging
from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray

from .config import VLMConfig

logger = logging.getLogger(__name__)


class VLMClient(ABC):
    """Abstract base class for all VLM backends.

    Subclasses implement backend-specific model loading, message construction,
    inference, and response parsing. The public `generate()` method ties these
    together and is called by the shared inference layer.
    """

    def __init__(self, config: VLMConfig) -> None:
        """Initialize the client and load the model.

        Args:
            config: Shared VLM configuration.
        """
        self.config = config
        self._initialize_model()

    @abstractmethod
    def _initialize_model(self) -> None:
        """Load / initialize the backend model. Called once at construction."""
        raise NotImplementedError

    @abstractmethod
    def _build_messages(
        self,
        system_prompt: str,
        user_text: str,
        resting_image: NDArray[np.uint8],
        clip_frames: list[NDArray[np.uint8]],
    ) -> object:
        """Construct backend-specific model input.

        Args:
            system_prompt: Shared system prompt.
            user_text: Region/task prompt + landmark evidence.
            resting_image: Resting baseline frame (BGR).
            clip_frames: Sampled clip frames (BGR), temporal order.

        Returns:
            Backend-specific message payload consumed by `_infer`.
        """
        raise NotImplementedError

    @abstractmethod
    def _infer(self, messages: object) -> str:
        """Run inference and return the raw text response.

        Args:
            messages: Payload produced by `_build_messages`.

        Returns:
            Raw model output string (expected to contain JSON).
        """
        raise NotImplementedError

    def _parse_response(self, raw: str) -> dict:
        """Parse the raw model output into a JSON dict.

        Shared default implementation; backends may override if needed.

        Args:
            raw: Raw model text output.

        Returns:
            Parsed JSON dict.

        Raises:
            ValueError: If no valid JSON can be extracted.
        """
        # Try direct parse first
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            pass

        # Fallback: extract the first {...} block
        start = raw.find("{")
        end = raw.rfind("}")
        if start != -1 and end != -1 and end > start:
            snippet = raw[start:end + 1]
            try:
                return json.loads(snippet)
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON from response: {e}")
                logger.error(f"Raw response: {raw}")
                raise ValueError(f"Backend did not return valid JSON: {e}") from e

        raise ValueError(f"No JSON object found in response: {raw[:200]}")

    def generate(
        self,
        system_prompt: str,
        user_text: str,
        resting_image: NDArray[np.uint8],
        clip_frames: list[NDArray[np.uint8]],
    ) -> dict:
        """Full generation flow: build messages → infer → parse.

        Args:
            system_prompt: Shared system prompt.
            user_text: Region/task prompt + landmark evidence.
            resting_image: Resting baseline frame (BGR).
            clip_frames: Sampled clip frames (BGR), temporal order.

        Returns:
            Parsed JSON evidence dict.
        """
        messages = self._build_messages(
            system_prompt, user_text, resting_image, clip_frames
        )
        raw = self._infer(messages)
        return self._parse_response(raw)
