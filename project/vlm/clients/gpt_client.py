"""
GPT backend client (OpenAI vision models).

Implements only: model initialization, message construction, inference.
API key is read from the OPENAI_API_KEY environment variable.
"""

import logging

import numpy as np
from numpy.typing import NDArray

from ..base import VLMClient
from ..config import VLMConfig
from ..sampling import encode_bgr_to_data_url

logger = logging.getLogger(__name__)


class GPTClient(VLMClient):
    """OpenAI GPT vision backend."""

    def _initialize_model(self) -> None:
        """Initialize the OpenAI client from OPENAI_API_KEY."""
        import os
        from openai import OpenAI

        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "OPENAI_API_KEY environment variable is not set."
            )
        self._client = OpenAI(api_key=api_key)
        logger.info(f"GPT backend initialized (model={self.config.gpt_model_name}).")

    def _build_messages(
        self,
        system_prompt: str,
        user_text: str,
        resting_image: NDArray[np.uint8],
        clip_frames: list[NDArray[np.uint8]],
    ) -> list[dict]:
        """Build OpenAI chat messages with interleaved labeled images."""
        detail = self.config.image_detail

        content: list[dict] = [{"type": "text", "text": user_text}]

        content.append({"type": "text", "text": "=== RESTING baseline ==="})
        content.append({
            "type": "image_url",
            "image_url": {"url": encode_bgr_to_data_url(resting_image), "detail": detail},
        })

        content.append({
            "type": "text",
            "text": f"=== Instructed movement clip ({len(clip_frames)} frames, temporal order) ===",
        })
        for frame in clip_frames:
            content.append({
                "type": "image_url",
                "image_url": {"url": encode_bgr_to_data_url(frame), "detail": detail},
            })

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content},
        ]

    def _infer(self, messages: list[dict]) -> str:
        """Run inference via the OpenAI chat completions API."""
        response = self._client.chat.completions.create(
            model=self.config.gpt_model_name,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            response_format={"type": "json_object"},
            messages=messages,
        )
        return response.choices[0].message.content
