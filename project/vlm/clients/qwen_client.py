"""
Qwen2.5-VL backend client (local HuggingFace inference).

Implements only: model initialization, message construction, inference,
and response parsing (inherits shared JSON parsing).

Requires:
    transformers (with Qwen2.5-VL support)
    qwen-vl-utils
    torch
"""

import logging

import numpy as np
from numpy.typing import NDArray

from ..base import VLMClient
from ..config import VLMConfig
from ..sampling import bgr_to_rgb

logger = logging.getLogger(__name__)


class QwenClient(VLMClient):
    """Qwen2.5-VL local backend."""

    def _initialize_model(self) -> None:
        """Load the Qwen2.5-VL model and processor."""
        import torch
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        model_path = self.config.qwen_model_path
        logger.info(f"Loading Qwen2.5-VL from {model_path} ...")

        self._torch = torch
        self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map=self.config.device,
        )
        self._processor = AutoProcessor.from_pretrained(model_path)
        logger.info("Qwen2.5-VL initialized.")

    def _build_messages(
        self,
        system_prompt: str,
        user_text: str,
        resting_image: NDArray[np.uint8],
        clip_frames: list[NDArray[np.uint8]],
    ) -> dict:
        """Build Qwen chat messages with PIL images embedded.

        Returns a dict carrying both the chat structure and the raw images,
        consumed by `_infer`.
        """
        from PIL import Image

        resting_pil = Image.fromarray(bgr_to_rgb(resting_image))
        frame_pils = [Image.fromarray(bgr_to_rgb(f)) for f in clip_frames]

        # Interleave labels and images in the user content
        user_content: list[dict] = [{"type": "text", "text": user_text}]
        user_content.append({"type": "text", "text": "=== RESTING baseline ==="})
        user_content.append({"type": "image", "image": resting_pil})
        user_content.append({
            "type": "text",
            "text": f"=== Instructed movement clip ({len(frame_pils)} frames, temporal order) ===",
        })
        for fp in frame_pils:
            user_content.append({"type": "image", "image": fp})

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]

        images = [resting_pil] + frame_pils
        return {"messages": messages, "images": images}

    def _infer(self, messages: dict) -> str:
        """Run inference via the Qwen2.5-VL model."""
        chat = messages["messages"]
        images = messages["images"]

        text = self._processor.apply_chat_template(
            chat, tokenize=False, add_generation_prompt=True
        )
        inputs = self._processor(
            text=[text],
            images=images,
            padding=True,
            return_tensors="pt",
        ).to(self._model.device)

        with self._torch.no_grad():
            generated = self._model.generate(
                **inputs,
                max_new_tokens=self.config.max_tokens,
                do_sample=self.config.temperature > 0.0,
                temperature=max(self.config.temperature, 1e-6),
            )

        # Strip the prompt tokens
        trimmed = generated[:, inputs.input_ids.shape[1]:]
        output = self._processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        return output
