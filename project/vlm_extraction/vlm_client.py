"""
GPT Vision API client wrapper for clinical evidence extraction.

Reads the API key from the OPENAI_API_KEY environment variable.
NEVER hardcode API keys.
"""

import json
import logging
import os

from openai import OpenAI

from .config import VLMConfig

logger = logging.getLogger(__name__)


def get_client() -> OpenAI:
    """Create an OpenAI client using the OPENAI_API_KEY environment variable.

    Returns:
        OpenAI client.

    Raises:
        EnvironmentError: If OPENAI_API_KEY is not set.
    """
    # Set your API key here
    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        # Fallback to environment variable if not set above
        api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        raise EnvironmentError(
            "No API key provided. Set it in the code above or via OPENAI_API_KEY env var."
        )
    return OpenAI(api_key=api_key)


def call_vision(
    client: OpenAI,
    system_prompt: str,
    content_blocks: list[dict],
    config: VLMConfig,
) -> dict:
    """Send a vision request with pre-built content blocks and parse the JSON.

    Args:
        client: OpenAI client.
        system_prompt: Shared system prompt.
        content_blocks: Interleaved list of {"type": "text"/"image_url", ...} blocks.
        config: VLM configuration.

    Returns:
        Parsed JSON dict of extracted evidence.

    Raises:
        ValueError: If the response is not valid JSON.
    """
    content = content_blocks
    n_images = sum(1 for b in content if b.get("type") == "image_url")
    logger.info(f"Calling {config.model} with {n_images} images...")

    response = client.chat.completions.create(
        model=config.model,
        temperature=config.temperature,
        max_tokens=config.max_tokens,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content},
        ],
    )

    raw = response.choices[0].message.content
    try:
        return json.loads(raw)
    except json.JSONDecodeError as e:
        logger.error(f"VLM response is not valid JSON: {e}")
        logger.error(f"Raw response: {raw}")
        raise ValueError(f"VLM did not return valid JSON: {e}") from e
