"""
LLM module — OpenAI API interaction functions.
"""

import json
import logging
import os
from pathlib import Path

from openai import OpenAI

logger = logging.getLogger(__name__)


def load_prompt(prompt_path: str) -> str:
    """Load a prompt from a text file.

    Args:
        prompt_path: Path to the prompt .txt file.

    Returns:
        Prompt string content.

    Raises:
        FileNotFoundError: If prompt file does not exist.
    """
    p = Path(prompt_path)
    if not p.exists():
        raise FileNotFoundError(f"Prompt file not found: {prompt_path}")
    content = p.read_text(encoding="utf-8")
    logger.info(f"Loaded prompt from: {prompt_path} ({len(content)} chars)")
    return content


def run_openai(
    prompt: str,
    payload: dict,
    model: str = "gpt-4o",
) -> tuple[dict, str]:
    """Send a request to the OpenAI API and parse the JSON response.

    Creates the OpenAI client internally using OPENAI_API_KEY env var.

    Args:
        prompt: System prompt string.
        payload: User payload dict.
        model: Model name to use.

    Returns:
        Tuple of (parsed JSON dict, raw response string).

    Raises:
        EnvironmentError: If OPENAI_API_KEY is not set.
        ValueError: If the API response is not valid JSON.
    """
    # Set your API key here
    api_key = ""

    if not api_key:
        # Fallback to environment variable if not set above
        api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        raise EnvironmentError("No API key provided. Set it in the code or via OPENAI_API_KEY env var.")

    client = OpenAI(api_key=api_key)
    logger.info(f"Sending request to OpenAI API (model={model})...")

    response = client.chat.completions.create(
        model=model,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "system",
                "content": prompt,
            },
            {
                "role": "user",
                "content": json.dumps(payload, ensure_ascii=False, indent=2),
            },
        ],
    )

    raw_content = response.choices[0].message.content
    logger.info("Received response from OpenAI API.")

    try:
        result = json.loads(raw_content)
    except json.JSONDecodeError as e:
        logger.error(f"API response is not valid JSON: {e}")
        logger.error(f"Raw response: {raw_content}")
        raise ValueError(f"Model did not return valid JSON: {e}") from e

    return result, raw_content
