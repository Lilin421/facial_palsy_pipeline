"""
GPT reasoning client for HB grading.

API key is read from the OPENAI_API_KEY environment variable.
NEVER hardcode API keys.
"""

import json
import logging
import os

from openai import OpenAI

from .config import DiagnosisConfig

logger = logging.getLogger(__name__)


def get_client() -> OpenAI:
    """Create an OpenAI client using OPENAI_API_KEY.

    Returns:
        OpenAI client.

    Raises:
        EnvironmentError: If OPENAI_API_KEY is not set.
    """
    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY environment variable is not set.")
    return OpenAI(api_key=api_key)


def run_diagnosis(
    client: OpenAI,
    system_prompt: str,
    user_text: str,
    config: DiagnosisConfig,
) -> dict:
    """Send the diagnosis request and parse the JSON response.

    Args:
        client: OpenAI client.
        system_prompt: HB grading system prompt.
        user_text: Combined VLM + landmark evidence text.
        config: Diagnosis configuration.

    Returns:
        Parsed JSON diagnosis dict.

    Raises:
        ValueError: If the response is not valid JSON.
    """
    logger.info(f"Running HB diagnosis with {config.model}...")
    response = client.chat.completions.create(
        model=config.model,
        temperature=config.temperature,
        max_tokens=config.max_tokens,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_text},
        ],
    )
    raw = response.choices[0].message.content
    try:
        return json.loads(raw)
    except json.JSONDecodeError as e:
        logger.error(f"Diagnosis response is not valid JSON: {e}")
        logger.error(f"Raw response: {raw}")
        raise ValueError(f"Reasoner did not return valid JSON: {e}") from e
