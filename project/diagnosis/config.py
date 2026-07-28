"""
Configuration for the facial palsy diagnosis (HB grading) module.
"""

from dataclasses import dataclass


@dataclass
class DiagnosisConfig:
    """Parameters for the HB-grading reasoning model.

    Attributes:
        model: OpenAI model name for reasoning.
        temperature: Sampling temperature (low for consistent reasoning).
        max_tokens: Max tokens in the response.
    """

    model: str = "gpt-4o"
    temperature: float = 0.0
    max_tokens: int = 2000
