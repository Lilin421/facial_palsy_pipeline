"""
VLM-based facial clinical evidence extraction package.

Uses the GPT Vision API to extract structured visual clinical findings from
facial movement clips, organized into regional experts.

This module extracts visual evidence only. It does NOT diagnose or grade.
"""

from .config import VLMConfig
from .pipeline import run_all_experts

__all__ = ["VLMConfig", "run_all_experts"]
