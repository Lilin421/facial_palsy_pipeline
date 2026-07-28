"""
Backend-independent VLM feature extraction package.

Supports multiple VLM backends (GPT-4o, Qwen2.5-VL, ...) through a common
interface. The upper pipeline never knows which backend is in use — switching
backends only requires changing one configuration parameter.

This module extracts structured visual clinical findings only.
It does NOT diagnose or grade.
"""

from .config import VLMConfig
from .inference import extract_features
from .factory import create_client

__all__ = ["VLMConfig", "extract_features", "create_client"]
