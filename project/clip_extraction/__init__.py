"""
Video clip extraction package.

Extracts per-task video clips from refined task boundaries.
"""

from .clipper import extract_task_clips, extract_resting_image, find_resting_frame

__all__ = ["extract_task_clips", "extract_resting_image", "find_resting_frame"]
