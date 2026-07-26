"""
Utility module — file I/O helpers.
"""

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def save_json(data: Any, output_path: str) -> None:
    """Save data as a JSON file.

    Args:
        data: Data to serialize.
        output_path: Destination file path.
    """
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved JSON: {output_path}")


def save_text(text: str, output_path: str) -> None:
    """Save text content to a file.

    Args:
        text: Text content to write.
        output_path: Destination file path.
    """
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        f.write(text)
    logger.info(f"Saved text: {output_path}")


def ensure_output_dir(output_dir: str) -> Path:
    """Ensure the output directory exists.

    Args:
        output_dir: Path to output directory.

    Returns:
        Path object for the output directory.
    """
    p = Path(output_dir)
    p.mkdir(parents=True, exist_ok=True)
    return p
