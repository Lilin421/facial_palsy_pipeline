"""
Debug script — runs the complete pipeline end-to-end.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline import setup_logging, process_audio


def main() -> None:
    """Run the full pipeline for debugging."""
    setup_logging("output/audio/debug.log")
    status = process_audio()
    print(f"\nPipeline status: {status}")


if __name__ == "__main__":
    main()
