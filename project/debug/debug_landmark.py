"""
Debug script — runs only face landmark detection on a video.
"""

import sys
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from landmark import process_video_landmarks


def main() -> None:
    """Run landmark detection in isolation for debugging."""
    output_dir = "output/landmark"
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.FileHandler(f"{output_dir}/debug.log", mode="w", encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )

    result = process_video_landmarks(
        video_path="C:/Users/lilia/Desktop/agentic_ai/test/pof_ver2/project/data/with_audio.mp4",
        output_path=f"{output_dir}/result.mp4",
        model_path="C:/Users/lilia/Desktop/agentic_ai/test/pof_ver2/project/landmark/face_landmarker.task",
    )

    print(f"Frames: {result['frame_count']}")
    print(f"Avg: {result['avg_ms_per_frame']:.1f} ms/frame")
    print(f"Saved to: {result['output_path']}")


if __name__ == "__main__":
    main()
