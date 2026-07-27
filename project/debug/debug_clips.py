"""
Debug script — extracts per-task video clips from refined_tasks.json.

Usage:
    python debug/debug_clips.py \
        --video data/with_audio.mp4 \
        --tasks output_2/refined_tasks.json \
        --output output_2/clips
"""

import sys
import argparse
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from clip_extraction import extract_task_clips, extract_resting_image


def main() -> None:
    parser = argparse.ArgumentParser(description="Task video clip extraction")
    parser.add_argument("--video", type=str, default="C:/Users/lilia/Desktop/agentic_ai/test/pof_ver2/project/data/with_audio.mp4")
    parser.add_argument("--tasks", type=str, default="C:/Users/lilia/Desktop/agentic_ai/test/pof_ver2/output_2/refined_tasks.json")
    parser.add_argument("--output", type=str, default="output_2/clips")
    parser.add_argument("--pre", type=float, default=0.5, help="Seconds before onset")
    parser.add_argument("--post", type=float, default=0.5, help="Seconds after offset")
    parser.add_argument("--ffmpeg", type=str, default="ffmpeg", help="Path to ffmpeg executable")
    parser.add_argument("--features", type=str, default="output_2/debug/features.csv",
                        help="Path to features.csv (for resting frame detection)")
    parser.add_argument("--fps", type=float, default=30.0)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    manifest = extract_task_clips(
        video_path=args.video,
        refined_tasks_path=args.tasks,
        output_dir=args.output,
        pre_pad_s=args.pre,
        post_pad_s=args.post,
        ffmpeg_path=args.ffmpeg,
    )

    # Resting image — most stable moment BEFORE the first task begins
    resting = extract_resting_image(
        video_path=args.video,
        features_csv_path=args.features,
        output_path=str(Path(args.output) / "00_resting.png"),
        fps=args.fps,
        ffmpeg_path=args.ffmpeg,
        refined_tasks_path=args.tasks,
    )

    print(f"\nExtracted {len(manifest)} clips + 1 resting image → {args.output}\n")
    print(f"  Resting: frame {resting['frame']} ({resting['time']:.2f}s)  "
          f"{Path(resting['image_path']).name}")
    for c in manifest:
        print(f"  {c['task_id']:>3}. {c['task_name']:<25s} "
              f"[{c['clip_start']:.2f}s - {c['clip_end']:.2f}s]  {Path(c['clip_path']).name}")


if __name__ == "__main__":
    main()
