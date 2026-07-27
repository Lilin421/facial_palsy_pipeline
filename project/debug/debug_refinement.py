"""
Debug script — runs temporal refinement on task.json + features.csv.

Usage:
    python debug/debug_refinement.py --tasks output_2/task.json --features output_2/debug/features.csv
"""

import sys
import json
import argparse
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from temporal_refinement import refine_all_tasks, RefinementConfig


def main() -> None:
    parser = argparse.ArgumentParser(description="Temporal refinement debug")
    parser.add_argument("--tasks", type=str, default="C:/Users/lilia/Desktop/agentic_ai/test/pof_ver2/output_2/task.json",
                        help="Path to task.json")
    parser.add_argument("--features", type=str, default="C:/Users/lilia/Desktop/agentic_ai/test/pof_ver2/output_2/debug/features.csv",
                        help="Path to features.csv")
    parser.add_argument("--fps", type=float, default=30.0,
                        help="Video FPS")
    parser.add_argument("--output", type=str, default="C:/Users/lilia/Desktop/agentic_ai/test/pof_ver2/output_2/refined_tasks.json",
                        help="Output path for refined annotations")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    config = RefinementConfig(fps=args.fps)

    results = refine_all_tasks(
        tasks_json_path=args.tasks,
        features_csv_path=args.features,
        config=config,
    )

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nRefined {len(results)} tasks → {output_path}")
    print()
    for r in results:
        print(f"  {r['task_id']:2d}. {r['task_name']:<25s} conf={r.get('confidence', '?')}")


if __name__ == "__main__":
    main()
