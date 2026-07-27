"""
Debug script — extracts clinical evidence from refined tasks + features.

Usage:
    python debug/debug_clinical_evidence.py \
        --tasks output_2/refined_tasks.json \
        --features output_2/debug/features.csv \
        --output output_2/clinical_evidence.json
"""

import sys
import argparse
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from clinical_evidence import extract_clinical_evidence, EvidenceConfig


def main() -> None:
    parser = argparse.ArgumentParser(description="Clinical evidence extraction debug")
    parser.add_argument("--tasks", type=str, default="output_2/refined_tasks.json")
    parser.add_argument("--features", type=str, default="output_2/debug/features.csv")
    parser.add_argument("--output", type=str, default="output_2/clinical_evidence.json")
    parser.add_argument("--fps", type=float, default=30.0)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    config = EvidenceConfig(fps=args.fps)

    evidence = extract_clinical_evidence(
        refined_tasks_path=args.tasks,
        features_csv_path=args.features,
        output_path=args.output,
        config=config,
    )

    print(f"\nExtracted evidence for {len(evidence)} tasks → {args.output}\n")
    for e in evidence:
        n_assoc = len(e.get("possible_associated_movements", []))
        n_asym = len(e.get("possible_asymmetry_regions", []))
        primary = e.get("primary_summary", {}).get("primary_feature", "n/a")
        n_regions = len(e.get("symmetry", {}).get("regions", {}))
        print(f"  {e.get('task_id'):>3}. {e['task_name']:<25s} "
              f"primary={primary:<15s} regions={n_regions} "
              f"asymmetry={n_asym} associated={n_assoc}")


if __name__ == "__main__":
    main()
