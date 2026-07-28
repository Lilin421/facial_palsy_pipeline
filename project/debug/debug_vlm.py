"""
Debug script — runs the VLM regional experts to extract clinical evidence.

Requires OPENAI_API_KEY environment variable.

Usage:
    set OPENAI_API_KEY=sk-...        (Windows CMD)
    $env:OPENAI_API_KEY="sk-..."     (PowerShell)

    python debug/debug_vlm.py \
        --resting output_2/clips/00_resting.png \
        --clips output_2/clips \
        --evidence output_2/clinical_evidence.json \
        --output output_2/vlm_evidence
"""

import sys
import argparse
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vlm_extraction import run_all_experts, VLMConfig


def main() -> None:
    parser = argparse.ArgumentParser(description="VLM regional expert extraction")
    parser.add_argument("--resting", type=str, default="C:/Users/lilia/Desktop/agentic_ai/test/pof_ver2/output_2/clips/00_resting.png")
    parser.add_argument("--clips", type=str, default="C:/Users/lilia/Desktop/agentic_ai/test/pof_ver2/output_2/clips")
    parser.add_argument("--evidence", type=str, default="C:/Users/lilia/Desktop/agentic_ai/test/pof_ver2/output_2/clinical_evidence.json")
    parser.add_argument("--output", type=str, default="C:/Users/lilia/Desktop/agentic_ai/test/pof_ver2/output_2/vlm_evidence")
    parser.add_argument("--model", type=str, default="gpt-4o")
    parser.add_argument("--frames", type=int, default=4, help="Frames sampled per clip")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    config = VLMConfig(model=args.model, frames_per_clip=args.frames)

    results = run_all_experts(
        resting_image_path=args.resting,
        clips_dir=args.clips,
        clinical_evidence_path=args.evidence,
        output_dir=args.output,
        config=config,
    )

    print(f"\nExtracted evidence from {len(results)} experts → {args.output}\n")
    for name, ev in results.items():
        n_obs = len(ev.get("visual_observations", []))
        n_assoc = len(ev.get("possible_associated_movements", []))
        err = " [ERROR]" if "error" in ev else ""
        print(f"  {name:<22s} observations={n_obs} associated={n_assoc}{err}")


if __name__ == "__main__":
    main()
