"""
Debug script — runs the unified VLM extraction over the regional experts.

Switch backend with --model gpt|qwen. No other change required.

Usage:
    python vlm/debug/debug_vlm_unified.py --model gpt
    python vlm/debug/debug_vlm_unified.py --model qwen
"""

import sys
import json
import argparse
import logging
import re
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from vlm import VLMConfig, extract_features, create_client


# Region → (clip task names) mapping for the regional experts.
REGION_TASKS: dict[str, list[str]] = {
    "temporal": ["raise eyebrow"],
    "zygomatic": ["gentle eye closure", "close eye", "tightly close eye",
                  "tight eye closure", "blink", "blink repeatedly"],
    "buccal": ["smile", "big smile", "puff cheek", "puff cheeks", "puff up cheeks",
               "raise upper lip", "raise top lip", "screw nose", "screw up nose",
               "blow kiss", "blow kisses", "blow kisses for 3 times"],
    "marginal_mandibular": ["lower bottom lip"],
    "cervical": ["angry neck"],
}


def _sanitize(name: str) -> str:
    cleaned = re.sub(r"[^\w\s-]", "", name).strip().lower()
    return re.sub(r"\s+", "_", cleaned)


def _label_from_path(clip_path: str) -> str:
    stem = re.sub(r"^\d+_", "", Path(clip_path).stem)
    return stem.replace("_", " ").title()


def main() -> None:
    parser = argparse.ArgumentParser(description="Unified VLM extraction")
    parser.add_argument("--model", type=str, default="gpt", help="Backend: gpt|qwen")
    parser.add_argument("--resting", type=str, default="output_2/clips/00_resting.png")
    parser.add_argument("--clips", type=str, default="output_2/clips")
    parser.add_argument("--evidence", type=str, default="output_2/clinical_evidence.json")
    parser.add_argument("--output", type=str, default="vlm/outputs")
    parser.add_argument("--fps", type=float, default=5.0)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    config = VLMConfig(model=args.model, sampling_fps=args.fps)

    # Load landmark evidence
    with open(args.evidence, "r", encoding="utf-8") as f:
        clinical = json.load(f)
    if isinstance(clinical, dict):
        clinical = clinical.get("tasks", [])
    evidence_by_task = {_sanitize(e.get("task_name", "")): e for e in clinical}

    # Build client once and reuse (avoids reloading local models per clip)
    client = create_client(config)

    Path(args.output).mkdir(parents=True, exist_ok=True)
    clip_files = list(Path(args.clips).glob("*.mp4"))

    for region, task_names in REGION_TASKS.items():
        wanted = {_sanitize(t) for t in task_names}
        region_results = []

        for clip in clip_files:
            stem = _sanitize(re.sub(r"^\d+_", "", clip.stem))
            if not any(w in stem or stem in w for w in wanted):
                continue

            task_label = _label_from_path(str(clip))
            landmark = evidence_by_task.get(stem, {})

            result = extract_features(
                region=region,
                task_name=task_label,
                video_path=str(clip),
                resting_image=args.resting,
                landmark_evidence=landmark,
                config=config,
                client=client,
            )
            region_results.append(result)
            print(f"  [{region}] {task_label}: "
                  f"{len(result.get('visual_observations', []))} observations")

        out_path = Path(args.output) / f"{region}_evidence.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(region_results, f, indent=2, ensure_ascii=False)
        print(f"  → wrote {out_path}\n")


if __name__ == "__main__":
    main()
