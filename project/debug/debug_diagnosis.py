"""
Debug script — runs HB-grade facial palsy diagnosis.

Combines VLM evidence + landmark clinical evidence.
Requires OPENAI_API_KEY environment variable.

Usage:
    python debug/debug_diagnosis.py \
        --vlm output_2/vlm_evidence \
        --evidence output_2/clinical_evidence.json \
        --output output_2/diagnosis.json
"""

import sys
import json
import argparse
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from diagnosis import diagnose_facial_palsy, DiagnosisConfig


def main() -> None:
    parser = argparse.ArgumentParser(description="Facial palsy HB diagnosis")
    parser.add_argument("--vlm", type=str, default="output_2/vlm_evidence",
                        help="Directory with regional VLM evidence JSON files")
    parser.add_argument("--evidence", type=str, default="output_2/clinical_evidence.json")
    parser.add_argument("--output", type=str, default="output_2/diagnosis.json")
    parser.add_argument("--model", type=str, default="gpt-4o")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    config = DiagnosisConfig(model=args.model)

    diagnosis = diagnose_facial_palsy(
        vlm_evidence_dir=args.vlm,
        clinical_evidence_path=args.evidence,
        output_path=args.output,
        config=config,
    )

    print(f"\nDiagnosis → {args.output}\n")
    print(f"  HB Grade:      {diagnosis.get('hb_grade')}")
    print(f"  Affected side: {diagnosis.get('affected_side')}")
    print(f"  Confidence:    {diagnosis.get('confidence')}")
    print(f"\n  Summary: {diagnosis.get('summary', '')}\n")


if __name__ == "__main__":
    main()
