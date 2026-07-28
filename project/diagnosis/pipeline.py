"""
Facial palsy diagnosis pipeline.

Loads VLM visual evidence + landmark-derived clinical evidence, combines them,
and produces a House-Brackmann grade via a GPT reasoning model.
"""

import json
import logging
from pathlib import Path

from .config import DiagnosisConfig
from .gpt_reasoner import get_client, run_diagnosis

logger = logging.getLogger(__name__)

_PROMPT_FILE = "C:/Users/lilia/Desktop/agentic_ai/test/pof_ver2/project/diagnosis/prompts/hb_diagnosis.txt"


def _load_prompt(prompt_file: str = _PROMPT_FILE) -> str:
    """Load the HB diagnosis prompt."""
    p = Path(prompt_file)
    if not p.exists():
        raise FileNotFoundError(f"Prompt file not found: {prompt_file}")
    return p.read_text(encoding="utf-8")


def _load_json(path: str) -> object:
    """Load a JSON file."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {path}")
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_vlm_evidence(vlm_dir: str) -> dict:
    """Load all *_evidence.json files from the VLM evidence directory.

    Args:
        vlm_dir: Directory containing regional VLM evidence JSON files.

    Returns:
        Dict mapping filename stem → evidence content.
    """
    evidence = {}
    d = Path(vlm_dir)
    if not d.exists():
        raise FileNotFoundError(f"VLM evidence directory not found: {vlm_dir}")

    for f in sorted(d.glob("*_evidence.json")):
        evidence[f.stem] = _load_json(str(f))
    if not evidence:
        logger.warning(f"No *_evidence.json files found in {vlm_dir}")
    return evidence


def _build_user_text(vlm_evidence: dict, clinical_evidence: object) -> str:
    """Assemble the combined evidence into the reasoning prompt input.

    Args:
        vlm_evidence: VLM regional evidence.
        clinical_evidence: Landmark-derived clinical evidence.

    Returns:
        Composed user text.
    """
    return (
        "=== VLM VISUAL EVIDENCE (qualitative, per region) ===\n"
        f"{json.dumps(vlm_evidence, ensure_ascii=False, indent=2)}\n\n"
        "=== LANDMARK-DERIVED CLINICAL EVIDENCE (quantitative, per task) ===\n"
        f"{json.dumps(clinical_evidence, ensure_ascii=False, indent=2)}\n\n"
        "Integrate both sources. Think twice about whether each feature truly "
        "indicates weakness, asymmetry, or synkinesis, or whether it is within "
        "normal physiological variation. Then assign the House-Brackmann grade."
    )


def diagnose_facial_palsy(
    vlm_evidence_dir: str,
    clinical_evidence_path: str,
    output_path: str,
    config: DiagnosisConfig | None = None,
) -> dict:
    """Produce an HB-grade diagnosis from VLM + landmark evidence.

    Args:
        vlm_evidence_dir: Directory with regional VLM evidence JSON files.
        clinical_evidence_path: Path to clinical_evidence.json.
        output_path: Path to write the diagnosis JSON.
        config: Optional diagnosis configuration.

    Returns:
        Diagnosis dict.
    """
    if config is None:
        config = DiagnosisConfig()

    prompt = _load_prompt()
    vlm_evidence = _load_vlm_evidence(vlm_evidence_dir)
    clinical_evidence = _load_json(clinical_evidence_path)

    user_text = _build_user_text(vlm_evidence, clinical_evidence)

    client = get_client()
    diagnosis = run_diagnosis(client, prompt, user_text, config)

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(diagnosis, f, indent=2, ensure_ascii=False)

    logger.info(f"Wrote diagnosis: {output_path}")
    return diagnosis
