"""
VLM extraction pipeline — runs each regional expert and writes evidence JSONs.

For each expert:
    1. Gather resting image + relevant clip frames.
    2. Attach landmark clinical evidence (as guidance).
    3. Call GPT Vision to extract structured visual findings.
    4. Save <expert>_evidence.json.
"""

import json
import logging
import re
from pathlib import Path

from .config import VLMConfig
from .experts import EXPERTS, Expert, SHARED_SYSTEM_PROMPT
from .frame_sampler import sample_clip_frames, encode_image_bgr, encode_image_file
from .vlm_client import get_client, call_vision

logger = logging.getLogger(__name__)


def _sanitize(name: str) -> str:
    """Normalize a task name for matching (lowercase, underscores)."""
    cleaned = re.sub(r"[^\w\s-]", "", name).strip().lower()
    return re.sub(r"\s+", "_", cleaned)


def _label_from_path(clip_path: str) -> str:
    """Derive a human-readable task label from a clip filename.

    Example: '03_tightly_close_eye.mp4' -> 'Tightly Close Eye'.

    Args:
        clip_path: Path to the clip.

    Returns:
        Readable task label.
    """
    stem = Path(clip_path).stem
    # Strip leading numeric id prefix like "03_"
    stem = re.sub(r"^\d+_", "", stem)
    return stem.replace("_", " ").title()


def _load_clip_manifest(clips_dir: str) -> list[dict]:
    """Load the clips manifest produced by clip_extraction.

    Args:
        clips_dir: Directory containing clips and clips_manifest.json.

    Returns:
        List of clip entries.
    """
    manifest_path = Path(clips_dir) / "clips_manifest.json"
    if not manifest_path.exists():
        logger.warning(f"No clips_manifest.json in {clips_dir}; matching by filename.")
        return []
    with open(manifest_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _find_clips_for_expert(
    expert: Expert,
    manifest: list[dict],
    clips_dir: str,
) -> list[str]:
    """Resolve clip file paths for an expert based on its task list.

    Args:
        expert: The expert definition.
        manifest: Clip manifest entries.
        clips_dir: Clips directory (fallback for filename matching).

    Returns:
        List of clip file paths relevant to this expert.
    """
    wanted = {_sanitize(t) for t in expert.clip_tasks}
    paths: list[str] = []

    if manifest:
        for entry in manifest:
            if _sanitize(entry.get("task_name", "")) in wanted:
                # Re-resolve against the local clips_dir so stale absolute paths
                # (e.g. from a previous Docker run) don't break loading.
                fname = Path(entry["clip_path"]).name
                local_path = Path(clips_dir) / fname
                if local_path.exists():
                    paths.append(str(local_path))
                elif Path(entry["clip_path"]).exists():
                    paths.append(entry["clip_path"])
    else:
        # Fallback: match by filename fragments
        for f in Path(clips_dir).glob("*.mp4"):
            stem = _sanitize(f.stem)
            if any(w in stem for w in wanted):
                paths.append(str(f))

    return paths


def _filter_evidence_for_expert(
    expert: Expert,
    clinical_evidence: list[dict],
) -> list[dict]:
    """Select clinical-evidence entries relevant to the expert's clips.

    Args:
        expert: Expert definition.
        clinical_evidence: Full clinical evidence list.

    Returns:
        Filtered evidence entries (task-name match).
    """
    wanted = {_sanitize(t) for t in expert.clip_tasks}
    return [
        e for e in clinical_evidence
        if _sanitize(e.get("task_name", "")) in wanted
    ]


def run_expert(
    expert: Expert,
    resting_image_path: str,
    clips_dir: str,
    clinical_evidence: list[dict],
    manifest: list[dict],
    client,
    config: VLMConfig,
) -> dict:
    """Run a single regional expert and return its extracted evidence.

    Args:
        expert: Expert definition.
        resting_image_path: Path to the resting baseline image.
        clips_dir: Directory of task clips.
        clinical_evidence: Full landmark-derived clinical evidence.
        manifest: Clip manifest.
        client: OpenAI client.
        config: VLM configuration.

    Returns:
        Extracted evidence dict.
    """
    # Gather images: resting baseline first
    clip_paths = _find_clips_for_expert(expert, manifest, clips_dir)
    if not clip_paths:
        logger.warning(f"[{expert.name}] no clips found; skipping.")
        return {
            "region": expert.region,
            "primary_findings": {},
            "secondary_findings": {},
            "visual_observations": [],
            "possible_associated_movements": [],
            "uncertain_findings": ["No clips available for this expert."],
        }

    # Landmark evidence relevant to this expert
    expert_evidence = _filter_evidence_for_expert(expert, clinical_evidence)

    # Build interleaved content: prompt text, resting image (labeled),
    # then each clip's frames preceded by a label naming the task.
    detail = config.image_detail
    content: list[dict] = [{
        "type": "text",
        "text": (
            f"{expert.prompt}\n\n"
            f"--- Landmark-derived clinical evidence (guidance only) ---\n"
            f"{json.dumps(expert_evidence, ensure_ascii=False, indent=2)}\n\n"
            f"You will receive labeled image groups. The RESTING baseline is "
            f"first, then each movement clip is introduced by a text label "
            f"naming the task, followed by its frames in temporal order. "
            f"Analyze each labeled task separately."
        ),
    }]

    # Resting baseline (labeled)
    content.append({"type": "text", "text": "=== RESTING baseline ==="})
    content.append({
        "type": "image_url",
        "image_url": {"url": encode_image_file(resting_image_path), "detail": detail},
    })

    # Each clip, labeled with its task name
    dynamic_keywords = ("blink", "blow")
    for cp in clip_paths:
        task_label = _label_from_path(cp)
        # Dynamic movements (blink, blow kiss) need denser sampling —
        # 4 stills can miss a fast blink entirely.
        is_dynamic = any(k in task_label.lower() for k in dynamic_keywords)
        n = config.dynamic_frames_per_clip if is_dynamic else config.frames_per_clip
        frames = sample_clip_frames(cp, n)
        content.append({
            "type": "text",
            "text": f"=== Task clip: {task_label} ({len(frames)} frames, temporal order) ===",
        })
        for frame in frames:
            content.append({
                "type": "image_url",
                "image_url": {"url": encode_image_bgr(frame), "detail": detail},
            })

    evidence = call_vision(client, SHARED_SYSTEM_PROMPT, content, config)
    evidence.setdefault("region", expert.region)
    return evidence


def run_all_experts(
    resting_image_path: str,
    clips_dir: str,
    clinical_evidence_path: str,
    output_dir: str,
    config: VLMConfig | None = None,
) -> dict[str, dict]:
    """Run all regional experts and write their evidence JSON files.

    Args:
        resting_image_path: Path to resting baseline image.
        clips_dir: Directory containing task clips + manifest.
        clinical_evidence_path: Path to clinical_evidence.json.
        output_dir: Directory to write expert evidence files.
        config: Optional VLM configuration.

    Returns:
        Dict mapping expert name → evidence dict.
    """
    if config is None:
        config = VLMConfig()

    with open(clinical_evidence_path, "r", encoding="utf-8") as f:
        clinical_evidence = json.load(f)
    if isinstance(clinical_evidence, dict):
        clinical_evidence = clinical_evidence.get("tasks", [])

    manifest = _load_clip_manifest(clips_dir)
    client = get_client()

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    results: dict[str, dict] = {}
    for expert in EXPERTS:
        logger.info(f"Running expert: {expert.name}")
        try:
            evidence = run_expert(
                expert, resting_image_path, clips_dir,
                clinical_evidence, manifest, client, config
            )
        except Exception as e:
            logger.error(f"[{expert.name}] failed: {e}")
            evidence = {
                "region": expert.region,
                "error": str(e),
                "primary_findings": {},
                "secondary_findings": {},
                "visual_observations": [],
                "possible_associated_movements": [],
                "uncertain_findings": [],
            }

        out_path = Path(output_dir) / expert.output_file
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(evidence, f, indent=2, ensure_ascii=False)
        logger.info(f"[{expert.name}] wrote {out_path}")

        results[expert.name] = evidence

    return results
