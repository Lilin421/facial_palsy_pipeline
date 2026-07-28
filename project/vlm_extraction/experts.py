"""
Regional expert definitions: prompts, clip mappings, and output filenames.

Each expert analyzes a specific facial region using the resting baseline image,
one or more task clips, and the landmark-derived clinical evidence.
"""

from dataclasses import dataclass, field


# Shared system prompt for all experts (extraction only, no diagnosis).
SHARED_SYSTEM_PROMPT = """You are an expert facial nerve specialist.
Your task is ONLY to extract objective visual clinical findings.

Do NOT diagnose facial palsy.
Do NOT estimate disease severity.
Do NOT assign House-Brackmann, Sunnybrook, Sydney or eFACE scores.

Use the resting image as the patient's baseline.
The supplied landmark evidence is quantitative guidance.
Do not simply repeat landmark measurements.
Instead, identify visual findings that complement or verify the landmark evidence.

Return structured JSON only, matching this schema exactly:
{
  "region": "...",
  "primary_findings": {},
  "secondary_findings": {},
  "visual_observations": [],
  "possible_associated_movements": [],
  "uncertain_findings": []
}"""


@dataclass
class Expert:
    """Definition of one regional expert.

    Attributes:
        name: Expert identifier.
        region: Region label used in output.
        prompt: Region-specific analysis prompt.
        clip_tasks: Task names (lowercase) whose clips this expert consumes.
        output_file: Output JSON filename.
    """

    name: str
    region: str
    prompt: str
    clip_tasks: list[str]
    output_file: str


EXPERTS: list[Expert] = [
    Expert(
        name="temporal",
        region="temporal",
        prompt="""Analyse only the forehead region.

Primary observations
- brow excursion
- brow symmetry
- forehead wrinkles
- movement quality
- brow elevation completeness

Secondary observations
- eye narrowing
- mouth movement
- contracture
- unexpected associated movement

Return objective visual findings only.""",
        clip_tasks=["raise eyebrow"],
        output_file="temporal_evidence.json",
    ),
    Expert(
        name="zygomatic",
        region="zygomatic",
        prompt="""Analyse only the eye region.

Primary observations
- eyelid closure completeness
- lagophthalmos
- eyelid symmetry
- blink quality
- orbicularis oculi contraction
- eye closure effort

Secondary observations
- brow movement
- mouth movement
- chin movement
- platysma activation

If abnormal associated movement is observed, describe it objectively without
diagnosing synkinesis.

Return structured JSON only.""",
        clip_tasks=["gentle eye closure", "close eye", "tight eye closure",
                    "tightly close eye", "blink", "blink repeatedly"],
        output_file="zygomatic_evidence.json",
    ),
    Expert(
        name="buccal",
        region="buccal",
        prompt="""Analyse only the oral and midface region.

Primary observations
- mouth corner excursion
- smile symmetry
- upper lip movement
- lip seal
- teeth visibility
- cheek elevation
- cheek fullness
- nasolabial fold depth
- nasolabial fold orientation
- nose wrinkle
- upper lip lift

Secondary observations
- eye narrowing
- mentalis contraction
- platysma activation
- contracture
- abnormal associated movement

Return structured JSON only.""",
        clip_tasks=["smile", "big smile", "puff cheek", "puff cheeks", "puff up cheeks",
                    "raise upper lip", "raise top lip", "screw nose", "screw up nose",
                    "blow kiss", "blow kisses", "blow kisses for 3 times"],
        output_file="buccal_evidence.json",
    ),
    Expert(
        name="marginal_mandibular",
        region="marginal_mandibular",
        prompt="""Analyse only the lower lip and chin.

Primary observations
- lower lip depression
- lower lip symmetry
- lower lip excursion
- mentalis contraction
- chin movement

Secondary observations
- eye movement
- platysma activation

Return structured JSON only.""",
        clip_tasks=["lower bottom lip"],
        output_file="marginal_evidence.json",
    ),
    Expert(
        name="cervical",
        region="cervical",
        prompt="""Analyse only the neck region.

Primary observations
- platysma activation
- neck symmetry
- contraction strength
- contraction quality

Secondary observations
- mouth movement
- eye movement

Return structured JSON only.""",
        clip_tasks=["angry neck"],
        output_file="cervical_evidence.json",
    ),
]
