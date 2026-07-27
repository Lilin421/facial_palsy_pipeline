"""
Action-specific feature templates.

Maps each action to its primary/secondary features, expected direction,
and temporal pattern type (step vs oscillatory).
"""

from dataclasses import dataclass


@dataclass
class ActionTemplate:
    """Defines which features to analyze for a given facial action.

    Attributes:
        pattern: "step" (Template A) or "oscillatory" (Template B).
        primary_features: Main features to detect boundaries.
        primary_directions: Direction per primary feature ("increase" or "decrease").
        secondary_features: Supporting features (optional validation).
        secondary_directions: Direction per secondary feature.
    """

    pattern: str
    primary_features: list[str]
    primary_directions: list[str]
    secondary_features: list[str]
    secondary_directions: list[str]


# Maps normalized task names to templates
ACTION_TEMPLATES: dict[str, ActionTemplate] = {
    "raise eyebrow": ActionTemplate(
        pattern="step",
        primary_features=["BrowHeight_L", "BrowHeight_R"],
        primary_directions=["increase", "increase"],
        secondary_features=["BrowSpeed_L", "BrowSpeed_R"],
        secondary_directions=["increase", "increase"],
    ),
    "close eye": ActionTemplate(
        pattern="step",
        primary_features=["EAR_L", "EAR_R"],
        primary_directions=["decrease", "decrease"],
        secondary_features=["EARSpeed_L", "EARSpeed_R"],
        secondary_directions=["decrease", "decrease"],
    ),
    "tightly close eye": ActionTemplate(
        pattern="step",
        primary_features=["EAR_L", "EAR_R"],
        primary_directions=["decrease", "decrease"],
        secondary_features=["BrowHeight_L", "BrowHeight_R"],
        secondary_directions=["decrease", "decrease"],
    ),
    "blink repeatedly": ActionTemplate(
        pattern="oscillatory",
        primary_features=["EAR_L", "EAR_R"],
        primary_directions=["decrease", "decrease"],
        secondary_features=["EARSpeed_L", "EARSpeed_R"],
        secondary_directions=["decrease", "decrease"],
    ),
    "smile": ActionTemplate(
        pattern="step",
        primary_features=["MouthWidth", "MouthArea"],
        primary_directions=["increase", "increase"],
        secondary_features=["UpperLipHeight"],
        secondary_directions=["decrease"],
    ),
    "big smile": ActionTemplate(
        pattern="step",
        primary_features=["MouthArea", "LipGap"],
        primary_directions=["increase", "increase"],
        secondary_features=["MouthWidth"],
        secondary_directions=["increase"],
    ),
    "puff cheeks": ActionTemplate(
        pattern="step",
        primary_features=["CheekWidth_L", "CheekWidth_R"],
        primary_directions=["increase", "increase"],
        secondary_features=["LipGap"],
        secondary_directions=["decrease"],
    ),
    "puff up cheeks": ActionTemplate(
        pattern="step",
        primary_features=["CheekWidth_L", "CheekWidth_R"],
        primary_directions=["increase", "increase"],
        secondary_features=["LipGap"],
        secondary_directions=["decrease"],
    ),
    "screw up nose": ActionTemplate(
        pattern="step",
        primary_features=["UpperLipHeight"],
        primary_directions=["decrease"],
        secondary_features=["LowerLipHeight"],
        secondary_directions=["increase"],
    ),
    "raise upper lip": ActionTemplate(
        pattern="step",
        primary_features=["UpperLipHeight"],
        primary_directions=["decrease"],
        secondary_features=["LowerLipHeight"],
        secondary_directions=["decrease"],
    ),
    "raise top lip": ActionTemplate(
        pattern="step",
        primary_features=["UpperLipHeight"],
        primary_directions=["decrease"],
        secondary_features=["LowerLipHeight"],
        secondary_directions=["decrease"],
    ),
    "lower bottom lip": ActionTemplate(
        pattern="step",
        primary_features=["LowerLipHeight"],
        primary_directions=["decrease"],
        secondary_features=["LipGap"],
        secondary_directions=["increase"],
    ),
    "angry neck": ActionTemplate(
        pattern="step",
        primary_features=["JawOpen"],
        primary_directions=["increase"],
        secondary_features=["MouthArea"],
        secondary_directions=["increase"],
    ),
    "blow kisses": ActionTemplate(
        pattern="oscillatory",
        primary_features=["MouthWidth", "LipGap", "MouthArea"],
        primary_directions=["decrease", "increase", "increase"],
        secondary_features=["UpperLipHeight", "LowerLipHeight"],
        secondary_directions=["increase", "increase"],
    ),
}


def get_template(task_name: str) -> ActionTemplate | None:
    """Look up the action template for a task name.

    Performs case-insensitive fuzzy matching.

    Args:
        task_name: Name from task annotation.

    Returns:
        ActionTemplate or None if no match found.
    """
    normalized = task_name.lower().strip()

    # Direct match
    if normalized in ACTION_TEMPLATES:
        return ACTION_TEMPLATES[normalized]

    # Substring match
    for key, template in ACTION_TEMPLATES.items():
        if key in normalized or normalized in key:
            return template

    return None
