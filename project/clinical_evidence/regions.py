"""
Facial region definitions and per-task primary/monitored region mappings.

A "region" groups left/right features so symmetry and associated-movement
analysis can be done generically without per-feature hardcoding.
"""

from dataclasses import dataclass


@dataclass
class RegionFeatures:
    """Left/right (or single) feature names for a facial region.

    Attributes:
        left: Left-side feature column name.
        right: Right-side feature column name.
        speed_left: Optional left-side speed feature.
        speed_right: Optional right-side speed feature.
        bilateral: If True, region is a single/central feature (no L/R split).
    """

    left: str
    right: str
    speed_left: str | None = None
    speed_right: str | None = None
    bilateral: bool = False


# Region → left/right feature mapping
REGIONS: dict[str, RegionFeatures] = {
    "eyebrow": RegionFeatures("BrowHeight_L", "BrowHeight_R", "BrowSpeed_L", "BrowSpeed_R"),
    "eye": RegionFeatures("EAR_L", "EAR_R", "EARSpeed_L", "EARSpeed_R"),
    "cheek": RegionFeatures("CheekWidth_L", "CheekWidth_R"),
    "mouth_corner": RegionFeatures("LipCornerLift_L", "LipCornerLift_R"),
    # Central / bilateral regions — use the same column for both "sides"
    "mouth": RegionFeatures("MouthWidth", "MouthWidth", bilateral=True),
    "jaw": RegionFeatures("JawOpen", "JawOpen", bilateral=True),
    "upper_lip": RegionFeatures("UpperLipHeight", "UpperLipHeight", bilateral=True),
    "lower_lip": RegionFeatures("LowerLipHeight", "LowerLipHeight", bilateral=True),
}


# Regions that have a genuine left/right split (usable for symmetry analysis)
BILATERAL_REGIONS: list[str] = ["eyebrow", "eye", "cheek", "mouth_corner"]


@dataclass
class TaskRegionSpec:
    """Primary and monitored regions for a task.

    Attributes:
        primary_region: The main region involved in the action.
        primary_feature: The main feature name (for the summary).
        primary_direction: "increase" or "decrease" of the primary feature.
        monitored_regions: Non-primary regions to watch for associated movement.
    """

    primary_region: str
    primary_feature: str
    primary_direction: str
    monitored_regions: list[str]


# Task name → region specification
TASK_REGIONS: dict[str, TaskRegionSpec] = {
    "raise eyebrow": TaskRegionSpec("eyebrow", "BrowHeight", "increase", ["eye", "mouth", "jaw"]),
    "close eye": TaskRegionSpec("eye", "EAR", "decrease", ["mouth", "jaw", "eyebrow"]),
    "tightly close eye": TaskRegionSpec("eye", "EAR", "decrease", ["mouth", "jaw", "eyebrow"]),
    "blink repeatedly": TaskRegionSpec("eye", "EAR", "decrease", ["mouth", "jaw", "eyebrow"]),
    "smile": TaskRegionSpec("mouth", "MouthWidth", "increase", ["eye", "eyebrow", "jaw"]),
    "big smile": TaskRegionSpec("mouth", "MouthArea", "increase", ["eye", "eyebrow", "jaw"]),
    "puff cheeks": TaskRegionSpec("cheek", "CheekWidth", "increase", ["eye", "eyebrow", "jaw"]),
    "puff up cheeks": TaskRegionSpec("cheek", "CheekWidth", "increase", ["eye", "eyebrow", "jaw"]),
    "screw up nose": TaskRegionSpec("upper_lip", "UpperLipHeight", "decrease", ["eye", "eyebrow", "mouth"]),
    "raise upper lip": TaskRegionSpec("upper_lip", "UpperLipHeight", "decrease", ["eye", "eyebrow", "jaw"]),
    "raise top lip": TaskRegionSpec("upper_lip", "UpperLipHeight", "decrease", ["eye", "eyebrow", "jaw"]),
    "lower bottom lip": TaskRegionSpec("lower_lip", "LowerLipHeight", "decrease", ["eye", "eyebrow", "cheek"]),
    "angry neck": TaskRegionSpec("jaw", "JawOpen", "increase", ["eye", "eyebrow", "mouth"]),
    "blow kisses": TaskRegionSpec("mouth", "MouthWidth", "increase", ["eye", "eyebrow", "jaw"]),
}


def get_task_spec(task_name: str) -> TaskRegionSpec | None:
    """Look up region spec for a task name (case-insensitive fuzzy match).

    Args:
        task_name: Task name from annotation.

    Returns:
        TaskRegionSpec or None if no match.
    """
    normalized = task_name.lower().strip()
    if normalized in TASK_REGIONS:
        return TASK_REGIONS[normalized]
    for key, spec in TASK_REGIONS.items():
        if key in normalized or normalized in key:
            return spec
    return None
