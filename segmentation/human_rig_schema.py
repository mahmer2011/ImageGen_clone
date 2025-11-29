"""
Static rig schema for human / human‑like characters.

This module does NOT talk to Mediapipe directly. It only defines how we
interpret Mediapipe Pose landmarks in terms of Spine‑style body parts
that will later be segmented and exported.

Landmark naming convention
--------------------------
The existing Mediapipe helper in `segmentation.py` (`detect_mediapipe_pose`)
returns a list of dicts with keys:

    {
        "name": "<landmark_name>",  # e.g. "left_shoulder"
        "x": <pixel_x>,
        "y": <pixel_y>,
        "z": <z>,
        "visibility": <0..1>,
    }

The `name` values come from `mp.solutions.pose.PoseLandmark(idx).name.lower()`,
so they are lower‑case with underscores, such as:

    "nose", "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow",
    "left_wrist", "right_wrist",
    "left_hip", "right_hip",
    "left_knee", "right_knee",
    "left_ankle", "right_ankle",
    "left_heel", "right_heel",
    "left_foot_index", "right_foot_index", ...

We keep everything in this file aligned with those names.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Literal, Optional


Side = Literal["left", "right"]
PartType = Literal["segment", "region"]


@dataclass(frozen=True)
class PartDefinition:
    """
    Describes a single Spine body part in terms of Mediapipe landmarks.

    Attributes:
        name: Logical part name, e.g. \"upper_arm\", \"head\", \"torso\".
        type: \"segment\" for bone‑like parts (upper/lower arms/legs),
              \"region\" for blob‑like parts (head, torso, hands, feet).
        sided: True if the part exists for both left and right sides.
        joints: For segments, the proximal and distal joint landmark
                names (may include \"{side}\" placeholder).
        landmarks: Additional landmarks used to bound the region (for
                   head/torso/hands/feet). Names may include
                   \"{side}\" placeholder.
        bounds: Optional extra information such as a lower boundary
                defined by two landmarks, e.g. shoulders for the head.
    """

    name: str
    type: PartType
    sided: bool = False
    joints: Optional[List[str]] = None
    landmarks: Optional[List[str]] = None
    bounds: Optional[Dict[str, List[str]]] = None


# Canonical list of human / human‑like Spine parts we will support.
#
# NOTE:
# - We intentionally keep the set fairly small so it works well for both
#   humans and anthropomorphic characters (like your bipedal cat).
# - Extra pieces like hair, ears, clothing, or tails are better handled
#   as attachments on top of these base parts, not driven directly from
#   pose keypoints.
HUMAN_PART_DEFINITIONS: List[PartDefinition] = [
    PartDefinition(
        name="head",
        type="region",
        sided=False,
        landmarks=["nose", "left_ear", "right_ear"],
        bounds={
            # Lower boundary for head: line through both shoulders.
            "lower_line": ["left_shoulder", "right_shoulder"]
        },
    ),
    PartDefinition(
        name="torso",
        type="region",
        sided=False,
        landmarks=["left_shoulder", "right_shoulder", "left_hip", "right_hip"],
    ),
    # Arms
    PartDefinition(
        name="upper_arm",
        type="segment",
        sided=True,
        joints=["{side}_shoulder", "{side}_elbow"],
    ),
    PartDefinition(
        name="lower_arm",
        type="segment",
        sided=True,
        joints=["{side}_elbow", "{side}_wrist"],
    ),
    PartDefinition(
        name="hand",
        type="region",
        sided=True,
        joints=["{side}_wrist"],
        landmarks=[
            "{side}_wrist",
            "{side}_index",
            "{side}_pinky",
            "{side}_thumb",
        ],
    ),
    # Legs
    PartDefinition(
        name="upper_leg",
        type="segment",
        sided=True,
        joints=["{side}_hip", "{side}_knee"],
    ),
    PartDefinition(
        name="lower_leg",
        type="segment",
        sided=True,
        joints=["{side}_knee", "{side}_ankle"],
    ),
    PartDefinition(
        name="foot",
        type="region",
        sided=True,
        joints=["{side}_ankle"],
        landmarks=[
            "{side}_ankle",
            "{side}_heel",
            "{side}_foot_index",
        ],
    ),
]


def expand_sided_parts(
    sides: List[Side] | None = None,
) -> List[Dict[str, object]]:
    """
    Expand HUMAN_PART_DEFINITIONS to a concrete list of per‑side parts.

    Each returned dict has:
        - part_name: e.g. \"upper_arm_L\" or \"head\"
        - base_name: e.g. \"upper_arm\" (without side suffix)
        - side: \"left\" / \"right\" / None
        - type: \"segment\" / \"region\"
        - joints: concrete landmark names
        - landmarks: concrete landmark names (if any)
        - bounds: bounds dict with concrete names (if any)
    """
    if sides is None:
        sides = ["left", "right"]

    expanded: List[Dict[str, object]] = []

    for definition in HUMAN_PART_DEFINITIONS:
        if not definition.sided:
            expanded.append(
                {
                    "part_name": definition.name,
                    "base_name": definition.name,
                    "side": None,
                    "type": definition.type,
                    "joints": list(definition.joints or []),
                    "landmarks": list(definition.landmarks or []),
                    "bounds": definition.bounds.copy() if definition.bounds else None,
                }
            )
            continue

        for side in sides:
            side_suffix = "L" if side == "left" else "R"

            def _sub(name: str) -> str:
                return name.format(side=side)

            joints = [_sub(j) for j in (definition.joints or [])]
            landmarks = [_sub(lm) for lm in (definition.landmarks or [])]

            bounds: Optional[Dict[str, List[str]]] = None
            if definition.bounds:
                bounds = {
                    key: [_sub(v) for v in vals] for key, vals in definition.bounds.items()
                }

            expanded.append(
                {
                    "part_name": f"{definition.name}_{side_suffix}",
                    "base_name": definition.name,
                    "side": side,
                    "type": definition.type,
                    "joints": joints,
                    "landmarks": landmarks,
                    "bounds": bounds,
                }
            )

    return expanded


__all__ = [
    "Side",
    "PartType",
    "PartDefinition",
    "HUMAN_PART_DEFINITIONS",
    "expand_sided_parts",
]


