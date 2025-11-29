"""
Simple Walk animation - only uses bones that exist in simplified skeleton.
Creates a basic walk cycle.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from spine.animation_engine import Animation, CurveType


def create_walk_animation(duration: float = 1.0) -> Animation:
    """
    Create a walking animation cycle using only basic bones.
    Only references: root, torso, head, upper_arm_L/R, lower_arm_L/R, upper_leg_L/R, lower_leg_L/R
    """
    animation = Animation(name="walk", duration=duration)
    
    half = duration / 2  # Time for one step
    
    # Root movement (subtle bob)
    animation.add_bone_translation("root", 0.0, 0, 0)
    animation.add_bone_translation("root", half / 2, 0, -3)  # Down on contact
    animation.add_bone_translation("root", half, 0, 0)
    animation.add_bone_translation("root", half + half / 2, 0, -3)
    animation.add_bone_translation("root", duration, 0, 0)
    
    # Torso (subtle rotation)
    animation.add_bone_rotation("torso", 0.0, 0)
    animation.add_bone_rotation("torso", half / 2, -3)
    animation.add_bone_rotation("torso", half, 0)
    animation.add_bone_rotation("torso", half + half / 2, 3)
    animation.add_bone_rotation("torso", duration, 0)
    
    # Head (subtle counter-rotation for stability)
    animation.add_bone_rotation("head", 0.0, 0)
    animation.add_bone_rotation("head", half / 2, 2)
    animation.add_bone_rotation("head", half, 0)
    animation.add_bone_rotation("head", half + half / 2, -2)
    animation.add_bone_rotation("head", duration, 0)
    
    # Left leg forward, right leg back
    # Upper leg L
    animation.add_bone_rotation("upper_leg_L", 0.0, 30)  # Forward
    animation.add_bone_rotation("upper_leg_L", half / 2, 0)  # Contact
    animation.add_bone_rotation("upper_leg_L", half, -30)  # Back
    animation.add_bone_rotation("upper_leg_L", half + half / 2, 0)
    animation.add_bone_rotation("upper_leg_L", duration, 30)  # Forward again
    
    # Lower leg L
    animation.add_bone_rotation("lower_leg_L", 0.0, -10)
    animation.add_bone_rotation("lower_leg_L", half / 4, -30)  # Swing
    animation.add_bone_rotation("lower_leg_L", half / 2, -5)  # Straighten
    animation.add_bone_rotation("lower_leg_L", half, 0)
    animation.add_bone_rotation("lower_leg_L", half + half / 4, 10)
    animation.add_bone_rotation("lower_leg_L", duration, -10)
    
    # Right leg (opposite phase)
    # Upper leg R
    animation.add_bone_rotation("upper_leg_R", 0.0, -30)  # Back
    animation.add_bone_rotation("upper_leg_R", half / 2, 0)
    animation.add_bone_rotation("upper_leg_R", half, 30)  # Forward
    animation.add_bone_rotation("upper_leg_R", half + half / 2, 0)
    animation.add_bone_rotation("upper_leg_R", duration, -30)  # Back again
    
    # Lower leg R
    animation.add_bone_rotation("lower_leg_R", 0.0, 0)
    animation.add_bone_rotation("lower_leg_R", half / 4, 10)
    animation.add_bone_rotation("lower_leg_R", half / 2, -5)
    animation.add_bone_rotation("lower_leg_R", half, -10)
    animation.add_bone_rotation("lower_leg_R", half + half / 4, -30)
    animation.add_bone_rotation("lower_leg_R", half + half / 2, -5)
    animation.add_bone_rotation("lower_leg_R", duration, 0)
    
    # Arms - opposite to legs (natural swing)
    # Left arm (swings back when left leg forward)
    animation.add_bone_rotation("upper_arm_L", 0.0, -20)  # Back
    animation.add_bone_rotation("upper_arm_L", half / 2, 0)
    animation.add_bone_rotation("upper_arm_L", half, 20)  # Forward
    animation.add_bone_rotation("upper_arm_L", half + half / 2, 0)
    animation.add_bone_rotation("upper_arm_L", duration, -20)
    
    animation.add_bone_rotation("lower_arm_L", 0.0, -10)
    animation.add_bone_rotation("lower_arm_L", half, -10)
    animation.add_bone_rotation("lower_arm_L", duration, -10)
    
    # Right arm (opposite to left)
    animation.add_bone_rotation("upper_arm_R", 0.0, 20)  # Forward
    animation.add_bone_rotation("upper_arm_R", half / 2, 0)
    animation.add_bone_rotation("upper_arm_R", half, -20)  # Back
    animation.add_bone_rotation("upper_arm_R", half + half / 2, 0)
    animation.add_bone_rotation("upper_arm_R", duration, 20)
    
    animation.add_bone_rotation("lower_arm_R", 0.0, -10)
    animation.add_bone_rotation("lower_arm_R", half, -10)
    animation.add_bone_rotation("lower_arm_R", duration, -10)
    
    return animation


if __name__ == "__main__":
    anim = create_walk_animation()
    print(f"Created walk animation: {anim.name}, duration: {anim.duration}s")

