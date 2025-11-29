"""
Simple Idle animation - only uses bones that exist in simplified skeleton.
Creates a subtle breathing/idle animation.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from spine.animation_engine import Animation, CurveType


def create_idle_animation(duration: float = 3.0) -> Animation:
    """
    Create an idle/breathing animation using only basic bones.
    Only references: root, torso, head, upper_arm_L/R, lower_arm_L/R, upper_leg_L/R, lower_leg_L/R
    """
    animation = Animation(name="idle", duration=duration)
    
    half = duration / 2  # Breath in/out cycle
    
    # Root - very subtle bob
    animation.add_bone_translation("root", 0.0, 0, 0)
    animation.add_bone_translation("root", half, 0, -2)  # Slight down
    animation.add_bone_translation("root", duration, 0, 0)
    
    # Torso - breathing motion (scale for expansion)
    animation.add_bone_scale("torso", 0.0, 1.0, 1.0)
    animation.add_bone_scale("torso", half, 1.02, 1.01)  # Expand
    animation.add_bone_scale("torso", duration, 1.0, 1.0)
    
    # Torso - subtle rotation
    animation.add_bone_rotation("torso", 0.0, 0)
    animation.add_bone_rotation("torso", half / 2, 1)
    animation.add_bone_rotation("torso", half, 0)
    animation.add_bone_rotation("torso", half + half / 2, -1)
    animation.add_bone_rotation("torso", duration, 0)
    
    # Head - subtle looking around
    animation.add_bone_rotation("head", 0.0, 0)
    animation.add_bone_rotation("head", duration / 3, 2)
    animation.add_bone_rotation("head", 2 * duration / 3, -2)
    animation.add_bone_rotation("head", duration, 0)
    
    # Arms - very subtle sway (only add if bones might exist - will be filtered if they don't)
    # Left arm
    animation.add_bone_rotation("upper_arm_L", 0.0, 0)
    animation.add_bone_rotation("upper_arm_L", half, -2)
    animation.add_bone_rotation("upper_arm_L", duration, 0)
    
    animation.add_bone_rotation("lower_arm_L", 0.0, -10)
    animation.add_bone_rotation("lower_arm_L", half, -12)
    animation.add_bone_rotation("lower_arm_L", duration, -10)
    
    # Right arm (slight phase offset)
    animation.add_bone_rotation("upper_arm_R", 0.0, -2)
    animation.add_bone_rotation("upper_arm_R", half, 0)
    animation.add_bone_rotation("upper_arm_R", duration, -2)
    
    animation.add_bone_rotation("lower_arm_R", 0.0, -12)
    animation.add_bone_rotation("lower_arm_R", half, -10)
    animation.add_bone_rotation("lower_arm_R", duration, -12)
    
    # Legs - stay still (only add if bones might exist - will be filtered if they don't)
    for side in ["L", "R"]:
        animation.add_bone_rotation(f"upper_leg_{side}", 0.0, 0)
        animation.add_bone_rotation(f"upper_leg_{side}", duration, 0)
        
        animation.add_bone_rotation(f"lower_leg_{side}", 0.0, 0)
        animation.add_bone_rotation(f"lower_leg_{side}", duration, 0)
    
    return animation


if __name__ == "__main__":
    anim = create_idle_animation()
    print(f"Created idle animation: {anim.name}, duration: {anim.duration}s")

