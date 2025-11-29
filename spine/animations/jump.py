"""
Simple Jump animation - only uses bones that exist in simplified skeleton.
Creates a basic jump animation.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from spine.animation_engine import Animation, CurveType


def create_jump_animation(duration: float = 1.2) -> Animation:
    """
    Create a jump animation using only basic bones.
    Only references: root, torso, head, upper_arm_L/R, lower_arm_L/R, upper_leg_L/R, lower_leg_L/R
    """
    animation = Animation(name="jump", duration=duration)
    
    anticipation = duration * 0.1  # Crouch down
    launch = duration * 0.2  # Take off
    peak = duration * 0.5  # At peak
    fall = duration * 0.8  # Falling
    landing = duration * 0.95  # Landing
    # duration = recovery
    
    # Root - jump arc
    animation.add_bone_translation("root", 0.0, 0, 0)
    animation.add_bone_translation("root", anticipation, 0, 5)  # Crouch down
    animation.add_bone_translation("root", launch, 0, -50)  # Jump up
    animation.add_bone_translation("root", peak, 0, -100)  # Peak height
    animation.add_bone_translation("root", fall, 0, -50)  # Falling
    animation.add_bone_translation("root", landing, 0, 5)  # Landing impact
    animation.add_bone_translation("root", duration, 0, 0)  # Recover
    
    # Torso - lean forward during jump
    animation.add_bone_rotation("torso", 0.0, 0)
    animation.add_bone_rotation("torso", anticipation, -5)  # Lean back
    animation.add_bone_rotation("torso", launch, 10)  # Lean forward
    animation.add_bone_rotation("torso", peak, 5)
    animation.add_bone_rotation("torso", fall, 10)
    animation.add_bone_rotation("torso", landing, -5)  # Impact
    animation.add_bone_rotation("torso", duration, 0)
    
    # Head - counter-balance
    animation.add_bone_rotation("head", 0.0, 0)
    animation.add_bone_rotation("head", peak, -5)
    animation.add_bone_rotation("head", duration, 0)
    
    # Arms - raise up
    for side in ["L", "R"]:
        animation.add_bone_rotation(f"upper_arm_{side}", 0.0, 0)
        animation.add_bone_rotation(f"upper_arm_{side}", anticipation, -30)  # Pull back
        animation.add_bone_rotation(f"upper_arm_{side}", launch, 60)  # Raise up
        animation.add_bone_rotation(f"upper_arm_{side}", peak, 80)
        animation.add_bone_rotation(f"upper_arm_{side}", fall, 60)
        animation.add_bone_rotation(f"upper_arm_{side}", landing, 0)
        animation.add_bone_rotation(f"upper_arm_{side}", duration, 0)
        
        animation.add_bone_rotation(f"lower_arm_{side}", 0.0, -10)
        animation.add_bone_rotation(f"lower_arm_{side}", launch, 20)
        animation.add_bone_rotation(f"lower_arm_{side}", peak, 30)
        animation.add_bone_rotation(f"lower_arm_{side}", landing, -10)
        animation.add_bone_rotation(f"lower_arm_{side}", duration, -10)
    
    # Legs - crouch and extend
    for side in ["L", "R"]:
        animation.add_bone_rotation(f"upper_leg_{side}", 0.0, 0)
        animation.add_bone_rotation(f"upper_leg_{side}", anticipation, 30)  # Crouch
        animation.add_bone_rotation(f"upper_leg_{side}", launch, -20)  # Extend
        animation.add_bone_rotation(f"upper_leg_{side}", peak, -10)
        animation.add_bone_rotation(f"upper_leg_{side}", fall, -5)
        animation.add_bone_rotation(f"upper_leg_{side}", landing, 20)  # Landing bend
        animation.add_bone_rotation(f"upper_leg_{side}", duration, 0)
        
        animation.add_bone_rotation(f"lower_leg_{side}", 0.0, 0)
        animation.add_bone_rotation(f"lower_leg_{side}", anticipation, -30)  # Crouch
        animation.add_bone_rotation(f"lower_leg_{side}", launch, 20)  # Extend
        animation.add_bone_rotation(f"lower_leg_{side}", peak, 10)
        animation.add_bone_rotation(f"lower_leg_{side}", fall, 5)
        animation.add_bone_rotation(f"lower_leg_{side}", landing, -20)  # Landing bend
        animation.add_bone_rotation(f"lower_leg_{side}", duration, 0)
    
    return animation


if __name__ == "__main__":
    anim = create_jump_animation()
    print(f"Created jump animation: {anim.name}, duration: {anim.duration}s")

