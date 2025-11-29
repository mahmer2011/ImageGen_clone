"""
Simple Wave animation - only uses bones that exist in simplified skeleton.
Creates a waving animation for one arm.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from spine.animation_engine import Animation, CurveType


def create_wave_animation(duration: float = 2.0, hand: str = "R") -> Animation:
    """
    Create a waving animation using only basic bones.
    Only references: root, torso, head, upper_arm_L/R, lower_arm_L/R
    
    Args:
        duration: Duration of wave cycle
        hand: "L" or "R" for left or right hand
    """
    animation = Animation(name=f"wave_{hand.lower()}", duration=duration)
    
    # Determine which side to wave
    suffix = "_L" if hand.upper() == "L" else "_R"
    other_suffix = "_R" if hand.upper() == "L" else "_L"
    
    # Root - slight lean
    animation.add_bone_translation("root", 0.0, 0, 0)
    animation.add_bone_translation("root", duration, 0, 0)
    
    # Torso - slight turn toward waving side
    turn_amount = 5 if hand.upper() == "R" else -5
    animation.add_bone_rotation("torso", 0.0, 0)
    animation.add_bone_rotation("torso", duration / 4, turn_amount)
    animation.add_bone_rotation("torso", duration / 2, turn_amount)
    animation.add_bone_rotation("torso", 3 * duration / 4, turn_amount)
    animation.add_bone_rotation("torso", duration, 0)
    
    # Head - look at waving hand
    animation.add_bone_rotation("head", 0.0, 0)
    animation.add_bone_rotation("head", duration / 4, turn_amount * 0.5)
    animation.add_bone_rotation("head", duration, 0)
    
    # Waving arm - raise and wave
    # Upper arm
    animation.add_bone_rotation(f"upper_arm{suffix}", 0.0, 0)
    animation.add_bone_rotation(f"upper_arm{suffix}", duration / 8, 60)  # Raise
    animation.add_bone_rotation(f"upper_arm{suffix}", duration / 4, 80)  # High
    animation.add_bone_rotation(f"upper_arm{suffix}", duration / 2, 60)  # Wave down
    animation.add_bone_rotation(f"upper_arm{suffix}", 3 * duration / 4, 80)  # Wave up
    animation.add_bone_rotation(f"upper_arm{suffix}", 7 * duration / 8, 60)  # Wave down
    animation.add_bone_rotation(f"upper_arm{suffix}", duration, 0)  # Lower
    
    # Lower arm - wave motion
    animation.add_bone_rotation(f"lower_arm{suffix}", 0.0, -10)
    animation.add_bone_rotation(f"lower_arm{suffix}", duration / 8, 20)
    animation.add_bone_rotation(f"lower_arm{suffix}", duration / 4, -20)  # Wave
    animation.add_bone_rotation(f"lower_arm{suffix}", duration / 2, 20)  # Wave
    animation.add_bone_rotation(f"lower_arm{suffix}", 3 * duration / 4, -20)  # Wave
    animation.add_bone_rotation(f"lower_arm{suffix}", 7 * duration / 8, 20)
    animation.add_bone_rotation(f"lower_arm{suffix}", duration, -10)
    
    # Other arm - stay relaxed
    animation.add_bone_rotation(f"upper_arm{other_suffix}", 0.0, 0)
    animation.add_bone_rotation(f"upper_arm{other_suffix}", duration, 0)
    
    animation.add_bone_rotation(f"lower_arm{other_suffix}", 0.0, -10)
    animation.add_bone_rotation(f"lower_arm{other_suffix}", duration, -10)
    
    # Legs - stay still
    for side in ["L", "R"]:
        animation.add_bone_rotation(f"upper_leg_{side}", 0.0, 0)
        animation.add_bone_rotation(f"upper_leg_{side}", duration, 0)
        
        animation.add_bone_rotation(f"lower_leg_{side}", 0.0, 0)
        animation.add_bone_rotation(f"lower_leg_{side}", duration, 0)
    
    return animation


if __name__ == "__main__":
    anim_l = create_wave_animation(hand="L")
    anim_r = create_wave_animation(hand="R")
    print(f"Created wave animations: {anim_l.name}, {anim_r.name}")

