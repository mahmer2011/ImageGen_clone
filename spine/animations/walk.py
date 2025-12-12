"""
Simple Walk animation - Dynamic Generator
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from spine.animation_engine import Animation

def create_walk_animation(
    duration: float = 1.0,
    stride_scale: float = 1.0,  # 1.0 = normal, 0.5 = baby steps, 1.5 = giant steps
    bounce_scale: float = 1.0,  # 1.0 = normal, 2.0 = happy/bouncy, 0.2 = robotic
    arm_swing: float = 1.0      # 1.0 = normal, 2.0 = energetic, 0.1 = zombie
) -> Animation:
    """
    Create a walking animation cycle with customizable parameters.
    """
    # Adjust rotation magnitudes based on params
    leg_rot = 30.0 * stride_scale
    arm_rot = 20.0 * arm_swing
    bob_y = -3.0 * bounce_scale
    
    animation = Animation(name="walk", duration=duration)
    half = duration / 2
    
    # Root movement (The Bob)
    animation.add_bone_translation("root", 0.0, 0, 0)
    animation.add_bone_translation("root", half / 2, 0, bob_y)  # Down on contact
    animation.add_bone_translation("root", half, 0, 0)
    animation.add_bone_translation("root", half + half / 2, 0, bob_y)
    animation.add_bone_translation("root", duration, 0, 0)
    
    # Torso (Counter-rotation)
    torso_rot = 3.0 * stride_scale
    animation.add_bone_rotation("torso", 0.0, 0)
    animation.add_bone_rotation("torso", half / 2, -torso_rot)
    animation.add_bone_rotation("torso", half, 0)
    animation.add_bone_rotation("torso", half + half / 2, torso_rot)
    animation.add_bone_rotation("torso", duration, 0)
    
    # Left leg (Forward/Back)
    animation.add_bone_rotation("upper_leg_L", 0.0, leg_rot)
    animation.add_bone_rotation("upper_leg_L", half / 2, 0)
    animation.add_bone_rotation("upper_leg_L", half, -leg_rot)
    animation.add_bone_rotation("upper_leg_L", half + half / 2, 0)
    animation.add_bone_rotation("upper_leg_L", duration, leg_rot)
    
    # Left Knee (Bend)
    knee_bend = -30.0 * stride_scale # More stride needs more knee bend
    animation.add_bone_rotation("lower_leg_L", 0.0, -10)
    animation.add_bone_rotation("lower_leg_L", half / 4, knee_bend)
    animation.add_bone_rotation("lower_leg_L", half / 2, -5)
    animation.add_bone_rotation("lower_leg_L", half, 0)
    animation.add_bone_rotation("lower_leg_L", half + half / 4, 10)
    animation.add_bone_rotation("lower_leg_L", duration, -10)
    
    # Right leg (Phase shifted)
    animation.add_bone_rotation("upper_leg_R", 0.0, -leg_rot)
    animation.add_bone_rotation("upper_leg_R", half / 2, 0)
    animation.add_bone_rotation("upper_leg_R", half, leg_rot)
    animation.add_bone_rotation("upper_leg_R", half + half / 2, 0)
    animation.add_bone_rotation("upper_leg_R", duration, -leg_rot)
    
    # Right Knee
    animation.add_bone_rotation("lower_leg_R", 0.0, 0)
    animation.add_bone_rotation("lower_leg_R", half / 4, 10)
    animation.add_bone_rotation("lower_leg_R", half / 2, -5)
    animation.add_bone_rotation("lower_leg_R", half, -10)
    animation.add_bone_rotation("lower_leg_R", half + half / 4, knee_bend)
    animation.add_bone_rotation("lower_leg_R", half + half / 2, -5)
    animation.add_bone_rotation("lower_leg_R", duration, 0)
    
    # Arms
    animation.add_bone_rotation("upper_arm_L", 0.0, -arm_rot)
    animation.add_bone_rotation("upper_arm_L", half, arm_rot)
    animation.add_bone_rotation("upper_arm_L", duration, -arm_rot)
    
    animation.add_bone_rotation("upper_arm_R", 0.0, arm_rot)
    animation.add_bone_rotation("upper_arm_R", half, -arm_rot)
    animation.add_bone_rotation("upper_arm_R", duration, arm_rot)
    
    return animation