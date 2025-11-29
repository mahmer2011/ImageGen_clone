"""
Animation Builder

Converts animation templates to Spine JSON format and applies them to skeletons.
Handles IK constraints, timing adjustments, and character proportions.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from spine.animation_engine import Animation, AnimationEngine
from spine.animations.walk import create_walk_animation
from spine.animations.jump import create_jump_animation
from spine.animations.wave import create_wave_animation
from spine.animations.idle import create_idle_animation


class AnimationBuilder:
    """
    Builds and manages animations for Spine skeletons.
    """
    
    def __init__(self):
        self.engine = AnimationEngine()
        self._register_default_animations()
    
    def _register_default_animations(self):
        """Register all default animation templates."""
        # Create default animations
        walk = create_walk_animation(duration=1.0)
        jump = create_jump_animation(duration=1.2)
        wave_r = create_wave_animation(duration=2.0, hand="R")
        wave_l = create_wave_animation(duration=2.0, hand="L")
        idle = create_idle_animation(duration=3.0)
        
        # Register them
        self.engine.animations[walk.name] = walk
        self.engine.animations[jump.name] = jump
        self.engine.animations[wave_r.name] = wave_r
        self.engine.animations[wave_l.name] = wave_l
        self.engine.animations[idle.name] = idle
    
    def get_animation(self, name: str) -> Optional[Animation]:
        """Get an animation by name."""
        return self.engine.get_animation(name)
    
    def list_animations(self) -> List[str]:
        """List all available animation names."""
        return list(self.engine.animations.keys())
    
    def add_animations_to_skeleton(
        self,
        skeleton_json: Dict[str, Any],
        animation_names: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Add animations to a Spine skeleton JSON.
        
        Args:
            skeleton_json: Spine skeleton JSON structure
            animation_names: List of animation names to add, or None for all
            
        Returns:
            Updated skeleton JSON with animations
        """
        if animation_names is None:
            # Add all default animations
            animation_names = self.list_animations()
        
        # Verify all animations exist
        for name in animation_names:
            if name not in self.engine.animations:
                print(f"Warning: Animation '{name}' not found, skipping")
                continue
        
        # Get list of bones that exist in the skeleton
        existing_bones = set()
        if "bones" in skeleton_json:
            for bone in skeleton_json["bones"]:
                if isinstance(bone, dict) and "name" in bone:
                    existing_bones.add(bone["name"])
        
        # Export animations to JSON format
        animations_json = self.engine.export_to_spine_json()
        
        # Filter to requested animations and remove bone references that don't exist
        filtered_anims = {}
        for name, anim_data in animations_json.items():
            if name not in animation_names:
                continue
            
            # Filter out bone tracks that don't exist
            # anim_data structure: {"bones": {bone_name: bone_data}, "slots": {}}
            filtered_anim = {"bones": {}, "slots": {}}
            bone_tracks = anim_data.get("bones", {})
            
            for bone_name, bone_data in bone_tracks.items():
                if bone_name in existing_bones:
                    filtered_anim["bones"][bone_name] = bone_data
                else:
                    print(f"Filtering out bone '{bone_name}' from animation '{name}' (bone doesn't exist)")
            
            # Only add animation if it has at least one valid bone track
            if filtered_anim["bones"]:
                filtered_anims[name] = filtered_anim
            else:
                print(f"Skipping animation '{name}' - no valid bone tracks")
        
        # Add to skeleton
        if "animations" not in skeleton_json:
            skeleton_json["animations"] = {}
        
        skeleton_json["animations"].update(filtered_anims)
        
        return skeleton_json
    
    def adjust_animation_speed(self, name: str, speed_factor: float):
        """
        Adjust the playback speed of an animation.
        
        Args:
            name: Animation name
            speed_factor: Speed multiplier (2.0 = twice as fast, 0.5 = half speed)
        """
        self.engine.scale_animation_speed(name, speed_factor)
    
    def create_custom_animation(
        self,
        name: str,
        duration: float
    ) -> Animation:
        """
        Create a new custom animation.
        
        Args:
            name: Name for the animation
            duration: Duration in seconds
            
        Returns:
            New Animation object
        """
        return self.engine.create_animation(name, duration)
    
    def clone_and_modify(
        self,
        source_name: str,
        new_name: str,
        modifications: Optional[Dict[str, Any]] = None
    ) -> Optional[Animation]:
        """
        Clone an animation and apply modifications.
        
        Args:
            source_name: Name of source animation
            new_name: Name for cloned animation
            modifications: Dict with modification parameters
            
        Returns:
            Cloned and modified animation
        """
        cloned = self.engine.clone_animation(source_name, new_name)
        
        if cloned and modifications:
            # Apply modifications
            if "speed_factor" in modifications:
                self.adjust_animation_speed(new_name, modifications["speed_factor"])
        
        return cloned
    
    def export_animations_json(
        self,
        animation_names: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Export animations as Spine JSON.
        
        Args:
            animation_names: List of animations to export, or None for all
            
        Returns:
            Animations JSON structure
        """
        all_anims = self.engine.export_to_spine_json()
        
        if animation_names is None:
            return all_anims
        
        return {
            name: anim_data
            for name, anim_data in all_anims.items()
            if name in animation_names
        }


# Global instance for easy access
_global_builder: Optional[AnimationBuilder] = None


def get_animation_builder() -> AnimationBuilder:
    """Get the global animation builder instance."""
    global _global_builder
    if _global_builder is None:
        _global_builder = AnimationBuilder()
    return _global_builder


def add_default_animations_to_skeleton(skeleton_json: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convenience function to add all default animations to a skeleton.
    
    Args:
        skeleton_json: Spine skeleton JSON
        
    Returns:
        Updated skeleton JSON
    """
    builder = get_animation_builder()
    return builder.add_animations_to_skeleton(skeleton_json)


if __name__ == "__main__":
    # Test
    builder = AnimationBuilder()
    print(f"Available animations: {builder.list_animations()}")
    
    # Test export
    anims_json = builder.export_animations_json()
    print(f"Exported {len(anims_json)} animations")

