"""
Animation Engine

Core animation system for creating and managing Spine animations.
Supports keyframes, curves, and timeline manipulation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Union


class CurveType(Enum):
    """Animation curve types."""
    LINEAR = "linear"
    STEPPED = "stepped"
    BEZIER = "bezier"


@dataclass
class Keyframe:
    """Represents a single keyframe in an animation."""
    time: float
    value: Union[float, List[float]]
    curve: CurveType = CurveType.LINEAR
    curve_params: Optional[List[float]] = None  # For bezier curves: [cx1, cy1, cx2, cy2]


@dataclass
class BoneTimeline:
    """Animation timeline for a single bone."""
    bone_name: str
    rotate_keys: List[Keyframe] = field(default_factory=list)
    translate_keys: List[Keyframe] = field(default_factory=list)
    scale_keys: List[Keyframe] = field(default_factory=list)
    
    def add_rotation(self, time: float, angle: float, curve: CurveType = CurveType.LINEAR):
        """Add a rotation keyframe."""
        self.rotate_keys.append(Keyframe(time=time, value=angle, curve=curve))
        self.rotate_keys.sort(key=lambda k: k.time)
    
    def add_translation(
        self,
        time: float,
        x: float,
        y: float,
        curve: CurveType = CurveType.LINEAR
    ):
        """Add a translation keyframe."""
        self.translate_keys.append(Keyframe(time=time, value=[x, y], curve=curve))
        self.translate_keys.sort(key=lambda k: k.time)
    
    def add_scale(
        self,
        time: float,
        scale_x: float,
        scale_y: float,
        curve: CurveType = CurveType.LINEAR
    ):
        """Add a scale keyframe."""
        self.scale_keys.append(Keyframe(time=time, value=[scale_x, scale_y], curve=curve))
        self.scale_keys.sort(key=lambda k: k.time)


@dataclass
class Animation:
    """Represents a complete animation."""
    name: str
    duration: float
    bone_timelines: Dict[str, BoneTimeline] = field(default_factory=dict)
    slot_timelines: Dict[str, List[Keyframe]] = field(default_factory=dict)
    
    def get_or_create_bone_timeline(self, bone_name: str) -> BoneTimeline:
        """Get existing timeline or create a new one for a bone."""
        if bone_name not in self.bone_timelines:
            self.bone_timelines[bone_name] = BoneTimeline(bone_name=bone_name)
        return self.bone_timelines[bone_name]
    
    def add_bone_rotation(
        self,
        bone_name: str,
        time: float,
        angle: float,
        curve: CurveType = CurveType.LINEAR
    ):
        """Add rotation keyframe to a bone."""
        timeline = self.get_or_create_bone_timeline(bone_name)
        timeline.add_rotation(time, angle, curve)
    
    def add_bone_translation(
        self,
        bone_name: str,
        time: float,
        x: float,
        y: float,
        curve: CurveType = CurveType.LINEAR
    ):
        """Add translation keyframe to a bone."""
        timeline = self.get_or_create_bone_timeline(bone_name)
        timeline.add_translation(time, x, y, curve)
    
    def add_bone_scale(
        self,
        bone_name: str,
        time: float,
        scale_x: float,
        scale_y: float,
        curve: CurveType = CurveType.LINEAR
    ):
        """Add scale keyframe to a bone."""
        timeline = self.get_or_create_bone_timeline(bone_name)
        timeline.add_scale(time, scale_x, scale_y, curve)


class AnimationEngine:
    """
    Manages animation creation and export to Spine JSON format.
    """
    
    def __init__(self):
        self.animations: Dict[str, Animation] = {}
    
    def create_animation(self, name: str, duration: float) -> Animation:
        """Create a new animation."""
        animation = Animation(name=name, duration=duration)
        self.animations[name] = animation
        return animation
    
    def get_animation(self, name: str) -> Optional[Animation]:
        """Get an existing animation by name."""
        return self.animations.get(name)
    
    def export_to_spine_json(self) -> Dict[str, Any]:
        """
        Export all animations to Spine JSON format.
        
        Returns:
            Dict in Spine animation format ready to be added to skeleton JSON
        """
        animations_json = {}
        
        for anim_name, animation in self.animations.items():
            anim_data = {
                "bones": {},
                "slots": {}
            }
            
            # Export bone timelines
            for bone_name, timeline in animation.bone_timelines.items():
                bone_data = {}
                
                # Rotation timeline
                if timeline.rotate_keys:
                    bone_data["rotate"] = [
                        self._export_keyframe(kf, "rotate")
                        for kf in timeline.rotate_keys
                    ]
                
                # Translation timeline
                if timeline.translate_keys:
                    bone_data["translate"] = [
                        self._export_keyframe(kf, "translate")
                        for kf in timeline.translate_keys
                    ]
                
                # Scale timeline
                if timeline.scale_keys:
                    bone_data["scale"] = [
                        self._export_keyframe(kf, "scale")
                        for kf in timeline.scale_keys
                    ]
                
                if bone_data:
                    anim_data["bones"][bone_name] = bone_data
            
            animations_json[anim_name] = anim_data
        
        return animations_json
    
    def _export_keyframe(self, keyframe: Keyframe, timeline_type: str) -> Dict[str, Any]:
        """
        Export a single keyframe to Spine JSON format.
        """
        kf_data = {"time": round(keyframe.time, 4)}
        
        if timeline_type == "rotate":
            kf_data["value"] = round(keyframe.value, 2)
        elif timeline_type == "translate":
            if isinstance(keyframe.value, list) and len(keyframe.value) == 2:
                kf_data["x"] = round(keyframe.value[0], 2)
                kf_data["y"] = round(keyframe.value[1], 2)
        elif timeline_type == "scale":
            if isinstance(keyframe.value, list) and len(keyframe.value) == 2:
                kf_data["x"] = round(keyframe.value[0], 4)
                kf_data["y"] = round(keyframe.value[1], 4)
        
        # Add curve data
        if keyframe.curve == CurveType.STEPPED:
            kf_data["curve"] = "stepped"
        elif keyframe.curve == CurveType.BEZIER and keyframe.curve_params:
            kf_data["curve"] = [round(p, 4) for p in keyframe.curve_params]
        # Linear is default, no need to specify
        
        return kf_data
    
    def scale_animation_speed(self, anim_name: str, speed_factor: float):
        """
        Scale animation speed by a factor.
        
        Args:
            anim_name: Name of animation to scale
            speed_factor: Multiplier (2.0 = twice as fast, 0.5 = half speed)
        """
        animation = self.get_animation(anim_name)
        if not animation:
            return
        
        # Scale duration
        animation.duration /= speed_factor
        
        # Scale all keyframe times
        for timeline in animation.bone_timelines.values():
            for keyframe in timeline.rotate_keys:
                keyframe.time /= speed_factor
            for keyframe in timeline.translate_keys:
                keyframe.time /= speed_factor
            for keyframe in timeline.scale_keys:
                keyframe.time /= speed_factor
    
    def merge_animations(self, output_name: str, animation_names: List[str]) -> Optional[Animation]:
        """
        Merge multiple animations into a sequence.
        
        Args:
            output_name: Name for the merged animation
            animation_names: List of animation names to merge in order
            
        Returns:
            The merged animation or None if any source animation doesn't exist
        """
        # Verify all animations exist
        source_anims = []
        for name in animation_names:
            anim = self.get_animation(name)
            if not anim:
                return None
            source_anims.append(anim)
        
        # Calculate total duration
        total_duration = sum(a.duration for a in source_anims)
        
        # Create merged animation
        merged = self.create_animation(output_name, total_duration)
        
        # Merge timelines
        time_offset = 0.0
        for anim in source_anims:
            for bone_name, timeline in anim.bone_timelines.items():
                merged_timeline = merged.get_or_create_bone_timeline(bone_name)
                
                # Copy and offset keyframes
                for kf in timeline.rotate_keys:
                    new_kf = Keyframe(
                        time=kf.time + time_offset,
                        value=kf.value,
                        curve=kf.curve,
                        curve_params=kf.curve_params
                    )
                    merged_timeline.rotate_keys.append(new_kf)
                
                for kf in timeline.translate_keys:
                    new_kf = Keyframe(
                        time=kf.time + time_offset,
                        value=kf.value.copy() if isinstance(kf.value, list) else kf.value,
                        curve=kf.curve,
                        curve_params=kf.curve_params
                    )
                    merged_timeline.translate_keys.append(new_kf)
                
                for kf in timeline.scale_keys:
                    new_kf = Keyframe(
                        time=kf.time + time_offset,
                        value=kf.value.copy() if isinstance(kf.value, list) else kf.value,
                        curve=kf.curve,
                        curve_params=kf.curve_params
                    )
                    merged_timeline.scale_keys.append(new_kf)
            
            time_offset += anim.duration
        
        return merged
    
    def clone_animation(self, source_name: str, new_name: str) -> Optional[Animation]:
        """
        Create a copy of an animation with a new name.
        """
        source = self.get_animation(source_name)
        if not source:
            return None
        
        new_anim = self.create_animation(new_name, source.duration)
        
        # Deep copy all timelines
        for bone_name, timeline in source.bone_timelines.items():
            new_timeline = new_anim.get_or_create_bone_timeline(bone_name)
            
            for kf in timeline.rotate_keys:
                new_timeline.rotate_keys.append(Keyframe(
                    time=kf.time,
                    value=kf.value,
                    curve=kf.curve,
                    curve_params=kf.curve_params.copy() if kf.curve_params else None
                ))
            
            for kf in timeline.translate_keys:
                new_timeline.translate_keys.append(Keyframe(
                    time=kf.time,
                    value=kf.value.copy() if isinstance(kf.value, list) else kf.value,
                    curve=kf.curve,
                    curve_params=kf.curve_params.copy() if kf.curve_params else None
                ))
            
            for kf in timeline.scale_keys:
                new_timeline.scale_keys.append(Keyframe(
                    time=kf.time,
                    value=kf.value.copy() if isinstance(kf.value, list) else kf.value,
                    curve=kf.curve,
                    curve_params=kf.curve_params.copy() if kf.curve_params else None
                ))
        
        return new_anim


if __name__ == "__main__":
    print("Animation engine module loaded successfully")

