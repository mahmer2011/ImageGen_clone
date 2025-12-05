"""
Simple Spine Builder - Clean, minimal pipeline using assembly preview as ground truth.

This module creates Spine skeletons by:
1. Using MediaPipe landmarks detected on the assembly preview
2. Creating bones from landmark positions
3. Mapping parts directly from assembly preview positions to attachments
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def image_to_spine_coords(x: float, y: float, img_width: int, img_height: int) -> Tuple[float, float]:
    """Convert image coordinates (top-left origin) to Spine coordinates (centered, Y-flipped)."""
    spine_x = x - (img_width / 2)
    spine_y = (img_height - y) - (img_height / 2)
    return spine_x, spine_y


def create_simple_skeleton(
    landmarks: List[Dict[str, Any]],
    assembly_width: int,
    assembly_height: int,
    part_metadata: Dict[str, Any],
    segmentation_metadata: Optional[Dict[str, Any]] = None,
    character_name: str = "character"
) -> Dict[str, Any]:
    """
    Create a complete Spine skeleton from landmarks and part metadata.
    
    Args:
        landmarks: MediaPipe landmarks detected on assembly preview
        assembly_width: Width of assembly preview image
        assembly_height: Height of assembly preview image
        part_metadata: Atlas part metadata (width, height, trim offsets)
        segmentation_metadata: Segmentation metadata with assembly positions
        character_name: Name for the skeleton
        
    Returns:
        Complete Spine JSON structure
    """
    # Build landmark lookup
    lm_dict: Dict[str, Dict[str, Any]] = {}
    for lm in landmarks:
        name = str(lm.get("name", "")).lower()
        if name:
            lm_dict[name] = lm
    
    # Build part positions from segmentation metadata
    part_positions: Dict[str, Dict[str, float]] = {}
    if segmentation_metadata:
        for part_entry in segmentation_metadata.get("parts", []):
            part_name = part_entry.get("name")
            assembly_info = part_entry.get("assembly")
            if part_name and assembly_info:
                center = assembly_info.get("center")
                origin = assembly_info.get("origin")
                if center and isinstance(center, dict):
                    part_positions[part_name] = {
                        "center_x": float(center.get("x", 0)),
                        "center_y": float(center.get("y", 0)),
                    }
                elif origin and isinstance(origin, dict):
                    size = assembly_info.get("size", {})
                    w = float(size.get("w", 0))
                    h = float(size.get("h", 0))
                    part_positions[part_name] = {
                        "center_x": float(origin.get("x", 0)) + w / 2.0,
                        "center_y": float(origin.get("y", 0)) + h / 2.0,
                    }
    
    # Create bones from landmarks
    bones = []
    
    # Root bone at hip center
    left_hip = lm_dict.get("left_hip")
    right_hip = lm_dict.get("right_hip")
    if left_hip and right_hip:
        root_x = (left_hip["x"] + right_hip["x"]) / 2
        root_y = (left_hip["y"] + right_hip["y"]) / 2
    else:
        root_x = assembly_width / 2
        root_y = assembly_height / 2
    
    root_spine_x, root_spine_y = image_to_spine_coords(root_x, root_y, assembly_width, assembly_height)
    bones.append({
        "name": "root",
        "parent": None,
        "x": round(root_spine_x, 2),
        "y": round(root_spine_y, 2),
        "rotation": 0,
        "length": 0
    })
    
    # Get list of parts that actually exist (before creating bones)
    existing_parts = set(part_metadata.keys())
    # Remove non-part files
    existing_parts = {p for p in existing_parts if p not in {
        "detection_visualization", "no_background", "outline_only", 
        "outline_overlay", "assembly_preview"
    }}
    print(f"Creating skeleton for {len(existing_parts)} parts: {sorted(existing_parts)}")
    
    # Torso bone (only if body/torso part exists)
    left_shoulder = lm_dict.get("left_shoulder")
    right_shoulder = lm_dict.get("right_shoulder")
    has_body = "body" in existing_parts or "torso" in existing_parts
    if has_body and left_shoulder and right_shoulder and left_hip and right_hip:
        torso_x = (left_shoulder["x"] + right_shoulder["x"] + left_hip["x"] + right_hip["x"]) / 4
        torso_y = (left_shoulder["y"] + right_shoulder["y"] + left_hip["y"] + right_hip["y"]) / 4
        torso_spine_x, torso_spine_y = image_to_spine_coords(torso_x, torso_y, assembly_width, assembly_height)
        bones.append({
            "name": "torso",
            "parent": "root",
            "x": round(torso_spine_x - root_spine_x, 2),
            "y": round(torso_spine_y - root_spine_y, 2),
            "rotation": 0,
            "length": 50
        })
    
    # Head bone (only if head part exists)
    nose = lm_dict.get("nose")
    has_head = "head" in existing_parts
    if has_head and nose and left_shoulder and right_shoulder:
        neck_x = (left_shoulder["x"] + right_shoulder["x"]) / 2
        neck_y = (left_shoulder["y"] + right_shoulder["y"]) / 2
        head_x = nose["x"]
        head_y = nose["y"] - 30  # Slightly above nose
        
        head_spine_x, head_spine_y = image_to_spine_coords(head_x, head_y, assembly_width, assembly_height)
        neck_spine_x, neck_spine_y = image_to_spine_coords(neck_x, neck_y, assembly_width, assembly_height)
        
        bones.append({
            "name": "head",
            "parent": "torso" if any(b["name"] == "torso" for b in bones) else "root",
            "x": round(head_spine_x - neck_spine_x, 2),
            "y": round(head_spine_y - neck_spine_y, 2),
            "rotation": 0,
            "length": 30
        })
    
    # Get list of parts that actually exist
    existing_parts = set(part_metadata.keys())
    # Remove non-part files
    existing_parts = {p for p in existing_parts if p not in {
        "detection_visualization", "no_background", "outline_only", 
        "outline_overlay", "assembly_preview"
    }}
    
    # Arm bones - only create if parts exist
    def create_arm_bones(side: str):
        shoulder = lm_dict.get(f"{side}_shoulder")
        elbow = lm_dict.get(f"{side}_elbow")
        wrist = lm_dict.get(f"{side}_wrist")
        
        # Convert side to suffix: "left" -> "_L", "right" -> "_R"
        suffix = "_L" if side == "left" else "_R"
        
        has_upper_arm = f"{side}_upper_arm" in existing_parts
        has_lower_arm = f"{side}_lower_arm" in existing_parts
        
        # Only create upper arm bone if we have the part OR if we need it as parent for lower arm
        if has_upper_arm and shoulder and elbow:
            elbow_spine_x, elbow_spine_y = image_to_spine_coords(
                elbow["x"], elbow["y"], assembly_width, assembly_height
            )
            shoulder_spine_x, shoulder_spine_y = image_to_spine_coords(
                shoulder["x"], shoulder["y"], assembly_width, assembly_height
            )
            
            bone_name = f"upper_arm{suffix}"
            bones.append({
                "name": bone_name,
                "parent": "torso" if any(b["name"] == "torso" for b in bones) else "root",
                "x": round(elbow_spine_x - shoulder_spine_x, 2),
                "y": round(elbow_spine_y - shoulder_spine_y, 2),
                "rotation": 0,
                "length": 40
            })
        
        # Create lower arm bone if we have the part
        if has_lower_arm and elbow and wrist:
            # Need parent bone - use upper_arm if it exists, otherwise torso
            parent_bone = f"upper_arm{suffix}" if has_upper_arm else ("torso" if any(b["name"] == "torso" for b in bones) else "root")
            
            # Get parent position for relative calculation
            parent_pos = (0, 0)
            if parent_bone != "root":
                parent_bone_obj = next((b for b in bones if b["name"] == parent_bone), None)
                if parent_bone_obj:
                    parent_pos = _bone_to_image_coords(parent_bone_obj, bones, assembly_width, assembly_height)
            
            elbow_spine_x, elbow_spine_y = image_to_spine_coords(
                elbow["x"], elbow["y"], assembly_width, assembly_height
            )
            wrist_spine_x, wrist_spine_y = image_to_spine_coords(
                wrist["x"], wrist["y"], assembly_width, assembly_height
            )
            
            # Calculate relative to parent
            parent_spine_x, parent_spine_y = image_to_spine_coords(parent_pos[0], parent_pos[1], assembly_width, assembly_height)
            elbow_rel_x = elbow_spine_x - parent_spine_x
            elbow_rel_y = elbow_spine_y - parent_spine_y
            wrist_rel_x = wrist_spine_x - parent_spine_x
            wrist_rel_y = wrist_spine_y - parent_spine_y
            
            bones.append({
                "name": f"lower_arm{suffix}",
                "parent": parent_bone,
                "x": round(wrist_rel_x - elbow_rel_x, 2),
                "y": round(wrist_rel_y - elbow_rel_y, 2),
                "rotation": 0,
                "length": 30
            })
    
    create_arm_bones("left")
    create_arm_bones("right")
    
    # Leg bones - only create if parts exist
    def create_leg_bones(side: str):
        hip = lm_dict.get(f"{side}_hip")
        knee = lm_dict.get(f"{side}_knee")
        ankle = lm_dict.get(f"{side}_ankle")
        
        # Convert side to suffix: "left" -> "_L", "right" -> "_R"
        suffix = "_L" if side == "left" else "_R"
        
        has_upper_leg = f"{side}_upper_leg" in existing_parts
        has_lower_leg = f"{side}_lower_leg" in existing_parts
        
        # Only create upper leg bone if we have the part
        if has_upper_leg and hip and knee:
            knee_spine_x, knee_spine_y = image_to_spine_coords(
                knee["x"], knee["y"], assembly_width, assembly_height
            )
            hip_spine_x, hip_spine_y = image_to_spine_coords(
                hip["x"], hip["y"], assembly_width, assembly_height
            )
            
            bone_name = f"upper_leg{suffix}"
            bones.append({
                "name": bone_name,
                "parent": "root",
                "x": round(knee_spine_x - hip_spine_x, 2),
                "y": round(knee_spine_y - hip_spine_y, 2),
                "rotation": 0,
                "length": 50
            })
        
        # Create lower leg bone if we have the part
        if has_lower_leg and knee and ankle:
            # Need parent bone - use upper_leg if it exists, otherwise root
            parent_bone = f"upper_leg{suffix}" if has_upper_leg else "root"
            
            # Get parent position for relative calculation
            parent_pos = (0, 0)
            if parent_bone != "root":
                parent_bone_obj = next((b for b in bones if b["name"] == parent_bone), None)
                if parent_bone_obj:
                    parent_pos = _bone_to_image_coords(parent_bone_obj, bones, assembly_width, assembly_height)
            
            knee_spine_x, knee_spine_y = image_to_spine_coords(
                knee["x"], knee["y"], assembly_width, assembly_height
            )
            ankle_spine_x, ankle_spine_y = image_to_spine_coords(
                ankle["x"], ankle["y"], assembly_width, assembly_height
            )
            
            # Calculate relative to parent
            parent_spine_x, parent_spine_y = image_to_spine_coords(parent_pos[0], parent_pos[1], assembly_width, assembly_height)
            knee_rel_x = knee_spine_x - parent_spine_x
            knee_rel_y = knee_spine_y - parent_spine_y
            ankle_rel_x = ankle_spine_x - parent_spine_x
            ankle_rel_y = ankle_spine_y - parent_spine_y
            
            bones.append({
                "name": f"lower_leg{suffix}",
                "parent": parent_bone,
                "x": round(ankle_rel_x - knee_rel_x, 2),
                "y": round(ankle_rel_y - knee_rel_y, 2),
                "rotation": 0,
                "length": 40
            })
    
    create_leg_bones("left")
    create_leg_bones("right")
    
    # Create slots and attachments
    slots = []
    attachments = {}
    
    # Map part names to bone names (parts use "left_", bones use "_L" suffix)
    # Only include parts that actually exist
    part_to_bone = {
        "head": "head",
        "body": "torso",  # Part is "body", bone is "torso"
        "torso": "torso",
        "left_upper_arm": "upper_arm_L",
        "left_lower_arm": "lower_arm_L",
        "right_upper_arm": "upper_arm_R",
        "right_lower_arm": "lower_arm_R",
        "left_upper_leg": "upper_leg_L",
        "left_lower_leg": "lower_leg_L",
        "right_upper_leg": "upper_leg_R",
        "right_lower_leg": "lower_leg_R",
    }
    # Filter to only parts that exist
    part_to_bone = {k: v for k, v in part_to_bone.items() if k in existing_parts}
    
    # Map bone names to slot names (for when bone name differs from part name)
    bone_to_slot = {
        "torso": "body"  # Bone is "torso", but slot should be "body" to match part name
    }
    
    # Create slots and attachments for each part
    for part_name, meta in part_metadata.items():
        # Skip non-part files
        if part_name in {"detection_visualization", "no_background", "outline_only", 
                        "outline_overlay", "assembly_preview"}:
            continue
        
        # Get part dimensions from atlas metadata
        if hasattr(meta, "width"):
            width = int(meta.width)
            height = int(meta.height)
            trim_x = float(meta.trim_offset_x)
            trim_y = float(meta.trim_offset_y)
        elif isinstance(meta, dict):
            width = int(meta.get("width", 0))
            height = int(meta.get("height", 0))
            trim_x = float(meta.get("trim_offset_x", 0))
            trim_y = float(meta.get("trim_offset_y", 0))
        else:
            continue
        
        if width <= 0 or height <= 0:
            continue
        
        # Find bone for this part
        bone_name = part_to_bone.get(part_name, "root")
        
        # Slot name is the part name (what we use in the atlas)
        # But if bone name differs, we might need to handle it
        slot_name = part_name
        
        # Special case: if part is "body" but bone is "torso", 
        # we still use "body" as slot name (matches atlas part name)
        
        # Get part center from assembly preview
        part_pos = part_positions.get(part_name)
        if not part_pos:
            # Fallback: use landmark position
            landmark = None
            if part_name == "head":
                landmark = lm_dict.get("nose")
            elif part_name in ("body", "torso"):
                if left_shoulder and right_shoulder and left_hip and right_hip:
                    part_pos = {
                        "center_x": (left_shoulder["x"] + right_shoulder["x"] + left_hip["x"] + right_hip["x"]) / 4,
                        "center_y": (left_shoulder["y"] + right_shoulder["y"] + left_hip["y"] + right_hip["y"]) / 4,
                    }
            else:
                # Try to find matching landmark
                for lm_name, lm_data in lm_dict.items():
                    if part_name.replace("_", "").replace("left", "").replace("right", "").lower() in lm_name:
                        landmark = lm_data
                        break
            
            if landmark and not part_pos:
                part_pos = {
                    "center_x": landmark["x"],
                    "center_y": landmark["y"],
                }
        
        if not part_pos:
            print(f"WARNING: Could not find position for part '{part_name}', skipping")
            continue
        
        # Get pivot point from segmentation metadata (where part should rotate from)
        pivot_point = None
        assembly_origin = None
        if segmentation_metadata:
            for part_entry in segmentation_metadata.get("parts", []):
                if part_entry.get("name") == part_name:
                    # Get pivot point (absolute coordinates in original image)
                    pivot_data = part_entry.get("pivot")
                    if pivot_data and isinstance(pivot_data, dict):
                        pivot_abs = pivot_data.get("absolute")
                        if pivot_abs and isinstance(pivot_abs, dict):
                            pivot_point = {
                                "x": float(pivot_abs.get("x", 0)),
                                "y": float(pivot_abs.get("y", 0))
                            }
                    
                    # Get assembly origin for trimmed part position
                    assembly_info = part_entry.get("assembly")
                    if assembly_info:
                        origin = assembly_info.get("origin")
                        if origin:
                            assembly_origin = (float(origin.get("x", 0)), float(origin.get("y", 0)))
                    break
        
        # Get bone position
        bone = next((b for b in bones if b["name"] == bone_name), None)
        if not bone:
            bone_name = "root"
            bone = bones[0]
        
        # Calculate bone position in image space
        bone_img_x, bone_img_y = _bone_to_image_coords(bone, bones, assembly_width, assembly_height)
        
        # Use pivot point if available, otherwise fall back to part center
        if pivot_point:
            # Pivot point is in original image coordinates
            pivot_x = pivot_point["x"]
            pivot_y = pivot_point["y"]
        else:
            # Fallback: use part center
            if assembly_origin:
                # Calculate trimmed center from origin
                pivot_x = assembly_origin[0] + trim_x + (width / 2.0)
                pivot_y = assembly_origin[1] + trim_y + (height / 2.0)
            else:
                # Use part center, adjust for trim
                pivot_x = part_pos["center_x"] - (width / 2.0) + trim_x + (width / 2.0)
                pivot_y = part_pos["center_y"] - (height / 2.0) + trim_y + (height / 2.0)
        
        # Convert pivot point and bone position to Spine coordinates
        pivot_spine_x, pivot_spine_y = image_to_spine_coords(
            pivot_x, pivot_y, assembly_width, assembly_height
        )
        bone_spine_x, bone_spine_y = image_to_spine_coords(
            bone_img_x, bone_img_y, assembly_width, assembly_height
        )
        
        # Attachment offset = pivot point - bone position
        # This ensures the part rotates around the pivot (joint) point, not its center
        attachment_x = round(pivot_spine_x - bone_spine_x, 2)
        attachment_y = round(pivot_spine_y - bone_spine_y, 2)
        
        # Create slot (slot name = part name)
        slots.append({
            "name": slot_name,
            "bone": bone_name,
            "attachment": slot_name,  # Default attachment name matches slot name
            "color": "ffffffff"
        })
        
        # Create attachment (organized by slot name, not bone name)
        if slot_name not in attachments:
            attachments[slot_name] = {}
        attachments[slot_name][slot_name] = {
            "type": "region",
            "x": attachment_x,
            "y": attachment_y,
            "width": width,
            "height": height,
            "rotation": 0
        }
    
    # Validate that all slots reference bones that exist
    bone_names = {b["name"] for b in bones}
    valid_slots = []
    for slot in slots:
        if slot["bone"] in bone_names:
            valid_slots.append(slot)
        else:
            print(f"WARNING: Slot '{slot['name']}' references non-existent bone '{slot['bone']}', removing slot")
            # Also remove from attachments if it exists
            if slot["name"] in attachments:
                del attachments[slot["name"]]
    slots = valid_slots
    
    # Build final JSON structure (Spine 4.x format)
    skeleton = {
        "skeleton": {
            "hash": character_name,
            "spine": "4.1.0",
            "width": assembly_width,
            "height": assembly_height,
            "images": "./"
        },
        "bones": bones,
        "slots": slots,
        "skins": [{
            "name": "default",
            "attachments": attachments
        }],
        "animations": {}
    }
    
    print(f"Created skeleton with {len(bones)} bones, {len(slots)} slots, {len(attachments)} attachment groups")
    return skeleton


def _bone_to_image_coords(bone: Dict[str, Any], all_bones: List[Dict[str, Any]], 
                          img_width: int, img_height: int) -> Tuple[float, float]:
    """Convert bone position to image coordinates."""
    # Start with root bone position
    root_bone = next((b for b in all_bones if b["name"] == "root"), None)
    if not root_bone:
        return img_width / 2, img_height / 2
    
    # Convert root to image coords
    root_img_x = root_bone["x"] + (img_width / 2)
    root_img_y = (img_height / 2) - root_bone["y"]
    
    # If this is the root bone, return its position
    if bone["name"] == "root":
        return root_img_x, root_img_y
    
    # Traverse up parent chain to accumulate position
    x = bone["x"]
    y = bone["y"]
    parent_name = bone.get("parent")
    
    while parent_name:
        parent = next((b for b in all_bones if b["name"] == parent_name), None)
        if not parent:
            break
        x += parent["x"]
        y += parent["y"]
        if parent_name == "root":
            break
        parent_name = parent.get("parent")
    
    # Convert accumulated Spine coords to image coords
    img_x = x + (img_width / 2)
    img_y = (img_height / 2) - y
    
    return img_x, img_y

