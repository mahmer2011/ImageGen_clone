"""
Advanced Spine Builder (v4 - Welded Joints)
- Hierarchy Fix: Legs parented to Torso (fixes floating hips).
- Weld Zone: Top vertices weighted 100% to parent (fixes shoulder gaps).
- High Density Mesh: 5x6 grid for smoother skin bending.
"""
from __future__ import annotations
import math
import os
from typing import Any, Dict, List, Optional, Tuple

def image_to_spine_coords(x: float, y: float, img_width: int, img_height: int) -> Tuple[float, float]:
    """Convert image coordinates to Spine coordinates (centered, Y-flipped)."""
    spine_x = x - (img_width / 2)
    spine_y = (img_height - y) - (img_height / 2)
    return spine_x, spine_y

def rotate_point(x: float, y: float, angle_degrees: float) -> Tuple[float, float]:
    """Rotates a point (x,y) around (0,0)."""
    rad = math.radians(angle_degrees)
    cos_a = math.cos(rad)
    sin_a = math.sin(rad)
    new_x = x * cos_a - y * sin_a
    new_y = x * sin_a + y * cos_a
    return new_x, new_y

def create_simple_skeleton(
    landmarks: List[Dict[str, Any]],
    assembly_width: int,
    assembly_height: int,
    part_metadata: Dict[str, Any],
    segmentation_metadata: Optional[Dict[str, Any]] = None,
    character_name: str = "character"
) -> Dict[str, Any]:
    
    # 1. SETUP LOOKUPS
    lm_dict = {str(lm.get("name", "")).lower(): lm for lm in landmarks}
    
    part_owner_map = {
        "head": "head",
        "body": "torso",
        "left_upper_arm": "upper_arm_L", "left_lower_arm": "lower_arm_L",
        "right_upper_arm": "upper_arm_R", "right_lower_arm": "lower_arm_R",
        "left_upper_leg": "upper_leg_L", "left_lower_leg": "lower_leg_L",
        "right_upper_leg": "upper_leg_R", "right_lower_leg": "lower_leg_R",
        "left_foot": "lower_leg_L", "right_foot": "lower_leg_R"
    }

    # 2. CREATE BONES
    # Tracker for absolute positions (needed for local transform math)
    bone_tracker = [] 
    
    # --- Root ---
    root_x, root_y = _get_center(lm_dict, "left_hip", "right_hip", fallback=(assembly_width/2, assembly_height/2))
    root_sx, root_sy = image_to_spine_coords(root_x, root_y, assembly_width, assembly_height)
    
    root_data = {"name": "root", "x": round(root_sx, 2), "y": round(root_sy, 2), "rotation": 0}
    bone_tracker.append({
        "name": "root", 
        "abs_x": root_sx, "abs_y": root_sy, "abs_rot": 0, 
        "data": root_data
    })

    def add_bone(name, parent_name, start_lm, end_lm=None):
        if not start_lm: return
        
        # Get Parent Absolute Info
        parent = next((b for b in bone_tracker if b["name"] == parent_name), bone_tracker[0])
        p_abs_x, p_abs_y = parent["abs_x"], parent["abs_y"]
        p_abs_rot = parent["abs_rot"]
        
        # Child Absolute Start
        sx, sy = start_lm['x'], start_lm['y']
        c_abs_x, c_abs_y = image_to_spine_coords(sx, sy, assembly_width, assembly_height)
        
        # Child Rotation & Length
        length = 0
        c_abs_rot = 0
        
        if end_lm:
            ex, ey = end_lm['x'], end_lm['y']
            e_abs_x, e_abs_y = image_to_spine_coords(ex, ey, assembly_width, assembly_height)
            dx, dy = e_abs_x - c_abs_x, e_abs_y - c_abs_y
            length = math.sqrt(dx*dx + dy*dy)
            c_abs_rot = math.degrees(math.atan2(dy, dx))
        else:
            c_abs_rot = p_abs_rot 

        # Convert Global to Local (Parent Space)
        dx_global = c_abs_x - p_abs_x
        dy_global = c_abs_y - p_abs_y
        local_x, local_y = rotate_point(dx_global, dy_global, -p_abs_rot)
        local_rot = (c_abs_rot - p_abs_rot + 180) % 360 - 180

        bone_data = {
            "name": name, "parent": parent_name,
            "x": round(local_x, 2), "y": round(local_y, 2),
            "length": round(length, 2), "rotation": round(local_rot, 2)
        }
        
        bone_tracker.append({
            "name": name, "abs_x": c_abs_x, "abs_y": c_abs_y, "abs_rot": c_abs_rot,
            "data": bone_data
        })

    # -- Hierarchy --
    sh_x, sh_y = _get_center(lm_dict, "left_shoulder", "right_shoulder")
    hip_x, hip_y = _get_center(lm_dict, "left_hip", "right_hip")
    
    add_bone("torso", "root", {'x':(sh_x+hip_x)/2, 'y':(sh_y+hip_y)/2}, {'x':sh_x, 'y':sh_y}) 
    add_bone("head", "torso", lm_dict.get("nose"), None)
    
    add_bone("upper_arm_L", "torso", lm_dict.get("left_shoulder"), lm_dict.get("left_elbow"))
    add_bone("lower_arm_L", "upper_arm_L", lm_dict.get("left_elbow"), lm_dict.get("left_wrist"))
    add_bone("upper_arm_R", "torso", lm_dict.get("right_shoulder"), lm_dict.get("right_elbow"))
    add_bone("lower_arm_R", "upper_arm_R", lm_dict.get("right_elbow"), lm_dict.get("right_wrist"))
    
    # FIX: Legs are now children of 'torso' to follow body movement
    add_bone("upper_leg_L", "torso", lm_dict.get("left_hip"), lm_dict.get("left_knee"))
    add_bone("lower_leg_L", "upper_leg_L", lm_dict.get("left_knee"), lm_dict.get("left_ankle"))
    add_bone("upper_leg_R", "torso", lm_dict.get("right_hip"), lm_dict.get("right_knee"))
    add_bone("lower_leg_R", "upper_leg_R", lm_dict.get("right_knee"), lm_dict.get("right_ankle"))

    # 3. CREATE MESHES
    slots = []
    skins = {"default": {}}
    draw_order = [
        "right_lower_leg", "right_upper_leg", "right_lower_arm", "right_upper_arm", 
        "body", "head", 
        "left_upper_leg", "left_lower_leg", "left_upper_arm", "left_lower_arm"
    ]
    
    for part_name in draw_order:
        if part_name not in part_metadata: continue
        
        meta = part_metadata[part_name]
        seg_info = _find_seg_info(segmentation_metadata, part_name)
        if not seg_info: continue
        
        bone_name = part_owner_map.get(part_name, "root")
        bone_obj = next((b for b in bone_tracker if b["name"] == bone_name), bone_tracker[0])
        
        mesh_data = _generate_mesh(
            part_name, meta, seg_info, 
            bone_obj, bone_tracker, 
            assembly_width, assembly_height
        )
        
        slots.append({
            "name": part_name, "bone": bone_name, "attachment": part_name
        })
        
        if part_name not in skins["default"]: skins["default"][part_name] = {}
        skins["default"][part_name][part_name] = mesh_data

    return {
        "skeleton": {"spine": "4.1.0", "width": assembly_width, "height": assembly_height},
        "bones": [b["data"] for b in bone_tracker],
        "slots": slots,
        "skins": [ {"name": "default", "attachments": skins["default"]} ],
        "animations": {} 
    }

# --- HELPERS ---

def _get_center(lm_dict, k1, k2, fallback=(0,0)):
    p1 = lm_dict.get(k1)
    p2 = lm_dict.get(k2)
    if p1 and p2: return (p1['x'] + p2['x'])/2, (p1['y'] + p2['y'])/2
    if p1: return p1['x'], p1['y']
    if p2: return p2['x'], p2['y']
    return fallback

def _find_seg_info(meta, name):
    if not meta: return None
    for p in meta.get("parts", []):
        if p["name"] == name: return p
    return None

def _generate_mesh(part_name, atlas_meta, seg_info, owner_bone_obj, all_bone_objs, aw, ah):
    """Generates High-Density Weighted Mesh."""
    w = atlas_meta.width
    h = atlas_meta.height
    bbox = seg_info['bbox']
    abs_x, abs_y = bbox['x'], bbox['y']
    
    # HIGH DENSITY GRID (5 cols x 6 rows) for smooth skinning
    cols = 5
    rows = 6 
    
    vertices = []
    uvs = []
    triangles = []
    
    b_abs_x, b_abs_y = owner_bone_obj["abs_x"], owner_bone_obj["abs_y"]
    b_abs_rot = owner_bone_obj["abs_rot"]
    
    parent_obj = next((b for b in all_bone_objs if b["name"] == owner_bone_obj["data"]["parent"]), None)
    
    for r in range(rows + 1):
        for c in range(cols + 1):
            u = c / cols
            v = r / rows
            
            px = abs_x + (u * bbox['w'])
            py = abs_y + (v * bbox['h'])
            sx, sy = image_to_spine_coords(px, py, aw, ah)
            uvs.extend([u, v])
            
            # Local Transform
            dx, dy = sx - b_abs_x, sy - b_abs_y
            lx, ly = rotate_point(dx, dy, -b_abs_rot)
            
            # --- WELDING LOGIC ---
            w_owner = 1.0
            w_parent = 0.0
            
            is_limb = "arm" in part_name or "leg" in part_name
            if is_limb and parent_obj and parent_obj["name"] != "root":
                # WELD ZONE: Top 15% is LOCKED to parent (Fixes gaps)
                if v < 0.15:
                    w_parent = 1.0
                    w_owner = 0.0
                # BLEND ZONE: 15% to 35% blends smoothly
                elif v < 0.35:
                    # Map 0.15->0.35 to 1.0->0.0
                    norm = (v - 0.15) / 0.20 
                    w_parent = 1.0 - norm
                    w_owner = 1.0 - w_parent

            if w_parent > 0.01:
                # Parent Relative
                p_dx, p_dy = sx - parent_obj["abs_x"], sy - parent_obj["abs_y"]
                plx, ply = rotate_point(p_dx, p_dy, -parent_obj["abs_rot"])
                
                idx_owner = all_bone_objs.index(owner_bone_obj)
                idx_parent = all_bone_objs.index(parent_obj)
                
                vertices.extend([
                    2, 
                    idx_owner, round(lx, 2), round(ly, 2), round(w_owner, 3),
                    idx_parent, round(plx, 2), round(ply, 2), round(w_parent, 3)
                ])
            else:
                idx_owner = all_bone_objs.index(owner_bone_obj)
                vertices.extend([1, idx_owner, round(lx, 2), round(ly, 2), 1.0])

    # Triangles
    for r in range(rows):
        for c in range(cols):
            v1 = r * (cols + 1) + c
            v2 = v1 + 1
            v3 = (r + 1) * (cols + 1) + c
            v4 = v3 + 1
            triangles.extend([v1, v3, v2, v2, v3, v4])

    return {
        "type": "mesh",
        "uvs": uvs,
        "triangles": triangles,
        "vertices": vertices,
        "hull": 4,
        "width": w,
        "height": h
    }