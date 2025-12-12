"""
Advanced Spine Builder (Mesh & Auto-Weights)
Generates weighted meshes for smooth bending characters.
"""
from __future__ import annotations
import json
import math
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
    
    # 1. SETUP LOOKUPS
    lm_dict = {str(lm.get("name", "")).lower(): lm for lm in landmarks}
    
    # Map Part Names -> Bone Names
    # (The bone that "owns" this part)
    part_owner_map = {
        "head": "head",
        "body": "torso",
        "left_upper_arm": "upper_arm_L", "left_lower_arm": "lower_arm_L",
        "right_upper_arm": "upper_arm_R", "right_lower_arm": "lower_arm_R",
        "left_upper_leg": "upper_leg_L", "left_lower_leg": "lower_leg_L",
        "right_upper_leg": "upper_leg_R", "right_lower_leg": "lower_leg_R",
        "left_foot": "lower_leg_L", "right_foot": "lower_leg_R" # Feet attach to lower legs for now
    }

    # 2. CREATE BONES
    # We calculate bone positions based on Landmarks
    bones = []
    
    # Root
    root_x, root_y = _get_center(lm_dict, "left_hip", "right_hip", fallback=(assembly_width/2, assembly_height/2))
    root_sx, root_sy = image_to_spine_coords(root_x, root_y, assembly_width, assembly_height)
    bones.append({"name": "root", "x": round(root_sx, 2), "y": round(root_sy, 2), "rotation": 0})

    # Helper to add bone
    def add_bone(name, parent, start_lm, end_lm=None):
        if not start_lm: return
        
        # Parent Position (Spine Coords)
        parent_bone = next((b for b in bones if b["name"] == parent), bones[0])
        # We need absolute spine coords of parent to calculate relative offset
        # (Simplified: assuming parent rotation is 0 for initial setup calculation)
        # For a robust system, we track absolute positions.
        
        # Calculate Start Point
        sx, sy = start_lm['x'], start_lm['y']
        spx, spy = image_to_spine_coords(sx, sy, assembly_width, assembly_height)
        
        # Calculate End Point (for length/rotation)
        length = 0
        rotation = 0
        if end_lm:
            ex, ey = end_lm['x'], end_lm['y']
            epx, epy = image_to_spine_coords(ex, ey, assembly_width, assembly_height)
            dx, dy = epx - spx, epy - spy
            length = math.sqrt(dx*dx + dy*dy)
            rotation = math.degrees(math.atan2(dy, dx))
            
            # Spine bones point relative to parent. 
            # If parent has rotation, we subtract it.
            # But here we are building a flat hierarchy relative to root/torso effectively
            # Let's keep it simple: Relative to Parent's absolute rotation?
            # To avoid complex transform math, we'll set parent rotations to 0 in setup pose
            # and encode the angle in the child.
        
        # Calculate Relative Position to Parent
        # (This assumes we tracked parent absolute position. Since we build in order, we can).
        parent_abs_x, parent_abs_y = _get_bone_abs_spine(parent_bone, bones)
        
        rel_x = spx - parent_abs_x
        rel_y = spy - parent_abs_y
        
        bones.append({
            "name": name,
            "parent": parent,
            "x": round(rel_x, 2),
            "y": round(rel_y, 2),
            "length": round(length, 2),
            "rotation": round(rotation, 2)
        })

    # -- Build Skeleton Hierarchy --
    
    # Torso
    # Center of shoulders + Center of hips
    sh_x, sh_y = _get_center(lm_dict, "left_shoulder", "right_shoulder")
    hip_x, hip_y = _get_center(lm_dict, "left_hip", "right_hip")
    add_bone("torso", "root", {'x':(sh_x+hip_x)/2, 'y':(sh_y+hip_y)/2}, {'x':sh_x, 'y':sh_y}) # Points up
    
    # Head
    add_bone("head", "torso", lm_dict.get("nose"), None) # Just a point for now
    
    # Arms
    add_bone("upper_arm_L", "torso", lm_dict.get("left_shoulder"), lm_dict.get("left_elbow"))
    add_bone("lower_arm_L", "upper_arm_L", lm_dict.get("left_elbow"), lm_dict.get("left_wrist"))
    add_bone("upper_arm_R", "torso", lm_dict.get("right_shoulder"), lm_dict.get("right_elbow"))
    add_bone("lower_arm_R", "upper_arm_R", lm_dict.get("right_elbow"), lm_dict.get("right_wrist"))
    
    # Legs
    add_bone("upper_leg_L", "root", lm_dict.get("left_hip"), lm_dict.get("left_knee"))
    add_bone("lower_leg_L", "upper_leg_L", lm_dict.get("left_knee"), lm_dict.get("left_ankle"))
    add_bone("upper_leg_R", "root", lm_dict.get("right_hip"), lm_dict.get("right_knee"))
    add_bone("lower_leg_R", "upper_leg_R", lm_dict.get("right_knee"), lm_dict.get("right_ankle"))

    # 3. CREATE MESH ATTACHMENTS
    slots = []
    skins = {"default": {}}
    
    # Sort parts for draw order (Z-index)
    # Reuse z-index logic from segmenter or list explicitly
    draw_order = [
        "right_lower_leg", "right_upper_leg", "right_lower_arm", "right_upper_arm", 
        "body", "head", 
        "left_upper_leg", "left_lower_leg", "left_upper_arm", "left_lower_arm"
    ]
    
    for part_name in draw_order:
        # Check if part exists
        if part_name not in part_metadata: continue
        
        # Get Metadata
        meta = part_metadata[part_name] # Atlas info
        seg_info = _find_seg_info(segmentation_metadata, part_name) # Positional info
        if not seg_info: continue
        
        bone_name = part_owner_map.get(part_name, "root")
        bone = next((b for b in bones if b["name"] == bone_name), bones[0])
        
        # Generate Mesh
        mesh_data = _generate_mesh(
            part_name, 
            meta, 
            seg_info, 
            bone, 
            bones,
            assembly_width, 
            assembly_height
        )
        
        # Add Slot
        slots.append({
            "name": part_name,
            "bone": bone_name,
            "attachment": part_name
        })
        
        # Add Skin Attachment
        if part_name not in skins["default"]:
            skins["default"][part_name] = {}
        
        skins["default"][part_name][part_name] = mesh_data

    # 4. COMPILE
    return {
        "skeleton": {"spine": "4.1.0", "width": assembly_width, "height": assembly_height},
        "bones": bones,
        "slots": slots,
        "skins": [ {"name": "default", "attachments": skins["default"]} ],
        "animations": {} 
    }

# --- HELPERS ---

def _get_bone_abs_spine(bone, all_bones):
    """Recursively calculate absolute spine position."""
    if bone["name"] == "root":
        return bone["x"], bone["y"]
    
    parent = next(b for b in all_bones if b["name"] == bone["parent"])
    px, py = _get_bone_abs_spine(parent, all_bones)
    return px + bone["x"], py + bone["y"]

def _get_center(lm_dict, k1, k2, fallback=(0,0)):
    p1 = lm_dict.get(k1)
    p2 = lm_dict.get(k2)
    if p1 and p2:
        return (p1['x'] + p2['x'])/2, (p1['y'] + p2['y'])/2
    if p1: return p1['x'], p1['y']
    if p2: return p2['x'], p2['y']
    return fallback

def _find_seg_info(meta, name):
    if not meta: return None
    for p in meta.get("parts", []):
        if p["name"] == name: return p
    return None

def _generate_mesh(part_name, atlas_meta, seg_info, owner_bone, all_bones, aw, ah):
    """
    Generates a Weighted Mesh Attachment.
    Creates a grid of vertices covering the part's bounding box.
    Calculates weights relative to the Owner Bone and its Parent/Child.
    """
    # 1. Get Image Dimensions
    # atlas_meta is PartPackingInfo(width, height, ...)
    w = atlas_meta.width
    h = atlas_meta.height
    
    # 2. Get Absolute Position on Canvas
    # seg_info['bbox'] gives x,y,w,h in original image pixels (top-left)
    bbox = seg_info['bbox']
    abs_x = bbox['x']
    abs_y = bbox['y']
    
    # 3. Create Grid (e.g. 3x3 for limbs, 4x4 for body)
    cols = 3
    rows = 4 if "leg" in part_name or "arm" in part_name else 3
    
    vertices = [] # [x1, y1, w1, ...] Weighted format
    uvs = []
    triangles = []
    
    # We need the Spine absolute coords of the owner bone
    bone_sx, bone_sy = _get_bone_abs_spine(owner_bone, all_bones)
    
    # Determine "Influence" Bones for weighting
    # (e.g. Lower Arm is influenced by Upper Arm at the top)
    parent_bone = next((b for b in all_bones if b["name"] == owner_bone["parent"]), None)
    
    parent_sx, parent_sy = (0,0)
    if parent_bone:
        parent_sx, parent_sy = _get_bone_abs_spine(parent_bone, all_bones)

    # 4. Generate Vertices & Weights
    for r in range(rows + 1):
        for c in range(cols + 1):
            # Normalized pos (0-1)
            u = c / cols
            v = r / rows
            
            # Pixel pos in Image
            px = abs_x + (u * bbox['w'])
            py = abs_y + (v * bbox['h'])
            
            # Spine Coords
            sx, sy = image_to_spine_coords(px, py, aw, ah)
            
            # UVs
            uvs.extend([u, v])
            
            # --- WEIGHT PAINTING LOGIC ---
            # Calculate distance to Owner Bone vs Parent Bone
            # This is a heuristic. Real rigging uses geodesic voxel binding.
            # Here: "If vertex is at top of part, blend with parent."
            
            # Relative pos to bone
            rel_x = sx - bone_sx
            rel_y = sy - bone_sy
            
            # Weight Calculation
            # Default: 100% Owner
            w_owner = 1.0
            w_parent = 0.0
            
            # Blending logic for limbs
            is_limb = "arm" in part_name or "leg" in part_name
            if is_limb and parent_bone and parent_bone["name"] != "torso" and parent_bone["name"] != "root":
                # If we are "upper" part of a limb (v < 0.2), blend with parent
                # Actually, check direction. Assuming standard T-pose or down-pose.
                # Simple heuristic: v (vertical) 0.0 is top.
                if v < 0.3: 
                    # Blend factor 0.0 -> 0.5 (at top edge)
                    w_parent = 0.5 * (1.0 - (v / 0.3))
                    w_owner = 1.0 - w_parent
            
            # Append Vertex Data (Weighted)
            # Format: [BoneCount, BoneIndex, RelX, RelY, Weight, ...]
            if w_parent > 0.01:
                # Two bones
                # Parent Relative
                p_rel_x = sx - parent_sx
                p_rel_y = sy - parent_sy
                
                # We need Indices of bones in the skeleton list?
                # Actually Spine JSON just needs vertices relative to the bone if not weighted?
                # Wait, "type": "mesh" requires specific vertex format.
                # Weighted Mesh Format: 
                # vertices: [ count, boneIndex1, x1, y1, w1, boneIndex2, x2, y2, w2, ... ]
                
                # Find indices (0-based index in the 'bones' array)
                idx_owner = all_bones.index(owner_bone)
                idx_parent = all_bones.index(parent_bone)
                
                vertices.extend([
                    2, # Count
                    idx_owner, round(rel_x, 2), round(rel_y, 2), round(w_owner, 3),
                    idx_parent, round(p_rel_x, 2), round(p_rel_y, 2), round(w_parent, 3)
                ])
            else:
                # One bone (Standard)
                # If mesh is weighted, we MUST use weighted format for all vertices? Yes.
                idx_owner = all_bones.index(owner_bone)
                vertices.extend([
                    1, # Count
                    idx_owner, round(rel_x, 2), round(rel_y, 2), 1.0
                ])

    # 5. Generate Triangles (Indices)
    for r in range(rows):
        for c in range(cols):
            # Vertex indices
            v1 = r * (cols + 1) + c
            v2 = v1 + 1
            v3 = (r + 1) * (cols + 1) + c
            v4 = v3 + 1
            
            # Two triangles per quad
            triangles.extend([v1, v3, v2])
            triangles.extend([v2, v3, v4])

    return {
        "type": "mesh",
        "uvs": uvs,
        "triangles": triangles,
        "vertices": vertices,
        "hull": 4, # Just the corners? Optional.
        # "edges": ... # Optional for editor lines
        "width": w,
        "height": h
    }

    

def visualize_skeleton(skeleton_json: Dict[str, Any], bg_image_path: str, output_path: str):
    """
    Draws the generated skeleton on top of the character image for debugging.
    """
    import cv2
    import numpy as np
    
    # Load background (Assembly Preview)
    if not os.path.exists(bg_image_path):
        print(f"Debug: Image not found {bg_image_path}")
        return

    img = cv2.imread(bg_image_path, cv2.IMREAD_UNCHANGED)
    if img is None: return
    
    # Create canvas (handle alpha)
    if img.shape[2] == 4:
        # Composite over white for visibility
        alpha = img[:, :, 3] / 255.0
        bg = np.ones_like(img[:, :, :3]) * 255
        for c in range(3):
            bg[:, :, c] = img[:, :, c] * alpha + bg[:, :, c] * (1 - alpha)
        img = bg.astype(np.uint8)
    
    # Get dimensions
    h, w = img.shape[:2]
    cx, cy = w / 2, h / 2
    
    # Helper to convert Spine Coords -> Image Pixel Coords
    # Spine: Center (0,0) is in middle, Y points UP
    # Image: (0,0) is top-left, Y points DOWN
    def to_pixel(sx, sy):
        px = cx + sx
        py = cy - sy # Flip Y
        return int(px), int(py)

    # Calculate Absolute Positions
    bones = skeleton_json["bones"]
    abs_pos = {} # {bone_name: (x, y)}
    
    # 1. Resolve Root
    root = next(b for b in bones if b["name"] == "root")
    abs_pos["root"] = (root["x"], root["y"])
    
    # 2. Resolve Children (Iterative to handle hierarchy order)
    # Simple approach: Iterate multiple times to resolve dependencies
    for _ in range(5): 
        for b in bones:
            if b["name"] in abs_pos: continue
            parent = b.get("parent")
            if parent and parent in abs_pos:
                px, py = abs_pos[parent]
                abs_pos[b["name"]] = (px + b["x"], py + b["y"])

    # Draw Bones
    for b in bones:
        name = b["name"]
        if name not in abs_pos: continue
        
        curr = abs_pos[name]
        start_pt = to_pixel(*curr)
        
        # Draw Joint (Circle)
        color = (0, 0, 255) if "leg" in name else ((0, 255, 0) if "arm" in name else (255, 0, 0))
        cv2.circle(img, start_pt, 5, color, -1)
        
        # Draw Bone (Line to Parent)
        parent = b.get("parent")
        if parent and parent in abs_pos:
            parent_pt = to_pixel(*abs_pos[parent])
            cv2.line(img, start_pt, parent_pt, (200, 200, 200), 2)
            
    # Save
    cv2.imwrite(output_path, img)
    print(f"Skeleton debug saved to: {output_path}")