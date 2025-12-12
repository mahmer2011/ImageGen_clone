"""
Geometric Body Part Detector (v2 - Offset Bones & Core Volume)
- Shifted Arm Bones: Protects chest/shoulders from being eaten by arms.
- Thick Body Bone: Simulates torso volume to claim chest pixels.
- Sleeve Bias: Prioritizes Upper Arm for elbow/baggy areas.
"""
import cv2
import numpy as np
from typing import Dict, List, Tuple
from .part_detector import BodyPartDetector as BaseDetector, BodyPart

class GeometricBodyPartDetector(BaseDetector):
    def detect_all_parts(self) -> Dict[str, BodyPart]:
        # 1. Get Landmarks
        super().detect_all_parts()
        kp = self.keypoints_pixel
        
        if not kp or len(kp) < 5:
            print("Geometric Rigging: No landmarks found.")
            return {}

        print(f"Geometric Rigging: Weighted Partitioning (Offset + Volume)...")
        alpha = self._get_alpha_mask()
        
        def p(name): return kp.get(name)

        # 2. Define Bones
        # Calculate virtual spine points
        neck = None
        pelvis = None
        shoulder_width = 0
        
        if p('left_shoulder') and p('right_shoulder'):
            neck = np.mean([p('left_shoulder'), p('right_shoulder')], axis=0).astype(int)
            shoulder_width = np.linalg.norm(np.array(p('left_shoulder')) - np.array(p('right_shoulder')))
            
        if p('left_hip') and p('right_hip'):
            pelvis = np.mean([p('left_hip'), p('right_hip')], axis=0).astype(int)

        bones = {
            'head':           [p('nose'), p('left_ear'), p('right_ear'), p('left_eye'), p('right_eye')],
            'body':           [neck, pelvis] if (neck is not None and pelvis is not None) else [], 
            
            # Limbs
            'left_upper_arm': [p('left_shoulder'), p('left_elbow')],
            'right_upper_arm':[p('right_shoulder'), p('right_elbow')],
            'left_lower_arm': [p('left_elbow'), p('left_wrist')],
            'right_lower_arm':[p('right_elbow'), p('right_wrist')],
            'left_upper_leg': [p('left_hip'), p('left_knee')],
            'right_upper_leg':[p('right_hip'), p('right_knee')],
            'left_lower_leg': [p('left_knee'), p('left_ankle')],
            'right_lower_leg':[p('right_knee'), p('right_ankle')]
        }

        # 3. Weights (Lower = Stronger Pull)
        # - Body: 0.6 (Strong base)
        # - Upper Arm: 0.8 (Stronger than lower arm to grab sleeves)
        # - Lower Arm: 1.1 (Weak, only grabs hands/wrists)
        weights = {
            'head': 0.8,
            'body': 0.6, 
            'left_upper_arm': 0.8, 
            'right_upper_arm': 0.8,
            'left_lower_arm': 1.1, 
            'right_lower_arm': 1.1,
            'left_upper_leg': 0.9,
            'right_upper_leg': 0.9,
            'left_lower_leg': 1.0,
            'right_lower_leg': 1.0
        }

        h, w = alpha.shape
        part_distances = {name: np.full((h, w), np.inf, dtype=np.float32) for name in bones}
        y_grid, x_grid = np.indices((h, w))
        
        # 4. Calculate Maps
        for name, points in bones.items():
            valid_points = [pt for pt in points if pt is not None]
            if not valid_points: continue
            
            dist_map = None
            
            # Point Cloud (Head)
            if name == 'head':
                for pt in valid_points:
                    d = np.sqrt((x_grid - pt[0])**2 + (y_grid - pt[1])**2)
                    if dist_map is None: dist_map = d
                    else: dist_map = np.minimum(dist_map, d)
            
            # Line Segments (Body/Limbs)
            elif len(valid_points) >= 2:
                # Convert to float for math
                p1 = np.array(valid_points[0]).astype(float)
                p2 = np.array(valid_points[1]).astype(float)

                # --- FIX 1: SHOULDER OFFSET ---
                # Shift Upper Arm start point 15% towards elbow.
                # This leaves a gap at the shoulder for the Body to claim.
                if name in ['left_upper_arm', 'right_upper_arm']:
                    vec = p2 - p1
                    p1 = p1 + (vec * 0.15) 

                # --- FIX 2: THICK BODY BONE ---
                # Draw Body as a thick line (Volume) instead of thin line (Spine).
                # This extends the body's "zero cost" zone outwards to the chest/pecs.
                thickness = 1
                if name == 'body':
                    # Use 25% of shoulder width as core thickness
                    thickness = int(shoulder_width * 0.25)
                    if thickness < 5: thickness = 5
                
                # Draw bone
                bone_mask = np.zeros((h, w), dtype=np.uint8)
                cv2.line(bone_mask, p1.astype(int), p2.astype(int), 255, thickness)
                
                # Dist Transform
                dist_map = cv2.distanceTransform(255 - bone_mask, cv2.DIST_L2, 5)

            if dist_map is not None:
                w_factor = weights.get(name, 1.0)
                part_distances[name] = dist_map * w_factor

        # 5. Assign
        part_names = list(part_distances.keys())
        dist_stack = np.stack([part_distances[name] for name in part_names])
        closest_part_idx = np.argmin(dist_stack, axis=0)
        
        detected_parts = {}
        
        for idx, name in enumerate(part_names):
            mask = np.zeros_like(alpha)
            mask[(closest_part_idx == idx) & (alpha > 0)] = 255
            
            if np.sum(mask) > 0:
                # Cleanup
                kernel_clean = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
                mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_clean)
                
                # Overlap Dilation
                # Body overlaps less (base), Limbs overlap more (rotation)
                dilate_size = 10 if name == 'body' else 20
                kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_size, dilate_size))
                
                expanded = cv2.dilate(mask, kernel_dilate, iterations=1)
                final_mask = cv2.bitwise_and(expanded, alpha)
                
                part = self._create_part_from_mask(name, final_mask)
                if part:
                    detected_parts[name] = part

        return detected_parts