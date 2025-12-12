"""
SAM-Based Body Part Detector (v5 - Asymmetric Padding & Low Negatives)
- Asymmetric Boxes: Expands UP for hair and DOWN for feet.
- Minimal Negatives: Prevents "half-leg" issues by removing opposing joint constraints.
- Deep Overlap: Allows parts to overlap significantly to prevent gaps.
"""
import os
import cv2
import numpy as np
import torch
import traceback
from typing import Dict, List, Optional, Tuple

try:
    from mobile_sam import sam_model_registry, SamPredictor
    MOBILE_SAM_AVAILABLE = True
except ImportError:
    MOBILE_SAM_AVAILABLE = False
    print("Warning: 'mobile_sam' not installed. Falling back to standard detector.")

from .part_detector import BodyPartDetector as BaseDetector, BodyPart

class SamBodyPartDetector(BaseDetector):
    def __init__(self, image: np.ndarray, model_type="vit_t", checkpoint="weights/mobile_sam.pt"):
        super().__init__(image)
        self.use_fallback = False
        
        if not MOBILE_SAM_AVAILABLE or not os.path.exists(checkpoint):
            self.use_fallback = True
            return

        try:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.sam = sam_model_registry[model_type](checkpoint=checkpoint)
            self.sam.to(device=self.device)
            self.sam.eval()
            self.predictor = SamPredictor(self.sam)
            
            if self.image.shape[2] == 4:
                rgb = cv2.cvtColor(self.image[:, :, :3], cv2.COLOR_BGR2RGB)
            else:
                rgb = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
            
            self.predictor.set_image(rgb)
            
        except Exception as e:
            print(f"SAM Init Error: {e}")
            self.use_fallback = True

    def detect_all_parts(self) -> Dict[str, BodyPart]:
        base_parts = super().detect_all_parts()
        if self.use_fallback: return base_parts

        kp = self.keypoints_pixel
        if not kp or len(kp) < 5: return base_parts

        print(f"SAM: Processing {len(kp)} landmarks (v5 Config)...")
        detected_parts = {}
        
        # 1. AGGRESSIVE LANDMARK RECOVERY
        # Extend vectors further to catch large boots/shoes
        self._recover_missing_foot(kp, 'left', scale=1.6)
        self._recover_missing_foot(kp, 'right', scale=1.6)

        def p(*names): return [kp.get(n) for n in names if kp.get(n)]

        # 2. ASYMMETRIC PADDING CONFIG
        # Format: (Left, Top, Right, Bottom) - Multipliers of width/height
        # Example: (0.2, 1.0, 0.2, 0.5) means add 100% height to Top, 50% to Bottom
        
        parts_def = {
            'head': {
                'box': p('nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear'),
                'pad_4': (0.5, 1.5, 0.5, 0.5), # TOP 1.5x = Massive space for hair
                'neg': [] # No negatives. Let it grab neck/shoulders.
            },
            'body': {
                'box': p('left_shoulder', 'right_shoulder', 'left_hip', 'right_hip'),
                'pad_4': (0.5, 0.5, 0.5, 0.5), # Wide box for coat/armor
                'neg': p('left_wrist', 'right_wrist') # Only avoid Hands. Let it grab knees.
            },
            
            # ARMS
            'left_upper_arm': {
                'box': p('left_shoulder', 'left_elbow'),
                'pad_4': (0.5, 0.5, 0.5, 0.5),
                'neg': p('body_center') # Only avoid deep center torso
            },
            'right_upper_arm': {
                'box': p('right_shoulder', 'right_elbow'),
                'pad_4': (0.5, 0.5, 0.5, 0.5),
                'neg': p('body_center')
            },
            'left_lower_arm': {
                'box': p('left_elbow', 'left_wrist', 'left_index'),
                'pad_4': (0.4, 0.4, 0.4, 0.4),
                'neg': p('left_shoulder') # Avoid shoulder
            },
            'right_lower_arm': {
                'box': p('right_elbow', 'right_wrist', 'right_index'),
                'pad_4': (0.4, 0.4, 0.4, 0.4),
                'neg': p('right_shoulder')
            },
            
            # LEGS - REMOVED OPPOSING NEGATIVES
            'left_upper_leg': {
                'box': p('left_hip', 'left_knee'),
                'pad_4': (0.6, 0.2, 0.6, 0.2), # Wide for skirts/baggy pants
                'neg': [] # No negatives!
            },
            'right_upper_leg': {
                'box': p('right_hip', 'right_knee'),
                'pad_4': (0.6, 0.2, 0.6, 0.2),
                'neg': [] 
            },
            'left_lower_leg': {
                'box': p('left_knee', 'left_ankle', 'left_foot_index'),
                'pad_4': (0.5, 0.2, 0.5, 0.8), # BOTTOM 0.8x = Massive space for Boots
                'neg': []
            },
            'right_lower_leg': {
                'box': p('right_knee', 'right_ankle', 'right_foot_index'),
                'pad_4': (0.5, 0.2, 0.5, 0.8), # BOTTOM 0.8x
                'neg': []
            }
        }

        # 3. PREDICT
        try:
            for name, config in parts_def.items():
                if not config['box']: continue

                # Get Asymmetric Box
                box = self._get_bounding_box(config['box'], config.get('pad_4', (0.2, 0.2, 0.2, 0.2)))
                
                # Predict
                mask = self._predict_mask(box, config['neg'])
                
                if mask is not None:
                    alpha = self._get_alpha_mask()
                    mask = cv2.bitwise_and(mask, alpha)
                    
                    # Cleanup: Open then Close to remove noise but keep solid shapes
                    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
                    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
                    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
                    
                    part = self._create_part_from_mask(name, mask)
                    if part:
                        detected_parts[name] = part
            
            if len(detected_parts) < 3: return base_parts
            return detected_parts

        except Exception as e:
            print(f"SAM Error: {e}")
            traceback.print_exc()
            return base_parts

    def _recover_missing_foot(self, kp, side, scale=1.6):
        """If foot/ankle is missing, guess where it is."""
        knee = kp.get(f'{side}_knee')
        hip = kp.get(f'{side}_hip')
        foot = kp.get(f'{side}_foot_index')
        ankle = kp.get(f'{side}_ankle')
        
        if knee and hip:
            # Always recalc foot if it's missing OR if it seems too close to ankle
            # Vector(Hip->Knee)
            vec = np.array(knee) - np.array(hip)
            
            h, w = self.image.shape[:2]
            
            if not ankle:
                new_ankle = np.array(knee) + (vec * 1.0)
                kp[f'{side}_ankle'] = (int(np.clip(new_ankle[0], 0, w)), int(np.clip(new_ankle[1], 0, h)))
            
            if not foot:
                # Extend significantly for the foot tip
                new_foot = np.array(knee) + (vec * scale)
                kp[f'{side}_foot_index'] = (int(np.clip(new_foot[0], 0, w)), int(np.clip(new_foot[1], 0, h)))

    def _get_bounding_box(self, points, pads):
        """
        Create 4-way asymmetric bounding box.
        pads: (Left, Top, Right, Bottom) multipliers
        """
        pts = np.array(points)
        x_min, y_min = np.min(pts, axis=0)
        x_max, y_max = np.max(pts, axis=0)
        
        w = x_max - x_min
        h = y_max - y_min
        
        # Calculate pixels for each side
        p_l = int(w * pads[0]) + 15
        p_t = int(h * pads[1]) + 15
        p_r = int(w * pads[2]) + 15
        p_b = int(h * pads[3]) + 15
        
        return np.array([
            max(0, x_min - p_l), 
            max(0, y_min - p_t),
            min(self.width, x_max + p_r), 
            min(self.height, y_max + p_b)
        ])

    def _predict_mask(self, box, neg_points):
        neg = [p for p in neg_points if p]
        
        # Anchor point: Center of the box
        center_x = (box[0] + box[2]) / 2
        center_y = (box[1] + box[3]) / 2
        
        coords = [[center_x, center_y]] + neg
        labels = [1] + [0] * len(neg)
        
        masks, _, _ = self.predictor.predict(
            point_coords=np.array(coords),
            point_labels=np.array(labels),
            box=box[None, :],
            multimask_output=False
        )
        return (masks[0] * 255).astype(np.uint8)