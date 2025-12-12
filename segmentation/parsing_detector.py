"""
Semantic Body Part Detector (v2 - Robust Recovery)
- Auto-reconstructs missing bodies using geometric hull
- Force-splits merged legs if landmarks fail
- Saves debug map to visualize what the AI sees
"""
import cv2
import numpy as np
import torch
import os
from PIL import Image
from transformers import SegformerImageProcessor, AutoModelForSemanticSegmentation
from typing import Dict, List, Optional, Tuple

from .part_detector import BodyPartDetector as BaseDetector, BodyPart

class ParsingBodyPartDetector(BaseDetector):
    def __init__(self, image: np.ndarray):
        super().__init__(image)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load Model
        self.processor = SegformerImageProcessor.from_pretrained("mattmdjaga/segformer_b2_clothes")
        self.model = AutoModelForSemanticSegmentation.from_pretrained("mattmdjaga/segformer_b2_clothes")
        self.model.to(self.device)
        self.model.eval()

        # 0:Background, 1:Hat, 2:Hair, 3:Glove, 4:Sunglasses, 5:UpperClothes
        # 6:Dress, 7:Coat, 8:Socks, 9:Pants, 10:Jumpsuits/Skin
        # 11:Scarf, 12:Skirt, 13:Face, 14:Left-arm, 15:Right-arm
        # 16:Left-leg, 17:Right-leg, 18:Left-shoe, 19:Right-shoe
        self.labels = {
            'head': [1, 2, 4, 13],      
            'body': [5, 6, 7, 10, 11],  
            'arms_parts': [14, 15, 3],  
            'legs_parts': [9, 12, 16, 17, 8], 
            'feet': [18, 19]            
        }

    def detect_all_parts(self) -> Dict[str, BodyPart]:
        # 1. Run MediaPipe for landmarks
        super().detect_all_parts()
        kp = self.keypoints_pixel
        
        print("Semantic Parsing: Running SegFormer (v2 Robust)...")
        
        # 2. Predict
        if self.image.shape[2] == 4:
            rgb = cv2.cvtColor(self.image[:, :, :3], cv2.COLOR_BGR2RGB)
        else:
            rgb = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        
        pil_img = Image.fromarray(rgb)
        inputs = self.processor(images=pil_img, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits.cpu()

        upsampled_logits = torch.nn.functional.interpolate(
            logits, size=pil_img.size[::-1], mode="bilinear", align_corners=False,
        )
        pred_seg = upsampled_logits.argmax(dim=1)[0].numpy().astype(np.uint8)
        
        # DEBUG: Save visualization
        self._save_debug_map(pred_seg)

        detected_parts = {}
        
        # --- HEAD & FACE ---
        # If face(13) is missing but hair(2) exists, try to recover face area
        # by checking holes inside the Hair+Body union? 
        # For now, we trust the model but dilate heavily.
        head_mask = self._get_merged_mask(pred_seg, self.labels['head'])
        head_mask = self._dilate(head_mask, 10) 
        self._add_part(detected_parts, 'head', head_mask)

        # --- BODY RECOVERY ---
        body_mask = self._get_merged_mask(pred_seg, self.labels['body'])
        
        # SAFETY NET: If body is too small (e.g. < 5% of image), reconstruct it
        total_pixels = self.width * self.height
        if np.sum(body_mask) < (total_pixels * 0.05):
            print("  ⚠️ Body missing or too small. Attempting geometric reconstruction...")
            body_mask = self._reconstruct_body_geometric(kp, head_mask)
        
        # Fill holes inside body
        body_mask = self._fill_holes(body_mask)
        self._add_part(detected_parts, 'body', body_mask)

        # --- ARMS ---
        # Left
        l_arm_mask = self._get_mask(pred_seg, 14) # Left-arm
        l_glove = self._get_mask(pred_seg, 3)     # Glove (assume left if on left side)
        
        # Assign gloves to side
        if np.sum(l_glove) > 0:
            h, w = l_glove.shape
            left_glove_mask = l_glove.copy()
            left_glove_mask[:, w//2:] = 0
            l_arm_mask = cv2.bitwise_or(l_arm_mask, left_glove_mask)

        if kp and kp.get('left_elbow'):
            u_mask, l_mask = self._split_limb_at_joint(l_arm_mask, kp.get('left_elbow'), kp.get('left_wrist'))
            self._add_part(detected_parts, 'left_upper_arm', self._dilate(u_mask, 10))
            self._add_part(detected_parts, 'left_lower_arm', self._dilate(l_mask, 10))
        else:
            self._add_part(detected_parts, 'left_upper_arm', l_arm_mask)

        # Right
        r_arm_mask = self._get_mask(pred_seg, 15)
        # Assign right glove
        if np.sum(l_glove) > 0:
            right_glove_mask = l_glove.copy()
            right_glove_mask[:, :w//2] = 0
            r_arm_mask = cv2.bitwise_or(r_arm_mask, right_glove_mask)

        if kp and kp.get('right_elbow'):
            u_mask, l_mask = self._split_limb_at_joint(r_arm_mask, kp.get('right_elbow'), kp.get('right_wrist'))
            self._add_part(detected_parts, 'right_upper_arm', self._dilate(u_mask, 10))
            self._add_part(detected_parts, 'right_lower_arm', self._dilate(l_mask, 10))
        else:
            self._add_part(detected_parts, 'right_upper_arm', r_arm_mask)

        # --- LEGS ---
        leg_group = self._get_merged_mask(pred_seg, self.labels['legs_parts'])
        
        # Split merged legs (Pants/Skirt)
        if kp and kp.get('left_knee') and kp.get('right_knee'):
            l_full, r_full = self._spatial_split_mask(leg_group, kp['left_knee'], kp['right_knee'])
        else:
            print("  ⚠️ Missing knee landmarks. Forcing vertical split for legs.")
            l_full, r_full = self._force_vertical_split(leg_group)

        # Split Upper/Lower
        # Left
        lu, ll = self._split_limb_at_joint(l_full, kp.get('left_knee'), kp.get('left_ankle'))
        self._add_part(detected_parts, 'left_upper_leg', self._dilate(lu, 15))
        self._add_part(detected_parts, 'left_lower_leg', self._dilate(ll, 10))
        
        # Right
        ru, rl = self._split_limb_at_joint(r_full, kp.get('right_knee'), kp.get('right_ankle'))
        self._add_part(detected_parts, 'right_upper_leg', self._dilate(ru, 15))
        self._add_part(detected_parts, 'right_lower_leg', self._dilate(rl, 10))

        # --- FEET ---
        # Try specific shoes classes first
        l_shoe = self._get_mask(pred_seg, 18)
        r_shoe = self._get_mask(pred_seg, 19)
        
        # If missing, try to steal from lower leg bottom?
        # For now, just add what we found
        self._add_part(detected_parts, 'left_foot', self._dilate(l_shoe, 5))
        self._add_part(detected_parts, 'right_foot', self._dilate(r_shoe, 5))

        return detected_parts

    def _reconstruct_body_geometric(self, kp, head_mask):
        """
        Creates a fallback body mask by connecting shoulders and hips.
        Useful when the AI thinks the torso is 'background'.
        """
        mask = np.zeros((self.height, self.width), dtype=np.uint8)
        
        # Get points
        ls = kp.get('left_shoulder')
        rs = kp.get('right_shoulder')
        lh = kp.get('left_hip')
        rh = kp.get('right_hip')
        
        # If we have all 4 points, draw a quad
        if ls and rs and lh and rh:
            pts = np.array([ls, rs, rh, lh], dtype=np.int32)
            cv2.fillConvexPoly(mask, pts, 255)
            # Dilate heavily to look like a torso
            mask = self._dilate(mask, 40)
        elif head_mask is not None:
            # Panic fallback: Take region below head
            y, x = np.where(head_mask > 0)
            if len(y) > 0:
                bottom_head = np.max(y)
                center_head = int(np.mean(x))
                # Draw a generic box below head
                cv2.rectangle(mask, (center_head-50, bottom_head), (center_head+50, bottom_head+200), 255, -1)
        
        # Intersect with original image alpha to don't go out of bounds
        alpha = self._get_alpha_mask()
        mask = cv2.bitwise_and(mask, alpha)
        
        # Remove head from body
        if head_mask is not None:
            mask[head_mask > 0] = 0
            
        return mask

    def _force_vertical_split(self, mask):
        """Splits a mask vertically in the middle."""
        if np.sum(mask) == 0:
            return mask, mask
            
        # Find bounding box of the mask
        y, x = np.where(mask > 0)
        if len(x) == 0: return mask, mask
        
        center_x = int(np.mean(x))
        
        left_mask = mask.copy()
        left_mask[:, center_x:] = 0 # Keep left side
        
        right_mask = mask.copy()
        right_mask[:, :center_x] = 0 # Keep right side
        
        return left_mask, right_mask

    def _save_debug_map(self, pred_seg):
        """Saves a color-coded map of what the AI detected."""
        # Map indices to distinct colors
        debug_img = np.zeros((pred_seg.shape[0], pred_seg.shape[1], 3), dtype=np.uint8)
        
        # Random colors for classes 1-19
        np.random.seed(42)
        colors = np.random.randint(0, 255, (20, 3))
        colors[0] = [0, 0, 0] # Background black
        
        for cls_id in range(20):
            debug_img[pred_seg == cls_id] = colors[cls_id]
            
        # Save to current working dir or similar
        try:
            cv2.imwrite("segformer_debug.png", debug_img)
        except:
            pass

    # ... (Keep _get_mask, _get_merged_mask, _add_part, _dilate, _fill_holes, _split_limb_at_joint, _spatial_split_mask from previous version) ...
    def _get_mask(self, seg, class_id):
        return np.where(seg == class_id, 255, 0).astype(np.uint8)

    def _get_merged_mask(self, seg, class_ids):
        mask = np.zeros_like(seg, dtype=np.uint8)
        for cid in class_ids:
            mask = cv2.bitwise_or(mask, self._get_mask(seg, cid))
        return mask

    def _add_part(self, parts_dict, name, mask):
        if np.sum(mask) > 0:
            part = self._create_part_from_mask(name, mask)
            if part:
                parts_dict[name] = part

    def _dilate(self, mask, pixels):
        if np.sum(mask) == 0: return mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (pixels, pixels))
        return cv2.dilate(mask, kernel, iterations=1)

    def _fill_holes(self, mask):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filled = np.zeros_like(mask)
        cv2.drawContours(filled, contours, -1, 255, -1)
        return filled

    def _split_limb_at_joint(self, limb_mask, joint, next_joint):
        if np.sum(limb_mask) == 0 or not joint:
            return limb_mask, np.zeros_like(limb_mask)
        if not next_joint: # Fallback if wrist/ankle missing
            return limb_mask, np.zeros_like(limb_mask)

        vec = np.array(next_joint) - np.array(joint)
        h, w = limb_mask.shape
        Y, X = np.ogrid[:h, :w]
        
        # Project: (P - Joint) dot Vector
        projection = (X - joint[0]) * vec[0] + (Y - joint[1]) * vec[1]
        
        lower_mask = np.where(projection > 0, limb_mask, 0).astype(np.uint8)
        upper_mask = np.where(projection <= 0, limb_mask, 0).astype(np.uint8)
        return upper_mask, lower_mask

    def _spatial_split_mask(self, mask, left_point, right_point):
        if np.sum(mask) == 0: return mask, mask
        h, w = mask.shape
        Y, X = np.ogrid[:h, :w]
        dist_L = (X - left_point[0])**2 + (Y - left_point[1])**2
        dist_R = (X - right_point[0])**2 + (Y - right_point[1])**2
        left_mask = np.where((dist_L < dist_R), mask, 0).astype(np.uint8)
        right_mask = np.where((dist_R <= dist_L), mask, 0).astype(np.uint8)
        return left_mask, right_mask