

"""
Body Part Detector with CLEAN Separation
No overlapping parts - each pixel belongs to only ONE body part
"""

import cv2
import numpy as np
import mediapipe as mp
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class BodyPart:
    """Represents a detected body part."""
    name: str
    bbox: Tuple[int, int, int, int]
    contour: np.ndarray
    center: Tuple[int, int]
    area: float
    mask: np.ndarray
    keypoints: Optional[Dict[str, Tuple[int, int]]] = None
    overlap_padding: int = 0


class BodyPartDetector:
    """Detects body parts with ZERO overlap - clean segmentation."""
    
    def __init__(self, image: np.ndarray):
        self.image = image
        self.height, self.width = image.shape[:2]
        self.parts: Dict[str, BodyPart] = {}
        self.pose_landmarks = None
        self.keypoints_pixel = {}
        
        # Priority map for conflict resolution (higher = higher priority)
        self.PART_PRIORITY = {
            'head': 10,           # Head claims neck pixels
            'left_lower_arm': 9,  # Hands over body
            'right_lower_arm': 9,
            'left_upper_arm': 8,
            'right_upper_arm': 8,
            'left_lower_leg': 7,  # Boots over pants
            'right_lower_leg': 7,
            'left_upper_leg': 6,
            'right_upper_leg': 6,
            'body': 5             # Body is the base layer
        }
        self.DEFAULT_OVERLAP_PIXELS = 20
        self.DEFAULT_OVERLAP_RATIO = 0.05
        self.OVERLAP_RULES = {
            'left_lower_leg': {
                'parent': 'left_upper_leg',
                'pixels': 32,
                'ratio': 0.08,
                'parent_ratio': 0.28,
                'parent_pad': 22,
                'min_overlap_area': 2800,
                'side_expansion': 14
            },
            'right_lower_leg': {
                'parent': 'right_upper_leg',
                'pixels': 32,
                'ratio': 0.08,
                'parent_ratio': 0.28,
                'parent_pad': 22,
                'min_overlap_area': 2800,
                'side_expansion': 14
            },
            'left_lower_arm': {
                'parent': 'left_upper_arm',
                'pixels': 26,
                'ratio': 0.08,
                'parent_ratio': 0.22,
                'parent_pad': 16,
                'min_overlap_area': 1800,
                'side_expansion': 12
            },
            'right_lower_arm': {
                'parent': 'right_upper_arm',
                'pixels': 26,
                'ratio': 0.08,
                'parent_ratio': 0.22,
                'parent_pad': 16,
                'min_overlap_area': 1800,
                'side_expansion': 12
            },
        }
        self.JOINT_OVERLAP_MIN_AREA = 900
        self.JOINT_BAND_PADDING = 12
        self.JOINT_BAND_PARENT_PAD = 10
        self.JOINT_SIDE_EXPANSION = 8
        self.JOINT_EXCLUDE_PARTS = {'body', 'head'}
        
        # MediaPipe setup - enhanced for non-human characters
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        # Lower confidence threshold and use highest model complexity
        self.pose = self.mp_pose.Pose(
            static_image_mode=True,
            model_complexity=2,  # Highest complexity for better detection
            enable_segmentation=True,
            min_detection_confidence=0.1,  # Very low threshold for non-human characters
            min_tracking_confidence=0.1
        )
        
        self.LANDMARK_NAMES = {
            0: 'nose', 1: 'left_eye_inner', 2: 'left_eye', 3: 'left_eye_outer',
            4: 'right_eye_inner', 5: 'right_eye', 6: 'right_eye_outer',
            7: 'left_ear', 8: 'right_ear', 9: 'mouth_left', 10: 'mouth_right',
            11: 'left_shoulder', 12: 'right_shoulder',
            13: 'left_elbow', 14: 'right_elbow',
            15: 'left_wrist', 16: 'right_wrist',
            17: 'left_pinky', 18: 'right_pinky',
            19: 'left_index', 20: 'right_index',
            21: 'left_thumb', 22: 'right_thumb',
            23: 'left_hip', 24: 'right_hip',
            25: 'left_knee', 26: 'right_knee',
            27: 'left_ankle', 28: 'right_ankle',
            29: 'left_heel', 30: 'right_heel',
            31: 'left_foot_index', 32: 'right_foot_index'
        }
    
    def detect_all_parts(self) -> Dict[str, BodyPart]:
        """Main detection with clean separation and enhanced MediaPipe."""
        print("Detecting pose landmarks with MediaPipe (enhanced mode)...")
        
        # Try multiple preprocessing strategies to help MediaPipe detect non-human characters
        preprocessing_strategies = [
            ("original", lambda img: img),
            ("enhanced_contrast", self._enhance_contrast),
            ("edge_enhanced", self._enhance_edges),
            ("normalized", self._normalize_image),
        ]
        
        best_results = None
        best_keypoints = {}
        best_strategy = None
        
        # Convert base image to RGB
        if self.image.shape[2] == 4:
            base_rgb = cv2.cvtColor(self.image[:, :, :3], cv2.COLOR_BGR2RGB)
        else:
            base_rgb = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        
        # Try each preprocessing strategy
        for strategy_name, preprocess_func in preprocessing_strategies:
            try:
                processed_image = preprocess_func(base_rgb.copy())
                results = self.pose.process(processed_image)
                
                if results.pose_landmarks:
                    keypoints = self._extract_keypoints(results.pose_landmarks)
                    valid_count = len(keypoints)
                    
                    # Keep the result with most keypoints
                    if valid_count > len(best_keypoints):
                        best_results = results
                        best_keypoints = keypoints
                        best_strategy = strategy_name
                        print(f"  Strategy '{strategy_name}': {valid_count} keypoints")
            except Exception as e:
                print(f"  Strategy '{strategy_name}' failed: {e}")
                continue
        
        if not best_results or not best_results.pose_landmarks:
            print("WARNING: No pose detected with any preprocessing, using fallback...")
            return self._fallback_contour_detection()
        
        print(f"Best strategy: '{best_strategy}' with {len(best_keypoints)} keypoints")
        self.pose_landmarks = best_results.pose_landmarks
        self.keypoints_pixel = best_keypoints
        
        if len(self.keypoints_pixel) < 6:  # Lowered threshold
            print(f"WARNING: Only {len(self.keypoints_pixel)} keypoints detected, using fallback...")
            return self._fallback_contour_detection()
        
        # Get alpha mask
        alpha_mask = self._get_alpha_mask()
        
        # Create assignment map (which part owns each pixel)
        assignment_map = np.zeros((self.height, self.width), dtype=np.uint8)
        
        # Define regions with boundaries
        regions = self._define_clean_regions()
        
        # Assign pixels to parts (NO OVERLAP)
        part_masks = {}
        for part_name, region_func in regions.items():
            mask = region_func(alpha_mask, assignment_map)
            if mask is not None and np.sum(mask > 0) > 100:
                part_masks[part_name] = mask
                # Mark these pixels as assigned
                assignment_map[mask > 0] = self.PART_PRIORITY.get(part_name, 1)
                print(f"  {part_name}: {np.sum(mask > 0)} pixels")
        
        # Apply multi-scale silhouette expansion to capture edge pixels that were missed
        refined_masks = {}
        for part_name, mask in part_masks.items():
            refined_mask = self._multi_scale_expand_mask(part_name, mask, alpha_mask, assignment_map)
            refined_masks[part_name] = refined_mask
        
        # Create BodyPart objects from refined masks
        detected_parts = {}
        for part_name, mask in refined_masks.items():
            part = self._create_part_from_mask(part_name, mask)
            if part:
                detected_parts[part_name] = part

        # Ensure entire silhouette is covered so no pixels remain missing
        detected_parts = self._ensure_silhouette_coverage(detected_parts, alpha_mask)
        detected_parts = self._apply_overlap_expansion(detected_parts)

        print(f"Segmented {len(detected_parts)} parts cleanly")
        return detected_parts

    def _multi_scale_expand_mask(
        self,
        part_name: str,
        mask: np.ndarray,
        alpha_mask: np.ndarray,
        assignment_map: np.ndarray
    ) -> np.ndarray:
        """Expand part mask in multiple distance bands to recover thin details."""
        part_priority = self.PART_PRIORITY.get(part_name, 1)
        expanded_mask = mask.copy()
        mask_binary = (expanded_mask > 0).astype(np.uint8) * 255

        for radius in (4, 8, 12):
            # Distance from current mask (0 for mask pixels)
            dist = cv2.distanceTransform(cv2.bitwise_not(mask_binary), cv2.DIST_L2, 5)
            band = ((dist > 0) & (dist <= radius)).astype(np.uint8) * 255

            # Stay inside silhouette
            band = cv2.bitwise_and(band, alpha_mask)

            # Respect higher-priority assignments
            band[assignment_map > part_priority] = 0

            before_pixels = np.sum(expanded_mask > 0)
            expanded_mask = cv2.bitwise_or(expanded_mask, band)
            mask_binary = (expanded_mask > 0).astype(np.uint8) * 255

            # Update assignment map for new pixels
            assignment_map[band > 0] = part_priority

            after_pixels = np.sum(expanded_mask > 0)
            if after_pixels > before_pixels:
                print(f"  {part_name} (radius {radius}): +{after_pixels - before_pixels} pixels")

        return expanded_mask

    def _ensure_silhouette_coverage(
        self,
        parts: Dict[str, BodyPart],
        alpha_mask: np.ndarray
    ) -> Dict[str, BodyPart]:
        """Assign any remaining silhouette pixels to the nearest part."""
        if not parts:
            return parts

        combined_mask = np.zeros_like(alpha_mask)
        for part in parts.values():
            combined_mask = cv2.bitwise_or(combined_mask, part.mask)

        missing = cv2.bitwise_and(alpha_mask, cv2.bitwise_not(combined_mask))
        missing_pixels = np.sum(missing > 0)
        if missing_pixels == 0:
            return parts

        print(f"  Filling {missing_pixels} unassigned silhouette pixels...")

        part_names = list(parts.keys())
        distance_stack = []
        for name in part_names:
            mask_binary = (parts[name].mask > 0).astype(np.uint8) * 255
            dist = cv2.distanceTransform(cv2.bitwise_not(mask_binary), cv2.DIST_L2, 5)
            distance_stack.append(dist)

        distance_stack = np.stack(distance_stack, axis=0)
        closest_indices = np.argmin(distance_stack, axis=0)

        for idx, name in enumerate(part_names):
            addition = np.zeros_like(alpha_mask)
            addition[(closest_indices == idx) & (missing > 0)] = 255
            if np.any(addition):
                parts[name].mask = cv2.bitwise_or(parts[name].mask, addition)

        # Recreate BodyPart objects to update bbox/center after adjustments
        updated_parts = {}
        for name in part_names:
            updated = self._create_part_from_mask(name, parts[name].mask)
            if updated:
                updated_parts[name] = updated

        return updated_parts

    def _apply_overlap_expansion(self, parts: Dict[str, BodyPart]) -> Dict[str, BodyPart]:
        """Extend selected parts upward with a rounded overlap into their parent regions."""
        if not parts:
            return parts

        for part_name, rule in self.OVERLAP_RULES.items():
            parent_name = rule['parent']
            if (
                part_name not in parts or
                parent_name not in parts or
                part_name in self.JOINT_EXCLUDE_PARTS or
                parent_name in self.JOINT_EXCLUDE_PARTS
            ):
                continue

            child = parts[part_name]
            parent = parts[parent_name]
            expansion = self._calculate_overlap_amount(child, rule, parent)
            if expansion <= 0:
                continue

            new_mask, added_pixels, effective_expansion = self._create_d_shape_overlap(
                child.mask,
                parent.mask,
                child.bbox,
                expansion
            )

            if added_pixels == 0 or effective_expansion == 0:
                continue

            parts[part_name] = self._build_part_with_new_mask(
                child,
                new_mask,
                effective_expansion
            )

        return self._ensure_joint_overlap(parts)

    def _ensure_joint_overlap(self, parts: Dict[str, BodyPart]) -> Dict[str, BodyPart]:
        """
        Guarantee that every child part shares some pixels with its parent part.
        This creates a small buffer of overlapping pixels so rigging software
        has room to hide seams during animation.
        """
        for child_name, rule in self.OVERLAP_RULES.items():
            parent_name = rule['parent']
            if (
                child_name not in parts or
                parent_name not in parts or
                child_name in self.JOINT_EXCLUDE_PARTS or
                parent_name in self.JOINT_EXCLUDE_PARTS
            ):
                continue

            child = parts[child_name]
            parent = parts[parent_name]

            overlap_mask = cv2.bitwise_and(child.mask, parent.mask)
            overlap_area = int(np.sum(overlap_mask > 0))
            min_required = rule.get('min_overlap_area', self.JOINT_OVERLAP_MIN_AREA)

            if overlap_area >= min_required:
                continue

            joint_band = self._extract_parent_band(parent, child, rule)
            added_pixels = int(np.sum(joint_band > 0))
            if added_pixels == 0:
                continue

            updated_child_mask = cv2.bitwise_or(child.mask, joint_band)
            parts[child_name] = self._build_part_with_new_mask(
                child,
                updated_child_mask,
                rule.get('pixels', self.DEFAULT_OVERLAP_PIXELS)
            )

        return parts

    def _extract_parent_band(
        self,
        parent: BodyPart,
        child: BodyPart,
        rule: Dict
    ) -> np.ndarray:
        """
        Slice a band from the parent mask around the joint area so it can be
        merged into the child mask. The band height scales with the requested
        overlap pixels or ratio.
        """
        band = np.zeros_like(parent.mask)
        height, width = band.shape

        requested_pixels = self._calculate_overlap_amount(child, rule, parent)
        if requested_pixels <= 0:
            return band

        child_x, child_y, child_w, child_h = child.bbox
        pad = rule.get('child_pad', self.JOINT_BAND_PADDING)
        if rule.get('use_parent_extent', True):
            parent_pad = rule.get('parent_pad', self.JOINT_BAND_PARENT_PAD)
            roi_left = max(0, parent.bbox[0] - parent_pad)
            roi_right = min(width, parent.bbox[0] + parent.bbox[2] + parent_pad)
        else:
            roi_left = max(0, child_x - pad)
            roi_right = min(width, child_x + child_w + pad)

        parent_top = parent.bbox[1]
        parent_bottom = parent.bbox[1] + parent.bbox[3]

        if parent.center[1] <= child.center[1]:
            # Parent is above child (typical case). Copy bottom strip of parent.
            strip_start = max(parent_top, parent_bottom - requested_pixels)
            strip_end = parent_bottom
        else:
            # Parent is below child (e.g., head vs body). Copy top strip.
            strip_start = parent_top
            strip_end = min(parent_bottom, parent_top + requested_pixels)

        if strip_end <= strip_start:
            return band

        band[strip_start:strip_end, roi_left:roi_right] = parent.mask[
            strip_start:strip_end, roi_left:roi_right
        ]

        if np.any(band):
            side_expand = rule.get('side_expansion', self.JOINT_SIDE_EXPANSION)
            kernel_width = max(3, side_expand * 2 + 1)
            kernel_height = max(3, requested_pixels if requested_pixels % 2 == 1 else requested_pixels + 1)
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (kernel_width, min(kernel_height, kernel_width * 2))
            )
            band = cv2.dilate(band, kernel, iterations=1)
            band = cv2.bitwise_and(band, parent.mask)

        return band

    def _calculate_overlap_amount(
        self,
        part: BodyPart,
        rule: Dict,
        parent: Optional[BodyPart] = None
    ) -> int:
        """Determine how many pixels to extend upward for a part."""
        base_pixels = rule.get('pixels', self.DEFAULT_OVERLAP_PIXELS)
        ratio = rule.get('ratio', self.DEFAULT_OVERLAP_RATIO)
        ratio_pixels = int(max(part.bbox[3], 1) * ratio)

        parent_ratio = rule.get('parent_ratio')
        parent_pixels = 0
        if parent is not None:
            if parent_ratio is not None:
                parent_pixels = int(max(parent.bbox[3], 1) * parent_ratio)
            parent_pixels = max(parent_pixels, rule.get('parent_pixels', 0))
        else:
            parent_pixels = rule.get('parent_pixels', 0)

        requested = max(base_pixels, ratio_pixels, parent_pixels, 0)
        max_pixels = rule.get('max_pixels')
        if max_pixels is not None:
            requested = min(requested, max_pixels)
        return requested

    def _create_d_shape_overlap(
        self,
        child_mask: np.ndarray,
        parent_mask: np.ndarray,
        bbox: Tuple[int, int, int, int],
        requested_expansion: int
    ) -> Tuple[np.ndarray, int, int]:
        """Generate a curved top cap for overlap constrained by the parent mask."""
        if requested_expansion <= 0:
            return child_mask, 0, 0

        child_binary = ((child_mask > 0).astype(np.uint8)) * 255
        parent_binary = ((parent_mask > 0).astype(np.uint8)) * 255

        coords = cv2.findNonZero(child_binary)
        if coords is None:
            return child_mask, 0, 0

        _, top_y, _, _ = cv2.boundingRect(coords)
        if top_y <= 0:
            return child_mask, 0, 0

        top_band_start = max(0, top_y - requested_expansion)
        effective_expansion = top_y - top_band_start
        if effective_expansion <= 0:
            return child_mask, 0, 0

        height, width = child_binary.shape

        top_row = child_binary[top_y]
        active_cols = np.where(top_row > 0)[0]
        if active_cols.size == 0:
            band_end = min(height, top_y + 5)
            active_cols = np.where(np.any(child_binary[top_y:band_end] > 0, axis=0))[0]

        if active_cols.size == 0:
            left_col = bbox[0]
            right_col = min(width - 1, bbox[0] + bbox[2] - 1)
        else:
            left_col = int(active_cols.min())
            right_col = int(active_cols.max())

        span = max(1, right_col - left_col + 1)
        width_margin = max(4, int(span * 0.2))
        radius_x = max(3, span // 2 + width_margin)
        radius_y = max(2, effective_expansion)

        center_x = int(np.clip((left_col + right_col) / 2, 0, width - 1))
        center_y = int(top_y)

        cap_mask = np.zeros_like(child_binary)
        rect_top = max(0, center_y - radius_y)
        rect_left = max(0, center_x - radius_x)
        rect_right = min(width - 1, center_x + radius_x)
        cv2.rectangle(cap_mask, (rect_left, rect_top), (rect_right, center_y), 255, -1)
        cv2.ellipse(cap_mask, (center_x, center_y), (radius_x, radius_y), 0, 0, 180, 255, -1)
        cap_mask[center_y + 1 :, :] = 0

        overlap_candidate = cv2.bitwise_and(cap_mask, cv2.bitwise_not(child_binary))
        overlap_candidate = cv2.bitwise_and(overlap_candidate, parent_binary)
        if not np.any(overlap_candidate):
            return child_mask, 0, 0

        candidate_bool = overlap_candidate > 0
        column_indices = np.where(np.any(candidate_bool, axis=0))[0]
        for col in column_indices:
            col_data = candidate_bool[:, col]
            first_idx = np.argmax(col_data)
            if col_data[first_idx]:
                overlap_candidate[first_idx:center_y, col] = 255

        smooth_w = max(3, ((radius_x // 4) * 2) + 1)
        smooth_h = max(3, ((max(1, radius_y // 2)) * 2) + 1)
        smooth_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (smooth_w, smooth_h))
        overlap_candidate = cv2.morphologyEx(overlap_candidate, cv2.MORPH_CLOSE, smooth_kernel, iterations=1)
        overlap_candidate = cv2.bitwise_and(overlap_candidate, parent_binary)
        overlap_candidate[center_y:, :] = 0

        added_pixels = int(np.sum(overlap_candidate > 0))
        if added_pixels == 0:
            return child_mask, 0, 0

        new_mask = cv2.bitwise_or(child_binary, overlap_candidate)
        return new_mask, added_pixels, effective_expansion

    def _build_part_with_new_mask(
        self,
        original_part: BodyPart,
        updated_mask: np.ndarray,
        overlap_padding: int
    ) -> BodyPart:
        """Create a refreshed BodyPart using the updated mask while keeping bbox."""
        contours, _ = cv2.findContours(updated_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            original_part.mask = updated_mask
            original_part.overlap_padding = max(original_part.overlap_padding, overlap_padding)
            return original_part

        main_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(main_contour)

        M = cv2.moments(main_contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        else:
            cx, cy = original_part.center

        new_bbox = cv2.boundingRect(main_contour)

        return BodyPart(
            name=original_part.name,
            bbox=new_bbox,
            contour=main_contour,
            center=(cx, cy),
            area=area,
            mask=updated_mask,
            keypoints=original_part.keypoints,
            overlap_padding=max(original_part.overlap_padding, overlap_padding)
        )
    
    
    
    def _define_clean_regions(self) -> Dict:
        """Define non-overlapping region functions with dynamic sizing."""
        kp = self.keypoints_pixel
        
        # Calculate dynamic scale based on shoulder width or height
        scale_ref = 40.0 # Default
        if 'left_shoulder' in kp and 'right_shoulder' in kp:
            # Width between shoulders
            w = np.linalg.norm(np.array(kp['left_shoulder']) - np.array(kp['right_shoulder']))
            scale_ref = w * 0.45 # Arms are roughly 45% of shoulder width
        elif self.height > 0:
            scale_ref = self.height * 0.08 # Fallback to % of height

        # Ensure minimum thickness
        limb_thickness = max(25, int(scale_ref))
        
        # Pass thickness to functions
        return {
            'head': lambda alpha, assigned: self._region_head(alpha, assigned),
            # Note: We calculate arms BEFORE body in the priority list, but here we define the logic
            'left_upper_arm': lambda alpha, assigned: self._region_limb(alpha, assigned, 'left', 'upper_arm', limb_thickness),
            'right_upper_arm': lambda alpha, assigned: self._region_limb(alpha, assigned, 'right', 'upper_arm', limb_thickness),
            'left_lower_arm': lambda alpha, assigned: self._region_limb(alpha, assigned, 'left', 'lower_arm', int(limb_thickness * 0.8)),
            'right_lower_arm': lambda alpha, assigned: self._region_limb(alpha, assigned, 'right', 'lower_arm', int(limb_thickness * 0.8)),
            'left_upper_leg': lambda alpha, assigned: self._region_limb(alpha, assigned, 'left', 'upper_leg', int(limb_thickness * 1.5)),
            'right_upper_leg': lambda alpha, assigned: self._region_limb(alpha, assigned, 'right', 'upper_leg', int(limb_thickness * 1.5)),
            'left_lower_leg': lambda alpha, assigned: self._region_limb(alpha, assigned, 'left', 'lower_leg', limb_thickness),
            'right_lower_leg': lambda alpha, assigned: self._region_limb(alpha, assigned, 'right', 'lower_leg', limb_thickness),
            'body': lambda alpha, assigned: self._region_body(alpha, assigned),
        }

    def _region_head(self, alpha: np.ndarray, assigned: np.ndarray) -> Optional[np.ndarray]:
        """Smart Head Region: Uses face landmarks + expansion."""
        kp = self.keypoints_pixel
        mask = np.zeros_like(alpha)
        
        # 1. Collect all face points
        face_points = []
        for name, pt in kp.items():
            if name in ['nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear', 'mouth_left', 'mouth_right']:
                face_points.append(pt)
        
        if not face_points:
            return None
            
        # 2. Create a convex hull around face points
        hull = cv2.convexHull(np.array(face_points))
        cv2.fillConvexPoly(mask, hull, 255)
        
        # 3. Expand outwards to capture hair/helmet (Dilate)
        # Calculate expansion based on face size
        if len(face_points) >= 2:
            x, y, w, h = cv2.boundingRect(np.array(face_points))
            dilation = max(20, int(w * 0.6)) 
        else:
            dilation = 30
            
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilation, dilation))
        mask = cv2.dilate(mask, kernel, iterations=1)
        
        # 4. Limit bottom by shoulders (to not eat the chest)
        if 'left_shoulder' in kp and 'right_shoulder' in kp:
            neck_y = min(kp['left_shoulder'][1], kp['right_shoulder'][1])
            # Allow a little dip for the chin (neck_y + 10)
            mask[neck_y + 10:, :] = 0 
            
        # 5. Intersect with alpha
        mask = cv2.bitwise_and(mask, alpha)
        
        # 6. Only take unassigned pixels (Head has highest priority usually, so this is fine)
        mask[assigned > 0] = 0
        
        return mask

    def _region_limb(self, alpha: np.ndarray, assigned: np.ndarray, side: str, part: str, thickness: int) -> Optional[np.ndarray]:
        """Generic limb segmenter with dynamic thickness."""
        kp = self.keypoints_pixel
        
        # Map part names to keypoint names
        map_start = {
            'upper_arm': f'{side}_shoulder', 'lower_arm': f'{side}_elbow',
            'upper_leg': f'{side}_hip',      'lower_leg': f'{side}_knee'
        }
        map_end = {
            'upper_arm': f'{side}_elbow',    'lower_arm': f'{side}_wrist',
            'upper_leg': f'{side}_knee',     'lower_leg': f'{side}_ankle'
        }
        
        start_key = map_start.get(part)
        end_key = map_end.get(part)
        
        if start_key not in kp or end_key not in kp:
            return None
            
        p1 = kp[start_key]
        p2 = kp[end_key]
        
        # Create mask
        mask = np.zeros_like(alpha)
        cv2.line(mask, p1, p2, 255, thickness)
        
        # Add hands/feet extensions
        if part == 'lower_arm':
            # Extend past wrist for hand
            vec = np.array(p2) - np.array(p1)
            length = np.linalg.norm(vec)
            if length > 0:
                extension = (vec / length) * (length * 0.4) # Add 40% length for hand
                p3 = (int(p2[0] + extension[0]), int(p2[1] + extension[1]))
                cv2.line(mask, p2, p3, 255, int(thickness * 1.2)) # Hand is wider
                
        elif part == 'lower_leg':
            # Extend for foot (check for foot index)
            foot_key = f'{side}_foot_index'
            if foot_key in kp:
                cv2.line(mask, p2, kp[foot_key], 255, thickness)
        
        # Intersect with alpha
        mask = cv2.bitwise_and(mask, alpha)
        
        # CRITICAL CHANGE: Soft Layering
        # If this limb (e.g. arm) is trying to claim pixels already assigned to something lower priority (like body),
        # we ALLOW it. We only block if assigned > current_priority.
        # But since we iterate in order, we just check if it's 0 for now.
        # For a truly robust fix, we rely on the `PART_PRIORITY` order defined in __init__.
        # Ensure your __init__ has high priority for arms!
        
        mask[assigned > 0] = 0
        
        return mask

    def _region_body(self, alpha: np.ndarray, assigned: np.ndarray) -> Optional[np.ndarray]:
        """Body region: The scavenger."""
        # The body simply takes whatever is left in the center area
        mask = alpha.copy()
        
        # Remove already assigned limbs/head
        mask[assigned > 0] = 0
        
        # Clean up noise - Remove small disconnected blobs
        # (e.g. stray pixels far from the center)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
            
        # Keep only the largest contour (the torso)
        largest = max(contours, key=cv2.contourArea)
        clean_mask = np.zeros_like(mask)
        cv2.drawContours(clean_mask, [largest], -1, 255, -1)
        
        return clean_mask
    
    def _region_upper_arm(self, alpha: np.ndarray, assigned: np.ndarray, side: str) -> Optional[np.ndarray]:
        """Upper arm region: shoulder to elbow."""
        kp = self.keypoints_pixel
        
        shoulder_key = f'{side}_shoulder'
        elbow_key = f'{side}_elbow'
        
        if shoulder_key not in kp or elbow_key not in kp:
            return None
        
        # Build upper arm path (shoulder to elbow)
        path_points = [kp[shoulder_key], kp[elbow_key]]
        
        # Create thick line along upper arm path
        mask = np.zeros_like(alpha)
        thickness = 40  # Upper arm thickness
        
        cv2.line(mask, path_points[0], path_points[1], 255, thickness)
        
        # Intersect with alpha
        mask = cv2.bitwise_and(mask, alpha)
        
        # Exclude assigned pixels
        mask[assigned > 0] = 0
        
        # Clean up with morphology
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        return mask
    
    def _region_lower_arm(self, alpha: np.ndarray, assigned: np.ndarray, side: str) -> Optional[np.ndarray]:
        """Lower arm region: elbow to wrist/hand."""
        kp = self.keypoints_pixel
        
        elbow_key = f'{side}_elbow'
        wrist_key = f'{side}_wrist'
        
        if elbow_key not in kp:
            return None
        
        # Build lower arm path
        path_points = [kp[elbow_key]]
        
        if wrist_key in kp:
            path_points.append(kp[wrist_key])
            # Add hand keypoints
            for finger in [f'{side}_pinky', f'{side}_index', f'{side}_thumb']:
                if finger in kp:
                    path_points.append(kp[finger])
        
        if len(path_points) < 2:
            return None
        
        # Create thick line along lower arm path
        mask = np.zeros_like(alpha)
        thickness = 35  # Lower arm thickness (slightly thinner)
        
        for i in range(len(path_points) - 1):
            cv2.line(mask, path_points[i], path_points[i+1], 255, thickness)
        
        # Intersect with alpha
        mask = cv2.bitwise_and(mask, alpha)
        
        # Exclude assigned pixels
        mask[assigned > 0] = 0
        
        # Clean up with morphology
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        return mask
    
    def _region_upper_leg(self, alpha: np.ndarray, assigned: np.ndarray, side: str) -> Optional[np.ndarray]:
        """Upper leg region: hip to knee."""
        kp = self.keypoints_pixel
        
        hip_key = f'{side}_hip'
        knee_key = f'{side}_knee'
        
        if hip_key not in kp or knee_key not in kp:
            return None
        
        # Build upper leg path (hip to knee)
        path_points = [kp[hip_key], kp[knee_key]]
        
        # Create thick line along upper leg path
        mask = np.zeros_like(alpha)
        thickness = 45  # Upper leg thickness
        
        cv2.line(mask, path_points[0], path_points[1], 255, thickness)
        
        # Intersect with alpha
        mask = cv2.bitwise_and(mask, alpha)
        
        # Exclude assigned pixels
        mask[assigned > 0] = 0
        
        # Clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        return mask
    
    def _region_lower_leg(self, alpha: np.ndarray, assigned: np.ndarray, side: str) -> Optional[np.ndarray]:
        """Lower leg region: knee to ankle/foot."""
        kp = self.keypoints_pixel
        
        knee_key = f'{side}_knee'
        ankle_key = f'{side}_ankle'
        
        if knee_key not in kp:
            return None
        
        # Build lower leg path
        path_points = [kp[knee_key]]
        
        if ankle_key in kp:
            path_points.append(kp[ankle_key])
            # Add foot keypoints
            for foot_part in [f'{side}_heel', f'{side}_foot_index']:
                if foot_part in kp:
                    path_points.append(kp[foot_part])
        
        if len(path_points) < 2:
            return None
        
        # Create thick line along lower leg path
        mask = np.zeros_like(alpha)
        thickness = 40  # Lower leg thickness (slightly thinner)
        
        for i in range(len(path_points) - 1):
            cv2.line(mask, path_points[i], path_points[i+1], 255, thickness)
        
        # Intersect with alpha
        mask = cv2.bitwise_and(mask, alpha)
        
        # Exclude assigned pixels
        mask[assigned > 0] = 0
        
        # Clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        return mask
    
    def _extract_keypoints(self, landmarks) -> Dict[str, Tuple[int, int]]:
        """Extract keypoints as pixel coordinates with lower visibility threshold."""
        keypoints = {}
        for idx, landmark in enumerate(landmarks.landmark):
            # Lower visibility threshold for non-human characters (0.2 instead of 0.3)
            if landmark.visibility < 0.2:
                continue
            x = int(landmark.x * self.width)
            y = int(landmark.y * self.height)
            x = max(0, min(x, self.width - 1))
            y = max(0, min(y, self.height - 1))
            name = self.LANDMARK_NAMES.get(idx, f'landmark_{idx}')
            keypoints[name] = (x, y)
        return keypoints
    
    def _enhance_contrast(self, image: np.ndarray) -> np.ndarray:
        """Enhance contrast to help MediaPipe detect edges better."""
        # Convert to LAB color space for better contrast enhancement
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        
        # Merge and convert back
        enhanced = cv2.merge([l, a, b])
        return cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB)
    
    def _enhance_edges(self, image: np.ndarray) -> np.ndarray:
        """Enhance edges to make body structure more visible."""
        # Convert to grayscale for edge detection
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Apply edge enhancement
        edges = cv2.Canny(gray, 50, 150)
        edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
        
        # Blend edges back into image
        edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
        enhanced = cv2.addWeighted(image, 0.7, edges_colored, 0.3, 0)
        
        return enhanced
    
    def _normalize_image(self, image: np.ndarray) -> np.ndarray:
        """Normalize image brightness and contrast."""
        # Normalize to 0-255 range
        normalized = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
        return normalized.astype(np.uint8)
    
    def _get_alpha_mask(self) -> np.ndarray:
        """Get alpha channel."""
        if self.image.shape[2] == 4:
            return self.image[:, :, 3]
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
        return mask
    
    def _create_part_from_mask(self, name: str, mask: np.ndarray) -> Optional[BodyPart]:
        """Create BodyPart from clean mask."""
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        
        main_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(main_contour)
        
        if area < 100:
            return None
        
        x, y, w, h = cv2.boundingRect(main_contour)
        
        M = cv2.moments(main_contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        else:
            cx, cy = x + w // 2, y + h // 2
        
        return BodyPart(
            name=name,
            bbox=(x, y, w, h),
            contour=main_contour,
            center=(cx, cy),
            area=area,
            mask=mask
        )
    
    def _fallback_contour_detection(self) -> Dict[str, BodyPart]:
        """Fallback: improved detection with arm detection."""
        print("Using fallback detection...")
        alpha = self._get_alpha_mask()
        coords = cv2.findNonZero(alpha)
        if coords is None:
            return {}
        
        x, y, w, h = cv2.boundingRect(coords)
        parts = {}
        
        # Step 1: Extract head (top region)
        head = self._extract_clean_region('head', alpha, x, y, w, h, 0, 0.20)
        if head:
            parts['head'] = head
        
        # Step 2: Extract body region (middle, but we'll refine it)
        body_y_start = int(y + h * 0.20)
        body_y_end = int(y + h * 0.55)
        body_h = body_y_end - body_y_start
        
        if body_h > 0:
            # Get body region mask
            body_mask = np.zeros_like(alpha)
            body_mask[body_y_start:body_y_end, x:x+w] = alpha[body_y_start:body_y_end, x:x+w]
            
            # Find center of character (for splitting arms)
            center_x = x + w // 2
            
            # Split body region into left and right halves for arms
            left_half = body_mask.copy()
            left_half[:, center_x:] = 0
            right_half = body_mask.copy()
            right_half[:, :center_x] = 0
            
            # Extract left and right arms (split into upper/lower if possible)
            # For fallback, we'll create combined arms but try to split them
            left_arm = self._create_part_from_mask('left_upper_arm', left_half)
            right_arm = self._create_part_from_mask('right_upper_arm', right_half)
            
            # Refine body to exclude arms (center portion only)
            body_center_mask = np.zeros_like(alpha)
            center_width = int(w * 0.4)  # 40% of width in center
            center_x_start = x + int(w * 0.3)
            center_x_end = center_x_start + center_width
            body_center_mask[body_y_start:body_y_end, center_x_start:center_x_end] = \
                body_mask[body_y_start:body_y_end, center_x_start:center_x_end]
            
            body = self._create_part_from_mask('body', body_center_mask)
            
            if left_arm and left_arm.area > 500:
                parts['left_upper_arm'] = left_arm
                # Try to create lower arm from same region (fallback doesn't have elbow detection)
                parts['left_lower_arm'] = left_arm  # Will be same as upper in fallback
            if right_arm and right_arm.area > 500:
                parts['right_upper_arm'] = right_arm
                parts['right_lower_arm'] = right_arm  # Will be same as upper in fallback
            if body:
                parts['body'] = body
        
        # Step 3: Extract legs (bottom region, split left/right)
        leg_y_start = int(y + h * 0.55)
        leg_y_end = int(y + h)
        leg_h = leg_y_end - leg_y_start
        
        if leg_h > 0:
            # Get leg region mask
            leg_mask = np.zeros_like(alpha)
            leg_mask[leg_y_start:leg_y_end, x:x+w] = alpha[leg_y_start:leg_y_end, x:x+w]
            
            # Split legs horizontally
            center_x = x + w // 2
            left_leg_mask = leg_mask.copy()
            left_leg_mask[:, center_x:] = 0
            right_leg_mask = leg_mask.copy()
            right_leg_mask[:, :center_x] = 0
            
            left_leg = self._create_part_from_mask('left_upper_leg', left_leg_mask)
            right_leg = self._create_part_from_mask('right_upper_leg', right_leg_mask)
            
            if left_leg:
                parts['left_upper_leg'] = left_leg
                parts['left_lower_leg'] = left_leg  # Fallback: same as upper
            if right_leg:
                parts['right_upper_leg'] = right_leg
                parts['right_lower_leg'] = right_leg  # Fallback: same as upper
        
        return parts
    
    def _extract_clean_region(self, name: str, alpha: np.ndarray, 
                             base_x: int, base_y: int, base_w: int, base_h: int,
                             y_start: float, y_end: float) -> Optional[BodyPart]:
        """Extract region with no overlap."""
        region_y = int(base_y + base_h * y_start)
        region_h = int(base_h * (y_end - y_start))
        
        mask = np.zeros_like(alpha)
        mask[region_y:region_y+region_h, base_x:base_x+base_w] = \
            alpha[region_y:region_y+region_h, base_x:base_x+base_w]
        
        return self._create_part_from_mask(name, mask)
    
    def visualize_detections(self, parts: Dict[str, BodyPart]) -> np.ndarray:
        """Visualize with pose skeleton."""
        vis = self.image.copy()
        
        if self.pose_landmarks:
            if vis.shape[2] == 4:
                vis_bgr = vis[:, :, :3].copy()
            else:
                vis_bgr = vis.copy()
            
            self.mp_drawing.draw_landmarks(
                vis_bgr, self.pose_landmarks, self.mp_pose.POSE_CONNECTIONS,
                self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),
                self.mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)
            )
            
            if vis.shape[2] == 4:
                vis[:, :, :3] = vis_bgr
            else:
                vis = vis_bgr
        
        colors = {
            'head': (0, 255, 255), 'body': (255, 0, 0),
            'left_upper_arm': (0, 255, 0), 'right_upper_arm': (0, 255, 0),
            'left_lower_arm': (0, 200, 0), 'right_lower_arm': (0, 200, 0),
            'left_upper_leg': (255, 0, 255), 'right_upper_leg': (255, 0, 255),
            'left_lower_leg': (200, 0, 200), 'right_lower_leg': (200, 0, 200)
        }
        
        for name, part in parts.items():
            color = colors.get(name, (128, 128, 128))
            if vis.shape[2] == 4:
                # Create a copy of BGR channels for drawing
                vis_bgr = vis[:, :, :3].copy()
                cv2.drawContours(vis_bgr, [part.contour], -1, color, 2)
                # Draw text on BGR copy
                x, y, _, _ = part.bbox
                label_pos = (x, max(y - 5, 15))
                cv2.putText(vis_bgr, name, label_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                # Assign back to original
                vis[:, :, :3] = vis_bgr
            else:
                cv2.drawContours(vis, [part.contour], -1, color, 2)
                x, y, _, _ = part.bbox
                label_pos = (x, max(y - 5, 15))
                cv2.putText(vis, name, label_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return vis
    
    def __del__(self):
        if hasattr(self, 'pose'):
            self.pose.close()


def main():
    """Standalone entry point for part detection."""
    import argparse
    import os
    import json
    from pathlib import Path
    
    parser = argparse.ArgumentParser(
        description='Detect and segment body parts from a character image',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage - detect parts and save visualization
  python part_detector.py input.png -o output/
  
  # Save individual part images and JSON metadata
  python part_detector.py input.png -o output/ --save-parts --save-json
  
  # Skip visualization, only save parts
  python part_detector.py input.png -o output/ --save-parts --no-visualization
        """
    )
    
    parser.add_argument(
        'image_path',
        type=str,
        help='Path to input image file'
    )
    
    parser.add_argument(
        '-o', '--output-dir',
        type=str,
        default=None,
        help='Output directory for results (default: same directory as input image)'
    )
    
    parser.add_argument(
        '--save-parts',
        action='store_true',
        help='Save individual body part images'
    )
    
    parser.add_argument(
        '--no-visualization',
        action='store_true',
        help='Skip saving detection visualization (visualization is saved by default)'
    )
    
    parser.add_argument(
        '--save-json',
        action='store_true',
        help='Save detection results as JSON metadata'
    )
    
    args = parser.parse_args()
    
    # Validate input file
    image_path = Path(args.image_path)
    if not image_path.exists():
        print(f"ERROR: Image file not found: {image_path}")
        return 1
    
    # Determine output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = image_path.parent / f"{image_path.stem}_parts"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    # Load image
    print(f"Loading image: {image_path}")
    image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    
    if image is None:
        print(f"ERROR: Could not load image from {image_path}")
        return 1
    
    print(f"Image loaded: {image.shape}")
    
    # Ensure image has alpha channel (BGRA)
    if image.shape[2] == 3:
        print("WARNING: Image has no alpha channel, adding opaque alpha...")
        alpha = np.ones((image.shape[0], image.shape[1]), dtype=np.uint8) * 255
        image = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
        image[:, :, 3] = alpha
    
    # Detect parts
    print("\nDetecting body parts...")
    detector = BodyPartDetector(image)
    parts = detector.detect_all_parts()
    
    if not parts:
        print("ERROR: No body parts detected!")
        return 1
    
    print(f"Detected {len(parts)} parts: {', '.join(parts.keys())}")
    
    # Save visualization (default: always save unless --no-visualization is set)
    if not args.no_visualization:
        vis = detector.visualize_detections(parts)
        vis_path = output_dir / "detection_visualization.png"
        cv2.imwrite(str(vis_path), vis)
        print(f"Saved visualization: {vis_path}")
    
    # Save individual parts
    if args.save_parts:
        print("\nSaving individual parts...")
        for part_name, part in parts.items():
            # Extract part image
            x, y, w, h = part.bbox
            pad = 5
            x = max(0, x - pad)
            y = max(0, y - pad)
            w = min(w + 2 * pad, image.shape[1] - x)
            h = min(h + 2 * pad, image.shape[0] - y)
            
            # Create part image with transparency
            part_image = np.zeros((h, w, 4), dtype=np.uint8)
            region = image[y:y+h, x:x+w].copy()
            mask_region = part.mask[y:y+h, x:x+w]
            
            part_image[:, :, :3] = region[:, :, :3]
            part_image[:, :, 3] = np.where(mask_region > 0, region[:, :, 3], 0)
            
            # Save part
            part_path = output_dir / f"{part_name}.png"
            cv2.imwrite(str(part_path), part_image)
            print(f"  {part_name}: {part_path}")
    
    # Save JSON metadata
    if args.save_json:
        metadata = {
            "image_path": str(image_path),
            "image_size": {
                "width": int(image.shape[1]),
                "height": int(image.shape[0])
            },
            "parts": []
        }
        
        for part_name, part in parts.items():
            part_info = {
                "name": part_name,
                "bbox": {
                    "x": int(part.bbox[0]),
                    "y": int(part.bbox[1]),
                    "w": int(part.bbox[2]),
                    "h": int(part.bbox[3])
                },
                "center": {
                    "x": int(part.center[0]),
                    "y": int(part.center[1])
                },
                "area": float(part.area)
            }
            metadata["parts"].append(part_info)
        
        json_path = output_dir / "detection_metadata.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        print(f"Saved metadata: {json_path}")
    
    print(f"\nDetection complete! Results saved to: {output_dir}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())