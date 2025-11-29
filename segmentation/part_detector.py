# # """
# # Body Part Detector Module
# # Detects and identifies different body parts from character images.
# # """

# # import cv2
# # import numpy as np
# # from typing import Dict, List, Tuple, Optional
# # from dataclasses import dataclass


# # @dataclass
# # class BodyPart:
# #     """Represents a detected body part."""
# #     name: str
# #     bbox: Tuple[int, int, int, int]  # x, y, w, h
# #     contour: np.ndarray
# #     center: Tuple[int, int]
# #     area: float
# #     mask: np.ndarray


# # class BodyPartDetector:
# #     """
# #     Detects and identifies body parts from character images.
# #     Uses multiple strategies: contour analysis, region division, and spatial relationships.
# #     """
    
# #     def __init__(self, image: np.ndarray):
# #         """
# #         Initialize detector with an image.
        
# #         Args:
# #             image: Input image (BGRA format with transparency)
# #         """
# #         self.image = image
# #         self.height, self.width = image.shape[:2]
# #         self.parts: List[BodyPart] = []
        
# #         # Body proportions (typical humanoid/animal ratios)
# #         self.PROPORTIONS = {
# #             'head_ratio': 0.20,      # Head is ~20% of total height
# #             'torso_ratio': 0.35,     # Torso is ~35% of total height
# #             'legs_ratio': 0.45,      # Legs are ~45% of total height
# #             'arm_width_ratio': 0.15, # Arms are ~15% of body width each
# #         }
    
# #     def detect_all_parts(self) -> Dict[str, BodyPart]:
# #         """
# #         Main detection method - detects all body parts.
        
# #         Returns:
# #             Dictionary of {part_name: BodyPart}
# #         """
# #         # Step 1: Get character bounding box
# #         char_bbox = self._get_character_bbox()
# #         if char_bbox is None:
# #             return {}
        
# #         # Step 2: Detect contours
# #         contours = self._find_contours()
        
# #         # Step 3: Apply multiple detection strategies
# #         detected_parts = {}
        
# #         # Strategy 1: Region-based detection (for simple characters)
# #         region_parts = self._detect_by_regions(char_bbox)
# #         detected_parts.update(region_parts)
        
# #         # Strategy 2: Contour-based detection (for complex characters)
# #         if len(contours) > 1:
# #             contour_parts = self._detect_by_contours(contours, char_bbox)
# #             # Merge with region-based (contour-based overrides if better)
# #             for name, part in contour_parts.items():
# #                 if name not in detected_parts or part.area > detected_parts[name].area * 0.5:
# #                     detected_parts[name] = part
        
# #         # Strategy 3: Spatial relationship refinement
# #         detected_parts = self._refine_by_spatial_relationships(detected_parts, char_bbox)
        
# #         return detected_parts
    
# #     def _get_character_bbox(self) -> Optional[Tuple[int, int, int, int]]:
# #         """Get the bounding box of the entire character."""
# #         if self.image.shape[2] == 4:  # Has alpha channel
# #             alpha = self.image[:, :, 3]
# #         else:
# #             gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
# #             _, alpha = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
        
# #         # Find non-zero pixels
# #         coords = cv2.findNonZero(alpha)
# #         if coords is None:
# #             return None
        
# #         x, y, w, h = cv2.boundingRect(coords)
# #         return (x, y, w, h)
    
# #     def _find_contours(self) -> List[np.ndarray]:
# #         """Find all contours in the image."""
# #         if self.image.shape[2] == 4:
# #             alpha = self.image[:, :, 3]
# #         else:
# #             gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
# #             _, alpha = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
        
# #         # Clean up with morphological operations
# #         kernel = np.ones((3, 3), np.uint8)
# #         alpha = cv2.morphologyEx(alpha, cv2.MORPH_CLOSE, kernel)
# #         alpha = cv2.morphologyEx(alpha, cv2.MORPH_OPEN, kernel)
        
# #         # Find contours
# #         contours, _ = cv2.findContours(alpha, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
# #         # Filter small contours (noise)
# #         min_area = (self.width * self.height) * 0.001  # 0.1% of image
# #         contours = [c for c in contours if cv2.contourArea(c) > min_area]
        
# #         # Sort by area (largest first)
# #         contours = sorted(contours, key=cv2.contourArea, reverse=True)
        
# #         return contours
    
# #     def _detect_by_regions(self, char_bbox: Tuple[int, int, int, int]) -> Dict[str, BodyPart]:
# #         """
# #         Detect body parts by dividing the character into regions.
# #         Works well for T-pose or neutral stance characters.
# #         """
# #         x, y, w, h = char_bbox
# #         parts = {}
        
# #         # Calculate region boundaries based on proportions
# #         head_h = int(h * self.PROPORTIONS['head_ratio'])
# #         torso_h = int(h * self.PROPORTIONS['torso_ratio'])
# #         legs_h = h - head_h - torso_h
        
# #         # Define regions
# #         regions = {
# #             'head': (x, y, w, head_h),
# #             'body': (x + int(w * 0.2), y + head_h, int(w * 0.6), torso_h),
# #             'left_arm': (x, y + head_h, int(w * 0.3), torso_h),
# #             'right_arm': (x + int(w * 0.7), y + head_h, int(w * 0.3), torso_h),
# #             'legs_combined': (x, y + head_h + torso_h, w, legs_h),
# #         }
        
# #         # Extract each region
# #         for part_name, (rx, ry, rw, rh) in regions.items():
# #             # Ensure bounds are valid
# #             rx = max(0, min(rx, self.width - 1))
# #             ry = max(0, min(ry, self.height - 1))
# #             rw = min(rw, self.width - rx)
# #             rh = min(rh, self.height - ry)
            
# #             if rw <= 0 or rh <= 0:
# #                 continue
            
# #             # Extract region mask
# #             if self.image.shape[2] == 4:
# #                 region_alpha = self.image[ry:ry+rh, rx:rx+rw, 3]
# #             else:
# #                 region_gray = cv2.cvtColor(self.image[ry:ry+rh, rx:rx+rw], cv2.COLOR_BGR2GRAY)
# #                 _, region_alpha = cv2.threshold(region_gray, 10, 255, cv2.THRESH_BINARY)
            
# #             # Check if region has significant content
# #             if np.sum(region_alpha > 0) < (rw * rh * 0.05):  # At least 5% filled
# #                 continue
            
# #             # Create full-size mask
# #             full_mask = np.zeros((self.height, self.width), dtype=np.uint8)
# #             full_mask[ry:ry+rh, rx:rx+rw] = region_alpha
            
# #             # Find contour for this region
# #             contours, _ = cv2.findContours(region_alpha, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# #             if not contours:
# #                 continue
            
# #             main_contour = max(contours, key=cv2.contourArea)
# #             # Adjust contour coordinates to full image
# #             main_contour = main_contour + np.array([rx, ry])
            
# #             # Calculate center
# #             M = cv2.moments(main_contour)
# #             if M["m00"] != 0:
# #                 cx = int(M["m10"] / M["m00"])
# #                 cy = int(M["m01"] / M["m00"])
# #             else:
# #                 cx, cy = rx + rw // 2, ry + rh // 2
            
# #             part = BodyPart(
# #                 name=part_name,
# #                 bbox=(rx, ry, rw, rh),
# #                 contour=main_contour,
# #                 center=(cx, cy),
# #                 area=cv2.contourArea(main_contour),
# #                 mask=full_mask
# #             )
            
# #             parts[part_name] = part
        
# #         # Try to split legs if combined
# #         if 'legs_combined' in parts:
# #             left_leg, right_leg = self._split_legs(parts['legs_combined'])
# #             if left_leg and right_leg:
# #                 parts['left_leg'] = left_leg
# #                 parts['right_leg'] = right_leg
# #                 del parts['legs_combined']
        
# #         return parts
    
# #     def _detect_by_contours(self, contours: List[np.ndarray], 
# #                            char_bbox: Tuple[int, int, int, int]) -> Dict[str, BodyPart]:
# #         """
# #         Detect body parts by analyzing individual contours.
# #         Works well for characters with clear separation between parts.
# #         """
# #         x, y, w, h = char_bbox
# #         parts = {}
        
# #         # Analyze each contour
# #         for contour in contours[:10]:  # Limit to top 10 largest
# #             # Get contour properties
# #             area = cv2.contourArea(contour)
# #             bbox = cv2.boundingRect(contour)
# #             cx_local, cy_local, cw, ch = bbox
            
# #             # Calculate center
# #             M = cv2.moments(contour)
# #             if M["m00"] != 0:
# #                 cx = int(M["m10"] / M["m00"])
# #                 cy = int(M["m01"] / M["m00"])
# #             else:
# #                 cx, cy = cx_local + cw // 2, cy_local + ch // 2
            
# #             # Classify based on position and size
# #             part_name = self._classify_contour(bbox, area, char_bbox)
            
# #             if part_name:
# #                 # Create mask for this contour
# #                 mask = np.zeros((self.height, self.width), dtype=np.uint8)
# #                 cv2.drawContours(mask, [contour], -1, 255, -1)
                
# #                 part = BodyPart(
# #                     name=part_name,
# #                     bbox=bbox,
# #                     contour=contour,
# #                     center=(cx, cy),
# #                     area=area,
# #                     mask=mask
# #                 )
                
# #                 # Keep largest if duplicate names
# #                 if part_name not in parts or area > parts[part_name].area:
# #                     parts[part_name] = part
        
# #         return parts
    
# #     def _classify_contour(self, bbox: Tuple[int, int, int, int], area: float,
# #                          char_bbox: Tuple[int, int, int, int]) -> Optional[str]:
# #         """
# #         Classify a contour as a specific body part based on its properties.
# #         """
# #         cx, cy, cw, ch = bbox
# #         char_x, char_y, char_w, char_h = char_bbox
        
# #         # Calculate relative position (0-1 range)
# #         rel_x = (cx - char_x) / char_w if char_w > 0 else 0
# #         rel_y = (cy - char_y) / char_h if char_h > 0 else 0
        
# #         # Calculate aspect ratio
# #         aspect = ch / cw if cw > 0 else 1
        
# #         # Calculate relative size
# #         rel_area = area / (char_w * char_h) if (char_w * char_h) > 0 else 0
        
# #         # Classification rules
        
# #         # Head: top region, roughly square, medium size
# #         if rel_y < 0.25 and 0.8 < aspect < 1.5 and 0.05 < rel_area < 0.3:
# #             return 'head'
        
# #         # Body: center region, larger area
# #         if 0.2 < rel_y < 0.6 and 0.3 < rel_x < 0.7 and rel_area > 0.15:
# #             return 'body'
        
# #         # Left arm: left side, upper-middle region
# #         if rel_x < 0.35 and 0.25 < rel_y < 0.65 and aspect > 1.2:
# #             return 'left_arm'
        
# #         # Right arm: right side, upper-middle region
# #         if rel_x > 0.65 and 0.25 < rel_y < 0.65 and aspect > 1.2:
# #             return 'right_arm'
        
# #         # Left leg: left side, lower region
# #         if rel_x < 0.5 and rel_y > 0.55 and aspect > 1.5:
# #             return 'left_leg'
        
# #         # Right leg: right side, lower region
# #         if rel_x >= 0.5 and rel_y > 0.55 and aspect > 1.5:
# #             return 'right_leg'
        
# #         return None
    
# #     def _split_legs(self, legs_part: BodyPart) -> Tuple[Optional[BodyPart], Optional[BodyPart]]:
# #         """Try to split combined legs into left and right."""
# #         x, y, w, h = legs_part.bbox
        
# #         # Split vertically down the middle
# #         mid_x = x + w // 2
        
# #         # Create masks for left and right
# #         left_mask = legs_part.mask.copy()
# #         left_mask[:, mid_x:] = 0
        
# #         right_mask = legs_part.mask.copy()
# #         right_mask[:, :mid_x] = 0
        
# #         # Find contours in each half
# #         left_contours, _ = cv2.findContours(left_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# #         right_contours, _ = cv2.findContours(right_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
# #         left_leg = None
# #         right_leg = None
        
# #         if left_contours:
# #             contour = max(left_contours, key=cv2.contourArea)
# #             bbox = cv2.boundingRect(contour)
# #             M = cv2.moments(contour)
# #             cx = int(M["m10"] / M["m00"]) if M["m00"] != 0 else bbox[0] + bbox[2] // 2
# #             cy = int(M["m01"] / M["m00"]) if M["m00"] != 0 else bbox[1] + bbox[3] // 2
            
# #             left_leg = BodyPart(
# #                 name='left_leg',
# #                 bbox=bbox,
# #                 contour=contour,
# #                 center=(cx, cy),
# #                 area=cv2.contourArea(contour),
# #                 mask=left_mask
# #             )
        
# #         if right_contours:
# #             contour = max(right_contours, key=cv2.contourArea)
# #             bbox = cv2.boundingRect(contour)
# #             M = cv2.moments(contour)
# #             cx = int(M["m10"] / M["m00"]) if M["m00"] != 0 else bbox[0] + bbox[2] // 2
# #             cy = int(M["m01"] / M["m00"]) if M["m00"] != 0 else bbox[1] + bbox[3] // 2
            
# #             right_leg = BodyPart(
# #                 name='right_leg',
# #                 bbox=bbox,
# #                 contour=contour,
# #                 center=(cx, cy),
# #                 area=cv2.contourArea(contour),
# #                 mask=right_mask
# #             )
        
# #         return left_leg, right_leg
    
# #     def _refine_by_spatial_relationships(self, parts: Dict[str, BodyPart],
# #                                         char_bbox: Tuple[int, int, int, int]) -> Dict[str, BodyPart]:
# #         """
# #         Refine detected parts based on spatial relationships.
# #         Ensures logical hierarchy (e.g., head above body).
# #         """
# #         if 'body' not in parts:
# #             return parts
        
# #         body = parts['body']
# #         body_center_y = body.center[1]
        
# #         # Head should be above body
# #         if 'head' in parts:
# #             head = parts['head']
# #             if head.center[1] > body_center_y:
# #                 # Head is below body - probably wrong, remove it
# #                 print("Warning: Head detected below body, removing head detection")
# #                 del parts['head']
        
# #         # Arms should be at body level (vertically)
# #         for arm_name in ['left_arm', 'right_arm']:
# #             if arm_name in parts:
# #                 arm = parts[arm_name]
# #                 # Arms should be roughly at body height
# #                 if abs(arm.center[1] - body_center_y) > char_bbox[3] * 0.3:
# #                     print(f"Warning: {arm_name} too far from body vertically")
        
# #         # Legs should be below body
# #         for leg_name in ['left_leg', 'right_leg']:
# #             if leg_name in parts:
# #                 leg = parts[leg_name]
# #                 if leg.center[1] < body_center_y:
# #                     print(f"Warning: {leg_name} detected above body")
        
# #         return parts
    
# #     def visualize_detections(self, parts: Dict[str, BodyPart]) -> np.ndarray:
# #         """
# #         Create visualization of detected parts.
# #         Useful for debugging.
# #         """
# #         vis = self.image.copy()
        
# #         # Color map for different parts
# #         colors = {
# #             'head': (0, 255, 255),      # Yellow
# #             'body': (255, 0, 0),        # Blue
# #             'left_arm': (0, 255, 0),    # Green
# #             'right_arm': (0, 255, 0),   # Green
# #             'left_leg': (255, 0, 255),  # Magenta
# #             'right_leg': (255, 0, 255), # Magenta
# #         }
        
# #         for name, part in parts.items():
# #             color = colors.get(name, (128, 128, 128))
            
# #             # Draw bounding box
# #             x, y, w, h = part.bbox
# #             cv2.rectangle(vis, (x, y), (x + w, y + h), color, 2)
            
# #             # Draw center point
# #             cv2.circle(vis, part.center, 5, color, -1)
            
# #             # Draw label
# #             cv2.putText(vis, name, (x, y - 10), 
# #                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
# #         return vis




# """
# Body Part Detector Module with MediaPipe Pose Detection
# Uses pose estimation to accurately detect and segment body parts.
# IMPROVED VERSION with proper part separation
# """

# import cv2
# import numpy as np
# import mediapipe as mp
# from typing import Dict, List, Tuple, Optional
# from dataclasses import dataclass


# @dataclass
# class BodyPart:
#     """Represents a detected body part."""
#     name: str
#     bbox: Tuple[int, int, int, int]  # x, y, w, h
#     contour: np.ndarray
#     center: Tuple[int, int]
#     area: float
#     mask: np.ndarray
#     keypoints: Optional[Dict[str, Tuple[int, int]]] = None


# class BodyPartDetector:
#     """
#     Detects and identifies body parts using MediaPipe Pose estimation.
#     Improved version with better part separation.
#     """
    
#     def __init__(self, image: np.ndarray):
#         self.image = image
#         self.height, self.width = image.shape[:2]
#         self.parts: Dict[str, BodyPart] = {}
#         self.pose_landmarks = None
#         self.keypoints_pixel = {}
        
#         # Initialize MediaPipe Pose
#         self.mp_pose = mp.solutions.pose
#         self.mp_drawing = mp.solutions.drawing_utils
#         self.pose = self.mp_pose.Pose(
#             static_image_mode=True,
#             model_complexity=2,
#             enable_segmentation=True,
#             min_detection_confidence=0.3  # Lower threshold for better detection
#         )
        
#         # MediaPipe landmark indices
#         self.LANDMARK_NAMES = {
#             0: 'nose',
#             1: 'left_eye_inner', 2: 'left_eye', 3: 'left_eye_outer',
#             4: 'right_eye_inner', 5: 'right_eye', 6: 'right_eye_outer',
#             7: 'left_ear', 8: 'right_ear',
#             9: 'mouth_left', 10: 'mouth_right',
#             11: 'left_shoulder', 12: 'right_shoulder',
#             13: 'left_elbow', 14: 'right_elbow',
#             15: 'left_wrist', 16: 'right_wrist',
#             17: 'left_pinky', 18: 'right_pinky',
#             19: 'left_index', 20: 'right_index',
#             21: 'left_thumb', 22: 'right_thumb',
#             23: 'left_hip', 24: 'right_hip',
#             25: 'left_knee', 26: 'right_knee',
#             27: 'left_ankle', 28: 'right_ankle',
#             29: 'left_heel', 30: 'right_heel',
#             31: 'left_foot_index', 32: 'right_foot_index'
#         }
    
#     def detect_all_parts(self) -> Dict[str, BodyPart]:
#         """Main detection method using MediaPipe Pose."""
#         print("Detecting pose landmarks with MediaPipe...")
        
#         # Convert to RGB for MediaPipe
#         if self.image.shape[2] == 4:
#             rgb_image = cv2.cvtColor(self.image[:, :, :3], cv2.COLOR_BGR2RGB)
#         else:
#             rgb_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        
#         # Process with MediaPipe
#         results = self.pose.process(rgb_image)
        
#         if not results.pose_landmarks:
#             print("⚠️ No pose landmarks detected! Falling back to contour-based detection...")
#             return self._fallback_contour_detection()
        
#         self.pose_landmarks = results.pose_landmarks
#         print(f"✓ Detected {len(results.pose_landmarks.landmark)} pose landmarks")
        
#         # Extract keypoints
#         self.keypoints_pixel = self._extract_keypoints(results.pose_landmarks)
#         print(f"✓ Extracted {len(self.keypoints_pixel)} valid keypoints")
        
#         if len(self.keypoints_pixel) < 10:
#             print("⚠️ Too few keypoints detected, using fallback...")
#             return self._fallback_contour_detection()
        
#         # Get alpha mask
#         alpha_mask = self._get_alpha_mask()
        
#         # Detect parts with improved segmentation
#         detected_parts = {}
        
#         # 1. Head (ears, eyes, nose, mouth)
#         head = self._segment_head(alpha_mask)
#         if head:
#             detected_parts['head'] = head
#             print(f"  ✓ Head segmented")
        
#         # 2. Body/Torso (between shoulders and hips)
#         body = self._segment_body(alpha_mask)
#         if body:
#             detected_parts['body'] = body
#             print(f"  ✓ Body segmented")
        
#         # 3. Left Arm (shoulder to hand)
#         left_arm = self._segment_arm(alpha_mask, 'left')
#         if left_arm:
#             detected_parts['left_arm'] = left_arm
#             print(f"  ✓ Left arm segmented")
        
#         # 4. Right Arm (shoulder to hand)
#         right_arm = self._segment_arm(alpha_mask, 'right')
#         if right_arm:
#             detected_parts['right_arm'] = right_arm
#             print(f"  ✓ Right arm segmented")
        
#         # 5. Left Leg (hip to foot)
#         left_leg = self._segment_leg(alpha_mask, 'left')
#         if left_leg:
#             detected_parts['left_leg'] = left_leg
#             print(f"  ✓ Left leg segmented")
        
#         # 6. Right Leg (hip to foot)
#         right_leg = self._segment_leg(alpha_mask, 'right')
#         if right_leg:
#             detected_parts['right_leg'] = right_leg
#             print(f"  ✓ Right leg segmented")
        
#         print(f"✓ Total parts segmented: {len(detected_parts)}")
        
#         return detected_parts
    
#     def _extract_keypoints(self, landmarks) -> Dict[str, Tuple[int, int]]:
#         """Extract keypoints as pixel coordinates."""
#         keypoints = {}
        
#         for idx, landmark in enumerate(landmarks.landmark):
#             # Lower visibility threshold for better detection
#             if landmark.visibility < 0.3:
#                 continue
            
#             x = int(landmark.x * self.width)
#             y = int(landmark.y * self.height)
            
#             x = max(0, min(x, self.width - 1))
#             y = max(0, min(y, self.height - 1))
            
#             name = self.LANDMARK_NAMES.get(idx, f'landmark_{idx}')
#             keypoints[name] = (x, y)
        
#         return keypoints
    
#     def _get_alpha_mask(self) -> np.ndarray:
#         """Get alpha channel as binary mask."""
#         if self.image.shape[2] == 4:
#             return self.image[:, :, 3]
#         else:
#             gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
#             _, mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
#             return mask
    
#     def _create_polygon_mask(self, points: List[Tuple[int, int]], 
#                             alpha_mask: np.ndarray) -> np.ndarray:
#         """Create a polygon mask from points and intersect with alpha."""
#         mask = np.zeros((self.height, self.width), dtype=np.uint8)
        
#         if len(points) < 3:
#             return mask
        
#         # Create polygon
#         points_array = np.array(points, dtype=np.int32)
#         cv2.fillPoly(mask, [points_array], 255)
        
#         # Intersect with alpha mask (only keep visible pixels)
#         mask = cv2.bitwise_and(mask, alpha_mask)
        
#         # Dilate slightly to include edges
#         kernel = np.ones((3, 3), np.uint8)
#         mask = cv2.dilate(mask, kernel, iterations=1)
        
#         return mask
    
#     def _segment_head(self, alpha_mask: np.ndarray) -> Optional[BodyPart]:
#         """Segment head using facial landmarks."""
#         kp = self.keypoints_pixel
        
#         # Required keypoints for head
#         required = ['nose']
#         if not all(k in kp for k in required):
#             return None
        
#         # Build head polygon
#         head_points = []
        
#         # Top of head (estimate above eyes/ears)
#         if 'left_ear' in kp and 'right_ear' in kp:
#             left_ear = kp['left_ear']
#             right_ear = kp['right_ear']
#             mid_x = (left_ear[0] + right_ear[0]) // 2
#             top_y = min(left_ear[1], right_ear[1]) - 50  # Above ears
#             head_points.append((mid_x, top_y))
#             head_points.append((left_ear[0] - 20, left_ear[1]))
#         elif 'left_eye' in kp:
#             eye = kp['left_eye']
#             head_points.append((eye[0], eye[1] - 60))
        
#         # Left side
#         if 'left_ear' in kp:
#             head_points.append(kp['left_ear'])
#         if 'mouth_left' in kp:
#             head_points.append(kp['mouth_left'])
        
#         # Bottom (chin/neck area)
#         if 'left_shoulder' in kp and 'right_shoulder' in kp:
#             ls = kp['left_shoulder']
#             rs = kp['right_shoulder']
#             neck_x = (ls[0] + rs[0]) // 2
#             neck_y = min(ls[1], rs[1]) - 10
#             head_points.append((neck_x - 30, neck_y))
#             head_points.append((neck_x + 30, neck_y))
        
#         # Right side
#         if 'mouth_right' in kp:
#             head_points.append(kp['mouth_right'])
#         if 'right_ear' in kp:
#             head_points.append(kp['right_ear'])
#             head_points.append((kp['right_ear'][0] + 20, kp['right_ear'][1]))
        
#         if len(head_points) < 3:
#             return self._fallback_region_part('head', alpha_mask, 0, 0.25)
        
#         mask = self._create_polygon_mask(head_points, alpha_mask)
#         return self._create_part_from_mask('head', mask, head_points)
    
#     def _segment_body(self, alpha_mask: np.ndarray) -> Optional[BodyPart]:
#         """Segment torso/body between shoulders and hips."""
#         kp = self.keypoints_pixel
        
#         required = ['left_shoulder', 'right_shoulder']
#         if not all(k in kp for k in required):
#             return None
        
#         # Body polygon: shoulders → hips
#         body_points = []
        
#         # Top: shoulders
#         if 'left_shoulder' in kp:
#             ls = kp['left_shoulder']
#             body_points.append((ls[0] - 15, ls[1]))
#             body_points.append(ls)
        
#         if 'right_shoulder' in kp:
#             rs = kp['right_shoulder']
#             body_points.append(rs)
#             body_points.append((rs[0] + 15, rs[1]))
        
#         # Bottom: hips
#         if 'right_hip' in kp:
#             rh = kp['right_hip']
#             body_points.append((rh[0] + 20, rh[1]))
#             body_points.append(rh)
#         elif 'right_shoulder' in kp:
#             rs = kp['right_shoulder']
#             body_points.append((rs[0] + 15, rs[1] + 150))
        
#         if 'left_hip' in kp:
#             lh = kp['left_hip']
#             body_points.append(lh)
#             body_points.append((lh[0] - 20, lh[1]))
#         elif 'left_shoulder' in kp:
#             ls = kp['left_shoulder']
#             body_points.append((ls[0] - 15, ls[1] + 150))
        
#         if len(body_points) < 4:
#             return self._fallback_region_part('body', alpha_mask, 0.25, 0.60)
        
#         mask = self._create_polygon_mask(body_points, alpha_mask)
#         return self._create_part_from_mask('body', mask, body_points)
    
#     def _segment_arm(self, alpha_mask: np.ndarray, side: str) -> Optional[BodyPart]:
#         """Segment arm (shoulder to wrist/hand)."""
#         kp = self.keypoints_pixel
#         prefix = side  # 'left' or 'right'
        
#         shoulder_key = f'{prefix}_shoulder'
#         elbow_key = f'{prefix}_elbow'
#         wrist_key = f'{prefix}_wrist'
        
#         if shoulder_key not in kp:
#             return None
        
#         # Build arm polygon
#         arm_points = []
        
#         # Start at shoulder
#         shoulder = kp[shoulder_key]
#         offset = 20 if side == 'left' else -20
        
#         # Shoulder area (wider)
#         arm_points.append((shoulder[0] + offset, shoulder[1] - 10))
#         arm_points.append((shoulder[0] + offset, shoulder[1] + 10))
        
#         # Along the arm
#         if elbow_key in kp:
#             elbow = kp[elbow_key]
#             arm_points.append((elbow[0] + offset//2, elbow[1] + 10))
            
#             if wrist_key in kp:
#                 wrist = kp[wrist_key]
#                 arm_points.append((wrist[0], wrist[1] + 15))
                
#                 # Hand area
#                 for finger in [f'{prefix}_pinky', f'{prefix}_index', f'{prefix}_thumb']:
#                     if finger in kp:
#                         arm_points.append(kp[finger])
                
#                 # Back up the arm
#                 arm_points.append((wrist[0], wrist[1] - 15))
            
#             arm_points.append((elbow[0] - offset//2, elbow[1] - 10))
        
#         arm_points.append((shoulder[0] - offset, shoulder[1] + 10))
#         arm_points.append((shoulder[0] - offset, shoulder[1] - 10))
        
#         if len(arm_points) < 4:
#             return None
        
#         mask = self._create_polygon_mask(arm_points, alpha_mask)
#         return self._create_part_from_mask(f'{side}_arm', mask, arm_points)
    
#     def _segment_leg(self, alpha_mask: np.ndarray, side: str) -> Optional[BodyPart]:
#         """Segment leg (hip to foot)."""
#         kp = self.keypoints_pixel
#         prefix = side
        
#         hip_key = f'{prefix}_hip'
#         knee_key = f'{prefix}_knee'
#         ankle_key = f'{prefix}_ankle'
        
#         if hip_key not in kp:
#             return None
        
#         leg_points = []
        
#         # Start at hip
#         hip = kp[hip_key]
#         offset = 25
        
#         leg_points.append((hip[0] - offset, hip[1]))
#         leg_points.append((hip[0] + offset, hip[1]))
        
#         # Along the leg
#         if knee_key in kp:
#             knee = kp[knee_key]
#             leg_points.append((knee[0] + offset//2, knee[1]))
            
#             if ankle_key in kp:
#                 ankle = kp[ankle_key]
#                 leg_points.append((ankle[0] + 10, ankle[1]))
                
#                 # Foot area
#                 for foot_part in [f'{prefix}_heel', f'{prefix}_foot_index']:
#                     if foot_part in kp:
#                         leg_points.append(kp[foot_part])
                
#                 leg_points.append((ankle[0] - 10, ankle[1]))
            
#             leg_points.append((knee[0] - offset//2, knee[1]))
        
#         if len(leg_points) < 4:
#             return None
        
#         mask = self._create_polygon_mask(leg_points, alpha_mask)
#         return self._create_part_from_mask(f'{side}_leg', mask, leg_points)
    
#     def _create_part_from_mask(self, name: str, mask: np.ndarray, 
#                                keypoints_list: List[Tuple[int, int]]) -> Optional[BodyPart]:
#         """Create BodyPart from mask."""
#         # Find contours in mask
#         contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
#         if not contours:
#             return None
        
#         # Get largest contour
#         main_contour = max(contours, key=cv2.contourArea)
#         area = cv2.contourArea(main_contour)
        
#         if area < 500:  # Too small
#             return None
        
#         # Bounding box
#         x, y, w, h = cv2.boundingRect(main_contour)
        
#         # Center
#         M = cv2.moments(main_contour)
#         if M["m00"] != 0:
#             cx = int(M["m10"] / M["m00"])
#             cy = int(M["m01"] / M["m00"])
#         else:
#             cx, cy = x + w // 2, y + h // 2
        
#         # Store keypoints
#         kp_dict = {}
#         for i, pt in enumerate(keypoints_list):
#             kp_dict[f'point_{i}'] = pt
        
#         return BodyPart(
#             name=name,
#             bbox=(x, y, w, h),
#             contour=main_contour,
#             center=(cx, cy),
#             area=area,
#             mask=mask,
#             keypoints=kp_dict
#         )
    
#     def _fallback_region_part(self, name: str, alpha_mask: np.ndarray, 
#                              start_ratio: float, end_ratio: float) -> Optional[BodyPart]:
#         """Fallback: extract part by vertical region."""
#         coords = cv2.findNonZero(alpha_mask)
#         if coords is None:
#             return None
        
#         x, y, w, h = cv2.boundingRect(coords)
        
#         region_y = int(y + h * start_ratio)
#         region_h = int(h * (end_ratio - start_ratio))
        
#         region_mask = np.zeros_like(alpha_mask)
#         region_mask[region_y:region_y+region_h, x:x+w] = alpha_mask[region_y:region_y+region_h, x:x+w]
        
#         contours, _ = cv2.findContours(region_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#         if not contours:
#             return None
        
#         main_contour = max(contours, key=cv2.contourArea)
#         area = cv2.contourArea(main_contour)
        
#         if area < 100:
#             return None
        
#         bbox = cv2.boundingRect(main_contour)
#         M = cv2.moments(main_contour)
#         cx = int(M["m10"] / M["m00"]) if M["m00"] != 0 else bbox[0] + bbox[2]//2
#         cy = int(M["m01"] / M["m00"]) if M["m00"] != 0 else bbox[1] + bbox[3]//2
        
#         return BodyPart(
#             name=name,
#             bbox=bbox,
#             contour=main_contour,
#             center=(cx, cy),
#             area=area,
#             mask=region_mask
#         )
    
#     def _fallback_contour_detection(self) -> Dict[str, BodyPart]:
#         """Complete fallback when pose detection fails."""
#         print("Using fallback contour-based detection...")
        
#         alpha_mask = self._get_alpha_mask()
#         coords = cv2.findNonZero(alpha_mask)
        
#         if coords is None:
#             return {}
        
#         x, y, w, h = cv2.boundingRect(coords)
        
#         parts = {}
        
#         # Simple vertical split
#         regions = {
#             'head': (0, 0.25),
#             'body': (0.25, 0.60),
#             'legs_combined': (0.60, 1.0)
#         }
        
#         for part_name, (start, end) in regions.items():
#             part = self._fallback_region_part(part_name, alpha_mask, start, end)
#             if part:
#                 parts[part_name] = part
        
#         # Split legs
#         if 'legs_combined' in parts:
#             mid_x = x + w // 2
#             left_mask = parts['legs_combined'].mask.copy()
#             left_mask[:, mid_x:] = 0
            
#             right_mask = parts['legs_combined'].mask.copy()
#             right_mask[:, :mid_x] = 0
            
#             left_leg = self._create_part_from_mask('left_leg', left_mask, [])
#             right_leg = self._create_part_from_mask('right_leg', right_mask, [])
            
#             if left_leg:
#                 parts['left_leg'] = left_leg
#             if right_leg:
#                 parts['right_leg'] = right_leg
            
#             del parts['legs_combined']
        
#         return parts
    
#     def visualize_detections(self, parts: Dict[str, BodyPart]) -> np.ndarray:
#         """Visualize detected parts with pose skeleton."""
#         vis = self.image.copy()
        
#         # Draw pose landmarks
#         if self.pose_landmarks:
#             if vis.shape[2] == 4:
#                 vis_bgr = vis[:, :, :3].copy()
#             else:
#                 vis_bgr = vis.copy()
            
#             self.mp_drawing.draw_landmarks(
#                 vis_bgr,
#                 self.pose_landmarks,
#                 self.mp_pose.POSE_CONNECTIONS,
#                 self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),
#                 self.mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)
#             )
            
#             if vis.shape[2] == 4:
#                 vis[:, :, :3] = vis_bgr
#             else:
#                 vis = vis_bgr
        
#         # Draw part boundaries
#         colors = {
#             'head': (0, 255, 255),
#             'body': (255, 0, 0),
#             'left_arm': (0, 255, 0),
#             'right_arm': (0, 255, 0),
#             'left_leg': (255, 0, 255),
#             'right_leg': (255, 0, 255),
#         }
        
#         for name, part in parts.items():
#             color = colors.get(name, (128, 128, 128))
            
#             # Draw contour
#             if vis.shape[2] == 4:
#                 vis_bgr = vis[:, :, :3]
#                 cv2.drawContours(vis_bgr, [part.contour], -1, color, 2)
#             else:
#                 cv2.drawContours(vis, [part.contour], -1, color, 2)
            
#             # Draw label
#             x, y, w, h = part.bbox
#             label_y = max(y - 5, 15)
#             cv2.putText(vis if vis.shape[2] == 3 else vis[:, :, :3], name, 
#                        (x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
#         return vis
    
#     def __del__(self):
#         """Cleanup MediaPipe resources."""
#         if hasattr(self, 'pose'):
#             self.pose.close()




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
            'head': 8,
            'left_upper_arm': 7,
            'right_upper_arm': 7,
            'left_lower_arm': 6,
            'right_lower_arm': 6,
            'body': 5,
            'left_upper_leg': 4,
            'right_upper_leg': 4,
            'left_lower_leg': 3,
            'right_lower_leg': 3
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
        """Define non-overlapping region functions."""
        kp = self.keypoints_pixel
        
        return {
            'head': lambda alpha, assigned: self._region_head(alpha, assigned),
            'body': lambda alpha, assigned: self._region_body(alpha, assigned),
            'left_upper_arm': lambda alpha, assigned: self._region_upper_arm(alpha, assigned, 'left'),
            'right_upper_arm': lambda alpha, assigned: self._region_upper_arm(alpha, assigned, 'right'),
            'left_lower_arm': lambda alpha, assigned: self._region_lower_arm(alpha, assigned, 'left'),
            'right_lower_arm': lambda alpha, assigned: self._region_lower_arm(alpha, assigned, 'right'),
            'left_upper_leg': lambda alpha, assigned: self._region_upper_leg(alpha, assigned, 'left'),
            'right_upper_leg': lambda alpha, assigned: self._region_upper_leg(alpha, assigned, 'right'),
            'left_lower_leg': lambda alpha, assigned: self._region_lower_leg(alpha, assigned, 'left'),
            'right_lower_leg': lambda alpha, assigned: self._region_lower_leg(alpha, assigned, 'right')
        }
    
    def _region_head(self, alpha: np.ndarray, assigned: np.ndarray) -> Optional[np.ndarray]:
        """Head region: Above shoulders, around face."""
        kp = self.keypoints_pixel
        
        if 'left_shoulder' not in kp or 'right_shoulder' not in kp:
            return None
        
        # Get shoulder line (neck boundary)
        ls = kp['left_shoulder']
        rs = kp['right_shoulder']
        neck_y = min(ls[1], rs[1]) - 5
        
        # Head is everything ABOVE neck_y
        mask = alpha.copy()
        mask[neck_y:, :] = 0  # Cut off below neck
        
        # Only unassigned pixels
        mask[assigned > 0] = 0
        
        return mask
    
    def _region_body(self, alpha: np.ndarray, assigned: np.ndarray) -> Optional[np.ndarray]:
        """Body region: Between shoulders and hips, center."""
        kp = self.keypoints_pixel
        
        if not all(k in kp for k in ['left_shoulder', 'right_shoulder']):
            return None
        
        ls = kp['left_shoulder']
        rs = kp['right_shoulder']
        
        # Vertical bounds
        top_y = min(ls[1], rs[1]) - 5
        
        if 'left_hip' in kp and 'right_hip' in kp:
            lh = kp['left_hip']
            rh = kp['right_hip']
            bottom_y = max(lh[1], rh[1]) + 10
        else:
            bottom_y = top_y + int((self.height - top_y) * 0.5)
        
        # Horizontal bounds (center torso)
        center_x = (ls[0] + rs[0]) // 2
        width = abs(ls[0] - rs[0]) + 40
        left_x = max(0, center_x - width // 2)
        right_x = min(self.width, center_x + width // 2)
        
        # Create body mask
        mask = np.zeros_like(alpha)
        mask[top_y:bottom_y, left_x:right_x] = alpha[top_y:bottom_y, left_x:right_x]
        
        # Exclude already assigned pixels
        mask[assigned > 0] = 0
        
        return mask
    
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