"""
Character Segmenter Module
Main module for segmenting character images into body parts for Spine2D animation.
"""

import cv2
import numpy as np
import os
import json
import io
from typing import Dict, List, Tuple, Optional
from PIL import Image
from rembg import remove

from .part_detector import BodyPartDetector, BodyPart


class CharacterSegmenter:
    """
    Segments a character image into body parts for Spine2D animation.
    Handles background removal, part detection, extraction, and metadata generation.
    """
    
    def __init__(self, image_path: str, output_dir: str):
        """
        Initialize segmenter.
        
        Args:
            image_path: Path to input character image
            output_dir: Directory to save segmented parts
        """
        self.image_path = image_path
        self.output_dir = output_dir
        self.original_image = None
        self.no_bg_image = None
        self.detected_parts: Dict[str, BodyPart] = {}
        self.part_offsets: Dict[str, Tuple[int, int]] = {}
        self.part_sizes: Dict[str, Tuple[int, int]] = {}
        self.part_images: Dict[str, np.ndarray] = {}
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
    
    def load_and_prepare(self) -> bool:
        """
        Load image and remove background.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            print(f"Loading image from: {self.image_path}")
            
            # Load original image
            self.original_image = cv2.imread(self.image_path, cv2.IMREAD_UNCHANGED)
            if self.original_image is None:
                print(f"Error: Could not load image from {self.image_path}")
                return False
            
            print(f"Image loaded: {self.original_image.shape}")
            
            # Remove background using rembg
            print("Removing background...")
            with open(self.image_path, 'rb') as f:
                input_data = f.read()
            
            # Remove background
            output_data = remove(input_data)
            
            # Convert to numpy array
            no_bg_pil = Image.open(io.BytesIO(output_data)).convert("RGBA")
            self.no_bg_image = cv2.cvtColor(np.array(no_bg_pil), cv2.COLOR_RGBA2BGRA)
            
            print(f"Background removed: {self.no_bg_image.shape}")
            
            # Save background-removed image
            no_bg_path = os.path.join(self.output_dir, "no_background.png")
            cv2.imwrite(no_bg_path, self.no_bg_image)
            print(f"Saved no-background image to: {no_bg_path}")
            
            return True
            
        except Exception as e:
            print(f"Error in load_and_prepare: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def detect_parts(self) -> bool:
        """
        Detect body parts using BodyPartDetector.
        
        Returns:
            True if parts detected, False otherwise
        """
        if self.no_bg_image is None:
            print("Error: No background-removed image available")
            return False
        
        try:
            print("Detecting body parts...")
            detector = BodyPartDetector(self.no_bg_image)
            self.detected_parts = detector.detect_all_parts()
            
            print(f"Detected {len(self.detected_parts)} parts: {list(self.detected_parts.keys())}")
            
            # Save visualization
            vis = detector.visualize_detections(self.detected_parts)
            vis_path = os.path.join(self.output_dir, "detection_visualization.png")
            cv2.imwrite(vis_path, vis)
            print(f"Saved detection visualization to: {vis_path}")
            
            return len(self.detected_parts) > 0
            
        except Exception as e:
            print(f"Error in detect_parts: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def extract_parts(self) -> Dict[str, str]:
        """
        Extract detected parts as individual images with transparency.
        
        Returns:
            Dictionary of {part_name: file_path}
        """
        if not self.detected_parts:
            print("Error: No parts detected")
            return {}
        
        extracted_paths = {}
        
        try:
            for part_name, part in self.detected_parts.items():
                # Extract part with transparency
                extraction = self._extract_part_image(part)
                
                if extraction is not None:
                    part_image, x_start, y_start = extraction
                    # Save part
                    part_filename = f"{part_name}.png"
                    part_path = os.path.join(self.output_dir, part_filename)
                    cv2.imwrite(part_path, part_image)
                    
                    extracted_paths[part_name] = part_path
                    self.part_offsets[part_name] = (x_start, y_start)
                    self.part_sizes[part_name] = (
                        int(part_image.shape[1]),
                        int(part_image.shape[0]),
                    )
                    self.part_images[part_name] = part_image
                    print(f"Extracted {part_name} to: {part_path}")
            
            return extracted_paths
            
        except Exception as e:
            print(f"Error in extract_parts: {e}")
            import traceback
            traceback.print_exc()
            return {}
    
    def _extract_part_image(self, part: BodyPart) -> Optional[Tuple[np.ndarray, int, int]]:
        """
        Extract a single body part as BGRA image with transparency.
        
        Args:
            part: BodyPart object to extract
            
        Returns:
            Tuple (image, x_offset, y_offset) where offsets indicate the
            placement location on the original canvas, or None if failed.
        """
        try:
            # Get bounding box with some padding
            x, y, w, h = part.bbox
            pad = 5  # pixels padding
            extra_top = getattr(part, "overlap_padding", 0)
            extra_side = max(0, min(extra_top // 2, 20))
            
            img_height, img_width = self.no_bg_image.shape[:2]
            
            x_start = max(0, x - pad - extra_side)
            y_start = max(0, y - pad - extra_top)
            x_end = min(img_width, x + w + pad + extra_side)
            y_end = min(img_height, y + h + pad)
            
            w = max(0, x_end - x_start)
            h = max(0, y_end - y_start)
            if w == 0 or h == 0:
                return None
            
            # Create blank BGRA image
            part_image = np.zeros((h, w, 4), dtype=np.uint8)
            
            # Extract region from original
            region = self.no_bg_image[y_start:y_end, x_start:x_end].copy()
            
            # Apply part mask (adjusted for cropped region)
            mask_region = part.mask[y_start:y_end, x_start:x_end]
            
            # Copy only the pixels that belong to this part
            part_image[:, :, :3] = region[:, :, :3]  # BGR channels
            part_image[:, :, 3] = np.where(mask_region > 0, region[:, :, 3], 0)  # Alpha
            
            return part_image, x_start, y_start
            
        except Exception as e:
            print(f"Error extracting part image: {e}")
            return None
    
    def generate_metadata(self, extracted_paths: Dict[str, str]) -> Dict:
        """
        Generate metadata for Spine2D rigging.
        
        Args:
            extracted_paths: Dictionary of {part_name: file_path}
            
        Returns:
            Metadata dictionary with part info, hierarchy, and pivots
        """
        metadata = {
            "parts": [],
            "hierarchy": self._get_bone_hierarchy(),
            "image_size": {
                "width": self.no_bg_image.shape[1] if self.no_bg_image is not None else 0,
                "height": self.no_bg_image.shape[0] if self.no_bg_image is not None else 0
            },
            "character_bounds": self._get_character_bounds()
        }
        
        for part_name, part_path in extracted_paths.items():
            if part_name not in self.detected_parts:
                continue
            
            part = self.detected_parts[part_name]
            
            # Calculate pivot point
            pivot = self._calculate_pivot(part_name, part)
            
            part_info = {
                "name": part_name,
                "file": os.path.basename(part_path),
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
                "pivot": pivot,
                "parent": self._get_parent_bone(part_name),
                "z_index": self._get_z_index(part_name),
                "area": float(part.area)
            }

            offset = self.part_offsets.get(part_name)
            width, height = self.part_sizes.get(part_name, (part.bbox[2], part.bbox[3]))
            width = int(width) if width else int(part.bbox[2])
            height = int(height) if height else int(part.bbox[3])

            origin_x = int(offset[0]) if offset else int(part.bbox[0])
            origin_y = int(offset[1]) if offset else int(part.bbox[1])
            center_x = origin_x + width / 2.0
            center_y = origin_y + height / 2.0

            part_info["crop_origin"] = {
                "x": origin_x,
                "y": origin_y,
            }

            assembly_info = {
                "origin": {
                    "x": origin_x,
                    "y": origin_y,
                },
                "size": {
                    "w": int(width),
                    "h": int(height),
                },
                "center": {
                    "x": int(center_x) if center_x else int(part.center[0]),
                    "y": int(center_y) if center_y else int(part.center[1]),
                },
            }

            pivot_abs = (pivot or {}).get("absolute")
            if isinstance(pivot_abs, dict):
                assembly_info["pivot_offset"] = {
                    "x": float(pivot_abs.get("x", center_x) - center_x),
                    "y": float(pivot_abs.get("y", center_y) - center_y),
                }

            part_info["assembly"] = assembly_info
            
            metadata["parts"].append(part_info)
        
        return metadata
    
    def _get_character_bounds(self) -> Dict:
        """Get overall character bounding box."""
        if not self.detected_parts:
            return {"x": 0, "y": 0, "w": 0, "h": 0}
        
        all_x = []
        all_y = []
        
        for part in self.detected_parts.values():
            x, y, w, h = part.bbox
            all_x.extend([x, x + w])
            all_y.extend([y, y + h])
        
        min_x, max_x = min(all_x), max(all_x)
        min_y, max_y = min(all_y), max(all_y)
        
        return {
            "x": int(min_x),
            "y": int(min_y),
            "w": int(max_x - min_x),
            "h": int(max_y - min_y)
        }
    
    def _calculate_pivot(self, part_name: str, part: BodyPart) -> Dict:
        """
        Calculate pivot point for a body part.
        Pivot is where the part rotates from.
        
        Args:
            part_name: Name of the body part
            part: BodyPart object
            
        Returns:
            Dictionary with normalized and absolute pivot coordinates
        """
        x, y, w, h = part.bbox
        
        # Default pivot points (normalized 0-1) for different parts
        # Format: (x_ratio, y_ratio) where 0,0 is top-left, 1,1 is bottom-right
        pivot_ratios = {
            'head': (0.5, 0.9),        # Bottom center (neck connection)
            'body': (0.5, 0.3),        # Upper center (chest/neck area)
            'left_upper_arm': (0.8, 0.2),    # Top right (shoulder)
            'right_upper_arm': (0.2, 0.2),   # Top left (shoulder)
            'left_lower_arm': (0.2, 0.2),    # Top left (elbow)
            'right_lower_arm': (0.8, 0.2),   # Top right (elbow)
            'left_upper_leg': (0.5, 0.1),    # Top center (hip)
            'right_upper_leg': (0.5, 0.1),   # Top center (hip)
            'left_lower_leg': (0.5, 0.1),    # Top center (knee)
            'right_lower_leg': (0.5, 0.1),   # Top center (knee)
            'tail': (0.1, 0.5),        # Left center (base attachment)
        }
        
        pivot_norm = pivot_ratios.get(part_name, (0.5, 0.5))
        
        # Calculate absolute pivot position
        pivot_x = x + w * pivot_norm[0]
        pivot_y = y + h * pivot_norm[1]
        
        return {
            "normalized": {
                "x": float(pivot_norm[0]),
                "y": float(pivot_norm[1])
            },
            "absolute": {
                "x": int(pivot_x),
                "y": int(pivot_y)
            },
            "relative_to_part": {
                "x": int(w * pivot_norm[0]),
                "y": int(h * pivot_norm[1])
            }
        }
    
    def _get_parent_bone(self, part_name: str) -> Optional[str]:
        """
        Get parent bone for hierarchy.
        Defines which bone this part is attached to.
        """
        hierarchy = {
            'head': 'body',
            'left_upper_arm': 'body',
            'right_upper_arm': 'body',
            'left_lower_arm': 'left_upper_arm',
            'right_lower_arm': 'right_upper_arm',
            'left_upper_leg': 'body',
            'right_upper_leg': 'body',
            'left_lower_leg': 'left_upper_leg',
            'right_lower_leg': 'right_upper_leg',
            'tail': 'body',
            'body': None  # Root bone
        }
        return hierarchy.get(part_name)
    
    def _get_z_index(self, part_name: str) -> int:
        """
        Get Z-index for layering (draw order).
        Higher number = drawn on top.
        """
        z_order = {
            'right_lower_leg': 0,    # Back lower leg
            'right_upper_leg': 1,    # Back upper leg
            'right_lower_arm': 2,    # Back lower arm
            'right_upper_arm': 3,    # Back upper arm
            'tail': 4,               # Behind body
            'body': 5,               # Main body
            'head': 6,               # Head on top of body
            'left_upper_arm': 7,     # Front upper arm
            'left_lower_arm': 8,     # Front lower arm
            'left_upper_leg': 9,     # Front upper leg
            'left_lower_leg': 10     # Front lower leg
        }
        return z_order.get(part_name, 3)
    
    def _get_bone_hierarchy(self) -> List[Dict]:
        """
        Define bone hierarchy for Spine2D.
        This defines the parent-child relationships for animation.
        """
        return [
            {"name": "root", "parent": None, "description": "Root bone (invisible)"},
            {"name": "body", "parent": "root", "description": "Main body/torso"},
            {"name": "head", "parent": "body", "description": "Head"},
            {"name": "left_upper_arm", "parent": "body", "description": "Left upper arm (shoulder to elbow)"},
            {"name": "left_lower_arm", "parent": "left_upper_arm", "description": "Left lower arm (elbow to wrist)"},
            {"name": "right_upper_arm", "parent": "body", "description": "Right upper arm (shoulder to elbow)"},
            {"name": "right_lower_arm", "parent": "right_upper_arm", "description": "Right lower arm (elbow to wrist)"},
            {"name": "left_upper_leg", "parent": "body", "description": "Left upper leg (hip to knee)"},
            {"name": "left_lower_leg", "parent": "left_upper_leg", "description": "Left lower leg (knee to ankle)"},
            {"name": "right_upper_leg", "parent": "body", "description": "Right upper leg (hip to knee)"},
            {"name": "right_lower_leg", "parent": "right_upper_leg", "description": "Right lower leg (knee to ankle)"}
        ]
    
    def segment_all_parts(self) -> Dict[str, str]:
        """
        Main segmentation method - performs complete segmentation pipeline.
        
        Returns:
            Dictionary of {part_name: file_path}, including 'metadata' key
        """
        print("=" * 60)
        print("Starting Character Segmentation Pipeline")
        print("=" * 60)
        
        # Step 1: Load and prepare image
        if not self.load_and_prepare():
            print("Failed to load and prepare image")
            return {}
        
        # Step 2: Detect body parts
        if not self.detect_parts():
            print("Failed to detect body parts")
            return {}
        
        # Step 3: Extract parts as separate images
        extracted_paths = self.extract_parts()
        if not extracted_paths:
            print("Failed to extract body parts")
            return {}
        
        # Step 4: Generate metadata
        print("Generating metadata...")
        metadata = self.generate_metadata(extracted_paths)
        metadata_path = os.path.join(self.output_dir, "metadata.json")
        
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        print(f"Saved metadata to: {metadata_path}")
        
        # Add metadata to results
        extracted_paths['metadata'] = metadata_path
        
        # Step 5: Generate outline overlay for full character
        outline_paths = self.create_outline_images()
        for key, path in outline_paths.items():
            extracted_paths[key] = path

        # Step 6: Generate an assembled preview rendered from the segmented parts
        assembly_preview = self.create_assembly_preview()
        if assembly_preview:
            extracted_paths["assembly_preview"] = assembly_preview
        
        print("=" * 60)
        print("Segmentation Complete!")
        print(f"Total parts extracted: {len(extracted_paths) - 1}")
        print(f"Output directory: {self.output_dir}")
        print("=" * 60)
        
        return extracted_paths
    
    def create_assembly_preview(self) -> Optional[str]:
        """
        Create a preview showing all parts assembled together.
        Useful for verifying segmentation quality.
        
        Returns:
            Path to preview image, or None if failed
        """
        if not self.detected_parts:
            return None
        
        try:
            # Create blank canvas
            h, w = self.no_bg_image.shape[:2]
            preview = np.zeros((h, w, 4), dtype=np.uint8)
            
            # Sort parts by z-index
            sorted_parts = sorted(
                self.detected_parts.items(),
                key=lambda x: self._get_z_index(x[0])
            )
            
            # Draw each part in order
            for part_name, _ in sorted_parts:
                offset = self.part_offsets.get(part_name)
                if not offset:
                    continue
                
                part_img = self.part_images.get(part_name)
                if part_img is None:
                    part_path = os.path.join(self.output_dir, f"{part_name}.png")
                    if not os.path.exists(part_path):
                        continue
                    part_img = cv2.imread(part_path, cv2.IMREAD_UNCHANGED)
                    if part_img is None or part_img.shape[2] < 4:
                        continue
                    self.part_images[part_name] = part_img
                
                x_start, y_start = offset
                part_h, part_w = part_img.shape[:2]
                
                # Calculate valid region bounds
                end_y = min(y_start + part_h, preview.shape[0])
                end_x = min(x_start + part_w, preview.shape[1])
                part_end_y = min(part_h, end_y - y_start)
                part_end_x = min(part_w, end_x - x_start)
                
                if end_y > y_start and end_x > x_start and part_end_y > 0 and part_end_x > 0:
                    alpha = part_img[:part_end_y, :part_end_x, 3:4] / 255.0
                    preview[y_start:end_y, x_start:end_x] = (
                        preview[y_start:end_y, x_start:end_x] * (1 - alpha) +
                        part_img[:part_end_y, :part_end_x] * alpha
                    ).astype(np.uint8)
            
            # Save preview
            preview_path = os.path.join(self.output_dir, "assembly_preview.png")
            cv2.imwrite(preview_path, preview)
            print(f"Saved assembly preview to: {preview_path}")
            
            return preview_path
            
        except Exception as e:
            print(f"Error creating assembly preview: {e}")
            return None

    def create_outline_images(
        self,
        outline_color: Tuple[int, int, int] = (0, 255, 0),
        outline_thickness: int = 4
    ) -> Dict[str, str]:
        """
        Create outline renderings of the full character silhouette.
        
        Returns:
            Dictionary mapping outline variant name to saved file path.
        """
        if self.no_bg_image is None:
            print("Outline generation skipped: no processed image available.")
            return {}
        
        alpha_channel = self.no_bg_image[:, :, 3]
        if alpha_channel is None or not np.any(alpha_channel):
            print("Outline generation skipped: alpha channel missing or empty.")
            return {}
        
        try:
            # Detect edges from the alpha channel (character silhouette)
            edges = cv2.Canny(alpha_channel, 30, 150)
            
            # Thicken edges to make the outline visible
            kernel_size = max(3, outline_thickness)
            if kernel_size % 2 == 0:
                kernel_size += 1  # ensure odd kernel size
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
            outline_mask = cv2.dilate(edges, kernel, iterations=1)
            
            outline_paths: Dict[str, str] = {}
            
            # Outline-only image (transparent background)
            outline_only = np.zeros_like(self.no_bg_image)
            outline_only[:, :, :3] = outline_color
            outline_only[:, :, 3] = outline_mask
            outline_only_path = os.path.join(self.output_dir, "outline_only.png")
            cv2.imwrite(outline_only_path, outline_only)
            outline_paths["outline_only"] = outline_only_path
            print(f"Saved outline-only image to: {outline_only_path}")
            
            # Overlay outline on the background-removed character for easier preview
            outline_overlay = self.no_bg_image.copy()
            mask_indices = outline_mask > 0
            outline_overlay[:, :, :3][mask_indices] = outline_color
            outline_overlay[:, :, 3][mask_indices] = 255
            outline_overlay_path = os.path.join(self.output_dir, "outline_overlay.png")
            cv2.imwrite(outline_overlay_path, outline_overlay)
            outline_paths["outline_overlay"] = outline_overlay_path
            print(f"Saved outline overlay image to: {outline_overlay_path}")
            
            return outline_paths
        
        except Exception as e:
            print(f"Error creating outline images: {e}")
            return {}