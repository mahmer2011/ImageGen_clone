"""
AnimatedDrawings Rigging Integration
Uses Meta's model to detect a stable skeleton structure.
"""
import cv2
import numpy as np
import yaml
from pathlib import Path
from pkg_resources import resource_filename

# Try importing the library
try:
    from animated_drawings import render
    from animated_drawings.model.pose_estimation import predict_pose
    AD_AVAILABLE = True
except ImportError:
    AD_AVAILABLE = False
    print("Warning: animated_drawings not installed.")

class AnimatedDrawingsRig:
    def __init__(self):
        self.available = AD_AVAILABLE

    def get_landmarks(self, image_path: str):
        """
        Runs AnimatedDrawings pose estimation on the image.
        Returns landmarks in the format expected by your simple_spine_builder.
        """
        if not self.available:
            return None

        print(f"Running AnimatedDrawings Pose Estimation on {image_path}...")
        
        # AD requires a specific config/predict call
        # We assume standard pretrained weights bundled with the library
        try:
            # Run inference
            # Note: predict_pose usually expects a file path or numpy array
            # and returns a dictionary of joints
            keypoints = predict_pose(image_path) 
            
            # Convert AD keypoints to our naming convention
            # AD Format: usually 0=Nose, 1=Neck, ... (similar to OpenPose)
            return self._convert_to_system_landmarks(keypoints)
            
        except Exception as e:
            print(f"AD Rigging Error: {e}")
            return None

    def _convert_to_system_landmarks(self, ad_kps):
        # Map AnimatedDrawings indices to your system names
        # (This mapping needs verification against AD's specific output)
        mapping = {
            0: 'nose', 1: 'neck', 2: 'right_shoulder', 3: 'right_elbow', 
            4: 'right_wrist', 5: 'left_shoulder', 6: 'left_elbow', 
            7: 'left_wrist', 8: 'right_hip', 9: 'right_knee', 
            10: 'right_ankle', 11: 'left_hip', 12: 'left_knee', 
            13: 'left_ankle', 14: 'right_eye', 15: 'left_eye', 
            16: 'right_ear', 17: 'left_ear'
        }
        
        system_lms = []
        for idx, name in mapping.items():
            if idx < len(ad_kps):
                x, y = ad_kps[idx][:2] # Take x,y (ignore confidence)
                system_lms.append({'name': name, 'x': x, 'y': y})
                
        return system_lms