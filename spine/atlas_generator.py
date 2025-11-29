"""
Texture Atlas Generator

Packs segmented body part images into a texture atlas and generates
a Spine .atlas file for efficient rendering.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from PIL import Image


@dataclass
class AtlasRegion:
    """Represents a region in a texture atlas."""
    name: str
    x: int
    y: int
    width: int
    height: int
    orig_width: int
    orig_height: int
    offset_x: int = 0
    offset_y: int = 0
    rotate: bool = False


@dataclass
class PartPackingInfo:
    """Metadata describing how a single part was packed into the atlas."""
    width: int
    height: int
    trim_offset_x: int
    trim_offset_y: int


class AtlasGenerator:
    """
    Generates texture atlas from segmented body part images.
    Uses a simple bin-packing algorithm to arrange images efficiently.
    """
    
    def __init__(self, max_width: int = 4096, max_height: int = 4096, padding: int = 2):
        self.max_width = max_width
        self.max_height = max_height
        self.padding = padding
        self.regions: List[AtlasRegion] = []
        self.part_metadata: Dict[str, PartPackingInfo] = {}
        
    def pack_images(
        self,
        image_paths: Dict[str, Path],
        output_atlas_path: Path,
        output_image_path: Path
    ) -> bool:
        """
        Pack multiple images into a single atlas.
        
        Args:
            image_paths: Dict mapping part names to image file paths
            output_atlas_path: Path to save .atlas file
            output_image_path: Path to save packed image
            
        Returns:
            True if packing was successful
        """
        print(f"DEBUG pack_images: Received {len(image_paths)} image paths")
        if not image_paths:
            print("DEBUG pack_images: No image paths provided")
            return False
        
        # Reset previous regions/metadata for a clean packing run
        self.regions = []
        self.part_metadata = {}
        
        # Load and trim all images
        images: Dict[str, Image.Image] = {}
        trim_offsets: Dict[str, Tuple[int, int]] = {}
        for name, path in image_paths.items():
            if not path.exists():
                print(f"DEBUG pack_images: Path does not exist: {path}")
                continue
            
            img = Image.open(path).convert("RGBA")
            alpha = img.split()[-1]
            bbox = alpha.getbbox()
            if bbox:
                left, upper, right, lower = bbox
                trimmed = img.crop(bbox)
                offset = (int(left), int(upper))
                print(
                    f"DEBUG pack_images: Trimmed {name} to "
                    f"{trimmed.width}x{trimmed.height} (offset={offset})"
                )
            else:
                trimmed = img
                offset = (0, 0)
                print(f"DEBUG pack_images: No alpha bbox for {name}, using full image")
            
            if trimmed.width == 0 or trimmed.height == 0:
                print(f"DEBUG pack_images: Skipping {name} - empty after trimming")
                continue
            
            images[name] = trimmed
            trim_offsets[name] = offset
            self.part_metadata[name] = PartPackingInfo(
                width=trimmed.width,
                height=trimmed.height,
                trim_offset_x=offset[0],
                trim_offset_y=offset[1],
            )
        
        print(f"DEBUG pack_images: Successfully prepared {len(images)} images")
        if not images:
            print("DEBUG pack_images: No images loaded")
            return False
        
        # Sort by area (largest first) for better packing
        sorted_names = sorted(
            images.keys(),
            key=lambda n: images[n].width * images[n].height,
            reverse=True
        )
        print(f"DEBUG pack_images: Sorted names: {sorted_names}")
        
        # Simple shelf packing algorithm
        packed = self._pack_shelf(images, sorted_names)
        print(f"DEBUG pack_images: Shelf packing result: {packed}")
        
        if not packed:
            print("DEBUG pack_images: Shelf packing failed")
            return False
        
        # Create atlas image
        used_width = max(r.x + r.width for r in self.regions)
        used_height = max(r.y + r.height for r in self.regions)
        atlas_width = self._next_power_of_2(used_width)
        atlas_height = self._next_power_of_2(used_height)
        atlas_width = min(self.max_width, atlas_width)
        atlas_height = min(self.max_height, atlas_height)
        
        # Round up to power of 2 for better GPU compatibility
        atlas_image = Image.new("RGBA", (atlas_width, atlas_height), (0, 0, 0, 0))
        
        # Paste images into atlas
        for region in self.regions:
            img = images[region.name]
            atlas_image.paste(img, (region.x, region.y), img)
        
        # Save atlas image
        output_image_path.parent.mkdir(parents=True, exist_ok=True)
        atlas_image.save(output_image_path, "PNG")
        print(f"DEBUG pack_images: Atlas image saved to {output_image_path}")
        
        # Generate .atlas file
        self._write_atlas_file(output_atlas_path, output_image_path.name, atlas_width, atlas_height)
        print(f"DEBUG pack_images: Atlas file saved to {output_atlas_path}")
        
        return True
    
    def _pack_shelf(self, images: Dict[str, Image.Image], names: List[str]) -> bool:
        """
        Shelf packing algorithm: place images on horizontal shelves.
        """
        print(f"DEBUG _pack_shelf: Starting with max_width={self.max_width}, max_height={self.max_height}")
        shelf_y = self.padding
        shelf_height = 0
        shelf_x = self.padding
        
        for name in names:
            img = images[name]
            width, height = img.size
            print(f"DEBUG _pack_shelf: Placing {name} ({width}x{height}) at shelf_x={shelf_x}, shelf_y={shelf_y}")
            
            # Check if image fits on current shelf
            if shelf_x + width > self.max_width:
                # Move to next shelf
                print(f"DEBUG _pack_shelf: Moving to next shelf (width exceeded)")
                shelf_y += shelf_height + self.padding
                shelf_x = self.padding
                shelf_height = 0
            
            # Check if we've exceeded max height
            if shelf_y + height > self.max_height:
                # Atlas is too small
                print(f"DEBUG _pack_shelf: Atlas too small! shelf_y={shelf_y}, height={height}, max_height={self.max_height}")
                return False
            
            # Place image
            region = AtlasRegion(
                name=name,
                x=shelf_x,
                y=shelf_y,
                width=width,
                height=height,
                orig_width=width,
                orig_height=height
            )
            self.regions.append(region)
            
            # Update shelf position
            shelf_x += width + self.padding
            shelf_height = max(shelf_height, height)
        
        return True
    
    def _next_power_of_2(self, n: int) -> int:
        """Round up to next power of 2."""
        n = max(1, n)
        return 2 ** math.ceil(math.log2(n))
    
    def _write_atlas_file(
        self,
        output_path: Path,
        image_filename: str,
        atlas_width: int,
        atlas_height: int
    ):
        """
        Write Spine .atlas file format.
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, "w", encoding="utf-8") as f:
            # Atlas header
            f.write(f"\n{image_filename}\n")
            f.write(f"size: {atlas_width},{atlas_height}\n")
            f.write("format: RGBA8888\n")
            f.write("filter: Linear,Linear\n")
            f.write("repeat: none\n")
            
            # Write each region
            for region in self.regions:
                f.write(f"{region.name}\n")
                f.write(f"  rotate: {'true' if region.rotate else 'false'}\n")
                f.write(f"  xy: {region.x}, {region.y}\n")
                f.write(f"  size: {region.width}, {region.height}\n")
                f.write(f"  orig: {region.orig_width}, {region.orig_height}\n")
                f.write(f"  offset: {region.offset_x}, {region.offset_y}\n")
                f.write(f"  index: -1\n")


def create_atlas_from_masks(
    masks_dir: Path,
    output_dir: Path,
    atlas_name: str = "character"
) -> Tuple[Optional[Path], Optional[Path], Dict[str, PartPackingInfo]]:
    """
    Create texture atlas from a directory of segmented mask images.
    
    Args:
        masks_dir: Directory containing segmented body part images
        output_dir: Directory to save atlas files
        atlas_name: Base name for atlas files
        
    Returns:
        Tuple of (atlas_file_path, image_file_path, part_metadata)
        or (None, None, {}) on failure
    """
    print(f"DEBUG: create_atlas_from_masks called with masks_dir={masks_dir}")
    if not masks_dir.exists():
        print(f"DEBUG: Masks directory does not exist: {masks_dir}")
        return None, None, {}
    
    # Collect all PNG images, excluding non-body-part files
    image_paths: Dict[str, Path] = {}
    all_files = list(masks_dir.glob("*.png"))
    print(f"DEBUG: Found {len(all_files)} PNG files in masks directory")
    
    # Files to exclude from atlas (visualization, outlines, full character images)
    exclude_files = {
        "detection_visualization",
        "no_background", 
        "outline_only",
        "outline_overlay",
        "assembly_preview"
    }
    
    for img_file in all_files:
        stem = img_file.stem
        print(f"DEBUG: Processing {img_file.name}, stem={stem}")
        
        # Skip excluded files
        if stem in exclude_files:
            print(f"DEBUG: Skipping excluded file: {img_file.name}")
            continue
        
        # Remove the image ID prefix (8 hex chars) if present
        if len(stem) > 8 and stem[8] == '_':
            stem = stem[9:]  # Remove "imageID_"
        
        # Remove "_no_bg" prefix if present
        if stem.startswith("no_bg_"):
            stem = stem[6:]  # Remove "no_bg_"
        
        if stem:
            part_name = stem
            print(f"DEBUG: Extracted part name: {part_name}")
            image_paths[part_name] = img_file
        else:
            print(f"DEBUG: Could not extract part name from {img_file.name}")
    
    print(f"DEBUG: Collected {len(image_paths)} images for atlas: {list(image_paths.keys())}")
    
    if not image_paths:
        print("DEBUG: No valid image paths found for atlas")
        return None, None, {}
    
    # Generate atlas
    output_dir.mkdir(parents=True, exist_ok=True)
    atlas_path = output_dir / f"{atlas_name}.atlas"
    image_path = output_dir / f"{atlas_name}.png"
    
    packing_configs = [
        {"max_width": 4096, "max_height": 4096, "padding": 2},
        {"max_width": 4096, "max_height": 4096, "padding": 0},
        {"max_width": 8192, "max_height": 8192, "padding": 2},
    ]
    
    for config in packing_configs:
        print(f"DEBUG: Attempting atlas pack with config={config}")
        generator = AtlasGenerator(**config)
        success = generator.pack_images(image_paths, atlas_path, image_path)
        print(f"DEBUG: Atlas packing success={success} for config={config}")
        if success:
            return atlas_path, image_path, generator.part_metadata
    
    print("DEBUG: All atlas packing attempts failed")
    return None, None, {}


if __name__ == "__main__":
    print("Atlas generator module loaded successfully")

