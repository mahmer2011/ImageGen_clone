# Model Weights Directory

This directory contains model weights for various segmentation and detection tasks.

## Required Models

### CartoonSegmentation ONNX Models

For anime/cartoon character segmentation, you need ONNX model files. The code supports multiple filename variations:

#### 1. ISNet Anime Segmentation (Mask Refiner) - **REQUIRED**

**Download:**
- **Actual Filename**: `mask_refiner_isnetdis_refine_last_simplified.onnx`
- **Recommended Name**: `isnet_anime.onnx` (rename after download)
- **Source**: Hugging Face (from comfyui_animeseg or similar repositories)
- **Usage**: Refines segmentation masks for better accuracy

**Action**: Download and rename to `isnet_anime.onnx` for consistency

**Alternative Names Supported:**
- `isnet_anime.onnx` (recommended - simplified name)
- `mask_refiner_isnetdis_refine_last_simplified.onnx` (actual HuggingFace filename)
- `isnetis.onnx` (from skytnt/anime-seg)

#### 2. RTMDet-based Anime Instance Segmentation (Instance Detector) - **OPTIONAL**

**Download:**
- **Actual Filename**: `anime_segmentor_rtmdet_e60_simplified.onnx`
- **Recommended Name**: `rtmdet_anime.onnx` (rename after download)
- **Source**: Hugging Face (from comfyui_animeseg or similar repositories)
- **Usage**: Detects anime character instance bounding boxes for ROI detection

**Action**: Download and rename to `rtmdet_anime.onnx` for consistency

**Alternative Names Supported:**
- `rtmdet_anime.onnx` (recommended - simplified name)
- `anime_segmentor_rtmdet_e60_simplified.onnx` (actual HuggingFace filename)
- `ranime_segmentor_rtmdet_e60_simplified.onnx` (alternative from comfyui_animeseg)

**Note**: The ComfyUI AnimeSeg repository provides ONNX models for advanced anime segmentation. You may need to check the repository for the exact download links.

### Download Instructions

**Step 1: Download the Models**

1. **ISNet Model (Required)**:
   - **Actual Filename**: `mask_refiner_isnetdis_refine_last_simplified.onnx`
   - **Search on Hugging Face** for repositories containing "anime segmentation" or "comfyui_animeseg"
   - Look for the file `mask_refiner_isnetdis_refine_last_simplified.onnx`
   - **Rename it to**: `isnet_anime.onnx`
   - Place it in the `weights/` folder

2. **RTMDet Model (Optional but Recommended)**:
   - **Actual Filename**: `anime_segmentor_rtmdet_e60_simplified.onnx`
   - **Search on Hugging Face** for repositories containing "anime segmentation" or "comfyui_animeseg"
   - Look for the file `anime_segmentor_rtmdet_e60_simplified.onnx`
   - **Rename it to**: `rtmdet_anime.onnx`
   - Place it in the `weights/` folder

**Finding the Models:**
- Search Hugging Face for: "anime segmentation onnx" or "comfyui animeseg"
- Check repositories like `craig-tanaka/comfyui_animeseg` or similar anime segmentation projects
- Look in the repository's "Files" or "Files and versions" section for `.onnx` files

**Step 2: Verify Your Setup**

After downloading and renaming, your `weights/` folder should look like:

```
weights/
├── rtmdet_anime.onnx          (optional - improves accuracy)
├── isnet_anime.onnx            (required)
└── mobile_sam.pt               (existing)
```

**Note**: 
- The code supports both the simplified names (`rtmdet_anime.onnx`, `isnet_anime.onnx`) and the actual HuggingFace filenames
- You can use either naming convention - the code will automatically detect them
- Renaming to simplified names is recommended for consistency

**Alternative**: If you cannot find pre-converted ONNX models, the code will automatically fall back to:
1. ParsingDetector (SegFormer) - Works well for most cases
2. GeometricDetector (MediaPipe) - Final fallback

### Alternative Sources

If you cannot find the direct ONNX links, the code includes automatic fallback:
1. **CartoonSegmentationDetector** (ONNX) - Primary method, cleanest masks
2. **ParsingDetector** (SegFormer) - Fallback if ONNX models unavailable
3. **GeometricDetector** (MediaPipe) - Final fallback using geometric partitioning

The app will automatically use the best available method, so it will continue to work even without the ONNX models.

### Model Requirements

**Minimum Required:**
- `isnetis.onnx` (or alternative ISNet filename) - **REQUIRED** for CartoonSegmentation

**Optional but Recommended:**
- `ranime_segmentor_rtmdet_e60_simplified.onnx` (or alternative RTMDet filename) - Improves accuracy by detecting character instances first

**Note**: The code can work with just ISNet (full-image segmentation), but RTMDet + ISNet provides better results by focusing on detected character instances.

### Current Models

- `mobile_sam.pt` - MobileSAM model for Segment Anything Model functionality

### Notes

- Place all model files directly in this `weights/` directory
- The code supports multiple filename variations (see supported names above)
- The code will automatically detect and use available models
- If models are missing, the system will automatically fall back to alternative detection methods (SegFormer or MediaPipe)

