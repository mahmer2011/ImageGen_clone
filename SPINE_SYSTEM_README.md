# Spine2D Animation Generation System

## Overview

A complete AI-powered system for creating, editing, and exporting Spine2D animation projects using natural language commands and automated character generation.

## Features

### 🎨 Character Generation
- AI-enhanced prompt generation for animation-ready characters
- Automatic image generation with proper body structure
- Full body segmentation using SAM (Segment Anything Model)
- MediaPipe pose detection for skeletal rigging
- Overlap-safe per-part masks with tightened head/limb regions for cleaner exports

### 🦴 Spine2D Integration
- Automatic skeleton creation from pose landmarks
- Bone hierarchy with proper parent-child relationships
- Texture atlas generation from segmented body parts
- Automatic weight painting for mesh deformation

### 🎬 Animation System
- Pre-defined animations:
  - **Idle**: Breathing/standing animation
  - **Walk**: Looping walk cycle
  - **Jump**: Jump with anticipation and landing
  - **Wave**: Waving hand gestures (left/right)
- Custom animation creation support
- Animation speed control
- Keyframe editing

### 🖥️ Web Preview
- Real-time Spine animation preview using PixiJS
- Interactive animation controls
- Playback speed adjustment
- Animation selection buttons

### 💬 Chat Interface
- Natural language command processing
- AI-powered prompt analysis using OpenAI
- Contextual responses
- Guided workflows

### 📦 Export
- Complete Spine project export as ZIP
- Includes .json, .atlas, and texture files
- Compatible with Spine Editor
- README documentation included

## Architecture

### Phase 1: Spine Export Foundation
- **spine_exporter.py**: Converts MediaPipe landmarks to Spine JSON
- **atlas_generator.py**: Packs segmented parts into texture atlas
- **weight_painter.py**: Automatic vertex weight assignment

### Phase 2: Animation Creation
- **animation_engine.py**: Keyframe system and timeline management
- **animations/*.py**: Pre-defined animation templates
- **animation_builder.py**: Animation application and management

### Phase 3: Web Runtime & Preview
- **static/js/spine_viewer.js**: PixiJS-based Spine runtime
- **static/css/spine_viewer.css**: Viewer styling
- API endpoints for animation data

### Phase 4: Chat Interface
- **chat/prompt_analyzer.py**: AI-powered command parsing
- **chat/command_processors.py**: Command execution
- **chat/chat_handler.py**: Session management
- **templates/chat.html**: Chat UI

### Phase 5: AI-Powered Editing
- **spine/animation_editor.py**: Animation modification system
- **spine/skeleton_editor.py**: Skeleton adjustment tools

### Phase 6: Export & Download
- **spine/project_packager.py**: ZIP packaging system
- Export endpoints in app.py

## Usage

### Standard Workflow

1. **Generate Character**
   - Enter character description (e.g., "a cool boy with multicolored hair")
   - Click "Enhance Prompt"
   - Review and edit enhanced prompt
   - Click "Generate Image"

2. **Segment Character**
   - Click "Segment Image" button
   - Wait for SAM to segment body parts
   - View segmented masks

3. **Create Spine Skeleton**
   - Click "Create Spine Skeleton"
   - System automatically:
     - Detects pose landmarks
     - Creates bone hierarchy
     - Generates texture atlas
     - Adds default animations

4. **Preview Animations**
   - Use animation buttons to preview
   - Adjust playback speed
   - Test different animations

5. **Export Project**
   - Click "Export Spine Project"
   - Download ZIP file
   - Open in Spine Editor

### Chat Interface Workflow

1. **Access Chat**
   - Click "Try the Chat Interface" link
   - Or navigate to `/chat`

2. **Use Natural Language**
   - "Create a blue cat walking"
   - "Make the jump faster"
   - "Add a wave animation"
   - "Export project"

3. **Follow Guided Steps**
   - Chat provides step-by-step guidance
   - Integrates with main interface
   - Contextual help and suggestions

## API Endpoints

### `/api/create-spine` (POST)
Creates Spine skeleton from segmented image.

**Request:**
```json
{
  "image_id": "abc12345"
}
```

**Response:**
```json
{
  "success": true,
  "spine_json": "/static/generated_images/spine/abc12345/abc12345.json",
  "spine_atlas": "/static/generated_images/spine/abc12345/abc12345.atlas",
  "image_id": "abc12345"
}
```

### `/api/animations` (GET)
Lists all available animations.

**Response:**
```json
{
  "animations": ["idle", "walk", "jump", "wave_r", "wave_l"]
}
```

### `/export-spine/<image_id>` (GET)
Downloads complete Spine project as ZIP.

### `/chat` (GET)
Renders chat interface.

### `/chat/message` (POST)
Processes chat messages.

**Request:**
```json
{
  "message": "Create a blue cat"
}
```

**Response:**
```json
{
  "response": "To create 'a blue cat'...",
  "action": "guide",
  "command": {...},
  "data": {...}
}
```

## Dependencies

### Core
- Flask >= 3.0.0
- OpenAI >= 1.0.0
- Pillow >= 10.0.0

### Image Processing
- rembg >= 2.0.0
- opencv-python >= 4.8.0
- numpy >= 1.24.0
- mediapipe >= 0.10.0

### Deep Learning
- torch >= 2.0.0
- segment-anything
- onnxruntime >= 1.15.0

### Animation
- scipy >= 1.11.0

### Frontend
- PixiJS (CDN)
- pixi-spine 4.0.4 (CDN)

## Configuration

### Environment Variables

```bash
# Required
BFL_API_KEY=your_bfl_api_key
OPENAI_API_KEY=your_openai_api_key

# Optional
FLASK_SECRET_KEY=your_secret_key
OPENAI_MODEL=gpt-4o-mini
BFL_ENDPOINT=https://api.bfl.ai/v1/flux-kontext-pro
BFL_ASPECT_RATIO=2:3
PORT=5006

# For segmentation
ENABLE_SAM=1
SAM_CHECKPOINT=path/to/sam_vit_b.pth
```

## File Structure

```
imgGen/
├── app.py                          # Main Flask application
├── requirements.txt                # Python dependencies
├── SPINE_SYSTEM_README.md         # This file
├── static/
│   ├── js/
│   │   └── spine_viewer.js        # Spine runtime viewer
│   ├── css/
│   │   └── spine_viewer.css       # Viewer styles
│   └── generated_images/
│       ├── *.png                   # Generated images
│       ├── segmented_masks/        # SAM segmentation output
│       └── spine/                  # Spine projects
│           └── <image_id>/
│               ├── *.json          # Spine skeleton
│               ├── *.atlas         # Texture atlas
│               └── *.png           # Atlas texture
├── templates/
│   ├── index.html                  # Main interface
│   └── chat.html                   # Chat interface
├── segmentation/
│   ├── segmentation.py             # Background removal & pose
│   ├── sam.py                      # SAM integration
│   ├── human_part_regions.py      # Region building
│   └── human_rig_schema.py        # Rig definition
├── spine/
│   ├── spine_exporter.py          # Skeleton exporter
│   ├── atlas_generator.py         # Atlas packer
│   ├── weight_painter.py          # Weight painting
│   ├── animation_engine.py        # Animation system
│   ├── animation_builder.py       # Animation manager
│   ├── animation_editor.py        # Animation editing
│   ├── skeleton_editor.py         # Skeleton editing
│   ├── project_packager.py        # ZIP export
│   └── animations/                # Animation templates
│       ├── walk.py
│       ├── jump.py
│       ├── wave.py
│       └── idle.py
└── chat/
    ├── chat_handler.py            # Chat session manager
    ├── command_processors.py      # Command execution
    └── prompt_analyzer.py         # AI prompt parsing
```

## Spine JSON Format

The system generates Spine 4.1 compatible JSON with:

- **skeleton**: Metadata and configuration
- **bones**: Hierarchical bone structure
- **slots**: Draw order definitions
- **skins**: Attachment mappings
- **animations**: Keyframe timelines

## Animation Templates

### Idle (3.0s)
Subtle breathing and movement for standing characters.

### Walk (1.0s)
Complete walk cycle with arm swing and leg movement.

### Jump (1.2s)
Jump with anticipation, launch, peak, fall, and landing phases.

### Wave (2.0s)
Waving gesture with configurable hand (left/right).

## Extending the System

### Adding New Animations

1. Create new animation file in `spine/animations/`:

```python
from spine.animation_engine import Animation

def create_custom_animation(duration: float = 1.0) -> Animation:
    animation = Animation(name="custom", duration=duration)
    
    # Add keyframes
    animation.add_bone_rotation("head", 0.0, 0)
    animation.add_bone_rotation("head", 0.5, 15)
    animation.add_bone_rotation("head", duration, 0)
    
    return animation
```

2. Register in `animation_builder.py`:

```python
from spine.animations.custom import create_custom_animation

# In AnimationBuilder._register_default_animations():
custom = create_custom_animation()
self.engine.animations[custom.name] = custom
```

### Adding New Chat Commands

1. Update `prompt_analyzer.py` to recognize new commands
2. Add handler in `command_processors.py`
3. Test in chat interface

## Troubleshooting

### Segmentation Not Working
- Ensure SAM checkpoint is downloaded
- Set `ENABLE_SAM=1` environment variable
- Check SAM_CHECKPOINT path

### Pose Detection Fails
- Ensure character is clearly visible
- Check for proper body separation (arms away from body)
- Verify MediaPipe installation

### Animation Preview Not Loading
- Check browser console for errors
- Ensure PixiJS and pixi-spine CDN are accessible
- Verify Spine JSON and atlas paths are correct

### Chat Commands Not Working
- Verify OPENAI_API_KEY is set
- Check API usage limits
- Review server logs for errors

## Performance Optimization

- Image generation: 10-30 seconds
- Segmentation: 5-15 seconds
- Skeleton creation: 1-2 seconds
- Animation preview: Real-time at 60fps

## Future Enhancements

- IK (Inverse Kinematics) constraints
- Mesh deformation support
- Custom animation from motion capture
- Multi-character scenes
- Animation blending and transitions
- Skin/clothing attachment system
- Real-time skeleton editing in UI
- Animation timeline editor

## Credits

- Spine2D format by Esoteric Software
- MediaPipe by Google
- SAM (Segment Anything) by Meta
- PixiJS and pixi-spine by PixiJS team
- Flask web framework
- OpenAI GPT models

## License

This system is for educational and development purposes. Spine2D is a commercial product - exported projects should be used in accordance with Esoteric Software's licensing terms.

---

Generated by the imgGen Spine2D Animation System

