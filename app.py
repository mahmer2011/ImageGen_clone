import os
import time
import uuid
import json
import zipfile
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple

import requests
from dotenv import load_dotenv
from flask import Flask, flash, render_template, request, jsonify, send_file
from openai import OpenAI
from PIL import Image, ImageOps
from werkzeug.datastructures import FileStorage

load_dotenv()

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", os.urandom(16))

# Lazy import to avoid heavy dependencies at startup when not needed
try:
    from segmentation.segmenter import CharacterSegmenter
    from segmentation.part_detector import BodyPartDetector
    import mediapipe as mp
    # Helper function to extract MediaPipe landmarks from detector
    def extract_landmarks_from_detector(detector: BodyPartDetector) -> Optional[List[Dict[str, Any]]]:
        """Extract MediaPipe landmarks in the format expected by simple_spine_builder."""
        if not hasattr(detector, 'keypoints_pixel') or not detector.keypoints_pixel:
            return None
        
        landmarks = []
        for name, (x, y) in detector.keypoints_pixel.items():
            landmarks.append({
                "name": name,
                "x": float(x),
                "y": float(y),
                "z": 0.0,
                "visibility": 1.0
            })
        return landmarks if landmarks else None
except Exception as e:
    print(f"Warning: New segmentation modules not available: {e}")
    CharacterSegmenter = None  # type: ignore[assignment]
    BodyPartDetector = None  # type: ignore[assignment]
    extract_landmarks_from_detector = None  # type: ignore[assignment]
    mp = None

# Spine-related imports
try:
    from spine.simple_spine_builder import create_simple_skeleton
    from spine.atlas_generator import create_atlas_from_masks
    from spine.animation_builder import add_default_animations_to_skeleton, get_animation_builder
    from spine.project_packager import package_spine_project
except Exception as e:
    print(f"Warning: Spine modules not available: {e}")
    create_simple_skeleton = None  # type: ignore[assignment]
    create_atlas_from_masks = None  # type: ignore[assignment]
    add_default_animations_to_skeleton = None  # type: ignore[assignment]
    get_animation_builder = None  # type: ignore[assignment]
    package_spine_project = None  # type: ignore[assignment]

# Chat system imports
try:
    from chat.chat_handler import ChatHandler
    from chat.command_processors import CommandProcessor
    from chat.prompt_analyzer import PromptAnalyzer
except Exception as e:
    print(f"Warning: Chat modules not available: {e}")
    ChatHandler = None  # type: ignore[assignment]
    CommandProcessor = None  # type: ignore[assignment]
    PromptAnalyzer = None  # type: ignore[assignment]

BFL_API_KEY = os.environ.get("BFL_API_KEY")
BFL_ENDPOINT = os.environ.get("BFL_ENDPOINT", "https://api.bfl.ai/v1/flux-kontext-pro")
BFL_POLL_INTERVAL = float(os.environ.get("BFL_POLL_INTERVAL", "1.0"))
BFL_POLL_TIMEOUT = float(os.environ.get("BFL_POLL_TIMEOUT", "180"))  # Increased to 3 minutes for complex prompts

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")


def build_landmarks_from_segmentation(metadata: Dict[str, Any]) -> Optional[List[Dict[str, Any]]]:
    """
    Construct pseudo-MediaPipe landmarks directly from segmentation metadata pivots.
    This keeps the skeleton aligned with the assembled parts when MediaPipe struggles.
    """
    parts_data = metadata.get("parts") or []
    if not parts_data:
        return None
    
    parts: Dict[str, Dict[str, Any]] = {}
    for part in parts_data:
        name = part.get("name")
        if name:
            parts[name] = part
    
    def _pivot_abs(part_name: str) -> Optional[Tuple[float, float]]:
        part = parts.get(part_name)
        if not part:
            return None
        pivot = (part.get("pivot") or {}).get("absolute")
        if not isinstance(pivot, dict):
            return None
        x = pivot.get("x")
        y = pivot.get("y")
        if isinstance(x, (int, float)) and isinstance(y, (int, float)):
            return float(x), float(y)
        return None
    
    def _bbox_point(part_name: str, rel_x: float, rel_y: float) -> Optional[Tuple[float, float]]:
        part = parts.get(part_name)
        if not part:
            return None
        bbox = part.get("bbox") or {}
        x = bbox.get("x")
        y = bbox.get("y")
        w = bbox.get("w")
        h = bbox.get("h")
        if not all(isinstance(val, (int, float)) for val in (x, y, w, h)):
            return None
        return (
            float(x) + float(w) * rel_x,
            float(y) + float(h) * rel_y,
        )
    
    def _bbox_bottom_center(part_name: str) -> Optional[Tuple[float, float]]:
        return _bbox_point(part_name, 0.5, 1.0)
    
    def _head_tip() -> Optional[Tuple[float, float]]:
        part = parts.get("head")
        if not part:
            return None
        bbox = part.get("bbox") or {}
        x = bbox.get("x")
        y = bbox.get("y")
        w = bbox.get("w")
        if not all(isinstance(val, (int, float)) for val in (x, y, w)):
            return None
        return float(x) + float(w) / 2.0, float(y) + (bbox.get("h", 0) or 0) * 0.1
    
    landmark_map = {
        "nose": _head_tip(),
        "left_shoulder": _pivot_abs("left_upper_arm"),
        "right_shoulder": _pivot_abs("right_upper_arm"),
        "left_elbow": _pivot_abs("left_lower_arm"),
        "right_elbow": _pivot_abs("right_lower_arm"),
        "left_wrist": _bbox_bottom_center("left_lower_arm"),
        "right_wrist": _bbox_bottom_center("right_lower_arm"),
        "left_hip": _pivot_abs("left_upper_leg"),
        "right_hip": _pivot_abs("right_upper_leg"),
        "left_knee": _pivot_abs("left_lower_leg"),
        "right_knee": _pivot_abs("right_lower_leg"),
        "left_ankle": _bbox_bottom_center("left_lower_leg"),
        "right_ankle": _bbox_bottom_center("right_lower_leg"),
    }
    
    landmarks: List[Dict[str, Any]] = []
    for name, point in landmark_map.items():
        if not point:
            continue
        landmarks.append({
            "name": name,
            "x": float(point[0]),
            "y": float(point[1]),
            "z": 0.0,
            "visibility": 1.0,
        })
    
    return landmarks if landmarks else None

# Supported file types for manual uploads
ALLOWED_UPLOAD_EXTENSIONS = {"png", "jpg", "jpeg", "webp"}

# Initialize OpenAI client if API key is available
openai_client = None
if OPENAI_API_KEY:
    openai_client = OpenAI(api_key=OPENAI_API_KEY)

# Global chat handler instance - initialize after openai_client
chat_handler = None
if ChatHandler and CommandProcessor and PromptAnalyzer and openai_client:
    analyzer = PromptAnalyzer(openai_client)
    processor = CommandProcessor(app)
    chat_handler = ChatHandler(analyzer, processor)


@app.route("/", methods=["GET", "POST"])
def index():
    prompt = ""
    enhanced_prompt = ""
    image_type = ""
    image_path: Optional[str] = None
    remote_url: Optional[str] = None
    segmented_masks: list[str] = []
    assembly_preview: Optional[str] = None

    if request.method == "POST":
        action = request.form.get("action", "")
        prompt = request.form.get("prompt", "").strip()
        enhanced_prompt = request.form.get("enhanced_prompt", "").strip()
        image_type = request.form.get("image_type", "").strip()

        if action == "enhance":
            # Just enhance the prompt and show it
            if not prompt:
                flash("Please enter a prompt before enhancing.")
            else:
                try:
                    enhanced_prompt = generate_enhanced_prompt(prompt)
                    if enhanced_prompt:
                        # Classify the image type based on enhanced prompt
                        image_type = classify_image_type(enhanced_prompt)
                        flash("Prompt enhanced successfully! You can edit it below and then generate the image.", "success")
                    else:
                        flash("Prompt enhancement is not available. Please set OPENAI_API_KEY in your .env file.")
                except Exception as exc:
                    flash(f"Error enhancing prompt: {exc}")
        
        elif action == "generate":
            # Generate image using the enhanced prompt (or original if no enhanced)
            final_prompt = enhanced_prompt if enhanced_prompt else prompt

            if not final_prompt:
                flash("Please enter a prompt or enhance one before generating.")
            elif not BFL_API_KEY:
                flash("Set the BFL_API_KEY environment variable before generating images.")
            else:
                try:
                    # Generate new image
                    submission = submit_generation(final_prompt)
                    polling_url = submission["polling_url"]
                    remote_url = poll_for_result(polling_url, submission.get("id"))

                    if remote_url:
                        # Get image type from form or classify if not present
                        if not image_type and enhanced_prompt:
                            image_type = classify_image_type(enhanced_prompt)
                        image_path = download_image(remote_url, final_prompt, enhanced_prompt if enhanced_prompt else None, image_type)
                        flash("Image generated successfully.", "success")
                    else:
                        flash(f"Image generation timed out after {BFL_POLL_TIMEOUT} seconds. Complex prompts may take longer. Try again or simplify the prompt.", "warning")
                except requests.HTTPError as http_error:
                    status_code = http_error.response.status_code if http_error.response else ""
                    detail = http_error.response.text if http_error.response else str(http_error)
                    flash(f"API request failed ({status_code}): {detail}")
                except Exception as exc:
                    flash(f"Unexpected error: {exc}")
        
        elif action == "upload":
            upload_prompt = request.form.get("upload_prompt", "").strip()
            upload_image_type = request.form.get("upload_image_type", "").strip()
            uploaded_file = request.files.get("uploaded_image")

            if not uploaded_file or uploaded_file.filename == "":
                flash("Please select an image to upload.", "error")
            else:
                try:
                    image_path = save_uploaded_image(
                        uploaded_file,
                        prompt=upload_prompt,
                        image_type=upload_image_type or None,
                    )
                    remote_url = None
                    segmented_masks = []
                    if upload_image_type:
                        image_type = upload_image_type
                    flash("Image uploaded successfully. You can now run segmentation.", "success")
                except ValueError as exc:
                    flash(str(exc), "error")
                except Exception as exc:
                    flash(f"Failed to upload image: {exc}", "error")
        
        elif action == "segment":
            # Run new segmentation pipeline using CharacterSegmenter
            image_path = request.form.get("image_path", "").strip()

            if not image_path:
                flash("No image available to segment. Please generate an image first.", "error")
            elif not CharacterSegmenter:
                flash("Segmentation utilities are not available on this server.", "error")
            else:
                try:
                    # Build absolute paths
                    static_root = os.path.join(app.root_path, "static")
                    input_abs = os.path.join(static_root, image_path.replace("/", os.sep))

                    if not os.path.exists(input_abs):
                        flash("Could not find the image file on the server for segmentation.", "error")
                    else:
                        # Extract image ID from filename (e.g., "abc12345.png" -> "abc12345")
                        base_dir, filename = os.path.split(input_abs)
                        name, ext = os.path.splitext(filename)
                        image_id = name
                        
                        # Create output directory for segmented parts
                        output_dir = os.path.join(base_dir, "segmented_masks", f"{image_id}_no_bg")
                        os.makedirs(output_dir, exist_ok=True)
                        
                        # Run segmentation
                        segmenter = CharacterSegmenter(input_abs, output_dir)
                        extracted_paths = segmenter.segment_all_parts()
                        
                        if extracted_paths:
                            exclude_parts = {"metadata", "outline_only", "outline_overlay", "assembly_preview"}
                            # Collect segmented part images for display (exclude metadata and previews)
                            for part_name, part_path in extracted_paths.items():
                                if not part_path or not os.path.exists(part_path):
                                    continue
                                
                                rel_path = os.path.relpath(part_path, static_root).replace(os.sep, "/")
                                
                                if part_name == "assembly_preview":
                                    assembly_preview = rel_path
                                    continue
                                
                                if part_name in exclude_parts:
                                    continue
                                
                                segmented_masks.append(rel_path)
                            
                            if segmented_masks:
                                flash(
                                    f"Segmentation complete! Detected {len(segmented_masks)} body parts.",
                                    "success",
                                )
                            else:
                                flash(
                                    "Segmentation completed but no part images were found to display.",
                                    "warning",
                                )
                        else:
                            flash("Segmentation failed - no parts were extracted.", "error")
                            
                except Exception as exc:
                    flash(f"Segmentation failed: {exc}", "error")
                    import traceback
                    traceback.print_exc()

        # If an image is displayed, try to read its metadata to get image type
        if image_path and not image_type:
            metadata = read_image_metadata(image_path)
            if metadata.get("image_type"):
                image_type = metadata["image_type"]
            # Also populate enhanced_prompt from metadata if available
            if not enhanced_prompt and metadata.get("enhanced_prompt"):
                enhanced_prompt = metadata["enhanced_prompt"]

    return render_template(
        "index.html",
        prompt=prompt,
        enhanced_prompt=enhanced_prompt,
        image_type=image_type,
        image_path=image_path,
        remote_url=remote_url,
        segmented_masks=segmented_masks,
        assembly_preview=assembly_preview,
    )


def submit_generation(prompt: str) -> dict:
    payload = {
        "prompt": prompt,
        # Use a taller canvas by default so the full body (head to feet) fits comfortably.
        # Can be overridden via BFL_ASPECT_RATIO if needed.
        "aspect_ratio": os.environ.get("BFL_ASPECT_RATIO", "2:3"),
    }
    headers = {
        "accept": "application/json",
        "Content-Type": "application/json",
        "x-key": require_api_key(),
    }

    response = requests.post(BFL_ENDPOINT, json=payload, headers=headers, timeout=30)
    response.raise_for_status()
    data = response.json()

    if "polling_url" not in data:
        raise ValueError("API response missing polling_url.")

    return data


def poll_for_result(polling_url: str, request_id: Optional[str]) -> Optional[str]:
    deadline = time.time() + BFL_POLL_TIMEOUT
    headers = {
        "accept": "application/json",
        "x-key": require_api_key(),
    }
    
    poll_count = 0
    start_time = time.time()
    status = None

    while time.time() < deadline:
        params = {"id": request_id} if request_id else None
        response = requests.get(polling_url, headers=headers, params=params, timeout=30)
        response.raise_for_status()
        payload = response.json()

        status = payload.get("status")
        poll_count += 1
        elapsed = time.time() - start_time

        # Log status every 10 polls or every 10 seconds
        if poll_count % 10 == 0 or elapsed > 10:
            print(f"Polling status: {status} (attempt {poll_count}, {elapsed:.1f}s elapsed)")

        if status == "Ready":
            result = payload.get("result", {})
            sample = result.get("sample")

            if isinstance(sample, list):
                return sample[0] if sample else None

            return sample

        if status in {"Error", "Failed"}:
            error_detail = payload.get("error") or payload
            raise RuntimeError(f"Generation failed: {error_detail}")
        
        # Handle content moderation - this is a terminal state
        if status == "Content Moderated":
            error_detail = payload.get("error") or payload.get("message") or "Content moderation flagged this prompt"
            raise RuntimeError(f"Content moderation blocked this prompt. The API flagged your request as potentially violating content policies. Please try rephrasing your prompt or removing any potentially problematic terms. Details: {error_detail}")
        
        # Handle other statuses that might indicate the API is still processing
        if status in {"Processing", "Pending", "Queued", "InProgress"}:
            # Continue polling
            pass
        elif status:
            # Unknown status - log it but continue polling (but limit how long we poll for unknown statuses)
            if poll_count > 20:  # After 20 polls of unknown status, treat as error
                raise RuntimeError(f"Received unknown status '{status}' repeatedly. The API may be stuck. Please try again.")
            if poll_count % 5 == 0:  # Only log every 5th poll to reduce spam
                print(f"Unknown status '{status}' received (attempt {poll_count}), continuing to poll...")

        time.sleep(BFL_POLL_INTERVAL)

    # Timeout reached
    elapsed_total = time.time() - start_time
    print(f"Polling timeout after {elapsed_total:.1f}s ({poll_count} attempts). Last status: {status}")
    return None


def download_image(image_url: str, prompt: str, enhanced_prompt: Optional[str] = None, image_type: Optional[str] = None) -> str:
    """
    Download image and save it with a unique filename.
    Also saves a metadata file with the prompt for reference.
    """
    print(f"DEBUG: Downloading image from: {image_url}")
    
    response = requests.get(image_url, stream=True, timeout=30)
    response.raise_for_status()

    content_type = response.headers.get("Content-Type", "image/png")
    extension = determine_extension(content_type)
    
    print(f"DEBUG: Content-Type: {content_type}, Extension: {extension}")

    # Generate unique filename
    unique_id = uuid.uuid4().hex[:8]  # Short unique ID to avoid collisions
    filename = f"{unique_id}{extension}"
    output_dir = os.path.join(app.root_path, "static", "generated_images")
    os.makedirs(output_dir, exist_ok=True)

    file_path = os.path.join(output_dir, filename)
    
    print(f"DEBUG: Saving image to: {file_path}")

    # Save the image
    bytes_written = 0
    with open(file_path, "wb") as file_handle:
        for chunk in response.iter_content(chunk_size=8192):
            file_handle.write(chunk)
            bytes_written += len(chunk)
    
    print(f"DEBUG: Image saved successfully! Size: {bytes_written/1024:.1f} KB")
    
    # Verify file exists
    if os.path.exists(file_path):
        file_size = os.path.getsize(file_path)
        print(f"DEBUG: Verified file exists, size: {file_size/1024:.1f} KB")
    else:
        print(f"ERROR: File was not saved properly!")

    # Save metadata file with prompt for reference
    metadata_filename = f"{unique_id}.txt"
    metadata_path = os.path.join(output_dir, metadata_filename)
    with open(metadata_path, "w", encoding="utf-8") as meta_file:
        meta_file.write(f"Prompt: {prompt}\n")
        if enhanced_prompt:
            meta_file.write(f"Enhanced Prompt: {enhanced_prompt}\n")
        if image_type:
            meta_file.write(f"Image Type: {image_type}\n")

    relative_path = os.path.join("generated_images", filename)
    print(f"DEBUG: Returning relative path: {relative_path}")
    return relative_path.replace(os.sep, "/")


def save_uploaded_image(
    file_storage: FileStorage,
    prompt: str = "",
    image_type: Optional[str] = None,
) -> str:
    """
    Persist an uploaded image into the generated_images directory so it can be
    segmented and exported just like AI-generated assets.
    """
    if not file_storage or not file_storage.filename:
        raise ValueError("Please choose an image to upload.")

    original_name = file_storage.filename
    extension = os.path.splitext(original_name)[1].lower().lstrip(".")
    if extension not in ALLOWED_UPLOAD_EXTENSIONS:
        raise ValueError("Unsupported file type. Please upload a PNG, JPG, or WEBP image.")

    unique_id = uuid.uuid4().hex[:8]
    output_dir = os.path.join(app.root_path, "static", "generated_images")
    os.makedirs(output_dir, exist_ok=True)

    filename = f"{unique_id}.png"
    file_path = os.path.join(output_dir, filename)

    try:
        file_storage.stream.seek(0)
        image = Image.open(file_storage.stream)
        image = ImageOps.exif_transpose(image)
        image = image.convert("RGBA")
        image.save(file_path, "PNG")
    except Exception as exc:
        raise ValueError("Uploaded file is not a valid image.") from exc
    finally:
        file_storage.stream.seek(0)

    metadata_path = os.path.join(output_dir, f"{unique_id}.txt")
    with open(metadata_path, "w", encoding="utf-8") as meta_file:
        if prompt:
            meta_file.write(f"Prompt: {prompt}\n")
        else:
            meta_file.write("Prompt: User uploaded image\n")
        meta_file.write(f"Source: User Upload ({original_name})\n")
        if image_type:
            meta_file.write(f"Image Type: {image_type}\n")

    relative_path = os.path.join("generated_images", filename)
    return relative_path.replace(os.sep, "/")


def determine_extension(content_type: str) -> str:
    if "jpeg" in content_type or "jpg" in content_type:
        return ".jpg"
    if "webp" in content_type:
        return ".webp"
    if "gif" in content_type:
        return ".gif"
    if "png" in content_type:
        return ".png"
    return ".png"


def read_image_metadata(image_path: str) -> dict:
    """
    Read metadata from the .txt file associated with an image.
    Returns a dictionary with 'prompt', 'enhanced_prompt', and 'image_type' if available.
    """
    metadata = {}
    
    try:
        # Extract the unique ID from the image path
        # Path format: generated_images/abc12345.png
        filename = os.path.basename(image_path)
        unique_id = os.path.splitext(filename)[0]
        
        # Construct metadata file path
        output_dir = os.path.join(app.root_path, "static", "generated_images")
        metadata_path = os.path.join(output_dir, f"{unique_id}.txt")
        
        if os.path.exists(metadata_path):
            with open(metadata_path, "r", encoding="utf-8") as meta_file:
                content = meta_file.read()
                
                # Parse the metadata file
                # Handle multi-line content by looking for section headers
                lines = content.split("\n")
                current_section = None
                current_content = []
                
                for line in lines:
                    if line.startswith("Prompt:"):
                        # Save previous section if any
                        if current_section and current_content:
                            metadata[current_section] = "\n".join(current_content).strip()
                        current_section = "prompt"
                        current_content = [line.replace("Prompt:", "").strip()]
                    elif line.startswith("Enhanced Prompt:"):
                        # Save previous section
                        if current_section and current_content:
                            metadata[current_section] = "\n".join(current_content).strip()
                        current_section = "enhanced_prompt"
                        current_content = [line.replace("Enhanced Prompt:", "").strip()]
                    elif line.startswith("Image Type:"):
                        # Save previous section
                        if current_section and current_content:
                            metadata[current_section] = "\n".join(current_content).strip()
                        current_section = "image_type"
                        # Image type is always a single line
                        metadata["image_type"] = line.replace("Image Type:", "").strip()
                        current_section = None
                        current_content = []
                    elif current_section:
                        # Continue collecting content for current section
                        if line.strip():  # Only add non-empty lines
                            current_content.append(line)
                
                # Save last section if any
                if current_section and current_content:
                    metadata[current_section] = "\n".join(current_content).strip()
    except Exception as e:
        print(f"Error reading metadata: {e}")
    
    return metadata


def require_api_key() -> str:
    if not BFL_API_KEY:
        raise RuntimeError("BFL_API_KEY environment variable is not set.")
    return BFL_API_KEY


def is_human_like_character(description: str) -> bool:
    """
    Determine if the character is human-like (requires arms separated from body) or a simple animal.
    Human-like includes: humans, humanoid robots, anthropomorphic animals, etc.
    Simple animals: regular cats, dogs, elephants, kangaroos, etc.
    """
    description_lower = description.lower()
    
    # Human-like keywords
    human_keywords = [
        'human', 'man', 'woman', 'person', 'people', 'boy', 'girl', 'kid', 'child', 
        'adult', 'teen', 'teenager', 'robot', 'android', 'humanoid', 'anthropomorphic',
        'human-like', 'human like', 'bipedal', 'standing on two legs', 'walks upright',
        'wears clothes', 'wearing', 'clothing', 'shirt', 'pants', 'jacket'
    ]
    
    # Simple animal keywords (four-legged, natural animals)
    simple_animal_keywords = [
        'cat', 'dog', 'elephant', 'kangaroo', 'lion', 'tiger', 'bear', 'wolf', 'fox',
        'rabbit', 'mouse', 'rat', 'horse', 'cow', 'pig', 'sheep', 'goat', 'deer',
        'monkey', 'ape', 'gorilla', 'chimpanzee', 'bird', 'eagle', 'hawk', 'owl',
        'snake', 'lizard', 'turtle', 'frog', 'fish', 'shark', 'whale', 'dolphin'
    ]
    
    # Check for human-like indicators first
    for keyword in human_keywords:
        if keyword in description_lower:
            return True
    
    # Check for simple animal keywords
    for keyword in simple_animal_keywords:
        if keyword in description_lower:
            # But exclude if it's anthropomorphic or human-like
            if 'anthropomorphic' in description_lower or 'human-like' in description_lower or 'human like' in description_lower:
                return True
            # Check if it's described as standing/walking like a human
            if any(phrase in description_lower for phrase in ['standing on two', 'walks on two', 'bipedal', 'upright']):
                return True
            return False
    
    # Default to human-like if unclear (safer for animation)
    return True


def classify_image_type(enhanced_prompt: str) -> str:
    """
    Classify the image type based on the enhanced prompt.
    Returns a classification like "Humanlike Character", "Animal", etc.
    This classification is NOT included in the image generation prompt.
    """
    if not enhanced_prompt:
        return "Unknown"
    
    prompt_lower = enhanced_prompt.lower()
    
    # Check for humanlike character indicators
    humanlike_keywords = [
        'human', 'man', 'woman', 'person', 'people', 'boy', 'girl', 'kid', 'child',
        'adult', 'teen', 'teenager', 'humanoid', 'anthropomorphic', 'human-like',
        'human like', 'bipedal', 'standing on two legs', 'walks upright',
        'wears clothes', 'wearing', 'clothing', 'shirt', 'pants', 'jacket',
        'robot', 'android', 'cyborg', 'mech', 'mechanical human'
    ]
    
    # Check for anthropomorphic animal indicators
    anthropomorphic_keywords = [
        'anthropomorphic', 'anthro', 'humanoid animal', 'animal with human',
        'standing upright', 'wearing clothes', 'animal person', 'furry character'
    ]
    
    # Check for robot/humanlike robot indicators
    robot_keywords = [
        'robot', 'android', 'cyborg', 'mechanical', 'mech', 'automaton',
        'humanoid robot', 'robot human', 'artificial human'
    ]
    
    # Check for regular animal indicators (non-anthropomorphic)
    animal_keywords = [
        'cat', 'dog', 'elephant', 'kangaroo', 'lion', 'tiger', 'bear', 'wolf', 'fox',
        'rabbit', 'mouse', 'rat', 'horse', 'cow', 'pig', 'sheep', 'goat', 'deer',
        'monkey', 'ape', 'gorilla', 'chimpanzee', 'bird', 'eagle', 'hawk', 'owl',
        'snake', 'lizard', 'turtle', 'frog', 'fish', 'shark', 'whale', 'dolphin',
        'dragon', 'dinosaur', 'creature', 'beast', 'wildlife'
    ]
    
    # Check for robot/humanlike robot first
    for keyword in robot_keywords:
        if keyword in prompt_lower:
            return "Humanlike Character (Robot/Humanoid Robot)"
    
    # Check for anthropomorphic animals
    for keyword in anthropomorphic_keywords:
        if keyword in prompt_lower:
            return "Humanlike Character (Anthropomorphic Animal)"
    
    # Check for humanlike characters
    for keyword in humanlike_keywords:
        if keyword in prompt_lower:
            # Check if it's anthropomorphic
            if any(anthro_kw in prompt_lower for anthro_kw in anthropomorphic_keywords):
                return "Humanlike Character (Anthropomorphic Animal)"
            return "Humanlike Character"
    
    # Check for regular animals
    for keyword in animal_keywords:
        if keyword in prompt_lower:
            # Exclude if it's anthropomorphic
            if any(anthro_kw in prompt_lower for anthro_kw in anthropomorphic_keywords):
                return "Humanlike Character (Anthropomorphic Animal)"
            # Exclude if described as human-like
            if any(phrase in prompt_lower for phrase in ['standing on two', 'walks on two', 'bipedal', 'upright', 'wearing', 'clothes']):
                return "Humanlike Character (Anthropomorphic Animal)"
            return "Animal"
    
    # Default classification
    return "Other"


def generate_enhanced_prompt(input_description: str) -> str:
    """
    Enhance a simple character description using OpenAI API.
    Expands the description with details suitable for animation-ready character generation
    with full body structure, appendages, and clear segmentation.
    """
    if not openai_client:
        # If OpenAI is not configured, return empty string to use original prompt
        return ""
    
    # Determine if character is human-like (needs arms separated from body) or simple animal
    is_human_like = is_human_like_character(input_description)
    
    try:
        # Build system prompt based on character type
        if is_human_like:
            arms_requirement = """4. ARMS - CRITICAL SEGMENTATION REQUIREMENT (FOR HUMAN-LIKE CHARACTERS):
   - UPPER ARMS MUST BE AWAY FROM THE BODY - this is ABSOLUTELY CRITICAL
   - The upper arms (from shoulder to elbow) MUST have clear, visible space between them and the torso
   - Upper arms MUST NOT touch the body, MUST NOT be close to the body, and MUST NOT press against the sides
   - There must be a noticeable gap between the upper arms and the torso - the upper arms should be positioned outward from the body
   - Arms should be in a natural pose (NOT in T-pose, NOT extended straight out to sides), but the upper arms MUST be angled away from the body
   - Arms can be slightly bent at the elbows, but the upper arm portion MUST maintain clear separation from the torso
   - The biceps and triceps area of the upper arms MUST be clearly separated from the body with visible space
   - NEVER describe upper arms touching the body, upper arms close to the body, or upper arms pressed against the sides
   - The entire upper arm (shoulder to elbow) must be positioned away from the body for proper segmentation
   - The space between upper arms and body is essential for animation segmentation
5. HANDS: Hands MUST be positioned away from the body, not touching the torso, hips, or any other body part. Hands should be open with fingers visible, creating clear space around them. NEVER describe hands on hips, hands in pockets, or hands touching the body.
6. LEGS: Legs MUST be straight and positioned apart with clear space between them. There must be visible separation between the legs - they should NOT be touching or pressed together. The space between legs should be clearly visible for animation segmentation.
7. FEET: Feet MUST be visible, separated, and positioned for walking. There must be clear space between the feet - they should NOT be touching or close together.
8. HAIR (FOR FEMALE/WOMAN CHARACTERS): If the character is a woman or female, the hair MUST be short. Hair should NOT extend below the shoulders and MUST NOT overlap with the body, shoulders, or back. Short hair is essential to prevent hair from covering body parts needed for segmentation."""
            critical_note = "CRITICAL: In your description, you MUST explicitly mention that the UPPER ARMS are positioned AWAY from the body with clear, visible space between the upper arms and torso. The upper arms (shoulder to elbow) MUST NOT touch the body and must be angled outward. For female characters, describe short hair that does not overlap with the body."
        else:
            arms_requirement = """4. POSE - NATURAL ANIMAL POSE:
   - The character should be in a natural standing pose appropriate for the animal type
   - For four-legged animals: describe all four legs clearly visible and separated, with the animal standing naturally
   - For bipedal animals: describe natural standing pose with limbs clearly separated from the body
   - All limbs must be clearly visible and separated for animation segmentation
   - The pose should be natural and appropriate for the animal species
5. LIMBS: All limbs (legs, arms if applicable, wings, etc.) MUST be clearly visible and separated from the body. There must be visible separation between all body parts for animation segmentation.
6. FEET/PAWS: All feet or paws MUST be visible, separated, and positioned naturally. There must be clear space between them."""
            critical_note = "CRITICAL: Describe the character in a natural pose appropriate for the animal type, with all body parts clearly visible and separated for animation."
        
        system_prompt = f"""You are a creative assistant that enhances character descriptions for animation-ready character generation. 
Your task is to expand simple character descriptions into detailed, visual descriptions optimized for animation.

IMPORTANT: Write ONLY the character description. Do NOT include technical instructions, formatting, or animation requirements in your response. Just describe the character naturally while following the requirements below.

MANDATORY REQUIREMENTS - THESE MUST ALWAYS BE INCLUDED IN YOUR DESCRIPTION:
1. SINGLE IMAGE ONLY: Describe ONE character in ONE image. NEVER describe multiple views, reference sheets, or multiple poses in one image. Only a single full-body character standing in one pose.
2. FULL BODY IN FRAME: ALWAYS describe the COMPLETE character from head to feet/paws/tail, fully inside the image with a small margin around the body. NEVER create half-body, bust, portrait, close-up, or cropped images. The \"camera\" must be far enough back so the entire character including shoes/feet is visible at once with nothing cut off at the knees, thighs, waist, or ankles.
3. VIEWING ANGLE: The character MUST be at a 3/4 angle (turned approximately 45 degrees to the side), NOT facing straight front. This 3/4 pose is essential for walking animation - the character should be positioned so they can walk forward in the frame. NEVER describe a front-facing camera view or multiple angles.
{arms_requirement}
8. APPENDAGES: If the character has tails, wings, horns, antennae, or other appendages, describe them clearly and ensure they are separate from the main body
9. APPEARANCE DETAILS: Include specific details about fur/feathers/scales color and texture, eye color, facial features, and any distinctive markings
10. EXPRESSION: Specify a neutral facial expression suitable for animation

{critical_note}"""

        if is_human_like:
            user_prompt = (
                "Expand the following character description into a detailed, animation-ready character description. "
                "Write ONLY the character description - do not include technical instructions. "
                "CRITICAL FRAMING RULES: The character must be shown as a SINGLE full-body figure from head to shoes fully inside the frame, "
                "with a little empty space above the head and below the feet so nothing is cropped. Do NOT describe close-ups, portraits, "
                "waist-up, or half-body shots. The character is standing at a 3/4 angle for walking animation (NOT front-facing). "
                "UPPER ARMS REQUIREMENT - ABSOLUTELY CRITICAL: The UPPER ARMS (from shoulder to elbow) MUST be positioned AWAY from the body "
                "with clear, visible space between the upper arms and torso. Upper arms MUST NOT touch the body, MUST be angled outward, and there must be a noticeable gap. "
                "Arms can be in a natural pose (NOT T-pose, NOT extended straight out), but the upper arm portion MUST maintain clear separation from the torso. "
                "Hands positioned away from body, legs straight and apart, feet separated. If the character is a woman or female, describe SHORT HAIR that does not extend below "
                "shoulders and does not overlap with the body. Describe the character naturally while strictly following these framing and pose rules: "
                f"{input_description}"
            )
        else:
            user_prompt = (
                "Expand the following character description into a detailed, animation-ready character description. "
                "Write ONLY the character description - do not include technical instructions. "
                "CRITICAL FRAMING RULES: The character must be shown as a SINGLE full-body figure from head to feet/paws fully inside the frame, "
                "with a little empty space above the head and below the feet so nothing is cropped. Do NOT describe close-ups, portraits, "
                "waist-up, or half-body shots. The character is in a natural standing pose at 3/4 angle (NOT front-facing). "
                "All body parts must be clearly visible and separated for animation. "
                "Describe the character in a natural pose appropriate for this animal type: "
                f"{input_description}"
            )

        response = openai_client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=250,
            temperature=0.7,
        )
        
        enhanced_prompt = response.choices[0].message.content.strip()
        
        # Clean up the enhanced prompt - remove any technical instructions that might have been inserted
        # Split by common instruction markers and take only the character description part
        if "SINGLE IMAGE ONLY" in enhanced_prompt or "MANDATORY BODY PART" in enhanced_prompt:
            # If instructions were inserted, try to extract just the description
            parts = enhanced_prompt.split("SINGLE IMAGE ONLY")
            if len(parts) > 1:
                enhanced_prompt = parts[0].strip()
            else:
                parts = enhanced_prompt.split("MANDATORY BODY PART")
                if len(parts) > 1:
                    enhanced_prompt = parts[0].strip()
        
        # Add mandatory animation-specific instructions to the END of the enhanced prompt
        if is_human_like:
            animation_instructions = (
                " SINGLE IMAGE ONLY - one character, one pose, no multiple views or reference sheets. "
                "FULL BODY VIEW - complete character from head to feet visible in a single frame, NOT half-body, portrait, or waist-up. "
                "The camera must be pulled back so the entire character including both shoes/feet and a bit of space below the feet are inside the image at once; "
                "NEVER crop the character at the ankles, shins, knees, thighs, waist, or shoulders. "
                "Character standing at 3/4 angle pose (turned 45 degrees to the side for walking animation), NOT facing straight front camera. "
                "UPPER ARMS REQUIREMENT - ABSOLUTELY CRITICAL: The UPPER ARMS (from shoulder to elbow) MUST be positioned AWAY from the body with clear, visible space between the upper arms and torso. "
                "Upper arms MUST NOT touch the body, MUST NOT be close to the body, MUST be angled outward from the body, and there must be a noticeable gap. "
                "Arms can be in a natural pose (NOT T-pose, NOT extended straight out to sides), but the upper arm portion MUST maintain clear separation from the torso. "
                "The biceps and triceps area of the upper arms MUST be clearly separated from the body. "
                "Hands positioned away from body with clear space (hands NOT touching torso, NOT on hips, NOT in pockets, NOT crossed). "
                "Legs straight and positioned apart with visible space between them (legs NOT touching, NOT pressed together, clear separation visible). "
                "Feet separated with clear space between them (feet NOT touching, NOT close together). "
                "SHORT HAIR for women/female characters - hair must be short, NOT extending below shoulders, and MUST NOT overlap with body, shoulders, or back. "
                "All body parts must be clearly separated and visible with no overlapping parts. Visible joints for animation rigging. "
                "If the character has appendages like tails, wings, or horns, ensure they are clearly visible and separated from the main body structure."
            )
        else:
            animation_instructions = (
                " SINGLE IMAGE ONLY - one character, one pose, no multiple views or reference sheets. "
                "FULL BODY VIEW - complete character from head to tail/paws visible in a single frame, NOT half-body, portrait, or waist-up. "
                "The camera must be pulled back so the entire character including all feet/paws and a bit of space below them are inside the image at once; "
                "NEVER crop the character at the ankles, shins, knees, thighs, waist, or shoulders. "
                "Character in natural standing pose at 3/4 angle (turned 45 degrees to the side for walking animation), NOT facing straight front camera. "
                "Natural pose appropriate for the animal type with all limbs clearly visible and separated from the body. "
                "All body parts must be clearly separated and visible with no overlapping parts. "
                "If the character has appendages like tails, wings, or horns, ensure they are clearly visible and separated from the main body structure."
            )
        
        return enhanced_prompt + animation_instructions
    
    except Exception as e:
        # If enhancement fails, log error but don't break the flow
        print(f"Prompt enhancement failed: {e}")
        return ""


@app.route("/api/create-spine", methods=["POST"])
def create_spine():
    """
    Create Spine skeleton and animations from segmented image.
    NEW PIPELINE: Uses assembly_preview.png as ground truth.
    """
    if not create_simple_skeleton or not create_atlas_from_masks:
        return jsonify({"error": "Spine modules not available"}), 500
    
    data = request.get_json()
    image_id = data.get("image_id")
    
    if not image_id:
        return jsonify({"error": "image_id required"}), 400
    
    try:
        static_root = Path(app.root_path) / "static"
        generated_dir = static_root / "generated_images"
        
        # Find the segmented masks directory
        masks_dir = generated_dir / "segmented_masks" / f"{image_id}_no_bg"
        
        if not masks_dir.exists():
            return jsonify({"error": f"Segmentation not found. Please run segmentation first for image: {image_id}"}), 404
        
        # STEP 1: Load assembly_preview.png (the correctly assembled character)
        assembly_preview_path = masks_dir / "assembly_preview.png"
        if not assembly_preview_path.exists():
            return jsonify({"error": f"Assembly preview not found. Please run segmentation first for image: {image_id}"}), 404
        
        print(f"Loading assembly preview from: {assembly_preview_path}")
        assembly_img = Image.open(assembly_preview_path)
        assembly_width, assembly_height = assembly_img.size
        print(f"Assembly preview dimensions: {assembly_width}x{assembly_height}")
        
        # STEP 2: Run MediaPipe pose detection on assembly preview
        landmarks = None
        segmentation_metadata: Optional[Dict[str, Any]] = None
        metadata_path = masks_dir / "metadata.json"
        
        if metadata_path.exists():
            try:
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    segmentation_metadata = json.load(f)
            except Exception as e:
                print(f"Could not load metadata: {e}")
        
        if not landmarks and BodyPartDetector:
            try:
                import cv2
                import numpy as np
                # Load assembly preview as OpenCV image
                assembly_cv = cv2.imread(str(assembly_preview_path), cv2.IMREAD_UNCHANGED)
                if assembly_cv is not None:
                    # Ensure it has alpha channel
                    if assembly_cv.shape[2] == 3:
                        alpha = np.ones((assembly_cv.shape[0], assembly_cv.shape[1]), dtype=np.uint8) * 255
                        assembly_cv = cv2.cvtColor(assembly_cv, cv2.COLOR_BGR2BGRA)
                        assembly_cv[:, :, 3] = alpha
                    
                    # Detect pose landmarks on the assembled character
                    print("Running MediaPipe pose detection on assembly preview...")
                    detector = BodyPartDetector(assembly_cv)
                    detected_parts = detector.detect_all_parts()
                    
                    if detected_parts and extract_landmarks_from_detector:
                        landmarks = extract_landmarks_from_detector(detector)
                        if landmarks:
                            print(f"Detected {len(landmarks)} pose landmarks from assembly preview")
            except Exception as e:
                print(f"Pose detection on assembly preview failed: {e}")
                import traceback
                traceback.print_exc()
        
        if not landmarks:
            return jsonify({"error": "Could not detect pose landmarks on assembly preview. Please ensure the character is clearly visible."}), 400
        
        # STEP 3: Create atlas from segmented parts (needed for part metadata)
        spine_dir = generated_dir / "spine" / image_id
        spine_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Creating atlas from segmented parts in: {masks_dir}")
        
        # Check for part images (exclude metadata.json, visualization, and outline files)
        exclude_files = {"metadata.json", "detection_visualization.png", "no_background.png", 
                        "outline_only.png", "outline_overlay.png", "assembly_preview.png"}
        mask_files = [f for f in masks_dir.glob("*.png") if f.name not in exclude_files]
        print(f"Found {len(mask_files)} part files: {[f.name for f in mask_files]}")
        
        if not mask_files:
            return jsonify({"error": f"No segmented part images found in {masks_dir}. Please run segmentation first."}), 404
        
        atlas_path, atlas_image_path, part_metadata = create_atlas_from_masks(
            masks_dir, spine_dir, image_id
        )
        
        if not atlas_path or not atlas_image_path:
            return jsonify({"error": f"Failed to create atlas. Found {len(mask_files)} mask files in {masks_dir}"}), 500
        
        # STEP 4: Create complete Spine skeleton with attachments positioned from assembly preview
        spine_json_path = spine_dir / f"{image_id}.json"
        print(f"Creating Spine skeleton from {len(landmarks)} landmarks with {len(part_metadata)} parts...")
        
        spine_json = create_simple_skeleton(
            landmarks=landmarks,
            assembly_width=assembly_width,
            assembly_height=assembly_height,
            part_metadata=part_metadata,
                segmentation_metadata=segmentation_metadata,
            character_name=image_id
        )
        
        # STEP 5: Add animations
        if add_default_animations_to_skeleton:
            spine_json = add_default_animations_to_skeleton(spine_json)
        
        # Save skeleton
            with open(spine_json_path, "w", encoding="utf-8") as f:
                json.dump(spine_json, f, indent=2)
        
        print(f"Spine skeleton saved to: {spine_json_path}")
        
        # Return paths relative to static
        rel_json = str(spine_json_path.relative_to(static_root)).replace(os.sep, "/")
        rel_atlas = str(atlas_path.relative_to(static_root)).replace(os.sep, "/")
        
        return jsonify({
            "success": True,
            "spine_json": f"/static/{rel_json}",
            "spine_atlas": f"/static/{rel_atlas}",
            "image_id": image_id
        })
    
    except Exception as e:
        print(f"Error creating Spine skeleton: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/api/animations", methods=["GET"])
def list_animations():
    """List all available animations."""
    if not get_animation_builder:
        return jsonify({"error": "Animation builder not available"}), 500
    
    try:
        builder = get_animation_builder()
        animations = builder.list_animations()
        return jsonify({"animations": animations})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/export-spine/<image_id>", methods=["GET"])
def export_spine_project(image_id):
    """Export complete Spine project as ZIP file."""
    try:
        static_root = Path(app.root_path) / "static"
        spine_dir = static_root / "generated_images" / "spine" / image_id
        
        if not spine_dir.exists():
            flash("Spine project not found. Please create skeleton first.", "error")
            return render_template("index.html")
        
        # Create ZIP file
        zip_path = spine_dir / f"{image_id}_spine.zip"
        
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
            # Add all files in spine directory
            for file_path in spine_dir.glob("*"):
                if file_path.is_file() and file_path != zip_path:
                    zipf.write(file_path, file_path.name)
        
        # Send file
        return send_file(
            zip_path,
            as_attachment=True,
            download_name=f"{image_id}_spine.zip",
            mimetype="application/zip"
        )
    
    except Exception as e:
        print(f"Error exporting Spine project: {e}")
        flash(f"Export failed: {e}", "error")
        return render_template("index.html")


@app.route("/chat", methods=["GET"])
def chat_interface():
    """Render the chat interface."""
    return render_template("chat.html")


@app.route("/chat/message", methods=["POST"])
def chat_message():
    """Handle chat messages from the user."""
    if not chat_handler:
        return jsonify({
            "response": "Chat system not available. Please configure OpenAI API key.",
            "action": "error"
        }), 500
    
    try:
        data = request.get_json()
        user_message = data.get("message", "").strip()
        
        if not user_message:
            return jsonify({
                "response": "Please enter a message.",
                "action": "error"
            }), 400
        
        # Process the message
        result = chat_handler.process_user_input(user_message)
        
        return jsonify(result)
    
    except Exception as e:
        print(f"Error processing chat message: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            "response": f"Error processing message: {str(e)}",
            "action": "error"
        }), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5006"))
    app.run(host="0.0.0.0", port=port, debug=True)
