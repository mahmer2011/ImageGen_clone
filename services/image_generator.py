# services/image_generator.py
import os
import time
import uuid
import requests
from typing import Optional, Dict

# Configuration
BFL_API_KEY = os.environ.get("BFL_API_KEY")
BFL_ENDPOINT = os.environ.get("BFL_ENDPOINT", "https://api.bfl.ai/v1/flux-kontext-pro")
BFL_POLL_INTERVAL = float(os.environ.get("BFL_POLL_INTERVAL", "1.0"))
BFL_POLL_TIMEOUT = float(os.environ.get("BFL_POLL_TIMEOUT", "180"))

def require_api_key() -> str:
    if not BFL_API_KEY:
        raise RuntimeError("BFL_API_KEY environment variable is not set.")
    return BFL_API_KEY

def submit_generation(prompt: str) -> dict:
    payload = {
        "prompt": prompt,
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
    headers = {"accept": "application/json", "x-key": require_api_key()}
    
    while time.time() < deadline:
        params = {"id": request_id} if request_id else None
        response = requests.get(polling_url, headers=headers, params=params, timeout=30)
        response.raise_for_status()
        payload = response.json()
        status = payload.get("status")

        if status == "Ready":
            result = payload.get("result", {})
            sample = result.get("sample")
            return sample[0] if isinstance(sample, list) and sample else sample
        
        if status in {"Error", "Failed", "Content Moderated"}:
            raise RuntimeError(f"Generation failed: {status}")

        time.sleep(BFL_POLL_INTERVAL)
    return None

def download_image(image_url: str, output_dir: str, prompt: str) -> str:
    """Downloads image and returns the relative path."""
    response = requests.get(image_url, stream=True, timeout=30)
    response.raise_for_status()
    
    # Determine extension
    content_type = response.headers.get("Content-Type", "image/png")
    ext = ".jpg" if "jpeg" in content_type else ".png"
    if "webp" in content_type: ext = ".webp"
    
    unique_id = uuid.uuid4().hex[:8]
    filename = f"{unique_id}{ext}"
    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, filename)
    
    with open(file_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            
    # Save simple metadata
    with open(os.path.join(output_dir, f"{unique_id}.txt"), "w", encoding="utf-8") as f:
        f.write(f"Prompt: {prompt}\n")
        
    return filename # Return just filename (ID)