"""
ElevenLabs Video Translation/Dubbing Module

Uses ElevenLabs Dubbing API to translate video audio to different languages.
"""

import os
import time
import httpx
from typing import Optional

ELEVENLABS_API_BASE = "https://api.elevenlabs.io/v1"

# Supported target languages for dubbing
SUPPORTED_LANGUAGES = {
    "en": "English",
    "es": "Spanish",
    "fr": "French",
    "de": "German",
    "it": "Italian",
    "pt": "Portuguese",
    "pl": "Polish",
    "hi": "Hindi",
    "ja": "Japanese",
    "ko": "Korean",
    "zh": "Chinese",
    "ar": "Arabic",
    "ru": "Russian",
    "tr": "Turkish",
    "nl": "Dutch",
    "sv": "Swedish",
    "id": "Indonesian",
    "fil": "Filipino",
    "ms": "Malay",
    "vi": "Vietnamese",
    "th": "Thai",
    "uk": "Ukrainian",
    "el": "Greek",
    "cs": "Czech",
    "fi": "Finnish",
    "ro": "Romanian",
    "da": "Danish",
    "bg": "Bulgarian",
    "hr": "Croatian",
    "sk": "Slovak",
    "ta": "Tamil",
}


def create_dubbing_project(
    video_path: str,
    target_language: str,
    api_key: str,
    source_language: Optional[str] = None,
) -> dict:
    """
    Create a new dubbing project with ElevenLabs.

    Args:
        video_path: Path to the video file
        target_language: Target language code (e.g., 'es', 'fr', 'de')
        api_key: ElevenLabs API key
        source_language: Source language code (auto-detected if None)

    Returns:
        dict with dubbing_id and expected_duration_sec
    """
    url = f"{ELEVENLABS_API_BASE}/dubbing"

    headers = {
        "xi-api-key": api_key,
    }

    # Prepare form data
    data = {
        "target_lang": target_language,
        "mode": "automatic",
        "num_speakers": "0",
        "watermark": "false",
    }

    if source_language:
        data["source_lang"] = source_language

    # Open and send the video file
    with open(video_path, "rb") as video_file:
        files = {
            "file": (os.path.basename(video_path), video_file, "video/mp4")
        }

        print(f"[ElevenLabs] Creating dubbing project for {target_language}...")
        with httpx.Client(timeout=300.0) as client:
            response = client.post(url, headers=headers, data=data, files=files)

    if response.status_code not in [200, 201]:
        error_msg = response.text
        try:
            error_data = response.json()
            error_msg = error_data.get("detail", {}).get("message", response.text)
        except:
            pass
        raise Exception(f"ElevenLabs API error: {error_msg}")

    result = response.json()
    print(f"[ElevenLabs] Dubbing project created: {result.get('dubbing_id')}")
    return result


def get_dubbing_status(dubbing_id: str, api_key: str) -> dict:
    """
    Check the status of a dubbing project.

    Returns:
        dict with status ('dubbing', 'dubbed', 'failed') and other metadata
    """
    url = f"{ELEVENLABS_API_BASE}/dubbing/{dubbing_id}"

    headers = {
        "xi-api-key": api_key,
    }

    with httpx.Client(timeout=30.0) as client:
        response = client.get(url, headers=headers)

    if response.status_code != 200:
        raise Exception(f"Failed to get dubbing status: {response.text}")

    return response.json()


def download_dubbed_video(
    dubbing_id: str,
    target_language: str,
    output_path: str,
    api_key: str
) -> str:
    """
    Download the dubbed video file.

    Args:
        dubbing_id: The dubbing project ID
        target_language: Target language code
        output_path: Where to save the dubbed video
        api_key: ElevenLabs API key

    Returns:
        Path to the downloaded file
    """
    url = f"{ELEVENLABS_API_BASE}/dubbing/{dubbing_id}/audio/{target_language}"

    headers = {
        "xi-api-key": api_key,
    }

    print(f"[ElevenLabs] Downloading dubbed video...")
    with httpx.Client(timeout=120.0) as client:
        with client.stream("GET", url, headers=headers) as response:
            if response.status_code != 200:
                raise Exception(f"Failed to download dubbed video: {response.text}")

            with open(output_path, "wb") as f:
                for chunk in response.iter_bytes(chunk_size=8192):
                    f.write(chunk)

    print(f"[ElevenLabs] Dubbed video saved to: {output_path}")
    return output_path


def translate_video(
    video_path: str,
    output_path: str,
    target_language: str,
    api_key: str,
    source_language: Optional[str] = None,
    max_wait_seconds: int = 600,
    poll_interval: int = 5,
) -> str:
    """
    Translate a video to a target language using ElevenLabs dubbing.

    This is a blocking call that waits for the dubbing to complete.

    Args:
        video_path: Path to input video
        output_path: Path to save translated video
        target_language: Target language code
        api_key: ElevenLabs API key
        source_language: Source language code (auto-detected if None)
        max_wait_seconds: Maximum time to wait for dubbing (default 10 min)
        poll_interval: Seconds between status checks

    Returns:
        Path to the translated video
    """
    # Create dubbing project
    project = create_dubbing_project(
        video_path=video_path,
        target_language=target_language,
        api_key=api_key,
        source_language=source_language,
    )

    dubbing_id = project["dubbing_id"]
    expected_duration = project.get("expected_duration_sec", 60)

    print(f"[ElevenLabs] Dubbing ID: {dubbing_id}, Expected duration: {expected_duration}s")

    # Poll for completion
    start_time = time.time()
    while True:
        elapsed = time.time() - start_time
        if elapsed > max_wait_seconds:
            raise Exception(f"Dubbing timed out after {max_wait_seconds} seconds")

        status = get_dubbing_status(dubbing_id, api_key)
        current_status = status.get("status", "unknown")

        print(f"[ElevenLabs] Status: {current_status} (elapsed: {int(elapsed)}s)")

        if current_status == "dubbed":
            # Download the result
            return download_dubbed_video(
                dubbing_id=dubbing_id,
                target_language=target_language,
                output_path=output_path,
                api_key=api_key,
            )

        elif current_status == "failed":
            error = status.get("error", "Unknown error")
            raise Exception(f"Dubbing failed: {error}")

        # Still processing, wait and poll again
        time.sleep(poll_interval)


def get_supported_languages() -> dict:
    """Return dict of supported language codes and names."""
    return SUPPORTED_LANGUAGES.copy()
