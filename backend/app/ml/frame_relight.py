"""Nano Banana relight: send a frame + relight prompts to Gemini's image
preview model and write the relit frame to disk.

Mirrors the call shape in ``backend/app/thumbnails/images.py:generate_thumbnail``.
Used by AI Restyle to produce the style-reference frame for downstream
video-to-video.
"""
from __future__ import annotations

import os
from typing import List

from google import genai
from google.genai import types

MODEL_NAME = "gemini-3.1-flash-image-preview"

SAFETY_CONSTRAINTS: List[str] = [
    "Keep the person, pose, clothing, and composition EXACTLY as in the source.",
    "Do not add or remove any people or objects.",
    "Do not change facial features or body proportions.",
    "Preserve the framing and camera angle.",
]


def build_relight_prompt(background_prompt: str, lighting_prompt: str) -> str:
    """Compose the Nano Banana prompt from user-controlled fragments + safety."""
    safety_block = "\n".join(f"- {c}" for c in SAFETY_CONSTRAINTS)
    return (
        "Relight this image with the following style. Only change the "
        "background and lighting.\n\n"
        f"Background: {background_prompt}\n"
        f"Lighting: {lighting_prompt}\n\n"
        "Constraints:\n"
        f"{safety_block}"
    )


def relight_frame(
    api_key: str,
    frame_path: str,
    background_prompt: str,
    lighting_prompt: str,
    out_path: str,
) -> str:
    """Call Nano Banana with the input frame + prompts. Writes the relit
    PNG to ``out_path``.

    Returns ``out_path`` on success. Raises ``FileNotFoundError`` if the
    input is missing, ``RuntimeError`` if the response carries no image
    data (typically a content-policy refusal).
    """
    if not os.path.exists(frame_path):
        raise FileNotFoundError(f"Input frame not found: {frame_path}")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    with open(frame_path, "rb") as f:
        image_bytes = f.read()

    client = genai.Client(api_key=api_key)
    prompt = build_relight_prompt(background_prompt, lighting_prompt)

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=[
            types.Part.from_bytes(data=image_bytes, mime_type="image/png"),
            prompt,
        ],
    )

    for part in response.parts or []:
        inline = getattr(part, "inline_data", None)
        if inline and getattr(inline, "data", None):
            with open(out_path, "wb") as f:
                f.write(inline.data)
            return out_path

    raise RuntimeError("Nano Banana returned no image (likely content policy)")
