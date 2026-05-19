"""Thumbnail image generation via Gemini multimodal image preview model."""

import os
from google import genai
from google.genai import types
from PIL import Image


def generate_thumbnail(api_key, title, session_id, face_image_path=None, bg_image_path=None, extra_prompt="", count=3, video_context=""):
    """
    Generates YouTube thumbnails using Gemini image generation.
    Returns list of saved image paths (relative URLs).
    """
    client = genai.Client(api_key=api_key)

    output_dir = os.path.join("output", "thumbnails", session_id)
    os.makedirs(output_dir, exist_ok=True)

    prompt_parts = []

    # Add face image if provided
    if face_image_path and os.path.exists(face_image_path):
        face_img = Image.open(face_image_path)
        prompt_parts.append(face_img)

    # Add background image if provided
    if bg_image_path and os.path.exists(bg_image_path):
        bg_img = Image.open(bg_image_path)
        prompt_parts.append(bg_img)

    # Build video context block
    context_block = ""
    if video_context:
        context_block = f"""
VIDEO CONTEXT (use this to understand the video and design a relevant thumbnail):
{video_context}
"""

    # Build extra instructions block (high priority)
    extra_block = ""
    if extra_prompt:
        extra_block = f"""
⚠️ MANDATORY USER INSTRUCTIONS (MUST follow these exactly — they override any default behavior):
{extra_prompt}
"""

    text_prompt = f"""Generate a professional, eye-catching YouTube thumbnail image.

VIDEO TITLE (for reference — do NOT put the full title on the thumbnail): "{title}"
{context_block}
TEXT ON THE THUMBNAIL:
- Based on the title AND the video context, create a SHORT visual hook: 1 to 5 words maximum
- It should capture the core emotion, surprise, or promise of the video
- The thumbnail text should COMPLEMENT the YouTube title (which appears below), not repeat it
- Examples: "$10K EN 30 DÍAS", "ESTO FUNCIONA", "NO LO SABÍAS", "GRATIS 🔥"
- Use ALL CAPS for maximum impact, split into 2-3 lines
{extra_block}
DESIGN REQUIREMENTS:
- The text MUST be large, bold, and high-contrast (readable at small sizes)
- Use vibrant, eye-catching colors that match the video's mood
- Professional YouTube thumbnail aesthetic
- Clean composition — text and face/subject as clear focal points
- NO clutter, NO small text, NO watermarks"""

    if face_image_path and os.path.exists(face_image_path):
        text_prompt += "\n- Include the provided face/person prominently with an exaggerated expression (surprise, excitement, shock)"

    if bg_image_path and os.path.exists(bg_image_path):
        text_prompt += "\n- Use the provided background image as the base/backdrop"

    prompt_parts.append(text_prompt)

    thumbnails = []
    last_error = None
    for i in range(count):
        print(f"🎨 [Thumbnail] Generating thumbnail {i + 1}/{count}...")
        try:
            response = client.models.generate_content(
                model="gemini-3.1-flash-image-preview",
                contents=prompt_parts,
                config=types.GenerateContentConfig(
                    response_modalities=["TEXT", "IMAGE"],
                    image_config=types.ImageConfig(
                        aspect_ratio="16:9",
                        image_size="2K"
                    )
                )
            )

            for part in response.parts:
                if part.text is not None:
                    print(f"📝 [Thumbnail] Gemini text: {part.text}")
                elif image := part.as_image():
                    filename = f"thumb_{i + 1}.jpg"
                    filepath = os.path.join(output_dir, filename)
                    image.save(filepath)
                    thumbnails.append(f"/thumbnails/{session_id}/{filename}")
                    print(f"✅ [Thumbnail] Saved: {filepath}")
                    break

        except Exception as e:
            last_error = str(e)
            print(f"❌ [Thumbnail] Generation {i + 1} failed: {e}")

    if not thumbnails and last_error:
        raise RuntimeError(f"All thumbnail generations failed. Last error: {last_error}")

    return thumbnails
