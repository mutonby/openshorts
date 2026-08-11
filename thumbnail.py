import os
import uuid
import time
import json

import ai_gateway
from PIL import Image, ImageDraw, ImageFont

# Text/analysis model (title, description, tags). The gateway chain handles
# model choice; legacy Gemini SDK (GEMINI_API_KEY) is used for image
# generation only.
TEXT_MODEL = os.environ.get("GEMINI_MODEL_THUMBNAIL") or os.environ.get("GEMINI_MODEL") or "gemini-3.1-flash-lite"


_THUMB_SYSTEM = (
    "You are a YouTube title / SEO expert. Return strict JSON only — "
    "no markdown, no commentary."
)


def analyze_video_for_titles(api_key, video_path, transcript=None):
    """
    Transcribes a video and uses the AI to suggest viral YouTube titles.
    If transcript is provided, skips Whisper transcription.
    Returns: { "titles": [...], "transcript_summary": "...", "language": "...", "segments": [...], "video_duration": ... }
    """
    if transcript is None:
        from main import transcribe_video
        print("🎬 [Thumbnail] Transcribing video...")
        transcript = transcribe_video(video_path)
    else:
        print("🎬 [Thumbnail] Using pre-computed transcript (Whisper already done)...")

    segments = transcript.get("segments", [])
    video_duration = segments[-1]["end"] if segments else 0
    transcript_text = (
        transcript.get("text")
        or " ".join(str(s.get("text") or "") for s in segments)
    ).strip()

    prompt = f"""You are a YouTube title expert who creates viral, click-worthy titles.

Analyze this video and its transcript, then suggest 10 YouTube titles that would maximize CTR (click-through rate).

TRANSCRIPT:
{transcript_text}

RULES:
- Titles must be under 70 characters
- Use power words, curiosity gaps, and emotional triggers
- Mix styles: how-to, listicle, story-driven, controversial, question-based
- Make them specific to the actual content, not generic
- Include numbers where appropriate
- Consider the language of the video (detected: {transcript['language']})
- Titles should be in the SAME LANGUAGE as the video transcript

Also provide a brief summary of the video content (2-3 sentences).

After generating all 10 titles, pick the TOP 2 you most recommend and explain concisely WHY (CTR potential, emotional hook, uniqueness, etc.). Reference them by their 0-based index in the titles array.

OUTPUT JSON:
{{
    "titles": ["title1", "title2", ...],
    "transcript_summary": "Brief summary of the video content...",
    "language": "{transcript['language']}",
    "recommended": [
        {{"index": 0, "reason": "Why this title is best..."}},
        {{"index": 3, "reason": "Why this title is second best..."}}
    ]
}}"""

    if ai_gateway.is_configured():
        print("🤖 [Thumbnail] Asking free AI for title suggestions...")
        try:
            result, _usage = ai_gateway.complete_json(
                system=_THUMB_SYSTEM, user=prompt, temperature=0.7)
            result["transcript_summary"] = result.get("transcript_summary", "")
            result["language"] = result.get("language", transcript["language"])
            result["segments"] = segments
            result["video_duration"] = video_duration
            return result
        except ai_gateway.AIGatewayError as e:
            print(f"❌ [Thumbnail] Title analysis failed: {e}")
            return {
                "titles": ["Could not generate titles - please try again"],
                "transcript_summary": transcript_text[:500],
                "language": transcript["language"],
                "segments": segments,
                "video_duration": video_duration
            }

    # Legacy Gemini path (BYOK deployments with a real GEMINI_API_KEY).
    from google import genai
    from google.genai import types
    print("📤 [Thumbnail] Uploading video to Gemini...")
    client = genai.Client(api_key=api_key)

    file_upload = client.files.upload(file=video_path)
    while True:
        file_info = client.files.get(name=file_upload.name)
        if file_info.state == "ACTIVE":
            break
        elif file_info.state == "FAILED":
            raise Exception("Video processing failed by Gemini.")
        time.sleep(2)

    print("🤖 [Thumbnail] Asking Gemini for title suggestions...")
    response = client.models.generate_content(
        model=TEXT_MODEL,
        contents=[file_upload, prompt],
        config=types.GenerateContentConfig(
            response_mime_type="application/json"
        )
    )

    try:
        text = response.text.strip()
        if text.startswith("```json"):
            text = text[7:]
        if text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()

        start_idx = text.find('{')
        end_idx = text.rfind('}')
        if start_idx != -1 and end_idx != -1:
            text = text[start_idx:end_idx + 1]

        result = json.loads(text)
        result["transcript_summary"] = result.get("transcript_summary", "")
        result["language"] = result.get("language", transcript["language"])
        result["segments"] = segments
        result["video_duration"] = video_duration
        return result
    except json.JSONDecodeError:
        print(f"❌ [Thumbnail] Failed to parse titles JSON: {response.text}")
        return {
            "titles": ["Could not generate titles - please try again"],
            "transcript_summary": transcript_text[:500],
            "language": transcript["language"],
            "segments": segments,
            "video_duration": video_duration
        }


def refine_titles(api_key, context, user_message, conversation_history=None):
    """
    Takes video context + user feedback and returns refined title suggestions.
    """
    history_text = ""
    if conversation_history:
        for msg in conversation_history:
            role = msg.get("role", "user")
            history_text += f"\n{role.upper()}: {msg['content']}"

    prompt = f"""You are a YouTube title expert. Based on the video context and the user's feedback, suggest 8 new refined YouTube titles.

VIDEO CONTEXT:
{context}

CONVERSATION HISTORY:{history_text}

USER'S NEW REQUEST:
{user_message}

RULES:
- Titles must be under 70 characters
- Incorporate the user's feedback/direction
- Keep titles viral and click-worthy
- If the user asks for a specific style, follow it
- Titles should be in the same language as the original content

OUTPUT JSON:
{{
    "titles": ["title1", "title2", ...]
}}"""

    if ai_gateway.is_configured():
        try:
            result, _usage = ai_gateway.complete_json(
                system=_THUMB_SYSTEM, user=prompt, temperature=0.7)
            return result
        except ai_gateway.AIGatewayError as e:
            print(f"❌ [Thumbnail] Title refinement failed: {e}")
            return {"titles": ["Could not refine titles - please try again"]}

    from google import genai
    from google.genai import types
    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model=TEXT_MODEL,
        contents=[prompt],
        config=types.GenerateContentConfig(
            response_mime_type="application/json"
        )
    )

    try:
        text = response.text.strip()
        if text.startswith("```json"):
            text = text[7:]
        if text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()

        start_idx = text.find('{')
        end_idx = text.rfind('}')
        if start_idx != -1 and end_idx != -1:
            text = text[start_idx:end_idx + 1]

        return json.loads(text)
    except json.JSONDecodeError:
        print(f"❌ [Thumbnail] Failed to parse refined titles: {response.text}")
        return {"titles": ["Could not refine titles - please try again"]}


def _local_thumbnail(title, output_path, face_image_path=None):
    """Zero-cost PIL fallback: bold typographic thumbnail on a vibrant gradient.

    Used only when no image-generation provider is configured/working, so the
    YouTube Studio feature still delivers something useful at $0.00.
    """
    W, H = 1280, 720
    img = Image.new("RGB", (W, H))
    draw = ImageDraw.Draw(img)
    for y in range(H):
        r = int(20 + 120 * y / H)
        g = int(30 + 40 * y / H)
        b = int(90 + 165 * y / H)
        draw.line([(0, y), (W, y)], fill=(r, g, b))

    if face_image_path and os.path.exists(face_image_path):
        try:
            face = Image.open(face_image_path).convert("RGB")
            face.thumbnail((480, 480))
            face = face.resize((480, 480))
            img.paste(face, (W - 560, H // 2 - 240))
        except Exception:
            pass

    text = (title or "WATCH THIS")[:60]
    font = None
    for name in ("DejaVuSans-Bold.ttf", "Arial Bold.ttf", "Arial.ttf"):
        try:
            font = ImageFont.truetype(name, 72)
            break
        except Exception:
            continue
    if font is None:
        font = ImageFont.load_default()

    lines = []
    words = text.split()
    current = ""
    for w in words:
        candidate = (current + " " + w).strip()
        if draw.textlength(candidate, font=font) > W - 160 or len(current.split()) >= 4:
            lines.append(current)
            current = w
        else:
            current = candidate
    lines.append(current)

    y = H // 2 - 60 * len(lines) // 2
    for line in lines[:3]:
        draw.text((80, y), line.upper(), fill=(255, 255, 255),
                  stroke_width=6, stroke_fill=(0, 0, 0), font=font)
        y += 110

    img.save(output_path, "JPEG", quality=88)
    return output_path


def generate_thumbnail(api_key, title, session_id, face_image_path=None, bg_image_path=None, extra_prompt="", count=3, video_context=""):
    """
    Generates YouTube thumbnails. Uses the free image provider when available
    (OpenRouter image model / Gemini image generation), and falls back to a
    local typographic thumbnail at zero cost.
    Returns list of saved image paths (relative URLs).
    """
    output_dir = os.path.join("output", "thumbnails", session_id)
    os.makedirs(output_dir, exist_ok=True)

    prompt_parts = []
    if face_image_path and os.path.exists(face_image_path):
        prompt_parts.append(Image.open(face_image_path))

    if bg_image_path and os.path.exists(bg_image_path):
        prompt_parts.append(Image.open(bg_image_path))

    context_block = ""
    if video_context:
        context_block = f"""
VIDEO CONTEXT (use this to understand the video and design a relevant thumbnail):
{video_context}
"""

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

    thumbnails = []
    last_error = None

    gemini_key = ai_gateway.gemini_image_key()
    for i in range(count):
        print(f"🎨 [Thumbnail] Generating thumbnail {i + 1}/{count}...")
        filename = f"thumb_{i + 1}.jpg"
        filepath = os.path.join(output_dir, filename)

        # 1) Gemini image generation (legacy BYOK with a real Gemini key).
        if gemini_key:
            try:
                from google import genai
                from google.genai import types
                client = genai.Client(api_key=gemini_key)
                contents = prompt_parts + [text_prompt]
                response = client.models.generate_content(
                    model="gemini-3.1-flash-image-preview",
                    contents=contents,
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
                        image.save(filepath)
                        thumbnails.append(f"/thumbnails/{session_id}/{filename}")
                        print(f"✅ [Thumbnail] Saved: {filepath}")
                        break
                else:
                    raise RuntimeError("Gemini returned no image part")
                continue
            except Exception as e:
                last_error = str(e)
                print(f"❌ [Thumbnail] Gemini generation {i + 1} failed: {e}")

        # 2) Free image model via the gateway (OpenRouter & co.).
        try:
            got = ai_gateway.generate_image(
                text_prompt, filepath, size="1792x1024")
            if got:
                thumbnails.append(f"/thumbnails/{session_id}/{filename}")
                print(f"✅ [Thumbnail] Saved (free AI): {filepath}")
                continue
            last_error = "image provider returned nothing"
        except Exception as e:
            last_error = str(e)
            print(f"❌ [Thumbnail] Free-AI generation {i + 1} failed: {e}")

        # 3) Zero-cost local fallback.
        try:
            _local_thumbnail(title, filepath, face_image_path)
            thumbnails.append(f"/thumbnails/{session_id}/{filename}")
            print(f"🖼️ [Thumbnail] Local fallback saved: {filepath}")
        except Exception as e:
            last_error = str(e)
            print(f"❌ [Thumbnail] Local fallback {i + 1} failed: {e}")

    if not thumbnails and last_error:
        raise RuntimeError(f"All thumbnail generations failed. Last error: {last_error}")

    return thumbnails


def generate_youtube_description(api_key, title, transcript_segments, language, video_duration):
    """
    Uses the AI to generate a YouTube description with chapter markers from transcript segments.
    Returns: { "description": "full description text with chapters" }
    """
    formatted_segments = []
    for seg in transcript_segments:
        start = seg.get("start", 0)
        mins = int(start // 60)
        secs = int(start % 60)
        timestamp = f"{mins}:{secs:02d}"
        formatted_segments.append(f"[{timestamp}] {seg.get('text', '').strip()}")

    segments_text = "\n".join(formatted_segments)

    dur_mins = int(video_duration // 60)
    dur_secs = int(video_duration % 60)
    duration_str = f"{dur_mins}:{dur_secs:02d}"

    prompt = f"""You are a YouTube SEO expert. Generate a complete YouTube video description for the following video.

VIDEO TITLE: "{title}"
VIDEO LANGUAGE: {language}
VIDEO DURATION: {duration_str}

TRANSCRIPT WITH TIMESTAMPS:
{segments_text}

REQUIREMENTS:
1. Write the description in the SAME LANGUAGE as the video ({language})
2. Start with a compelling 2-3 sentence summary/hook
3. Add relevant CTAs (subscribe, like, comment)
4. Generate YouTube CHAPTERS based on the transcript timestamps:
   - First chapter MUST start at 0:00
   - Minimum 3 chapters, each at least 10 seconds apart
   - Chapter titles should be concise and descriptive
   - Format: 0:00 Chapter Title
   - Place chapters in their own section with a blank line before and after
5. Add 5-10 relevant hashtags at the end
6. Keep the total description under 5000 characters

OUTPUT: Return ONLY the description text (no JSON wrapper, no markdown code blocks). The description should be ready to paste directly into YouTube."""

    print("🤖 [Thumbnail] Generating YouTube description with chapters...")
    if ai_gateway.is_configured():
        try:
            result = ai_gateway.complete(
                system="You are a YouTube SEO expert. Answer with plain text only.",
                user=prompt, temperature=0.7)
            description = result.text
        except ai_gateway.AIGatewayError as e:
            print(f"❌ [Thumbnail] Description generation failed: {e}")
            return {"description": "Could not generate description - please try again."}
    else:
        from google import genai
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model=TEXT_MODEL,
            contents=[prompt],
        )
        description = response.text.strip()

    # Clean up any accidental markdown wrappers
    if description.startswith("```"):
        lines = description.split("\n")
        description = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])

    return {"description": description}
