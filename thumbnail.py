import io
import os
import uuid
import json
from concurrent.futures import ThreadPoolExecutor

from google import genai
from google.genai import types
from PIL import Image, ImageDraw, ImageFont, ImageFilter

import rtl_text

# Text/analysis model (titles, concepts, description). Deliberately NOT tied to
# GEMINI_MODEL: the pipeline runs flash-lite for a closed-choice layout pick,
# but titles are the one place where the creative gap between lite and flash
# shows, and ten titles per video cost cents either way. The image model
# stays on gemini-3.1-flash-image; the text models cannot draw.
TEXT_MODEL = os.environ.get("GEMINI_MODEL_THUMBNAIL") or "gemini-3.7-flash"
IMAGE_MODEL = os.environ.get("GEMINI_IMAGE_MODEL") or "gemini-3.1-flash-image"

# Frames sent with the transcript instead of the whole video. Gemini bills
# video at ~300 tokens/s, so an hour is ~1M tokens for what is a text task;
# ten 1024px frames cost ~3k tokens however long the source runs.
TITLE_FRAMES = 10
TITLE_FRAME_WIDTH = 1024

# YouTube shows ~50 characters of a title on a phone before cutting it.
TITLE_MOBILE_CHARS = 50
TITLE_MAX_CHARS = 65

THUMB_W, THUMB_H = 1280, 720
THUMB_MAX_BYTES = 2 * 1024 * 1024  # YouTube's upload limit
THUMB_FONT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fonts", "Anton-Regular.ttf")
TEXT_POSITIONS = ("left", "right", "top", "bottom")


def _parse_json(text):
    """Gemini JSON, tolerant of code fences and prose around the object."""
    text = (text or "").strip()
    start = text.find("{")
    if start == -1:
        raise json.JSONDecodeError("no object", text, 0)
    # raw_decode stops at the end of the first complete object, so a model
    # that appends a stray "]}" (seen with flash) does not break the parse.
    obj, _ = json.JSONDecoder().raw_decode(text[start:])
    return obj


def _frame_parts(video_path, n=TITLE_FRAMES, width=TITLE_FRAME_WIDTH):
    """Evenly spaced frames as Gemini image parts (empty list on any failure)."""
    try:
        from layout_picker import sample_frames
        return [types.Part.from_bytes(data=b, mime_type="image/jpeg")
                for b in sample_frames(video_path, n=n, width=width)]
    except Exception as e:
        print(f"⚠️ [Thumbnail] Could not sample frames: {e}")
        return []


# ---------------------------------------------------------------------------
# Titles
# ---------------------------------------------------------------------------

def analyze_video_for_titles(api_key, video_path, transcript=None):
    """
    Suggests YouTube titles from the transcript plus a handful of frames.
    Two calls: a wide brainstorm, then a critic that scores and picks a
    diverse top 10 and pairs each title with a 1-4 word thumbnail hook.
    Returns: { "titles", "thumbnail_texts", "transcript_summary", "language",
               "segments", "video_duration", "recommended" }
    """
    if transcript is None:
        from main import transcribe_video
        print("🎬 [Thumbnail] Transcribing video...")
        transcript = transcribe_video(video_path)
    else:
        print("🎬 [Thumbnail] Using pre-computed transcript (Whisper already done)...")

    client = genai.Client(api_key=api_key)
    frames = _frame_parts(video_path)
    language = transcript.get("language", "en")
    segments = transcript.get("segments", [])
    video_duration = segments[-1]["end"] if segments else 0
    # Enough transcript for a title; a 3-hour podcast does not need all of it.
    transcript_text = transcript.get("text", "")[:60000]

    brainstorm_prompt = f"""You are a YouTube packaging expert (titles + thumbnails) for a channel that wants maximum CTR without lying.

The images are frames sampled evenly across the video. The transcript is below.

TRANSCRIPT (language: {language}):
{transcript_text}

TASK 1: Summarize the video in 2-3 sentences (what it is, who it is for, the single most surprising or valuable point).

TASK 2: Brainstorm 25 candidate titles. Cover ALL of these styles, at least 3 each:
- specific outcome / result ("I did X and Y happened")
- curiosity gap (withhold the key fact, never clickbait the video does not pay off)
- contrarian / myth-busting
- how-to with a concrete promise
- number / list
- question the viewer already asks themselves
- story / confession

TITLE RULES:
- Write in the SAME LANGUAGE as the transcript ({language}). Never translate.
- Ideal length {TITLE_MOBILE_CHARS} characters or less; hard maximum {TITLE_MAX_CHARS}. Phones cut titles after ~{TITLE_MOBILE_CHARS} characters, so the payoff must land inside the first {TITLE_MOBILE_CHARS}.
- Put the main keyword / subject in the first 3 words.
- Be specific to THIS video: names, numbers, tools, places from the transcript. Nothing generic.
- No ALL CAPS words, at most one exclamation mark, no emojis.

OUTPUT JSON:
{{
  "transcript_summary": "...",
  "candidates": ["title", ...]
}}"""

    print("🤖 [Thumbnail] Brainstorming titles...")
    response = client.models.generate_content(
        model=TEXT_MODEL,
        contents=frames + [brainstorm_prompt],
        config=types.GenerateContentConfig(response_mime_type="application/json"),
    )
    try:
        draft = _parse_json(response.text)
    except (json.JSONDecodeError, AttributeError):
        print(f"❌ [Thumbnail] Failed to parse brainstorm JSON: {getattr(response, 'text', '')}")
        return {
            "titles": ["Could not generate titles - please try again"],
            "thumbnail_texts": [],
            "transcript_summary": transcript.get("text", "")[:500],
            "language": language,
            "segments": segments,
            "video_duration": video_duration,
            "recommended": [],
        }

    summary = draft.get("transcript_summary", "")
    candidates = [c for c in draft.get("candidates", []) if isinstance(c, str) and c.strip()]

    critic_prompt = f"""You are a ruthless YouTube title critic. Below are candidate titles for one video.

VIDEO SUMMARY: {summary}
LANGUAGE: {language}

CANDIDATES:
{json.dumps(candidates, ensure_ascii=False, indent=2)}

Score every candidate 1-10 on each of: specificity (concrete to this video), curiosity (makes you need to click), clarity (understood in 1 second on a phone), honesty (the video delivers it). Reject anything over {TITLE_MAX_CHARS} characters or whose hook lands after character {TITLE_MOBILE_CHARS}.

Then return the 10 best, ordered best first, with NO two titles using the same angle (different styles, different hooks). You may lightly edit a title to fix length or sharpen it, keeping the language ({language}).

For each of the 10, write the THUMBNAIL TEXT: 1-4 words, ALL CAPS, in {language}, that COMPLEMENTS the title instead of repeating it (the title says what, the thumbnail says the emotion, the number or the twist). Examples of the relationship: title "How I got 10k subscribers in 30 days" -> thumbnail "0 TO 10K"; title "Por qué dejé mi trabajo en Google" -> thumbnail "ME FUI".

Pick the top 2 and explain in one sentence each why they will win the click.

OUTPUT JSON:
{{
  "titles": ["...", ... 10 items],
  "thumbnail_texts": ["...", ... 10 items, same order],
  "recommended": [
    {{"index": 0, "reason": "..."}},
    {{"index": 3, "reason": "..."}}
  ]
}}"""

    print("🧐 [Thumbnail] Scoring titles...")
    response = client.models.generate_content(
        model=TEXT_MODEL,
        contents=[critic_prompt],
        config=types.GenerateContentConfig(response_mime_type="application/json"),
    )
    try:
        picked = _parse_json(response.text)
        titles = [t for t in picked.get("titles", []) if isinstance(t, str) and t.strip()]
        if not titles:
            raise ValueError("no titles")
    except (json.JSONDecodeError, AttributeError, ValueError):
        print(f"⚠️ [Thumbnail] Critic failed, falling back to the brainstorm: {getattr(response, 'text', '')[:300]}")
        titles = candidates[:10]
        picked = {"thumbnail_texts": [], "recommended": []}

    thumbnail_texts = [str(t) for t in picked.get("thumbnail_texts", [])][:len(titles)]
    thumbnail_texts += [""] * (len(titles) - len(thumbnail_texts))
    return {
        "titles": titles,
        "thumbnail_texts": thumbnail_texts,
        "transcript_summary": summary,
        "language": language,
        "segments": segments,
        "video_duration": video_duration,
        "recommended": [r for r in picked.get("recommended", [])
                        if isinstance(r, dict) and isinstance(r.get("index"), int)
                        and 0 <= r["index"] < len(titles)],
    }


def refine_titles(api_key, context, user_message, conversation_history=None):
    """
    Takes video context + user feedback and returns refined title suggestions.
    """
    client = genai.Client(api_key=api_key)

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
- Ideal length {TITLE_MOBILE_CHARS} characters or less, hard maximum {TITLE_MAX_CHARS} (phones cut after ~{TITLE_MOBILE_CHARS})
- Main keyword / subject in the first 3 words
- Incorporate the user's feedback/direction; if they ask for a style, follow it
- Specific to this video (names, numbers, tools), never generic
- Same language as the original content unless the user asks for another language: then write ALL titles in the requested language

For each title also write its THUMBNAIL TEXT: 1-4 words, ALL CAPS, same language as the title, complementing it (emotion, number or twist), never repeating it.

OUTPUT JSON:
{{
    "titles": ["title1", "title2", ...],
    "thumbnail_texts": ["...", ... same order],
    "language": "ISO 639-1 code of the language the titles are written in"
}}"""

    response = client.models.generate_content(
        model=TEXT_MODEL,
        contents=[prompt],
        config=types.GenerateContentConfig(
            response_mime_type="application/json"
        )
    )

    try:
        result = _parse_json(response.text)
        titles = [t for t in result.get("titles", []) if isinstance(t, str) and t.strip()]
        texts = [str(t) for t in result.get("thumbnail_texts", [])][:len(titles)]
        texts += [""] * (len(titles) - len(texts))
        return {"titles": titles, "thumbnail_texts": texts,
                "language": str(result.get("language") or "")[:5]}
    except (json.JSONDecodeError, AttributeError):
        print(f"❌ [Thumbnail] Failed to parse refined titles: {response.text}")
        return {"titles": ["Could not refine titles - please try again"], "thumbnail_texts": [], "language": ""}


# ---------------------------------------------------------------------------
# Frames from the video
# ---------------------------------------------------------------------------

def extract_face_frames(video_path, session_id, n=5, samples=40):
    """
    Frames from the video worth building a thumbnail on: a large, sharp face,
    spread across the runtime so the user gets different moments, not five
    near-duplicates of one second. Saved as JPEGs under the session's
    thumbnails dir. Returns [{"url", "path", "time", "face": [x, y, w, h]}].
    """
    import cv2
    from main import detect_face_candidates

    out_dir = os.path.join("output", "thumbnails", session_id, "frames")
    os.makedirs(out_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    scored = []
    try:
        if total <= 0:
            return []
        for i in range(samples):
            idx = int((i + 0.5) * total / samples)
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if not ok:
                continue
            faces = detect_face_candidates(frame)
            if not faces:
                continue
            x, y, w, h = max(faces, key=lambda f: f["score"])["box"]
            fh, fw = frame.shape[:2]
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            scored.append(rank_frame(idx, idx / fps, frame, [x, y, w, h], (fw, fh),
                                     cv2.Laplacian(gray, cv2.CV_64F).var()))
    finally:
        cap.release()

    picked = pick_spread(scored, n, total)

    results = []
    for k, cand in enumerate(picked):
        frame = cand["frame"]
        fh, fw = frame.shape[:2]
        # Keep native resolution up to 1920 wide: the face crop sent as the
        # person reference comes from this file, and a 500px face gives the
        # image model too little to keep the likeness.
        scale = min(1.0, 1920 / float(fw))
        resized = cv2.resize(frame, (int(fw * scale), max(2, int(fh * scale))), interpolation=cv2.INTER_AREA) if scale < 1.0 else frame
        path = os.path.join(out_dir, f"frame_{k + 1}.jpg")
        cv2.imwrite(path, resized, [cv2.IMWRITE_JPEG_QUALITY, 90])
        results.append({
            "url": f"/thumbnails/{session_id}/frames/frame_{k + 1}.jpg",
            "path": path,
            "time": round(cand["time"], 1),
            "face": [int(v * scale) for v in cand["face"]],
        })
    print(f"🖼️ [Thumbnail] {len(results)} face frames picked from {len(scored)} candidates")
    return results


def rank_frame(idx, time, frame, face_box, frame_size, sharpness):
    """Score one candidate frame: face area, boosted by sharpness (capped so a
    noisy frame cannot outrank a bigger face). None below 1% face area."""
    x, y, w, h = face_box
    fw, fh = frame_size
    area = (w * h) / float(fw * fh)
    if area < 0.01:
        return None
    return {
        "idx": idx, "time": time, "frame": frame,
        "face": [int(x), int(y), int(w), int(h)],
        "score": area * (1.0 + min(sharpness, 500.0) / 500.0),
    }


def pick_spread(scored, n, total_frames, min_gap_fraction=0.04):
    """Best-first, but every pick at least `min_gap_fraction` of the runtime
    away from the others, returned in timeline order."""
    scored = sorted((s for s in scored if s), key=lambda s: -s["score"])
    picked = []
    min_gap = total_frames * min_gap_fraction
    for cand in scored:
        if all(abs(cand["idx"] - p["idx"]) >= min_gap for p in picked):
            picked.append(cand)
        if len(picked) >= n:
            break
    picked.sort(key=lambda s: s["idx"])
    return picked


def _crop_face_reference(frame_path, face_box, out_path):
    """Head-and-shoulders crop around the detected face, as a person reference
    for the image model. Falls back to the whole frame without a box."""
    img = Image.open(frame_path).convert("RGB")
    if face_box:
        x, y, w, h = face_box
        cx, cy = x + w / 2, y + h / 2
        side = max(w, h) * 2.4
        left = max(0, int(cx - side / 2))
        top = max(0, int(cy - side * 0.45))
        right = min(img.width, int(cx + side / 2))
        bottom = min(img.height, int(cy + side * 0.75))
        img = img.crop((left, top, right, bottom))
    img.save(out_path, quality=92)
    return out_path


# ---------------------------------------------------------------------------
# Thumbnails
# ---------------------------------------------------------------------------

def plan_thumbnail_concepts(client, title, count, video_context="", extra_prompt="",
                            thumbnail_text_hint="", has_person=False, language="en"):
    """
    One text call that designs `count` DIFFERENT thumbnails before any pixel is
    drawn. Asking the image model to invent the concept and render it in one
    go gives N variations of one idea; splitting it gives N ideas.
    """
    person_line = ("A real photo of the presenter is provided and MUST be the focal point, "
                   "with a clear, natural expression that reads at small size (no caricature)."
                   if has_person else
                   "No photo of the presenter is available: build the thumbnail around an object, "
                   "a scene, a before/after, or a symbol; do NOT invent a specific person's face "
                   "unless the user asks for one.")
    hint_line = (f'\nSUGGESTED THUMBNAIL TEXT (use it for the first concept unless it is poor): "{thumbnail_text_hint}"'
                 if thumbnail_text_hint else "")
    extra_line = f"\nMANDATORY USER INSTRUCTIONS (override everything else):\n{extra_prompt}\n" if extra_prompt else ""
    prompt = f"""You are a YouTube thumbnail art director. Design {count} DIFFERENT thumbnail concepts for this video.

VIDEO TITLE: "{title}"
LANGUAGE OF THE AUDIENCE: {language}
VIDEO CONTEXT: {video_context or "(none)"}
{person_line}{hint_line}{extra_line}
The thumbnail must show WHAT the video is about, not only how the presenter feels: name the concrete subject (the product, the tool, the result, the numbers) and build the picture around it. A face alone with two words is the lazy default; use it at most once, and only combined with a strong prop or scene.

Each concept must use a different device, e.g.: the presenter + a prop that IS the topic (a phone showing vertical clips, a laptop with the app, a counter/badge with the number); before/after or "long video -> short clips" split; product/UI hero shot with the presenter small; a giant number or stat as the hero with the subject behind; "X vs Y" comparison; a red circle/arrow on a detail; a dramatic reveal with depth (foreground object, blurred background). Layered, cinematic, premium: real props, depth of field, rim light, particles, glow on the key object. Never flat, never a plain wall behind the person. No two concepts may share the same text or the same layout.

Per concept give:
- "text": 1-4 words, ALL CAPS, in {language}. Complements the title (the title says what, the text says the emotion, the number or the twist). Never repeat the title. No emoji.
- "text_position": one of "left", "right", "top", "bottom" (where the text block sits; the subject goes on the opposite side).
- "text_color": "white" or "yellow" (yellow for money/energy, white otherwise).
- "scene": a precise image-generation prompt in English, 2-4 sentences: subject, expression/pose, background, lighting, colours, camera. Say that the text_position side is clean negative space (simple, dark or plain) so a headline can sit there. It must NOT ask for any text, letters, captions or logos in the image.
- The image model REFUSES prompts that name or depict real people, companies or brands. Never write a real name (person, company, product, logo) in "scene". Refer to the presenter only as "the person in the provided photo" when a photo is provided, otherwise as a generic, unnamed person or use objects and symbols.
- "why": one short sentence on why this earns the click.

OUTPUT JSON:
{{"concepts": [{{"text": "...", "text_position": "left", "text_color": "yellow", "scene": "...", "why": "..."}}, ...]}}"""

    response = client.models.generate_content(
        model=TEXT_MODEL,
        contents=[prompt],
        config=types.GenerateContentConfig(response_mime_type="application/json"),
    )
    try:
        concepts = _parse_json(response.text).get("concepts", [])
    except (json.JSONDecodeError, AttributeError):
        print(f"⚠️ [Thumbnail] Concept JSON unreadable, using a generic concept: {getattr(response, 'text', '')[:200]}")
        concepts = []
    return normalise_concepts(concepts, count, title, thumbnail_text_hint)


def normalise_concepts(concepts, count, title, thumbnail_text_hint=""):
    """Clamp model output to what the renderer accepts and pad to `count`."""
    cleaned = []
    for c in concepts:
        if not isinstance(c, dict) or not c.get("scene"):
            continue
        cleaned.append({
            "text": str(c.get("text") or thumbnail_text_hint or "").strip().upper()[:40],
            "text_position": c.get("text_position") if c.get("text_position") in TEXT_POSITIONS else "left",
            "text_color": "yellow" if c.get("text_color") == "yellow" else "white",
            "scene": str(c["scene"]),
            "why": str(c.get("why", "")),
        })
    while len(cleaned) < count:
        cleaned.append({
            "text": thumbnail_text_hint.upper()[:40],
            "text_position": "left",
            "text_color": "white",
            "scene": (f"A bold, high-contrast YouTube thumbnail scene for a video titled '{title}'. "
                      "Dramatic lighting, vibrant colours, the left third is clean dark negative space."),
            "why": "fallback",
        })
    return cleaned[:count]


def _wrap_lines(draw, text, font, max_width, **opts):
    words = text.split()
    lines, cur = [], []
    for w in words:
        test = " ".join(cur + [w])
        if draw.textlength(test, font=font, **opts) <= max_width or not cur:
            cur.append(w)
        else:
            lines.append(" ".join(cur))
            cur = [w]
    if cur:
        lines.append(" ".join(cur))
    return lines


def burn_thumbnail_text(img, text, position="left", color="white"):
    """
    Draw the hook text with PIL: Anton, heavy black stroke, drop shadow,
    filling the chosen side of the canvas. Image models misspell accents and
    split words; this never does.
    """
    text = (text or "").strip()
    if not text:
        return img
    img = img.convert("RGBA")
    W, H = img.size
    horizontal = position in ("left", "right")
    box_w = int(W * (0.46 if horizontal else 0.92))
    box_h = int(H * (0.84 if horizontal else 0.40))
    fill = (255, 221, 0, 255) if color == "yellow" else (255, 255, 255, 255)

    draw = ImageDraw.Draw(img)
    size = int(H * 0.30)
    while True:
        arabic = rtl_text.is_arabic(text)
        opts = rtl_text.layout_opts() if arabic else {}
        font = ImageFont.truetype(rtl_text.ARABIC_FONT_PATH if arabic else THUMB_FONT, size)
        lines = _wrap_lines(draw, text, font, box_w, **opts)
        # Anton's accents sit above the cap height: 1.15 keeps a tilde on
        # line 2 off the baseline of line 1.
        line_h = int(size * (1.5 if arabic else 1.15))
        widest = max(draw.textlength(l, font=font, **opts) for l in lines)
        if (len(lines) <= 3 and widest <= box_w and line_h * len(lines) <= box_h) or size <= 24:
            break
        size = int(size * 0.92)
    stroke = max(4, size // 12)
    total_h = line_h * len(lines)
    # PIL draws from the ascender line; shift so the block is measured from the
    # tallest glyph actually present (caps sit lower than accents).
    top_pad = min(font.getbbox(l, **opts)[1] for l in lines)

    if position == "left":
        x0, y0 = int(W * 0.05), (H - total_h) // 2 - top_pad
    elif position == "right":
        x0, y0 = W - box_w - int(W * 0.05), (H - total_h) // 2 - top_pad
    elif position == "top":
        x0, y0 = int(W * 0.04), int(H * 0.05) - top_pad
    else:
        x0, y0 = int(W * 0.04), H - total_h - int(H * 0.06) - top_pad

    def line_x(line):
        if arabic and horizontal:  # RTL lines share their right edge
            return x0 + widest - draw.textlength(line, font=font, **opts)
        return x0 if horizontal else (W - draw.textlength(line, font=font, **opts)) // 2

    shadow = Image.new("RGBA", img.size, (0, 0, 0, 0))
    sdraw = ImageDraw.Draw(shadow)
    for i, line in enumerate(lines):
        sdraw.text((line_x(line) + stroke, y0 + i * line_h + stroke * 1.5), line, font=font,
                   fill=(0, 0, 0, 170), stroke_width=stroke, stroke_fill=(0, 0, 0, 170), **opts)
    shadow = shadow.filter(ImageFilter.GaussianBlur(stroke))
    img = Image.alpha_composite(img, shadow)
    draw = ImageDraw.Draw(img)
    for i, line in enumerate(lines):
        draw.text((line_x(line), y0 + i * line_h), line, font=font, fill=fill,
                  stroke_width=stroke, stroke_fill=(0, 0, 0, 255), **opts)
    return img


# AI Act art. 50(2): a synthetic image has to be marked in a machine-readable
# way, and these thumbnails are exactly the case the article is about — an
# image model painting a scene, often around a reference photo of the real
# presenter. Videos get the mp4 `comment` tag (ffmpeg_utils.mark_ai_generated);
# a JPEG's equivalent is XMP, and the interoperable vocabulary is IPTC's
# DigitalSourceType, whose `trainedAlgorithmicMedia` value is what Google,
# LinkedIn, TikTok and Adobe already read. Written alongside the EXIF Software
# tag because the two survive different pipelines: a re-encode usually keeps
# EXIF and drops XMP, a metadata scrub keeps neither, and this is the honest
# ceiling of what a JPEG can carry without a C2PA signing key.
AI_XMP_PACKET = (
    '<?xpacket begin="\ufeff" id="W5M0MpCehiHzreSzNTczkc9d"?>'
    '<x:xmpmeta xmlns:x="adobe:ns:meta/">'
    '<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">'
    '<rdf:Description rdf:about=""'
    ' xmlns:Iptc4xmpExt="http://iptc.org/std/Iptc4xmpExt/2008-02-29/"'
    ' xmlns:xmp="http://ns.adobe.com/xap/1.0/"'
    ' xmlns:dc="http://purl.org/dc/elements/1.1/"'
    ' Iptc4xmpExt:DigitalSourceType='
    '"http://cv.iptc.org/newscodes/digitalsourcetype/trainedAlgorithmicMedia"'
    ' xmp:CreatorTool="OpenShorts (openshorts.app)">'
    '<dc:description><rdf:Alt><rdf:li xml:lang="x-default">'
    'AI-generated image produced with OpenShorts (openshorts.app)'
    '</rdf:li></rdf:Alt></dc:description>'
    '</rdf:Description></rdf:RDF></x:xmpmeta>'
    '<?xpacket end="w"?>'
).encode("utf-8")

AI_SOFTWARE_TAG = "OpenShorts (openshorts.app) — AI-generated image"


def _ai_save_kwargs(img):
    """XMP + EXIF marking for a generated image. Never raises.

    Pillow grew the ``xmp=`` save argument for JPEG relatively late, so an
    older Pillow simply gets the EXIF half rather than failing the render.
    """
    kwargs = {}
    try:
        exif = img.getexif()
        exif[0x0131] = AI_SOFTWARE_TAG                    # Software
        exif[0x010E] = AI_SOFTWARE_TAG                    # ImageDescription
        kwargs["exif"] = exif.tobytes()
    except Exception:
        pass
    kwargs["xmp"] = AI_XMP_PACKET
    return kwargs


def finalize_thumbnail(img, out_path):
    """Cover-crop to 1280x720 and save a JPEG under YouTube's 2 MB limit.

    Every file that leaves here is machine-marked as AI-generated — see
    AI_XMP_PACKET.
    """
    img = img.convert("RGB")
    scale = max(THUMB_W / img.width, THUMB_H / img.height)
    img = img.resize((max(THUMB_W, int(img.width * scale + 0.5)),
                      max(THUMB_H, int(img.height * scale + 0.5))), Image.LANCZOS)
    left = (img.width - THUMB_W) // 2
    top = (img.height - THUMB_H) // 2
    img = img.crop((left, top, left + THUMB_W, top + THUMB_H))
    marks = _ai_save_kwargs(img)
    for q in (92, 88, 84, 78, 70, 60):
        try:
            img.save(out_path, "JPEG", quality=q, optimize=True, **marks)
        except (TypeError, ValueError):
            # An older Pillow without the xmp= argument: keep the EXIF mark
            # rather than losing the thumbnail.
            marks.pop("xmp", None)
            img.save(out_path, "JPEG", quality=q, optimize=True, **marks)
        if os.path.getsize(out_path) <= THUMB_MAX_BYTES:
            break
    return out_path


def _image_part(path):
    """A JPEG byte part for the image model (re-encoded so HEIC/PNG uploads
    and odd modes all arrive the same way)."""
    buf = io.BytesIO()
    Image.open(path).convert("RGB").save(buf, "JPEG", quality=92)
    return types.Part.from_bytes(data=buf.getvalue(), mime_type="image/jpeg")


def _generate_one(client, concept, reference_images, out_path, burn_text):
    """One image call for one concept; returns the saved path or raises."""
    if burn_text:
        pos = concept['text_position']
        share = "45% of the width" if pos in ("left", "right") else "40% of the height"
        text_rule = ("Do NOT render any text, letters, numbers, captions, logos or watermarks anywhere "
                     f"in the image. The {pos} {share} of the frame must be clean, simple negative space "
                     "(plain, dark or softly blurred, no objects, no detail): a headline will be placed "
                     "there afterwards. Put the subject and every object in the remaining part.")
    else:
        text_rule = (f'Render the text "{concept["text"]}" in huge bold condensed sans-serif capitals, '
                     f'{concept["text_color"]} with a thick black outline, on the {concept["text_position"]} '
                     "side, spelled EXACTLY as given. No other text.")
    prompt = f"""Generate a professional YouTube thumbnail, 16:9.

{concept['scene']}

{text_rule}

Style: high contrast, saturated colours, crisp subject separation, cinematic lighting, sharp focus on the subject, readable at 168x94 pixels. No clutter, no small details, no borders, no watermark."""
    if reference_images:
        prompt += ("\nIDENTITY: the person in the provided photo must appear as EXACTLY the same real person: "
                   "identical face shape, skin, eyes, glasses, facial hair, hairstyle and hair length, age and "
                   "body type. Photorealistic, like a photo of them; do not idealize, slim, rejuvenate or "
                   "stylize them. Expression may change slightly but must stay natural and true to their face.")

    response = client.models.generate_content(
        model=IMAGE_MODEL,
        contents=reference_images + [prompt],
        config=types.GenerateContentConfig(
            response_modalities=["TEXT", "IMAGE"],
            image_config=types.ImageConfig(aspect_ratio="16:9", image_size="2K"),
        ),
    )
    if not response.parts:
        cand = (response.candidates or [None])[0]
        reason = getattr(cand, "finish_reason", None) or getattr(response, "prompt_feedback", None)
        raise RuntimeError(f"Gemini returned no image (finish_reason={reason})")
    for part in response.parts:
        if part.text is not None:
            print(f"📝 [Thumbnail] Gemini text: {part.text[:200]}")
            continue
        if part.inline_data is None or not part.inline_data.data:
            continue
        pil = Image.open(io.BytesIO(part.inline_data.data))
        if burn_text:
            pil = burn_thumbnail_text(pil, concept["text"], concept["text_position"], concept["text_color"])
        return finalize_thumbnail(pil, out_path)
    raise RuntimeError("Gemini returned no image")


def generate_thumbnail(api_key, title, session_id, face_image_path=None, bg_image_path=None,
                       extra_prompt="", count=3, video_context="", burn_text=True,
                       thumbnail_text_hint="", language="en", frame_reference=None):
    """
    Generates `count` thumbnails, each from its own concept, in parallel.
    frame_reference: {"path", "face"} from extract_face_frames, used as the
    person reference when the user picked a frame instead of uploading a photo.
    Returns [{"url", "text", "why"}] (only the ones that rendered).
    """
    client = genai.Client(api_key=api_key)
    output_dir = os.path.join("output", "thumbnails", session_id)
    os.makedirs(output_dir, exist_ok=True)

    # References travel as immutable byte parts: one PIL Image shared by the
    # worker threads below raced inside the SDK's encoder and every call died.
    reference_images = []
    if face_image_path and os.path.exists(face_image_path):
        reference_images.append(_image_part(face_image_path))
    elif frame_reference and os.path.exists(frame_reference.get("path", "")):
        ref_path = os.path.join(output_dir, "person_reference.jpg")
        _crop_face_reference(frame_reference["path"], frame_reference.get("face"), ref_path)
        reference_images.append(_image_part(ref_path))
    if bg_image_path and os.path.exists(bg_image_path):
        reference_images.append(_image_part(bg_image_path))

    concepts = plan_thumbnail_concepts(
        client, title, count, video_context=video_context, extra_prompt=extra_prompt,
        thumbnail_text_hint=thumbnail_text_hint, has_person=bool(reference_images), language=language)
    for i, c in enumerate(concepts):
        print(f"🎨 [Thumbnail] Concept {i + 1}: \"{c['text']}\" ({c['text_position']}) - {c['why']}")

    # Unique names per batch so a regenerate never serves a stale cached file.
    batch = uuid.uuid4().hex[:6]

    def run(i):
        out_path = os.path.join(output_dir, f"thumb_{batch}_{i + 1}.jpg")
        concept = concepts[i]
        try:
            try:
                _generate_one(client, concept, reference_images, out_path, burn_text)
                fallback = False
            except RuntimeError as e:
                if "no image" not in str(e):
                    raise
                # Gemini refuses recognisable public figures, by reference photo
                # and by name alike (IMAGE_OTHER). Retry once with no reference
                # and a generic presenter instead of failing the whole batch.
                print(f"⚠️ [Thumbnail] Generation {i + 1} blocked ({e}); retrying without the person")
                generic = dict(concept, scene=(
                    "Do not depict any real, named or recognisable person; if a presenter is needed, "
                    "show a generic one seen from behind or in silhouette, or use an object instead. "
                    + concept["scene"]))
                _generate_one(client, generic, [], out_path, burn_text)
                fallback = True
            print(f"✅ [Thumbnail] Saved: {out_path}")
            return {"url": f"/thumbnails/{session_id}/{os.path.basename(out_path)}",
                    "text": concept["text"], "why": concept["why"], "fallback": fallback}
        except Exception as e:
            print(f"❌ [Thumbnail] Generation {i + 1} failed: {e}")
            return {"error": str(e)}

    with ThreadPoolExecutor(max_workers=min(count, 4)) as pool:
        results = list(pool.map(run, range(len(concepts))))

    thumbnails = [r for r in results if "url" in r]
    if not thumbnails:
        last_error = next((r["error"] for r in results if "error" in r), "unknown")
        raise RuntimeError(f"All thumbnail generations failed. Last error: {last_error}")
    return thumbnails


# ---------------------------------------------------------------------------
# Description
# ---------------------------------------------------------------------------

def generate_youtube_description(api_key, title, transcript_segments, language, video_duration,
                                 max_chars=200000):
    """
    Uses Gemini to generate a YouTube description with chapter markers from transcript segments.
    Returns: { "description": "full description text with chapters" }
    """
    client = genai.Client(api_key=api_key)

    # Format segments for the prompt
    formatted_segments = []
    for seg in transcript_segments:
        start = seg.get("start", 0)
        mins = int(start // 60)
        secs = int(start % 60)
        timestamp = f"{mins}:{secs:02d}"
        formatted_segments.append(f"[{timestamp}] {seg.get('text', '').strip()}")

    # Bounded: the source length is capped upstream, this caps the prompt.
    segments_text = "\n".join(formatted_segments)[:max_chars]

    # Format total duration
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
