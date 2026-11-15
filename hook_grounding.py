"""Rewrite a clip's hook and title from what is actually on its screen.

The moment picker chooses clips and writes their hook from the transcript
alone; it never sees a frame. On a talking head that is fine. On a clip the
reframe rendered as SCREENCAST / WIDE / INSET the meaning is on the screen
(a settings dialog, a spreadsheet, a terminal) and the hook comes out as a
summary of the video's topic ("I automated my clips with AI") while the
viewer is watching an MCP connector being configured.

So, for those clips only, once the render has said which stretches live on
the screen (the ``<clip>.layout.json`` sidecar), three frames from those
stretches at 1024px plus the clip's own transcript go to Gemini and the hook
and title are rewritten to name what is shown. Same principle as the layout
picker: a few frames at a legible resolution answering one concrete
question. About 3k tokens per clip.

Frames need a model that can see, so this is Gemini-only: with just a local
LLM (``LLM_BASE_URL``) the function returns None and the transcript-based
hook stands. Never raises: a hook problem must never cost the clip.
"""
from __future__ import annotations

import json
import os
from typing import Optional

SCREEN_LAYOUTS = {"screencast", "wide", "inset"}
FRAMES = int(os.environ.get("HOOK_GROUNDING_FRAMES", "3"))
WIDTH = int(os.environ.get("HOOK_GROUNDING_WIDTH", "1024"))
# Ignore a blip: the screen stretches must cover this share of the clip.
MIN_SHARE = float(os.environ.get("HOOK_GROUNDING_MIN_SHARE", "0.25"))


def enabled() -> bool:
    return os.environ.get("HOOK_GROUNDING", "1").strip() != "0"


def screen_video() -> bool:
    """True when the layout picker (or the user) called this source a
    screencast. On such a video a GENERAL stretch, which the scene classifier
    emits for "no face in frame", is a full-screen slide, dialog or terminal
    (verified on a product update video: every GENERAL scene was a stats
    card), even when the width gate did not upgrade it to SCREENCAST."""
    try:
        import screencast_layout
        return bool(getattr(screencast_layout, "ENABLED", False))
    except Exception:
        return False


def screen_ranges(ranges, on_screen_video=None):
    """(start, end) pairs, clip seconds, of the stretches rendered on-screen."""
    import layout_ranges
    layouts = set(SCREEN_LAYOUTS)
    if screen_video() if on_screen_video is None else on_screen_video:
        layouts.add("general")
    return [(r["start"], r["end"]) for r in layout_ranges.normalise(ranges)
            if r["layout"] in layouts]


def wanted(ranges, clip_duration) -> bool:
    """True when enough of the clip lives on the screen to reground the hook."""
    if not enabled() or not clip_duration or clip_duration <= 0:
        return False
    covered = sum(e - s for s, e in screen_ranges(ranges))
    return covered / float(clip_duration) >= MIN_SHARE


def clip_words(transcript, start, end) -> str:
    """What is said between ``start`` and ``end`` (source seconds)."""
    out = []
    for seg in (transcript or {}).get("segments", []) or []:
        words = seg.get("words") or []
        if words:
            out.extend(w.get("word", "") for w in words
                       if w.get("end", 0) > start and w.get("start", 0) < end)
        elif seg.get("end", 0) > start and seg.get("start", 0) < end:
            out.append(seg.get("text", ""))
    return " ".join(t.strip() for t in out if t and t.strip())


def sample_times(ranges, clip_duration, n=None):
    """``n`` timestamps spread over the on-screen stretches (whole clip if
    the sidecar has none), never on the very first frame of a stretch, where
    a cut is still settling."""
    n = n or FRAMES
    spans = screen_ranges(ranges) or [(0.0, float(clip_duration))]
    total = sum(e - s for s, e in spans) or float(clip_duration)
    times = []
    for i in range(n):
        target = total * (i + 0.5) / n
        for s, e in spans:
            if target <= e - s:
                times.append(s + target)
                break
            target -= e - s
    return times


def frames_at(video_path, times, width=None):
    """JPEG bytes of the frame at each timestamp, at ``width`` px wide."""
    import cv2

    width = width or WIDTH
    cap = cv2.VideoCapture(video_path)
    out = []
    try:
        for t in times:
            cap.set(cv2.CAP_PROP_POS_MSEC, float(t) * 1000.0)
            ok, frame = cap.read()
            if not ok:
                continue
            h, w = frame.shape[:2]
            scaled = cv2.resize(frame, (width, max(2, int(h * width / w))),
                                interpolation=cv2.INTER_AREA)
            ok, buf = cv2.imencode(".jpg", scaled, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if ok:
                out.append(buf.tobytes())
    finally:
        cap.release()
    return out


def _ask_gemini(frames, prompt, api_key):
    """One vision call; returns the parsed dict. Split out so tests can stub it."""
    import llm_backend
    import gemini_worker

    if llm_backend.active():
        answer, _ = llm_backend.generate_json(
            prompt, gemini_worker.GroundedHook, images=frames)
        return answer

    from google import genai
    from google.genai import types as genai_types

    client = genai.Client(api_key=api_key)
    model_name = os.environ.get("GEMINI_MODEL") or "gemini-3.1-flash-lite"
    parts = [genai_types.Part.from_bytes(data=b, mime_type="image/jpeg") for b in frames]
    response = client.models.generate_content(
        model=model_name,
        contents=parts + [prompt],
        config=genai_types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=gemini_worker.GroundedHook,
        ))
    gemini_worker.raise_if_blocked(response)
    return json.loads(response.text) or {}


def reground(clip_path, clip, transcript, start, end) -> Optional[dict]:
    """Rewrite ``clip['viral_hook_text']`` / ``video_title_for_youtube_short``
    in place from the clip's frames. Returns what changed (also stored under
    ``clip['hook_grounding']``), or None when skipped or failed."""
    import llm_backend

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key and not llm_backend.active():
        print("   🪝 Hook grounding skipped: needs a Gemini key (frames), "
              "keeping the transcript hook.")
        return None
    try:
        import gemini_worker

        times = sample_times(clip.get("layout_ranges"), float(end) - float(start))
        frames = frames_at(clip_path, times)
        if not frames:
            return None
        language = str((transcript or {}).get("language") or "unknown")
        prompt = gemini_worker.GROUNDED_HOOK_PROMPT.format(
            language=language,
            current_hook=clip.get("viral_hook_text") or "",
            current_title=clip.get("video_title_for_youtube_short") or "",
            transcript=clip_words(transcript, start, end)[:4000] or "(no speech)")
        answer = _ask_gemini(frames, prompt, api_key)
        hook = str(answer.get("viral_hook_text") or "").strip()
        title = str(answer.get("video_title_for_youtube_short") or "").strip()
        if not hook:
            return None
        before = {"viral_hook_text": clip.get("viral_hook_text"),
                  "video_title_for_youtube_short": clip.get("video_title_for_youtube_short")}
        clip["viral_hook_text"] = hook
        if title:
            clip["video_title_for_youtube_short"] = title[:100]
        clip["hook_grounding"] = {
            "on_screen": str(answer.get("on_screen") or "")[:200],
            "before": before,
            "frames": len(frames),
        }
        print(f"   🪝 Hook regrounded on screen ({clip['hook_grounding']['on_screen'][:60]}): {hook}")
        return clip["hook_grounding"]
    except Exception as e:
        print(f"   ⚠️ Hook grounding failed ({type(e).__name__}: {e}) — keeping the transcript hook.")
        return None
