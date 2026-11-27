"""Assess proposed moments before rendering and keep the review on disk."""
import json
import os
from concurrent.futures import ThreadPoolExecutor

from clip_selection import trim_to_best
from jsonfile import write_json_atomic


PLAN_FILE = ".clip-review.json"
STATUS = "review"
MAX_CANDIDATES = 15
RATINGS = ("hook", "visual_action", "reaction", "clarity", "payoff")


def text(value, limit):
    return " ".join(value.split())[:limit] if isinstance(value, str) else ""


def _valid_span(item, duration):
    if not isinstance(item, dict):
        return False
    start, end = item.get("start"), item.get("end")
    return (type(start) in (int, float) and type(end) in (int, float)
            and 0 <= start < end <= duration)


def read(directory):
    try:
        with open(os.path.join(directory, PLAN_FILE)) as f:
            data = json.load(f)
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict):
        return None
    source, duration, shorts = (data.get(key) for key in ("source_video", "source_duration", "shorts"))
    if (not isinstance(source, str) or not source or os.path.basename(source) != source
            or type(duration) not in (int, float) or not 0 < duration < float("inf")
            or not isinstance(shorts, list) or len(shorts) > MAX_CANDIDATES
            or not all(_valid_span(clip, duration) for clip in shorts)):
        return None
    # An audio-first review names each candidate's range files by its slot
    # (livestream.preview_file / original_file): a small integer, nothing else.
    if data.get("deferred") and not all(
            type(clip.get("preview_slot")) is int and 1 <= clip["preview_slot"] <= MAX_CANDIDATES
            for clip in shorts):
        return None
    return data


def write(directory, data):
    write_json_atomic(os.path.join(directory, PLAN_FILE), data)


def _preview_offset(clip):
    origin = clip.get("preview_origin")
    if type(origin) not in (int, float) or not 0 <= origin <= clip["start"]:
        return 0
    return round(clip["start"] - origin, 3)


def _gameplay_like(method):
    import framing
    return framing.gameplay_like(method) if isinstance(method, str) else False


def public_result(data, *, directory=None):
    """Expose candidate descriptions and timestamps without the full transcript."""
    from livestream import preview_file

    deferred = bool(data.get("deferred"))

    def preview_available(clip):
        slot = clip.get("preview_slot")
        return (deferred and type(slot) is int and 1 <= slot <= MAX_CANDIDATES
                and (directory is None or preview_file(directory, slot).is_file()))

    return {
        "candidates": [
            {"id": i, "start": clip["start"], "end": clip["end"],
             "title": text(clip.get("video_title_for_youtube_short"), 100),
             "assessment": clip.get("assessment") or {},
             "signals": clip.get("supporting_signals") or {},
             "hype": clip.get("hype") if isinstance(clip.get("hype"), dict) else None,
             "preview": preview_available(clip),
             # A judged clip can start after the window its preview file
             # begins at: the page seeks the preview by this much.
             "preview_offset": _preview_offset(clip)}
            for i, clip in enumerate(data["shorts"], start=1)
        ],
        "source_duration": data["source_duration"],
        "source_offset": data.get("source_offset", 0),
        "deferred": deferred,
        # A game framing gets a still of the vertical composition per
        # candidate (/api/jobs/{id}/candidate-layout/{n}).
        "layout_preview": bool(_gameplay_like(data.get("framing_method"))
                               and data.get("output_format") != "horizontal"),
        "signals": data.get("signals"),
        "engine": "hype" if data.get("hype") is True else "transcript",
        "performance": data.get("performance"),
        "analysis_usage": data.get("analysis_usage"),
        "cost_analysis": data.get("cost_analysis"),
        "transcript_segments": len((data.get("transcript") or {}).get("segments") or []),
    }


def select(data, ids):
    if (not ids or len(set(ids)) != len(ids)
            or any(type(i) is not int or i < 1 or i > len(data["shorts"]) for i in ids)):
        raise ValueError("Select distinct candidate numbers from this review.")
    return {**data, "shorts": [data["shorts"][i - 1] for i in sorted(ids)]}


def _assessment(video, clip, transcript, api_key, frame_origin=0.0):
    import gemini_worker
    import hook_grounding

    start, end = clip["start"], clip["end"]
    times = [start + (end - start) * share for share in (0.06, 0.28, 0.52, 0.76, 0.94)]
    frames = hook_grounding.frames_at(video, [time - frame_origin for time in times], width=640)
    if len(frames) < 3:
        return None
    prompt = gemini_worker.MOMENT_REVIEW_PROMPT.format(
        language=text((transcript or {}).get("language"), 32) or "the source language",
        transcript=hook_grounding.clip_words(transcript, start, end)[:4000] or "(no speech)")
    raw = hook_grounding._ask_gemini(frames, prompt, api_key, schema=gemini_worker.MomentReview)
    return gemini_worker.MomentReview.model_validate(raw)


def assess_clip(video, clip, transcript, api_key, gameplay=False, frame_origin=0.0):
    """Use a single event description across platforms. Unclear moments go to review."""
    answer = None
    if api_key:
        try:
            answer = (_assessment(video, clip, transcript, api_key, frame_origin)
                      if frame_origin else _assessment(video, clip, transcript, api_key))
        except Exception as exc:
            # The provider response could contain internal details or credentials.
            print(f"   ⚠️ Could not check a candidate's source frames ({type(exc).__name__}).")

    event = text(answer.event, 240) if answer else ""
    payoff = text(answer.outcome, 160) if answer else ""
    ratings = ({key: getattr(answer, key) for key in RATINGS} if answer else {})
    verdict = "review"
    if answer and answer.decision == "weak":
        verdict = "weak"
    elif (answer and answer.decision == "keep" and answer.evidence != "uncertain"
          and event and payoff and ratings["payoff"] != "weak" and ratings["clarity"] != "weak"):
        verdict = "supported"
    # With uncertain evidence even the draft title can overpromise. Give it a
    # neutral title and leave the opening overlay empty for the editor.
    neutral = "Gameplay moment" if gameplay else "Moment from the video"
    title = (text(answer.title, 100) or event[:100]) if verdict == "supported" else neutral
    summary = event if verdict == "supported" else neutral
    excerpt = ""
    if not event:
        from hook_grounding import clip_words
        excerpt = text(clip_words(transcript, clip["start"], clip["end"]), 200)
    assessed = {
        **clip,
        "video_title_for_youtube_short": title,
        "video_description_for_tiktok": summary,
        "video_description_for_instagram": summary,
        "viral_hook_text": text(answer.opening_title, 80) if verdict == "supported" else "",
        "why": payoff if verdict == "supported" else "",
        "assessment": {"verdict": verdict, "event": event, "payoff": payoff, "ratings": ratings,
                       "excerpt": excerpt,
                       "evidence": answer.evidence if answer else "uncertain",
                       "source_captions": bool(answer and answer.source_captions)},
    }
    return assessed


def shortlist(shorts, duration):
    return trim_to_best([clip for clip in shorts if _valid_span(clip, duration)], MAX_CANDIDATES)


def rank(clips):
    """Use explicit payoff and evidence first. Supporting cues break editorial ties."""
    levels = {"weak": 0, "clear": 1, "strong": 2}

    def key(clip):
        analysis = clip["assessment"]
        ratings = analysis["ratings"]
        def level(name):
            return -levels.get(ratings.get(name), 0)
        return (analysis["verdict"] != "supported", level("payoff"), level("clarity"),
                analysis["evidence"] != "visible", level("visual_action"), level("hook"),
                level("reaction"), -len(clip.get("supporting_signals") or {}))

    return sorted(clips, key=key)


def assess(video, shorts, transcript, duration, *, gameplay, manual_review, signals=None):
    """Limit work to the shortlist and avoid filling an empty quota with filler."""
    import highlight_signals

    items = shortlist(shorts, duration)
    if not items:
        return []
    print(f"   Checking source evidence for {len(items)} proposed moment(s)…")
    key = os.environ.get("GEMINI_API_KEY")
    with ThreadPoolExecutor(max_workers=min(2, len(items))) as pool:
        assessed = list(pool.map(lambda clip: assess_clip(video, clip, transcript, key, gameplay), items))
    for clip in assessed:
        clip["supporting_signals"] = highlight_signals.related(clip["start"], clip["end"], signals)
    kept = [clip for clip in assessed if clip["assessment"]["verdict"] == "supported"
            or (manual_review and clip["assessment"]["verdict"] == "review")]
    return rank(kept)
