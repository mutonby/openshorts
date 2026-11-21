"""Fast transcript path for Plic AI.

Uses YouTube's own original-language automatic captions as a timestamped transcript.
The pipeline still runs local ASR on the selected video ranges before rendering, so
final burned captions use fresh word timings rather than relying on approximate
YouTube caption timings.
"""
from __future__ import annotations

import re
from urllib.parse import urlparse


def _is_youtube(url: str) -> bool:
    host = (urlparse(url).hostname or "").lower()
    return host.endswith(("youtube.com", "youtu.be", "youtube-nocookie.com"))


def _caption_words(data: dict) -> list[dict]:
    words = []
    for event in data.get("events") or []:
        segs = [s for s in event.get("segs") or [] if (s.get("utf8") or "").strip()]
        if not segs:
            continue
        start_ms = float(event.get("tStartMs") or 0)
        duration_ms = float(event.get("dDurationMs") or 0)
        timed = len(segs) > 1 or "tOffsetMs" in segs[0]
        for seg in segs:
            parts = (seg.get("utf8") or "").split()
            for index, token in enumerate(parts):
                if re.fullmatch(r"[\[(].*[\])]", token) or token in ("[", "]", "__"):
                    continue
                offset = float(seg.get("tOffsetMs") or 0) if timed else duration_ms * index / max(1, len(parts))
                at = (start_ms + offset) / 1000.0
                words.append({"word": token, "start": round(at, 2)})
    words.sort(key=lambda item: item["start"])
    for index, word in enumerate(words):
        nxt = words[index + 1]["start"] if index + 1 < len(words) else word["start"] + 0.4
        word["end"] = round(max(word["start"] + 0.08, min(nxt, word["start"] + 0.65)), 2)
    return words


def _segments(words: list[dict]) -> list[dict]:
    result = []
    current = []

    def flush():
        if not current:
            return
        result.append({
            "start": current[0]["start"],
            "end": current[-1]["end"],
            "text": " ".join(w["word"].strip() for w in current),
            "words": [dict(w) for w in current],
        })
        current.clear()

    for word in words:
        if current and (word["start"] - current[-1]["end"] > 1.8 or len(current) >= 28):
            flush()
        current.append(word)
    flush()
    return result


def fetch(url: str, duration: float | None = None) -> dict | None:
    """Return a faster-whisper-shaped transcript, or None when no useful track exists."""
    if not _is_youtube(url):
        return None

    import json
    import yt_dlp
    import ssrf_guard

    options = {
        "quiet": True,
        "no_warnings": True,
        "noplaylist": True,
        "skip_download": True,
    }
    with ssrf_guard.guarded():
        with yt_dlp.YoutubeDL(options) as ydl:
            info = ydl.extract_info(url, download=False)
            automatic = info.get("automatic_captions") or {}
            spoken = str(info.get("language") or "").strip().lower()
            base = spoken.split("-")[0] if spoken else ""
            candidates = [f"{spoken}-orig", spoken, f"{base}-orig", base]
            key = next((candidate for candidate in candidates if candidate and candidate in automatic), None)
            if not key:
                key = next((name for name in automatic if name.endswith("-orig")), None)
            if not key:
                return None
            track = next(
                (item for item in automatic.get(key) or []
                 if item.get("ext") == "json3" and item.get("url")),
                None,
            )
            if not track:
                return None
            raw = ydl.urlopen(track["url"]).read(40_000_000)

    words = _caption_words(json.loads(raw))
    if len(words) < 40:
        return None
    if duration and duration > 0:
        # Reject tracks that cover only the opening minutes of a long stream.
        if words[-1]["end"] < duration * 0.60:
            return None
        # Very sparse captions are not reliable enough for long-stream selection.
        if len(words) < max(40, int(duration / 60 * 1.5)):
            return None

    segments = _segments(words)
    if not segments:
        return None
    return {"language": key.replace("-orig", ""), "segments": segments}
