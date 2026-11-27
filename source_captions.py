"""Read the captions a source already has for the two-pass ASR mode.

The clip pipeline normally transcribes the whole source. With
``ASR_CLIP_WORDS`` it only needs a coarse transcript for moment picking, so a
YouTube link can reuse the captions YouTube already serves. ``free_tools``
owns the caption fetching (static proxies, cookies, json3 parsing); this
module keeps the pipeline side of it small, importable without the ML stack,
and testable.

Kept out of ``free_tools`` itself because that module is the public web tool:
this is the job-side adapter, with one policy of its own — a requested
language that YouTube cannot supply must fall back to ASR, never to a
different-language track.
"""
from __future__ import annotations

import os
import re

_YOUTUBE_HOST = re.compile(
    r"(?:^|//|\.)(?:youtube\.com|youtu\.be|youtube-nocookie\.com)(?:/|$)", re.I)


def is_youtube_url(url) -> bool:
    """True for a YouTube watch/shorts/live link (the caption adapter only
    handles those). Used to keep early-audio ASR enabled for other sites."""
    return bool(_YOUTUBE_HOST.search(url or ""))


def source_captions_transcript(url, lang=None):
    """Return a coarse transcript from the source's captions, or ``None``.

    ``lang`` wins over ``SOURCE_CAPTIONS_LANG``; both are plain codes ("ru")
    or YouTube's ``<code>-orig`` form. When a language was requested and the
    track served is a different one, returns ``None`` so the caller can
    transcribe the audio instead of feeding the wrong language to the model.
    Any network/parsing failure also returns ``None``; the caller falls back.
    """
    try:
        import youtube_captions as captions
        video_id = captions.extract_video_id(url or "")
        if not video_id:
            return None
        wanted = (lang or os.environ.get("SOURCE_CAPTIONS_LANG") or "").strip() or None
        data = captions.fetch_transcript(video_id, wanted)
        served = (data.get("language") or "").split("-")[0].lower()
        if wanted and served != wanted.split("-")[0].lower():
            print(f"⚠️ Source captions are '{data.get('language')}', wanted "
                  f"'{wanted}' — using ASR instead of the wrong-language track.")
            return None
        segments = []
        for item in data.get("segments") or []:
            text = str(item.get("text") or "").strip()
            if not text:
                continue
            start = float(item.get("start", 0))
            segments.append({"start": start,
                             "end": start + float(item.get("dur", 0)),
                             "text": text})
        if not segments:
            return None
        print(f"⏩ Using YouTube captions ({data.get('language') or 'unknown'}, "
              f"{len(segments)} segments) — skipping source ASR.")
        return {
            "language": data.get("language") or "unknown",
            "text": " ".join(segment["text"] for segment in segments),
            "segments": segments,
        }
    except Exception as e:  # noqa: BLE001 — any caption failure means ASR
        print(f"⚠️ Source captions unavailable ({type(e).__name__}: {e}) — "
              f"transcribing the source.")
        return None
