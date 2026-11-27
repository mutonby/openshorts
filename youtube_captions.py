"""YouTube caption fetching shared by the public transcript tool and the job
pipeline.

Deliberately free of FastAPI and cloud imports: ``main.py`` imports this
module for the two-pass ASR mode. The public tool (``free_tools.py``) keeps
the rate limits and the 24 h cache around these functions.
"""
from __future__ import annotations

import json
import os
import random
import re
import tempfile
from typing import Optional
from urllib.parse import parse_qs, urlparse

_VIDEO_ID = re.compile(r"^[A-Za-z0-9_-]{11}$")


def extract_video_id(raw: str) -> Optional[str]:
    """The 11-character id of one YouTube video, or None.

    Accepts watch / youtu.be / shorts / live / embed links and a bare id. The
    request that goes out is always rebuilt from the id, so nothing the user
    typed (a playlist, another host, a redirect) ever reaches yt-dlp.
    """
    s = (raw or "").strip()
    if _VIDEO_ID.match(s):
        return s
    if not re.match(r"^https?://", s, re.I):
        s = "https://" + s
    try:
        u = urlparse(s)
    except ValueError:
        return None
    host = (u.hostname or "").lower()
    if host == "youtu.be":
        cand = u.path.strip("/").split("/")[0]
        return cand if _VIDEO_ID.match(cand) else None
    if not (host == "youtube.com" or host.endswith(".youtube.com")
            or host == "youtube-nocookie.com" or host.endswith(".youtube-nocookie.com")):
        return None
    parts = [p for p in u.path.split("/") if p]
    if parts and parts[0] in ("watch", "watch_popup"):
        cand = (parse_qs(u.query).get("v") or [""])[0]
        return cand if _VIDEO_ID.match(cand) else None
    if len(parts) >= 2 and parts[0] in ("shorts", "live", "embed", "v", "e"):
        return parts[1] if _VIDEO_ID.match(parts[1]) else None
    return None


def _base_lang(code: str) -> str:
    return (code or "").split("-")[0].lower()


def choose_track(info: dict, lang: Optional[str] = None):
    """Pick the caption track to return: ``(kind, key, languages)``.

    kind is "manual" or "auto" (YouTube's speech recognition), key the entry
    in ``subtitles`` / ``automatic_captions``, languages what the user can
    switch to. Machine translations of the auto track (every other key in
    ``automatic_captions``) are never offered: they are not a transcript.
    Returns ``(None, None, languages)`` when the video has no captions.
    """
    manual = {k: v for k, v in (info.get("subtitles") or {}).items()
              if k != "live_chat" and v}
    auto = info.get("automatic_captions") or {}
    video_lang = info.get("language") or ""
    orig = [k for k in auto if k.endswith("-orig")]
    # Some videos list the ASR track under the plain language code only.
    if not orig and video_lang and video_lang in auto:
        orig = [video_lang]

    # ``request`` is what the page sends back as ``lang`` to switch track:
    # the plain code for a human track, ``<code>-orig`` for YouTube's ASR one.
    languages = [{"code": k, "kind": "manual", "request": k,
                  "name": _track_name(manual[k], k)} for k in manual]
    for k in orig:
        code = k[:-5] if k.endswith("-orig") else k
        languages.append({"code": code, "kind": "auto", "request": f"{code}-orig",
                          "name": _track_name(auto[k], code) + " (auto-generated)"})

    video_base = _base_lang(video_lang)
    languages.sort(key=lambda l: (_base_lang(l["code"]) != video_base,
                                  _base_lang(l["code"]) != "en", l["kind"] != "manual",
                                  l["name"].lower()))

    def auto_key(code):
        for k in orig:
            if k == f"{code}-orig" or k == code:
                return k
        return None

    if lang and lang.endswith("-orig"):
        k = auto_key(lang[:-5])
        if k:
            return "auto", k, languages
        lang = lang[:-5]
    if lang:
        if lang in manual:
            return "manual", lang, languages
        k = auto_key(lang)
        if k:
            return "auto", k, languages
    # Default: a human track in the spoken language, then the auto track in
    # the spoken language, then any human track (English first), then any auto.
    if video_lang:
        for k in manual:
            if k == video_lang or _base_lang(k) == _base_lang(video_lang):
                return "manual", k, languages
    if orig:
        return "auto", orig[0], languages
    if manual:
        for k in manual:
            if _base_lang(k) == "en":
                return "manual", k, languages
        return "manual", next(iter(manual)), languages
    return None, None, languages


def _track_name(formats, code):
    for f in formats or []:
        if f.get("name"):
            return re.sub(r"\s*\(.*?\)\s*$", "", f["name"]).strip() or code
    return code


def parse_json3(data) -> list:
    """YouTube json3 captions -> [{start, dur, text}] in seconds.

    Events without segments are window/style markers; ``\\n``-only events are
    the line breaks of the rolling auto captions. Both are dropped.
    """
    if isinstance(data, (bytes, str)):
        data = json.loads(data)
    out = []
    for ev in data.get("events") or []:
        segs = ev.get("segs")
        if not segs:
            continue
        text = "".join(s.get("utf8", "") for s in segs)
        text = re.sub(r"\s+", " ", text.replace("\n", " ")).strip()
        if not text:
            continue
        start = (ev.get("tStartMs") or 0) / 1000.0
        dur = (ev.get("dDurationMs") or 0) / 1000.0
        out.append({"start": round(start, 3), "dur": round(dur, 3), "text": text})
    # Auto captions overlap (each line stays on screen until the next one
    # ends); clamp so exported SRT cues never collide.
    for a, b in zip(out, out[1:]):
        if a["start"] + a["dur"] > b["start"]:
            a["dur"] = round(max(0.0, b["start"] - a["start"]), 3)
    return out


def is_degraded(info: dict) -> bool:
    """True for a watch page YouTube served without its player response.

    Keyed on the duration, not on the formats: a finished live stream answers
    with a duration and zero formats, and it genuinely has no captions (that
    one used to be reported as "YouTube is blocking us").
    """
    return not info.get("formats") and not info.get("duration")


class TranscriptError(Exception):
    def __init__(self, code: str, message: str, status: int):
        super().__init__(message)
        self.code, self.message, self.status = code, message, status


_UNAVAILABLE = re.compile(
    r"private video|video unavailable|has been removed|not available|members-only|"
    r"confirm your age|age-restricted|copyright|terminated|does not exist|premieres in",
    re.I)
_BLOCKED = re.compile(r"not a bot|sign in to confirm|429|too many requests|403|"
                      r"unable to download|timed out|proxy|connection", re.I)


def transcript_routes():
    """Proxies to try, in order. Statics only (rotated); direct when there are
    none. Never the per-GB PROXY_URL."""
    statics = [p.strip() for p in os.environ.get("STATIC_PROXY_URLS", "").split(",")
               if p.strip()]
    if statics:
        k = random.randrange(len(statics))
        statics = statics[k:] + statics[:k]
        routes = statics[:3]
        if os.environ.get("DIRECT_FIRST", "").strip() == "1":
            routes = [None] + routes
        return routes
    return [None]


class _QuietLogger:
    def debug(self, msg): pass
    def info(self, msg): pass
    def warning(self, msg): pass
    def error(self, msg): pass


def fetch_transcript(video_id: str, lang: Optional[str] = None) -> dict:
    """Blocking: resolve the caption track and download it. Raises
    TranscriptError with a user-facing message."""
    import yt_dlp

    url = f"https://www.youtube.com/watch?v={video_id}"
    cookies_env = os.environ.get("YOUTUBE_COOKIES")
    ck_path = None
    errors = []
    try:
        if cookies_env:
            fd, ck_path = tempfile.mkstemp(prefix="tool_ck_", suffix=".txt")
            with os.fdopen(fd, "w") as f:
                f.write(cookies_env)
        for proxy in transcript_routes():
            for use_cookies in ([False, True] if ck_path else [False]):
                opts = {"skip_download": True, "quiet": True, "no_warnings": True,
                        "noplaylist": True, "logger": _QuietLogger(),
                        "socket_timeout": 15,
                        # Captions only: without this a video whose formats are
                        # SABR-only fails with "Requested format is not available"
                        # although its caption list came back fine.
                        "ignore_no_formats_error": True,
                        "extractor_args": {"youtube": {"player_client": ["default"]}}}
                if proxy:
                    opts["proxy"] = proxy
                if use_cookies:
                    opts["cookiefile"] = ck_path
                try:
                    with yt_dlp.YoutubeDL(opts) as ydl:
                        info = ydl.extract_info(url, download=False)
                        kind, key, languages = choose_track(info, lang)
                        meta = {
                            "video_id": video_id,
                            "title": info.get("title") or "",
                            "channel": info.get("channel") or info.get("uploader") or "",
                            "duration": info.get("duration") or 0,
                            "languages": languages,
                        }
                        if not kind and is_degraded(info):
                            # A degraded answer: anonymous requests from two
                            # of the three statics got the watch page without
                            # its player response (title, but no duration, no
                            # formats, no captions) for a TED talk with 49
                            # human tracks (23-sep-2026); the same routes with
                            # the cookies answered in full. That is the route,
                            # not the video: try the next attempt.
                            errors.append("empty player response")
                            continue
                        if not kind:
                            raise TranscriptError(
                                "no_captions",
                                "This video has no captions on YouTube, not even automatic ones.",
                                404)
                        pool = (info.get("subtitles") if kind == "manual"
                                else info.get("automatic_captions")) or {}
                        fmt = next((f for f in pool.get(key, []) if f.get("ext") == "json3"), None)
                        if not fmt:
                            raise TranscriptError(
                                "no_captions",
                                "YouTube lists captions for this video but does not serve them in a readable format.",
                                404)
                        raw = ydl.urlopen(fmt["url"]).read()
                    segments = parse_json3(raw)
                    if not segments:
                        raise TranscriptError(
                            "no_captions", "The caption track for this video is empty.", 404)
                    code = key[:-5] if key.endswith("-orig") else key
                    return {**meta, "language": code, "kind": kind, "segments": segments}
                except TranscriptError:
                    raise
                except Exception as e:  # noqa: BLE001 — classified below
                    msg = str(e)
                    errors.append(msg[:300])
                    if _UNAVAILABLE.search(msg) and not re.search(r"not a bot", msg, re.I):
                        raise TranscriptError(
                            "unavailable",
                            "YouTube says this video is private, removed, age-restricted or not "
                            "available, so its captions cannot be read.", 422)
        raise TranscriptError(
            "youtube_blocked",
            "YouTube is not answering our servers right now. Try again in a minute, or open "
            "the video on YouTube and use \"Show transcript\" under the description.", 503)
    finally:
        if ck_path:
            try:
                os.remove(ck_path)
            except OSError:
                pass
