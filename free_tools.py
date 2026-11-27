"""Free public tools behind the static SEO pages (/tools on openshorts.app).

Three endpoints, all anonymous, all deliberately cheap:

* ``GET /api/tools/youtube-transcript`` reads the captions a video **already
  has** on YouTube (manual subtitles first, then YouTube's own automatic
  track). It never downloads media and never touches the GPU: no Parakeet, no
  render. A video without captions gets an honest "no captions" answer and the
  page points at the clip generator, which does transcribe.
* ``GET /api/tools/tiktok-transcript`` does the same for a TikTok video: the
  WebVTT captions TikTok's own player shows, read through yt-dlp inside
  ``ssrf_guard.guarded()`` (a share link redirects), direct then statics.
* ``POST /api/tools/youtube-metadata`` writes titles, a description or tags
  for a video from a short brief, with one Gemini text call.

Why yt-dlp and the static proxies. YouTube answers every anonymous request
from a datacenter IP with "Sign in to confirm you're not a bot" (measured from
the prod container on 23-sep-2026: direct = bot check in 0.8 s; the static ISP
pool = full caption list in 1.1 s). So the transcript goes out through the same
flat-rate ``STATIC_PROXY_URLS`` the download uses, anonymous first and with the
account cookies only as the second attempt on a route. It **never** uses the
per-GB ``PROXY_URL``: a free tool must not be able to run up a bandwidth bill.
With no statics configured (self-host) it goes direct.

Cost and abuse guards, all in-process (a restart resets them, which is fine
for limits this coarse): per-IP windows, a global daily cap per tool, a
semaphore on concurrent YouTube lookups, a hard timeout, and a 24 h cache so a
popular video is fetched once.
"""
from __future__ import annotations

import asyncio
import json
import os
import random
import re
import tempfile
import threading
import time
from collections import OrderedDict, defaultdict, deque
from typing import Optional
from urllib.parse import parse_qs, urlparse

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel
from youtube_captions import (TranscriptError, _QuietLogger, _base_lang,
                              _track_name, choose_track, extract_video_id,
                              fetch_transcript, is_degraded, parse_json3,
                              transcript_routes)

router = APIRouter()

# --------------------------------------------------------------------------- #
# Rate limiting
# --------------------------------------------------------------------------- #
# (window seconds, max hits) per IP, plus one global cap per UTC day.
LIMITS = {
    "transcript": {"per_ip": [(600, 10), (86400, 60)], "global_day": 1500},
    "tiktok_transcript": {"per_ip": [(600, 10), (86400, 60)], "global_day": 1500},
    "metadata": {"per_ip": [(600, 12), (86400, 50)], "global_day": 2000},
}


class RateLimiter:
    """Sliding windows per (tool, ip) and a fixed UTC-day counter per tool."""

    def __init__(self, limits, clock=time.time):
        self.limits = limits
        self.clock = clock
        self._hits = defaultdict(deque)
        self._day = {}
        self._lock = threading.Lock()

    def check(self, tool: str, ip: str) -> Optional[str]:
        """Record a hit and return None, or return the reason it is refused
        (nothing is recorded then)."""
        cfg = self.limits[tool]
        now = self.clock()
        day = int(now // 86400)
        with self._lock:
            d, n = self._day.get(tool, (day, 0))
            if d != day:
                d, n = day, 0
            if n >= cfg["global_day"]:
                return "global"
            hits = self._hits[(tool, ip)]
            longest = max(w for w, _ in cfg["per_ip"])
            while hits and now - hits[0] > longest:
                hits.popleft()
            for window, cap in cfg["per_ip"]:
                if sum(1 for t in hits if now - t <= window) >= cap:
                    return "ip"
            hits.append(now)
            self._day[tool] = (d, n + 1)
            # Keep the map from growing without bound on a long-lived process.
            if len(self._hits) > 20000:
                for key in [k for k, v in self._hits.items() if not v or now - v[-1] > longest]:
                    del self._hits[key]
        return None


_limiter = RateLimiter(LIMITS)


def client_ip(request: Request) -> str:
    # Traefik overwrites X-Forwarded-For from untrusted clients and uvicorn
    # resolves it into request.client, so this is the real peer address.
    return (request.client.host if request.client else "") or "unknown"


def _enforce(tool: str, request: Request):
    reason = _limiter.check(tool, client_ip(request))
    if reason == "ip":
        raise HTTPException(status_code=429, detail={
            "code": "rate_limited",
            "message": "Too many requests from your network. Try again in a few minutes."})
    if reason == "global":
        raise HTTPException(status_code=429, detail={
            "code": "busy",
            "message": "This free tool hit its daily limit. It resets at midnight UTC."})


# --------------------------------------------------------------------------- #
# YouTube transcript
# --------------------------------------------------------------------------- #
_TRANSCRIPT_CACHE: "OrderedDict[tuple, tuple]" = OrderedDict()
_CACHE_TTL = 24 * 3600
_CACHE_MAX = 400
_cache_lock = threading.Lock()
_yt_slots = threading.BoundedSemaphore(int(os.environ.get("FREE_TOOLS_YT_CONCURRENCY", "3")))
TRANSCRIPT_TIMEOUT = float(os.environ.get("FREE_TOOLS_YT_TIMEOUT", "45"))


def _cache_get(key):
    with _cache_lock:
        hit = _TRANSCRIPT_CACHE.get(key)
        if not hit:
            return None
        ts, value = hit
        if time.time() - ts > _CACHE_TTL:
            del _TRANSCRIPT_CACHE[key]
            return None
        _TRANSCRIPT_CACHE.move_to_end(key)
        return value


def _cache_put(key, value):
    with _cache_lock:
        _TRANSCRIPT_CACHE[key] = (time.time(), value)
        _TRANSCRIPT_CACHE.move_to_end(key)
        while len(_TRANSCRIPT_CACHE) > _CACHE_MAX:
            _TRANSCRIPT_CACHE.popitem(last=False)


def _fetch_with_slot(video_id, lang):
    if not _yt_slots.acquire(timeout=20):
        raise TranscriptError("busy", "The transcript tool is busy. Try again in a few seconds.", 503)
    try:
        return fetch_transcript(video_id, lang)
    finally:
        _yt_slots.release()


@router.get("/api/tools/youtube-transcript")
async def youtube_transcript(request: Request, url: str = "", lang: str = ""):
    video_id = extract_video_id(url)
    if not video_id:
        raise HTTPException(status_code=400, detail={
            "code": "bad_url",
            "message": "Paste the link of one YouTube video (youtube.com/watch?v=..., youtu.be/... or a Shorts link)."})
    lang = lang.strip() if re.match(r"^[A-Za-z0-9-]{1,20}$", lang.strip() or "") else None
    key = (video_id, lang or "")
    cached = _cache_get(key)
    if cached and "error" in cached:
        e = cached["error"]
        raise HTTPException(status_code=e["status"], detail={"code": e["code"], "message": e["message"]})
    if cached:
        return {**cached, "cached": True}
    _enforce("transcript", request)
    try:
        result = await asyncio.wait_for(
            asyncio.to_thread(_fetch_with_slot, video_id, lang), timeout=TRANSCRIPT_TIMEOUT)
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail={
            "code": "timeout", "message": "YouTube took too long to answer. Try again."})
    except TranscriptError as e:
        if e.code in ("no_captions", "unavailable"):
            # Cache the verdict about the video itself (not a blocked route):
            # it will not change within the day, and a retry loop on it would
            # spend the static pool for nothing.
            _cache_put(key, {"error": {"code": e.code, "message": e.message, "status": e.status}})
        raise HTTPException(status_code=e.status, detail={"code": e.code, "message": e.message})
    _cache_put(key, result)
    if not lang:
        req = result["language"] + ("-orig" if result["kind"] == "auto" else "")
        _cache_put((video_id, req), result)
    return {**result, "cached": False}


# --------------------------------------------------------------------------- #
# TikTok transcript
# --------------------------------------------------------------------------- #
# TikTok's web player ships the captions it shows (its own speech recognition,
# or the creator's edited version) as WebVTT next to the video; yt-dlp lists
# them under ``subtitles`` once subtitle extraction is switched on. Same
# promise as the YouTube tool: read what is already there, never media, never
# the GPU, never the per-GB PROXY_URL. TikTok does not bot-check a plain
# request the way YouTube does, so the route order is the download's own:
# direct first, then up to two statics.

_TIKTOK_ID = re.compile(r"^\d{15,22}$")
_TIKTOK_SHORT = re.compile(r"^[A-Za-z0-9]{5,20}$")
# Only TikTok's own extractors may run: a short link that redirects anywhere
# else fails as "no suitable extractor" instead of reaching the generic one.
_TIKTOK_EXTRACTORS = ["TikTok", "vm.tiktok"]


def extract_tiktok_target(raw: str):
    """``(cache_key, url)`` for one TikTok video, or None.

    The URL handed to yt-dlp is rebuilt from the numeric id (the username is
    not needed and is dropped) or, for a share link from the app, from the
    short code on TikTok's own short-link host.
    """
    s = (raw or "").strip()
    if _TIKTOK_ID.match(s):
        return s, f"https://www.tiktok.com/@_/video/{s}"
    if not re.match(r"^https?://", s, re.I):
        s = "https://" + s
    try:
        u = urlparse(s)
    except ValueError:
        return None
    host = (u.hostname or "").lower()
    parts = [p for p in u.path.split("/") if p]
    if host in ("vm.tiktok.com", "vt.tiktok.com"):
        if len(parts) == 1 and _TIKTOK_SHORT.match(parts[0]):
            return f"{host[:2]}:{parts[0]}", f"https://{host}/{parts[0]}/"
        return None
    if host not in ("tiktok.com", "www.tiktok.com", "m.tiktok.com"):
        return None
    if len(parts) == 2 and parts[0] == "t" and _TIKTOK_SHORT.match(parts[1]):
        return f"t:{parts[1]}", f"https://www.tiktok.com/t/{parts[1]}"
    cand = None
    if len(parts) >= 3 and parts[0].startswith("@") and parts[1] == "video":
        cand = parts[2]
    elif len(parts) >= 3 and parts[0] == "embed" and parts[1] == "v2":
        cand = parts[2]
    elif len(parts) == 2 and parts[0] == "embed":
        cand = parts[1]
    elif len(parts) == 2 and parts[0] == "v" and parts[1].endswith(".html"):
        cand = parts[1][:-5]
    if cand and _TIKTOK_ID.match(cand):
        return cand, f"https://www.tiktok.com/@_/video/{cand}"
    return None


_CUE_TIME = re.compile(
    r"(?:(\d+):)?(\d{1,2}):(\d{2})[.,](\d{3})\s*-->\s*(?:(\d+):)?(\d{1,2}):(\d{2})[.,](\d{3})")


def _secs(h, m, s, ms):
    return int(h or 0) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000.0


def parse_timed_text(data) -> list:
    """WebVTT or SRT -> [{start, dur, text}] in seconds, the shape
    ``parse_json3`` returns. Cue ids, NOTE/STYLE blocks and tags are dropped."""
    if isinstance(data, bytes):
        data = data.decode("utf-8", "replace")
    out = []
    for block in re.split(r"\r?\n\s*\r?\n", (data or "").replace("﻿", "")):
        lines = [ln.strip() for ln in block.strip().splitlines()]
        for i, line in enumerate(lines):
            m = _CUE_TIME.search(line)
            if not m:
                continue
            start = _secs(*m.group(1, 2, 3, 4))
            end = _secs(*m.group(5, 6, 7, 8))
            text = " ".join(lines[i + 1:])
            text = re.sub(r"<[^>]+>", "", text)
            text = re.sub(r"\s+", " ", text).strip()
            if text:
                out.append({"start": round(start, 3), "dur": round(max(0.0, end - start), 3),
                            "text": text})
            break
    out.sort(key=lambda c: c["start"])
    for a, b in zip(out, out[1:]):
        if a["start"] + a["dur"] > b["start"]:
            a["dur"] = round(max(0.0, b["start"] - a["start"]), 3)
    return out


def _tiktok_lang_name(code: str) -> str:
    base = _base_lang(code)
    names = {"en": "English", "es": "Spanish", "pt": "Portuguese", "fr": "French",
             "de": "German", "it": "Italian", "id": "Indonesian", "ja": "Japanese",
             "ko": "Korean", "vi": "Vietnamese", "th": "Thai", "tr": "Turkish",
             "ru": "Russian", "ar": "Arabic", "nl": "Dutch", "pl": "Polish",
             "eng": "English", "spa": "Spanish", "por": "Portuguese", "fra": "French",
             "deu": "German", "ita": "Italian", "ind": "Indonesian", "jpn": "Japanese",
             "kor": "Korean", "vie": "Vietnamese", "tha": "Thai", "tur": "Turkish",
             "rus": "Russian", "ara": "Arabic", "nld": "Dutch", "pol": "Polish",
             "cmn": "Chinese", "zho": "Chinese", "hin": "Hindi"}
    name = names.get(base)
    return f"{name} ({code})" if name else code


def choose_tiktok_track(info: dict, lang: Optional[str] = None):
    """``(key, fmt, languages)`` for the TikTok caption track to return.

    TikTok lists the spoken language first; a requested language wins when it
    is there. Only WebVTT / SRT are readable (the json "creator_caption"
    variant is skipped). ``(None, None, [])`` when there are none.
    """
    readable = {}
    for k, fmts in (info.get("subtitles") or {}).items():
        for f in fmts or []:
            if f.get("ext") in ("vtt", "srt") and (f.get("url") or f.get("data")):
                readable.setdefault(k, f)
    languages = [{"code": k, "kind": "tiktok", "request": k, "name": _tiktok_lang_name(k)}
                 for k in readable]
    if not readable:
        return None, None, languages
    key = lang if lang in readable else next(iter(readable))
    return key, readable[key], languages


_TT_UNAVAILABLE = re.compile(
    r"private|not available|unavailable|removed|deleted|log in for access|"
    r"no suitable extractor|unsupported url|status code 10\d{3}|does not exist", re.I)
_TT_RETRY = re.compile(r"ip address is blocked|429|too many requests|403|timed out|"
                       r"proxy|connection|unable to download|unable to extract", re.I)


def tiktok_routes():
    """Direct, then up to two statics. Never the per-GB PROXY_URL."""
    statics = [p.strip() for p in os.environ.get("STATIC_PROXY_URLS", "").split(",")
               if p.strip()]
    if statics:
        k = random.randrange(len(statics))
        statics = statics[k:] + statics[:k]
    return [None] + statics[:2]


def fetch_tiktok_transcript(target_url: str, lang: Optional[str] = None) -> dict:
    """Blocking: read the captions TikTok already shows for one video.
    Raises TranscriptError with a user-facing message."""
    import yt_dlp
    import ssrf_guard

    errors = []
    for proxy in tiktok_routes():
        opts = {"skip_download": True, "quiet": True, "no_warnings": True,
                "noplaylist": True, "logger": _QuietLogger(), "socket_timeout": 15,
                "ignore_no_formats_error": True,
                # yt-dlp only fills ``subtitles`` for TikTok when asked to.
                # extract_info(download=False) writes nothing to disk.
                "writesubtitles": True,
                "allowed_extractors": _TIKTOK_EXTRACTORS}
        if proxy:
            opts["proxy"] = proxy
        try:
            # A short link redirects, and caption URLs come out of TikTok's
            # page: every connection is checked at resolve time.
            with ssrf_guard.guarded(), yt_dlp.YoutubeDL(opts) as ydl:
                info = ydl.extract_info(target_url, download=False)
                if info.get("_type") == "playlist" or not _TIKTOK_ID.match(str(info.get("id") or "")):
                    raise TranscriptError(
                        "bad_url", "That link is not a single TikTok video.", 400)
                key, fmt, languages = choose_tiktok_track(info, lang)
                video_id = str(info["id"])
                handle = info.get("uploader") or ""
                meta = {
                    "platform": "tiktok",
                    "video_id": video_id,
                    "page_url": (f"https://www.tiktok.com/@{handle}/video/{video_id}"
                                 if re.match(r"^[\w.-]+$", handle)
                                 else f"https://www.tiktok.com/@_/video/{video_id}"),
                    "title": (info.get("title") or info.get("description") or "")[:300],
                    "channel": info.get("channel") or handle,
                    "duration": info.get("duration") or 0,
                    "languages": languages,
                }
                if not key:
                    raise TranscriptError(
                        "no_captions",
                        "TikTok shows no captions for this video. The creator may have turned "
                        "auto captions off, or the video has no speech.",
                        404)
                raw = fmt.get("data") or ydl.urlopen(fmt["url"]).read()
            segments = parse_timed_text(raw)
            if not segments:
                raise TranscriptError(
                    "no_captions", "The caption track for this TikTok is empty.", 404)
            return {**meta, "language": key, "kind": "tiktok", "segments": segments}
        except TranscriptError:
            raise
        except Exception as e:  # noqa: BLE001 — classified below
            msg = str(e)
            errors.append(msg[:300])
            if _TT_RETRY.search(msg):
                continue
            if _TT_UNAVAILABLE.search(msg):
                raise TranscriptError(
                    "unavailable",
                    "TikTok says this video is private, removed or not available, or the "
                    "short link has expired, so its captions cannot be read.", 422)
    raise TranscriptError(
        "tiktok_blocked",
        "TikTok is not showing this video to our servers right now. Try again in a minute, "
        "or open it in the TikTok app and use its captions.", 503)


_tt_slots = threading.BoundedSemaphore(int(os.environ.get("FREE_TOOLS_TT_CONCURRENCY", "3")))


def _fetch_tiktok_with_slot(target_url, lang):
    if not _tt_slots.acquire(timeout=20):
        raise TranscriptError("busy", "The transcript tool is busy. Try again in a few seconds.", 503)
    try:
        return fetch_tiktok_transcript(target_url, lang)
    finally:
        _tt_slots.release()


@router.get("/api/tools/tiktok-transcript")
async def tiktok_transcript(request: Request, url: str = "", lang: str = ""):
    target = extract_tiktok_target(url)
    if not target:
        raise HTTPException(status_code=400, detail={
            "code": "bad_url",
            "message": "Paste the link of one TikTok video (tiktok.com/@user/video/... or a "
                       "vm.tiktok.com share link)."})
    ref, target_url = target
    lang = lang.strip() if re.match(r"^[A-Za-z0-9-]{1,20}$", lang.strip() or "") else None
    key = ("tiktok", ref, lang or "")
    cached = _cache_get(key)
    if cached and "error" in cached:
        e = cached["error"]
        raise HTTPException(status_code=e["status"], detail={"code": e["code"], "message": e["message"]})
    if cached:
        return {**cached, "cached": True}
    _enforce("tiktok_transcript", request)
    try:
        result = await asyncio.wait_for(
            asyncio.to_thread(_fetch_tiktok_with_slot, target_url, lang), timeout=TRANSCRIPT_TIMEOUT)
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail={
            "code": "timeout", "message": "TikTok took too long to answer. Try again."})
    except TranscriptError as e:
        if e.code in ("no_captions", "unavailable"):
            _cache_put(key, {"error": {"code": e.code, "message": e.message, "status": e.status}})
        raise HTTPException(status_code=e.status, detail={"code": e.code, "message": e.message})
    _cache_put(key, result)
    _cache_put(("tiktok", result["video_id"], lang or ""), result)
    return {**result, "cached": False}


# --------------------------------------------------------------------------- #
# YouTube title / description / tag generator
# --------------------------------------------------------------------------- #
MODES = ("titles", "description", "tags")
MAX_BRIEF = 2000


class MetadataRequest(BaseModel):
    mode: str
    topic: str
    keyword: Optional[str] = ""
    audience: Optional[str] = ""


_COMMON_RULES = """
- Write in the same language as the BRIEF. Never translate it.
- Use only facts present in the BRIEF. Do not invent numbers, results, names, prices, dates or guests.
- No emojis, no ALL CAPS words, no em dashes.
- Plain, specific wording a creator would actually publish. Avoid filler such as "in today's video", "ultimate guide", "you won't believe", "game-changer", "unlock", "dive into".
"""

PROMPTS = {
    "titles": """You write YouTube video titles.

BRIEF (what the video is about):
{topic}
{extra}
Write 10 title options, each with a different angle: specific outcome, how-to with a concrete promise, question the viewer already asks, number or list, contrarian take, curiosity gap that the video actually pays off, short and direct.

Rules:
- Hard limit 100 characters (YouTube's maximum). Aim for 60 or fewer: search results and phones cut long titles, so the payoff must come first.
- Put the main keyword early when it reads naturally, but vary the structure: no more than three titles may start with the same word, and never use a "Keyword: ..." colon pattern more than twice.{keyword_rule}
- Never promise something the brief does not support.
{common}
Return JSON: {{"titles": [{{"title": "...", "angle": "2-4 word label for the angle"}}]}}""",

    "description": """You write YouTube video descriptions.

BRIEF (what the video is about):
{topic}
{extra}
Write one description, 120 to 250 words, structured like this:
1. First two sentences (under 150 characters together): what the viewer gets from the video, with the main keyword in the first sentence. These are the only lines shown before "more", in search results and under the player.
2. A short paragraph with the specifics from the brief: what is covered, for whom.
3. If the BRIEF contains timestamps, a chapter list in "00:00 Title" format starting at 00:00. If it contains no timestamps, write no chapters and no timestamps at all.
4. One line inviting a specific next step (subscribe, comment with a question about the topic). Put placeholders like [link] where the creator should paste links; never invent URLs.
5. A final line with exactly 3 relevant hashtags (YouTube shows up to three next to the title).{keyword_rule}
{common}
Return JSON: {{"description": "..."}}""",

    "tags": """You pick YouTube tags.

BRIEF (what the video is about):
{topic}
{extra}
Return 15 to 25 tags ordered from most to least important:
- the exact main keyword first{keyword_rule_short}
- close variations and long-tail phrasings people type into YouTube search
- common misspellings of names or terms in the brief, if any (YouTube says tags mainly help when the topic is commonly misspelled)
- 2 or 3 broader topic tags
Rules: lowercase unless a proper noun, no hashtags, no duplicates, each tag under 30 characters, and all tags joined with commas must stay under 450 characters (YouTube's limit is 500).
{common}
Return JSON: {{"tags": ["...", "..."]}}""",
}


def build_prompt(mode: str, topic: str, keyword: str = "", audience: str = "") -> str:
    extra = f"TARGET AUDIENCE: {audience}\n" if audience else ""
    kw_rule = (f"\n- The main keyword is \"{keyword}\": use it naturally, exactly once in the "
               f"part that matters most." if keyword else "")
    kw_short = f" (\"{keyword}\")" if keyword else ""
    return PROMPTS[mode].format(topic=topic, extra=extra, keyword_rule=kw_rule,
                                keyword_rule_short=kw_short, common=_COMMON_RULES)


def _clean(s: str) -> str:
    return re.sub(r"\s*—\s*", ", ", (s or "")).strip()


def shape_result(mode: str, data: dict) -> dict:
    """Validate and tidy the model's JSON into what the page renders."""
    if mode == "titles":
        items = []
        for t in (data.get("titles") or [])[:12]:
            if isinstance(t, str):
                t = {"title": t, "angle": ""}
            title = _clean(str(t.get("title", "")))[:100]
            if title:
                items.append({"title": title, "angle": _clean(str(t.get("angle", "")))[:40]})
        if not items:
            raise ValueError("no titles")
        return {"titles": items}
    if mode == "description":
        desc = _clean(str(data.get("description", "")))[:5000]
        if not desc:
            raise ValueError("no description")
        return {"description": desc}
    seen, tags, total = set(), [], 0
    for t in data.get("tags") or []:
        t = re.sub(r"\s+", " ", str(t).replace("#", "").replace(",", " ")).strip()[:60]
        if not t or t.lower() in seen:
            continue
        if total + len(t) + (1 if tags else 0) > 500:
            break
        seen.add(t.lower())
        tags.append(t)
        total += len(t) + (1 if len(tags) > 1 else 0)
    if not tags:
        raise ValueError("no tags")
    return {"tags": tags, "characters": len(",".join(tags))}


def _gemini_key():
    try:
        from cloud import managed_keys
        key = managed_keys.gemini_key()
        if key:
            return key
    except Exception:
        pass
    return os.environ.get("GEMINI_API_KEY", "").strip() or None


METADATA_MODEL = (os.environ.get("FREE_TOOLS_GEMINI_MODEL")
                  or os.environ.get("GEMINI_MODEL_THUMBNAIL") or "gemini-3.7-flash")


def _thinking_config(types):
    try:
        return types.ThinkingConfig(thinking_level="low")
    except Exception:  # older SDK without thinking_level
        return None


def _generate(prompt: str) -> dict:
    from google import genai
    from google.genai import types
    client = genai.Client(api_key=_gemini_key())
    resp = client.models.generate_content(
        model=METADATA_MODEL,
        contents=[prompt],
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            # Thinking tokens count against max_output_tokens on the 3.x
            # models; at 2048 with default thinking the description came back
            # cut mid-string. Low thinking + a roomier cap keeps it whole and
            # the cost per call at a few thousand tokens.
            max_output_tokens=6000,
            temperature=0.9,
            thinking_config=_thinking_config(types),
        ),
    )
    text = (resp.text or "").strip()
    text = re.sub(r"^```(?:json)?|```$", "", text).strip()
    return json.loads(text)


@router.post("/api/tools/youtube-metadata")
async def youtube_metadata(payload: MetadataRequest, request: Request):
    mode = (payload.mode or "").strip().lower()
    if mode not in MODES:
        raise HTTPException(status_code=400, detail={"code": "bad_mode", "message": "Unknown mode."})
    topic = (payload.topic or "").strip()
    if len(topic) < 15:
        raise HTTPException(status_code=400, detail={
            "code": "too_short",
            "message": "Describe the video in at least a sentence: what happens in it and who it is for."})
    topic = topic[:MAX_BRIEF]
    keyword = re.sub(r"[\"\n\r{}]", " ", (payload.keyword or "")).strip()[:80]
    audience = re.sub(r"[\n\r{}]", " ", (payload.audience or "")).strip()[:120]
    if not _gemini_key():
        raise HTTPException(status_code=503, detail={
            "code": "not_configured", "message": "This tool needs a Gemini API key on the server."})
    _enforce("metadata", request)
    prompt = build_prompt(mode, topic.replace("{", "(").replace("}", ")"), keyword, audience)
    last = None
    for _ in range(2):
        try:
            data = await asyncio.wait_for(asyncio.to_thread(_generate, prompt), timeout=40)
            return {"mode": mode, **shape_result(mode, data)}
        except asyncio.TimeoutError:
            last = "timeout"
            break
        except Exception as e:  # noqa: BLE001 — model/JSON errors retry once
            last = str(e)[:200]
    print(f"[free_tools] metadata generation failed: {last}")
    raise HTTPException(status_code=502, detail={
        "code": "generation_failed", "message": "The generator did not answer. Try again."})
