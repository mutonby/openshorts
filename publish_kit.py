"""publish_kit.py — Manual-publish helper: viral title, description, hashtags.

Nothing here posts anywhere. It builds a copy-ready "publish kit" for one
selected clip — a scroll-stopping title, a description, and a mix of RELEVANT
niche hashtags + TODAY's TRENDING hashtags — so the user can review the clip
and paste it into YouTube/TikTok/IG themselves.

Trending refresh (automatic, day by day):
  * Primary: free public trend sources (Trends24 — no API key).
  * Fallback: the free AI gateway lists today's trending topics for the region
    (seeded with today's date).
  * The result is cached on disk and refreshed automatically when the cached
    day is stale (or the file is older than CACHE_TTL_HOURS), so every day the
    app serves a fresh trend list without any manual step.
"""

import json
import os
import time
from datetime import date
from typing import Any, Dict, List, Optional, Tuple

import httpx

import ai_gateway

CACHE_TTL_HOURS = 12
# How many trending topics to feed the AI as context (top N by rank).
TRENDING_CONTEXT_LIMIT = 30

# Common platform/region trend API params (Trends24 country codes).
REGIONS = {
    "US": "United States", "GB": "United Kingdom", "IN": "India",
    "BD": "Bangladesh", "PK": "Pakistan", "CA": "Canada", "AU": "Australia",
    "DE": "Germany", "FR": "France", "ES": "Spain", "BR": "Brazil",
    "JP": "Japan", "ID": "Indonesia", "PH": "Philippines", "NG": "Nigeria",
    "MX": "Mexico", "IT": "Italy", "TR": "Turkey", "KR": "South Korea",
    "SA": "Saudi Arabia", "AE": "UAE",
}


def _cache_path() -> str:
    data_dir = os.environ.get("DATA_DIR", "").strip() or "output"
    return os.path.join(data_dir, "trending_cache.json")


def _load_cache() -> dict:
    try:
        with open(_cache_path(), "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _save_cache(cache: dict) -> None:
    try:
        os.makedirs(os.path.dirname(_cache_path()), exist_ok=True)
        with open(_cache_path(), "w", encoding="utf-8") as f:
            json.dump(cache, f, indent=2)
    except Exception as e:
        print(f"⚠️ [publish_kit] Could not persist trend cache: {e}")


# --------------------------------------------------------------------------- #
# Trend sources
# --------------------------------------------------------------------------- #
def _fetch_trends24(region: str) -> List[Dict[str, Any]]:
    """Today's trending topics from Trends24 (Twitter trends, free, no key).

    Returns [{"topic", "rank"?, "volume"?}, ...] or [] on any failure.
    """
    url = f"https://api.trends24.in/api/v1/today?geo={region}"
    resp = httpx.get(url, timeout=20.0, headers={
        "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                       "AppleWebKit/537.36 (KHTML, like Gecko) "
                       "Chrome/120.0.0.0 Safari/537.36"),
    })
    resp.raise_for_status()
    data = resp.json()
    trends = data.get("trends") if isinstance(data, dict) else None
    if not isinstance(trends, list):
        return []
    out: List[Dict[str, Any]] = []
    for t in trends:
        if isinstance(t, str):
            out.append({"topic": t})
        elif isinstance(t, dict):
            name = t.get("name") or t.get("title") or t.get("query")
            if name:
                out.append({
                    "topic": str(name).strip(),
                    "rank": t.get("rank"),
                    "volume": t.get("tweet_count") or t.get("volume"),
                })
    return [t for t in out if t.get("topic")]


_TRENDS_AI_SYSTEM = (
    "You are a social-media trend analyst. Return strict JSON only — no "
    "markdown, no commentary."
)


def _fetch_trends_ai(region: str, language: str) -> List[Dict[str, Any]]:
    """AI-generated 'today' trend list (fallback when the trend API is down)."""
    if not ai_gateway.is_configured():
        return []
    today = date.today().isoformat()
    region_name = REGIONS.get(region, region)
    prompt = (
        f"Today is {today}. List the top 20 topics that are TRENDING right "
        f"now on YouTube Shorts / TikTok / Instagram Reels in "
        f"{region_name} (language: {language or 'any'}). Mix entertainment, "
        f"news, sports, tech, viral moments and evergreen niches.\n"
        f"Return JSON: {{\"trends\": [{{\"topic\": \"...\"}}, ...]}}"
    )
    try:
        parsed, _result = ai_gateway.complete_json(
            system=_TRENDS_AI_SYSTEM, user=prompt, temperature=0.7)
        trends = parsed.get("trends") or []
        out = []
        for t in trends:
            if isinstance(t, str):
                out.append({"topic": t})
            elif isinstance(t, dict) and t.get("topic"):
                out.append({"topic": str(t["topic"])})
        return out
    except Exception as e:
        print(f"⚠️ [publish_kit] AI trend fetch failed: {e}")
        return []


def fetch_trending_topics(region: str = "US",
                          language: str = "en") -> Tuple[List[str], str]:
    """(topics, source) for today, refreshed automatically when stale.

    source is one of: "trends24" | "ai" | "cache" | "none".
    """
    region = (region or "US").upper()[:2]
    if region not in REGIONS:
        region = "US"
    today = date.today().isoformat()
    cache = _load_cache()
    entry = cache.get(f"{region}:{language}") or {}
    fresh = (entry.get("date") == today
             and time.time() - float(entry.get("fetched_at") or 0) < CACHE_TTL_HOURS * 3600)
    if fresh and entry.get("topics"):
        return list(entry["topics"]), "cache"

    topics: List[str] = []
    source = "none"
    try:
        raw = _fetch_trends24(region)
        if raw:
            topics = [t["topic"] for t in raw if t.get("topic")]
            source = "trends24"
    except Exception as e:
        print(f"⚠️ [publish_kit] Trends24 unavailable ({e}) — using AI fallback.")
    if not topics:
        raw = _fetch_trends_ai(region, language)
        if raw:
            topics = [t["topic"] for t in raw if t.get("topic")]
            source = "ai"

    if topics:
        cache[f"{region}:{language}"] = {
            "date": today,
            "fetched_at": time.time(),
            "topics": topics,
        }
        _save_cache(cache)
    elif fresh:
        return list(entry.get("topics") or []), "cache"
    return topics[:TRENDING_CONTEXT_LIMIT], source


# --------------------------------------------------------------------------- #
# Publish kit generation
# --------------------------------------------------------------------------- #
_KIT_SYSTEM = (
    "You are a viral short-form content strategist and copywriter. Return "
    "strict JSON only — no markdown, no commentary."
)

KIT_PROMPT_TEMPLATE = """Write a copy-ready publish kit for THIS ONE CLIP.

CLIP TITLE (context): {clip_title}
CLIP TRANSCRIPT (what is said in the clip):
{clip_text}

VIDEO LANGUAGE: {language}
TODAY'S DATE: {today}
TRENDING TOPICS RIGHT NOW (source: {trend_source}):
{trending_json}

Return JSON exactly like this:
{{
  "title": "viral, curiosity-driven title (max 100 chars for YouTube)",
  "description": "2-4 sentences hook + summary + soft CTA, written in {language}, no hashtags inside",
  "hashtags": [
    {{"tag": "#example", "source": "trending", "why": "why it fits this clip"}},
    {{"tag": "#example2", "source": "niche", "why": "content-specific tag"}}
  ]
}}

RULES:
- Title: max 100 characters, curiosity + specificity, no clickbait lies.
- Hashtags: 4-7 total — mix of (a) TRENDING topics from the list above that
  genuinely fit this clip and (b) niche/content-specific tags. Prefer
  lowercase, no spaces. Include at most one of #shorts / #fyp / #viral /
  #trending, only if it makes sense. No hashtag spam.
- Everything in the video's language ({language}); hashtags may be in English
  when that's standard for the platform.
- Be specific to the transcript — never generic filler."""


def build_clip_context(metadata: dict,
                       clip_index: int) -> Tuple[Optional[dict], str]:
    """The clip's own metadata + the exact words spoken inside it."""
    shorts = metadata.get("shorts") or []
    if clip_index >= len(shorts):
        return None, ""
    clip = shorts[clip_index]
    try:
        start = float(clip.get("start") or 0)
        end = float(clip.get("end") or 0)
    except (TypeError, ValueError):
        start, end = 0.0, 0.0
    segments = (metadata.get("transcript") or {}).get("segments") or []
    words = []
    for seg in segments:
        for w in seg.get("words", []):
            try:
                ws, we = float(w.get("start", 0)), float(w.get("end", 0))
            except (TypeError, ValueError):
                continue
            # Any overlap with the clip window counts (words may straddle
            # the cut points slightly).
            if ws < end and we > start:
                words.append(str(w.get("word", "")))
    text = " ".join(w for w in words if w)
    if not text:  # word-level missing → use segment text in range
        parts = []
        for seg in segments:
            try:
                s, e = float(seg.get("start", 0)), float(seg.get("end", 0))
            except (TypeError, ValueError):
                continue
            if e > start and s < end:
                parts.append(str(seg.get("text", "")))
        text = " ".join(parts)
    return clip, text.strip()


def generate_publish_kit(metadata: dict, clip_index: int,
                         language: str = "auto",
                         region: str = "US") -> Dict[str, Any]:
    """Full publish kit for one clip. Raises on failure."""
    clip, clip_text = build_clip_context(metadata, clip_index)
    if clip is None:
        raise ValueError(f"Clip index {clip_index} out of range.")
    if not clip_text:
        raise ValueError("No transcript text found for this clip — reprocess "
                         "the video so captions can be generated.")

    lang = language if language and language != "auto" else (
        (metadata.get("transcript") or {}).get("language") or "english")
    topics, source = fetch_trending_topics(region, lang)

    prompt = KIT_PROMPT_TEMPLATE.format(
        clip_title=clip.get("title") or clip.get("video_title_for_youtube_short")
        or "Untitled clip",
        clip_text=clip_text[:6000],
        language=lang,
        today=date.today().isoformat(),
        trend_source=source,
        trending_json=json.dumps([{"topic": t} for t in topics[:TRENDING_CONTEXT_LIMIT]],
                                 ensure_ascii=False),
    )

    parsed, _result = ai_gateway.complete_json(
        system=_KIT_SYSTEM, user=prompt, temperature=0.8, max_tokens=3000)

    title = str(parsed.get("title") or "").strip()[:100]
    description = str(parsed.get("description") or "").strip()
    hashtags_raw = parsed.get("hashtags") or []
    hashtags = []
    for h in hashtags_raw:
        if isinstance(h, str):
            tag = h
            why = ""
        elif isinstance(h, dict):
            tag = str(h.get("tag") or "")
            why = str(h.get("why") or "")
        else:
            continue
        tag = tag.strip()
        if not tag:
            continue
        if not tag.startswith("#"):
            tag = "#" + tag.lstrip("#")
        hashtags.append({"tag": tag, "source": h.get("source") if isinstance(h, dict) else "niche", "why": why})
    # Deduplicate, keep order.
    seen, deduped = set(), []
    for h in hashtags:
        if h["tag"].lower() not in seen:
            seen.add(h["tag"].lower())
            deduped.append(h)

    return {
        "title": title or (clip.get("video_title_for_youtube_short") or ""),
        "description": description,
        "hashtags": deduped[:12],
        "trending_topics": topics[:15],
        "trend_source": source,
        "language": lang,
        "region": region,
        "generated_at": date.today().isoformat(),
    }


def format_kit_text(kit: Dict[str, Any]) -> str:
    """The copy-ready block: title + description + hashtags."""
    lines = [kit.get("title") or "", ""]
    if kit.get("description"):
        lines.append(kit["description"])
        lines.append("")
    tags = [h["tag"] for h in (kit.get("hashtags") or [])]
    if tags:
        lines.append(" ".join(tags))
    return "\n".join(lines).strip()
