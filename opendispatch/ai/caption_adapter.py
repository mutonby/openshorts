"""Per-platform caption adaptation.

Takes one source caption + a list of target platforms, returns a dict ready
to drop into `ContentUnit.formats`. Each platform gets a rewrite that
respects:

- Hard character limits (server-side trim is the safety net)
- Stylistic conventions (LinkedIn formal, Twitter punchy, IG hashtag-heavy)
- Format keys the rest of Open-Dispatch already understands

Provider abstraction
- Default: OpenRouter (any model)
- Local: Ollama (set OLLAMA_HOST and OLLAMA_MODEL — no API key needed)
- Stub: if neither is configured, falls back to a non-LLM heuristic so the
  caller still gets *something* usable. The heuristic is documented + tested.

Env vars
- OPENROUTER_API_KEY           — preferred
- OPENROUTER_MODEL             — default "openai/gpt-4o-mini"
- OPENROUTER_BASE_URL          — default "https://openrouter.ai/api/v1"
- OLLAMA_HOST                  — if set, prefers Ollama over OpenRouter
- OLLAMA_MODEL                 — default "qwen2.5:latest"
- AI_CAPTION_TEMPERATURE       — default 0.4
- AI_CAPTION_TIMEOUT_SECONDS   — default 30
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any

import httpx

log = logging.getLogger("open-dispatch.ai.caption")


class AdaptError(Exception):
    """Raised when caption adaptation can't produce any usable output."""


# ─── Platform spec ────────────────────────────────────────────────────────
# (format_key, max_chars, style_notes, output_template)
# `output_template` is what the LLM should populate. It maps directly to
# the format dict that Open-Dispatch's adapters consume.
PLATFORM_SPECS: dict[str, dict[str, Any]] = {
    "twitter": {
        "format_key": "twitter_thread",
        "max_chars": 280,
        "style": (
            "Punchy and conversational. One single tweet unless the source is "
            "long enough to merit a thread (>250 chars). For threads, split at "
            "natural sentence boundaries. Up to 1 tasteful emoji. No hashtags "
            "unless the source explicitly used them."
        ),
        "schema": "{ \"tweets\": [\"<=280 chars\", ...] }",
    },
    "bluesky": {
        "format_key": "bluesky_post",
        "max_chars": 300,
        "style": (
            "Similar to Twitter but slightly more raw, indie, dev-skewed. "
            "No hashtag stuffing. Single post unless source is very long."
        ),
        "schema": "{ \"text\": \"<=300 chars\" }",
    },
    "telegram": {
        "format_key": "telegram_message",
        "max_chars": 4096,
        "style": (
            "Casual but informative. Line breaks fine. Can be longer than "
            "Twitter — use the room. No hashtags."
        ),
        "schema": "{ \"text\": \"<=4096 chars\" }",
    },
    "instagram": {
        "format_key": "instagram_post",
        "max_chars": 2200,
        "style": (
            "Caption-heavy with double-newline paragraph breaks. End with "
            "5-12 relevant hashtags on their own line, each prefixed with #. "
            "Conversational, slightly hyped."
        ),
        "schema": "{ \"caption\": \"<=2200 chars including hashtags\" }",
    },
    "linkedin": {
        "format_key": "linkedin_post",
        "max_chars": 3000,
        "style": (
            "Professional, structured, no slang. Use line breaks for "
            "scanability. Lead with a hook line. No hashtags. No emojis "
            "unless the source had them."
        ),
        "schema": "{ \"text\": \"<=3000 chars\" }",
    },
    "threads": {
        "format_key": "threads_post",
        "max_chars": 500,
        "style": (
            "Conversational, slightly more relaxed than Twitter. Single "
            "post. Up to 1 emoji. No hashtags."
        ),
        "schema": "{ \"text\": \"<=500 chars\" }",
    },
    "youtube": {
        "format_key": "youtube_short",
        "max_chars": 5000,
        "style": (
            "Generate a title (<=100 chars) and a description (<=5000 chars). "
            "Title should be a hook. Description can be longer with context, "
            "credits, and 3-7 relevant hashtags at the end. Always include "
            "#Shorts in the description so YouTube classifies correctly."
        ),
        "schema": "{ \"title\": \"<=100\", \"description\": \"<=5000 incl #Shorts\" }",
    },
}


# ─── Heuristic fallback (no LLM) ──────────────────────────────────────────

def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    # Cut at a word boundary so we don't end mid-token
    cut = text[: limit - 1].rsplit(" ", 1)[0]
    return cut + "…"


def _extract_hashtags(text: str) -> list[str]:
    return re.findall(r"#\w+", text)


def _heuristic_adapt(source: str, platform: str) -> dict[str, Any]:
    """LLM-free fallback. Truncates to limits, light per-platform shaping.
    Always returns something usable — never raises.
    """
    spec = PLATFORM_SPECS[platform]
    limit = spec["max_chars"]
    clean = source.strip()

    if platform == "twitter":
        # Split into a thread if it overflows
        if len(clean) <= limit:
            return {"tweets": [clean]}
        # Naive sentence split, repack into tweets
        sentences = re.split(r"(?<=[.!?])\s+", clean)
        tweets: list[str] = []
        buf = ""
        for s in sentences:
            if len(buf) + len(s) + 1 <= limit - 4:  # leave room for "1/n"
                buf = (buf + " " + s).strip()
            else:
                if buf:
                    tweets.append(buf)
                buf = _truncate(s, limit)
        if buf:
            tweets.append(buf)
        return {"tweets": tweets[:25]}  # cap thread length

    if platform == "bluesky":
        return {"text": _truncate(clean, limit)}
    if platform == "telegram":
        return {"text": _truncate(clean, limit)}
    if platform == "threads":
        return {"text": _truncate(clean, limit)}
    if platform == "linkedin":
        # Strip hashtags for LinkedIn
        no_tags = re.sub(r"#\w+", "", clean).strip()
        no_tags = re.sub(r"\s+", " ", no_tags)
        return {"text": _truncate(no_tags, limit)}
    if platform == "instagram":
        # Keep existing hashtags, room for caption
        existing_tags = _extract_hashtags(clean)
        body = re.sub(r"#\w+", "", clean).strip()
        body = re.sub(r"\s+", " ", body)
        tag_line = " ".join(existing_tags[:12])
        caption = (body + ("\n\n" + tag_line if tag_line else "")).strip()
        return {"caption": _truncate(caption, limit)}

    if platform == "youtube":
        # Title = first sentence or first 100 chars
        first_sentence = re.split(r"(?<=[.!?])\s+", clean, 1)[0]
        title = _truncate(first_sentence, 100)
        description = clean
        if "#shorts" not in clean.lower():
            description = (description + "\n\n#Shorts").strip()
        description = _truncate(description, limit)
        return {"title": title, "description": description}

    # Unknown platform — return source as-is in a generic shape
    return {"text": _truncate(clean, limit)}


def _heuristic_all(source: str, platforms: list[str]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for p in platforms:
        spec = PLATFORM_SPECS.get(p)
        if not spec:
            continue
        out[spec["format_key"]] = _heuristic_adapt(source, p)
    return out


# ─── LLM call shape ───────────────────────────────────────────────────────

def _build_prompt(source: str, platforms: list[str]) -> tuple[str, str]:
    """Return (system, user) messages."""
    spec_lines = []
    for p in platforms:
        spec = PLATFORM_SPECS.get(p)
        if not spec:
            continue
        spec_lines.append(
            f"- {p} (format_key={spec['format_key']}, max_chars={spec['max_chars']}):\n"
            f"    style: {spec['style']}\n"
            f"    output_shape: {spec['schema']}"
        )
    specs_block = "\n".join(spec_lines)

    system = (
        "You adapt one source caption into platform-native posts. "
        "Stay strictly on-topic. Don't editorialize. Don't add facts the "
        "source didn't include. Respect every character limit. "
        "Return a single JSON object — no prose, no markdown, no commentary. "
        "Top-level keys are the format_keys listed in the platform spec. "
        "Each value is an object matching the platform's output_shape."
    )

    user = (
        f"PLATFORMS:\n{specs_block}\n\n"
        f"SOURCE CAPTION:\n```\n{source}\n```\n\n"
        f"Return ONLY the JSON object. Keys: "
        f"{', '.join(PLATFORM_SPECS[p]['format_key'] for p in platforms if p in PLATFORM_SPECS)}."
    )
    return system, user


def _strip_code_fences(text: str) -> str:
    """LLMs often wrap JSON in ```json … ``` despite being told not to."""
    t = text.strip()
    if t.startswith("```"):
        t = re.sub(r"^```(?:json)?\s*", "", t)
        t = re.sub(r"\s*```$", "", t)
    return t.strip()


def _parse_llm_json(raw: str) -> dict[str, Any]:
    cleaned = _strip_code_fences(raw)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        # Try to find the first {...} block — sometimes models prepend text
        m = re.search(r"\{[\s\S]*\}", cleaned)
        if not m:
            raise
        return json.loads(m.group(0))


async def _call_openrouter(system: str, user: str) -> str:
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise AdaptError("OPENROUTER_API_KEY not set")
    model = os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini")
    base = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    timeout = float(os.getenv("AI_CAPTION_TIMEOUT_SECONDS", "30"))
    temperature = float(os.getenv("AI_CAPTION_TEMPERATURE", "0.4"))

    async with httpx.AsyncClient(timeout=timeout) as client:
        r = await client.post(
            f"{base}/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                # OpenRouter recommends these for analytics
                "HTTP-Referer": os.getenv("OPENROUTER_REFERER", "https://github.com/Matthew-Selvam/Open-Dispatch"),
                "X-Title": "Open-Dispatch",
            },
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "temperature": temperature,
                "response_format": {"type": "json_object"},
            },
        )
        if r.status_code >= 400:
            raise AdaptError(f"OpenRouter {r.status_code}: {r.text[:200]}")
        data = r.json()
        return data["choices"][0]["message"]["content"]


async def _call_ollama(system: str, user: str) -> str:
    host = os.getenv("OLLAMA_HOST", "").rstrip("/")
    if not host:
        raise AdaptError("OLLAMA_HOST not set")
    model = os.getenv("OLLAMA_MODEL", "qwen2.5:latest")
    timeout = float(os.getenv("AI_CAPTION_TIMEOUT_SECONDS", "60"))

    async with httpx.AsyncClient(timeout=timeout) as client:
        r = await client.post(
            f"{host}/api/chat",
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "format": "json",
                "stream": False,
                "options": {
                    "temperature": float(os.getenv("AI_CAPTION_TEMPERATURE", "0.4")),
                },
            },
        )
        if r.status_code >= 400:
            raise AdaptError(f"Ollama {r.status_code}: {r.text[:200]}")
        data = r.json()
        return data["message"]["content"]


# ─── Public API ───────────────────────────────────────────────────────────

def _enforce_limits(adapted: dict[str, Any]) -> dict[str, Any]:
    """Belt-and-suspenders trimming. If the LLM ignored a limit, we cut."""
    out: dict[str, Any] = {}
    for platform, spec in PLATFORM_SPECS.items():
        fk = spec["format_key"]
        if fk not in adapted:
            continue
        block = adapted[fk]
        limit = spec["max_chars"]
        if fk == "twitter_thread":
            tweets = block.get("tweets") or []
            out[fk] = {"tweets": [_truncate(t, limit) for t in tweets if isinstance(t, str)]}
        elif fk == "instagram_post":
            cap = block.get("caption") or ""
            out[fk] = {"caption": _truncate(cap, limit)}
        elif fk == "youtube_short":
            title = (block.get("title") or "").strip()[:100]
            desc = block.get("description") or ""
            out[fk] = {"title": title, "description": _truncate(desc, limit)}
        else:
            txt = block.get("text") or ""
            out[fk] = {"text": _truncate(txt, limit)}
    return out


async def adapt_caption_async(
    source: str,
    platforms: list[str],
    *,
    provider: str | None = None,
) -> dict[str, dict[str, Any]]:
    """Adapt one source caption for each requested platform.

    Returns a dict shaped like ContentUnit.formats. Always returns at least
    a heuristic fallback for every platform — never raises on transient
    network errors, only on bad inputs.
    """
    if not source.strip():
        raise AdaptError("source caption is empty")
    valid = [p for p in platforms if p in PLATFORM_SPECS]
    if not valid:
        raise AdaptError(f"no supported platforms in {platforms}")

    # Decide provider
    has_openrouter = bool(os.getenv("OPENROUTER_API_KEY"))
    has_ollama = bool(os.getenv("OLLAMA_HOST"))
    chosen = provider or ("ollama" if has_ollama else ("openrouter" if has_openrouter else "heuristic"))

    if chosen == "heuristic":
        log.info("caption adapter: no LLM configured, using heuristic")
        return _heuristic_all(source, valid)

    system, user = _build_prompt(source, valid)
    try:
        if chosen == "ollama":
            raw = await _call_ollama(system, user)
        else:
            raw = await _call_openrouter(system, user)
        parsed = _parse_llm_json(raw)
        return _enforce_limits(parsed)
    except (AdaptError, httpx.HTTPError, json.JSONDecodeError, KeyError) as e:
        log.warning("caption adapter falling back to heuristic: %s", e)
        return _heuristic_all(source, valid)


def adapt_caption(
    source: str,
    platforms: list[str],
    *,
    provider: str | None = None,
) -> dict[str, dict[str, Any]]:
    """Sync wrapper for adapt_caption_async.

    Works when called from a thread with no running event loop (CLI,
    scripts, worker threads). If the calling code is itself running on
    an event loop (e.g. inside an async FastAPI handler), blocking is
    impossible — this raises a clear RuntimeError instead of the old
    behaviour (a guaranteed 'loop is already running' crash from
    run_until_complete). Use adapt_caption_async() there.
    """
    import asyncio

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        pass  # no loop in this thread — safe to run one
    else:
        raise RuntimeError(
            "adapt_caption() called from inside a running event loop — "
            "use await adapt_caption_async(...) instead"
        )

    return asyncio.run(adapt_caption_async(source, platforms, provider=provider))
