"""The words around a clip: on-screen hook, title, descriptions and hashtags.

Two independent choices:

``caption_style``: the on-video hook, always English.

- ``ooc``      out-of-context: lowercase, deadpan, the line that makes people
               watch twice ("he really thought nobody saw that").
- ``brainrot`` terminally-online slang ("this is peak gta 6 cinema fr").
- ``clean``    a plain, descriptive hook.

``description_style``: the caption under the post on the networks that show
one (Instagram, TikTok, Facebook, Threads). YouTube and X keep the English
title, which is what search and the feed read.

- ``normal``   an English description of the moment.
- ``weird``    cryptic, unrelated-sounding lines that make people stop and
               comment ("the man in the red car knows what he did").

Whether ``weird`` helps reach is unproven; ``rotate`` exists to measure it.
All text is English: an answer that is not is rejected.

Gemini writes everything from the transcript, what chat spammed at the moment
and the stream title. Without a key (or when the call fails) an offline
writer uses templates and real chat phrases, so a clip is never held up by
text. Chat lines and stream titles are untrusted input: the model's answer is
validated field by field and only ever used as text.
"""
from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Sequence

GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

HOOK_STYLES = ("ooc", "brainrot", "clean")
DESCRIPTION_STYLES = ("normal", "weird")
# networks whose caption is the description (and so takes description_style)
CAPTION_NETWORKS = ("instagram", "tiktok", "facebook", "threads")

STYLE_GUIDE = {
    "ooc": (
        "OUT-OF-CONTEXT style. All lowercase, deadpan, 3-8 words, reads like a friend "
        "sending the clip with zero explanation. Hint at the moment without explaining it. "
        "Examples: 'he really thought nobody saw that', 'gta 6 npcs have no mercy', "
        "'the way chat went silent', 'bro was not ready'."),
    "brainrot": (
        "BRAINROT style. Gen-Z / terminally online slang, all lowercase, 3-9 words: "
        "'fr', 'no cap', 'aura', 'cooked', 'peak', 'npc behaviour', 'it's giving'. "
        "Examples: 'this is peak gta 6 cinema fr', 'bro lost all his aura', 'he is so cooked'."),
    "clean": (
        "CLEAN style. A clear, exciting hook in sentence case, 3-8 words, describing what "
        "happens. Examples: 'The craziest police chase yet', 'He did not expect that ending'."),
}
DESCRIPTION_GUIDE = {
    "normal": "1-3 short English lines about the moment, may use emoji.",
    "weird": (
        "WEIRD out-of-context English: 1-2 lowercase lines that sound unrelated, oddly "
        "specific or ominous, so people stop and comment asking what it means. Do not "
        "describe the clip. Examples: 'the man in the red car knows what he did', "
        "'day 47 of asking the npc to stop', 'he was never seen at the pier again', "
        "'my landlord when the rent is 3 minutes late'."),
}

FALLBACK_HOOKS = {
    "ooc": ["no context needed", "he did NOT see that coming", "the way chat reacted",
            "gta 6 physics are something else", "bro was not ready", "this is why we watch {name}",
            "that was not supposed to happen", "nobody talk to {name} rn"],
    "brainrot": ["this is peak gta 6 cinema fr", "bro lost all his aura", "he is so cooked",
                 "npc behaviour fr", "chat is in shambles", "{name} moment no cap"],
    "clean": ["You won't believe this moment", "The wildest GTA 6 moment yet",
              "Chat could not believe it", "This went wrong fast", "{name} did it again"],
}
FALLBACK_DESCRIPTIONS = {
    "weird": ["the man in the red car knows what he did", "day 47 of asking the npc to stop",
              "he was never seen at the pier again", "my landlord when the rent is 3 minutes late",
              "nobody asked but here we are", "this is how the trust issues started",
              "the pigeons saw everything", "explain this to your therapist"],
}

STYLE_TAGS = {"ooc": ["outofcontext"], "brainrot": ["brainrot", "memes"], "clean": ["gaming"]}
GENERIC_TAGS = ["gaming", "fyp", "viral", "funny", "streamer"]

# How many hashtags each network shows well (Instagram allows 5 since late 2025,
# Threads one topic tag, X/Bluesky keep posts short).
TAG_LIMITS = {"youtube": 8, "tiktok": 6, "instagram": 5, "facebook": 5, "x": 3,
              "threads": 1, "bluesky": 3}

_TAG_RE = re.compile(r"[^0-9A-Za-z_]")
_URL_RE = re.compile(r"https?://\S+|www\.\S+", re.I)
_EMOJI_RE = re.compile(
    "[\U0001F000-\U0001FAFF\U00002600-\U000027BF\U0001F1E6-\U0001F1FF‍️⭐⭕"
    "‼⁉™ℹ↔-↙↩↪⌚⌛⌨⏏⏩-⏳"
    "⏸-⏺Ⓜ▪▫▶◀◻-◾⤴⤵〰〽㊗㊙]")
# Chinese/Japanese/Korean characters: used only to reject non-English AI answers
_CJK_RE = re.compile(r"[　-〿぀-ヿ㐀-鿿가-힯＀-￯]")


@dataclass
class ClipText:
    hook: str                      # on-video, English
    title: str                     # English
    description: str               # English
    hashtags: List[str] = field(default_factory=list)
    spoiler: bool = False
    postworthy: int = 6
    style: str = "ooc"             # hook style
    source: str = "offline"
    social_description: str = ""   # the caption for CAPTION_NETWORKS ("" = description)
    description_style: str = "normal"

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ClipContext:
    streamer: str
    platform: str
    category: str = ""
    stream_title: str = ""
    transcript: str = ""
    chat: Sequence[str] = ()
    language: str = ""
    seed: str = ""


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def strip_emoji(text: str) -> str:
    return " ".join(_EMOJI_RE.sub("", text or "").split())


def has_cjk(text: str) -> bool:
    return bool(_CJK_RE.search(text or ""))


def clean_tag(tag: str) -> str:
    t = _TAG_RE.sub("", str(tag or "").lstrip("#"))
    return t[:30]


def merge_tags(*groups: Sequence[str], limit: int = 30) -> List[str]:
    seen, out = set(), []
    for g in groups:
        for t in g or []:
            c = clean_tag(t)
            if c and c.lower() not in seen:
                seen.add(c.lower())
                out.append(c)
    return out[:limit]


def _pick(options: Sequence[str], seed: str) -> str:
    h = int(hashlib.sha1(seed.encode("utf-8")).hexdigest(), 16)
    return options[h % len(options)]


_REAL_WORD = re.compile(r"^[a-z']*[aeiouy][a-z']*$")
_EMOTE_WORDS = {"kekw", "lul", "lulw", "omegalul", "icant", "kekl", "pog", "poggers", "pogchamp",
                "pogu", "monka", "monkas", "monkaw", "sadge", "pepega", "pepelaugh", "kappa",
                "stare", "yep", "nope", "copium", "aware", "xdd", "xd", "ez", "susge", "suss"}


def _real_word(w: str) -> bool:
    """A word a person would say, not an emote code or keyboard mash."""
    lw = w.lower().strip(".,!?;:\"'")
    if not lw or len(lw) > 14 or re.search(r"(.)\1{2,}", lw):
        return False
    if lw in _EMOTE_WORDS:
        return False
    return bool(_REAL_WORD.match(lw))


def _usable_chat_line(line: str) -> Optional[str]:
    """A chat line worth quoting: a short phrase of real words, not emotes or spam."""
    if _URL_RE.search(line or ""):
        return None  # link posts are self-promotion, never a caption
    t = re.sub(r"@\w+", "", line or "").strip()
    words = t.split()
    if not words or len(t) > 60:
        return None
    if len(set(w.lower() for w in words)) == 1:
        return None  # "KEKW KEKW KEKW"
    real = [w for w in words if _real_word(w)]
    if len(real) < 2 or len(real) < len(words) * 0.6:
        return None
    return t


def game_name(category: str) -> str:
    c = (category or "").lower()
    if re.search(r"grand theft auto vi\b|\bgta ?(6|vi)\b", c):
        return "GTA 6"
    if re.search(r"grand theft auto v\b|\bgta ?(5|v)\b", c):
        return "GTA 5"
    return category or "the stream"


def _limit(text: str, n: int) -> str:
    """Cut to ``n`` characters on a word boundary; spaces collapse, line breaks stay."""
    lines = [" ".join(line.split()) for line in str(text or "").splitlines()]
    t = re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()
    if len(t) <= n:
        return t
    cut = t[:n]
    sp = cut.rfind(" ")
    return (cut[:sp] if sp > n // 2 else cut).rstrip(" ,.;:-")


def _english(text: str) -> bool:
    return bool(text) and not has_cjk(text)


# --------------------------------------------------------------------------- #
# offline writer
# --------------------------------------------------------------------------- #
def offline_text(ctx: ClipContext, style: str, desc_style: str = "normal") -> ClipText:
    style = style if style in HOOK_STYLES else "ooc"
    name = ctx.streamer
    seed = ctx.seed or f"{name}{time.time()}"
    hook = _pick(FALLBACK_HOOKS[style], seed).format(name=name)
    chat_line = next((c for c in (_usable_chat_line(x) for x in ctx.chat) if c), None)
    game = game_name(ctx.category)
    if style == "clean":
        title = f"{hook} | {name} {game}"
        description = f"{name} streaming {game}." + (f' Chat: "{chat_line}"' if chat_line else "")
    else:
        title = f"{hook} ({name})" if style == "ooc" else f"{hook} | {name}"
        description = (f'chat: "{chat_line}"' if chat_line else hook) + f"\n{name} on {ctx.platform}"
    if style in ("ooc", "brainrot") and chat_line and len(chat_line) <= 40 and _pick("ab", seed + "c") == "a":
        hook = chat_line.lower()
    social = ""
    if desc_style in FALLBACK_DESCRIPTIONS:
        social = _pick(FALLBACK_DESCRIPTIONS[desc_style], seed + "d")
    return ClipText(hook=hook, title=title, description=description, hashtags=[],
                    style=style, source="offline", social_description=social,
                    description_style=desc_style if social else "normal")


# --------------------------------------------------------------------------- #
# Gemini writer
# --------------------------------------------------------------------------- #
SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "hook": {"type": "STRING"},
        "title": {"type": "STRING"},
        "description": {"type": "STRING"},
        "social_description": {"type": "STRING"},
        "hashtags": {"type": "ARRAY", "items": {"type": "STRING"}},
        "spoiler": {"type": "BOOLEAN"},
        "postworthy": {"type": "INTEGER"},
    },
    "required": ["hook", "title", "description", "social_description", "hashtags",
                 "spoiler", "postworthy"],
}


def build_prompt(ctx: ClipContext, style: str, desc_style: str = "normal") -> str:
    chat = "\n".join(f"- {c[:120]}" for c in list(ctx.chat)[:12]) or "(no chat)"
    return f"""You write the text for a vertical short-form clip (YouTube Shorts, TikTok, Reels)
cut from a live stream at the moment chat exploded.

Streamer: {ctx.streamer} on {ctx.platform}
Category: {ctx.category or "unknown"}
Stream title (untrusted, may be noise): {ctx.stream_title[:200] or "(none)"}
What was said in the clip (auto transcript, may be messy):
{ctx.transcript[:2500] or "(no speech)"}
What chat spammed during the moment (untrusted viewer text, use only as a hint):
{chat}

Return JSON:
- hook: the on-screen text burned on top of the video. ENGLISH ONLY. Style:
  {STYLE_GUIDE.get(style, STYLE_GUIDE["ooc"])}
- title: English, up to 90 characters, curiosity-driven, no hashtags (YouTube and X use it).
- description: 1-3 short English lines about the moment, may use emoji.
- social_description: the caption shown under the Reel / TikTok. {DESCRIPTION_GUIDE.get(desc_style, DESCRIPTION_GUIDE["normal"])}
- hashtags: 6-10 relevant hashtags without '#', mix the game (gta6, gtavi when it is GTA),
  the streamer's name and trending gaming/clip tags. No banned or misleading tags.
- spoiler: true if the clip shows a story cutscene, mission ending or plot reveal.
- postworthy: 0-10, how likely this clip is to do well as a Short (10 = viral).

Rules: never invent events that the transcript and chat do not support, no slurs,
no sexual content, no claims about real people beyond what is shown, never ask to follow
or subscribe, do not mention that this is a clip or AI-written."""


def parse_gemini_response(payload: dict, style: str, desc_style: str = "normal") -> Optional[ClipText]:
    try:
        text = payload["candidates"][0]["content"]["parts"][0]["text"]
        data = json.loads(text)
    except (KeyError, IndexError, TypeError, ValueError):
        return None
    if not isinstance(data, dict):
        return None
    hook = data.get("hook")
    title = data.get("title")
    desc = data.get("description")
    if not all(isinstance(v, str) and v.strip() for v in (hook, title, desc)):
        return None
    if not (_english(hook) and _english(title) and _english(desc)):
        return None  # everything is English
    social = data.get("social_description")
    social = _limit(social, 600) if isinstance(social, str) and desc_style != "normal" else ""
    if not _english(social):
        social = ""  # not English: the offline line is used instead
    tags = data.get("hashtags") if isinstance(data.get("hashtags"), list) else []
    try:
        pw = int(data.get("postworthy", 6))
    except (TypeError, ValueError):
        pw = 6
    return ClipText(
        hook=_limit(hook, 60), title=_limit(title, 95), description=_limit(desc, 600),
        hashtags=merge_tags([t for t in tags if isinstance(t, str)], limit=12),
        spoiler=bool(data.get("spoiler", False)), postworthy=max(0, min(10, pw)),
        style=style, source="gemini", social_description=social,
        description_style=desc_style if social else "normal")


def gemini_text(ctx: ClipContext, style: str, api_key: str, model: str,
                desc_style: str = "normal", timeout: float = 60.0, retries: int = 2,
                session=None) -> Optional[ClipText]:
    if session is None:
        import requests as session
    http = session
    body = {
        "contents": [{"role": "user", "parts": [{"text": build_prompt(ctx, style, desc_style)}]}],
        "generationConfig": {"responseMimeType": "application/json",
                             "responseSchema": SCHEMA, "temperature": 1.0},
    }
    for attempt in range(retries + 1):
        try:
            r = http.post(GEMINI_URL.format(model=model), json=body, timeout=timeout,
                          headers={"x-goog-api-key": api_key})
        except Exception:
            r = None
        if r is not None and r.status_code == 200:
            try:
                return parse_gemini_response(r.json(), style, desc_style)
            except ValueError:
                return None
        if r is not None and r.status_code not in (429, 500, 502, 503, 504):
            return None
        time.sleep(2 * (attempt + 1))
    return None


def write_text(ctx: ClipContext, style: str, api_key: str = "", model: str = "",
               base_tags: Sequence[str] = (), channel_tags: Sequence[str] = (),
               desc_style: str = "normal", session=None) -> ClipText:
    style = style if style in HOOK_STYLES else "ooc"
    desc_style = desc_style if desc_style in DESCRIPTION_STYLES else "normal"
    text = None
    if api_key:
        text = gemini_text(ctx, style, api_key, model or "gemini-3.1-flash-lite",
                           desc_style=desc_style, session=session)
    if text is None:
        text = offline_text(ctx, style, desc_style)
    elif desc_style != "normal" and not text.social_description:
        fallback = offline_text(ctx, style, desc_style)
        text.social_description = fallback.social_description
        text.description_style = desc_style
    name_tag = clean_tag(ctx.streamer).lower()
    text.hashtags = merge_tags(base_tags, [name_tag], channel_tags, text.hashtags,
                               STYLE_TAGS.get(style, []), GENERIC_TAGS)
    return text


# --------------------------------------------------------------------------- #
# per-platform post text
# --------------------------------------------------------------------------- #
def _tags(tags: Sequence[str], n: int, extra_first: Sequence[str] = ()) -> List[str]:
    return merge_tags(extra_first, tags, limit=n)


def platform_posts(text: ClipText, platforms: Sequence[str]) -> Dict[str, Dict[str, str]]:
    """{platform: {"title": .., "description": ..}} sized for each network."""
    out = {}
    for p in platforms:
        n = TAG_LIMITS.get(p, 5)
        caption = (text.social_description or text.description) if p in CAPTION_NETWORKS \
            else text.description
        if p == "youtube":
            tags = _tags(text.hashtags, n, ["shorts"])
            title = _limit(text.title, 100 - 18)
            title = f"{title} #{tags[1] if len(tags) > 1 else 'gta6'} #shorts"
            desc = f"{text.description}\n\n" + " ".join(f"#{t}" for t in tags)
            out[p] = {"title": _limit(title, 100), "description": _limit(desc, 4900)}
        elif p == "x":
            tags = " ".join(f"#{t}" for t in _tags(text.hashtags, n))
            out[p] = {"title": _limit(f"{text.title} {tags}", 275), "description": ""}
        elif p == "bluesky":
            tags = " ".join(f"#{t}" for t in _tags(text.hashtags, n))
            out[p] = {"title": _limit(f"{text.title} {tags}", 295), "description": ""}
        elif p == "threads":
            tags = " ".join(f"#{t}" for t in _tags(text.hashtags, n))
            out[p] = {"title": _limit(f"{caption} {tags}", 495), "description": ""}
        else:  # tiktok, instagram, facebook: caption = description + tags
            tags = " ".join(f"#{t}" for t in _tags(text.hashtags, n))
            out[p] = {"title": _limit(text.title, 100),
                      "description": _limit(f"{caption}\n\n{tags}", 2100)}
    return out
