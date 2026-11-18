"""Settings: one TOML file (``livewatch.toml``) plus secrets from the environment.

Secrets never live in the TOML: ``GEMINI_API_KEY``, ``UPLOADPOST_API_KEY``
(and optionally ``TWITCH_CHAT_TOKEN``) come from the environment or a ``.env``
file next to the config. Everything else has a default that suits a 2-core
laptop, so a config with only ``[[channel]]`` entries works.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence

try:  # Python 3.11+
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - 3.10 and older
    import tomli as tomllib  # type: ignore

PLATFORMS = ("twitch", "kick", "youtube")
CAPTION_STYLES = ("ooc", "brainrot", "clean")          # on-video hook, always English
DESCRIPTION_STYLES = ("normal", "weird")              # caption under Reels/TikToks
POST_PLATFORMS = ("youtube", "tiktok", "instagram", "facebook", "x", "threads", "bluesky")

_NAME_RE = re.compile(r"^[A-Za-z0-9_.@-]{1,100}$")


class ConfigError(ValueError):
    pass


@dataclass
class Channel:
    platform: str
    name: str
    label: str = ""
    # Only clip while the stream's category contains one of these (case
    # insensitive). Empty: clip whatever is streamed. YouTube exposes no
    # category, so the filter is ignored there.
    game_filter: List[str] = field(default_factory=list)
    # Facecam rectangle as fractions of the frame: [x, y, w, h]. Set it and the
    # Short is facecam on top, gameplay below; leave it out for the blurred-fill
    # layout, which suits any stream.
    facecam: Optional[List[float]] = None
    tags: List[str] = field(default_factory=list)
    language: str = "auto"
    caption_style: str = ""  # empty: the global style

    @property
    def key(self) -> str:
        return f"{self.platform}:{self.name.lower()}"

    @property
    def display(self) -> str:
        return self.label or self.name.lstrip("@")

    @property
    def url(self) -> str:
        n = self.name
        if self.platform == "twitch":
            return f"https://www.twitch.tv/{n.lower()}"
        if self.platform == "kick":
            return f"https://kick.com/{n.lower()}"
        # youtube: @handle, UC.. channel id, or a full URL to a live video
        if n.startswith("http"):
            return n
        if n.startswith("UC") and len(n) == 24:
            return f"https://www.youtube.com/channel/{n}/live"
        return f"https://www.youtube.com/@{n.lstrip('@')}/live"

    @property
    def chat_url(self) -> str:
        if self.platform == "twitch":
            return f"https://www.twitch.tv/{self.name.lower()}"
        return self.url


@dataclass
class Settings:
    # general
    data_dir: Path = Path("livewatch_data")
    buffer_seconds: int = 600
    quality: str = "720p60,720p,best"
    check_interval: int = 90
    max_live_channels: int = 3
    keep_raw_clips: bool = False
    # clip
    pre_seconds: float = 30.0
    post_seconds: float = 12.0
    min_seconds: float = 15.0
    max_seconds: float = 75.0
    cooldown_seconds: float = 150.0
    max_clips_per_hour: int = 6
    # detect
    threshold: float = 3.5
    warmup_seconds: float = 120.0
    min_chat_rate: float = 0.3  # messages/s; quieter chats lean on audio
    audio_weight: float = 1.0
    # style
    caption_style: str = "ooc"
    rotate_styles: bool = False
    description_style: str = "normal"
    rotate_descriptions: bool = False
    captions: bool = True
    hook: bool = True
    whisper_model: str = "base"
    fps: int = 30
    encoder: str = "auto"  # auto | qsv | x264
    # music
    music_mode: str = "off"        # off | always | quiet (only clips with little speech)
    music_volume_db: float = -20.0
    music_dir: Optional[Path] = None  # default: <data_dir>/music
    # ai
    ai_model: str = "gemini-3.1-flash-lite"
    # post
    post_enabled: bool = False
    post_platforms: List[str] = field(default_factory=lambda: ["youtube", "tiktok", "instagram"])
    upload_post_user: str = ""
    min_gap_minutes: int = 20
    daily_cap: int = 12
    skip_spoilers: bool = True
    base_hashtags: List[str] = field(default_factory=lambda: ["gta6", "gtavi", "gta"])
    channels: List[Channel] = field(default_factory=list)
    # secrets (environment only)
    gemini_api_key: str = ""
    upload_post_api_key: str = ""
    twitch_chat_token: str = ""

    def style_for(self, channel: Channel, n: int = 0) -> str:
        if channel.caption_style:
            return channel.caption_style
        if self.rotate_styles:
            return CAPTION_STYLES[n % len(CAPTION_STYLES)]
        return self.caption_style

    def description_for(self, n: int = 0) -> str:
        if self.rotate_descriptions:
            return DESCRIPTION_STYLES[n % len(DESCRIPTION_STYLES)]
        return self.description_style


def load_dotenv(path: Path) -> None:
    """``KEY=value`` lines into ``os.environ`` without overriding what is set."""
    if not path.is_file():
        return
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k = k.strip().removeprefix("export ").strip()
        v = v.strip()
        if len(v) >= 2 and v[0] == v[-1] and v[0] in "\"'":
            v = v[1:-1]
        if k and k not in os.environ:
            os.environ[k] = v


def _num(section: dict, key: str, default, lo, hi, cast=float):
    if key not in section:
        return default
    try:
        v = cast(section[key])
    except (TypeError, ValueError):
        raise ConfigError(f"{key} must be a number")
    if not (lo <= v <= hi):
        raise ConfigError(f"{key} must be between {lo} and {hi}")
    return v


def _str_list(value, what: str) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, (list, tuple)) or not all(isinstance(v, str) for v in value):
        raise ConfigError(f"{what} must be a list of strings")
    return [v.strip() for v in value if v.strip()]


def parse_channel(raw: dict) -> Channel:
    platform = str(raw.get("platform", "")).strip().lower()
    if platform not in PLATFORMS:
        raise ConfigError(f"channel platform must be one of {', '.join(PLATFORMS)}, got {platform!r}")
    name = str(raw.get("name", "")).strip()
    if platform == "youtube" and name.startswith("https://"):
        if not re.match(r"^https://(www\.)?youtube\.com/", name):
            raise ConfigError("a YouTube channel URL must be on youtube.com")
    elif not _NAME_RE.match(name):
        raise ConfigError(f"invalid channel name {name!r}")
    facecam = raw.get("facecam")
    if facecam is not None:
        if (not isinstance(facecam, (list, tuple)) or len(facecam) != 4
                or not all(isinstance(v, (int, float)) for v in facecam)):
            raise ConfigError(f"{name}: facecam must be [x, y, w, h] fractions")
        x, y, w, h = (float(v) for v in facecam)
        if not (0 <= x < 1 and 0 <= y < 1 and 0.05 <= w <= 1 and 0.05 <= h <= 1
                and x + w <= 1.0001 and y + h <= 1.0001):
            raise ConfigError(f"{name}: facecam fractions must stay inside the frame")
        facecam = [x, y, w, h]
    style = str(raw.get("caption_style", "")).strip().lower()
    if style == "chinese":  # removed style in older configs: fall back to the global one
        style = ""
    if style and style not in CAPTION_STYLES:
        raise ConfigError(f"{name}: caption_style must be one of {', '.join(CAPTION_STYLES)}")
    return Channel(
        platform=platform, name=name, label=str(raw.get("label", "")).strip(),
        game_filter=_str_list(raw.get("game_filter"), "game_filter"),
        facecam=facecam,
        tags=[t.lstrip("#") for t in _str_list(raw.get("tags"), "tags")],
        language=str(raw.get("language", "auto")).strip() or "auto",
        caption_style=style,
    )


def from_dict(data: dict, base_dir: Path = Path(".")) -> Settings:
    s = Settings()
    g = data.get("general", {}) or {}
    c = data.get("clip", {}) or {}
    d = data.get("detect", {}) or {}
    st = data.get("style", {}) or {}
    ai = data.get("ai", {}) or {}
    p = data.get("post", {}) or {}

    if "data_dir" in g:
        s.data_dir = Path(str(g["data_dir"]))
    if not s.data_dir.is_absolute():
        s.data_dir = (base_dir / s.data_dir).resolve()
    s.buffer_seconds = _num(g, "buffer_seconds", s.buffer_seconds, 120, 3600, int)
    s.quality = str(g.get("quality", s.quality))
    if not re.match(r"^[a-z0-9_,]+$", s.quality):
        raise ConfigError("quality must look like 720p60,720p,best")
    s.check_interval = _num(g, "check_interval", s.check_interval, 30, 3600, int)
    s.max_live_channels = _num(g, "max_live_channels", s.max_live_channels, 1, 10, int)
    s.keep_raw_clips = bool(g.get("keep_raw_clips", s.keep_raw_clips))

    s.pre_seconds = _num(c, "pre_seconds", s.pre_seconds, 3, 120)
    s.post_seconds = _num(c, "post_seconds", s.post_seconds, 0, 60)
    s.min_seconds = _num(c, "min_seconds", s.min_seconds, 5, 180)
    s.max_seconds = _num(c, "max_seconds", s.max_seconds, 10, 180)
    if s.min_seconds > s.max_seconds:
        raise ConfigError("min_seconds must not exceed max_seconds")
    if s.pre_seconds + s.post_seconds > s.max_seconds:
        raise ConfigError("pre_seconds + post_seconds must fit in max_seconds")
    if s.pre_seconds + s.post_seconds + 20 > s.buffer_seconds:
        raise ConfigError("buffer_seconds is too short for the clip window")
    s.cooldown_seconds = _num(c, "cooldown_seconds", s.cooldown_seconds, 10, 3600)
    s.max_clips_per_hour = _num(c, "max_clips_per_hour", s.max_clips_per_hour, 1, 60, int)

    s.threshold = _num(d, "threshold", s.threshold, 1.0, 20.0)
    s.warmup_seconds = _num(d, "warmup_seconds", s.warmup_seconds, 0, 1800)
    s.min_chat_rate = _num(d, "min_chat_rate", s.min_chat_rate, 0, 100)
    s.audio_weight = _num(d, "audio_weight", s.audio_weight, 0, 5)

    style = str(st.get("caption_style", s.caption_style)).strip().lower()
    if style == "chinese":  # removed style in older configs
        style = "ooc"
    if style == "rotate":
        s.rotate_styles = True
    elif style in CAPTION_STYLES:
        s.caption_style = style
    else:
        raise ConfigError(f"caption_style must be one of {', '.join(CAPTION_STYLES + ('rotate',))}")
    dstyle = str(st.get("description_style", s.description_style)).strip().lower()
    if dstyle == "chinese":  # removed style
        dstyle = "normal"
    if dstyle == "rotate":
        s.rotate_descriptions = True
    elif dstyle in DESCRIPTION_STYLES:
        s.description_style = dstyle
    else:
        raise ConfigError("description_style must be one of "
                          f"{', '.join(DESCRIPTION_STYLES + ('rotate',))}")
    s.captions = bool(st.get("captions", s.captions))
    s.hook = bool(st.get("hook", s.hook))
    s.whisper_model = str(st.get("whisper_model", s.whisper_model))
    if s.whisper_model not in ("off", "tiny", "base", "small", "medium",
                               "tiny.en", "base.en", "small.en", "large-v3-turbo"):
        raise ConfigError("whisper_model must be off, tiny, base, small or medium")
    s.fps = _num(st, "fps", s.fps, 24, 60, int)
    s.encoder = str(st.get("encoder", s.encoder)).lower()
    if s.encoder not in ("auto", "qsv", "x264"):
        raise ConfigError("encoder must be auto, qsv or x264")

    mu = data.get("music", {}) or {}
    s.music_mode = str(mu.get("mode", s.music_mode)).strip().lower()
    if s.music_mode not in ("off", "always", "quiet"):
        raise ConfigError("music mode must be off, always or quiet")
    s.music_volume_db = _num(mu, "volume_db", s.music_volume_db, -40, 0)
    folder = str(mu.get("folder", "") or "").strip()
    s.music_dir = Path(folder) if folder else s.data_dir / "music"
    if not s.music_dir.is_absolute():
        s.music_dir = (base_dir / s.music_dir).resolve()

    s.ai_model = str(ai.get("model", os.environ.get("GEMINI_MODEL", s.ai_model)))

    s.post_enabled = bool(p.get("enabled", s.post_enabled))
    if "platforms" in p:
        plats = [x.lower() for x in _str_list(p["platforms"], "post.platforms")]
        bad = [x for x in plats if x not in POST_PLATFORMS]
        if bad:
            raise ConfigError(f"unknown post platforms: {', '.join(bad)}")
        s.post_platforms = plats
    s.upload_post_user = str(p.get("upload_post_user", s.upload_post_user)).strip()
    s.min_gap_minutes = _num(p, "min_gap_minutes", s.min_gap_minutes, 0, 1440, int)
    s.daily_cap = _num(p, "daily_cap", s.daily_cap, 1, 200, int)
    s.skip_spoilers = bool(p.get("skip_spoilers", s.skip_spoilers))
    if "base_hashtags" in p:
        s.base_hashtags = [t.lstrip("#") for t in _str_list(p["base_hashtags"], "base_hashtags")]

    chans = data.get("channel", []) or []
    if not isinstance(chans, list):
        raise ConfigError("[[channel]] entries expected")
    s.channels = [parse_channel(ch) for ch in chans]
    seen = set()
    for ch in s.channels:
        if ch.key in seen:
            raise ConfigError(f"channel listed twice: {ch.key}")
        seen.add(ch.key)

    s.gemini_api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    s.upload_post_api_key = (os.environ.get("UPLOADPOST_API_KEY")
                             or os.environ.get("UPLOAD_POST_API_KEY") or "").strip()
    s.twitch_chat_token = os.environ.get("TWITCH_CHAT_TOKEN", "").strip()
    return s


def load(path: Path) -> Settings:
    path = Path(path)
    if not path.is_file():
        raise ConfigError(f"config not found: {path} (copy livewatch.example.toml)")
    load_dotenv(path.parent / ".env")
    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as e:
        raise ConfigError(f"{path}: {e}")
    return from_dict(data, base_dir=path.parent.resolve())


def matches_game(category: Optional[str], filters: Sequence[str]) -> bool:
    if not filters:
        return True
    if not category:
        return False
    cat = category.lower()
    return any(f.lower() in cat for f in filters)
