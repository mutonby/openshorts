"""Profile store — named sets of platform credentials.

A profile groups per-platform credentials under a friendly name so users can
dispatch to different accounts (personal, work, brand X…) without touching .env.

Profiles are persisted as plain JSON in $OPEN_DISPATCH_DATA/profiles.json.
Security note: credentials are stored in plaintext — protect the data directory.

Thread safety: profile_env() uses a lock so the worker never races when
multiple rows referencing different profiles are processed in quick succession.
"""

from __future__ import annotations

import json
import os
import threading
import uuid
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterator


# ── credential field → ENV_VAR for each platform ──────────────────────────────

PLATFORM_CRED_MAP: dict[str, dict[str, str]] = {
    "twitter": {
        "api_key":       "TWITTER_API_KEY",
        "api_secret":    "TWITTER_API_SECRET",
        "access_token":  "TWITTER_ACCESS_TOKEN",
        "access_secret": "TWITTER_ACCESS_SECRET",
    },
    "bluesky": {
        "handle":       "BLUESKY_HANDLE",
        "app_password": "BLUESKY_APP_PASSWORD",
    },
    "telegram": {
        "bot_token": "TELEGRAM_BOT_TOKEN",
        "chat_id":   "TELEGRAM_CHAT_ID",
    },
    "instagram": {
        "user_id": "IG_USER_ID",
        "token":   "IG_TOKEN",
    },
    "linkedin": {
        "access_token": "LINKEDIN_ACCESS_TOKEN",
        "author_urn":   "LINKEDIN_AUTHOR_URN",
    },
    "threads": {
        "user_id":      "THREADS_USER_ID",
        "access_token": "THREADS_ACCESS_TOKEN",
    },
    "youtube": {
        "client_id":     "YOUTUBE_CLIENT_ID",
        "client_secret": "YOUTUBE_CLIENT_SECRET",
        "refresh_token": "YOUTUBE_REFRESH_TOKEN",
    },
    "tiktok": {
        "access_token": "TIKTOK_ACCESS_TOKEN",
    },
    "facebook": {
        "page_id":      "FACEBOOK_PAGE_ID",
        "access_token": "FACEBOOK_ACCESS_TOKEN",
    },
    "discord": {
        "webhook_url": "DISCORD_WEBHOOK_URL",
    },
}

# Human-readable labels for the form
CRED_LABELS: dict[str, str] = {
    "api_key":       "API Key",
    "api_secret":    "API Secret",
    "access_token":  "Access Token",
    "access_secret": "Access Token Secret",
    "handle":        "Handle (e.g. user.bsky.social)",
    "app_password":  "App Password",
    "bot_token":     "Bot Token",
    "chat_id":       "Chat ID or @channel",
    "user_id":       "User ID",
    "token":         "Access Token",
    "author_urn":    "Author URN (urn:li:person:…)",
    "client_id":     "OAuth Client ID",
    "client_secret": "OAuth Client Secret",
    "refresh_token": "OAuth Refresh Token",
    "page_id":       "Facebook Page ID (numeric)",
    "webhook_url":   "Discord Webhook URL",
}

# Fields that should be rendered as password inputs
PASSWORD_FIELDS = {
    "api_secret", "access_token", "access_secret", "app_password",
    "bot_token", "token", "client_secret", "refresh_token",
    "webhook_url",
}

# Nice emoji per platform
PLATFORM_EMOJI: dict[str, str] = {
    "twitter":   "𝕏",
    "bluesky":   "🦋",
    "telegram":  "✈️",
    "instagram": "📷",
    "linkedin":  "💼",
    "threads":   "🧵",
    "youtube":    "▶️",
    "tiktok":     "🎵",
    "facebook":   "📘",
    "discord":    "💬",
}

PRESET_COLORS = ["#4ade80", "#60a5fa", "#f472b6", "#fb923c", "#a78bfa", "#34d399", "#facc15"]


# ── Profile dataclass ──────────────────────────────────────────────────────────

@dataclass
class Profile:
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    name: str = ""
    emoji: str = "🎭"
    color: str = "#4ade80"
    # {platform: {field_name: value}}
    platforms: dict[str, dict[str, str]] = field(default_factory=dict)

    def configured_platforms(self) -> list[str]:
        """Platforms that have at least one non-empty credential field."""
        return [
            p for p, creds in self.platforms.items()
            if any(v.strip() for v in creds.values())
        ]

    def to_env(self) -> dict[str, str]:
        """Flat {ENV_VAR: value} dict for all non-empty credentials."""
        result: dict[str, str] = {}
        for platform, creds in self.platforms.items():
            for field_name, env_var in PLATFORM_CRED_MAP.get(platform, {}).items():
                value = creds.get(field_name, "").strip()
                if value:
                    result[env_var] = value
        return result

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Profile":
        return cls(
            id=d.get("id") or str(uuid.uuid4())[:8],
            name=d.get("name", ""),
            emoji=d.get("emoji", "🎭"),
            color=d.get("color", "#4ade80"),
            platforms={
                p: dict(creds)
                for p, creds in (d.get("platforms") or {}).items()
            },
        )


# ── Persistent store ───────────────────────────────────────────────────────────

class ProfileStore:
    def __init__(self) -> None:
        data_dir = Path(os.getenv("OPEN_DISPATCH_DATA", str(Path.home() / ".open-dispatch")))
        self._path = data_dir / "profiles.json"

    def _load(self) -> list[Profile]:
        if not self._path.exists():
            return []
        try:
            raw = json.loads(self._path.read_text())
            return [Profile.from_dict(p) for p in raw.get("profiles", [])]
        except Exception:
            return []

    def _save(self, profiles: list[Profile]) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.chmod(self._path.parent, 0o700)
        except OSError:
            pass
        self._path.write_text(
            json.dumps({"profiles": [p.to_dict() for p in profiles]}, indent=2, ensure_ascii=False)
        )
        try:
            os.chmod(self._path, 0o600)
        except OSError:
            pass

    def list(self) -> list[Profile]:
        return self._load()

    def get(self, profile_id: str) -> Profile | None:
        for p in self._load():
            if p.id == profile_id:
                return p
        return None

    def save(self, profile: Profile) -> Profile:
        profiles = self._load()
        ids = [p.id for p in profiles]
        if profile.id in ids:
            profiles = [profile if p.id == profile.id else p for p in profiles]
        else:
            profiles.append(profile)
        self._save(profiles)
        return profile

    def delete(self, profile_id: str) -> bool:
        profiles = self._load()
        new = [p for p in profiles if p.id != profile_id]
        if len(new) == len(profiles):
            return False
        self._save(new)
        return True


# ── Env injection context manager ─────────────────────────────────────────────

_env_lock = threading.Lock()


@contextmanager
def profile_env(profile: Profile | None) -> Iterator[None]:
    """Temporarily inject profile credentials into os.environ.

    Uses a threading lock so the worker never corrupts env state when
    processing rows for different profiles concurrently.
    """
    if profile is None:
        yield
        return

    overrides = profile.to_env()
    if not overrides:
        yield
        return

    with _env_lock:
        saved = {k: os.environ.get(k) for k in overrides}
        os.environ.update(overrides)
        try:
            yield
        finally:
            for k, v in saved.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v
