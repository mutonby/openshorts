"""Credentials the local process already has in its environment.

Nothing here starts an OAuth flow. Put the tokens in the environment, then
the poster reads them. They are never written into the queue.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class InstagramCreds:
    user_id: str
    access_token: str


@dataclass(frozen=True)
class YouTubeCreds:
    client_id: str
    client_secret: str
    refresh_token: str


@dataclass(frozen=True)
class Credentials:
    instagram: InstagramCreds | None = None
    youtube: YouTubeCreds | None = None


def from_env() -> Credentials:
    ig_id = os.getenv("IG_USER_ID", "").strip()
    ig_token = os.getenv("IG_TOKEN", "").strip()
    yt_id = os.getenv("YOUTUBE_CLIENT_ID", "").strip()
    yt_secret = os.getenv("YOUTUBE_CLIENT_SECRET", "").strip()
    yt_refresh = os.getenv("YOUTUBE_REFRESH_TOKEN", "").strip()
    return Credentials(
        instagram=InstagramCreds(ig_id, ig_token) if ig_id and ig_token else None,
        youtube=YouTubeCreds(yt_id, yt_secret, yt_refresh) if yt_id and yt_secret and yt_refresh else None,
    )


def r2_missing() -> list[str]:
    """Env names still empty. Instagram needs every one of them."""
    required = (
        "R2_ENDPOINT",
        "R2_BUCKET",
        "R2_ACCESS_KEY_ID",
        "R2_SECRET_ACCESS_KEY",
        "R2_PUBLIC_BASE",
    )
    return [name for name in required if not os.getenv(name, "").strip()]


def config_flags() -> dict[str, bool]:
    creds = from_env()
    return {
        "youtube": creds.youtube is not None,
        "instagram": creds.instagram is not None,
        "r2": not r2_missing(),
    }
