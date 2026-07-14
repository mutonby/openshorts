"""Resolution of server-owned (managed) provider keys for entitled users.

Only Gemini and Upload-Post are managed in v1. ElevenLabs and fal stay BYOK.
The functions here are sync and read from the ``CurrentUser`` snapshot so they
can be called inline from ``app.py``'s resolve helpers without awaiting.
"""
from typing import Optional
from .config import settings


def has_active_entitlement(user) -> bool:
    """True if this user may consume managed keys (active plan or top-up credit).

    The ``entitled`` flag is computed once in Fase 2/3 when the user is loaded
    (active subscription OR remaining top-up minutes).
    """
    return bool(user is not None and getattr(user, "entitled", False))


def gemini_key() -> Optional[str]:
    return settings.managed_gemini_key or None


def upload_post_key() -> Optional[str]:
    return settings.managed_upload_post_key or None
