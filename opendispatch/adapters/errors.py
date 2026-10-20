"""Shared adapter error helpers — redaction and retry classification.

Two things every adapter needs and none had:

1. **Redaction.** Telegram puts the bot token in the URL *path*, Instagram put
   an access token in the query string. `raise_for_status()` and httpx's INFO
   logger both interpolate the full URL, so a token can reach logs and, in
   some paths, the queue's `last_error` — which the dashboard, the API, and the
   failure webhook all expose. Anything user-visible goes through `redact()`.

2. **Classification.** The worker retries every non-ok result identically
   (`attempts+1`, 30s x 2^n, dead at 3), so a revoked token gets retried twice
   before dead-lettering exactly like a rate limit. `classify()` lets an
   adapter say "auth" or "permanent" and the worker can skip the retries.

Error strings are capped at 500 chars to match what the queue stores
(`api/queue.py` truncates with `err[:500]`); adapters were cutting at 200-400
and losing the actionable tail of Meta/Google error bodies.
"""
from __future__ import annotations

import re
from typing import Any

# 500 chars matches the queue's own storage cap; adapters were cutting lower.
MAX_ERR_CHARS = 500

# Telegram-style bot tokens: 123456789:AA...  Also catches any long opaque
# token that follows a known secret-ish parameter name.
_TELEGRAM_TOKEN = re.compile(r"\b/?bot([0-9]+):[A-Za-z0-9_-]{8,}")
_SECRET_PARAM = re.compile(
    r"(?i)\b(access_token|api_key|apikey|client_secret|refresh_token|bot_token"
    r"|app_password|token|secret)=([^&\s\"'\\]+)"
)
_BEARER = re.compile(r"(?i)\b(Bearer|Basic)\s+[A-Za-z0-9._~+/=-]{12,}")


def redact(text: Any) -> str:
    """Strip credentials from anything that may reach a log, the API, or a webhook.

    Applied to error bodies and messages, not to successful payloads.
    """
    s = "" if text is None else str(text)
    s = _TELEGRAM_TOKEN.sub("bot***", s)
    s = _SECRET_PARAM.sub(lambda m: f"{m.group(1)}=***", s)
    s = _BEARER.sub(lambda m: f"{m.group(1)} ***", s)
    return s


def clip(text: Any, limit: int = MAX_ERR_CHARS) -> str:
    """Redact then cap, in that order — clip first could split a token in half
    and leave a partial secret in the output."""
    return redact(text)[:limit]


# ── Retry classification ─────────────────────────────────────────────────────
# "retryable" -> the worker backs off and tries again (429, 5xx, network).
# "auth"      -> credentials are wrong; retrying cannot help.
# "permanent" -> the request itself is wrong; retrying cannot help.
# "published" -> the post went out but we could not read the id; must NOT be
#                retried or the user gets a duplicate.
RETRYABLE = "retryable"
AUTH = "auth"
PERMANENT = "permanent"
PUBLISHED = "published"


def classify(status: int | None) -> str:
    """Map an HTTP status onto a retry class.

    401/403 are AUTH rather than PERMANENT so the message can say "check your
    credentials" — the most common misconfiguration, and the one where a retry
    is guaranteed useless.
    """
    if status is None:
        return RETRYABLE
    if status in (401, 403):
        return AUTH
    if status == 429:
        return RETRYABLE
    if status >= 500:
        return RETRYABLE
    if status >= 400:
        return PERMANENT
    return RETRYABLE


def prefix_error(cls: str, message: str) -> str:
    """Tag an error string with its class so the worker can act on it.

    The worker parses this leading token; see `scheduler/worker.py`.
    """
    return f"{cls}: {clip(message)}"
