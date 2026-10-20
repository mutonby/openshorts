"""Discord adapter — Webhook API (no bot token required).

Env (per account suffix; uppercase):
  DISCORD_WEBHOOK_URL[_<ACCT>]  — full webhook URL from Channel → Integrations → Webhooks

Format key: `discord_message`
  content     (required) — message text, up to 2000 chars
  username    (optional) — override the webhook display name
  avatar_url  (optional) — override the webhook avatar
  embeds      (optional) — list of Discord embed objects (dicts)

Discord webhook limits: 2000 chars for content, 10 embeds per message.
"""

from __future__ import annotations

import logging
import os

import httpx

from opendispatch.adapters.errors import classify, clip, prefix_error
from opendispatch.api.schema import ContentUnit

log = logging.getLogger("open-dispatch.discord")


def _webhook_url(account: str | None) -> str:
    suffix = f"_{account.upper()}" if account else ""
    return os.getenv(f"DISCORD_WEBHOOK_URL{suffix}") or os.getenv("DISCORD_WEBHOOK_URL", "")


def publish(unit: ContentUnit, account: str | None = None, *, creds=None) -> tuple[bool, str, str]:
    webhook_url = _webhook_url(account)
    if not webhook_url:
        return False, "", "DISCORD_WEBHOOK_URL missing"

    fmt = unit.formats.get("discord_message") or {}
    content = (fmt.get("content") or "").strip()[:2000]
    username = fmt.get("username")
    avatar_url = fmt.get("avatar_url")
    embeds = fmt.get("embeds") or []

    if not content and not embeds:
        return False, "", "discord_message.content or embeds is required"

    body: dict = {}
    if content:
        body["content"] = content
    if username:
        body["username"] = username
    if avatar_url:
        body["avatar_url"] = avatar_url
    if embeds:
        body["embeds"] = embeds[:10]

    try:
        # ?wait=true makes Discord return the message object (gives us a message ID)
        r = httpx.post(f"{webhook_url}?wait=true", json=body, timeout=15)
        r.raise_for_status()
        message_id = str(r.json().get("id", ""))
        return True, message_id, ""

    except httpx.HTTPStatusError as e:
        return False, "", prefix_error(classify(e.response.status_code), f"HTTP {e.response.status_code}: {clip(e.response.text)}")
    except Exception as e:  # noqa: BLE001
        return False, "", f"discord error: {e}"
