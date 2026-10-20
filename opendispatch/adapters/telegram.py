"""Telegram adapter — text/photo/video via Bot API.

Env: TELEGRAM_BOT_TOKEN (required), TELEGRAM_CHAT_ID (required for default account).
Per-account override: TELEGRAM_CHAT_ID_<ACCOUNT> when target is `telegram:<account>`.

Format key: `telegram_message`
  text (required), photo_path (optional), video_path (optional), parse_mode (default "HTML")
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import httpx

from opendispatch.adapters.errors import AUTH, PERMANENT, PUBLISHED, clip, classify, prefix_error
from opendispatch.api.schema import CAPTION_LIMITS, ContentUnit
from opendispatch.media.paths import resolve_media_path

log = logging.getLogger("open-dispatch.telegram")


def _chunk(text: str, limit: int) -> list[str]:
    if len(text) <= limit:
        return [text]
    return [text[i:i + limit] for i in range(0, len(text), limit)]


def _result_message_id(r: Any) -> tuple[str, str]:
    """Return (message_id, error). Telegram reports failure as HTTP 200 with
    `"ok": false` in the body, so raise_for_status() alone reports success for
    a message that was never sent. Every response must go through this."""
    try:
        body = r.json()
    except ValueError:
        return "", f"telegram: non-JSON response: {clip(r.text)}"
    if not body.get("ok"):
        code = body.get("error_code", "?")
        desc = body.get("description", "unknown error")
        cls = AUTH if code in (401, 403) else PERMANENT if code in (400, 404) else "retryable"
        return "", prefix_error(cls, f"telegram {code}: {desc}")
    try:
        return str(body["result"]["message_id"]), ""
    except (KeyError, TypeError):
        return "", prefix_error("published", f"telegram: sent but no message_id: {clip(r.text)}")


def _chat_id(account: str | None) -> str:
    if account:
        v = os.getenv(f"TELEGRAM_CHAT_ID_{account.upper()}")
        if v:
            return v
    return os.getenv("TELEGRAM_CHAT_ID", "")


def publish(unit: ContentUnit, account: str | None = None, *, creds=None) -> tuple[bool, str, str]:
    token = os.getenv("TELEGRAM_BOT_TOKEN", "")
    chat_id = _chat_id(account)
    if not (token and chat_id):
        return False, "", "TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID missing"

    data = unit.formats.get("telegram_message") or {}
    text = (data.get("text") or data.get("caption") or "").strip()
    parse_mode = data.get("parse_mode", "HTML")
    photo = data.get("photo_path")
    video = data.get("video_path")

    base = f"https://api.telegram.org/bot{token}"
    try:
        if photo:
            p = resolve_media_path(photo, strict_root=True)
            with p.open("rb") as f:
                r = httpx.post(
                    f"{base}/sendPhoto",
                    data={"chat_id": chat_id, "caption": text[:1024], "parse_mode": parse_mode},
                    files={"photo": f},
                    timeout=60,
                )
            r.raise_for_status()
            msg_id, perr = _result_message_id(r)
            if perr:
                return False, "", perr
            if len(text) > 1024:
                for chunk in _chunk(text[1024:], CAPTION_LIMITS["telegram"]):
                    cr = httpx.post(f"{base}/sendMessage",
                                    data={"chat_id": chat_id, "text": chunk, "parse_mode": parse_mode},
                                    timeout=30)
                    cr.raise_for_status()
                    _cid, cerr = _result_message_id(cr)
                    if cerr:
                        # The media (and possibly earlier chunks) already went
                        # out. Retrying would duplicate them.
                        return False, "", prefix_error(
                            PUBLISHED,
                            f"telegram: media published but caption chunk failed, "
                            f"NOT retried to avoid duplicates: {cerr}",
                        )
            return True, msg_id, ""

        if video:
            v = resolve_media_path(video, strict_root=True)
            with v.open("rb") as f:
                r = httpx.post(
                    f"{base}/sendVideo",
                    data={"chat_id": chat_id, "caption": text[:1024], "parse_mode": parse_mode},
                    files={"video": f},
                    timeout=120,
                )
            r.raise_for_status()
            msg_id, perr = _result_message_id(r)
            if perr:
                return False, "", perr
            if len(text) > 1024:
                for chunk in _chunk(text[1024:], CAPTION_LIMITS["telegram"]):
                    cr = httpx.post(f"{base}/sendMessage",
                                    data={"chat_id": chat_id, "text": chunk, "parse_mode": parse_mode},
                                    timeout=30)
                    cr.raise_for_status()
                    _cid, cerr = _result_message_id(cr)
                    if cerr:
                        # The media (and possibly earlier chunks) already went
                        # out. Retrying would duplicate them.
                        return False, "", prefix_error(
                            PUBLISHED,
                            f"telegram: media published but caption chunk failed, "
                            f"NOT retried to avoid duplicates: {cerr}",
                        )
            return True, msg_id, ""

        if not text:
            return False, "", "text empty"
        first_id = None
        sent_chunks = 0
        chunks = _chunk(text, CAPTION_LIMITS["telegram"])
        total_chunks = len(chunks)
        for chunk in chunks:
            r = httpx.post(
                f"{base}/sendMessage",
                data={"chat_id": chat_id, "text": chunk, "parse_mode": parse_mode},
                timeout=30,
            )
            try:
                r.raise_for_status()
            except httpx.HTTPStatusError as e:
                # An HTTP failure mid-sequence means earlier chunks already
                # went out; never let this be retried into duplicates.
                detail = prefix_error(classify(e.response.status_code),
                                      f"HTTP {e.response.status_code}: {clip(e.response.text)}")
                if sent_chunks:
                    return False, "", prefix_error(
                        PUBLISHED,
                        f"telegram: {sent_chunks}/{total_chunks} chunks sent, "
                        f"NOT retried to avoid duplicates: {detail}",
                    )
                return False, "", detail
            chunk_id, perr = _result_message_id(r)
            if perr:
                if sent_chunks:
                    # Earlier chunks are already in the chat; a retry would
                    # re-send them.
                    return False, "", prefix_error(
                        PUBLISHED,
                        f"telegram: {sent_chunks}/{total_chunks} chunks sent, "
                        f"NOT retried to avoid duplicates: {perr}",
                    )
                return False, "", perr
            sent_chunks += 1
            if first_id is None:
                first_id = chunk_id
        if not first_id:
            return False, "", prefix_error("published", "telegram: sent but no message_id")
        return True, first_id, ""

    except httpx.HTTPStatusError as e:
        # Build the message from the response, never from str(e): the exception
        # interpolates the full URL and the bot token lives in its path.
        return False, "", prefix_error(
            classify(e.response.status_code),
            f"HTTP {e.response.status_code}: {clip(e.response.text)}",
        )
    except Exception as e:  # noqa: BLE001
        return False, "", prefix_error("retryable", f"telegram error: {clip(e)}")
