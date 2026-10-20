"""TikTok adapter — Content Posting API v2.

Env (per account suffix; uppercase):
  TIKTOK_ACCESS_TOKEN[_<ACCT>]   — OAuth 2.0 user token (scope: video.publish)
  TIKTOK_PRIVACY_LEVEL           — default privacy: PUBLIC_TO_EVERYONE | MUTUAL_FOLLOW_FRIENDS | SELF_ONLY
                                    (default: PUBLIC_TO_EVERYONE)

Format key: `tiktok_post`
  video_url  (required) — publicly accessible URL to the .mp4; TikTok pulls it directly
  caption    (optional) — up to 2200 chars
  privacy    (optional) — overrides TIKTOK_PRIVACY_LEVEL for this post

Requires the TikTok Content Posting API v2 (apply at developers.tiktok.com).
Access token must have scope: video.publish
"""

from __future__ import annotations

import logging
import os
import time

import httpx

from opendispatch.adapters.errors import PUBLISHED, classify, clip, prefix_error
from opendispatch.api.schema import ContentUnit

log = logging.getLogger("open-dispatch.tiktok")

_BASE = "https://open.tiktokapis.com/v2"


def _token(account: str | None) -> str:
    suffix = f"_{account.upper()}" if account else ""
    return os.getenv(f"TIKTOK_ACCESS_TOKEN{suffix}") or os.getenv("TIKTOK_ACCESS_TOKEN", "")


def publish(unit: ContentUnit, account: str | None = None, *, creds=None) -> tuple[bool, str, str]:
    token = _token(account)
    if not token:
        return False, "", "TIKTOK_ACCESS_TOKEN missing"

    fmt = unit.formats.get("tiktok_post") or {}
    video_url = (fmt.get("video_url") or "").strip()
    if not video_url:
        return False, "", "tiktok_post.video_url is required"

    caption = (fmt.get("caption") or "")[:2200]
    privacy = (
        fmt.get("privacy")
        or os.getenv("TIKTOK_PRIVACY_LEVEL", "PUBLIC_TO_EVERYONE")
    )

    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json; charset=UTF-8",
    }

    try:
        # Step 1 — initialise the post (PULL_FROM_URL = TikTok fetches video itself)
        init_r = httpx.post(
            f"{_BASE}/post/publish/video/init/",
            headers=headers,
            json={
                "post_info": {
                    "title": caption,
                    "privacy_level": privacy,
                    "disable_duet": False,
                    "disable_comment": False,
                    "disable_stitch": False,
                    "video_cover_timestamp_ms": 1000,
                },
                "source_info": {
                    "source": "PULL_FROM_URL",
                    "video_url": video_url,
                },
            },
            timeout=30,
        )
        init_r.raise_for_status()
        init_data = init_r.json()

        if init_data.get("error", {}).get("code", "ok") != "ok":
            msg = init_data["error"].get("message", "unknown error")
            return False, "", f"TikTok init error: {msg}"

        publish_id = init_data.get("data", {}).get("publish_id", "")
        if not publish_id:
            return False, "", "TikTok returned no publish_id"

        # Step 2 — poll for publish status (TikTok processes async)
        for _ in range(12):
            time.sleep(5)
            status_r = httpx.post(
                f"{_BASE}/post/publish/status/fetch/",
                headers=headers,
                json={"publish_id": publish_id},
                timeout=15,
            )
            status_r.raise_for_status()
            status_data = status_r.json()
            # A status response can carry an error envelope with HTTP 200.
            err_env = status_data.get("error")
            if err_env:
                code = err_env.get("code", "?")
                msg = err_env.get("message", "unknown")
                cls = classify(429 if str(code) == "22000" else None)
                return False, "", prefix_error(
                    cls, f"TikTok status error {code}: {msg} (publish_id={publish_id})")
            status = (status_data.get("data") or {}).get("status", "")
            if status == "PUBLISH_COMPLETE":
                # The field is a list in the API, but be defensive: a bare
                # string would index to its first CHARACTER and be returned as
                # the post id, and an empty list raised IndexError.
                ids = (status_data.get("data") or {}).get("publicaly_available_post_id") or []
                if isinstance(ids, str):
                    ids = [ids]
                if not ids:
                    return False, "", prefix_error(
                        PUBLISHED,
                        f"TikTok reports complete but no post id (publish_id={publish_id})")
                return True, str(ids[0]), ""
            if status in ("FAILED", "PUBLISH_FAILED"):
                reason = (status_data.get("data") or {}).get("fail_reason", "unknown")
                return False, "", prefix_error(
                    "retryable", f"TikTok publish failed: {reason} (publish_id={publish_id})")

        # Polls exhausted. The old code returned ok=True with publish_id, so a
        # video that never published was marked published and a published
        # webhook fired for it. publish_id is NOT a post id. Report it as
        # published-but-unconfirmed so the worker never retries into a double.
        return False, "", prefix_error(
            PUBLISHED,
            f"TikTok still processing after 12 polls; publish_id={publish_id} (not a post id)",
        )

    except httpx.HTTPStatusError as e:
        return False, "", prefix_error(
            classify(e.response.status_code),
            f"HTTP {e.response.status_code}: {clip(e.response.text)}",
        )
    except Exception as e:  # noqa: BLE001
        return False, "", prefix_error("retryable", f"tiktok error: {clip(e)}")
