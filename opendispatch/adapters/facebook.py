"""Facebook adapter — Pages API via Meta Graph API v19.

Env (per account suffix; uppercase):
  FACEBOOK_PAGE_ID[_<ACCT>]       — numeric Page ID
  FACEBOOK_ACCESS_TOKEN[_<ACCT>] — Page access token (never a user token)

Format key: `facebook_post`
  text       (required for text-only posts)
  image_url  (optional) — public URL for a photo post
  video_url  (optional) — public URL for a video post; takes precedence over image_url
  link       (optional) — URL to attach as a link preview (text posts only)
"""

from __future__ import annotations

import logging
import os

import httpx

from opendispatch.adapters.errors import classify, clip, prefix_error
from opendispatch.api.schema import ContentUnit

log = logging.getLogger("open-dispatch.facebook")

_BASE = "https://graph.facebook.com/v19.0"


def _creds(account: str | None) -> tuple[str, str]:
    suffix = f"_{account.upper()}" if account else ""
    page_id = os.getenv(f"FACEBOOK_PAGE_ID{suffix}") or os.getenv("FACEBOOK_PAGE_ID", "")
    token = os.getenv(f"FACEBOOK_ACCESS_TOKEN{suffix}") or os.getenv("FACEBOOK_ACCESS_TOKEN", "")
    return page_id, token


def publish(unit: ContentUnit, account: str | None = None, *, creds=None) -> tuple[bool, str, str]:
    page_id, token = _creds(account)
    if not (page_id and token):
        return False, "", "FACEBOOK_PAGE_ID / FACEBOOK_ACCESS_TOKEN missing"

    fmt = unit.formats.get("facebook_post") or {}
    text = (fmt.get("text") or "").strip()
    image_url = (fmt.get("image_url") or "").strip()
    video_url = (fmt.get("video_url") or "").strip()
    link = (fmt.get("link") or "").strip()

    params = {"access_token": token}

    try:
        if video_url:
            # Video post — use /videos endpoint
            r = httpx.post(
                f"{_BASE}/{page_id}/videos",
                params=params,
                json={"file_url": video_url, "description": text},
                timeout=60,
            )
            r.raise_for_status()
            post_id = str(r.json().get("id", ""))
            return True, post_id, ""

        if image_url:
            # Photo post
            r = httpx.post(
                f"{_BASE}/{page_id}/photos",
                params=params,
                json={"url": image_url, "caption": text},
                timeout=30,
            )
            r.raise_for_status()
            post_id = str(r.json().get("post_id") or r.json().get("id", ""))
            return True, post_id, ""

        # Text (+ optional link preview)
        if not text:
            return False, "", "facebook_post.text is required for text-only posts"
        body: dict = {"message": text}
        if link:
            body["link"] = link
        r = httpx.post(
            f"{_BASE}/{page_id}/feed",
            params=params,
            json=body,
            timeout=30,
        )
        r.raise_for_status()
        post_id = str(r.json().get("id", ""))
        return True, post_id, ""

    except httpx.HTTPStatusError as e:
        body_text = clip(e.response.text)
        return False, "", f"HTTP {e.response.status_code}: {body_text}"
    except Exception as e:  # noqa: BLE001
        return False, "", f"facebook error: {e}"
