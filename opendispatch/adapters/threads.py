"""Threads adapter — Meta Threads API via Graph endpoint.

Two-step publish:
  1. POST /{user_id}/threads      → returns creation_id
  2. POST /{user_id}/threads_publish → returns post id

Docs: https://developers.facebook.com/docs/threads/posts/

Env:
  THREADS_USER_ID[_<ACCT>]     — numeric Threads user id
  THREADS_ACCESS_TOKEN[_<ACCT>] — long-lived token with threads_basic +
                                  threads_content_publish

Format key: `threads_post`
  text:  str   (<=500 chars, required for TEXT posts)
  image_url:  str   (optional; public URL for IMAGE post)
  video_url:  str   (optional; public URL for VIDEO post)

If both image_url and video_url are set, video takes precedence.
"""

from __future__ import annotations

import logging
import os
import time

import httpx

from opendispatch.adapters.errors import classify, clip, prefix_error
from opendispatch.api.schema import ContentUnit

log = logging.getLogger("open-dispatch.threads")

GRAPH_BASE = "https://graph.threads.net/v1.0"
PUBLISH_SETTLE_SECONDS = 30  # Meta recommends ~30s before calling threads_publish
# Hard ceiling: the worker is single-threaded, so an unbounded sleep would stop
# the whole queue draining for every platform.
MAX_SETTLE_SECONDS = 120


def _creds(account: str | None) -> tuple[str, str]:
    suffix = f"_{account.upper()}" if account else ""
    user_id = os.getenv(f"THREADS_USER_ID{suffix}") or os.getenv("THREADS_USER_ID", "")
    token = os.getenv(f"THREADS_ACCESS_TOKEN{suffix}") or os.getenv("THREADS_ACCESS_TOKEN", "")
    return user_id, token


def _create_container(
    client: httpx.Client,
    user_id: str,
    token: str,
    *,
    text: str = "",
    image_url: str = "",
    video_url: str = "",
) -> tuple[str | None, str]:
    """Returns (creation_id, error_message). One of them is empty."""
    if video_url:
        media_type = "VIDEO"
    elif image_url:
        media_type = "IMAGE"
    else:
        media_type = "TEXT"

    params: dict[str, str] = {
        "access_token": token,
        "media_type": media_type,
    }
    if text:
        params["text"] = text[:500]
    if media_type == "IMAGE":
        params["image_url"] = image_url
    elif media_type == "VIDEO":
        params["video_url"] = video_url

    try:
        r = client.post(f"{GRAPH_BASE}/{user_id}/threads", data=params, timeout=30)
    except (httpx.HTTPError, httpx.InvalidURL) as e:
        return None, f"network error creating container: {e}"
    if r.status_code >= 400:
        return None, prefix_error(classify(r.status_code), f"create container HTTP {r.status_code}: {clip(r.text)}")
    try:
        cid = r.json().get("id")
    except ValueError:
        return None, "create container: non-JSON response"
    if not cid:
        return None, "create container: missing id in response"
    return cid, ""


def _publish_container(
    client: httpx.Client,
    user_id: str,
    token: str,
    creation_id: str,
) -> tuple[str | None, str]:
    """Returns (post_id, error_message). One of them is empty."""
    try:
        r = client.post(
            f"{GRAPH_BASE}/{user_id}/threads_publish",
            data={"access_token": token, "creation_id": creation_id},
            timeout=30,
        )
    except (httpx.HTTPError, httpx.InvalidURL) as e:
        return None, f"network error publishing: {e}"
    if r.status_code >= 400:
        return None, prefix_error(classify(r.status_code), f"publish HTTP {r.status_code}: {clip(r.text)}")
    try:
        post_id = r.json().get("id")
    except ValueError:
        return None, "publish: non-JSON response"
    if not post_id:
        return None, "publish: missing id in response"
    return post_id, ""


def publish(unit: ContentUnit, account: str | None = None, *, creds=None) -> tuple[bool, str, str]:
    fmt = unit.formats.get("threads_post") or {}
    user_id, token = _creds(account)
    if not (user_id and token):
        return False, "", "THREADS_USER_ID / THREADS_ACCESS_TOKEN missing"

    text = (fmt.get("text") or "").strip()
    image_url = (fmt.get("image_url") or "").strip()
    video_url = (fmt.get("video_url") or "").strip()

    if not (text or image_url or video_url):
        return False, "", "threads_post requires text, image_url, or video_url"

    # A typo here used to raise ValueError out of publish(), which left the row
    # stuck in "publishing" forever (list_due() only returns "queued" rows and
    # nothing reaps a stuck claim), silently losing the post. Clamp it.
    try:
        settle = int(os.getenv("THREADS_SETTLE_SECONDS", "") or PUBLISH_SETTLE_SECONDS)
    except ValueError:
        settle = PUBLISH_SETTLE_SECONDS
    settle = max(0, min(settle, MAX_SETTLE_SECONDS))

    with httpx.Client() as client:
        creation_id, err = _create_container(
            client, user_id, token,
            text=text, image_url=image_url, video_url=video_url,
        )
        if not creation_id:
            return False, "", err

        # Meta recommends waiting before publishing so their pipeline finishes
        # ingesting media. For pure-text posts this is mostly unnecessary but
        # cheap insurance.
        if settle > 0:
            time.sleep(settle if (image_url or video_url) else min(settle, 5))


        post_id, err = _publish_container(client, user_id, token, creation_id)
        if not post_id:
            return False, "", err

    return True, post_id, ""
