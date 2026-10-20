"""Bluesky adapter — AT Protocol posting via atproto SDK.

Env:
  BLUESKY_HANDLE[_<ACCT>], BLUESKY_APP_PASSWORD[_<ACCT>]
  (App password from https://bsky.app/settings/app-passwords — never use main pwd.)

Format key: `bluesky_post`
  text: str  (<=300 chars; threads supported via reply chain)
  images: list[{path: str, alt: str}]  (<=4)
  thread: list[str]  (if multiple posts, treats text as ignored)
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from opendispatch.adapters.errors import PERMANENT, PUBLISHED, clip, prefix_error
from opendispatch.api.schema import ContentUnit
from opendispatch.media.paths import resolve_media_path

log = logging.getLogger("open-dispatch.bluesky")
POST_LIMIT = 300
MAX_IMAGES = 4


def _creds(account: str | None) -> tuple[str, str]:
    suffix = f"_{account.upper()}" if account else ""
    handle = os.getenv(f"BLUESKY_HANDLE{suffix}") or os.getenv("BLUESKY_HANDLE", "")
    pwd = os.getenv(f"BLUESKY_APP_PASSWORD{suffix}") or os.getenv("BLUESKY_APP_PASSWORD", "")
    return handle, pwd


def publish(unit: ContentUnit, account: str | None = None, *, creds=None) -> tuple[bool, str, str]:
    fmt = unit.formats.get("bluesky_post") or {}
    handle, password = _creds(account)
    if not (handle and password):
        return False, "", "BLUESKY_HANDLE / BLUESKY_APP_PASSWORD missing"

    try:
        from atproto import Client, client_utils  # noqa: F401
    except ImportError:
        return False, "", "atproto not installed (pip install atproto)"

    try:
        client = Client()
        client.login(handle, password)

        thread = fmt.get("thread") or []
        posted = 0
        if thread:
            parent = None
            root = None
            first_uri = ""
            posted = 0
            for piece in thread:
                if len(piece) > POST_LIMIT:
                    return False, "", prefix_error(
                        PERMANENT,
                        f"thread item is {len(piece)} chars, over the {POST_LIMIT} limit",
                    )
                ref = client.send_post(text=piece,
                                       reply_to=parent)
                posted += 1
                if root is None:
                    root = ref
                    first_uri = ref.uri
                parent = {
                    "root": {"uri": root.uri, "cid": root.cid},
                    "parent": {"uri": ref.uri, "cid": ref.cid},
                }
            return True, first_uri, ""

        text = (fmt.get("text") or "").strip()
        if not text:
            return False, "", "bluesky_post.text empty"
        if len(text) > POST_LIMIT:
            return False, "", prefix_error(
                PERMANENT, f"text is {len(text)} chars, over the {POST_LIMIT} limit")

        images = fmt.get("images") or []
        if images:
            embed_images = []
            if len(images) > MAX_IMAGES:
                return False, "", prefix_error(
                    PERMANENT,
                    f"images supports at most {MAX_IMAGES} items (got {len(images)})",
                )
            for img in images:
                p = resolve_media_path(img["path"], strict_root=True)
                uploaded = client.upload_blob(p.read_bytes())
                embed_images.append({"image": uploaded.blob, "alt": img.get("alt", "")})
            ref = client.send_images(text=text, images_alt=[i["alt"] for i in embed_images],
                                     images=[p["image"] for p in embed_images])
        else:
            ref = client.send_post(text=text)
        return True, ref.uri, ""
    except Exception as e:  # noqa: BLE001
        # Earlier posts in a reply chain are already live; retrying the whole
        # chain would duplicate them.
        if posted:
            return False, "", prefix_error(
                PUBLISHED,
                f"bluesky: {posted}/{len(thread)} posts published before failing, "
                f"NOT retried to avoid duplicates: {clip(e)}",
            )
        return False, "", prefix_error("retryable", f"bluesky error: {clip(e)}")
