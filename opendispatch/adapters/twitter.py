"""Twitter / X adapter — v2 API via tweepy.

Env (per account suffix; uppercase):
  TWITTER_API_KEY, TWITTER_API_SECRET (consumer creds, shared)
  TWITTER_ACCESS_TOKEN[_<ACCT>], TWITTER_ACCESS_SECRET[_<ACCT>]

Format key: `twitter_thread`
  tweets: list[str] — each <=280 chars; posted as a reply chain
  media_paths (optional, applied to first tweet): list[str]
"""

from __future__ import annotations

import logging
import os

from opendispatch.adapters.errors import PERMANENT, PUBLISHED, clip, prefix_error
from opendispatch.api.schema import ContentUnit
from opendispatch.media.paths import resolve_media_path, MediaPathError

log = logging.getLogger("open-dispatch.twitter")
TWEET_LIMIT = 280
MAX_MEDIA = 4


def _tokens(account: str | None) -> tuple[str, str]:
    suffix = f"_{account.upper()}" if account else ""
    at = os.getenv(f"TWITTER_ACCESS_TOKEN{suffix}") or os.getenv("TWITTER_ACCESS_TOKEN", "")
    asec = os.getenv(f"TWITTER_ACCESS_SECRET{suffix}") or os.getenv("TWITTER_ACCESS_SECRET", "")
    return at, asec


def publish(unit: ContentUnit, account: str | None = None, *, creds=None) -> tuple[bool, str, str]:
    fmt = unit.formats.get("twitter_thread") or {}
    tweets = [str(t).strip() for t in fmt.get("tweets", []) if str(t).strip()]
    if not tweets:
        return False, "", "twitter_thread.tweets is empty"

    consumer_key = os.getenv("TWITTER_API_KEY", "").strip()
    consumer_secret = os.getenv("TWITTER_API_SECRET", "").strip()
    access_token, access_secret = _tokens(account)
    if not (consumer_key and consumer_secret and access_token and access_secret):
        return False, "", "twitter credentials missing"

    try:
        import tweepy
    except ImportError:
        return False, "", "tweepy not installed (pip install tweepy)"

    for i, text in enumerate(tweets):
        if len(text) > TWEET_LIMIT:
            # text[:280] posted a broken mid-sentence tweet with no warning.
            return False, "", prefix_error(
                PERMANENT,
                f"tweet {i + 1} is {len(text)} chars, over the {TWEET_LIMIT} limit",
            )

    try:
        client = tweepy.Client(
            consumer_key=consumer_key,
            consumer_secret=consumer_secret,
            access_token=access_token,
            access_token_secret=access_secret,
        )
        in_reply_to: str | None = None
        first_id = ""
        media_ids: list[str] = []
        posted: list[str] = []

        media_paths = fmt.get("media_paths") or []
        if len(media_paths) > MAX_MEDIA:
            return False, "", prefix_error(
                PERMANENT,
                f"media_paths supports at most {MAX_MEDIA} files (got {len(media_paths)})",
            )
        if media_paths:
            api_v1 = tweepy.API(tweepy.OAuth1UserHandler(
                consumer_key, consumer_secret, access_token, access_secret,
            ))
            for path in media_paths:
                m = api_v1.media_upload(filename=str(resolve_media_path(path, strict_root=True)))
                media_ids.append(str(m.media_id))

        for i, text in enumerate(tweets):
            kwargs: dict = {"text": text}
            if in_reply_to:
                kwargs["in_reply_to_tweet_id"] = in_reply_to
            if i == 0 and media_ids:
                kwargs["media_ids"] = media_ids
            resp = client.create_tweet(**kwargs)
            tid = str(resp.data["id"])
            posted.append(tid)
            if i == 0:
                first_id = tid
            in_reply_to = tid
        return True, first_id, ""
    except Exception as e:  # noqa: BLE001
        # If some tweets already went out, a plain retry would re-post them —
        # a duplicate thread is worse than an incomplete one. Report it as
        # `published` so the worker never retries, and name what did publish.
        if posted:
            return False, "", prefix_error(
                PUBLISHED,
                f"tweet {len(posted)}/{len(tweets)} published before failing "
                f"(ids {', '.join(posted[:5])}); NOT retried to avoid duplicates: {clip(e)}",
            )
        return False, "", prefix_error("retryable", f"twitter error: {clip(e)}")
