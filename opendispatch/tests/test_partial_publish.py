"""Partial publishes must never be retried into duplicates.

A Twitter thread of 5 where tweet 3 fails, or a long Telegram message split
into chunks where chunk 2 fails: tweets/chunks 1-2 are already live. The
adapter used to return a plain failure, so the worker retried and re-posted
the ones that had already gone out — on every attempt, up to MAX_ATTEMPTS.

These tests assert the partial-publish contract: report `published`, never
`retryable`.
"""
from __future__ import annotations

import sys
import types

import httpx
import pytest

from opendispatch.api.schema import ContentUnit
from opendispatch.adapters.errors import PUBLISHED


def _unit(fmt_key: str, payload: dict) -> ContentUnit:
    return ContentUnit(targets=["x"], formats={fmt_key: payload})


def _resp(status: int, json_body=None, url="https://api.test/x"):
    return httpx.Response(status, json=json_body, request=httpx.Request("POST", url))


def _fake_tweepy(monkeypatch, fail_on_call: int):
    """tweepy stub whose Nth create_tweet raises."""
    calls = {"n": 0}
    ids = []

    class FakeResp:
        def __init__(self, i): self.data = {"id": i}

    class FakeClient:
        def create_tweet(self, **kw):
            calls["n"] += 1
            if calls["n"] == fail_on_call:
                raise RuntimeError("tweet 3 rejected")
            tid = f"tweet-{calls['n']}"
            ids.append(tid)
            return FakeResp(tid)

    monkeypatch.setitem(sys.modules, "tweepy", types.SimpleNamespace(
        Client=lambda **kw: FakeClient(), API=None, OAuth1UserHandler=lambda *a: None))
    return ids


# ── Twitter thread ─────────────────────────────────────────────────────────

def test_twitter_thread_failure_after_partial_publish_is_not_retried(
        monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TWITTER_API_KEY", "k")
    monkeypatch.setenv("TWITTER_API_SECRET", "s")
    monkeypatch.setenv("TWITTER_ACCESS_TOKEN", "at")
    monkeypatch.setenv("TWITTER_ACCESS_SECRET", "as")
    from opendispatch.adapters import twitter

    ids = _fake_tweepy(monkeypatch, fail_on_call=3)
    ok, _pid, err = twitter.publish(_unit("twitter_thread", {
        "tweets": ["one", "two", "three", "four", "five"]}))

    assert ok is False
    assert err.startswith(f"{PUBLISHED}:"), err
    assert "2/5" in err, err
    # The already-published ids must be named so an operator can find them.
    assert "tweet-1" in err and "tweet-2" in err
    assert len(ids) == 2, "the first two tweets did publish"


def test_twitter_thread_failure_on_first_tweet_is_retryable(
        monkeypatch: pytest.MonkeyPatch) -> None:
    """Nothing went out, so this one SHOULD be retried."""
    monkeypatch.setenv("TWITTER_API_KEY", "k")
    monkeypatch.setenv("TWITTER_API_SECRET", "s")
    monkeypatch.setenv("TWITTER_ACCESS_TOKEN", "at")
    monkeypatch.setenv("TWITTER_ACCESS_SECRET", "as")
    from opendispatch.adapters import twitter

    _fake_tweepy(monkeypatch, fail_on_call=1)
    ok, _pid, err = twitter.publish(_unit("twitter_thread", {"tweets": ["one", "two"]}))

    assert ok is False
    assert err.startswith("retryable:"), err
    assert PUBLISHED not in err


def test_twitter_full_thread_succeeds(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TWITTER_API_KEY", "k")
    monkeypatch.setenv("TWITTER_API_SECRET", "s")
    monkeypatch.setenv("TWITTER_ACCESS_TOKEN", "at")
    monkeypatch.setenv("TWITTER_ACCESS_SECRET", "as")
    from opendispatch.adapters import twitter

    ids = _fake_tweepy(monkeypatch, fail_on_call=99)
    ok, pid, err = twitter.publish(_unit("twitter_thread", {"tweets": ["a", "b", "c"]}))

    assert ok is True, err
    assert pid == "tweet-1"
    assert len(ids) == 3


# ── Telegram chunked message ───────────────────────────────────────────────

@pytest.mark.parametrize("fail_second", [
    # HTTP 5xx -> raise_for_status() raises before the body check
    lambda url: _resp(503, {"ok": False, "description": "unavailable"}, url=url),
    # HTTP 200 + ok:false -> body-level failure
    lambda url: _resp(200, {"ok": False, "error_code": 500,
                            "description": "upstream"}, url=url),
], ids=["http_5xx", "ok_false_200"])
def test_telegram_chunk_failure_after_partial_send_is_not_retried(
        monkeypatch: pytest.MonkeyPatch, fail_second) -> None:
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "123456789:AAFAKEFAKEFAKEFAKEFAKEFAKEFAKE")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "@c")
    from opendispatch.adapters import telegram
    from opendispatch.api.schema import CAPTION_LIMITS

    sent = {"n": 0}
    limit = CAPTION_LIMITS["telegram"]

    def fake_post(url, **kw):
        sent["n"] += 1
        if sent["n"] == 2:
            return fail_second(url)
        return _resp(200, {"ok": True, "result": {"message_id": sent["n"]}}, url=url)

    monkeypatch.setattr(telegram.httpx, "post", fake_post)
    # Force multiple chunks.
    long_text = "y" * (limit + 50)

    ok, _pid, err = telegram.publish(_unit("telegram_message", {"text": long_text}))

    assert ok is False
    assert err.startswith(f"{PUBLISHED}:"), err
    assert "1/2" in err, err


def test_telegram_first_chunk_failure_is_retryable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "123456789:AAFAKEFAKEFAKEFAKEFAKEFAKEFAKE")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "@c")
    from opendispatch.adapters import telegram

    def fake_post(url, **kw):
        return _resp(500, {"ok": False, "description": "upstream"}, url=url)

    monkeypatch.setattr(telegram.httpx, "post", fake_post)
    ok, _pid, err = telegram.publish(_unit("telegram_message", {"text": "hello"}))

    assert ok is False
    assert err.startswith("retryable:"), err


def test_telegram_single_chunk_success_unaffected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "123456789:AAFAKEFAKEFAKEFAKEFAKEFAKEFAKE")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "@c")
    from opendispatch.adapters import telegram

    monkeypatch.setattr(telegram.httpx, "post",
                        lambda url, **kw: _resp(200, {"ok": True, "result": {"message_id": 9}}, url=url))
    assert telegram.publish(_unit("telegram_message", {"text": "hi"})) == (True, "9", "")


# ── Bluesky reply chain ────────────────────────────────────────────────────

def test_bluesky_thread_partial_publish_is_not_retried(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BLUESKY_HANDLE", "a.bsky.social")
    monkeypatch.setenv("BLUESKY_APP_PASSWORD", "pw")
    from opendispatch.adapters import bluesky

    sent = {"n": 0}

    class FakeClient:
        def login(self, *a, **k): return None
        def send_post(self, text, **k):
            sent["n"] += 1
            if sent["n"] == 3:
                raise RuntimeError("reply chain broke")
            return types.SimpleNamespace(uri=f"at://x/{sent['n']}", cid="c")

    import atproto
    monkeypatch.setattr(atproto, "Client", lambda: FakeClient())

    ok, _pid, err = bluesky.publish(_unit("bluesky_post", {
        "thread": ["a", "b", "c", "d"]}))

    assert ok is False
    assert err.startswith(f"{PUBLISHED}:"), err
    assert "2/4" in err, err


def test_bluesky_first_post_failure_is_retryable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BLUESKY_HANDLE", "a.bsky.social")
    monkeypatch.setenv("BLUESKY_APP_PASSWORD", "pw")
    from opendispatch.adapters import bluesky

    class FakeClient:
        def login(self, *a, **k): return None
        def send_post(self, text, **k):
            raise RuntimeError("nope")

    import atproto
    monkeypatch.setattr(atproto, "Client", lambda: FakeClient())

    ok, _pid, err = bluesky.publish(_unit("bluesky_post", {"thread": ["a", "b"]}))
    assert ok is False
    assert err.startswith("retryable:"), err
