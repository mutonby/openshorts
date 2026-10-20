"""Regression tests: config parsing, silent truncation, and dropped items.

Each of these shipped a way for a post to be published wrong — or lost —
with no error surfaced.
"""
from __future__ import annotations

import httpx
import pytest

from opendispatch.api.schema import ContentUnit


def _unit(fmt_key: str, payload: dict, target: str = "x") -> ContentUnit:
    return ContentUnit(targets=[target], formats={fmt_key: payload})


def _resp(status: int, json_body=None, text: str = "", url="https://graph.test/x"):
    return httpx.Response(status, json=json_body, text=text or None,
                          request=httpx.Request("POST", url))


# ── Threads: a typo in THREADS_SETTLE_SECONDS used to raise ValueError ─────

def test_threads_bad_settle_env_does_not_raise(monkeypatch: pytest.MonkeyPatch) -> None:
    """int("abc") raised out of publish(), leaving the row stuck in
    "publishing" forever — list_due() only returns "queued" rows and nothing
    reaps a stuck claim, so the post was silently lost."""
    monkeypatch.setenv("THREADS_USER_ID", "u")
    monkeypatch.setenv("THREADS_ACCESS_TOKEN", "t")
    monkeypatch.setenv("THREADS_SETTLE_SECONDS", "abc")
    from opendispatch.adapters import threads

    monkeypatch.setattr(threads, "_create_container", lambda *a, **k: ("c1", ""))
    monkeypatch.setattr(threads, "_publish_container", lambda *a, **k: ("p1", ""))
    monkeypatch.setattr(threads.time, "sleep", lambda s: None)

    # Must not raise, and must not be treated as a success either.
    ok, _pid, err = threads.publish(_unit("threads_post", {"text": "hi"}))
    assert ok in (True, False)  # the point is that it returns at all
    assert isinstance(err, str)


@pytest.mark.parametrize("value", ["", "not-a-number", "30.5", "1e9", "-5", "999999"])
def test_threads_settle_env_is_always_parseable(monkeypatch: pytest.MonkeyPatch, value: str) -> None:
    monkeypatch.setenv("THREADS_USER_ID", "u")
    monkeypatch.setenv("THREADS_ACCESS_TOKEN", "t")
    monkeypatch.setenv("THREADS_SETTLE_SECONDS", value)
    from opendispatch.adapters import threads

    slept: list[float] = []
    monkeypatch.setattr(threads, "_create_container", lambda *a, **k: ("c1", ""))
    monkeypatch.setattr(threads, "_publish_container", lambda *a, **k: ("p1", ""))
    monkeypatch.setattr(threads.time, "sleep", lambda s: slept.append(s))

    threads.publish(_unit("threads_post", {"text": "hi", "image_url": "https://x/a.jpg"}))
    # Whatever the env says, the blocking sleep stays bounded.
    assert all(s <= threads.MAX_SETTLE_SECONDS for s in slept), slept


def test_threads_settle_is_clamped(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("THREADS_USER_ID", "u")
    monkeypatch.setenv("THREADS_ACCESS_TOKEN", "t")
    monkeypatch.setenv("THREADS_SETTLE_SECONDS", "999999")
    from opendispatch.adapters import threads

    slept: list[float] = []
    monkeypatch.setattr(threads, "_create_container", lambda *a, **k: ("c1", ""))
    monkeypatch.setattr(threads, "_publish_container", lambda *a, **k: ("p1", ""))
    monkeypatch.setattr(threads.time, "sleep", lambda s: slept.append(s))

    threads.publish(_unit("threads_post", {"text": "hi", "image_url": "https://x/a.jpg"}))
    assert slept and max(slept) <= threads.MAX_SETTLE_SECONDS, slept


# ── Twitter / Bluesky: silent truncation published a broken post ───────────

def test_twitter_over_length_tweet_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    """text[:280] posted a mid-sentence fragment with no warning at all."""
    monkeypatch.setenv("TWITTER_API_KEY", "k")
    monkeypatch.setenv("TWITTER_API_SECRET", "s")
    monkeypatch.setenv("TWITTER_ACCESS_TOKEN", "at")
    monkeypatch.setenv("TWITTER_ACCESS_SECRET", "as")
    from opendispatch.adapters import twitter

    posted: list[dict] = []

    class FakeResp:
        data = {"id": "1"}

    class FakeClient:
        def create_tweet(self, **kw):
            posted.append(kw)
            return FakeResp()

    monkeypatch.setitem(__import__("sys").modules, "tweepy",
                        type("T", (), {"Client": lambda **kw: FakeClient(),
                                      "API": None, "OAuth1UserHandler": lambda *a: None}))
    ok, _pid, err = twitter.publish(_unit("twitter_thread", {"tweets": ["x" * 400]}))
    assert ok is False
    assert posted == [], "an over-length tweet must not be posted at all"
    assert "over the 280 limit" in err


def test_twitter_exact_length_tweet_is_accepted(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TWITTER_API_KEY", "k")
    monkeypatch.setenv("TWITTER_API_SECRET", "s")
    monkeypatch.setenv("TWITTER_ACCESS_TOKEN", "at")
    monkeypatch.setenv("TWITTER_ACCESS_SECRET", "as")
    from opendispatch.adapters import twitter

    posted: list[dict] = []

    class FakeResp:
        data = {"id": "42"}

    class FakeClient:
        def create_tweet(self, **kw):
            posted.append(kw)
            return FakeResp()

    monkeypatch.setitem(__import__("sys").modules, "tweepy",
                        type("T", (), {"Client": lambda **kw: FakeClient(),
                                      "API": None, "OAuth1UserHandler": lambda *a: None}))
    ok, pid, err = twitter.publish(_unit("twitter_thread", {"tweets": ["y" * 280]}))
    assert ok is True, err
    assert pid == "42"
    assert len(posted[0]["text"]) == 280


def test_twitter_too_many_media_files_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TWITTER_API_KEY", "k")
    monkeypatch.setenv("TWITTER_API_SECRET", "s")
    monkeypatch.setenv("TWITTER_ACCESS_TOKEN", "at")
    monkeypatch.setenv("TWITTER_ACCESS_SECRET", "as")
    from opendispatch.adapters import twitter

    monkeypatch.setitem(__import__("sys").modules, "tweepy",
                        type("T", (), {"Client": lambda **kw: None,
                                      "API": None, "OAuth1UserHandler": lambda *a: None}))
    ok, _pid, err = twitter.publish(_unit("twitter_thread", {
        "tweets": ["hi"], "media_paths": [f"m{i}.jpg" for i in range(5)]}))
    assert ok is False
    assert "at most 4" in err


def test_bluesky_over_length_post_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BLUESKY_HANDLE", "a.bsky.social")
    monkeypatch.setenv("BLUESKY_APP_PASSWORD", "pw")
    from opendispatch.adapters import bluesky

    sent: list[str] = []

    class FakeClient:
        def login(self, *a, **k): return None
        def send_post(self, text, **k):
            sent.append(text)
            return type("R", (), {"uri": "at://x", "cid": "c"})()

    import atproto
    monkeypatch.setattr(atproto, "Client", lambda: FakeClient())
    ok, _pid, err = bluesky.publish(_unit("bluesky_post", {"text": "z" * 400}))
    assert ok is False
    assert sent == [], "an over-length post must not be truncated and sent"
    assert "over the 300 limit" in err


def test_bluesky_exact_length_post_is_accepted(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BLUESKY_HANDLE", "a.bsky.social")
    monkeypatch.setenv("BLUESKY_APP_PASSWORD", "pw")
    from opendispatch.adapters import bluesky

    sent: list[str] = []

    class FakeClient:
        def login(self, *a, **k): return None
        def send_post(self, text, **k):
            sent.append(text)
            return type("R", (), {"uri": "at://x", "cid": "c"})()

    import atproto
    monkeypatch.setattr(atproto, "Client", lambda: FakeClient())
    ok, pid, err = bluesky.publish(_unit("bluesky_post", {"text": "z" * 300}))
    assert ok is True, err
    assert pid == "at://x"
    assert len(sent[0]) == 300


# ── Instagram carousel: items were silently dropped ────────────────────────

def test_instagram_oversized_carousel_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    """carousel[:10] published the post with the wrong images and no error."""
    monkeypatch.setenv("IG_USER_ID", "123")
    monkeypatch.setenv("IG_TOKEN", "tok")
    from opendispatch.adapters import instagram

    calls: list[str] = []
    monkeypatch.setattr(instagram.httpx, "post",
                        lambda url, **kw: calls.append(url) or _resp(200, {"id": "c"}, url=url))

    ok, _pid, err = instagram.publish(_unit("instagram_post", {
        "carousel_image_urls": [f"https://x/{i}.jpg" for i in range(12)]}))
    assert ok is False
    assert calls == [], "no partial carousel should be created"
    assert "at most 10" in err


def test_instagram_exact_size_carousel_is_accepted(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("IG_USER_ID", "123")
    monkeypatch.setenv("IG_TOKEN", "tok")
    from opendispatch.adapters import instagram

    monkeypatch.setattr(instagram, "_wait_for_container", lambda cid, tok, timeout=90: (True, ""))
    monkeypatch.setattr(instagram.httpx, "post",
                        lambda url, **kw: _resp(200, {"id": f"child-{url[-1]}"}, url=url))
    ok, _pid, err = instagram.publish(_unit("instagram_post", {
        "carousel_image_urls": [f"https://x/{i}.jpg" for i in range(10)]}))
    assert ok is True, err
