"""Regression tests for the adapter correctness and credential-leak fixes.

Each test reproduces a specific failure that shipped, then asserts the fixed
behavior. All HTTP is stubbed at the httpx boundary, so no network.
"""
from __future__ import annotations

import httpx
import pytest

from opendispatch.api.schema import ContentUnit
from opendispatch.adapters import errors


def _unit(fmt_key: str, payload: dict, target: str = "x") -> ContentUnit:
    return ContentUnit(targets=[target], formats={fmt_key: payload})


def _resp(status: int, json_body=None, text: str = "", headers=None, url="https://api.test/x"):
    return httpx.Response(status, json=json_body, text=text or None,
                          headers=headers or {}, request=httpx.Request("POST", url))


# ── Telegram: HTTP 200 + ok:false must not report success ───────────────────

def test_telegram_ok_false_is_a_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """Telegram signals failure with HTTP 200 and "ok": false. The old code
    never checked it, so a chat-not-found returned (True, '', '') — the row was
    marked published, no webhook fired, and the message was lost with no retry."""
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "123456789:AAFAKEFAKEFAKEFAKEFAKEFAKEFAKE")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "@chan")
    from opendispatch.adapters import telegram

    monkeypatch.setattr(telegram.httpx, "post", lambda url, **kw: _resp(
        200, {"ok": False, "error_code": 400, "description": "chat not found"},
        url=url))

    ok, pid, err = telegram.publish(_unit("telegram_message", {"text": "hi"}))
    assert ok is False
    assert pid == ""
    assert "chat not found" in err
    assert err.startswith("permanent:"), "a 400 is not retryable"


def test_telegram_success_still_returns_message_id(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "123456789:AAFAKEFAKEFAKEFAKEFAKEFAKE")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "@chan")
    from opendispatch.adapters import telegram

    monkeypatch.setattr(telegram.httpx, "post", lambda url, **kw: _resp(
        200, {"ok": True, "result": {"message_id": 4242}}, url=url))
    assert telegram.publish(_unit("telegram_message", {"text": "hi"})) == (True, "4242", "")


def test_telegram_401_error_never_contains_the_bot_token(monkeypatch: pytest.MonkeyPatch) -> None:
    token = "123456789:AAFAKEFAKEFAKEFAKEFAKEFAKEFAKE"
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", token)
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "@chan")
    from opendispatch.adapters import telegram

    monkeypatch.setattr(telegram.httpx, "post", lambda url, **kw: _resp(
        401, {"ok": False, "description": "Unauthorized"}, url=url))
    ok, _pid, err = telegram.publish(_unit("telegram_message", {"text": "hi"}))
    assert ok is False
    assert token not in err
    assert "AAFAKEFAKEFAKEFAKEFAKEFAKEFAKE" not in err
    assert err.startswith("auth:")


def test_telegram_non_json_response_is_not_a_success(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "123456789:AAFAKEFAKEFAKEFAKEFAKEFAKE")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "@chan")
    from opendispatch.adapters import telegram

    monkeypatch.setattr(telegram.httpx, "post", lambda url, **kw: _resp(
        200, None, text="<html>gateway</html>", url=url))
    ok, pid, _err = telegram.publish(_unit("telegram_message", {"text": "hi"}))
    assert ok is False
    assert pid == ""


# ── LinkedIn: a 2xx means the post is live, so never return ok=False ────────

def test_linkedin_missing_restli_id_is_published_not_failed(monkeypatch: pytest.MonkeyPatch) -> None:
    """A 201 with lifecycleState PUBLISHED means the post went out. The old code
    called r.json() on LinkedIn's empty 201 body, caught the JSONDecodeError,
    and returned ok=False — so the worker retried and re-published the post."""
    monkeypatch.setenv("LINKEDIN_ACCESS_TOKEN", "tok")
    monkeypatch.setenv("LINKEDIN_AUTHOR_URN", "urn:li:person:x")
    from opendispatch.adapters import linkedin

    monkeypatch.setattr(linkedin.httpx, "post", lambda url, **kw: _resp(
        201, None, text="", url=url))
    ok, pid, err = linkedin.publish(_unit("linkedin_post", {"text": "hi"}))
    assert ok is False, "worker must not retry a post that is already live"
    assert err.startswith(f"{errors.PUBLISHED}:")
    assert "x-restli-id" in err


def test_linkedin_success_returns_urn(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LINKEDIN_ACCESS_TOKEN", "tok")
    monkeypatch.setenv("LINKEDIN_AUTHOR_URN", "urn:li:person:x")
    from opendispatch.adapters import linkedin

    monkeypatch.setattr(linkedin.httpx, "post", lambda url, **kw: _resp(
        201, None, text="", headers={"x-restli-id": "urn:li:share:42"}, url=url))
    assert linkedin.publish(_unit("linkedin_post", {"text": "hi"})) == (True, "urn:li:share:42", "")


def test_linkedin_4xx_is_classified_and_redacted(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LINKEDIN_ACCESS_TOKEN", "SUPERSECRETTOKEN")
    monkeypatch.setenv("LINKEDIN_AUTHOR_URN", "urn:li:person:x")
    from opendispatch.adapters import linkedin

    monkeypatch.setattr(linkedin.httpx, "post", lambda url, **kw: _resp(
        403, {"message": "insufficient scope"}, url=url))
    ok, _pid, err = linkedin.publish(_unit("linkedin_post", {"text": "hi"}))
    assert ok is False
    assert err.startswith("auth:")


# ── Instagram: 200 without an id is not a success ───────────────────────────

def test_instagram_200_without_id_is_a_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("IG_USER_ID", "123")
    monkeypatch.setenv("IG_TOKEN", "IGSECRET")
    from opendispatch.adapters import instagram

    calls: list[str] = []

    def fake_post(url, **kw):
        calls.append(url)
        if "media_publish" in url:
            return _resp(200, {"status": "ok"}, url=url)
        if "media" in url and "images" not in url:
            return _resp(200, {"id": "container-1"}, url=url)
        return _resp(200, {"status_code": "FINISHED"}, url=url)

    def fake_get(url, **kw):
        return _resp(200, {"status_code": "FINISHED"}, url=url)

    monkeypatch.setattr(instagram.httpx, "post", fake_post)
    monkeypatch.setattr(instagram.httpx, "get", fake_get)
    monkeypatch.setattr(instagram, "_wait_for_container", lambda cid, tok, timeout=90: (True, ""))

    ok, pid, err = instagram.publish(_unit("instagram_post", {"image_url": "https://x/a.jpg"}))
    assert ok is False
    assert pid == ""
    assert "no id" in err


def test_instagram_never_puts_the_token_in_the_query_string(monkeypatch: pytest.MonkeyPatch) -> None:
    """httpx logs the full URL at INFO, and the worker enables INFO, so a token
    in the query string lands in the log file."""
    monkeypatch.setenv("IG_USER_ID", "123")
    monkeypatch.setenv("IG_TOKEN", "IGSECRETCANARY")
    from opendispatch.adapters import instagram

    seen: list[str] = []

    def fake_get(url, **kw):
        seen.append(url)
        return _resp(200, {"status_code": "FINISHED"}, url=url)

    monkeypatch.setattr(instagram.httpx, "get", fake_get)

    ok, err = instagram._wait_for_container("c1", "IGSECRETCANARY")
    assert ok is True
    assert seen, "the container was never polled"
    assert all("IGSECRETCANARY" not in u for u in seen), seen
    # And it must be a header, not a query param.
    assert "access_token" not in seen[0]


# ── TikTok: poll exhaustion is not success ─────────────────────────────────

def test_tiktok_poll_exhaustion_is_not_success(monkeypatch: pytest.MonkeyPatch) -> None:
    """The old code returned (True, publish_id, "") after 12 polls. publish_id
    is not a post id, so a video that never published was marked published."""
    monkeypatch.setenv("TIKTOK_ACCESS_TOKEN", "tok")
    from opendispatch.adapters import tiktok

    def fake_post(url, **kw):
        if "/video/init/" in url:
            return _resp(200, {"data": {"publish_id": "PUB123"}}, url=url)
        return _resp(200, {"data": {"status": "PROCESSING_UPLOAD"}}, url=url)

    monkeypatch.setattr(tiktok.httpx, "post", fake_post)
    monkeypatch.setattr(tiktok.time, "sleep", lambda s: None)

    ok, _pid, err = tiktok.publish(_unit("tiktok_post", {"video_url": "https://x/v.mp4"}))
    assert ok is False
    assert err.startswith(f"{errors.PUBLISHED}:"), err
    assert "PUB123" in err


def test_tiktok_string_post_id_is_not_indexed_to_one_char(monkeypatch: pytest.MonkeyPatch) -> None:
    """A bare string post id used to be indexed with [0], returning the first
    CHARACTER of the real id as the post id, with ok=True."""
    monkeypatch.setenv("TIKTOK_ACCESS_TOKEN", "tok")
    from opendispatch.adapters import tiktok

    def fake_post(url, **kw):
        if "/video/init/" in url:
            return _resp(200, {"data": {"publish_id": "PUB123"}}, url=url)
        return _resp(200, {"data": {"status": "PUBLISH_COMPLETE",
                                    "publicaly_available_post_id": "189876543210"}}, url=url)

    monkeypatch.setattr(tiktok.httpx, "post", fake_post)
    ok, pid, _err = tiktok.publish(_unit("tiktok_post", {"video_url": "https://x/v.mp4"}))
    assert ok is True
    assert pid == "189876543210", "the whole id, not '1'"


def test_tiktok_status_error_envelope_is_a_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TIKTOK_ACCESS_TOKEN", "tok")
    from opendispatch.adapters import tiktok

    def fake_post(url, **kw):
        if "/video/init/" in url:
            return _resp(200, {"data": {"publish_id": "PUB123"}}, url=url)
        return _resp(200, {"error": {"code": 22000, "message": "Too many requests"}}, url=url)

    monkeypatch.setattr(tiktok.httpx, "post", fake_post)
    ok, _pid, err = tiktok.publish(_unit("tiktok_post", {"video_url": "https://x/v.mp4"}))
    assert ok is False
    assert "Too many requests" in err


# ── YouTube: untrusted Location host must be refused ───────────────────────

@pytest.mark.parametrize("location", [
    "https://attacker.example.net/collect",
    "http://169.254.169.254/latest/meta-data/",
    "http://graph.youtube.com/cleartext-session",
    "https://googleapis.com.evil.net/collect",
    "ftp://www.googleapis.com/upload",
])
def _youtube_env(tmp_path, monkeypatch, location: str | None = None):
    """Common setup for the YouTube upload tests: a real file inside a media
    root, and an OAuth refresh stub so nothing touches the network."""
    import opendispatch.media.paths as media_paths
    from opendispatch.adapters import youtube

    media = tmp_path / "media"
    media.mkdir()
    (media / "clip.mp4").write_bytes(b"fake video bytes")
    monkeypatch.setattr(media_paths, "MEDIA_ROOT", media.resolve())
    monkeypatch.setenv("YOUTUBE_CLIENT_ID", "cid")
    monkeypatch.setenv("YOUTUBE_CLIENT_SECRET", "csec")
    monkeypatch.setenv("YOUTUBE_REFRESH_TOKEN", "rtok")
    monkeypatch.setattr(youtube, "_refresh_access_token", lambda *a, **k: ("at", ""))

    put_calls: list[str] = []

    class FakeClient:
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def post(self, url, **kw):
            hdrs = {"location": location} if location else {}
            return _resp(200, None, text="", headers=hdrs, url=url)
        def put(self, url, **kw):
            put_calls.append(url)
            return _resp(200, None, text="ok", url=url)

    monkeypatch.setattr(youtube.httpx, "Client", lambda **kw: FakeClient())
    return youtube, put_calls


@pytest.mark.parametrize("location", [
    "https://attacker.example.net/collect",
    "http://169.254.169.254/latest/meta-data/",
    "http://graph.youtube.com/cleartext-session",
    "https://googleapis.com.evil.net/collect",
    "ftp://www.googleapis.com/upload",
])
def test_youtube_refuses_untrusted_upload_host(tmp_path, monkeypatch, location: str) -> None:
    """The Location header dictates where the whole video file is PUT. Pinning
    scheme and host means an intercepting proxy or DNS compromise cannot
    redirect the upload."""
    youtube, put_calls = _youtube_env(tmp_path, monkeypatch, location)

    ok, _pid, err = youtube.publish(_unit("youtube_short", {
        "video_path": "clip.mp4", "caption": "hi"}))

    assert put_calls == [], f"the video file was PUT to {location}"
    assert ok is False
    assert "untrusted Location host" in err


@pytest.mark.parametrize("location", [
    "https://www.googleapis.com/upload/session/v1",
    "https://storage.googleapis.com/upload",
])
def test_youtube_allows_google_upload_hosts(tmp_path, monkeypatch, location: str) -> None:
    """The allowlist must not be so tight that real uploads break."""
    youtube, put_calls = _youtube_env(tmp_path, monkeypatch, location)

    ok, _pid, err = youtube.publish(_unit("youtube_short", {
        "video_path": "clip.mp4", "caption": "hi"}))

    assert "untrusted Location host" not in err, err
    assert put_calls == [location], f"expected the upload to proceed, got {err}"


def test_youtube_upload_is_bounded_and_streamed(tmp_path, monkeypatch) -> None:
    """timeout=None hangs the single-threaded worker forever, and f.read()
    buffers the whole video in RAM."""
    youtube, _ = _youtube_env(tmp_path, monkeypatch, "https://www.googleapis.com/upload")

    seen: dict = {}

    class FakeClient:
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def post(self, url, **kw):
            return _resp(200, None, text="", headers={"location": "https://www.googleapis.com/up"}, url=url)
        def put(self, url, **kw):
            seen.update(kw)
            return _resp(200, None, text="ok", url=url)

    monkeypatch.setattr(youtube.httpx, "Client", lambda **kw: FakeClient())
    youtube.publish(_unit("youtube_short", {"video_path": "clip.mp4", "caption": "hi"}))

    assert "timeout" in seen, "the upload had no timeout at all"
    assert seen["timeout"] is not None
    assert seen.get("content") is not None and hasattr(seen["content"], "read"), \
        "the file was read into memory instead of streamed"


def test_youtube_invalid_privacy_is_rejected_not_defaulted(tmp_path, monkeypatch) -> None:
    """Defaulting an unknown privacy value to "public" is a visibility
    escalation — a typo like "private " would publish the video to the world."""
    youtube, _ = _youtube_env(tmp_path, monkeypatch)

    ok, _pid, err = youtube.publish(_unit("youtube_short", {
        "video_path": "clip.mp4", "caption": "hi", "privacy": "secured"}))

    assert ok is False
    assert "invalid privacy" in err, err
    assert "invalid privacy 'secured'" in err
    # The crucial part: it must NOT have fallen back to public.
    assert "privacyStatus" not in err


@pytest.mark.parametrize("value", ["public", "unlisted", "private", "Private", " PUBLIC "])
def test_youtube_valid_privacy_is_accepted(tmp_path, monkeypatch, value: str) -> None:
    """The new check must not reject legitimate values."""
    youtube, _ = _youtube_env(tmp_path, monkeypatch, "https://www.googleapis.com/up")

    _ok, _pid, err = youtube.publish(_unit("youtube_short", {
        "video_path": "clip.mp4", "caption": "hi", "privacy": value}))

    assert "invalid privacy" not in err, f"{value!r} was wrongly rejected: {err}"


# ── Worker: adapter exceptions must not strand rows ─────────────────────────

def test_worker_records_an_adapter_exception_as_a_failure(tmp_path, monkeypatch) -> None:
    """An adapter that raised left the row stuck in "publishing" forever:
    list_due() only returns "queued" rows and nothing reaps a stuck claim."""
    monkeypatch.setenv("OPEN_DISPATCH_DATA", str(tmp_path))
    from opendispatch.api.queue import get_queue
    import opendispatch.scheduler.worker as worker

    q = get_queue()
    unit = {"id": "u-boom", "targets": ["bluesky"],
            "formats": {"bluesky_post": {"text": "hi"}}, "scheduled_for": None}
    rid = q.enqueue(unit, "bluesky", "2020-01-01T00:00:00+00:00")

    def boom(row):
        raise ValueError("invalid literal for int() with base 10: 'abc'")

    monkeypatch.setattr(worker, "_publish", boom)
    monkeypatch.setattr(worker, "get_queue", lambda: q)
    worker.run_once()

    row = q.get(rid)
    assert row["status"] != "publishing", "row was stranded in publishing"
    assert row["status"] in ("queued", "dead")
    assert "ValueError" in (row.get("last_error") or "")


def test_worker_dead_letters_auth_errors_without_retrying(tmp_path, monkeypatch) -> None:
    """A 401 cannot succeed on retry, so it must not burn the attempt budget."""
    monkeypatch.setenv("OPEN_DISPATCH_DATA", str(tmp_path))
    from opendispatch.api.queue import get_queue
    import opendispatch.scheduler.worker as worker

    q = get_queue()
    unit = {"id": "u-auth", "targets": ["bluesky"],
            "formats": {"bluesky_post": {"text": "hi"}}, "scheduled_for": None}
    rid = q.enqueue(unit, "bluesky", "2020-01-01T00:00:00+00:00")

    monkeypatch.setattr(worker, "_publish", lambda row: (False, "", "auth: HTTP 401: bad token"))
    monkeypatch.setattr(worker, "get_queue", lambda: q)
    worker.run_once()

    row = q.get(rid)
    assert row["status"] == "dead", f"expected immediate dead-letter, got {row['status']}"
    assert row["attempts"] == 1, "auth failure must not consume 3 attempts"


def test_worker_never_retries_a_published_but_unconfirmed_post(tmp_path, monkeypatch) -> None:
    """Retrying a post that already went out is how you get duplicates."""
    monkeypatch.setenv("OPEN_DISPATCH_DATA", str(tmp_path))
    from opendispatch.api.queue import get_queue
    import opendispatch.scheduler.worker as worker

    q = get_queue()
    unit = {"id": "u-dup", "targets": ["bluesky"],
            "formats": {"bluesky_post": {"text": "hi"}}, "scheduled_for": None}
    rid = q.enqueue(unit, "bluesky", "2020-01-01T00:00:00+00:00")

    monkeypatch.setattr(
        worker, "_publish",
        lambda row: (False, "", f"{errors.PUBLISHED}: sent but id unreadable"))
    monkeypatch.setattr(worker, "get_queue", lambda: q)
    worker.run_once()

    row = q.get(rid)
    assert row["status"] == "published", f"got {row['status']}"
    assert row["status"] != "dead"


# ── Every adapter must be importable and self-consistent ───────────────────

def test_every_adapter_imports_and_declares_publish() -> None:
    import importlib

    from opendispatch.adapters import ADAPTERS

    for name in ADAPTERS:
        mod = importlib.import_module(f"opendispatch.adapters.{name}")
        assert callable(mod.publish), name
        import inspect

        params = list(inspect.signature(mod.publish).parameters)
        assert params[0] == "unit", name
