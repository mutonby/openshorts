"""Threads adapter tests — mocks httpx to avoid hitting Meta's API."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from opendispatch.adapters import threads
from opendispatch.api.schema import ContentUnit


def _unit(text: str = "", **fmt) -> ContentUnit:
    return ContentUnit(
        targets=["threads:default"],
        formats={"threads_post": {"text": text, **fmt}},
    )


def test_missing_creds(monkeypatch):
    monkeypatch.delenv("THREADS_USER_ID", raising=False)
    monkeypatch.delenv("THREADS_ACCESS_TOKEN", raising=False)
    ok, post_id, err = threads.publish(_unit("hi"))
    assert not ok
    assert "missing" in err.lower()


def test_empty_format(monkeypatch):
    monkeypatch.setenv("THREADS_USER_ID", "1234")
    monkeypatch.setenv("THREADS_ACCESS_TOKEN", "token")
    ok, _, err = threads.publish(_unit(""))
    assert not ok
    assert "requires text" in err.lower()


class _MockResp:
    def __init__(self, status: int, body: dict):
        self.status_code = status
        self._body = body
        self.text = str(body)

    def json(self):
        return self._body


def test_text_post_happy_path(monkeypatch):
    monkeypatch.setenv("THREADS_USER_ID", "1234")
    monkeypatch.setenv("THREADS_ACCESS_TOKEN", "token")
    monkeypatch.setenv("THREADS_SETTLE_SECONDS", "0")  # skip the sleep

    calls = []

    def fake_post(self, url, **kwargs):
        calls.append((url, kwargs.get("data", {})))
        if url.endswith("/threads"):
            return _MockResp(200, {"id": "container_123"})
        if url.endswith("/threads_publish"):
            return _MockResp(200, {"id": "post_999"})
        return _MockResp(404, {"error": "unknown"})

    with patch("httpx.Client.post", new=fake_post):
        ok, post_id, err = threads.publish(_unit("hello threads"))
    assert ok, err
    assert post_id == "post_999"
    # 2 API calls: create container + publish
    assert len(calls) == 2
    # First call had text + TEXT media_type
    assert calls[0][1]["text"] == "hello threads"
    assert calls[0][1]["media_type"] == "TEXT"
    # Second call references the creation_id
    assert calls[1][1]["creation_id"] == "container_123"


def test_image_post(monkeypatch):
    monkeypatch.setenv("THREADS_USER_ID", "1234")
    monkeypatch.setenv("THREADS_ACCESS_TOKEN", "token")
    monkeypatch.setenv("THREADS_SETTLE_SECONDS", "0")

    def fake_post(self, url, **kwargs):
        if url.endswith("/threads"):
            assert kwargs["data"]["media_type"] == "IMAGE"
            assert kwargs["data"]["image_url"] == "https://example.com/a.jpg"
            return _MockResp(200, {"id": "c1"})
        return _MockResp(200, {"id": "p1"})

    unit = _unit("caption", image_url="https://example.com/a.jpg")
    with patch("httpx.Client.post", new=fake_post):
        ok, post_id, err = threads.publish(unit)
    assert ok, err
    assert post_id == "p1"


def test_create_container_failure(monkeypatch):
    monkeypatch.setenv("THREADS_USER_ID", "1234")
    monkeypatch.setenv("THREADS_ACCESS_TOKEN", "bad-token")
    monkeypatch.setenv("THREADS_SETTLE_SECONDS", "0")

    def fake_post(self, url, **kwargs):
        return _MockResp(401, {"error": "invalid token"})

    with patch("httpx.Client.post", new=fake_post):
        ok, _, err = threads.publish(_unit("oops"))
    assert not ok
    assert "create container" in err.lower()
    assert "401" in err


def test_publish_failure(monkeypatch):
    monkeypatch.setenv("THREADS_USER_ID", "1234")
    monkeypatch.setenv("THREADS_ACCESS_TOKEN", "token")
    monkeypatch.setenv("THREADS_SETTLE_SECONDS", "0")

    def fake_post(self, url, **kwargs):
        if url.endswith("/threads"):
            return _MockResp(200, {"id": "c1"})
        return _MockResp(500, {"error": "internal"})

    with patch("httpx.Client.post", new=fake_post):
        ok, _, err = threads.publish(_unit("hi"))
    assert not ok
    assert "publish" in err.lower()
    assert "500" in err


def test_text_truncated_to_500_chars(monkeypatch):
    monkeypatch.setenv("THREADS_USER_ID", "1234")
    monkeypatch.setenv("THREADS_ACCESS_TOKEN", "token")
    monkeypatch.setenv("THREADS_SETTLE_SECONDS", "0")

    long_text = "x" * 800
    sent_text: list[str] = []

    def fake_post(self, url, **kwargs):
        if url.endswith("/threads"):
            sent_text.append(kwargs["data"]["text"])
            return _MockResp(200, {"id": "c1"})
        return _MockResp(200, {"id": "p1"})

    with patch("httpx.Client.post", new=fake_post):
        threads.publish(_unit(long_text))
    assert sent_text[0] == "x" * 500


def test_account_suffix_picks_up_scoped_creds(monkeypatch):
    monkeypatch.setenv("THREADS_USER_ID_WORK", "9999")
    monkeypatch.setenv("THREADS_ACCESS_TOKEN_WORK", "work-token")
    monkeypatch.setenv("THREADS_SETTLE_SECONDS", "0")

    captured = {}

    def fake_post(self, url, **kwargs):
        captured["user_id_in_url"] = "9999" in url
        captured["token"] = kwargs["data"]["access_token"]
        if url.endswith("/threads"):
            return _MockResp(200, {"id": "c1"})
        return _MockResp(200, {"id": "p1"})

    with patch("httpx.Client.post", new=fake_post):
        threads.publish(_unit("hi"), account="work")
    assert captured["user_id_in_url"]
    assert captured["token"] == "work-token"


def test_compose_page_shows_threads(monkeypatch, tmp_path):
    """The web composer should list 'threads' as a selectable platform."""
    import importlib
    monkeypatch.setenv("OPEN_DISPATCH_DATA", str(tmp_path))
    import opendispatch.api.queue as q
    importlib.reload(q)
    import opendispatch.api.app as appmod
    importlib.reload(appmod)
    from fastapi.testclient import TestClient
    client = TestClient(appmod.app)
    r = client.get("/compose", headers={"Accept": "text/html"})
    assert r.status_code == 200
    assert 'value="threads"' in r.text
