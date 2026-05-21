"""Tests for integrations/fal: submit_and_poll + upload_file.

httpx is mocked. We verify request shape (headers, URLs, JSON body),
poll loop behavior (COMPLETED / FAILED / timeout), and the synchronous
fast-path (when the queue returns a result immediately without a
request_id).
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from app.integrations.fal import (
    FAL_QUEUE_BASE,
    FalError,
    submit_and_poll,
    upload_file,
)


# ── submit_and_poll ─────────────────────────────────────────────────────


def _mock_client_factory(responses):
    """Return a function that yields a context-manager-style httpx.Client
    whose .post/.get pop responses from the supplied list."""
    iter_resp = iter(responses)

    def _factory(*args, **kwargs):
        ctx = MagicMock()

        def _next(*_a, **_kw):
            r = next(iter_resp)
            if isinstance(r, Exception):
                raise r
            return r

        ctx.__enter__.return_value = MagicMock(post=_next, get=_next)
        ctx.__exit__.return_value = False
        return ctx

    return _factory


def test_submit_and_poll_returns_completed_result():
    submit_resp = MagicMock(status_code=200)
    submit_resp.json.return_value = {
        "request_id": "abc",
        "status_url": f"{FAL_QUEUE_BASE}/m/requests/abc/status",
        "response_url": f"{FAL_QUEUE_BASE}/m/requests/abc",
    }
    status_resp = MagicMock(status_code=200)
    status_resp.json.return_value = {"status": "COMPLETED"}
    result_resp = MagicMock(status_code=200)
    result_resp.json.return_value = {"video": {"url": "https://fake.fal.ai/out.mp4"}}

    factory = _mock_client_factory([submit_resp, status_resp, result_resp])

    with patch("app.integrations.fal.httpx.Client", side_effect=factory), \
         patch("app.integrations.fal.time.sleep"):
        out = submit_and_poll("m", {"k": "v"}, "fkey", poll_interval=0)

    assert out == {"video": {"url": "https://fake.fal.ai/out.mp4"}}


def test_submit_and_poll_synchronous_fastpath():
    """If queue returns a result without request_id, it's a sync gen."""
    submit_resp = MagicMock(status_code=200)
    submit_resp.json.return_value = {"video": {"url": "x"}}
    factory = _mock_client_factory([submit_resp])

    with patch("app.integrations.fal.httpx.Client", side_effect=factory):
        out = submit_and_poll("m", {}, "fkey")
    assert out == {"video": {"url": "x"}}


def test_submit_and_poll_raises_on_submit_error():
    bad = MagicMock(status_code=500, text="nope")
    factory = _mock_client_factory([bad])
    with patch("app.integrations.fal.httpx.Client", side_effect=factory):
        with pytest.raises(FalError, match="500"):
            submit_and_poll("m", {}, "fkey")


def test_submit_and_poll_raises_on_failed_status():
    submit_resp = MagicMock(status_code=200)
    submit_resp.json.return_value = {
        "request_id": "abc",
        "status_url": f"{FAL_QUEUE_BASE}/m/requests/abc/status",
        "response_url": f"{FAL_QUEUE_BASE}/m/requests/abc",
    }
    failed = MagicMock(status_code=200)
    failed.json.return_value = {"status": "FAILED", "error": "model crashed"}
    factory = _mock_client_factory([submit_resp, failed])

    with patch("app.integrations.fal.httpx.Client", side_effect=factory), \
         patch("app.integrations.fal.time.sleep"):
        with pytest.raises(FalError, match="model crashed"):
            submit_and_poll("m", {}, "fkey", poll_interval=0)


def test_submit_and_poll_times_out():
    submit_resp = MagicMock(status_code=200)
    submit_resp.json.return_value = {
        "request_id": "abc",
        "status_url": f"{FAL_QUEUE_BASE}/m/requests/abc/status",
        "response_url": f"{FAL_QUEUE_BASE}/m/requests/abc",
    }
    in_progress = MagicMock(status_code=200)
    in_progress.json.return_value = {"status": "IN_QUEUE"}
    # Repeat in_progress forever — the time mock will trip the timeout
    factory = _mock_client_factory([submit_resp] + [in_progress] * 100)

    fake_time = iter([0.0, 1.0, 99999.0])

    with patch("app.integrations.fal.httpx.Client", side_effect=factory), \
         patch("app.integrations.fal.time.sleep"), \
         patch("app.integrations.fal.time.time", lambda: next(fake_time)):
        with pytest.raises(FalError, match="timed out"):
            submit_and_poll("m", {}, "fkey", timeout=10, poll_interval=0)


def test_submit_and_poll_authorization_header():
    """The Authorization header must use the ``Key <fal_key>`` scheme."""
    submit_resp = MagicMock(status_code=200)
    submit_resp.json.return_value = {"video": {"url": "x"}}
    captured = {}

    def _factory(*args, **kwargs):
        ctx = MagicMock()

        def _post(url, *, headers, json):
            captured["headers"] = headers
            captured["json"] = json
            captured["url"] = url
            return submit_resp

        ctx.__enter__.return_value = MagicMock(post=_post)
        ctx.__exit__.return_value = False
        return ctx

    with patch("app.integrations.fal.httpx.Client", side_effect=_factory):
        submit_and_poll("my-model", {"video_url": "u"}, "secret")

    assert captured["headers"]["Authorization"] == "Key secret"
    assert captured["url"] == f"{FAL_QUEUE_BASE}/my-model"
    assert captured["json"] == {"video_url": "u"}


# ── upload_file ────────────────────────────────────────────────────────


def test_upload_file_two_step(tmp_path):
    src = tmp_path / "ref.png"
    src.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 50)

    initiate_resp = MagicMock(status_code=200)
    initiate_resp.json.return_value = {
        "upload_url": "https://upload.fal/abc",
        "file_url": "https://cdn.fal/abc.png",
    }
    initiate_resp.raise_for_status = MagicMock()
    put_resp = MagicMock(status_code=200)
    put_resp.raise_for_status = MagicMock()

    calls = []

    def _factory(*args, **kwargs):
        ctx = MagicMock()

        def _post(url, **kw):
            calls.append(("POST", url, kw))
            return initiate_resp

        def _put(url, **kw):
            calls.append(("PUT", url, kw))
            return put_resp

        ctx.__enter__.return_value = MagicMock(post=_post, put=_put)
        ctx.__exit__.return_value = False
        return ctx

    with patch("app.integrations.fal.httpx.Client", side_effect=_factory):
        out_url = upload_file(str(src), "fkey")

    assert out_url == "https://cdn.fal/abc.png"
    # POST initiate then PUT bytes
    assert calls[0][0] == "POST"
    assert calls[1][0] == "PUT"
    assert calls[1][1] == "https://upload.fal/abc"


def test_upload_file_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        upload_file(str(tmp_path / "nope.png"), "fkey")


# ── URL allowlist (Codex HIGH-2: queue URL trust) ───────────────────────


def test_submit_and_poll_rejects_attacker_controlled_status_url():
    """A submit response carrying a status_url to a non-fal host must be
    rejected before we send the fal API key in an Authorization header."""
    submit_resp = MagicMock(status_code=200)
    submit_resp.json.return_value = {
        "request_id": "abc",
        "status_url": "http://evil.example.com/leak",
        "response_url": f"{FAL_QUEUE_BASE}/m/requests/abc",
    }
    factory = _mock_client_factory([submit_resp])

    with patch("app.integrations.fal.httpx.Client", side_effect=factory):
        with pytest.raises(FalError, match="untrusted fal queue URL"):
            submit_and_poll("m", {}, "secret")


def test_submit_and_poll_rejects_attacker_controlled_response_url():
    """Same defense applies to response_url."""
    submit_resp = MagicMock(status_code=200)
    submit_resp.json.return_value = {
        "request_id": "abc",
        "status_url": f"{FAL_QUEUE_BASE}/m/requests/abc/status",
        "response_url": "https://attacker.test/exfil",
    }
    factory = _mock_client_factory([submit_resp])

    with patch("app.integrations.fal.httpx.Client", side_effect=factory):
        with pytest.raises(FalError, match="untrusted fal queue URL"):
            submit_and_poll("m", {}, "secret")


def test_submit_and_poll_rejects_non_https_queue_url():
    """Plain HTTP queue URL must be rejected — the API key would travel
    in cleartext."""
    submit_resp = MagicMock(status_code=200)
    submit_resp.json.return_value = {
        "request_id": "abc",
        "status_url": "http://queue.fal.run/m/requests/abc/status",
        "response_url": f"{FAL_QUEUE_BASE}/m/requests/abc",
    }
    factory = _mock_client_factory([submit_resp])

    with patch("app.integrations.fal.httpx.Client", side_effect=factory):
        with pytest.raises(FalError, match="untrusted fal queue URL"):
            submit_and_poll("m", {}, "secret")
