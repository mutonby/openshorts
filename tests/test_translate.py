"""Tests for the ElevenLabs retry helper."""
import pytest

import translate
from translate import _with_retry, _TransientHTTPError


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    monkeypatch.setattr(translate.time, "sleep", lambda seconds: None)


def test_succeeds_after_transient_failures():
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise _TransientHTTPError("HTTP 502")
        return "ok"

    assert _with_retry(flaky, "test op") == "ok"
    assert calls["n"] == 3


def test_gives_up_after_max_attempts():
    def always_fails():
        raise _TransientHTTPError("HTTP 503")

    with pytest.raises(Exception, match="failed after 3 attempts"):
        _with_retry(always_fails, "test op")


def test_non_transient_fails_fast():
    calls = {"n": 0}

    def bad_request():
        calls["n"] += 1
        raise ValueError("HTTP 400 - bad request")

    with pytest.raises(ValueError):
        _with_retry(bad_request, "test op")
    assert calls["n"] == 1
