"""Adapter error redaction and retry classification.

The two Telegram/Instagram credential-leak paths and the ok=True-on-failure
bugs in this repo shared a root cause: nothing scrubbed tokens out of strings
before they reached logs, the queue, the API, or the failure webhook.
"""
from __future__ import annotations

import pytest

from opendispatch.adapters.errors import (
    AUTH,
    MAX_ERR_CHARS,
    PERMANENT,
    PUBLISHED,
    RETRYABLE,
    classify,
    clip,
    prefix_error,
    redact,
)


# ── redaction ───────────────────────────────────────────────────────────────

def test_redacts_telegram_bot_token_in_url_path() -> None:
    url = "https://api.telegram.org/bot123456789:AAHdqTcvCH1vGWJxfSeofSAs0K5PALDsaw/sendMessage"
    out = redact(url)
    assert "123456789:AAHdqTcvCH1vGWJxfSeofSAs0K5PALDsaw" not in out
    assert "sendMessage" in out, "the useful part of the URL must survive"


def test_redacts_access_token_query_param() -> None:
    out = redact("GET /123?fields=status_code&access_token=IGSUPERSECRETVALUE")
    assert "IGSUPERSECRETVALUE" not in out
    assert "fields=status_code" in out


@pytest.mark.parametrize("param", [
    "access_token", "api_key", "apikey", "client_secret", "refresh_token",
    "bot_token", "app_password", "token", "secret",
])
def test_redacts_every_secret_param_name(param: str) -> None:
    out = redact(f"https://x/y?{param}=LEAKCANARY123")
    assert "LEAKCANARY123" not in out, param


def test_redacts_bearer_header_value() -> None:
    out = redact("Authorization: Bearer AAAAAAAAAAAAAAAAAAAAABBBBCCCC")
    assert "AAAAAAAAAAAAAAAAAAAAAABBBBCCCC" not in out
    assert "Bearer" in out


def test_redacts_a_token_split_across_an_exception_message() -> None:
    msg = "Client error '401 Unauthorized' for url 'https://api.telegram.org/bot999999:ZZZSECRETTOKENVALUE/sendPhoto'"
    out = redact(msg)
    assert "ZZZSECRETTOKENVALUE" not in out


def test_redaction_is_idempotent() -> None:
    once = redact("bot111111:AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA")
    assert redact(once) == once


def test_clip_redacts_before_capping() -> None:
    # If we clipped first we could split a token and leak a valid prefix.
    secret = "123456789:" + "A" * 60
    out = clip("bot" + secret + "x" * 600)
    assert secret not in out
    assert len(out) <= MAX_ERR_CHARS


def test_clip_keeps_the_full_500_the_queue_stores() -> None:
    assert len(clip("y" * 900)) == MAX_ERR_CHARS == 500


def test_redact_handles_non_string_input() -> None:
    assert redact(None) == ""
    assert redact(401) == "401"


# ── classification ──────────────────────────────────────────────────────────

@pytest.mark.parametrize("status,expected", [
    (None, RETRYABLE),
    (401, AUTH),
    (403, AUTH),
    (429, RETRYABLE),
    (500, RETRYABLE),
    (502, RETRYABLE),
    (503, RETRYABLE),
    (400, PERMANENT),
    (404, PERMANENT),
    (422, PERMANENT),
    (200, RETRYABLE),
])
def test_classify(status, expected) -> None:
    assert classify(status) == expected


def test_prefix_error_tags_and_caps() -> None:
    out = prefix_error(AUTH, "token rejected")
    assert out.startswith(f"{AUTH}: ")
    assert "token rejected" in out


def test_prefix_error_redacts_inside_the_message() -> None:
    out = prefix_error(PERMANENT, "bad bot999999:AAAABBBBCCCCDDDDEEEEFFFF token")
    assert "AAAABBBBCCCCDDDDEEEEFFFF" not in out


def test_published_class_exists_for_do_not_retry_post() -> None:
    # A post that went out but whose id we could not read must never be
    # retried, or the user gets a duplicate.
    assert PUBLISHED not in (RETRYABLE,)
    assert PUBLISHED == "published"
