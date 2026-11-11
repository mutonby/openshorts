"""A failed Bot API call must not print or propagate the bot token or chat id.

httpx puts the request URL (``.../bot<token>/sendMessage``) in its error
messages; the old handler printed ``{e}`` and, with ``raise_errors``, re-raised
it. The client is faked here: nothing leaves the process.
"""
import asyncio

import pytest

alerts = pytest.importorskip("cloud.alerts")
httpx = pytest.importorskip("httpx")

TOKEN = "123456789:AAFakeTokenForTests_abcdefghijklmno"
CHAT = "-1009876543210"


@pytest.fixture
def bot(monkeypatch):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", TOKEN)
    monkeypatch.setenv("TELEGRAM_CHAT_ID", CHAT)

    class FailingClient:
        def __init__(self, *a, **k):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def post(self, url, json=None):
            req = httpx.Request("POST", url, json=json)
            resp = httpx.Response(401, request=req, text=f"chat {CHAT} unauthorized")
            resp.raise_for_status()

    monkeypatch.setattr(httpx, "AsyncClient", FailingClient)


def _clean(text):
    assert TOKEN not in text and TOKEN.split(":")[1] not in text
    assert CHAT not in text


def test_logged_failure_is_redacted(bot, capsys):
    asyncio.run(alerts.send_telegram("hi"))
    out = capsys.readouterr().out
    assert "Telegram alert failed" in out and "<bot-token>" in out
    _clean(out)


def test_propagated_failure_is_redacted_and_unchained(bot):
    with pytest.raises(alerts.TelegramSendError) as e:
        asyncio.run(alerts.send_telegram("hi", raise_errors=True))
    _clean(str(e.value))
    _clean(repr(e.value.args))
    assert e.value.__cause__ is None and e.value.__context__ is None


def test_connect_error_is_redacted_too(bot, monkeypatch):
    class Down:
        def __init__(self, *a, **k):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def post(self, url, json=None):
            raise httpx.ConnectError(f"cannot reach {url}")
    monkeypatch.setattr(httpx, "AsyncClient", Down)
    with pytest.raises(alerts.TelegramSendError) as e:
        asyncio.run(alerts.send_telegram("hi", raise_errors=True))
    _clean(str(e.value))


def test_redact_handles_any_bot_url():
    assert alerts.redact_telegram("x /bot42:abc-DEF_1/sendMessage") == "x /bot<bot-token>/sendMessage"
