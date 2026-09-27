import asyncio
from unittest.mock import AsyncMock


def test_formats_feedback_as_plain_data(monkeypatch):
    from cloud import cancellation_notifications as notifications
    sender = AsyncMock()
    monkeypatch.setattr(notifications, "send_telegram", sender)
    asyncio.run(notifications.notify_cancellation_feedback(
        "12345678-1234-1234-1234-123456789012", "too_expensive",
        "<b>Too expensive</b>\nPlease improve pricing", rating=2,
        plan="pro", feedback_id="feedback-abc"))
    text = sender.await_args.args[0]
    assert "cancellation requested/feedback received, not confirmed" in text
    assert "user 12345678" in text
    assert "Reason: too_expensive" in text
    assert "Rating: 2" in text
    assert "Plan: pro" in text
    assert "Feedback ID: feedback-abc" in text
    assert "Comment (user-supplied text):\n> <b>Too expensive</b>\n> Please improve pricing" in text
    assert sender.await_args.kwargs == {"raise_errors": True}


def test_redacts_obvious_pii_in_all_free_text(monkeypatch):
    from cloud import cancellation_notifications as notifications
    sender = AsyncMock()
    monkeypatch.setattr(notifications, "send_telegram", sender)
    asyncio.run(notifications.notify_cancellation_feedback(
        "12345678", "email me alice@example.com",
        "Call +1 (212) 555-0199 or 612345678; card 4111 1111 1111 1111. Keep the pricing feedback.",
        plan="bob@example.org", feedback_id="4111111111111111"))
    text = sender.await_args.args[0]
    for secret in ("alice@example.com", "bob@example.org", "212", "555", "0199", "612345678", "4111"):
        assert secret not in text
    assert "[email redacted]" in text
    assert "[number redacted]" in text
    assert "Keep the pricing feedback." in text


def test_trims_fields_and_total_including_telegram_prefix(monkeypatch):
    from cloud import cancellation_notifications as notifications
    from cloud.alerts import TELEGRAM_PREFIX
    sender = AsyncMock()
    monkeypatch.setattr(notifications, "send_telegram", sender)
    asyncio.run(notifications.notify_cancellation_feedback(
        "12345678", "r" * 10000, "😀\n" * 10000,
        plan="p" * 10000, feedback_id="f" * 10000))
    text = sender.await_args.args[0]
    assert len((TELEGRAM_PREFIX + text).encode("utf-16-le")) // 2 < 4096
    assert "…" in text
    assert "Comment (user-supplied text):" in text
    assert "r" * 201 not in text
    assert "p" * 81 not in text


def test_optional_rating_omitted(monkeypatch):
    from cloud import cancellation_notifications as notifications
    sender = AsyncMock()
    monkeypatch.setattr(notifications, "send_telegram", sender)
    asyncio.run(notifications.notify_cancellation_feedback("12345678", "other", ""))
    assert "Rating:" not in sender.await_args.args[0]


def test_send_exceptions_are_swallowed_without_logging_raw_text(monkeypatch, capsys, caplog):
    from cloud import cancellation_notifications as notifications
    sender = AsyncMock(side_effect=RuntimeError("secret raw feedback"))
    monkeypatch.setattr(notifications, "send_telegram", sender)
    assert asyncio.run(notifications.notify_cancellation_feedback(
        "12345678", "other", "secret raw feedback")) is None
    assert "secret raw feedback" not in capsys.readouterr().out + caplog.text


def test_unconfigured_destination_is_a_noop(monkeypatch):
    from cloud import cancellation_notifications as notifications
    from cloud import alerts
    from types import SimpleNamespace
    monkeypatch.setattr(alerts, "settings", SimpleNamespace(telegram_configured=False))
    # Real existing sender takes its no-config branch; any HTTP attempt fails the test.
    import httpx
    from unittest.mock import Mock
    client = Mock(side_effect=AssertionError("network forbidden"))
    monkeypatch.setattr(httpx, "AsyncClient", client)
    assert asyncio.run(notifications.notify_cancellation_feedback("12345678", "other", "")) is None
    client.assert_not_called()
