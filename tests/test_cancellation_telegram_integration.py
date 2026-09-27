"""Cancellation feedback reaches existing Telegram without a live network call."""
from unittest.mock import AsyncMock
from test_cancellation_feedback import setup  # noqa: F401
from cloud import cancellation_notifications as alerts


def test_feedback_reaches_existing_telegram(setup, monkeypatch):
    send = AsyncMock()
    monkeypatch.setattr(alerts, 'send_telegram', send)
    response = setup.client.post('/api/billing/cancel', json={
        'reason': 'quality', 'comment': 'The clips cut off too early', 'rating': 2})
    assert response.status_code == 200
    send.assert_awaited_once()
    message = send.call_args.args[0]
    assert 'The clips cut off too early' in message
    assert '2' in message
    assert setup.user.email not in message


def test_skip_does_not_send_telegram(setup, monkeypatch):
    send = AsyncMock()
    monkeypatch.setattr(alerts, 'send_telegram', send)
    assert setup.client.post('/api/billing/cancel', json={}).status_code == 200
    send.assert_not_awaited()


def test_telegram_failure_does_not_block_portal(setup, monkeypatch):
    send = AsyncMock(side_effect=RuntimeError('Telegram down'))
    monkeypatch.setattr(alerts, 'send_telegram', send)
    response = setup.client.post('/api/billing/cancel', json={'comment': 'Too slow'})
    assert response.status_code == 200
    assert response.json()['url'].startswith('https://billing.stripe.test/')
