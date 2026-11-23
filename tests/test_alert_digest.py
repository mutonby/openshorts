"""Telegram policy: only urgent alerts go out on the spot, the rest are held
and counted in the daily digest (cloud/alerts.py, cloud/digest.py)."""
import asyncio
import json

import pytest

alerts = pytest.importorskip("cloud.alerts")


@pytest.fixture
def chat(monkeypatch, tmp_path):
    sent = []

    async def fake_send(text, **kw):
        sent.append(text)

    async def no_email(*a, **k):
        raise AssertionError("no email expected")

    monkeypatch.setattr(alerts, "send_telegram", fake_send)
    monkeypatch.setattr(alerts, "send_email", no_email)
    from types import SimpleNamespace
    monkeypatch.setattr(alerts, "settings", SimpleNamespace(
        admin_email="", smtp_configured=False, telegram_configured=True))
    monkeypatch.delenv("TELEGRAM_ALERTS", raising=False)
    alerts._held_mem.clear()
    alerts.configure_digest(str(tmp_path))
    yield sent
    alerts.configure_digest(None)
    alerts._held_mem.clear()


def test_a_routine_alert_is_held_not_sent(chat, tmp_path):
    asyncio.run(alerts.send_admin_alert("💵 Payment received: $12.00 USD", "user 3f9a1c2b"))
    asyncio.run(alerts.notify("🗑️ Account deleted (plan: free)"))
    assert chat == []
    lines = (tmp_path / ".alerts_held.jsonl").read_text().splitlines()
    assert [json.loads(ln)["k"] for ln in lines] == ["💵 Payment received", "🗑️ Account deleted"]


def test_an_urgent_alert_goes_out_now(chat):
    asyncio.run(alerts.send_admin_alert("🔴 ALL YouTube proxies DOWN", "body", urgent=True))
    asyncio.run(alerts.notify("⛔ Paid proxy daily budget hit", urgent=True))
    assert len(chat) == 2 and "DOWN" in chat[0]


def test_per_event_mode_sends_everything(chat, monkeypatch):
    monkeypatch.setenv("TELEGRAM_ALERTS", "all")
    asyncio.run(alerts.send_admin_alert("🎉 New subscriber", "user x"))
    assert len(chat) == 1


def test_digest_key_drops_the_changing_figures():
    assert alerts.digest_key("⚠️ La cola de render sigue atascada (desde hace 3 h)") \
        == "⚠️ La cola de render sigue atascada"
    assert alerts.digest_key("💰 Top-up purchased ($5.00 USD)") == "💰 Top-up purchased"
    assert alerts.digest_key("🎬 Clips created by a paying user (pro)\nmore") \
        == "🎬 Clips created by a paying user"


def test_take_held_counts_sums_and_empties(chat):
    for amount in ("$12.00", "$29.00"):
        asyncio.run(alerts.send_admin_alert(f"💵 Payment received: {amount} USD", ""))
    asyncio.run(alerts.send_admin_alert("🔻 Subscription canceled", ""))
    taken = alerts.take_held(until=10 ** 12)
    assert alerts.summarize_held(taken) == [
        "💵 Payment received ×2 ($41.00)", "🔻 Subscription canceled ×1"]
    assert alerts.take_held(until=10 ** 12) == []


def test_take_held_keeps_what_came_after(chat, tmp_path):
    path = tmp_path / ".alerts_held.jsonl"
    path.write_text(json.dumps({"t": 100, "k": "old"}) + "\n"
                    + json.dumps({"t": 300, "k": "new"}) + "\n")
    assert [e["k"] for e in alerts.take_held(until=200)] == ["old"]
    assert [e["k"] for e in alerts.take_held(until=400)] == ["new"]


def test_failure_rate_pages_for_an_outage_not_for_user_videos(chat, monkeypatch):
    calls = []

    async def fake_admin(subject, body, *, urgent=False):
        calls.append((subject, urgent))
    monkeypatch.setattr(alerts, "send_admin_alert", fake_admin)
    alerts._recent.clear()
    alerts._last_alert.clear()
    for _ in range(6):
        asyncio.run(alerts.record_job_outcome(False, "Gemini did not return usable clips"))
    assert calls and calls[-1][1] is False
    alerts._recent.clear()
    alerts._last_alert.clear()
    calls.clear()
    for _ in range(6):
        asyncio.run(alerts.record_job_outcome(False, "ffmpeg exited with code 187"))
    assert calls == [("⚠️ High job failure rate (ffmpeg/render)", True)]


def test_a_depleted_ai_account_pages_on_the_first_job(chat, monkeypatch):
    calls = []

    async def fake_admin(subject, body, *, urgent=False):
        calls.append((subject, urgent))
    monkeypatch.setattr(alerts, "send_admin_alert", fake_admin)
    alerts._recent.clear()
    alerts._last_alert.clear()
    err = ("Gemini Error: 429 RESOURCE_EXHAUSTED. Your prepayment credits are depleted. "
           "RuntimeError: Clip detection failed — the AI model did not return usable clips")
    asyncio.run(alerts.record_job_outcome(False, err))
    assert len(calls) == 1 and calls[0][1] is True and "AI provider" in calls[0][0]
    for _ in range(8):                                   # one page an hour, not one per job
        asyncio.run(alerts.record_job_outcome(False, err))
    assert len(calls) == 1
    alerts._last_alert["ai_account"] -= 3601
    asyncio.run(alerts.record_job_outcome(False, err))
    assert len(calls) == 2
    alerts._recent.clear()
    alerts._last_alert.clear()


def test_proxy_watch_pages_only_what_matters(monkeypatch):
    calls = []

    async def fake_admin(subject, body, *, urgent=False):
        calls.append((subject, urgent))
    monkeypatch.setattr(alerts, "send_admin_alert", fake_admin)
    monkeypatch.setenv("STATIC_PROXY_URLS", "http://s1")
    monkeypatch.setenv("PROXY_URL", "http://paid")
    for d in (alerts._watch_down, alerts._watch_nag, alerts._watch_strikes, alerts._watch_paged):
        d.clear()

    async def go():
        for _ in range(2):
            await alerts._watch_update(alerts._PAID_TARGET, False, "timeout")
        await alerts._watch_update(alerts._PAID_TARGET, True, "")
        for _ in range(2):
            await alerts._watch_update(alerts._STATIC_TARGET, False, "403")
        await alerts._watch_update(alerts._STATIC_TARGET, True, "")
    asyncio.run(go())
    assert [u for _, u in calls] == [False, False, True, True]


def test_digest_message_lists_queue_and_held_alerts():
    digest = pytest.importorskip("cloud.digest")
    stats = {"signups": 10, "jobs": 5, "processors": 4, "refunded": 1, "minutes": 50.0,
             "clips": 12, "stripe": {"invoice.paid": 2}, "paid_proxy_mb": 340.0}
    text = digest._format(stats, ["💵 Payment received ×2 ($41.00)"], ["En cola: 3 vídeos."])
    assert "Altas: 10" in text and "Proxy de pago: 340 MB" in text
    assert "Cola ahora:\nEn cola: 3 vídeos." in text
    assert text.endswith("💵 Payment received ×2 ($41.00)")
