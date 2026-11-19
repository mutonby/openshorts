"""The proxy watcher must say what is actually affected, and not page on one miss."""
import asyncio
import pytest

alerts = pytest.importorskip("cloud.alerts")


@pytest.fixture()
def sent(monkeypatch):
    out = []
    async def fake_send(title, body, **kw):
        out.append((title, body))
    monkeypatch.setattr(alerts, "send_admin_alert", fake_send)
    monkeypatch.setenv("STATIC_PROXY_URLS", "http://s1,http://s2")
    monkeypatch.setenv("PROXY_URL", "http://paid")
    alerts._watch_down.clear(); alerts._watch_nag.clear(); alerts._watch_strikes.clear()
    return out


def _fail(name, n=1):
    for _ in range(n):
        asyncio.run(alerts._watch_update(name, False, "ReadTimeout: x"))


def test_one_miss_is_silent(sent):
    _fail(alerts._PAID_TARGET)
    assert sent == []


def test_paid_down_with_statics_up_is_informational(sent):
    _fail(alerts._PAID_TARGET, 2)
    title, body = sent[-1]
    assert title.startswith("ℹ️") and "fallback" in title
    assert "No impact" in body and "static pool is UP" in body
    assert "failing" not in body.split("Probe error")[0]


def test_statics_down_is_a_cost_warning(sent):
    _fail(alerts._STATIC_TARGET, 2)
    title, body = sent[-1]
    assert title.startswith("🟠") and "costs money" in body


def test_both_down_is_red(sent):
    _fail(alerts._STATIC_TARGET, 2)
    _fail(alerts._PAID_TARGET, 2)
    title, body = sent[-1]
    assert title.startswith("🔴") and "ALL YouTube proxies" in title


def test_recovery_resets_strikes(sent):
    _fail(alerts._PAID_TARGET, 2)
    asyncio.run(alerts._watch_update(alerts._PAID_TARGET, True, ""))
    assert sent[-1][0].startswith("✅")
    _fail(alerts._PAID_TARGET, 1)
    assert len(sent) == 2  # a single new miss after recovery stays quiet
