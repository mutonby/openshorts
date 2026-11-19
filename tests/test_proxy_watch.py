"""Proxy watcher: alert on down, nag until fixed, confirm recovery.

Born from the 19-aug-2026 incident: DataImpulse ran out of traffic
(407 TRAFFIC_EXHAUSTED), every YouTube-URL job failed, and nobody was told
more than once. The watcher probes every configured route (static ISP pool
first, per-GB paid proxy last) and keeps nagging while one is down.
"""
import asyncio

import pytest

from cloud import alerts


def _run(coro):
    return asyncio.run(coro)


@pytest.fixture(autouse=True)
def _reset_state(monkeypatch):
    alerts._watch_down.clear()
    alerts._watch_nag.clear()
    alerts._watch_strikes.clear()
    alerts._last_alert.clear()
    sent = []

    async def fake_alert(subject, body, **kw):
        sent.append((subject, body))

    monkeypatch.setattr(alerts, "send_admin_alert", fake_alert)
    yield sent
    alerts._watch_down.clear()
    alerts._watch_nag.clear()
    alerts._watch_strikes.clear()


def _probes(results):
    """Fake _probe_one returning per-url results (dict url -> (ok, detail))."""
    async def fake_probe(url):
        return results[url]
    return fake_probe


class TestWatchTargets:
    def test_no_env_no_targets(self, monkeypatch):
        monkeypatch.delenv("STATIC_PROXY_URLS", raising=False)
        monkeypatch.delenv("PROXY_URL", raising=False)
        assert alerts._watch_targets() == []

    def test_both_pools_in_order(self, monkeypatch):
        monkeypatch.setenv("STATIC_PROXY_URLS", "http://s1, http://s2 ,")
        monkeypatch.setenv("PROXY_URL", "http://paid")
        assert alerts._watch_targets() == [
            (alerts._STATIC_TARGET, ["http://s1", "http://s2"]),
            (alerts._PAID_TARGET, ["http://paid"]),
        ]


class TestWatchTick:
    def _env(self, monkeypatch, statics="http://s1,http://s2", paid="http://paid"):
        if statics:
            monkeypatch.setenv("STATIC_PROXY_URLS", statics)
        else:
            monkeypatch.delenv("STATIC_PROXY_URLS", raising=False)
        if paid:
            monkeypatch.setenv("PROXY_URL", paid)
        else:
            monkeypatch.delenv("PROXY_URL", raising=False)

    def test_all_healthy_is_silent(self, _reset_state, monkeypatch):
        self._env(monkeypatch)
        monkeypatch.setattr(alerts, "_probe_one", _probes({
            "http://s1": (True, ""), "http://paid": (True, "")}))
        _run(alerts.proxy_watch_tick())
        assert _reset_state == []

    def test_pool_is_up_if_any_ip_answers(self, _reset_state, monkeypatch):
        self._env(monkeypatch)
        monkeypatch.setattr(alerts, "_probe_one", _probes({
            "http://s1": (False, "HTTP 407"), "http://s2": (True, ""),
            "http://paid": (True, "")}))
        _run(alerts.proxy_watch_tick())
        assert _reset_state == []

    def test_static_pool_down_alerts_with_cost_hint(self, _reset_state, monkeypatch):
        self._env(monkeypatch)
        monkeypatch.setattr(alerts, "_probe_one", _probes({
            "http://s1": (False, "boom"), "http://s2": (False, "boom"),
            "http://paid": (True, "")}))
        _run(alerts.proxy_watch_tick())
        assert _reset_state == []  # one miss is not an outage
        _run(alerts.proxy_watch_tick())
        assert len(_reset_state) == 1
        subject, body = _reset_state[0]
        # The statics being down while the paid proxy answers is a cost
        # warning, not an outage — hence "down" rather than "DOWN".
        assert alerts._STATIC_TARGET in subject and "down" in subject.lower()
        assert "costs money" in body

    def test_paid_down_alerts_and_nags_after_window(self, _reset_state, monkeypatch):
        self._env(monkeypatch, statics=None)
        monkeypatch.setattr(alerts, "_probe_one", _probes({
            "http://paid": (False, "HTTP 407")}))
        for _ in range(alerts._PROXY_STRIKES):
            _run(alerts.proxy_watch_tick())
        alerts._watch_nag[alerts._PAID_TARGET] -= alerts._PROXY_RENOTIFY + 1
        _run(alerts.proxy_watch_tick())
        assert len(_reset_state) == 2
        # The nag repeats the headline and adds how long it has been down.
        assert alerts._PAID_TARGET in _reset_state[1][0]
        assert "h)" in _reset_state[1][0]

    def test_recovery_confirms_and_resets(self, _reset_state, monkeypatch):
        self._env(monkeypatch, statics=None)
        monkeypatch.setattr(alerts, "_probe_one", _probes({
            "http://paid": (False, "HTTP 407")}))
        for _ in range(alerts._PROXY_STRIKES):
            _run(alerts.proxy_watch_tick())
        monkeypatch.setattr(alerts, "_probe_one", _probes({
            "http://paid": (True, "")}))
        _run(alerts.proxy_watch_tick())
        assert any("recovered" in s for s, _ in _reset_state)
        assert not alerts._watch_down.get(alerts._PAID_TARGET)
        # A later failure is a NEW incident: strikes were reset by the
        # recovery, so it has to miss twice again before it alerts.
        monkeypatch.setattr(alerts, "_probe_one", _probes({
            "http://paid": (False, "HTTP 407")}))
        for _ in range(alerts._PROXY_STRIKES):
            _run(alerts.proxy_watch_tick())
        assert sum("DOWN" in s for s, _ in _reset_state) == 2


class TestJobFailureOpensIncident:
    def test_proxy_job_error_opens_paid_incident_and_alerts(self, _reset_state):
        _run(alerts.record_job_outcome(
            False,
            "yt_dlp.utils.DownloadError: Unable to connect to proxy "
            "('Tunnel connection failed: 407 TRAFFIC_EXHAUSTED')"))
        assert alerts._watch_down.get(alerts._PAID_TARGET)
        assert len(_reset_state) == 1

    def test_non_proxy_error_leaves_incident_closed(self, _reset_state):
        _run(alerts.record_job_outcome(False, "ffmpeg exploded"))
        assert not alerts._watch_down.get(alerts._PAID_TARGET)


class TestCookieSession:
    """The cookies are a route too: a rotated session makes every download
    anonymous while the pool probe still says UP (30-sep-2026)."""

    def test_note_session_reads_the_ytcfg_marker(self):
        alerts._session_seen["logged_in"] = None
        alerts._note_session('..."LOGGED_IN":false,"EVENT_ID"...')
        assert alerts._session_seen["logged_in"] is False
        alerts._note_session('"LOGGED_IN":true')
        assert alerts._session_seen["logged_in"] is True
        alerts._session_seen["logged_in"] = None
        alerts._note_session("no player at all")
        assert alerts._session_seen["logged_in"] is None

    def test_expired_session_alerts_on_first_probe_with_the_fix(self, _reset_state, monkeypatch):
        monkeypatch.setenv("STATIC_PROXY_URLS", "http://s1")
        monkeypatch.setattr(alerts, "_probe_cookies", lambda: object())

        async def fake_probe(url):
            alerts._session_seen["logged_in"] = False
            return True, ""  # the IP still answers: pool is UP
        monkeypatch.setattr(alerts, "_probe_one", fake_probe)
        _run(alerts.proxy_watch_tick())
        assert len(_reset_state) == 1
        subject, body = _reset_state[0]
        assert "cookies expired" in subject
        assert "YOUTUBE_COOKIES" in body and "private window" in body
        # Still down on the next tick: no second alert inside the nag window.
        _run(alerts.proxy_watch_tick())
        assert len(_reset_state) == 1

    def test_live_session_and_no_cookies_stay_silent(self, _reset_state, monkeypatch):
        monkeypatch.setenv("STATIC_PROXY_URLS", "http://s1")

        async def fake_probe(url):
            alerts._session_seen["logged_in"] = True
            return True, ""
        monkeypatch.setattr(alerts, "_probe_one", fake_probe)
        monkeypatch.setattr(alerts, "_probe_cookies", lambda: object())
        _run(alerts.proxy_watch_tick())
        assert _reset_state == []
        # Self-host without cookies: the marker says false, nobody is paged.
        monkeypatch.setattr(alerts, "_probe_cookies", lambda: None)

        async def anon_probe(url):
            alerts._session_seen["logged_in"] = False
            return True, ""
        monkeypatch.setattr(alerts, "_probe_one", anon_probe)
        _run(alerts.proxy_watch_tick())
        assert _reset_state == []
