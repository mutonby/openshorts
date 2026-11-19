"""Where the paid proxy bytes come from, and the trail that survives a deploy.

DataImpulse showed ~0.2 $/day of traffic with the static pool healthy, and
$14 on 28-aug-2026 that nothing on the server could explain (in-memory
counter, hourly log rotation). These pin the two leaks found and the ledger.
"""
import asyncio
import json
import os

import pytest

from cloud import metering, proxy_ledger


def _patch_ydl(monkeypatch, cls):
    """Route yt-dlp construction through a stub; skips where the CI's minimal
    env has no yt_dlp (cv2/torch-style: the module itself is the heavy dep)."""
    yt_dlp = pytest.importorskip("yt_dlp")
    monkeypatch.setattr(yt_dlp, "YoutubeDL", cls)


# --- probe policy ----------------------------------------------------------

class TestStaticFailureWarrantsPaid:
    def test_ip_specific_failures_do(self):
        for err in ("ERROR: Sign in to confirm you're not a bot",
                    # fake on Decodo IPs: 5 videos "unavailable" on all three
                    # statics downloaded fine through the paid proxy (1-sep)
                    "ERROR: [youtube] BJW5gAgk4bg: Video unavailable",
                    "HTTP Error 403: Forbidden", "HTTP Error 429: Too Many Requests",
                    "ProxyError: Tunnel connection failed: 407",
                    "Unable to download webpage: The read operation timed out"):
            assert metering.static_failure_warrants_paid(err), err

    def test_content_failures_do_not(self):
        for err in ("ERROR: Private video. Sign in if you've been granted access",
                    "This video has been removed by the uploader",
                    "Join this channel to get access to members-only content",
                    "no duration in metadata", "Premieres in 3 hours",
                    "Unsupported URL: https://shopee.vn/product", "Sign in to confirm your age",
                    # the paid pool exits from the same blocked countries: 5 of 6
                    # paid probes 3-5 sep 2026 failed with the identical error
                    "The uploader has not made this video available in your country"):
            assert not metering.static_failure_warrants_paid(err), err

    def test_unknown_errors_stay_free(self):
        assert not metering.static_failure_warrants_paid("something odd happened")
        assert not metering.static_failure_warrants_paid("")


def test_non_youtube_urls_are_recognised():
    assert metering.is_youtube_url("https://www.youtube.com/watch?v=x")
    assert metering.is_youtube_url("https://youtu.be/x")
    assert metering.is_youtube_url("https://m.youtube.com/shorts/x")
    for u in ("https://www.twitch.tv/videos/1", "https://kick.com/x", "https://rumble.com/v",
              "https://drive.google.com/file/d/x", "https://www.openshorts.app/videos/x.mp4"):
        assert not metering.is_youtube_url(u), u


def test_probe_never_reaches_paid_for_non_youtube(monkeypatch):
    """Twitch/Kick/Rumble/product pages were all showing up on the paid proxy."""
    monkeypatch.setenv("PROXY_URL", "http://paid")
    monkeypatch.setenv("STATIC_PROXY_URLS", "http://s1")
    monkeypatch.delenv("DIRECT_FIRST", raising=False)
    monkeypatch.setenv("BGUTIL_SCRIPT_PATH", "")
    monkeypatch.setenv("BGUTIL_BASE_URL", "")
    seen = []

    class _FakeYDL:
        def __init__(self, opts): seen.append(opts.get("proxy"))
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def extract_info(self, url, download=False): raise RuntimeError("HTTP Error 403: Forbidden")

    _patch_ydl(monkeypatch, _FakeYDL)
    monkeypatch.setattr(metering, "_ffprobe_url_seconds", lambda url, timeout=30: 0.0)
    with pytest.raises(ValueError):
        metering.probe_url_minutes("https://www.twitch.tv/videos/123")
    assert "http://paid" not in seen
    assert metering.pop_paid_probe_events() == []


def test_probe_reaches_paid_only_for_ip_specific_failures(monkeypatch):
    monkeypatch.setenv("PROXY_URL", "http://paid")
    monkeypatch.setenv("STATIC_PROXY_URLS", "http://s1")
    monkeypatch.delenv("DIRECT_FIRST", raising=False)
    monkeypatch.setenv("BGUTIL_SCRIPT_PATH", "")
    monkeypatch.setenv("BGUTIL_BASE_URL", "")

    def run(static_error):
        seen = []

        class _FakeYDL:
            def __init__(self, opts): self.proxy = opts.get("proxy"); seen.append(self.proxy)
            def __enter__(self): return self
            def __exit__(self, *a): return False
            def extract_info(self, url, download=False):
                if self.proxy == "http://paid":
                    return {"duration": 600}
                raise RuntimeError(static_error)

        _patch_ydl(monkeypatch, _FakeYDL)
        monkeypatch.setattr(metering, "_ffprobe_url_seconds", lambda url, timeout=30: 0.0)
        metering.pop_paid_probe_events()
        try:
            minutes = metering.probe_url_minutes("https://www.youtube.com/watch?v=abc")
        except ValueError:
            minutes = None
        return minutes, seen, metering.pop_paid_probe_events()

    minutes, seen, events = run("Sign in to confirm you're not a bot")
    assert minutes == 10.0 and "http://paid" in seen
    assert len(events) == 1 and "static1" in events[0]["static_errors"]

    minutes, seen, events = run("ERROR: Private video")
    assert minutes is None and "http://paid" not in seen and events == []


# --- the trail -------------------------------------------------------------

def _route(winner, attempts):
    return {"winner": winner, "paid_bytes": sum(a["bytes"] for a in attempts if a.get("paid")),
            "attempts": attempts}


def test_route_line_round_trip():
    route = _route("fallback", [
        {"label": "HD-static1", "ok": False, "bytes": 0, "paid": False, "error": "HTTP Error 403"},
        {"label": "HD", "ok": False, "bytes": 31_000_000, "paid": True, "error": "read timed out"},
        {"label": "fallback", "ok": True, "bytes": 70_000_000, "paid": True},
    ])
    parsed = proxy_ledger.parse_route_line("PROXY_ROUTE=" + json.dumps(route))
    assert parsed == route
    assert proxy_ledger.parse_route_line("PROXY_ROUTE=not json") is None
    assert proxy_ledger.parse_route_line("other") is None


def test_paid_bytes_count_failed_paid_attempts_too():
    route = _route("fallback", [
        {"label": "HD", "ok": False, "bytes": 31_000_000, "paid": True, "error": "x"},
        {"label": "fallback", "ok": True, "bytes": 70_000_000, "paid": True},
    ])
    assert proxy_ledger.paid_bytes_of(route) == 101_000_000
    assert proxy_ledger.used_paid(route)


def test_free_download_is_not_paid():
    route = _route("HD-static1", [{"label": "HD-static1", "ok": True, "bytes": 90_000_000, "paid": False}])
    assert not proxy_ledger.used_paid(route)
    assert proxy_ledger.describe_failures(route) == ""


def test_failure_summary_names_the_free_routes_that_failed():
    route = _route("HD", [
        {"label": "HD-static1", "ok": False, "bytes": 0, "paid": False, "error": "Sign in to confirm you're not a bot"},
        {"label": "HD-static2", "ok": False, "bytes": 0, "paid": False, "error": "HTTP Error 429"},
        {"label": "HD", "ok": True, "bytes": 50_000_000, "paid": True},
    ])
    text = proxy_ledger.describe_failures(route)
    assert "HD-static1: Sign in" in text and "HD-static2: HTTP Error 429" in text
    assert "HD:" not in text


def test_alert_folds_a_burst_into_one_message(monkeypatch):
    sent = []

    async def fake_send(text):
        sent.append(text)

    import cloud.alerts as alerts
    monkeypatch.setattr(alerts, "send_telegram", fake_send)
    # The per-event mode: by default these go to the daily digest instead.
    monkeypatch.setenv("TELEGRAM_ALERTS", "all")
    monkeypatch.setattr(proxy_ledger, "_persist", lambda *a, **k: asyncio.sleep(0))
    proxy_ledger._pending.update(count=0, bytes=0, lines=[], last_sent=0.0)

    async def go():
        for i in range(4):
            await proxy_ledger.record_download(f"job{i}xxxx", _route("fallback", [
                {"label": "HD-static1", "ok": False, "bytes": 0, "paid": False, "error": "HTTP Error 403"},
                {"label": "fallback", "ok": True, "bytes": 10_000_000, "paid": True}]),
                "https://www.youtube.com/watch?v=x")
    asyncio.run(go())
    assert len(sent) == 1                      # the other three wait for the cooldown
    assert "DataImpulse" in sent[0] and "10.0 MB" in sent[0] and "HD-static1: HTTP Error 403" in sent[0]
    assert proxy_ledger._pending["count"] == 3


def test_host_of_strips_credentials_and_paths():
    assert proxy_ledger.host_of("https://user:pw@www.youtube.com/watch?v=1") == "www.youtube.com"
    assert proxy_ledger.host_of(None) is None


def test_job_source_url_skips_the_interpreter_flag():
    app = pytest.importorskip("app")
    job = {"cmd": ["/usr/bin/python3", "-u", "main.py", "-u", "https://www.youtube.com/watch?v=x", "-o", "out"]}
    assert app._job_source_url(job) == "https://www.youtube.com/watch?v=x"
    assert app._job_source_url({"cmd": ["/usr/bin/python3", "-u", "main.py", "-i", "file.mp4"]}) is None


# --- daily budget ----------------------------------------------------------

class TestDailyBudget:
    def _prime(self, monkeypatch, used_mb, budget_mb=500):
        async def fake_today():
            return int(used_mb * 1e6)
        monkeypatch.setattr(proxy_ledger, "paid_bytes_today", fake_today)
        monkeypatch.setattr(proxy_ledger, "DAILY_BUDGET_MB", budget_mb)
        proxy_ledger._budget_alerted["day"] = None

    def test_under_budget_allows_paid(self, monkeypatch):
        self._prime(monkeypatch, used_mb=100)
        assert asyncio.run(proxy_ledger.budget_exceeded()) is False

    def test_over_budget_blocks_and_alerts_once_per_day(self, monkeypatch):
        self._prime(monkeypatch, used_mb=600)
        sent = []

        async def fake_send(text):
            sent.append(text)
        import cloud.alerts as alerts
        monkeypatch.setattr(alerts, "send_telegram", fake_send)
        assert asyncio.run(proxy_ledger.budget_exceeded()) is True
        assert asyncio.run(proxy_ledger.budget_exceeded()) is True
        assert len(sent) == 1 and "budget" in sent[0].lower()

    def test_zero_disables_the_cap(self, monkeypatch):
        self._prime(monkeypatch, used_mb=99999, budget_mb=0)
        assert asyncio.run(proxy_ledger.budget_exceeded()) is False


def test_probe_allow_paid_false_never_uses_the_paid_proxy(monkeypatch):
    monkeypatch.setenv("PROXY_URL", "http://paid")
    monkeypatch.setenv("STATIC_PROXY_URLS", "http://s1")
    monkeypatch.delenv("DIRECT_FIRST", raising=False)
    monkeypatch.setenv("BGUTIL_SCRIPT_PATH", "")
    monkeypatch.setenv("BGUTIL_BASE_URL", "")
    seen = []

    class _FakeYDL:
        def __init__(self, opts): seen.append(opts.get("proxy"))
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def extract_info(self, url, download=False):
            raise RuntimeError("Sign in to confirm you're not a bot")

    _patch_ydl(monkeypatch, _FakeYDL)
    monkeypatch.setattr(metering, "_ffprobe_url_seconds", lambda url, timeout=30: 0.0)
    with pytest.raises(ValueError):
        metering.probe_url_minutes("https://www.youtube.com/watch?v=x", allow_paid=False)
    assert "http://paid" not in seen


# --- the watcher's static probe --------------------------------------------

class TestStaticProbeAgainstYouTube:
    """google.com answered 204 through statics YouTube was refusing (28-aug):
    the static probe now has to see a playable watch page."""

    def _client(self, monkeypatch, status=200, body="", raise_exc=None):
        import cloud.alerts as alerts
        calls = {}

        class _Resp:
            status_code = status
            text = body

        class _Client:
            def __init__(self, **kw): calls["proxy"] = kw.get("proxy")
            async def __aenter__(self): return self
            async def __aexit__(self, *a): return False
            async def get(self, url, **kw):
                calls["url"] = url
                if raise_exc:
                    raise raise_exc
                return _Resp()

        import httpx
        monkeypatch.setattr(httpx, "AsyncClient", _Client)
        return alerts, calls

    def test_static_needs_the_playable_markers(self, monkeypatch):
        monkeypatch.setenv("PROXY_URL", "http://paid")
        alerts, calls = self._client(monkeypatch, body='{"playabilityStatus":{"status":"OK"}}')
        ok, _ = asyncio.run(alerts._probe_one("http://static1"))
        assert ok and "youtube.com/watch" in calls["url"]

    def test_static_answering_without_player_is_a_miss(self, monkeypatch):
        monkeypatch.setenv("PROXY_URL", "http://paid")
        alerts, _ = self._client(monkeypatch, body="<html>consent page</html>")
        ok, detail = asyncio.run(alerts._probe_one("http://static1"))
        assert not ok and "not playable" in detail

    def test_the_detail_names_what_youtube_said(self, monkeypatch):
        """"IP flagged" blamed the IP for every miss. LOGIN_REQUIRED with
        cookies configured is an expired session, not a banned address."""
        monkeypatch.setenv("PROXY_URL", "http://paid")
        alerts, _ = self._client(
            monkeypatch,
            body='{"playabilityStatus":{"status":"LOGIN_REQUIRED","reason":"x"}}')
        ok, detail = asyncio.run(alerts._probe_one("http://static1"))
        assert not ok and "LOGIN_REQUIRED" in detail

    def test_paid_probe_keeps_the_cheap_http_204(self, monkeypatch):
        monkeypatch.setenv("PROXY_URL", "http://paid")
        alerts, calls = self._client(monkeypatch, status=204, body="")
        ok, _ = asyncio.run(alerts._probe_one("http://paid"))
        assert ok and calls["url"].startswith("http://www.google.com")


def test_probe_uses_the_download_cookies(monkeypatch, tmp_path):
    """An anonymous probe gets "Sign in to confirm you're not a bot" from the
    static IPs (4-sep-2026: ~10 probes/day, 1.8 MB each on the per-GB proxy)
    while the cookie-authenticated download passes on those same IPs."""
    monkeypatch.setenv("YOUTUBE_COOKIES", "# Netscape HTTP Cookie File\n.youtube.com\tTRUE\t/\tTRUE\t0\tSID\tx\n")
    monkeypatch.setenv("STATIC_PROXY_URLS", "http://s1")
    monkeypatch.delenv("PROXY_URL", raising=False)
    monkeypatch.delenv("DIRECT_FIRST", raising=False)
    monkeypatch.setenv("BGUTIL_SCRIPT_PATH", "")
    monkeypatch.setenv("BGUTIL_BASE_URL", "")
    seen = {}

    class _FakeYDL:
        def __init__(self, opts):
            seen["cookiefile"] = opts.get("cookiefile")
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def extract_info(self, url, download=False):
            # The cookie file must exist while the extraction runs.
            assert seen["cookiefile"] and os.path.exists(seen["cookiefile"])
            with open(seen["cookiefile"]) as f:
                seen["body"] = f.read()
            return {"duration": 300}

    _patch_ydl(monkeypatch, _FakeYDL)
    assert metering.probe_url_minutes("https://www.youtube.com/watch?v=x") == 5.0
    assert "SID" in seen["body"]
    # and it is cleaned up afterwards
    assert not os.path.exists(seen["cookiefile"])


def test_probe_without_cookies_configured_still_works(monkeypatch):
    monkeypatch.delenv("YOUTUBE_COOKIES", raising=False)
    monkeypatch.delenv("STATIC_PROXY_URLS", raising=False)
    monkeypatch.delenv("PROXY_URL", raising=False)
    monkeypatch.setenv("BGUTIL_SCRIPT_PATH", "")
    monkeypatch.setenv("BGUTIL_BASE_URL", "")

    class _FakeYDL:
        def __init__(self, opts): assert "cookiefile" not in opts
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def extract_info(self, url, download=False): return {"duration": 120}

    _patch_ydl(monkeypatch, _FakeYDL)
    assert metering.probe_url_minutes("https://www.youtube.com/watch?v=x") == 2.0
