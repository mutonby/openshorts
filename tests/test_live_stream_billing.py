"""A live stream still on air is refused before billing, and a finished job
never costs more than the source it really downloaded.

A YouTube live in progress reports the time elapsed so far as its duration;
the download only gets the last seconds at the live edge. Billing the probe's
number charged a whole balance for a two-second source and a one-clip job.
"""
import asyncio
import sys
import types
from decimal import Decimal

import pytest

metering = pytest.importorskip("cloud.metering")
from yt_clients import LiveStreamOnAir, NotASingleVideo, live_not_ready  # noqa: E402

URL = "https://www.youtube.com/watch?v=abcdefghijk"


def _fake_ytdl(monkeypatch, attempts, info):
    class _YDL:
        def __init__(self, opts):
            self.opts = opts

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def extract_info(self, url, download=False):
            attempts.append(dict(self.opts))
            return dict(info)

    monkeypatch.setitem(sys.modules, "yt_dlp", types.SimpleNamespace(YoutubeDL=_YDL))


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    import security_utils
    import source_cache
    monkeypatch.setattr(security_utils, "assert_public_url", lambda u: u)
    monkeypatch.setattr(source_cache, "probe_get", lambda url: None)
    monkeypatch.setattr(source_cache, "probe_put", lambda *a, **k: None)
    monkeypatch.setenv("STATIC_PROXY_URLS", "http://s1,http://s2")
    monkeypatch.setenv("PROXY_URL", "http://paid")
    monkeypatch.delenv("YOUTUBE_COOKIES", raising=False)
    monkeypatch.delenv("DIRECT_FIRST", raising=False)
    monkeypatch.setattr(metering, "_ffprobe_url_seconds", lambda url, timeout=30: 0.0)
    metering.pop_paid_probe_events()
    yield
    metering.pop_paid_probe_events()


class TestLiveStatus:
    @pytest.mark.parametrize("info", [
        {"live_status": "is_live"}, {"live_status": "is_upcoming"},
        {"live_status": "post_live"}, {"is_live": True},
    ])
    def test_not_ready(self, info):
        assert live_not_ready(info)

    @pytest.mark.parametrize("info", [
        {"live_status": "was_live"}, {"live_status": "not_live"}, {}, None,
    ])
    def test_ready(self, info):
        assert not live_not_ready(info)

    def test_is_a_link_error_with_its_own_message(self):
        e = LiveStreamOnAir()
        assert isinstance(e, NotASingleVideo)
        assert "still on air" in str(e)


class TestProbeRefusesALiveOnAir:
    def test_raises_once_without_trying_other_routes(self, monkeypatch):
        attempts = []
        _fake_ytdl(monkeypatch, attempts, {"extractor": "youtube", "duration": 2885,
                                           "live_status": "is_live"})
        with pytest.raises(LiveStreamOnAir):
            metering.probe_url_minutes(URL)
        assert len(attempts) == 1
        assert metering.pop_paid_probe_events() == []

    def test_a_finished_live_is_billed_normally(self, monkeypatch):
        attempts = []
        _fake_ytdl(monkeypatch, attempts, {"extractor": "youtube", "duration": 600,
                                           "live_status": "was_live"})
        assert abs(metering.probe_url_minutes(URL) - 10) < 1e-6

    def test_the_probe_process_keeps_the_kind(self):
        out = {"error": LiveStreamOnAir.MESSAGE, "single": True, "live": True}
        with pytest.raises(LiveStreamOnAir):
            metering._merge_child_probe(URL, out)


# --- trim_reservation -------------------------------------------------------
class _Row:
    def __init__(self, **kw):
        self.__dict__.update(kw)


class _Session:
    def __init__(self, objs):
        self.objs = objs

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    def begin(self):
        return self

    async def get(self, model, key):
        return self.objs.get(str(key))

    async def execute(self, *a, **k):
        return None


def _patch_db(monkeypatch, objs):
    monkeypatch.setattr(metering.database, "session", lambda: _Session(objs))


def _run(coro):
    return asyncio.new_event_loop().run_until_complete(coro)


class TestTrimReservation:
    def test_gives_back_bought_minutes_first_then_plan(self, monkeypatch):
        t1 = _Row(id="t1", minutes_consumed=Decimal(30), minutes_total=Decimal(30))
        t2 = _Row(id="t2", minutes_consumed=Decimal(10), minutes_total=Decimal(50))
        row = _Row(id="r", user_id="u", status="reserved", minutes=Decimal(48),
                   minutes_from_plan=Decimal(8), minutes_from_topup=Decimal(40),
                   topup_allocations=[{"topup_id": "t1", "minutes": 30.0},
                                      {"topup_id": "t2", "minutes": 10.0}])
        _patch_db(monkeypatch, {"r": row, "t1": t1, "t2": t2})

        assert _run(metering.trim_reservation("r", 1)) == 47.0
        assert row.minutes == 1
        assert row.minutes_from_topup == 0 and row.minutes_from_plan == 1
        assert t1.minutes_consumed == 0 and t2.minutes_consumed == 0
        assert row.topup_allocations is None

    def test_partial_topup_refund_keeps_the_oldest_allocation(self, monkeypatch):
        t1 = _Row(id="t1", minutes_consumed=Decimal(20), minutes_total=Decimal(20))
        t2 = _Row(id="t2", minutes_consumed=Decimal(10), minutes_total=Decimal(50))
        row = _Row(id="r", user_id="u", status="reserved", minutes=Decimal(30),
                   minutes_from_plan=Decimal(0), minutes_from_topup=Decimal(30),
                   topup_allocations=[{"topup_id": "t1", "minutes": 20.0},
                                      {"topup_id": "t2", "minutes": 10.0}])
        _patch_db(monkeypatch, {"r": row, "t1": t1, "t2": t2})

        assert _run(metering.trim_reservation("r", 15)) == 15.0
        assert t2.minutes_consumed == 0 and t1.minutes_consumed == 15
        assert row.topup_allocations == [{"topup_id": "t1", "minutes": 15.0}]
        assert row.minutes_from_topup == 15 and row.minutes == 15

    @pytest.mark.parametrize("status,new", [("committed", 1), ("released", 1),
                                            ("reserved", 48), ("reserved", 60)])
    def test_never_raises_nor_touches_settled_rows(self, monkeypatch, status, new):
        row = _Row(id="r", user_id="u", status=status, minutes=Decimal(48),
                   minutes_from_plan=Decimal(48), minutes_from_topup=Decimal(0),
                   topup_allocations=None)
        _patch_db(monkeypatch, {"r": row})
        assert _run(metering.trim_reservation("r", new)) == 0.0
        assert row.minutes == 48


# --- the settle step --------------------------------------------------------
class TestChargeForTheDownloadedSource:
    @pytest.fixture
    def app_mod(self, monkeypatch):
        app = pytest.importorskip("app")
        from cloud import config as cloud_config
        # Tests import the app with billing off: lend it the cloud modules.
        monkeypatch.setattr(app, "cloud", types.SimpleNamespace(metering=metering))
        monkeypatch.setattr(app, "_cloud_config", cloud_config)
        monkeypatch.setattr(app, "_locate_source", lambda job_id: "/x/source.mp4")
        return app

    def test_two_seconds_of_a_48_minute_reservation_costs_one(self, app_mod, monkeypatch):
        monkeypatch.setattr(app_mod.cloud.metering, "probe_file_minutes", lambda p: 2 / 60)
        assert app_mod._billable_for_downloaded_source("j", 48) == 1

    @pytest.mark.parametrize("actual", [47.5, 48.0, 49.9, 120.0])
    def test_rounding_and_longer_sources_change_nothing(self, app_mod, monkeypatch, actual):
        monkeypatch.setattr(app_mod.cloud.metering, "probe_file_minutes", lambda p: actual)
        assert app_mod._billable_for_downloaded_source("j", 48) is None

    def test_no_source_or_unreadable_changes_nothing(self, app_mod, monkeypatch):
        def boom(p):
            raise RuntimeError("ffprobe")
        monkeypatch.setattr(app_mod.cloud.metering, "probe_file_minutes", boom)
        assert app_mod._billable_for_downloaded_source("j", 48) is None
        monkeypatch.setattr(app_mod, "_locate_source", lambda job_id: None)
        assert app_mod._billable_for_downloaded_source("j", 48) is None
