"""One YouTube account shared by every probe and download.

YouTube limits an account that asks for too many videos ("rate-limited by
YouTube for up to an hour"). Until 12-nov-2026 every submission then hit the
limited account up to eight times (two probes and the download, each on every
route) and the videos failed. Now the account is paced, nobody sends its
cookies during a cooldown, and a video that failed only because of the limit
goes back in line instead of failing.
"""
import asyncio
import os
import sys
import threading
import time
import types

import pytest

import yt_account

LIMITED = ("ERROR: [youtube] abc: Video unavailable. This content isn't available, try again "
           "later. Your account has been rate-limited by YouTube for up to an hour.")
URL = "https://www.youtube.com/watch?v=e_04ZrNroTo"


@pytest.fixture(autouse=True)
def state(tmp_path, monkeypatch):
    monkeypatch.setenv("YT_ACCOUNT_STATE_DIR", str(tmp_path))
    monkeypatch.setattr(yt_account, "COOLDOWN_SECONDS", 900.0)
    monkeypatch.setattr(yt_account, "STRIKES", 3)
    monkeypatch.setattr(yt_account, "STRIKE_WINDOW", 180.0)
    monkeypatch.setattr(yt_account, "MIN_INTERVAL", 0.0)
    monkeypatch.setattr(yt_account, "SLOTS", 2)
    monkeypatch.setattr(yt_account, "MAX_WAIT", 90.0)
    return tmp_path


def test_the_error_text_is_recognised():
    assert yt_account.is_rate_limit(LIMITED)
    assert not yt_account.is_rate_limit("ERROR: [youtube] abc: Video unavailable")
    assert not yt_account.is_rate_limit("Sign in to confirm you’re not a bot")
    assert not yt_account.is_rate_limit(None)


def _shut(now=None):
    """Enough strikes in a row to start the cooldown."""
    last = 0
    for i in range(yt_account.STRIKES):
        last = yt_account.note_rate_limited(now=None if now is None else now + i)
    return last


def test_a_stray_limit_answer_does_not_stop_the_account():
    """A limited answer among requests that work is routine: it was there for
    hours on 12-nov-2026 while every download still succeeded."""
    assert yt_account.note_rate_limited(now=1000.0) == 0
    assert yt_account.note_rate_limited(now=1001.0) == 0
    assert yt_account.cooling_down(now=1002.0) == 0
    yt_account.note_ok()                                   # one that worked
    assert yt_account.note_rate_limited(now=1003.0) == 0   # the count starts over
    assert yt_account.note_rate_limited(now=1004.0) == 0
    assert yt_account.cooling_down(now=1005.0) == 0
    # Strikes far apart are not a streak either.
    assert yt_account.note_rate_limited(now=1004.0 + 500) == 0
    assert yt_account.cooling_down(now=1600.0) == 0


def test_a_cooldown_starts_lasts_and_ends():
    assert yt_account.cooling_down() == 0
    assert _shut(now=1000.0) == 900
    assert yt_account.cooling_down(now=1003.0) == pytest.approx(899)
    assert yt_account.cooling_down(now=1002.0 + 901) == 0
    assert yt_account.note_rate_limited(now=2000.0) == 0   # a new streak starts from one
    # A stamp from a clock that jumped ahead is not a cooldown of days.
    _shut(now=10 ** 9)
    assert yt_account.cooling_down(now=1000.0) == 0


def test_the_cooldown_is_off_unless_configured(monkeypatch):
    """Without a route that works anonymously a cooldown stops every download,
    so it is opt-in: strikes are counted, the cookies keep going out."""
    monkeypatch.setattr(yt_account, "COOLDOWN_SECONDS", 0.0)
    assert _shut(now=1000.0) == 0
    assert yt_account.cooling_down(now=1001.0) == 0
    # A stamp left on disk by an earlier setting means nothing either.
    monkeypatch.setattr(yt_account, "COOLDOWN_SECONDS", 900.0)
    _shut(now=2000.0)
    monkeypatch.setattr(yt_account, "COOLDOWN_SECONDS", 0.0)
    assert yt_account.cooling_down(now=2003.0) == 0
    assert yt_account._float_env("YT_ACCOUNT_COOLDOWN_SECONDS_UNSET_NAME", 0) == 0


def test_no_state_directory_never_blocks_a_request(monkeypatch):
    monkeypatch.setenv("YT_ACCOUNT_STATE_DIR", "/nonexistent/dir")
    with yt_account.request():
        pass
    assert yt_account.cooling_down() == 0


def test_requests_take_a_slot_and_wait_for_one(monkeypatch):
    monkeypatch.setattr(yt_account, "SLOTS", 1)
    order = []

    def second():
        with yt_account.request():
            order.append("second in")

    with yt_account.request():
        t = threading.Thread(target=second)
        t.start()
        time.sleep(0.4)
        order.append("first still holds it")
    t.join(5)
    assert order == ["first still holds it", "second in"]


def test_a_slot_that_never_frees_does_not_stall_a_job(monkeypatch):
    monkeypatch.setattr(yt_account, "SLOTS", 1)
    monkeypatch.setattr(yt_account, "MAX_WAIT", 0.3)
    with yt_account.request():
        start = time.time()
        with yt_account.request():           # goes ahead after MAX_WAIT
            waited = time.time() - start
    assert 0.25 <= waited < 3


def test_requests_keep_their_distance(monkeypatch):
    monkeypatch.setattr(yt_account, "MIN_INTERVAL", 0.3)
    start = time.time()
    for _ in range(3):
        with yt_account.request():
            pass
    assert time.time() - start >= 0.55               # two gaps of 0.3 s


# --- the metering probe -------------------------------------------------------

metering = pytest.importorskip("cloud.metering")
NETSCAPE = "# Netscape HTTP Cookie File\n.youtube.com\tTRUE\t/\tTRUE\t0\tSID\tx\n"


def _fake_ytdl(monkeypatch, calls, limited=True):
    class _YDL:
        def __init__(self, opts):
            self.opts = opts

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def extract_info(self, url, download=False):
            authed = bool(self.opts.get("cookiefile"))
            calls.append({"proxy": self.opts.get("proxy"), "cookies": authed})
            if authed and limited:
                raise RuntimeError(LIMITED)
            return {"extractor": "youtube", "duration": 600,
                    "formats": [{"height": 1080, "vcodec": "avc1"}, {"height": 360, "vcodec": "avc1"},
                                {"vcodec": "none"}]}

    monkeypatch.setitem(sys.modules, "yt_dlp", types.SimpleNamespace(YoutubeDL=_YDL))


@pytest.fixture
def probe_env(monkeypatch):
    import security_utils
    monkeypatch.setattr(security_utils, "assert_public_url", lambda u: u)
    monkeypatch.setenv("STATIC_PROXY_URLS", "http://s1,http://s2,http://s3")
    monkeypatch.setenv("PROXY_URL", "http://paid")
    monkeypatch.setenv("YOUTUBE_COOKIES", NETSCAPE)
    monkeypatch.setenv("BGUTIL_BASE_URL", "http://bgutil")      # an HD (cookies) strategy exists
    monkeypatch.delenv("DIRECT_FIRST", raising=False)
    metering._probe_facts.clear()
    metering.pop_paid_probe_events()
    yield
    metering.pop_paid_probe_events()


def test_the_probe_stops_sending_the_cookies_once_the_account_is_limited(probe_env, monkeypatch):
    calls = []
    _fake_ytdl(monkeypatch, calls)
    assert metering.probe_url_minutes(URL) == 10.0
    # One request with the cookies, the answer that it is limited, then the
    # anonymous step of that route, which has the answer.
    assert [c["cookies"] for c in calls] == [True, False]
    assert yt_account.cooling_down() == 0                 # one answer is not a shut account

    metering.probe_url_minutes(URL)
    metering.probe_url_minutes(URL)                       # three in a row: shut
    assert yt_account.cooling_down() > 0
    calls.clear()
    assert metering.probe_url_minutes(URL) == 10.0
    assert [c["cookies"] for c in calls] == [False]       # during the cooldown: none at all


def test_the_next_route_is_still_asked_after_one_limit_answer(probe_env, monkeypatch):
    """The limit is often on one exit IP: the second route has the video."""
    calls = []

    class _YDL:
        def __init__(self, opts):
            self.opts = opts

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def extract_info(self, url, download=False):
            authed = bool(self.opts.get("cookiefile"))
            calls.append((self.opts.get("proxy"), authed))
            if authed and len([c for c in calls if c[1]]) == 1:
                raise RuntimeError(LIMITED)                 # the first route only
            if not authed:
                raise RuntimeError("Sign in to confirm you’re not a bot")
            return {"extractor": "youtube", "duration": 600}

    monkeypatch.setitem(sys.modules, "yt_dlp", types.SimpleNamespace(YoutubeDL=_YDL))
    assert metering.probe_url_minutes(URL) == 10.0
    authed = [c for c in calls if c[1]]
    assert len(authed) == 2 and authed[0][0] != authed[1][0]    # a second route, with the cookies


def test_a_limited_video_is_not_asked_on_every_route(probe_env, monkeypatch):
    """Anonymous fails too (bot check): the probe walks every route, and the
    cookies go out twice (ANSWERS_PER_VIDEO), not once per route."""
    calls = []

    class _YDL:
        def __init__(self, opts):
            self.opts = opts

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def extract_info(self, url, download=False):
            authed = bool(self.opts.get("cookiefile"))
            calls.append(authed)
            raise RuntimeError(LIMITED if authed else "Sign in to confirm you’re not a bot")

    monkeypatch.setitem(sys.modules, "yt_dlp", types.SimpleNamespace(YoutubeDL=_YDL))
    monkeypatch.setattr(metering, "_ffprobe_url_seconds", lambda *a, **k: 0)
    with pytest.raises(ValueError):
        metering.probe_url_minutes(URL)
    assert calls.count(True) == yt_account.ANSWERS_PER_VIDEO == 2 and calls.count(False) >= 3


def test_the_quality_gate_reuses_what_the_probe_just_learned(probe_env, monkeypatch):
    calls = []
    _fake_ytdl(monkeypatch, calls, limited=False)
    assert metering.probe_facts(URL) is None
    metering.probe_url_minutes(URL)
    facts = metering.probe_facts(URL)
    assert facts["max_height"] == 1080 and facts["duration"] == 600
    metering._probe_facts[URL]["at"] -= 3600
    assert metering.probe_facts(URL) is None              # too old to trust


# --- the job: wait for the account instead of failing ---------------------------

app_module = pytest.importorskip("app")


@pytest.fixture
def api(tmp_path, monkeypatch):
    m = app_module
    (tmp_path / "output").mkdir()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(m, "OUTPUT_DIR", "output")
    for name, value in (("_draining", False), ("jobs", {}), ("_running_jobs", set()),
                        ("_remote_jobs", set()), ("AUTO_RETRY_DELAY_SECONDS", 30.0),
                        ("job_queue", asyncio.PriorityQueue())):
        monkeypatch.setattr(m, name, value)
    return m


def _failed_job(m, job_id="j1"):
    d = os.path.join("output", job_id)
    os.makedirs(d, exist_ok=True)
    job = {"status": "failed", "cmd": ["python", "main.py", "-u", URL], "env": {},
           "output_dir": d, "logs": m._TimedLog(["❌ FATAL ERROR: YOUTUBE DOWNLOAD FAILED",
                                                  LIMITED])}
    m.jobs[job_id] = job
    return job


def test_the_marker_is_consumed_and_only_this_host_may_send_it(api):
    m = api
    job = _failed_job(m)
    m._handle_job_line("j1", "YT_ACCOUNT_LIMITED 840")
    assert job["yt_limited"] is True
    assert not any("YT_ACCOUNT_LIMITED" in l for l in job["logs"])
    other = _failed_job(m, "j2")
    m._handle_job_line("j2", "YT_ACCOUNT_LIMITED 840", remote=True)
    assert "yt_limited" not in other


def test_a_video_stopped_by_the_account_limit_goes_back_in_line(api, monkeypatch):
    m = api
    job = _failed_job(m)
    # The text alone reads as a broken video ("Video unavailable"): no retry.
    assert not m._should_auto_retry(job)
    job["yt_limited"] = True
    assert m._should_auto_retry(job)
    _shut()
    waits = []

    async def fake_sleep(s):
        waits.append(s)
        raise asyncio.CancelledError()

    async def run():
        monkeypatch.setattr(m.asyncio, "sleep", fake_sleep)
        m._schedule_auto_retry("j1", job)
        await asyncio.gather(*[t for t in asyncio.all_tasks()
                               if t is not asyncio.current_task()], return_exceptions=True)
    asyncio.run(run())
    assert job["status"] == "queued" and job["yt_retries"] == 1
    assert "auto_retries" not in job and "yt_limited" not in job      # its real retry is intact
    assert 900 <= waits[0] <= 900 + 150                               # after the cooldown
    assert any(l.startswith("⏳ YouTube is limiting downloads") for l in job["logs"])
    from log_view import friendly_logs
    assert any("YouTube is limiting downloads" in l for l in friendly_logs(job["logs"]))


def test_it_waits_a_few_times_then_fails_for_real(api):
    m = api
    job = _failed_job(m)
    job["yt_limited"] = True
    job["yt_retries"] = m.YT_LIMIT_RETRIES
    assert not m._should_auto_retry(job)
