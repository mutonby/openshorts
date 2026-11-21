"""Each YouTube account leaves through its own exit, and a download waits
when every account rests.

The exit of a request used to be drawn at random, so one signed-in session
showed up from every address of the pool within seconds, and the accounts
were limited far below the request rate YouTube lets one through. And with
every account resting, each download asked the one closest to waking, which
answered "rate-limited" again and stayed limited for as long as there were
downloads to try.
"""
import asyncio
import os
import sys
import time
import types

import pytest

import yt_account

NETSCAPE = "# Netscape HTTP Cookie File\n.youtube.com\tTRUE\t/\tTRUE\t0\tSID\tx\n"
URL = "https://www.youtube.com/watch?v=tJbzMqJGH4k"
STATICS = ["http://s1", "http://s2", "http://s3"]


@pytest.fixture(autouse=True)
def env(tmp_path, monkeypatch):
    import security_utils
    monkeypatch.setattr(security_utils, "assert_public_url", lambda u: u)
    monkeypatch.setenv("YT_ACCOUNT_STATE_DIR", str(tmp_path))
    monkeypatch.setattr(yt_account, "MIN_INTERVAL", 0.0)
    monkeypatch.setattr(yt_account, "REST_SECONDS", 900.0)
    monkeypatch.setattr(yt_account, "REST_STRIKES", 2)
    monkeypatch.setattr(yt_account, "REST_WAIT", 600.0)
    for i, name in enumerate(["YOUTUBE_COOKIES"] + [f"YOUTUBE_COOKIES_{n}" for n in range(2, 6)]):
        monkeypatch.setenv(name, NETSCAPE.replace("SID\t", f"SID\tacct{i}-"))
    monkeypatch.setenv("STATIC_PROXY_URLS", ",".join(STATICS))
    monkeypatch.setenv("PROXY_URL", "http://paid")
    monkeypatch.setenv("BGUTIL_BASE_URL", "http://bgutil")
    monkeypatch.delenv("DIRECT_FIRST", raising=False)
    monkeypatch.delenv("SOURCE_CACHE_DIR", raising=False)
    return tmp_path


def _rest_everyone(n=5):
    now = time.time()
    for a in range(n):
        yt_account.note_account_limited(a, now=now)
        yt_account.note_account_limited(a, now=now)


# --- one account, one exit ----------------------------------------------------

def test_an_account_always_leaves_through_the_same_exit():
    homes = [yt_account.home_exit(a, 3) for a in range(5)]
    assert homes == [0, 1, 2, 0, 1]                 # more accounts than exits: they share
    assert homes == [yt_account.home_exit(a, 3) for a in range(5)]
    assert yt_account.home_exit(4, 0) == 0 and yt_account.home_exit(None, 3) == 0


def _which_account(opts):
    ck = opts.get("cookiefile")
    return int(open(ck).read().split("acct")[1][0]) if ck else None


def test_the_probe_asks_through_its_accounts_own_exit(monkeypatch):
    from cloud import metering
    metering._probe_facts.clear()
    seen = []

    class _YDL:
        def __init__(self, opts):
            self.opts = opts

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def extract_info(self, url, download=False):
            seen.append((_which_account(self.opts), self.opts.get("proxy")))
            return {"extractor": "youtube", "duration": 600,
                    "formats": [{"height": 1080, "vcodec": "avc1"}]}

    monkeypatch.setitem(sys.modules, "yt_dlp", types.SimpleNamespace(YoutubeDL=_YDL))
    for _ in range(10):
        metering.probe_url_minutes(URL)
    assert {a for a, _ in seen} == {0, 1, 2, 3, 4}          # the accounts still take turns
    for account, proxy in seen:
        assert proxy == STATICS[account % 3]                 # each through its own exit


def _download(monkeypatch):
    main = pytest.importorskip("main")
    seen = []

    class _YDL:
        def __init__(self, opts):
            self.opts = opts

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def extract_info(self, url, download=False, process=True):
            seen.append((_which_account(self.opts), self.opts.get("proxy")))
            return {"title": "GTA", "_type": "video", "formats": [{"format_id": "1"}],
                    "duration": 600, "extractor_key": "Youtube"}

        def process_ie_result(self, info, download=True):
            with open(self.opts["outtmpl"].replace("%(ext)s", "mp4"), "wb") as f:
                f.write(b"v")
            return info

    monkeypatch.setattr(main, "yt_dlp", types.SimpleNamespace(
        YoutubeDL=_YDL, version=types.SimpleNamespace(__version__="test"),
        utils=sys.modules.get("yt_dlp.utils")))
    return main, seen


def test_the_download_asks_through_its_accounts_own_exit(monkeypatch, tmp_path):
    main, seen = _download(monkeypatch)
    for i in range(10):
        d = tmp_path / f"job{i}"
        d.mkdir()
        main._download_youtube_video(URL, str(d))
    assert len(seen) == 10 and {a for a, _ in seen} == {0, 1, 2, 3, 4}
    for account, proxy in seen:
        assert proxy == STATICS[account % 3]


# --- every account resting -------------------------------------------------------

def test_all_resting_says_when_the_first_account_wakes():
    assert yt_account.all_resting() == 0                     # nobody rests
    now = time.time()
    for a in range(4):
        yt_account.note_account_limited(a, now=now)
        yt_account.note_account_limited(a, now=now)
    assert yt_account.all_resting() == 0                     # one is still awake
    yt_account.note_account_limited(4, now=now + 60)
    yt_account.note_account_limited(4, now=now + 60)
    assert 880 < yt_account.all_resting() <= 900             # until the first four wake
    assert yt_account.all_resting(now=now + 901) == 0        # four awake: nobody waits


def test_one_account_never_makes_anybody_wait(monkeypatch):
    for n in range(2, 6):
        monkeypatch.delenv(f"YOUTUBE_COOKIES_{n}")
    assert yt_account.all_resting() == 0
    assert yt_account.wait_for_awake(sleep=lambda s: pytest.fail("waited")) == 0


def test_a_download_waits_for_an_account_and_not_for_ever():
    slept, said = [], []
    assert yt_account.wait_for_awake(sleep=slept.append, say=said.append) == 0
    _rest_everyone()
    waited = yt_account.wait_for_awake(max_wait=40, sleep=slept.append, say=said.append)
    assert waited == 40 and slept == [15, 15, 10]            # in steps, up to the limit
    assert len(said) == 1 and 880 < said[0] <= 900           # told once, with what is left
    # 0 turns the wait off: the download goes ahead with a resting account, as before.
    assert yt_account.wait_for_awake(max_wait=0, sleep=lambda s: pytest.fail("waited")) == 0


def test_it_stops_waiting_as_soon_as_one_account_wakes(monkeypatch):
    _rest_everyone()
    clock = {"t": time.time()}
    real = yt_account.resting
    monkeypatch.setattr(yt_account, "resting",
                        lambda account, now=None: real(account, now or clock["t"]))

    def sleep(s):
        clock["t"] += 450
    assert yt_account.wait_for_awake(sleep=sleep) == 30      # two steps, then one is awake
    assert yt_account.all_resting(now=clock["t"]) == 0


def test_only_a_jobs_download_waits(monkeypatch, tmp_path):
    main, seen = _download(monkeypatch)
    _rest_everyone()
    waits = []
    monkeypatch.setattr(yt_account, "wait_for_awake", lambda **k: waits.append(k) or 0)
    main._download_youtube_video(URL, str(tmp_path))          # inside the API: somebody waits on it
    assert waits == []
    (tmp_path / "job").mkdir()
    main.download_youtube_video(URL, str(tmp_path / "job"), use_cache=True)
    assert len(waits) == 1


def test_a_limited_video_goes_back_in_line_for_when_an_account_wakes(monkeypatch, tmp_path):
    app = pytest.importorskip("app")
    (tmp_path / "output").mkdir()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(app, "OUTPUT_DIR", "output")
    for name, value in (("_draining", False), ("jobs", {}), ("_running_jobs", set()),
                        ("_remote_jobs", set()), ("AUTO_RETRY_DELAY_SECONDS", 30.0),
                        ("job_queue", asyncio.PriorityQueue())):
        monkeypatch.setattr(app, name, value)
    os.makedirs("output/j1")
    job = {"status": "failed", "cmd": ["python", "main.py", "-u", URL], "env": {},
           "output_dir": "output/j1", "logs": app._TimedLog([]), "yt_limited": True}
    app.jobs["j1"] = job
    _rest_everyone()
    waits = []

    async def fake_sleep(s):
        waits.append(s)
        raise asyncio.CancelledError()

    async def run():
        monkeypatch.setattr(app.asyncio, "sleep", fake_sleep)
        app._schedule_auto_retry("j1", job)
        await asyncio.gather(*[t for t in asyncio.all_tasks()
                               if t is not asyncio.current_task()], return_exceptions=True)
    asyncio.run(run())
    assert 880 <= waits[0] <= 900 + 150                      # not half a minute later
