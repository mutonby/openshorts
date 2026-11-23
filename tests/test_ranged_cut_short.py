"""A ranged download that drops mid-stream is noticed, retried once, and kept.

yt-dlp hands a ranged fetch to ffmpeg, one long HTTP read per stream. When the
connection dropped, ffmpeg logged "Error during demuxing: Input/output error",
still exited 0, and the attempt "succeeded" with the first two or three
minutes of a 20-minute range: clips only from the start, billed in full.
Now the file's length is checked against the range, a short one is fetched
again on the same route, and a second short one is kept as it is (never
walking the other routes or the per-GB proxy for a range). ffmpeg is not told
to reconnect: a stream the site keeps ending then crawls for the best part of
an hour, with the job and the worker waiting for its source held meanwhile.
"""
import os
import sys
import types

import pytest

import yt_account

NETSCAPE = "# Netscape HTTP Cookie File\n.youtube.com\tTRUE\t/\tTRUE\t0\tSID\tx\n"
URL = "https://www.youtube.com/watch?v=tJbzMqJGH4k"
STATICS = ["http://user:pw@s1", "http://user:pw@s2"]


@pytest.fixture(autouse=True)
def env(tmp_path, monkeypatch):
    import security_utils
    monkeypatch.setattr(security_utils, "assert_public_url", lambda u: u)
    monkeypatch.setenv("SOURCE_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("SOURCE_CACHE_INFO", "0")
    monkeypatch.setenv("YT_ACCOUNT_STATE_DIR", str(tmp_path))
    monkeypatch.setattr(yt_account, "MIN_INTERVAL", 0.0)
    monkeypatch.setenv("YOUTUBE_COOKIES", NETSCAPE)
    monkeypatch.setenv("STATIC_PROXY_URLS", ",".join(STATICS))
    monkeypatch.setenv("PROXY_URL", "http://paid")
    monkeypatch.setenv("BGUTIL_BASE_URL", "http://bgutil")
    monkeypatch.delenv("DIRECT_FIRST", raising=False)
    monkeypatch.delenv("SOURCE_CAP_MINUTES", raising=False)
    monkeypatch.delenv("MAX_SOURCE_MINUTES", raising=False)


def _download(monkeypatch, lengths, duration=40000):
    """Fake yt-dlp; ``lengths`` are the seconds each fetched file lasts."""
    main = pytest.importorskip("main")
    monkeypatch.setattr(main.time, "sleep", lambda s: None)
    downloaded = []

    class _YDL:
        def __init__(self, opts):
            self.opts = opts

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def extract_info(self, url, download=False, process=True):
            return {"title": "GTA", "_type": "video", "formats": [{"format_id": "1"}],
                    "duration": duration, "extractor_key": "Youtube"}

        def process_ie_result(self, info, download=True):
            downloaded.append({"proxy": self.opts.get("proxy"),
                               "ranged": "download_ranges" in self.opts,
                               "ffmpeg_i": (self.opts.get("external_downloader_args")
                                            or {}).get("ffmpeg_i")})
            with open(self.opts["outtmpl"].replace("%(ext)s", "mp4"), "wb") as f:
                f.write(b"v")
            return info

    seconds = iter(lengths)
    monkeypatch.setattr(main, "_media_seconds", lambda path: next(seconds))
    monkeypatch.setattr(main, "yt_dlp", types.SimpleNamespace(
        YoutubeDL=_YDL, version=types.SimpleNamespace(__version__="test"),
        utils=sys.modules.get("yt_dlp.utils")))
    return main, downloaded


def test_short_is_only_clearly_short():
    ranged_cut_short = pytest.importorskip("main").ranged_cut_short
    assert ranged_cut_short(160, 1260)
    assert not ranged_cut_short(1255, 1260)          # keyframe rounding
    assert not ranged_cut_short(1200, 1260)          # the paid part is all there
    assert not ranged_cut_short(None, 1260)          # unreadable: never called short
    assert not ranged_cut_short(160, 0)              # unknown length asked


def test_a_ranged_fetch_is_never_told_to_reconnect(monkeypatch, tmp_path):
    """A dropped stream ends the fetch, and the check below sees it: resumed,
    it comes in far below playback speed and holds the job for as long."""
    main, downloaded = _download(monkeypatch, [1260])
    main._download_youtube_video(URL, str(tmp_path), cap_minutes=20)
    assert len(downloaded) == 1 and downloaded[0]["ranged"]
    assert downloaded[0]["ffmpeg_i"] is None
    assert not hasattr(main, "RANGED_FFMPEG_INPUT_ARGS")


def test_a_whole_download_is_left_alone(monkeypatch, tmp_path):
    main, downloaded = _download(monkeypatch, [], duration=600)
    main._download_youtube_video(URL, str(tmp_path), cap_minutes=20)
    assert downloaded == [{"proxy": STATICS[0], "ranged": False, "ffmpeg_i": None}]


def test_a_short_range_is_fetched_again_on_the_same_route(monkeypatch, tmp_path):
    main, downloaded = _download(monkeypatch, [160, 1260])
    route = {}
    path, _ = main._download_youtube_video(URL, str(tmp_path), cap_minutes=20, route_out=route)
    assert os.path.exists(path)
    assert [d["proxy"] for d in downloaded] == [STATICS[0], STATICS[0]]
    assert [a["ok"] for a in route["attempts"]] == [False, True]
    assert "cut short" in route["attempts"][0]["error"]


def test_short_twice_keeps_what_came_and_never_reaches_the_paid_proxy(monkeypatch, tmp_path):
    main, downloaded = _download(monkeypatch, [160, 170])
    route = {}
    path, _ = main._download_youtube_video(URL, str(tmp_path), cap_minutes=20, route_out=route)
    assert os.path.exists(path)
    assert [d["proxy"] for d in downloaded] == [STATICS[0], STATICS[0]]
    assert route["attempts"][-1]["ok"] and route["attempts"][-1]["short"]
    assert route["paid_bytes"] == 0 and "http://paid" not in [d["proxy"] for d in downloaded]
