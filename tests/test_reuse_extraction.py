"""The download uses what the probe just asked YouTube.

A submission asked YouTube twice for the same thing with the account's
cookies: the probe (how long is this video) and, minutes later, the download
(its formats and media links). Each is a request of an account whose budget
every download waits for. The probe's answer is now kept until its media
links expire, and the download starts from it: only on the exit that asked,
with the same client list, never on the per-GB proxy, and when it does not
work the download asks again as if nothing had been kept.
"""
import json
import os
import sys
import time
import types

import pytest

import source_cache
import yt_account

NETSCAPE = "# Netscape HTTP Cookie File\n.youtube.com\tTRUE\t/\tTRUE\t0\tSID\tx\n"
URL = "https://www.youtube.com/watch?v=tJbzMqJGH4k"
OTHER = "https://www.youtube.com/watch?v=e_04ZrNroTo"
STATICS = ["http://user:pw@s1", "http://user:pw@s2", "http://user:pw@s3"]


def _media(expires_in=6 * 3600):
    return (f"https://rr2---sn-x.googlevideo.com/videoplayback?expire={int(time.time() + expires_in)}"
            f"&ei=x&ip=11.0.0.1&id=o-A&itag=137")


def _info(expires_in=6 * 3600, **more):
    return {"_type": "video", "id": "tJbzMqJGH4k", "title": "GTA", "duration": 600,
            "extractor": "youtube", "extractor_key": "Youtube",
            "formats": [{"format_id": "137", "height": 1080, "vcodec": "avc1",
                         "url": _media(expires_in)},
                        {"format_id": "140", "vcodec": "none", "url": _media(expires_in)}],
            **more}


@pytest.fixture(autouse=True)
def env(tmp_path, monkeypatch):
    import security_utils
    monkeypatch.setattr(security_utils, "assert_public_url", lambda u: u)
    monkeypatch.setenv("SOURCE_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.delenv("SOURCE_CACHE_INFO", raising=False)
    monkeypatch.setenv("YT_ACCOUNT_STATE_DIR", str(tmp_path))
    monkeypatch.setattr(yt_account, "MIN_INTERVAL", 0.0)
    monkeypatch.setenv("YOUTUBE_COOKIES", NETSCAPE)
    monkeypatch.setenv("STATIC_PROXY_URLS", ",".join(STATICS))
    monkeypatch.setenv("PROXY_URL", "http://paid")
    monkeypatch.setenv("BGUTIL_BASE_URL", "http://bgutil")
    monkeypatch.delenv("DIRECT_FIRST", raising=False)
    return tmp_path


# --- what is kept -------------------------------------------------------------

def test_an_extraction_is_kept_until_its_links_expire():
    assert source_cache.info_put(URL, _info(), STATICS[1], True, {"youtube": {"c": ["mweb"]}})
    kept = source_cache.info_get(URL)
    assert kept["info"]["title"] == "GTA" and kept["authed"] is True
    assert kept["args"] == {"youtube": {"c": ["mweb"]}}
    assert kept["route"] == source_cache.route_key(STATICS[1])
    # Not once the links are about to stop working, and never another video's.
    soon = time.time() + 6 * 3600 - source_cache.INFO_MARGIN_SECONDS + 60
    assert source_cache.info_get(URL, now=soon) is None
    assert source_cache.info_get(OTHER) is None
    source_cache.info_drop(URL)
    assert source_cache.info_get(URL) is None


def test_a_download_that_takes_long_needs_links_that_live_longer():
    source_cache.info_put(URL, _info(expires_in=6 * 3600), None, True, {})
    # A ranged fetch runs at about the speed of the video: it needs that long
    # on top of the margin, or the source would come out short.
    assert source_cache.info_get(URL, need_seconds=3 * 3600) is not None
    assert source_cache.info_get(URL, need_seconds=4.5 * 3600) is None


@pytest.mark.parametrize("info", [
    _info(expires_in=300),                                    # links nearly dead already
    {"_type": "video", "formats": [{"url": "https://x.example/v.mp4"}]},   # no expiry to trust
    {"_type": "playlist", "entries": []},
    None,
])
def test_what_cannot_be_downloaded_from_later_is_not_kept(info):
    assert not source_cache.info_put(URL, info, None, True, {})
    assert source_cache.info_get(URL) is None


def test_the_route_is_named_without_the_proxys_credentials(env):
    source_cache.info_put(URL, _info(), STATICS[0], True, {})
    name = source_cache.route_key(STATICS[0])
    assert "pw" not in name and "s1" not in name and name != source_cache.route_key(STATICS[1])
    assert source_cache.route_key(None) == source_cache.route_key("") == "direct"
    stored = (env / "cache" / "tJbzMqJGH4k.info.json").read_text()
    assert "user:pw" not in stored


def test_it_can_be_switched_off_and_old_files_are_swept(env, monkeypatch):
    assert source_cache.info_put(URL, _info(), None, True, {})
    path = env / "cache" / "tJbzMqJGH4k.info.json"
    monkeypatch.setenv("SOURCE_CACHE_INFO", "0")
    assert source_cache.info_get(URL) is None
    assert not source_cache.info_put(OTHER, _info(), None, True, {})
    old = time.time() - source_cache.INFO_FILE_SECONDS - 60
    os.utime(path, (old, old))
    source_cache.prune()
    assert not path.exists()


# --- the probe leaves it ------------------------------------------------------

def _probe_ytdl(monkeypatch, calls, fail_free_routes=False):
    class _YDL:
        def __init__(self, opts):
            self.opts = opts

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        @staticmethod
        def sanitize_info(info, remove_private_keys=False):
            return json.loads(json.dumps({k: v for k, v in info.items() if k != "private"}))

        def extract_info(self, url, download=False):
            proxy = self.opts.get("proxy")
            calls.append({"proxy": proxy, "cookies": bool(self.opts.get("cookiefile"))})
            if fail_free_routes and proxy != "http://paid":
                raise RuntimeError("ERROR: Sign in to confirm you're not a bot")
            return _info(private=object())

    monkeypatch.setitem(sys.modules, "yt_dlp", types.SimpleNamespace(YoutubeDL=_YDL))


def test_the_probe_leaves_its_answer_for_the_download(monkeypatch):
    from cloud import metering
    metering._probe_facts.clear()
    calls = []
    _probe_ytdl(monkeypatch, calls)
    assert metering.probe_url_minutes(URL) == 10.0
    kept = source_cache.info_get(URL)
    assert kept and kept["authed"] is True
    assert kept["route"] == source_cache.route_key(calls[0]["proxy"])
    assert "private" not in kept["info"]                    # yt-dlp's sanitized result


def test_what_is_kept_is_what_yt_dlp_can_pick_formats_from_again():
    """With the real yt-dlp, offline: a kept extraction selects the formats a
    fresh one does, keeps the flag yt-dlp tests a doubtful format by, and
    carries no captions (most of its size, never read by a download)."""
    yt_dlp = pytest.importorskip("yt_dlp")
    from cloud import metering
    spec = ("bestvideo[vcodec^=avc1][height<=1080][ext=mp4]+bestaudio[ext=m4a]/"
            "best[height<=1080][ext=mp4]/best")

    def fresh():
        return {"id": "tJbzMqJGH4k", "title": "GTA", "duration": 600, "extractor": "youtube",
                "extractor_key": "Youtube", "webpage_url": URL,
                "automatic_captions": {"en": [{"url": "https://x.example/c", "ext": "vtt"}]},
                "formats": [
                    {"format_id": "18", "ext": "mp4", "height": 360, "width": 640, "vcodec": "avc1.42",
                     "acodec": "mp4a.40.2", "protocol": "https", "url": _media()},
                    {"format_id": "137", "ext": "mp4", "height": 1080, "width": 1920, "tbr": 4000,
                     "vcodec": "avc1.64", "acodec": "none", "protocol": "https", "url": _media()},
                    {"format_id": "140", "ext": "m4a", "vcodec": "none", "acodec": "mp4a.40.2",
                     "abr": 128, "protocol": "https", "url": _media()},
                    {"format_id": "270", "ext": "mp4", "height": 1080, "width": 1920, "tbr": 4500,
                     "vcodec": "vp09", "acodec": "none", "protocol": "m3u8_native",
                     "url": _media(), "__needs_testing": True}]}

    def picked(info):
        opts = {"format": spec, "simulate": True, "quiet": True, "no_warnings": True}
        with yt_dlp.YoutubeDL(opts) as ydl:
            return ydl.process_ie_result(info, download=True)["format_id"]

    metering._keep_extraction(URL, fresh(), STATICS[0], True, {"youtube": {"c": ["mweb"]}})
    kept = source_cache.info_get(URL)["info"]
    assert picked(kept) == picked(fresh()) == "137+140"
    assert [f.get("__needs_testing") for f in kept["formats"]] == [None, None, None, True]
    assert "automatic_captions" not in kept


def test_nothing_is_kept_from_the_per_gb_proxy(monkeypatch):
    from cloud import metering
    metering._probe_facts.clear()
    metering.pop_paid_probe_events()
    calls = []
    _probe_ytdl(monkeypatch, calls, fail_free_routes=True)
    assert metering.probe_url_minutes(URL) == 10.0
    assert calls[-1]["proxy"] == "http://paid"
    assert source_cache.info_get(URL) is None
    metering.pop_paid_probe_events()


# --- the download starts from it ------------------------------------------------

def _download(monkeypatch, reused_works=True):
    main = pytest.importorskip("main")
    asked, downloaded = [], []

    class _YDL:
        def __init__(self, opts):
            self.opts = opts

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def extract_info(self, url, download=False, process=True):
            asked.append({"proxy": self.opts.get("proxy"),
                          "cookies": bool(self.opts.get("cookiefile"))})
            return {"title": "GTA", "_type": "video", "formats": [{"format_id": "1"}],
                    "duration": 600, "extractor_key": "Youtube"}

        def process_ie_result(self, info, download=True):
            kept = bool(info.get("kept"))
            downloaded.append({"proxy": self.opts.get("proxy"), "kept": kept,
                               "cookies": bool(self.opts.get("cookiefile"))})
            if kept and not reused_works:
                raise RuntimeError("HTTP Error 403: Forbidden")
            with open(self.opts["outtmpl"].replace("%(ext)s", "mp4"), "wb") as f:
                f.write(b"v")
            return info

    monkeypatch.setattr(main, "yt_dlp", types.SimpleNamespace(
        YoutubeDL=_YDL, version=types.SimpleNamespace(__version__="test"),
        utils=sys.modules.get("yt_dlp.utils")))
    return main, asked, downloaded


def _hd_args():
    from yt_clients import hd_extractor_args
    return hd_extractor_args("http://bgutil", "")


def test_the_download_asks_youtube_nothing_when_the_probe_just_did(monkeypatch, tmp_path, capsys):
    main, asked, downloaded = _download(monkeypatch)
    assert source_cache.info_put(URL, _info(kept=True), STATICS[2], True, _hd_args())
    route = {}
    path, title = main._download_youtube_video(URL, str(tmp_path), route_out=route)
    assert os.path.exists(path) and title == "GTA"
    assert asked == []                                       # no request of the account
    # Through the exit that asked, and without the account's cookie file.
    assert downloaded == [{"proxy": STATICS[2], "kept": True, "cookies": False}]
    assert route["winner"].startswith("HD-static") and route["paid_bytes"] == 0
    assert route["attempts"][-1]["reused"] is True
    assert "extraction reused" in capsys.readouterr().out


def test_a_kept_extraction_that_does_not_download_is_dropped_and_asked_again(monkeypatch, tmp_path):
    main, asked, downloaded = _download(monkeypatch, reused_works=False)
    source_cache.info_put(URL, _info(kept=True), STATICS[0], True, _hd_args())
    route = {}
    path, _ = main._download_youtube_video(URL, str(tmp_path), route_out=route)
    assert os.path.exists(path)
    assert downloaded[0]["kept"] and not downloaded[-1]["kept"]
    assert len(asked) == 1 and asked[0]["cookies"]           # the usual first attempt
    assert [a["ok"] for a in route["attempts"]] == [False, True]
    assert source_cache.info_get(URL) is None                 # the next job does not try it


def test_a_long_ranged_fetch_does_not_start_on_links_that_die_before_it_ends(monkeypatch, tmp_path):
    main, asked, downloaded = _download(monkeypatch)
    source_cache.info_put(URL, _info(kept=True, duration=40000), STATICS[0], True, _hd_args())
    # 300 paid minutes of a much longer video: five hours of fetching, on
    # links with six hours left.
    main._download_youtube_video(URL, str(tmp_path), cap_minutes=300)
    assert len(asked) == 1 and not downloaded[0]["kept"]
    # A short cap fits: the same kept extraction is used.
    source_cache.info_put(URL, _info(kept=True, duration=40000), STATICS[0], True, _hd_args())
    asked.clear(), downloaded.clear()
    (tmp_path / "b").mkdir()
    main._download_youtube_video(URL, str(tmp_path / "b"), cap_minutes=20)
    assert asked == [] and downloaded[0]["kept"]


@pytest.mark.parametrize("proxy,args", [
    ("http://paid", None),                      # the per-GB proxy: its exit changes
    ("http://user:pw@elsewhere", None),        # an exit this download does not have
    (STATICS[1], {"youtube": {"player_client": ["something-else"]}}),   # another client list
])
def test_only_the_same_exit_and_client_list_may_use_it(monkeypatch, tmp_path, proxy, args):
    main, asked, downloaded = _download(monkeypatch)
    source_cache.info_put(URL, _info(kept=True), proxy, True, args or _hd_args())
    main._download_youtube_video(URL, str(tmp_path))
    assert len(asked) == 1 and not downloaded[0]["kept"]
