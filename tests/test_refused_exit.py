"""A site an exit refuses is asked through the next one, and nobody is paged
about credits for it.

A proxy provider that does not carry a site answers the CONNECT itself with a
403. The download of a non-YouTube link tried the server's own address and
then one static route only, so when that route was one that refuses the site
the job failed with a proxy error, and the alert read any proxy error as "may
be out of credits".
"""
import sys
import types

import pytest

import yt_account

REFUSED = ("ERROR: [twitch:vod] 1: Unable to download JSON metadata: ('Unable to connect to "
           "proxy', OSError('Tunnel connection failed: 403 Forbidden')) (caused by ProxyError)")
UNREACHABLE = ("ERROR: [twitch:vod] 1: Failed to download m3u8 information: HTTPSConnection("
               "host='x.example', port=443): Failed to establish a new connection: "
               "[Errno 101] Network is unreachable")
URL = "https://www.twitch.tv/videos/1"
STATICS = ["http://s1", "http://s2", "http://s3"]


@pytest.fixture(autouse=True)
def env(tmp_path, monkeypatch):
    import security_utils
    monkeypatch.setattr(security_utils, "assert_public_url", lambda u: u)
    monkeypatch.setenv("YT_ACCOUNT_STATE_DIR", str(tmp_path))
    monkeypatch.setattr(yt_account, "MIN_INTERVAL", 0.0)
    monkeypatch.setenv("STATIC_PROXY_URLS", ",".join(STATICS))
    monkeypatch.setenv("PROXY_URL", "http://paid")
    monkeypatch.setenv("BGUTIL_BASE_URL", "http://bgutil")
    monkeypatch.delenv("DIRECT_FIRST", raising=False)
    monkeypatch.delenv("SOURCE_CACHE_DIR", raising=False)
    for name in ["YOUTUBE_COOKIES"] + [f"YOUTUBE_COOKIES_{n}" for n in range(2, 10)]:
        monkeypatch.delenv(name, raising=False)
    return tmp_path


def _download(monkeypatch, answers):
    """``answers``: proxy (None for direct) -> list of results, one per call:
    an error text raises it, anything else downloads."""
    main = pytest.importorskip("main")
    seen, slept = [], []

    class _YDL:
        def __init__(self, opts):
            self.opts = opts

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def extract_info(self, url, download=False, process=True):
            proxy = self.opts.get("proxy")
            seen.append(proxy)
            answer = answers[proxy].pop(0)
            if isinstance(answer, str):
                raise RuntimeError(answer)
            return {"title": "VOD", "_type": "video", "formats": [{"format_id": "1"}],
                    "duration": 600, "extractor_key": "TwitchVod"}

        def process_ie_result(self, info, download=True):
            with open(self.opts["outtmpl"].replace("%(ext)s", "mp4"), "wb") as f:
                f.write(b"v")
            return info

    monkeypatch.setattr(main, "yt_dlp", types.SimpleNamespace(
        YoutubeDL=_YDL, version=types.SimpleNamespace(__version__="test"),
        utils=sys.modules.get("yt_dlp.utils")))
    monkeypatch.setattr(main.time, "sleep", slept.append)
    return main, seen, slept


def test_the_two_failures_are_told_apart():
    main = pytest.importorskip("main")
    assert main.exit_refused(REFUSED) and not main.exit_refused(UNREACHABLE)
    assert main.transient_connection(UNREACHABLE) and not main.transient_connection(REFUSED)
    assert not main.exit_refused("HTTP Error 403: Forbidden")      # the site said no, not the exit
    assert not main.exit_refused(None) and not main.transient_connection("")


def test_a_refused_tunnel_moves_on_to_the_next_exit(monkeypatch, tmp_path):
    main, seen, slept = _download(monkeypatch, {
        None: ["HTTP Error 500"], "http://s1": [REFUSED], "http://s2": [REFUSED],
        "http://s3": [{}]})
    main._download_youtube_video(URL, str(tmp_path))
    # The statics come in no fixed order. One request per refusing exit (asking
    # it again gets the same answer), until the one that opens the tunnel;
    # never the per-GB proxy.
    assert seen[0] is None and seen[-1] == "http://s3" and slept == []
    assert set(seen[1:-1]) <= {"http://s1", "http://s2"} and len(set(seen)) == len(seen)


def test_an_exit_that_reached_the_site_is_the_last_one_asked(monkeypatch, tmp_path):
    gone = "ERROR: [twitch:vod] 1: The video does not exist"
    main, seen, _slept = _download(monkeypatch, {
        None: ["HTTP Error 500"], "http://s1": [REFUSED], "http://s2": [gone],
        "http://s3": [gone]})
    with pytest.raises(Exception):
        main._download_youtube_video(URL, str(tmp_path))
    # It stops at the first exit that reached the site: the other one that
    # would have reached it too, and the paid proxy, are never asked.
    assert seen[0] is None and seen[-1] in ("http://s2", "http://s3")
    assert set(seen[1:-1]) <= {"http://s1"} and len(set(seen)) == len(seen)


def test_a_connection_that_never_came_up_is_tried_once_more(monkeypatch, tmp_path):
    main, seen, slept = _download(monkeypatch, {None: [UNREACHABLE, {}]})
    main._download_youtube_video(URL, str(tmp_path))
    assert seen == [None, None] and slept == [3]                 # no proxy needed after all


def test_every_exit_refusing_is_a_failure_without_the_paid_proxy(monkeypatch, tmp_path):
    main, seen, _slept = _download(monkeypatch, {
        None: ["HTTP Error 500"], "http://s1": [REFUSED], "http://s2": [REFUSED],
        "http://s3": [REFUSED]})
    with pytest.raises(Exception):
        main._download_youtube_video(URL, str(tmp_path))
    assert seen[0] is None and sorted(seen[1:]) == STATICS       # each once, no paid proxy


def test_a_refused_site_is_not_a_credits_alert():
    from cloud import alerts
    assert not alerts._looks_like_proxy_error(REFUSED)
    assert alerts._classify_failure(REFUSED) != "proxy"
    # What does speak of the account or of a dead proxy still pages.
    for text in ("Tunnel connection failed: 407 Proxy Authentication Required",
                 "ProxyError: Unable to connect to proxy (connection refused)",
                 "HTTP Error 402 Payment Required: out of credits",
                 REFUSED + " ... later: 407 Proxy Authentication Required"):
        assert alerts._looks_like_proxy_error(text), text
