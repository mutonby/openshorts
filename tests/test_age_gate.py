"""An age-restricted video moves to another YouTube account.

YouTube shows an age-restricted video only to an account that proved it is
an adult. When one of the cookie accounts is not age-verified, every job that
drew it walked all the routes (the per-GB proxy included) and failed, while
the other accounts open the same video. Now the video moves to an account it
has not tried, and with none left the job stops there with a clear message.
"""
import os
import sys
import types

import pytest

import yt_account

NETSCAPE = "# Netscape HTTP Cookie File\n.youtube.com\tTRUE\t/\tTRUE\t0\tSID\tx\n"
ADULT = NETSCAPE.replace("SID\t", "SID\tadult-")
URL = "https://www.youtube.com/watch?v=tJbzMqJGH4k"
AGE_SIGNED_IN = ("ERROR: [youtube] tJbzMqJGH4k: Sorry, this content is age-restricted. To view "
                 "this video, please provide more info so we can be sure you're an adult.")
AGE_ANON = ("ERROR: [youtube] tJbzMqJGH4k: Sign in to confirm your age. This video may be "
            "inappropriate for some users.")


@pytest.fixture
def two_accounts(tmp_path, monkeypatch):
    monkeypatch.setenv("YT_ACCOUNT_STATE_DIR", str(tmp_path))
    monkeypatch.setattr(yt_account, "MIN_INTERVAL", 0.0)
    monkeypatch.setenv("YOUTUBE_COOKIES", NETSCAPE)
    monkeypatch.setenv("YOUTUBE_COOKIES_2", ADULT)


def test_the_age_gate_texts_are_recognised():
    assert yt_account.is_age_gate(AGE_SIGNED_IN)
    assert yt_account.is_age_gate(AGE_ANON)
    assert not yt_account.is_age_gate("ERROR: [youtube] x: Video unavailable")
    assert not yt_account.is_rate_limit(AGE_SIGNED_IN)


def test_an_age_gated_video_moves_to_the_account_it_has_not_tried(two_accounts):
    s = yt_account.Session()
    try:
        first = s.account
        assert s.age_gated() is True
        assert s.account == 1 - first
        assert s.age_gated() is False                  # both tried: nowhere left
    finally:
        s.close()
    # Not a strike: the account is fine for every other video.
    assert yt_account.resting(0) == 0 and yt_account.resting(1) == 0


def test_the_probe_moves_to_an_adult_account(two_accounts, monkeypatch):
    metering = pytest.importorskip("cloud.metering")
    import security_utils
    monkeypatch.setattr(security_utils, "assert_public_url", lambda u: u)
    monkeypatch.setenv("STATIC_PROXY_URLS", "http://s1,http://s2")
    monkeypatch.setenv("BGUTIL_BASE_URL", "http://bgutil")
    monkeypatch.delenv("PROXY_URL", raising=False)
    monkeypatch.delenv("SOURCE_CACHE_DIR", raising=False)
    calls = []

    class _YDL:
        def __init__(self, opts):
            self.opts = opts

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def extract_info(self, url, download=False):
            ck = self.opts.get("cookiefile")
            adult = bool(ck) and "adult-" in open(ck).read()
            calls.append(adult if ck else None)
            if not ck:
                raise RuntimeError(AGE_ANON)
            if not adult:
                raise RuntimeError(AGE_SIGNED_IN)
            return {"extractor": "youtube", "duration": 600}

    monkeypatch.setitem(sys.modules, "yt_dlp", types.SimpleNamespace(YoutubeDL=_YDL))
    assert metering.probe_url_minutes(URL) == 10.0
    assert True in calls


# --- the download (main.py) -------------------------------------------------

def _download_env(monkeypatch, adult_accounts):
    main = pytest.importorskip("main")
    import security_utils
    monkeypatch.setattr(security_utils, "assert_public_url", lambda u: u)
    monkeypatch.setenv("STATIC_PROXY_URLS", "http://s1,http://s2,http://s3")
    monkeypatch.setenv("PROXY_URL", "http://paid")
    monkeypatch.setenv("BGUTIL_BASE_URL", "http://bgutil")
    monkeypatch.delenv("DIRECT_FIRST", raising=False)
    monkeypatch.delenv("SOURCE_CACHE_DIR", raising=False)
    calls = []

    class _YDL:
        def __init__(self, opts):
            self.opts = opts

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def _check(self):
            ck = self.opts.get("cookiefile")
            adult = bool(ck) and "adult-" in open(ck).read()
            calls.append({"proxy": self.opts.get("proxy"), "cookies": bool(ck), "adult": adult})
            if not ck:
                raise RuntimeError(AGE_ANON)
            if not (adult and adult_accounts):
                raise RuntimeError(AGE_SIGNED_IN)

        def extract_info(self, url, download=False, process=True):
            self._check()
            return {"title": "GTA", "_type": "video", "formats": [{"format_id": "1"}],
                    "duration": 600, "extractor_key": "Youtube"}

        def process_ie_result(self, info, download=True):
            path = self.opts["outtmpl"].replace("%(ext)s", "mp4")
            with open(path, "wb") as f:
                f.write(b"v")
            return info

    fake = types.SimpleNamespace(YoutubeDL=_YDL, version=types.SimpleNamespace(__version__="test"),
                                 utils=sys.modules.get("yt_dlp.utils"))
    monkeypatch.setattr(main, "yt_dlp", fake)
    return main, calls


def test_the_download_moves_to_an_adult_account(two_accounts, monkeypatch, tmp_path):
    main, calls = _download_env(monkeypatch, adult_accounts=True)
    path, title = main.download_youtube_video(URL, str(tmp_path))
    assert os.path.exists(path)
    assert calls[-1]["adult"] is True
    assert all(c["proxy"] != "http://paid" for c in calls)


def test_with_no_adult_account_it_stops_before_the_paid_proxy(two_accounts, monkeypatch,
                                                              tmp_path, capsys):
    main, calls = _download_env(monkeypatch, adult_accounts=False)
    with pytest.raises(Exception):
        main.download_youtube_video(URL, str(tmp_path))
    out = capsys.readouterr().out
    assert main.AGE_RESTRICTED_MESSAGE in out
    assert "YT_ACCOUNT_LIMITED" not in out             # not retried later: the video is the issue
    assert all(c["proxy"] != "http://paid" for c in calls)
    assert len([c for c in calls if c["cookies"]]) == 2   # one request per account, no more
