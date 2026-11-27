"""Stage 3: audio-first discovery and full-quality ranges for Twitch VODs.

Twitch serves VODs as muxed HLS whose clock starts near 62 s, not 0. These
tests pin the URL and host checks, the variant choice, the HLS range command
(its timeline arithmetic and protocol whitelist) and the viewer-clip boost.
No network: yt-dlp and FFmpeg are stubbed."""
import json
import subprocess
from pathlib import Path

import numpy as np
import pytest

import hype_engine
import livestream
import range_media

CDN = "https://d1example.cloudfront.net/abc/{}/index-dvr.m3u8"


def _fmt(fid, height, fps=30.0, tbr=1000.0, vcodec="avc1.4D401F", acodec="mp4a.40.2", host=CDN):
    return {"format_id": fid, "protocol": "m3u8_native", "ext": "mp4", "height": height, "fps": fps,
            "tbr": tbr, "vcodec": vcodec, "acodec": acodec, "url": host.format(fid)}


VOD_INFO = {
    "id": "v1234567890", "duration": 27395, "title": "BEATING IT", "uploader": "streamer",
    "formats": [_fmt("1080p60", 1080, 60, 7897), _fmt("720p60", 720, 60, 3056), _fmt("480p", 480, 30, 1409),
                _fmt("360p", 360, 30, 698), _fmt("160p", 160, 30, 289),
                _fmt("Audio Only", None, None, 216, vcodec="none"),
                {"format_id": "sb0", "protocol": "mhtml", "url": "https://x.cloudfront.net/sb"}],
}


@pytest.mark.parametrize("url,expected", [
    ("https://www.twitch.tv/videos/1234567890", "https://www.twitch.tv/videos/1234567890"),
    ("https://twitch.tv/videos/1234567890?t=1h2m3s", "https://www.twitch.tv/videos/1234567890"),
    ("https://www.twitch.tv/somestreamer", None),
    ("https://clips.twitch.tv/SomeClip", None),
])
def test_twitch_url_is_canonical(url, expected):
    assert livestream.twitch_url(url) == expected


def test_staged_url_accepts_youtube_and_twitch_only():
    assert livestream.staged_url("https://youtu.be/dQw4w9WgXcQ") == "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
    assert livestream.staged_url("https://www.twitch.tv/videos/123") == "https://www.twitch.tv/videos/123"
    assert livestream.staged_url("https://vimeo.com/123") is None


@pytest.mark.parametrize("url", [
    "http://d1.cloudfront.net/a/index.m3u8",            # not HTTPS
    "https://evil.example/a/index.m3u8",                 # not a Twitch CDN
    "https://user:pw@d1.cloudfront.net/a/index.m3u8",    # credentials
    "https://d1.cloudfront.net:8443/a/index.m3u8",       # odd port
    "https://d1.cloudfront.net/a/segment.ts",            # not a playlist
])
def test_twitch_tracks_must_be_https_cdn_playlists(url):
    with pytest.raises(livestream.DiscoveryUnavailable):
        livestream._twitch_track_url({"url": url})


def _stub_ydl(monkeypatch, info):
    class FakeYDL:
        def __init__(self, opts):
            self.opts = opts

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def extract_info(self, url, download=False, process=True):
            assert url == "https://www.twitch.tv/videos/1234567890"
            return json.loads(json.dumps(info))

    # CI installs no yt-dlp: the stand-in module is all these tests need.
    import sys
    import types
    monkeypatch.setitem(sys.modules, "yt_dlp", types.SimpleNamespace(YoutubeDL=FakeYDL))


def test_inspect_twitch_picks_best_preview_and_audio(monkeypatch):
    _stub_ydl(monkeypatch, VOD_INFO)
    source = livestream.inspect_source("https://www.twitch.tv/videos/1234567890")
    assert source.kind == "twitch" and source.duration == 27395
    assert source.best["format_id"] == "1080p60"
    assert source.preview["format_id"] == "480p"
    assert source.audio["format_id"] == "Audio Only"
    assert source.summary["video_format"] == "1080p60" and source.summary["height"] == 1080


def test_inspect_twitch_refuses_a_live_or_changed_vod(monkeypatch):
    _stub_ydl(monkeypatch, {**VOD_INFO, "is_live": True})
    with pytest.raises(livestream.DiscoveryUnavailable):
        livestream.inspect_source("https://www.twitch.tv/videos/1234567890")
    _stub_ydl(monkeypatch, VOD_INFO)
    with pytest.raises(livestream.DiscoveryUnavailable):
        livestream.inspect_source("https://www.twitch.tv/videos/1234567890",
                                  {"id": "v1234567890", "duration": 27395, "video_format": "1080p60",
                                   "height": 720, "audio_format": "Audio Only"})


def test_twitch_approval_uses_the_pinned_variant_when_a_higher_one_appears(monkeypatch):
    url = "https://www.twitch.tv/videos/1234567890"
    initial = {**VOD_INFO, "formats": [f for f in VOD_INFO["formats"] if f.get("height") != 1080]}
    _stub_ydl(monkeypatch, initial)
    saved = livestream.inspect_source(url).summary
    assert saved["video_format"] == "720p60"

    _stub_ydl(monkeypatch, VOD_INFO)
    assert livestream.inspect_source(url).best["format_id"] == "1080p60"
    approved = livestream.inspect_source(url, saved)
    assert approved.best["format_id"] == "720p60"
    assert approved.best["height"] == 720

    missing = {**VOD_INFO, "formats": [f for f in VOD_INFO["formats"] if f.get("format_id") != "720p60"]}
    _stub_ydl(monkeypatch, missing)
    with pytest.raises(livestream.DiscoveryUnavailable, match="approved source formats"):
        livestream.inspect_source(url, saved)


def test_inspect_twitch_without_audio_only_uses_the_smallest_variant(monkeypatch):
    info = {**VOD_INFO, "formats": [f for f in VOD_INFO["formats"] if f["format_id"] != "Audio Only"]}
    _stub_ydl(monkeypatch, info)
    assert livestream.inspect_source("https://www.twitch.tv/videos/1234567890").audio["format_id"] == "160p"


def test_twitch_section_fetches_an_hls_range_on_the_source_clock(tmp_path, monkeypatch):
    _stub_ydl(monkeypatch, VOD_INFO)
    source = livestream.inspect_source("https://www.twitch.tv/videos/1234567890")
    monkeypatch.setattr(range_media, "playlist_base", lambda url, timeout=60: 62.033)
    monkeypatch.setattr(livestream, "check_space", lambda d, b: None)
    seen = {}

    def fake_fetch_hls(url, path, start, end, *, base, height, timeout=180):
        seen.update(url=url, path=Path(path), start=start, end=end, base=base, height=height)
        return range_media.Span(Path(path), start - 8, end + 1, 1920, height), 37_000_000

    monkeypatch.setattr(range_media, "fetch_hls", fake_fetch_hls)
    span, size = source.section(tmp_path, 3, 3000.0, 3030.0, preview=False)
    assert seen["url"].endswith("/1080p60/index-dvr.m3u8") and seen["base"] == 62.033
    assert seen["path"].name == "source-3.mkv" and seen["height"] == 1080
    assert span.local(3000.0) == 8.0 and size == 37_000_000
    source.section(tmp_path, 3, 3000.0, 3030.0, preview=True)
    assert seen["url"].endswith("/480p/index-dvr.m3u8") and seen["path"].name == "proxy-3.mkv"


def test_fetch_hls_cuts_on_the_packet_clock_with_a_tight_whitelist(tmp_path, monkeypatch):
    calls = []

    def fake_run(cmd, **kw):
        calls.append(cmd)
        Path(cmd[-1]).write_bytes(b"x" * 10)
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(range_media.subprocess, "run", fake_run)
    monkeypatch.setattr(range_media, "inspect",
                        lambda path, start, end, height=None, base=0.0: range_media.Span(Path(path), 2992.0, 3031.05, 1920, 1080))
    span, size = range_media.fetch_hls("https://d1.cloudfront.net/v/index-dvr.m3u8", tmp_path / "s.mkv",
                                       3000.0, 3030.0, base=62.033, height=1080)
    cmd = calls[0]
    whitelist = cmd[cmd.index("-protocol_whitelist") + 1].split(",")
    assert "http" not in whitelist and "file" not in whitelist and "https" in whitelist
    assert cmd[cmd.index("-ss") + 1] == "2997.0000"
    assert cmd[cmd.index("-to") + 1] == f"{62.033 + 3031:.4f}"      # absolute: -copyts
    assert span.offset == 2992.0 and size == 10 and (tmp_path / "s.mkv").is_file()


def test_inspect_with_a_base_reports_source_seconds(tmp_path, monkeypatch):
    probe = {"format": {"start_time": "3054.033", "duration": "3093.083"},
             "streams": [{"codec_type": "video", "start_time": "3054.033", "width": 1920, "height": 1080,
                          "tags": {"DURATION": "00:51:33.083000000"}},
                         {"codec_type": "audio", "start_time": "3054.059",
                          "tags": {"DURATION": "00:51:33.100000000"}}]}
    monkeypatch.setattr(range_media.subprocess, "run",
                        lambda *a, **k: subprocess.CompletedProcess(a, 0, stdout=json.dumps(probe).encode()))
    span = range_media.inspect(tmp_path / "r.mkv", 3000.0, 3030.0, height=1080, base=62.033)
    assert span.offset == pytest.approx(2992.0) and span.end == pytest.approx(3031.05)
    with pytest.raises(range_media.RangeUnavailable):
        range_media.inspect(tmp_path / "r.mkv", 3000.0, 3030.0, height=1080, base=0.0)


def test_viewer_clips_raise_the_hype_where_people_clipped():
    curve = hype_engine.viewer_curve([{"start": 100, "end": 130, "views": 5000},
                                      {"start": 110, "end": 125, "views": 20},
                                      {"start": "bad"}], 300)
    assert curve[115] > curve[105] > 0 and curve[200] == 0
    assert np.isfinite(curve).all()


def test_audio_format_matches_after_yt_dlp_renames_it():
    """yt-dlp rewrites "Audio Only" to "Audio_Only" while processing: the
    selector must still find it (a live run failed on exactly this)."""
    assert livestream._format_key("Audio Only") == livestream._format_key("Audio_Only")
    assert livestream._format_key("140") == "140"
