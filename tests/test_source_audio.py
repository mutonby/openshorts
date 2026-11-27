import asyncio
from copy import deepcopy
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace

import pytest

import youtube_io


URL = "https://youtu.be/12345678901"


def test_remote_render_reuses_the_prepared_track():
    from remote_worker import worker_args

    command = ["python", "main.py", "-u", URL, "--format", "vertical", "--keep-original"]
    assert worker_args(command + ["--source-audio-language", "pt-br"]) == (URL, ["--format", "vertical"])
    assert worker_args(command + ["--source-audio-language", "invalid/lang"]) is None


def _formats(urls):
    video, english, portuguese = urls
    return [
        {"format_id": "video", "url": video, "ext": "mp4", "height": 180,
         "vcodec": "avc1", "acodec": "none", "language": "en"},
        {"format_id": "english", "url": english, "ext": "m4a", "vcodec": "none",
         "acodec": "mp4a.40.2", "language": "en", "language_preference": 10},
        {"format_id": "portuguese", "url": portuguese, "ext": "m4a", "vcodec": "none",
         "acodec": "mp4a.40.2", "language": "pt-BR"},
    ]


def test_audio_language_inventory_excludes_silent_formats_and_rejects_unavailable_tracks():
    formats = _formats(("video", "english", "portuguese"))
    formats.extend([
        {"format_id": "spanish-muxed", "vcodec": "avc1", "acodec": "mp4a", "language": "es"},
        {"format_id": "video-only", "vcodec": "avc1", "acodec": "none", "language": "zh"},
        {"format_id": "unknown-audio", "vcodec": "avc1", "acodec": "mp4a"},
    ])
    info = {"formats": formats}
    assert youtube_io.audio_languages(info) == ["en", "es", "pt-br"]
    selected = youtube_io.select_audio_language(info, "pt-br")
    assert {f["format_id"] for f in selected["formats"]} == {"video", "video-only", "portuguese"}
    assert len(info["formats"]) == 6
    with pytest.raises(youtube_io.SourceAudioUnavailable, match="fr audio track"):
        youtube_io.select_audio_language(info, "fr")


def _stub_audio_first_source(monkeypatch, info):
    class FixtureDownloader:
        def __init__(self, _opts):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_exc):
            pass

        def extract_info(self, *_args, **_kwargs):
            return deepcopy(info)

    monkeypatch.setitem(sys.modules, "yt_dlp", SimpleNamespace(YoutubeDL=FixtureDownloader))


def test_audio_first_review_keeps_the_requested_track(monkeypatch):
    import livestream

    urls = [f"https://fixture.googlevideo.com/{kind}" for kind in ("v", "en", "pt")]
    info = {"id": "12345678901", "duration": 90, "formats": _formats(urls)}
    _stub_audio_first_source(monkeypatch, info)
    assert livestream.inspect_source(URL).audio["format_id"] == "english"
    reviewed = livestream.inspect_source(URL, audio_language="pt-br")
    assert reviewed.summary["audio_language"] == "pt-br"
    assert livestream.inspect_source(URL, reviewed.summary).audio["format_id"] == "portuguese"
    with pytest.raises(livestream.DiscoveryUnavailable, match="fr audio track"):
        livestream.inspect_source(URL, audio_language="fr")


@pytest.mark.parametrize("extension", ["mp4", "webm"])
def test_youtube_approval_keeps_pinned_video_and_audio_when_variants_change(monkeypatch, extension):
    import livestream

    urls = [f"https://fixture.googlevideo.com/{kind}" for kind in ("v", "en", "pt")]
    formats = _formats(urls)
    formats[0]["height"] = 720
    info = {"id": "12345678901", "duration": 90, "formats": formats}
    _stub_audio_first_source(monkeypatch, info)
    saved = livestream.inspect_source(URL, audio_language="pt-br").summary

    formats.extend([
        {**formats[0], "format_id": "new-1080", "height": 1080, "ext": extension},
        {**formats[2], "format_id": "new-portuguese", "language_preference": 10},
    ])
    approved = livestream.inspect_source(URL, saved)
    assert (approved.best["format_id"], approved.best["height"]) == ("video", 720)
    assert approved.audio["format_id"] == "portuguese"
    assert approved.requested_audio_language == "pt-br"

    if extension == "mp4":
        fresh = livestream.inspect_source(URL, audio_language="pt-br")
        assert (fresh.best["format_id"], fresh.audio["format_id"]) == ("new-1080", "new-portuguese")
    else:
        with pytest.raises(livestream.DiscoveryUnavailable, match="best source is not seekable"):
            livestream.inspect_source(URL, audio_language="pt-br")

    formats[:] = [track for track in formats if track["format_id"] != "portuguese"]
    with pytest.raises(livestream.DiscoveryUnavailable, match="approved source formats"):
        livestream.inspect_source(URL, saved)


def test_requested_audio_language_is_queued_and_metadata_only_is_discoverable(monkeypatch, tmp_path):
    from unittest.mock import AsyncMock

    from httpx import ASGITransport, AsyncClient

    api = pytest.importorskip("app")
    monkeypatch.setattr(api, "OUTPUT_DIR", str(tmp_path))
    monkeypatch.setattr(api, "_validate_source_url", AsyncMock())
    monkeypatch.setattr(api, "_check_probe_rate", lambda _actor: None)
    monkeypatch.setattr(api, "_enqueue_job", lambda *_args: None)
    monkeypatch.setattr(api, "jobs", {})
    monkeypatch.setattr(api, "BILLING_ENABLED", False)
    monkeypatch.setattr(api, "DISABLE_YOUTUBE_URL", False)
    probe = AsyncMock(return_value={"duration": 90, "max_height": 1080, "audio_languages": ["en", "pt-br"]})
    monkeypatch.setattr(api, "_probe_youtube_quality", probe)

    async def request():
        async with AsyncClient(transport=ASGITransport(app=api.app), base_url="http://test") as client:
            discovery = await client.post("/api/process/audio-tracks", json={"url": URL})
            assert discovery.json() == {"audio_languages": ["en", "pt-br"]}
            probe.assert_awaited_once_with(URL, include_audio=True)
            common = {"acknowledged": True, "source_audio_language": "PT-BR"}
            headers = {"X-Gemini-Key": "test"}
            response = await client.post("/api/process", headers=headers, json={**common, "url": URL})
            assert response.status_code == 200, response.text
            job_id = response.json()["job_id"]
            cmd = api.jobs[job_id]["cmd"]
            assert cmd[cmd.index("--source-audio-language") + 1] == "pt-br"
            assert api._read_manifest(job_id)["cmd"] == cmd
            invalid = await client.post("/api/process", headers=headers, json={
                **common, "url": URL, "source_audio_language": "en]/best",
            })
            assert invalid.status_code == 400
            unsupported = await client.post("/api/process", headers=headers, json={
                **common, "url": "https://vimeo.com/123",
            })
            assert unsupported.status_code == 400

    asyncio.run(request())


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="Needs FFmpeg")
def test_requested_language_is_muxed_instead_of_the_default(monkeypatch, tmp_path):
    main = pytest.importorskip("main")
    import security_utils

    monkeypatch.setattr(security_utils, "assert_public_url", lambda _url: None)
    monkeypatch.delenv("STATIC_PROXY_URLS", raising=False)
    monkeypatch.delenv("PROXY_URL", raising=False)
    monkeypatch.delenv("SOURCE_CACHE_DIR", raising=False)
    media = [tmp_path / name for name in ("video.mp4", "english.m4a", "portuguese.m4a")]
    subprocess.run([
        "ffmpeg", "-v", "error", "-f", "lavfi", "-i", "color=s=320x180:r=15",
        "-f", "lavfi", "-i", "sine=frequency=440", "-f", "lavfi", "-i", "sine=frequency=880",
        "-map", "0:v", "-c:v", "libx264", "-threads", "1", "-t", "1", str(media[0]),
        "-map", "1:a", "-c:a", "aac", "-t", "1", str(media[1]),
        "-map", "2:a", "-c:a", "aac", "-t", "1", str(media[2]),
    ], check=True, timeout=15)
    formats = _formats(path.as_uri() for path in media)
    mock_info = {"id": "12345678901", "title": "Audio fixture", "duration": 1, "extractor_key": "Youtube",
                 "webpage_url": URL, "formats": formats}

    class FixtureDownloader(main.yt_dlp.YoutubeDL):
        def __init__(self, opts):
            super().__init__({**opts, "enable_file_urls": True, "quiet": True})

        def extract_info(self, url, download=False, process=False):
            assert url == URL and not download and not process
            return deepcopy(mock_info)

    monkeypatch.setattr(main.yt_dlp, "YoutubeDL", FixtureDownloader)
    rendered, _ = main.download_youtube_video(URL, str(tmp_path), audio_language="pt-br")

    def audio_hash(path):
        return subprocess.check_output([
            "ffmpeg", "-v", "error", "-i", str(path), "-map", "0:a:0", "-c", "copy", "-f", "md5", "-",
        ], timeout=15)

    assert Path(rendered).is_file()
    assert audio_hash(rendered) == audio_hash(media[2])
    assert audio_hash(rendered) != audio_hash(media[1])
