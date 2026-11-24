import asyncio
from copy import deepcopy
from pathlib import Path
import shutil
import subprocess

import pytest

import youtube_io


URL = "https://youtu.be/12345678901"


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
            manifest = Path(tmp_path, job_id, api._RESUME_FILE).read_text()
            assert "--source-audio-language" in manifest
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


def test_worker_renders_prepared_audio_without_replacing_it_with_the_default_track():
    import remote_worker

    command = ["python", "main.py", "-u", URL, "--format", "vertical",
               "--source-audio-language", "pt-br", "--keep-original"]
    assert remote_worker.worker_args(command) == (URL, ["--format", "vertical"])
    assert remote_worker.link_of(command) is None
    assert remote_worker.link_of(command[:6]) == URL
