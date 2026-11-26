import asyncio
import io
import json
from pathlib import Path
import shutil
import subprocess

import pytest
from PIL import Image

import hooks


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="Needs FFmpeg")
def test_closing_text_replaces_the_whole_clip_hook_on_the_final_frames(tmp_path):
    source = tmp_path / "black.mp4"
    output = tmp_path / "text.mp4"
    subprocess.run([
        "ffmpeg", "-v", "error", "-f", "lavfi", "-i", "color=c=black:s=360x640:r=10",
        "-f", "lavfi", "-i", "sine=frequency=440", "-t", "5", "-c:v", "libx264",
        "-threads", "1", "-c:a", "aac", str(source),
    ], check=True, timeout=20)

    hooks.add_hook_and_cta_to_video(
        str(source), "Opening", str(output), position="top", face_aware=False,
        cta={"text": "Follow for more", "position": "bottom", "size": "M", "duration_seconds": 2},
    )

    def bright_regions(time):
        png = subprocess.check_output([
            "ffmpeg", "-v", "error", "-ss", str(time), "-i", str(output), "-frames:v", "1",
            "-c:v", "png", "-threads", "1", "-f", "image2pipe", "-",
        ], timeout=20)
        with Image.open(io.BytesIO(png)) as frame:
            gray = frame.convert("L")
            return tuple(gray.crop(box).getextrema()[1] > 100
                         for box in ((0, 0, 360, 320), (0, 320, 360, 640)))

    assert bright_regions(2.9) == (True, False)
    assert bright_regions(3.0) == (False, True)
    assert bright_regions(4.9) == (False, True)
    info = subprocess.check_output([
        "ffprobe", "-v", "error", "-show_entries", "stream=codec_type", "-of", "json", str(output),
    ], timeout=20)
    assert {stream["codec_type"] for stream in json.loads(info)["streams"]} == {"video", "audio"}


def test_editing_closing_text_uses_the_clean_file_and_preserves_the_saved_captions(monkeypatch, tmp_path):
    app = pytest.importorskip("app")
    import httpx

    job_id = "test-clip-text"
    clean = "sample_clip_1.mp4"
    previous = f"subtitled_1_hooked_1_{clean}"
    job_dir = tmp_path / job_id
    job_dir.mkdir()
    for name in (clean, f"hooked_1_{clean}", previous):
        (job_dir / name).write_bytes(b"video")
    metadata = job_dir / "sample_metadata.json"
    clip = {"start": 0, "end": 5, "clip_index": 0,
            "video_url": f"/videos/{job_id}/{previous}", "auto_hook": {"text": "Old hook"}}
    metadata.write_text(json.dumps({"shorts": [clip]}))
    job = {"status": "completed", "result": {"clips": [dict(clip)]}, "ready_files": {0: previous}}
    monkeypatch.setattr(app, "OUTPUT_DIR", str(tmp_path))
    monkeypatch.setitem(app.jobs, job_id, job)

    def render(video_path, text, path, *, cta, **kwargs):
        assert video_path == str(job_dir / clean)
        assert text == ""
        assert cta["position"] == "top"
        assert kwargs["has_captions"]
        assert kwargs["design"]["text_color"] == "#B5E812"
        with open(path, "wb") as media:
            media.write(b"closing text")

    def caption(_job_id, _index, path):
        rendered = Path(path)
        captioned = rendered.with_name(f"subtitled_2_{rendered.name}")
        captioned.write_bytes(b"captions")
        return str(captioned)

    monkeypatch.setattr(app, "add_hook_and_cta_to_video", render)
    monkeypatch.setattr(app, "_reapply_captions", caption)

    async def edit_and_remove():
        transport = httpx.ASGITransport(app=app.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            request = {"job_id": job_id, "clip_index": 0}
            response = await client.post("/api/hook", json={
                **request, "cta": {"text": "Follow", "position": "top", "duration_seconds": 4},
                "design": {"shape": "pill", "text_color": "#B5E812", "outline_width": 4},
            })
            assert response.status_code == 200, response.text
            saved = json.loads(metadata.read_text())["shorts"][0]
            assert saved["auto_hook"]["cta"]["text"] == "Follow"
            assert saved["auto_hook"]["design"]["outline_width"] == 4
            assert response.json()["burned_hook"]["design"] == saved["auto_hook"]["design"]
            assert saved["video_url"] == response.json()["new_video_url"]
            assert job["ready_files"][0].startswith("subtitled_2_hooked_")

            reapply = app._recut_text_overlay(saved, {})
            recut_output = reapply(str(job_dir / clean), has_captions=True)
            assert Path(recut_output).read_bytes() == b"closing text"

            response = await client.post("/api/hook", json={**request, "remove": True})
            assert response.status_code == 200, response.text
            saved = json.loads(metadata.read_text())["shorts"][0]
            assert "auto_hook" not in saved
            assert saved["video_url"].endswith(f"/subtitled_2_{clean}")

    asyncio.run(edit_and_remove())
