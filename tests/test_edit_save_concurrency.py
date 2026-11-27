"""Long renders cannot overwrite another editor save's metadata."""
import asyncio
import json
from pathlib import Path
from uuid import uuid4

import httpx
import pytest

import app as api
import ffmpeg_utils


@pytest.mark.parametrize("clip_index", [0, 1])
def test_dubbing_and_project_credit_both_survive_overlapping_saves(tmp_path, monkeypatch, clip_index):
    job = str(uuid4())
    directory = tmp_path / job
    directory.mkdir()
    name = f"video_clip_{clip_index + 1}.mp4"
    (directory / name).write_bytes(b"body")
    clip = {"clip_index": clip_index, "start": 0, "end": 2, "video_url": f"/videos/{job}/{name}"}
    metadata = directory / "video_metadata.json"
    shorts = [{"start": 0, "end": 2, "quality": {"valid": False}}] * clip_index + [clip]
    metadata.write_text(json.dumps({"shorts": shorts}))
    monkeypatch.setattr(api, "OUTPUT_DIR", str(tmp_path))
    monkeypatch.setattr(api, "jobs", {job: {"status": "completed", "result": {"clips": [clip.copy()]}}})
    monkeypatch.setattr(api, "_rerender_locks", {})
    monkeypatch.setattr(ffmpeg_utils, "mark_ai_generated", lambda *_a: None)
    captured = []
    monkeypatch.setattr(api, "_archive_clip_edit_bg", lambda *args: captured.append((args, json.loads(metadata.read_text()))))
    def translate(**kwargs):
        Path(kwargs["output_path"]).write_bytes(b"dubbed")
    monkeypatch.setattr(api, "translate_video", translate)
    async def run():
        entered, release = asyncio.Event(), asyncio.Event()
        async def card(_directory, filename, _config):
            entered.set()
            await release.wait()
            return filename
        monkeypatch.setattr(api, "_append_endcard", card)
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=api.app), base_url="http://test") as client:
            dub = asyncio.create_task(client.post("/api/translate", headers={"X-ElevenLabs-Key": "test"},
                json={"job_id": job, "clip_index": clip_index, "target_language": "es"}))
            await asyncio.wait_for(entered.wait(), 3)
            credit = asyncio.create_task(client.put(f"/api/projects/{job}/publishing",
                json={"source_credit_url": "https://example.com/source"}))
            await asyncio.sleep(0.02)
            release.set()
            dubbed, saved = await asyncio.gather(dub, credit)
            assert dubbed.status_code == saved.status_code == 200
            data = json.loads(metadata.read_text())
            assert data["shorts"][clip_index]["video_url"] == dubbed.json()["new_video_url"]
            assert api.jobs[job]["result"]["clips"][0]["video_url"] == dubbed.json()["new_video_url"]
            assert data["publishing"]["source_credit_url"] == "https://example.com/source"
            assert captured[-1][1] == data
            assert captured[-1][0][0:2] == (job, clip_index)
    asyncio.run(run())
