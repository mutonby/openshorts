"""Contract tests: legacy per-clip routes reject negative clip_index.

Codex Phase 5 audit found 5 routes that index ``clips[req.clip_index]`` with
only a ``>= len(clips)`` check, so ``clip_index=-1`` would silently mutate
the *last* clip. /api/colorgrade and /api/silencecut already use
``_resolve_clip_input`` which rejects negatives; /api/merge has its own
explicit ``idx < 0`` check. These tests cover the remaining surface.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest


@pytest.fixture
def app_client(tmp_path, monkeypatch):
    """Seed a job with 2 clips + a metadata.json so routes that read from
    disk find a transcript."""
    (tmp_path / "uploads").mkdir(exist_ok=True)
    (tmp_path / "output").mkdir(exist_ok=True)
    monkeypatch.chdir(tmp_path)

    from fastapi.testclient import TestClient
    from app.main import app as fastapi_app, jobs

    job_id = "neg-idx-job"
    job_dir = tmp_path / "output" / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    # Seed clip files + metadata.json (read by /api/subtitle, /api/hook,
    # /api/clip/.../transcript).
    for i in range(2):
        (job_dir / f"_clip_{i}.mp4").write_bytes(b"fake clip")
    metadata = {
        "transcript": {
            "segments": [{
                "words": [
                    {"start": 0.0, "end": 1.0, "word": "hi"},
                ],
            }],
        },
        "shorts": [
            {"start": 0.0, "end": 5.0, "video_url": f"/videos/{job_id}/_clip_0.mp4"},
            {"start": 5.0, "end": 10.0, "video_url": f"/videos/{job_id}/_clip_1.mp4"},
        ],
    }
    (job_dir / "test_metadata.json").write_text(json.dumps(metadata))

    jobs[job_id] = {
        "id": job_id,
        "status": "completed",
        "result": {
            "clips": [
                {"video_url": f"/videos/{job_id}/_clip_0.mp4"},
                {"video_url": f"/videos/{job_id}/_clip_1.mp4"},
            ],
        },
    }

    with TestClient(fastapi_app) as client:
        yield client, job_id, job_dir

    jobs.pop(job_id, None)


def test_edit_rejects_negative_clip_index(app_client):
    client, job_id, _ = app_client
    r = client.post(
        "/api/edit",
        headers={"X-Gemini-Key": "test"},
        json={"job_id": job_id, "clip_index": -1},
    )
    # 404 = our new route-entry guard. With the guard the route returns
    # before any Gemini/FFmpeg work fires.
    assert r.status_code in (400, 404, 422), r.text


def test_effects_generate_rejects_negative_clip_index(app_client):
    client, job_id, _ = app_client
    r = client.post(
        "/api/effects/generate",
        headers={"X-Gemini-Key": "test"},
        json={"job_id": job_id, "clip_index": -1},
    )
    assert r.status_code in (400, 404, 422), r.text


def test_subtitle_rejects_negative_clip_index(app_client):
    client, job_id, _ = app_client
    r = client.post(
        "/api/subtitle",
        json={"job_id": job_id, "clip_index": -1, "language": "en"},
    )
    assert r.status_code in (400, 404, 422), r.text


def test_hook_rejects_negative_clip_index(app_client):
    client, job_id, _ = app_client
    r = client.post(
        "/api/hook",
        json={"job_id": job_id, "clip_index": -1, "text": "hi"},
    )
    assert r.status_code in (400, 404, 422), r.text


def test_clip_transcript_rejects_negative_clip_index(app_client):
    client, job_id, _ = app_client
    r = client.get(f"/api/clip/{job_id}/-1/transcript")
    # FastAPI may resolve path-param "-1" as int -1 OR as a 422 if int
    # parsing accepts negatives; either way the route must NOT serve
    # clips[-1] (the last clip's transcript).
    assert r.status_code in (400, 404, 422), r.text
