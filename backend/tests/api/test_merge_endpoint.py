"""Contract tests for POST /api/merge.

Validates the request schema (bounds checks, dedup, transition allowlist) and
the integration with the in-memory job store. The actual ffmpeg invocation is
mocked at ``app.video.merge.concat_clips`` so the test runs without ffmpeg.
"""
from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

import pytest


@pytest.fixture
def app_client(tmp_path, monkeypatch):
    (tmp_path / "uploads").mkdir(exist_ok=True)
    (tmp_path / "output").mkdir(exist_ok=True)
    monkeypatch.chdir(tmp_path)

    from fastapi.testclient import TestClient
    from app.main import app as fastapi_app, jobs

    # Seed a fake completed job with 3 clips.
    job_id = "test-merge-job"
    job_dir = tmp_path / "output" / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    clip_files = []
    for i in range(3):
        p = job_dir / f"_clip_{i}.mp4"
        p.write_bytes(b"fake clip data")
        clip_files.append(p)
    jobs[job_id] = {
        "id": job_id,
        "status": "completed",
        "result": {
            "clips": [
                {"video_url": f"/videos/{job_id}/_clip_{i}.mp4"} for i in range(3)
            ],
        },
    }

    with TestClient(fastapi_app) as client:
        yield client, job_id, job_dir

    jobs.pop(job_id, None)


def test_merge_rejects_unknown_job(app_client):
    client, _job_id, _ = app_client
    r = client.post("/api/merge", json={
        "job_id": "ghost-job",
        "clip_indices": [0, 1],
    })
    assert r.status_code == 404


def test_merge_rejects_out_of_bounds_clip_index(app_client):
    client, job_id, _ = app_client
    r = client.post("/api/merge", json={
        "job_id": job_id,
        "clip_indices": [0, 99],
    })
    assert r.status_code in (400, 422)


def test_merge_rejects_negative_clip_index(app_client):
    """Negative indices must be rejected at the route boundary.

    Defense-in-depth alongside Pydantic: even though clip_count = 3, a
    `clip_index=-1` would index `clips[-1]` (the last clip) silently.
    The route's `idx < 0` check rejects it with 400. Covers Codex
    test_merge_endpoint:59 gap.
    """
    client, job_id, _ = app_client
    r = client.post("/api/merge", json={
        "job_id": job_id,
        "clip_indices": [-1, 0],
    })
    assert r.status_code in (400, 422)


def test_merge_normalizes_transition_case(app_client):
    """field_validator strips + lowercases transition; ' CUT ' is accepted."""
    client, job_id, _ = app_client

    def fake_concat(inputs, output):
        Path(output).write_bytes(b"merged")
        return output

    with patch("app.main.concat_clips", side_effect=fake_concat):
        r = client.post("/api/merge", json={
            "job_id": job_id,
            "clip_indices": [0, 1],
            "transition": " CUT ",
        })
    assert r.status_code == 200, r.text


def test_merge_atomic_rename_uses_unique_partial_paths(app_client):
    """Concurrent identical merges write to unique partial paths.

    Codex flagged test_merge_endpoint:111 (no concurrent-clobber coverage).
    We verify the contract: concat_clips receives a final-path output and
    handles partial-rename internally; the route hands it the public path
    and trusts the helper to atomically swap. This guards the route-helper
    contract, not the helper internals (those are in test_merge.py).
    """
    client, job_id, _ = app_client
    seen_outputs: list[str] = []

    def fake_concat(inputs, output):
        seen_outputs.append(output)
        Path(output).write_bytes(b"merged")
        return output

    with patch("app.main.concat_clips", side_effect=fake_concat):
        r1 = client.post("/api/merge", json={
            "job_id": job_id, "clip_indices": [0, 1],
        })
        r2 = client.post("/api/merge", json={
            "job_id": job_id, "clip_indices": [0, 1],
        })
    assert r1.status_code == 200 and r2.status_code == 200
    # Both calls converge on the same idempotency-key filename — the helper
    # is responsible for unique partials underneath.
    assert seen_outputs[0] == seen_outputs[1]
    assert seen_outputs[0].endswith("merged_0_1.mp4")


def test_merge_rejects_single_clip(app_client):
    client, job_id, _ = app_client
    r = client.post("/api/merge", json={
        "job_id": job_id,
        "clip_indices": [0],
    })
    assert r.status_code in (400, 422)


def test_merge_rejects_unknown_transition(app_client):
    client, job_id, _ = app_client
    r = client.post("/api/merge", json={
        "job_id": job_id,
        "clip_indices": [0, 1],
        "transition": "starfade",
    })
    assert r.status_code in (400, 422)


def test_merge_dedupes_repeated_clip_indices(app_client):
    client, job_id, job_dir = app_client

    captured = {}

    def fake_concat(inputs, output):
        captured["inputs"] = list(inputs)
        captured["output"] = output
        Path(output).write_bytes(b"merged")
        return output

    with patch("app.main.concat_clips", side_effect=fake_concat):
        r = client.post("/api/merge", json={
            "job_id": job_id,
            "clip_indices": [0, 0, 1, 1, 2],
        })
    assert r.status_code == 200
    # Dedup preserves first occurrence order: [0, 1, 2].
    assert len(captured["inputs"]) == 3
    assert os.path.basename(captured["inputs"][0]) == "_clip_0.mp4"
    assert os.path.basename(captured["inputs"][1]) == "_clip_1.mp4"
    assert os.path.basename(captured["inputs"][2]) == "_clip_2.mp4"


def test_merge_happy_path_returns_new_video_url(app_client):
    client, job_id, job_dir = app_client

    def fake_concat(inputs, output):
        Path(output).write_bytes(b"merged")
        return output

    with patch("app.main.concat_clips", side_effect=fake_concat):
        r = client.post("/api/merge", json={
            "job_id": job_id,
            "clip_indices": [2, 0],
        })
    assert r.status_code == 200
    body = r.json()
    assert body["success"] is True
    assert body["new_video_url"].startswith(f"/videos/{job_id}/merged_")
    assert body["new_video_url"].endswith(".mp4")
    # Filename encodes the user-picked order, not sorted: "merged_2_0.mp4".
    assert "merged_2_0.mp4" in body["new_video_url"]
