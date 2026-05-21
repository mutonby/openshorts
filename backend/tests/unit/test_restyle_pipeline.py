"""Tests for restyle/pipeline: the AI Restyle orchestrator.

Each ML / FFmpeg dep is mocked. We verify:
- Steps run in order on the happy path
- Progress + status are written to the supplied jobs dict
- A failure in any step marks the job 'failed' and short-circuits
- Videos over the 30s cap are rejected with a clear log line
"""
from __future__ import annotations

import asyncio
from unittest.mock import MagicMock, patch

import pytest

from app.restyle.pipeline import MAX_DURATION_SEC, run_restyle_job


@pytest.fixture
def fake_jobs(tmp_path):
    job_id = "test-restyle-job"
    input_file = tmp_path / "in.mp4"
    input_file.write_bytes(b"\x00\x00\x00\x18ftypisom" + b"\x00" * 200)
    return {
        job_id: {
            "status": "processing",
            "logs": [],
            "result": None,
            "progress_pct": 0,
        },
    }, job_id, input_file


def test_pipeline_happy_path(tmp_path, fake_jobs, monkeypatch):
    jobs, job_id, input_file = fake_jobs
    monkeypatch.setattr("app.restyle.pipeline.OUTPUT_DIR", str(tmp_path / "output"))

    with patch("app.restyle.pipeline.probe_duration", return_value=12.0), \
         patch("app.restyle.pipeline.extract_first_frame") as fake_extract, \
         patch("app.restyle.pipeline.relight_frame") as fake_relight, \
         patch("app.restyle.pipeline.restyle_video") as fake_v2v, \
         patch("app.restyle.pipeline.mux_video_audio") as fake_mux:
        fake_extract.side_effect = lambda src, dst: dst
        fake_relight.side_effect = lambda **kw: kw["out_path"]
        fake_v2v.side_effect = lambda **kw: kw["out_path"]
        fake_mux.side_effect = lambda video, audio, out: out

        asyncio.run(run_restyle_job(
            jobs=jobs,
            job_id=job_id,
            input_path=str(input_file),
            background_prompt="bahamas",
            lighting_prompt="golden",
            gemini_key="g",
            fal_key="f",
        ))

    job = jobs[job_id]
    assert job["status"] == "completed"
    assert job["progress_pct"] == 100
    assert job["result"]["video_url"].endswith(".mp4")
    assert job["result"]["duration_sec"] == 12.0
    assert job["result"]["original_url"].endswith(input_file.name)
    fake_extract.assert_called_once()
    fake_relight.assert_called_once()
    fake_v2v.assert_called_once()
    fake_mux.assert_called_once()


def test_pipeline_marks_failed_when_relight_raises(tmp_path, fake_jobs, monkeypatch):
    jobs, job_id, input_file = fake_jobs
    monkeypatch.setattr("app.restyle.pipeline.OUTPUT_DIR", str(tmp_path / "output"))

    with patch("app.restyle.pipeline.probe_duration", return_value=10.0), \
         patch("app.restyle.pipeline.extract_first_frame", side_effect=lambda src, dst: dst), \
         patch("app.restyle.pipeline.relight_frame", side_effect=RuntimeError("content policy")), \
         patch("app.restyle.pipeline.restyle_video") as fake_v2v, \
         patch("app.restyle.pipeline.mux_video_audio") as fake_mux:
        asyncio.run(run_restyle_job(
            jobs=jobs,
            job_id=job_id,
            input_path=str(input_file),
            background_prompt="x",
            lighting_prompt="y",
            gemini_key="g",
            fal_key="f",
        ))

    job = jobs[job_id]
    assert job["status"] == "failed"
    assert any("content policy" in line for line in job["logs"])
    fake_v2v.assert_not_called()
    fake_mux.assert_not_called()


def test_pipeline_rejects_videos_longer_than_cap(tmp_path, fake_jobs, monkeypatch):
    jobs, job_id, input_file = fake_jobs
    monkeypatch.setattr("app.restyle.pipeline.OUTPUT_DIR", str(tmp_path / "output"))

    with patch("app.restyle.pipeline.probe_duration", return_value=MAX_DURATION_SEC + 15.0), \
         patch("app.restyle.pipeline.extract_first_frame") as fake_extract:
        asyncio.run(run_restyle_job(
            jobs=jobs,
            job_id=job_id,
            input_path=str(input_file),
            background_prompt="x",
            lighting_prompt="y",
            gemini_key="g",
            fal_key="f",
        ))

    job = jobs[job_id]
    assert job["status"] == "failed"
    assert any("30s" in line or "duration" in line.lower() for line in job["logs"])
    fake_extract.assert_not_called()


def test_pipeline_writes_progress_increments(tmp_path, fake_jobs, monkeypatch):
    """Progress pct should rise monotonically across the steps."""
    jobs, job_id, input_file = fake_jobs
    monkeypatch.setattr("app.restyle.pipeline.OUTPUT_DIR", str(tmp_path / "output"))

    seen_pct = []

    def _spy_step(*args, **kwargs):
        seen_pct.append(jobs[job_id]["progress_pct"])
        # Return any "result" the step needs to look successful
        return kwargs.get("out_path", args[-1] if args else None)

    with patch("app.restyle.pipeline.probe_duration", return_value=8.0), \
         patch("app.restyle.pipeline.extract_first_frame", side_effect=_spy_step), \
         patch("app.restyle.pipeline.relight_frame", side_effect=_spy_step), \
         patch("app.restyle.pipeline.restyle_video", side_effect=_spy_step), \
         patch("app.restyle.pipeline.mux_video_audio", side_effect=_spy_step):
        asyncio.run(run_restyle_job(
            jobs=jobs,
            job_id=job_id,
            input_path=str(input_file),
            background_prompt="x",
            lighting_prompt="y",
            gemini_key="g",
            fal_key="f",
        ))

    # Each step recorded the pct BEFORE doing its work; final job.progress_pct == 100
    assert seen_pct == sorted(seen_pct), f"progress should be monotonic, got {seen_pct}"
    assert jobs[job_id]["progress_pct"] == 100


def test_pipeline_handles_missing_job_id_gracefully(tmp_path, monkeypatch):
    """If the caller never populated jobs[job_id], the pipeline must not
    KeyError — it should set up the entry itself."""
    jobs = {}
    monkeypatch.setattr("app.restyle.pipeline.OUTPUT_DIR", str(tmp_path / "output"))
    input_file = tmp_path / "in.mp4"
    input_file.write_bytes(b"x" * 100)

    with patch("app.restyle.pipeline.probe_duration", return_value=5.0), \
         patch("app.restyle.pipeline.extract_first_frame", side_effect=lambda src, dst: dst), \
         patch("app.restyle.pipeline.relight_frame", side_effect=lambda **kw: kw["out_path"]), \
         patch("app.restyle.pipeline.restyle_video", side_effect=lambda **kw: kw["out_path"]), \
         patch("app.restyle.pipeline.mux_video_audio", side_effect=lambda v, a, o: o):
        asyncio.run(run_restyle_job(
            jobs=jobs,
            job_id="new-job",
            input_path=str(input_file),
            background_prompt="x",
            lighting_prompt="y",
            gemini_key="g",
            fal_key="f",
        ))

    assert jobs["new-job"]["status"] == "completed"
