"""HTTP tests for the Thumbnail Studio -> clip generator handover (issue #68).

Same conventions as test_rerender_endpoint.py: a real ASGI round-trip against
the imported app (BILLING_ENABLED=0 via conftest). The job is only ENQUEUED —
the worker never runs under ASGITransport (no lifespan) — so these tests own
the /api/process contract for `thumbnail_session_id`: session validation, the
hardlinked source under the job's name, the transcript file ridden along via
--transcript, and the attestation source.
"""

import asyncio
import json
import os

import httpx
import pytest

app_module = pytest.importorskip("app")

TRANSCRIPT = {
    "text": "hello world again",
    "language": "en",
    "segments": [
        {
            "start": 0.0, "end": 60.0, "text": "hello world again",
            "words": [
                {"word": "hello", "start": 12.0, "end": 12.5},
                {"word": "world", "start": 20.0, "end": 20.4},
                {"word": "again", "start": 50.0, "end": 50.5},
            ],
        },
    ],
}


def _post_process(json_body):
    async def _do():
        transport = httpx.ASGITransport(app=app_module.app)
        async with httpx.AsyncClient(transport=transport,
                                     base_url="http://testserver") as client:
            return await client.post(
                "/api/process", json=json_body,
                headers={"X-Gemini-Key": "test-key"},
            )
    return asyncio.run(_do())


@pytest.fixture()
def dirs(tmp_path, monkeypatch):
    out_root = tmp_path / "output"
    up_root = tmp_path / "uploads"
    out_root.mkdir()
    up_root.mkdir()
    monkeypatch.setattr(app_module, "OUTPUT_DIR", str(out_root))
    monkeypatch.setattr(app_module, "UPLOAD_DIR", str(up_root))
    return out_root, up_root


@pytest.fixture()
def session(dirs, monkeypatch):
    """A Thumbnail Studio session with its video on disk and a ready transcript."""
    _, up_root = dirs
    video = up_root / "thumb_sess1_source.mp4"
    video.write_bytes(b"fake-video-bytes")
    sess = {
        "user_id": None,
        "video_path": str(video),
        "transcript_ready": True,
        "transcript": TRANSCRIPT,
    }
    monkeypatch.setitem(app_module.thumbnail_sessions, "sess1", sess)
    return sess


def _submitted_job(resp):
    assert resp.status_code == 200, resp.text
    job_id = resp.json()["job_id"]
    return job_id, app_module.jobs[job_id]


def test_handover_links_source_and_transcript(dirs, session):
    out_root, up_root = dirs
    resp = _post_process({"thumbnail_session_id": "sess1", "acknowledged": True})
    job_id, job = _submitted_job(resp)
    cmd = job["cmd"]

    # Source: hardlinked under the job's name so /api/source, the clip editor
    # and retention treat it exactly like a normal upload.
    input_path = cmd[cmd.index("-i") + 1]
    assert os.path.dirname(input_path) == str(up_root)
    assert os.path.basename(input_path).startswith(job_id)
    assert os.path.exists(input_path)
    assert open(input_path, "rb").read() == b"fake-video-bytes"
    # The original session file must still exist (link, not move).
    assert os.path.exists(session["video_path"])

    # Transcript: written into the job dir and passed via --transcript.
    transcript_path = cmd[cmd.index("--transcript") + 1]
    assert os.path.dirname(transcript_path) == str(out_root / job_id)
    with open(transcript_path) as f:
        assert json.load(f) == TRANSCRIPT

    assert job["attestation"]["source"] == "thumbnail_session"


def test_handover_without_ready_transcript_still_processes(dirs, session):
    session["transcript_ready"] = False
    session["transcript"] = None
    resp = _post_process({"thumbnail_session_id": "sess1", "acknowledged": True})
    _, job = _submitted_job(resp)
    assert "-i" in job["cmd"]
    assert "--transcript" not in job["cmd"]


def test_empty_transcript_is_not_forwarded(dirs, session):
    """A silent/music-only source yields {"segments": []} — main.py would
    reject it, so the endpoint must not bother writing it."""
    session["transcript"] = {"text": "", "language": "en", "segments": []}
    resp = _post_process({"thumbnail_session_id": "sess1", "acknowledged": True})
    _, job = _submitted_job(resp)
    assert "--transcript" not in job["cmd"]


def test_unknown_session_404s(dirs):
    resp = _post_process({"thumbnail_session_id": "nope", "acknowledged": True})
    assert resp.status_code == 404


def test_session_with_missing_video_404s(dirs, session):
    os.remove(session["video_path"])
    resp = _post_process({"thumbnail_session_id": "sess1", "acknowledged": True})
    assert resp.status_code == 404


def test_handover_still_requires_attestation(dirs, session):
    resp = _post_process({"thumbnail_session_id": "sess1"})
    assert resp.status_code == 400


def test_url_wins_over_session_id(dirs, session, monkeypatch):
    """A request carrying both keeps the historical URL behavior untouched."""
    monkeypatch.setattr(app_module, "QUALITY_GATE_MIN_HEIGHT", 0)
    resp = _post_process({
        "thumbnail_session_id": "sess1",
        "url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        "acknowledged": True,
    })
    _, job = _submitted_job(resp)
    assert "-u" in job["cmd"]
    assert "--transcript" not in job["cmd"]
    assert job["attestation"]["source"] == "url"


def _post_raw(content):
    async def _do():
        transport = httpx.ASGITransport(app=app_module.app)
        async with httpx.AsyncClient(transport=transport,
                                     base_url="http://testserver") as client:
            return await client.post(
                "/api/process", content=content,
                headers={"X-Gemini-Key": "test-key",
                         "Content-Type": "application/json"},
            )
    return asyncio.run(_do())


@pytest.mark.parametrize("raw", [b"{'url': 'x'}", b"not json", b"\xff\xfe", b"[1, 2]", b"null"])
def test_malformed_json_body_is_400_not_500(dirs, raw):
    r = _post_raw(raw)
    assert r.status_code == 400
