"""HTTP tests for POST /api/edit (the "auto edit" button).

The model and ffmpeg are stubbed at the VideoEditor seam; these tests own what
the endpoint does around them: each request works on temp files of its own,
and an edit that wrote no video is an error, never a clip pointing at a file
that does not exist."""

import asyncio
import json
import os

import httpx
import pytest

app_module = pytest.importorskip("app")
pytest.importorskip("cv2")

JOB_ID = "edit-endpoint-test-job"


def _request(json_body):
    async def _do():
        transport = httpx.ASGITransport(app=app_module.app)
        async with httpx.AsyncClient(transport=transport,
                                     base_url="http://testserver") as client:
            return await client.post("/api/edit", json=json_body,
                                     headers={"X-Gemini-Key": "test-key"})
    return asyncio.run(_do())


@pytest.fixture()
def job(tmp_path, monkeypatch):
    out_root = tmp_path / "output"
    job_dir = out_root / JOB_ID
    job_dir.mkdir(parents=True)
    monkeypatch.setattr(app_module, "OUTPUT_DIR", str(out_root))
    clips = [{"start": 10.0 * i, "end": 10.0 * i + 30,
              "video_url": f"/videos/{JOB_ID}/t_clip_{i + 1}.mp4"} for i in range(2)]
    for i in range(2):
        (job_dir / f"t_clip_{i + 1}.mp4").write_bytes(f"clip{i + 1}".encode())
    meta_path = job_dir / "t_metadata.json"
    meta_path.write_text(json.dumps({"shorts": [dict(c) for c in clips],
                                     "transcript": {"language": "en", "segments": []}}))
    app_module.jobs[JOB_ID] = {"status": "completed", "logs": [], "user_id": None,
                               "result": {"clips": clips, "cost_analysis": {}}}
    try:
        yield {"dir": job_dir, "meta_path": meta_path}
    finally:
        app_module.jobs.pop(JOB_ID, None)


@pytest.fixture()
def editor(monkeypatch):
    """A VideoEditor that copies its input to its output, and records both."""
    calls = []

    class Fake:
        writes = True

        def __init__(self, api_key):
            pass

        def upload_video(self, path):
            assert os.path.exists(path)
            return object()

        def get_ffmpeg_filter(self, *a, **kw):
            return {"filter": "null"}

        def apply_edits(self, input_path, output_path, filter_data):
            calls.append((os.path.basename(input_path), os.path.basename(output_path)))
            if Fake.writes:
                with open(input_path, "rb") as src, open(output_path, "wb") as dst:
                    dst.write(b"edited:" + src.read())

    monkeypatch.setattr(app_module, "VideoEditor", Fake)
    return Fake, calls


def test_the_edit_becomes_the_clips_file(job, editor):
    resp = _request({"job_id": JOB_ID, "clip_index": 0})
    assert resp.status_code == 200, resp.text
    assert resp.json()["new_video_url"] == f"/videos/{JOB_ID}/edited_t_clip_1.mp4"
    assert (job["dir"] / "edited_t_clip_1.mp4").read_bytes() == b"edited:clip1"
    assert not [f for f in os.listdir(job["dir"]) if f.startswith("temp_")]


def test_two_clips_of_a_job_never_share_temp_files(job, editor):
    """With one temp name per job, edits running at once overwrote each
    other's input and moved each other's result."""
    _fake, calls = editor
    assert _request({"job_id": JOB_ID, "clip_index": 0}).status_code == 200
    assert _request({"job_id": JOB_ID, "clip_index": 1}).status_code == 200
    (in_a, out_a), (in_b, out_b) = calls
    assert in_a != in_b and out_a != out_b
    assert (job["dir"] / "edited_t_clip_2.mp4").read_bytes() == b"edited:clip2"


def test_an_edit_that_wrote_no_video_is_an_error(job, editor):
    """It used to answer 200 and leave the clip pointing at a file that was
    never written: the post, the download and the next edit all failed on it."""
    fake, _calls = editor
    fake.writes = False
    before = job["meta_path"].read_text()
    resp = _request({"job_id": JOB_ID, "clip_index": 0})
    assert resp.status_code == 500
    assert job["meta_path"].read_text() == before
    assert app_module.jobs[JOB_ID]["result"]["clips"][0]["video_url"].endswith("/t_clip_1.mp4")
    assert not (job["dir"] / "edited_t_clip_1.mp4").exists()
    assert not [f for f in os.listdir(job["dir"]) if f.startswith("temp_")]


def test_a_job_without_a_result_yet_is_a_400_not_a_500(job, editor):
    """A queued or failed job has result=None; `'clips' in None` used to be a 500."""
    app_module.jobs[JOB_ID]["result"] = None
    r = _request({"job_id": JOB_ID, "clip_index": 0})
    assert r.status_code == 400
    assert r.json()["detail"] == "Job result not available"
