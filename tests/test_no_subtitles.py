"""The no-subtitles option survives edits, restarts and downloads.
Run: python tests/test_no_subtitles.py"""
import asyncio
import io
import json
import os
import sys
import tempfile
import zipfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import httpx

import app as app_module
import recut

JOB = "nosubs-test-job"
CLEAN = "t_clip_1.mp4"
HOOKED = "hooked_1_" + CLEAN
CAPTIONED = "subtitled_2_" + HOOKED


def _run(current, method, path, body=None, files=(CLEAN, HOOKED, CAPTIONED), in_metadata=False):
    """One request against a job serving ``current``. Returns (response, recut kwargs)."""
    calls = []

    def fake(**kw):
        calls.append(kw)
        name = "recut_1_" + kw["clean_name"]
        open(os.path.join(kw["output_dir"], name), "wb").write(b"recut")
        return name, name

    async def send():
        transport = httpx.ASGITransport(app=app_module.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as c:
            return await c.request(method, path, json=body)

    saved = app_module.OUTPUT_DIR, recut.perform_recut, app_module._locate_source
    with tempfile.TemporaryDirectory() as tmp:
        d = os.path.join(tmp, JOB)
        os.mkdir(d)
        for name in files:
            open(os.path.join(d, name), "wb").write(name.encode())
        open(os.path.join(d, "src.mp4"), "wb").write(b"source")
        clip = {"start": 10.0, "end": 40.0}
        live = dict(clip, video_url=f"/videos/{JOB}/{current}")
        meta = {"shorts": [live if in_metadata else clip], "transcript": {"language": "ar", "segments": [
            {"start": 10, "end": 39, "text": "x", "words": [{"word": " x", "start": 12, "end": 13}]}]}}
        json.dump(meta, open(os.path.join(d, "t_metadata.json"), "w"))
        app_module.OUTPUT_DIR, recut.perform_recut = tmp, fake
        app_module._locate_source = lambda _job: os.path.join(d, "src.mp4")
        app_module.jobs[JOB] = {"status": "completed", "logs": [], "user_id": None, "result": {"clips": [live]}}
        try:
            return asyncio.run(send()), calls
        finally:
            app_module.OUTPUT_DIR, recut.perform_recut, app_module._locate_source = saved
            app_module.jobs.pop(JOB, None)


def _recut(current, **extra):
    body = {"job_id": JOB, "clip_index": 0, "segments": [{"start": 12, "end": 22}], **extra}
    r, calls = _run(current, "POST", "/api/clip/rerender", body)
    assert r.status_code == 200, r.text
    return calls[0]["captions_transcript"]


def _reframe(current):
    r, calls = _run(current, "POST", "/api/clip/reframe",
                    {"job_id": JOB, "clip_index": 0, "crop_overrides": {"0": .5}})
    assert r.status_code == 200, r.text
    return calls[0]["captions_transcript"]


def _zip(current, query="", **kw):
    return _run(current, "GET", f"/api/jobs/{JOB}/download-all{query}", **kw)[0]


def test_recut_follows_the_clip():
    assert _recut(CAPTIONED)
    assert _recut(HOOKED) is None  # an older subtitled_ file on disk must not count
    assert _recut(CLEAN) is None


def test_recut_explicit_choice_wins():
    assert _recut(HOOKED, reapply_captions=True)
    assert _recut(CAPTIONED, reapply_captions=False) is None


def test_reframe_follows_the_clip():
    assert _reframe(CAPTIONED)
    assert _reframe(HOOKED) is None


def test_editor_is_told_the_truth():
    for current, expected in ((CAPTIONED, True), (HOOKED, False)):
        r, _ = _run(current, "GET", f"/api/clip/{JOB}/0/edl")
        assert r.json()["has_captions"] is expected


def test_zip_without_subtitles():
    r = _zip(CAPTIONED, "?subtitles=0")
    z = zipfile.ZipFile(io.BytesIO(r.content))
    assert [z.read(n) for n in z.namelist()] == [HOOKED.encode()]
    assert "_no_subtitles" in r.headers["content-disposition"]


def test_plain_zip_is_unchanged():
    z = zipfile.ZipFile(io.BytesIO(_zip(CAPTIONED).content))
    assert [z.read(n) for n in z.namelist()] == [CAPTIONED.encode()]


def test_zip_refuses_rather_than_include_subtitles():
    assert _zip(CAPTIONED, "?subtitles=0", files=(CAPTIONED,)).status_code == 409


def test_restart_keeps_the_choice():
    saved = app_module.OUTPUT_DIR, app_module._enqueue_job
    with tempfile.TemporaryDirectory() as tmp:
        os.mkdir(os.path.join(tmp, JOB))
        app_module.OUTPUT_DIR, app_module._enqueue_job = tmp, lambda *a, **k: None
        try:
            app_module._write_resume_manifest(
                JOB, ["true"], 2, None, None, False,
                env={"AUTO_CAPTIONS": "0", "AUTO_HOOK": "1", "GEMINI_API_KEY": "must-not-be-stored"})
            assert "must-not-be-stored" not in open(os.path.join(tmp, JOB, app_module._RESUME_FILE)).read()
            # As after a crash: the previous instance's heartbeat is gone.
            app_module._resume_interrupted_jobs()
            env = app_module.jobs[JOB]["env"]
            assert env["AUTO_CAPTIONS"] == "0" and env["AUTO_HOOK"] == "1"
        finally:
            app_module.OUTPUT_DIR, app_module._enqueue_job = saved
            app_module.jobs.pop(JOB, None)


if __name__ == "__main__":
    failed = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            try:
                fn()
                print("ok  ", name)
            except Exception as e:
                failed += 1
                print("FAIL", name, repr(e)[:260])
    sys.exit(1 if failed else 0)
