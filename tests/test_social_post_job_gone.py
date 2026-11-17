"""POST /api/social/post when the job is dropped (cleanup, account delete)
while the Upload-Post key is being resolved: a 404, never a KeyError 500."""
import asyncio

import httpx
import pytest

app_module = pytest.importorskip("app")


def test_job_dropped_during_key_resolution_is_a_404(monkeypatch):
    job_id = "job-gone-mid-request"
    app_module.jobs[job_id] = {"status": "completed", "result": {"clips": []}}

    async def fake_ensure(jid, request):
        return True

    async def fake_resolve(request, body_key=None):
        app_module.jobs.pop(job_id, None)  # cleanup runs during this await
        return "key", None

    monkeypatch.setattr(app_module, "_ensure_job_files", fake_ensure)
    monkeypatch.setattr(app_module, "resolve_upload_post", fake_resolve)

    async def _post():
        transport = httpx.ASGITransport(app=app_module.app, raise_app_exceptions=False)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as c:
            return await c.post("/api/social/post", json={
                "job_id": job_id, "clip_index": 0, "platforms": ["tiktok"], "user_id": "u"})

    try:
        resp = asyncio.run(_post())
    finally:
        app_module.jobs.pop(job_id, None)
    assert resp.status_code == 404
    assert resp.json()["detail"] == "Job not found"


def test_a_clip_pointing_at_a_file_never_written_posts_the_one_it_has(tmp_path, monkeypatch):
    """A failed edit used to record a name it never wrote; the post then died
    on "Video file not found" although the clip's real file was right there."""
    job_id = "job-with-a-stale-url"
    job_dir = tmp_path / job_id
    job_dir.mkdir()
    (job_dir / "t_metadata.json").write_text("{}")
    (job_dir / "t_clip_1.mp4").write_bytes(b"clean")
    (job_dir / "subtitled_5_t_clip_1.mp4").write_bytes(b"captioned")
    clip = {"video_url": f"/videos/{job_id}/edited_t_clip_1.mp4", "title": "t"}
    app_module.jobs[job_id] = {"status": "completed", "result": {"clips": [clip]}}
    monkeypatch.setattr(app_module, "OUTPUT_DIR", str(tmp_path))
    sent = {}

    async def fake_ensure(jid, request):
        return True

    async def fake_resolve(request, body_key=None):
        return "key", None

    class Accepted:
        status_code = 200
        text = "{}"

        def json(self):
            return {"ok": True}

    def fake_upload(url, headers, data, file_path, filename, timeout):
        sent.update(path=file_path, name=filename)
        return Accepted()

    monkeypatch.setattr(app_module, "_ensure_job_files", fake_ensure)
    monkeypatch.setattr(app_module, "resolve_upload_post", fake_resolve)
    monkeypatch.setattr(app_module, "_post_video_blocking", fake_upload)

    async def _post():
        transport = httpx.ASGITransport(app=app_module.app, raise_app_exceptions=False)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as c:
            return await c.post("/api/social/post", json={
                "job_id": job_id, "clip_index": 0, "platforms": ["tiktok"], "user_id": "u"})

    try:
        resp = asyncio.run(_post())
    finally:
        app_module.jobs.pop(job_id, None)
    assert resp.status_code == 200, resp.text
    assert sent["name"] == "subtitled_5_t_clip_1.mp4"
    # The clip is pointed back at a file that exists.
    assert clip["video_url"] == f"/videos/{job_id}/subtitled_5_t_clip_1.mp4"
