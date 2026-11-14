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
