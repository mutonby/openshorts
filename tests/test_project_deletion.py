"""Deleting a project is scoped, recoverable, and excludes active work."""
import asyncio
from contextlib import asynccontextmanager
from types import SimpleNamespace
from uuid import uuid4

import httpx
import pytest
from fastapi import HTTPException

import app as app_module
import project_files


@pytest.fixture
def workspace(monkeypatch, tmp_path):
    output, uploads = tmp_path / "output", tmp_path / "uploads"
    output.mkdir()
    uploads.mkdir()
    job_id, sibling_id = str(uuid4()), str(uuid4())
    for value in (job_id, sibling_id):
        directory = output / value
        directory.mkdir()
        (directory / "job_metadata.json").write_text('{"shorts": []}')
        (directory / "clip.mp4").touch()
        (uploads / f"{value}_source.mp4").touch()
    monkeypatch.setattr(app_module, "BILLING_ENABLED", False)
    monkeypatch.setattr(app_module, "OUTPUT_DIR", str(output))
    monkeypatch.setattr(app_module, "UPLOAD_DIR", str(uploads))
    monkeypatch.setattr(app_module, "_running_jobs", set())
    monkeypatch.setattr(app_module, "_delivering_jobs", set())
    monkeypatch.setattr(app_module, "_rerender_locks", {})
    monkeypatch.setattr(app_module, "jobs", {
        value: {"status": "completed", "logs": [], "result": {"clips": []}}
        for value in (job_id, sibling_id)
    })
    return output, uploads, job_id, sibling_id


async def _request(method, path):
    transport = httpx.ASGITransport(app=app_module.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        return await client.request(method, path)


def test_self_host_removes_only_the_project_and_its_source(workspace):
    output, uploads, job_id, sibling_id = workspace
    stale_restore = output / f"{job_id}.restoring"
    stale_restore.mkdir()
    (stale_restore / "clip.mp4").touch()
    path = f"/api/projects/{job_id}"

    async def run():
        result = await _request("DELETE", path)
        assert result.status_code == 200
        assert result.json() == {"deleted": True, "job_id": job_id}
        assert (await _request("DELETE", path)).status_code == 200
        assert (await _request("GET", f"/api/status/{job_id}")).status_code == 404
        assert (await _request("GET", f"/api/source/{job_id}")).status_code == 404

    asyncio.run(run())
    assert job_id not in app_module.jobs
    assert not (output / job_id).exists()
    assert not (uploads / f"{job_id}_source.mp4").exists()
    assert not stale_restore.exists()
    assert (output / sibling_id / "clip.mp4").is_file()
    assert (uploads / f"{sibling_id}_source.mp4").is_file()


def test_processing_or_a_project_lease_blocks_deletion(workspace):
    output, _uploads, job_id, _sibling = workspace
    path = f"/api/projects/{job_id}"

    async def run():
        app_module.jobs[job_id]["status"] = "processing"
        assert (await _request("DELETE", path)).status_code == 409
        app_module.jobs[job_id]["status"] = "completed"
        with project_files.lock(output, job_id):
            assert (await _request("DELETE", path)).status_code == 409
        assert (output / job_id / "clip.mp4").is_file()

    asyncio.run(run())
    assert not project_files.deleted(output, job_id)


def test_managed_storage_failure_leaves_a_retryable_deletion(workspace, monkeypatch):
    from cloud import project_deletion

    output, _uploads, job_id, sibling_id = workspace
    owner_id = uuid4()
    app_module.jobs[job_id]["user_id"] = owner_id
    monkeypatch.setattr(app_module, "BILLING_ENABLED", True)
    monkeypatch.setattr(
        app_module, "_user_from_request", _async_result(SimpleNamespace(id=owner_id)),
    )
    events = []

    async def owner(user_id, key, local_owner):
        assert (user_id, key, local_owner) == (owner_id, job_id, owner_id)

    async def begin(*args):
        await owner(*args)
        events.append("begin")
        return False, True

    async def purge(user_id, key, archived):
        assert (user_id, key, archived) == (owner_id, job_id, True)
        assert project_files.deleted(output, job_id)
        events.append("purge")
        if events.count("purge") == 1:
            raise OSError("partial storage failure")

    async def finish(user_id, key):
        assert (user_id, key) == (owner_id, job_id)
        assert not (output / job_id).exists()
        events.append("finish")

    monkeypatch.setattr(project_deletion, "require_owner", owner)
    monkeypatch.setattr(project_deletion, "begin", begin)
    monkeypatch.setattr(project_deletion, "purge_archive", purge)
    monkeypatch.setattr(project_deletion, "finish", finish)

    async def run():
        path = f"/api/projects/{job_id}"
        failed = await _request("DELETE", path)
        assert failed.status_code == 503
        assert (output / job_id / "clip.mp4").is_file()
        assert (await _request("GET", f"/api/status/{job_id}")).status_code == 404
        assert (await _request("DELETE", path)).status_code == 200

    asyncio.run(run())
    assert events == ["begin", "purge", "begin", "purge", "finish"]
    assert (output / sibling_id / "clip.mp4").is_file()


def _async_result(value):
    async def result(*_args):
        return value
    return result


def test_cloud_deletion_rejects_the_wrong_owner_before_marking(workspace, monkeypatch):
    from cloud import project_deletion

    output, _uploads, job_id, _sibling = workspace
    caller, actual_owner = uuid4(), uuid4()
    monkeypatch.setattr(app_module, "BILLING_ENABLED", True)
    monkeypatch.setattr(app_module, "_user_from_request", _async_result(SimpleNamespace(id=caller)))

    class Session:
        async def execute(self, _statement):
            return SimpleNamespace(scalars=lambda: [actual_owner])

        async def get(self, _model, _key):
            return None

    @asynccontextmanager
    async def session():
        yield Session()

    monkeypatch.setattr(project_deletion.database, "session", session)
    result = asyncio.run(_request("DELETE", f"/api/projects/{job_id}"))
    assert result.status_code == 404
    assert (output / job_id / "clip.mp4").is_file()
    assert not project_files.deleted(output, job_id)


def test_tombstone_prevents_restore_of_a_stale_project(workspace, monkeypatch):
    from cloud import project_deletion

    output, _uploads, job_id, _sibling = workspace
    monkeypatch.setattr(app_module, "BILLING_ENABLED", True)
    monkeypatch.setattr(project_deletion, "is_deleted", _async_result(True))
    with pytest.raises(HTTPException) as raised:
        asyncio.run(app_module._restore_job_files(job_id, SimpleNamespace(job_id=job_id), "owner"))
    assert raised.value.status_code == 404
    assert (output / job_id / "clip.mp4").is_file()


def test_storage_reports_partial_object_deletes_as_failures(monkeypatch):
    from cloud import storage

    prefix = "users/owner/project/"
    request = []

    def remove(**kwargs):
        request.append(kwargs["Delete"]["Objects"])
        return {"Errors": [{"Code": "AccessDenied", "Key": f"{prefix}clip.mp4"}]}

    monkeypatch.setattr(storage, "client", lambda: SimpleNamespace(
        list_objects_v2=lambda **kwargs: {"Contents": [{"Key": f"{prefix}clip.mp4"}]},
        delete_objects=remove,
    ))
    with pytest.raises(OSError, match="refused"):
        storage.delete_prefix(prefix, strict=True)
    assert request == [[{"Key": f"{prefix}clip.mp4"}]]
