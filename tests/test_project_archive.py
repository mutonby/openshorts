"""A published project revision must reference one complete durable snapshot."""
import asyncio
from contextlib import asynccontextmanager
import json
from types import SimpleNamespace
from uuid import uuid4

import pytest

from cloud import videos


@pytest.fixture
def archive(monkeypatch, tmp_path):
    objects, commits = {}, []
    monkeypatch.setattr(videos, "settings", SimpleNamespace(r2_configured=True))
    monkeypatch.setattr(videos.storage, "object_size", lambda key: len(objects[key]) if key in objects else None)
    monkeypatch.setattr(videos.storage, "upload_fileobj", lambda handle, key, _kind: objects.__setitem__(key, handle.read()))
    async def publish(*args):
        commits.append(args)
        return True
    monkeypatch.setattr(videos, "_publish_archive_revision", publish)
    monkeypatch.setattr(videos.media_redirect, "forget", lambda _job: None)
    return tmp_path, objects, commits


def snapshot(job, names):
    return json.dumps({"shorts": [{"start": 0, "end": 2, "video_url": f"/videos/{job}/{name}"} for name in names]}).encode()


def test_captured_snapshot_archives_every_clip_and_card_body(archive):
    directory, objects, commits = archive
    names = ["endcard_deadbeef_subtitled_1_video_clip_1.mp4", "video_clip_2.mp4"]
    for name in [*names, "subtitled_1_video_clip_1.mp4", "video_clip_1.mp4"]:
        (directory / name).write_bytes(name.encode())
    frozen = snapshot("job", names)
    # A later edit replaced the shared path before this upload began.
    (directory / "video_metadata.json").write_bytes(snapshot("job", ["wrong.mp4"]))
    asyncio.run(videos.archive_clip_edit("owner", "job", 0, directory, names[0],
        metadata_name="video_metadata.json", metadata_snapshot=frozen, revision=12))
    metadata_key = commits[0][2]
    assert objects[metadata_key] == frozen
    assert len(commits[0][4]) == 2
    assert objects["users/owner/job/subtitled_1_video_clip_1.mp4"]
    assert objects["users/owner/job/video_clip_1.mp4"]


def test_missing_body_or_upload_failure_cannot_publish(archive, monkeypatch):
    directory, objects, commits = archive
    name = "endcard_deadbeef_video_clip_1.mp4"
    (directory / name).write_bytes(b"card")
    async def run():
        await videos.archive_clip_edit("owner", "job", 0, directory, name,
            metadata_name="video_metadata.json", metadata_snapshot=snapshot("job", [name]), revision=12)
    with pytest.raises(FileNotFoundError):
        asyncio.run(run())
    assert not commits
    (directory / "video_clip_1.mp4").write_bytes(b"body")
    def fail(*_args):
        raise OSError("storage failure")
    monkeypatch.setattr(videos.storage, "upload_fileobj", fail)
    with pytest.raises(OSError, match="storage failure"):
        asyncio.run(run())
    assert not commits


def test_new_revision_preserves_old_snapshot_media(archive):
    directory, objects, commits = archive
    (directory / "video_clip_1.mp4").write_bytes(b"body")
    for revision in (12, 13):
        name = f"subtitled_{revision}_video_clip_1.mp4"
        (directory / name).write_bytes(str(revision).encode())
        asyncio.run(videos.archive_clip_edit("owner", "job", 0, directory, name,
            metadata_name="video_metadata.json", metadata_snapshot=snapshot("job", [name]), revision=revision))
    assert commits[0][2] != commits[1][2]
    assert objects["users/owner/job/subtitled_12_video_clip_1.mp4"] == b"12"
    assert objects[commits[0][2]]


def test_lazy_clean_twin_keeps_complete_recipe_without_encoding(archive):
    import clean_twin
    directory, objects, commits = archive
    served = "wm_subtitled_12_video_clip_1.mp4"
    (directory / served).write_bytes(b"marked")
    canonical = directory / "video_clip_1.mp4"
    canonical.write_bytes(b"body")
    captions = directory / "captions.ass"
    captions.write_text("[Script Info]\n")
    clean = "subtitled_12_video_clip_1.mp4"
    clean_twin.write_recipe(str(directory / clean), str(canonical), str(captions))
    def run(revision):
        return asyncio.run(videos.archive_clip_edit("owner", "job", 0, directory, served,
            metadata_name="video_metadata.json", metadata_snapshot=snapshot("job", [served]), revision=revision))
    run(12)
    assert len(commits) == 1
    assert "users/owner/job/" + clean not in objects
    json_path, ass_path = clean_twin.recipe_paths(directory, clean)
    from pathlib import Path
    assert objects["users/owner/job/" + Path(json_path).name]
    assert objects["users/owner/job/" + Path(ass_path).name]
    # Even if the first snapshot exists, a new incomplete local recipe cannot
    # be published into storage that does not already contain its inputs.
    objects.clear()
    Path(ass_path).unlink()
    with pytest.raises(FileNotFoundError, match="recipe"):
        run(13)
    assert len(commits) == 1


def test_stale_archive_cannot_repoint_history_and_browser_state_survives(monkeypatch):
    project = SimpleNamespace(user_id="owner", state={"v": 1, "clips": [{"index": 0,
        "original_file": "original.mp4", "server_file": "old.mp4", "active_layers": {"hook": "saved"}}]},
        metadata_r2_key="old-meta", size_bytes=1)
    video = SimpleNamespace(clip_index=0, r2_key="old.mp4", title="old", size_bytes=1)
    class Result:
        def __init__(self, value): self.value = value
        def scalar_one_or_none(self): return self.value
        def scalars(self): return self.value
    class Session:
        @asynccontextmanager
        async def begin(self): yield self
        async def execute(self, statement):
            if statement.column_descriptions[0]["entity"] is videos.Project:
                assert statement._for_update_arg is not None
                return Result(project)
            return Result([video])
    @asynccontextmanager
    async def session(): yield Session()
    monkeypatch.setattr(videos.database, "session", session)
    async def run():
        assert await videos._publish_archive_revision("owner", "job", "new-meta", 13,
            [(0, "new.mp4", "new-key", "new", 4)], None)
        assert not await videos._publish_archive_revision("owner", "job", "stale-meta", 12,
            [(0, "stale.mp4", "stale-key", "stale", 5)], None)
    asyncio.run(run())
    assert project.metadata_r2_key == "new-meta" and video.r2_key == "new-key"
    assert project.state["clips"][0] == {"index": 0, "original_file": "original.mp4", "server_file": "new.mp4", "active_layers": {"hook": "saved"}}


def test_restore_uses_recorded_metadata_key_when_basenames_collide(monkeypatch, tmp_path):
    import app
    job = str(uuid4())
    prefix = f"users/owner/{job}/"
    current = prefix + "metadata/13-new/video_metadata.json"
    objects = {prefix + "metadata/12-old/video_metadata.json": snapshot(job, ["old.mp4"]),
        current: snapshot(job, ["video_clip_1.mp4"]), prefix + "video_clip_1.mp4": b"video"}
    pulled = []
    def download(key, path):
        pulled.append(key)
        from pathlib import Path
        Path(path).write_bytes(objects[key])
    monkeypatch.setattr(app, "OUTPUT_DIR", str(tmp_path))
    monkeypatch.setattr(app, "jobs", {})
    monkeypatch.setattr(app, "_restore_locks", {})
    monkeypatch.setattr(videos.storage, "list_keys", lambda _prefix: list(objects))
    monkeypatch.setattr(videos.storage, "download_file", download)
    project = SimpleNamespace(metadata_r2_key=current, state={"clips": [{"server_file": "video_clip_1.mp4"}]})
    assert asyncio.run(app._restore_project_files(job, project, "owner"))
    assert prefix + "metadata/12-old/video_metadata.json" not in pulled
    assert app.jobs[job]["result"]["clips"][0]["video_url"].endswith("video_clip_1.mp4")


@pytest.fixture
def restore_store(monkeypatch, tmp_path):
    import app
    job = str(uuid4())
    prefix = f"users/owner/{job}/"
    objects = {prefix + "video_clip_1.mp4": b"body"}
    def download(key, path):
        from pathlib import Path
        Path(path).write_bytes(objects[key])
    monkeypatch.setattr(app, "OUTPUT_DIR", str(tmp_path))
    monkeypatch.setattr(app, "jobs", {})
    monkeypatch.setattr(app, "_restore_locks", {})
    monkeypatch.setattr(app, "_rerender_locks", {})
    monkeypatch.setattr(videos.storage, "list_keys", lambda _prefix: list(objects))
    monkeypatch.setattr(videos.storage, "download_file", download)
    return app, job, prefix, objects, tmp_path / job


def test_upgrade_pointer_survives_restore_and_metadata_reload(restore_store):
    app, job, prefix, objects, directory = restore_store
    clean = "subtitled_1_video_clip_1.mp4"
    marked = "wm_" + clean
    objects[prefix + clean] = b"clean"
    objects[prefix + marked] = b"marked"
    key = prefix + "metadata/12/video_metadata.json"
    objects[key] = snapshot(job, [marked])
    project = SimpleNamespace(metadata_r2_key=key, state={"archive_revision": "12", "clips": [
        {"index": 0, "original_file": clean, "server_file": clean}]})
    asyncio.run(app._restore_project_files(job, project, "owner"))
    assert app.jobs[job]["result"]["clips"][0]["video_url"].endswith("/" + clean)
    assert json.loads((directory / "video_metadata.json").read_text())["shorts"][0]["video_url"].endswith("/" + clean)
    assert not (directory / ".marked").exists()


def test_warm_restore_after_upgrade_opens_clean_delivery_guard(restore_store):
    app, job, prefix, objects, directory = restore_store
    key = prefix + "metadata/12/video_metadata.json"
    marked = "wm_video_clip_1.mp4"
    objects[key] = snapshot(job, [marked])
    objects[prefix + marked] = b"marked"
    project = SimpleNamespace(metadata_r2_key=key, state={"clips": [{"index": 0, "server_file": marked}]})
    asyncio.run(app._restore_project_files(job, project, "owner"))
    assert (directory / ".marked").exists()
    project.state["clips"][0]["server_file"] = "video_clip_1.mp4"
    asyncio.run(app._restore_project_files(job, project, "owner"))
    assert not (directory / ".marked").exists()
    assert app._media_guard(f"{job}/video_clip_1.mp4")


def test_restored_results_keep_source_ids_when_an_earlier_clip_failed(restore_store):
    app, job, prefix, objects, _directory = restore_store
    key = prefix + "video_metadata.json"
    data = json.loads(snapshot(job, ["video_clip_1.mp4", "video_clip_2.mp4"]))
    data["shorts"][0]["quality"] = {"valid": False}
    objects[key] = json.dumps(data).encode()
    objects[prefix + "video_clip_2.mp4"] = b"second"
    project = SimpleNamespace(metadata_r2_key=key, state={"clips": []})
    asyncio.run(app._restore_project_files(job, project, "owner"))
    assert [clip["clip_index"] for clip in app.jobs[job]["result"]["clips"]] == [1]


def test_same_video_filename_does_not_hide_new_metadata_revision(restore_store):
    app, job, prefix, objects, directory = restore_store
    old = prefix + "metadata/12/video_metadata.json"
    new = prefix + "metadata/13/video_metadata.json"
    def content(credit):
        data = json.loads(snapshot(job, ["video_clip_1.mp4"]))
        data["publishing"] = {"source_credit_url": credit}
        return json.dumps(data).encode()
    objects[old] = content("https://example.com/old")
    project = SimpleNamespace(metadata_r2_key=old, state={"clips": [{"index": 0, "server_file": "video_clip_1.mp4"}]})
    asyncio.run(app._restore_project_files(job, project, "owner"))
    objects[new] = content("https://example.com/new")
    project.metadata_r2_key = new
    assert asyncio.run(app._restore_project_files(job, project, "owner"))
    assert json.loads((directory / "video_metadata.json").read_text())["publishing"]["source_credit_url"] == "https://example.com/new"


def test_restore_preserves_local_edit_while_archive_upload_is_pending(restore_store):
    app, job, prefix, objects, directory = restore_store
    key = prefix + "metadata/12/video_metadata.json"
    objects[key] = snapshot(job, ["video_clip_1.mp4"])
    project = SimpleNamespace(metadata_r2_key=key, state={"clips": [{"index": 0, "server_file": "video_clip_1.mp4"}]})
    asyncio.run(app._restore_project_files(job, project, "owner"))
    local = json.loads((directory / "video_metadata.json").read_text())
    local["publishing"] = {"source_credit_url": "https://example.com/pending"}
    (directory / "video_metadata.json").write_text(json.dumps(local))
    project.metadata_r2_key = prefix + "metadata/13/video_metadata.json"
    objects[project.metadata_r2_key] = snapshot(job, ["video_clip_1.mp4"])
    assert not asyncio.run(app._restore_project_files(job, project, "owner"))
    assert json.loads((directory / "video_metadata.json").read_text()) == local


def test_deleted_asset_stays_hidden_in_current_restore_but_old_snapshot_retains_it(restore_store):
    app, job, prefix, objects, directory = restore_store
    name = "broll_asset_" + "a" * 32 + ".mp4"
    key = prefix + "metadata/12/video_metadata.json"
    data = json.loads(snapshot(job, ["video_clip_1.mp4"]))
    data["deleted_editor_assets"] = [name, "../outside.mp4"]
    objects[key] = json.dumps(data).encode()
    objects[prefix + name] = b"older snapshot media"
    project = SimpleNamespace(metadata_r2_key=key, state={"clips": []})
    asyncio.run(app._restore_project_files(job, project, "owner"))
    assert not (directory / name).exists()
    assert objects[prefix + name] == b"older snapshot media"


def test_failed_restore_drains_downloads_before_releasing_its_directory(restore_store, monkeypatch):
    import threading
    app, job, prefix, objects, directory = restore_store
    key = prefix + "video_metadata.json"
    objects[key] = snapshot(job, ["video_clip_1.mp4"])
    started, failed, release = threading.Event(), threading.Event(), threading.Event()
    def download(key, path):
        if key.endswith(".mp4"):
            started.set()
            assert release.wait(3)
            from pathlib import Path
            Path(path).write_bytes(objects[key])
        else:
            assert started.wait(3)
            failed.set()
            raise OSError("simulated download failure")
    monkeypatch.setattr(videos.storage, "download_file", download)
    project = SimpleNamespace(metadata_r2_key=key, state={"clips": []})
    async def run():
        task = asyncio.create_task(app._restore_project_files(job, project, "owner"))
        try:
            assert await asyncio.to_thread(failed.wait, 3)
            await asyncio.sleep(0.02)
            assert not task.done()
        finally:
            release.set()
        with pytest.raises(Exception, match="502"):
            await task
    asyncio.run(run())
    assert not directory.exists() and not directory.with_name(directory.name + ".restoring").exists()


def test_archive_finishing_after_upgrade_reconciles_marked_snapshot(restore_store, monkeypatch):
    app, job, _prefix, _objects, directory = restore_store
    directory.mkdir()
    marked = "wm_video_clip_1.mp4"
    (directory / "video_metadata.json").write_bytes(snapshot(job, [marked]))
    # The upgrade has already removed .marked, before this upload finished.
    app.jobs[job] = {"user_id": "owner", "status": "completed"}
    monkeypatch.setattr(app, "BILLING_ENABLED", True)
    monkeypatch.setattr(app, "cloud", SimpleNamespace(videos=videos))
    async def paid(_owners):
        return {"owner"}
    monkeypatch.setattr(app, "_owner_live_plans", paid)
    calls = []
    async def run():
        finished = asyncio.Event()
        async def archive(*_args, **_kwargs):
            calls.append("archive-marked")
        async def unmark(owner, job_id):
            assert (owner, job_id) == ("owner", job)
            calls.append("unmark-paid")
            finished.set()
        monkeypatch.setattr(app.cloud.videos, "archive_clip_edit", archive)
        monkeypatch.setattr(app.cloud.videos, "unmark_user_library", unmark)
        app._archive_clip_edit_bg(job, 0, marked)
        await asyncio.wait_for(finished.wait(), 2)
    asyncio.run(run())
    assert calls == ["archive-marked", "unmark-paid"]
