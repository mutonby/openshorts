"""The hourly sweeps must never delete a job that is still queued or running.

22-sep-2026: with the queue backed up, free jobs waited over an hour and the
age-only sweep removed their dir (and source upload) under the worker. The
user got no clips and the minute reservation stayed held for 3 hours.
"""
import asyncio
import os
import time
import types

import pytest

app_module = pytest.importorskip("app")

OLD = 2 * 3600  # older than JOB_RETENTION_SECONDS in cloud (1 h)


@pytest.fixture
def disk(tmp_path, monkeypatch):
    out = tmp_path / "output"
    up = tmp_path / "uploads"
    out.mkdir()
    up.mkdir()
    monkeypatch.setattr(app_module, "OUTPUT_DIR", str(out))
    monkeypatch.setattr(app_module, "UPLOAD_DIR", str(up))
    monkeypatch.setattr(app_module, "THUMBNAILS_DIR", str(out / "thumbnails"))
    monkeypatch.setattr(app_module, "JOB_RETENTION_SECONDS", 3600)
    monkeypatch.setattr(app_module, "_running_jobs", set())
    monkeypatch.setattr(app_module, "jobs", {})
    return out, up


def _age(path, seconds):
    t = time.time() - seconds
    os.utime(path, (t, t))


def _job_dir(out, job_id, manifest=False, age=OLD):
    d = out / job_id
    d.mkdir()
    if manifest:
        m = d / app_module._RESUME_FILE
        m.write_text("{}")
        _age(m, age)
    _age(d, age)
    return d


def _upload(up, job_id, age=OLD):
    f = up / f"{job_id}_video.mp4"
    f.write_bytes(b"x")
    _age(f, age)
    return f


class TestJobDirSweep:
    def test_finished_old_job_is_purged(self, disk):
        out, _ = disk
        d = _job_dir(out, "done")
        app_module.jobs["done"] = {"status": "completed"}
        app_module._sweep_expired_job_dirs(time.time())
        assert not d.exists()
        assert "done" not in app_module.jobs

    def test_queued_old_job_survives(self, disk):
        out, _ = disk
        d = _job_dir(out, "waiting")
        app_module.jobs["waiting"] = {"status": "queued"}
        app_module._sweep_expired_job_dirs(time.time())
        assert d.exists() and "waiting" in app_module.jobs

    def test_running_old_job_survives(self, disk):
        out, _ = disk
        d = _job_dir(out, "busy")
        app_module.jobs["busy"] = {"status": "processing"}
        app_module._running_jobs.add("busy")
        app_module._sweep_expired_job_dirs(time.time())
        assert d.exists() and "busy" in app_module.jobs

    def test_manifest_of_other_instance_survives(self, disk):
        out, _ = disk
        # Not in this instance's memory, but a fresh manifest says it is live.
        d = _job_dir(out, "handover", manifest=True, age=30)
        _age(d, OLD)
        app_module._sweep_expired_job_dirs(time.time())
        assert d.exists()

    def test_abandoned_manifest_does_not_pin_forever(self, disk):
        out, _ = disk
        d = _job_dir(out, "ghost", manifest=True,
                     age=app_module.ACTIVE_JOB_MAX_SECONDS + 60)
        app_module._sweep_expired_job_dirs(time.time())
        assert not d.exists()

    def test_size_cap_skips_live_jobs(self, disk, monkeypatch):
        out, _ = disk
        live = _job_dir(out, "live")
        (live / "clip.mp4").write_bytes(b"x" * 2048)
        app_module.jobs["live"] = {"status": "queued"}
        dead = _job_dir(out, "dead")
        (dead / "clip.mp4").write_bytes(b"x" * 2048)
        monkeypatch.setattr(app_module, "OUTPUT_MAX_GB", 1 / 1024 ** 3)  # 1 byte
        app_module._enforce_output_size_cap()
        assert live.exists() and not dead.exists()


class TestUploadSweep:
    def test_source_of_queued_job_survives(self, disk):
        _, up = disk
        f = _upload(up, "waiting")
        app_module.jobs["waiting"] = {"status": "queued"}
        app_module._sweep_expired_uploads(time.time())
        assert f.exists()

    def test_source_of_finished_job_is_removed(self, disk):
        _, up = disk
        f = _upload(up, "done")
        app_module.jobs["done"] = {"status": "failed"}
        app_module._sweep_expired_uploads(time.time())
        assert not f.exists()

    def test_size_cap_skips_live_sources(self, disk, monkeypatch):
        _, up = disk
        live = _upload(up, "live")
        app_module._running_jobs.add("live")
        dead = _upload(up, "dead")
        monkeypatch.setattr(app_module, "UPLOADS_MAX_GB", 1 / 1024 ** 3)
        app_module._enforce_uploads_size_cap()
        assert live.exists() and not dead.exists()


class TestSettleWithoutEntry:
    def test_reservation_released_when_entry_vanished(self, monkeypatch):
        calls = []

        class FakeMetering:
            @staticmethod
            async def commit_reservation(rid):
                calls.append(("commit", rid))

            @staticmethod
            async def release_reservation(rid):
                calls.append(("release", rid))

        monkeypatch.setattr(app_module, "BILLING_ENABLED", True)
        monkeypatch.setattr(app_module, "cloud", types.SimpleNamespace(metering=FakeMetering), raising=False)
        monkeypatch.setattr(app_module, "jobs", {})
        job = {"status": "processing", "reservation_id": "res-1"}
        asyncio.run(app_module._settle_reservation("gone", job))
        assert calls == [("release", "res-1")]


class TestRestoreSurvivesTheSizeCap:
    """16-nov-2026: with the disk over OUTPUT_MAX_GB every sweep purged every
    finished dir, a library restore included: mid-download ("Restore download
    failed: No such file or directory") and right after it, so the
    connector's get_job_status answered "Job not found" for a job the
    customer had in their library."""

    JOB = "8741b47a-ab74-43bd-ab88-ba4167d8fc4c"

    @pytest.fixture
    def r2(self, disk, monkeypatch):
        storage = pytest.importorskip("cloud.storage")
        out, _ = disk
        keys = [f"users/u/{self.JOB}/{n}" for n in ("v_metadata.json", "v_clip_0.mp4")]
        monkeypatch.setattr(app_module, "OUTPUT_MAX_GB", 1 / 1024 ** 3)  # always over
        monkeypatch.setattr(app_module, "_restore_locks", {})
        monkeypatch.setattr(app_module, "_restored_at", {})
        monkeypatch.setattr(storage, "list_keys", lambda prefix: keys)

        def download(key, local_path):
            # The cleanup task runs while the restore waits on R2.
            app_module._enforce_output_size_cap()
            body = (b'{"shorts": [{"start": 0, "end": 30}]}'
                    if key.endswith(".json") else b"x" * 4096)
            with open(local_path, "wb") as f:
                f.write(body)
        monkeypatch.setattr(storage, "download_file", download)
        return out

    def _restore(self):
        proj = types.SimpleNamespace(
            metadata_r2_key=f"users/u/{self.JOB}/v_metadata.json",
            state={"clips": [{"index": 0, "server_file": "v_clip_0.mp4"}]})
        return asyncio.run(app_module._restore_job_files(self.JOB, proj, "u"))

    def test_a_restore_is_not_purged_while_it_downloads_nor_right_after(self, r2):
        other = _job_dir(r2, "finished-long-ago")
        (other / "clip.mp4").write_bytes(b"x" * 2048)
        assert self._restore() is True
        job_dir = r2 / self.JOB
        assert (job_dir / "v_clip_0.mp4").exists()
        app_module._enforce_output_size_cap()      # the next sweep
        assert job_dir.exists() and app_module.jobs[self.JOB]["status"] == "completed"
        assert not other.exists()                  # the cap still trims the rest

    def test_after_the_grace_the_restored_dir_goes_like_any_other(self, r2):
        self._restore()
        app_module._restored_at[self.JOB] -= app_module.RESTORE_GRACE_SECONDS + 1
        app_module._enforce_output_size_cap()
        assert not (r2 / self.JOB).exists() and self.JOB not in app_module.jobs
