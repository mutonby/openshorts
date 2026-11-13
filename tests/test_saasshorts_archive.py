"""A finished AI Short (SaaSShorts) lands in the owner's library.

Its working dir is swept with every other job dir after JOB_RETENTION_SECONDS
and its state lives only in ``saas_jobs``, so the video used to vanish an hour
after rendering (or at the next deploy) and never showed in History. The real
``/api/saasshorts/generate`` path runs here, with the renderer, R2 and the
database as doubles: the final MP4 must be uploaded under the owner's prefix
and recorded as a ``UserVideo`` row, which is what ``/api/history`` lists.
"""
import asyncio
import os
import uuid
from types import SimpleNamespace

import pytest

app = pytest.importorskip("app")
cloud = pytest.importorskip("cloud")
from cloud import database, storage, videos  # noqa: E402
from cloud.models import UserVideo  # noqa: E402

OWNER = uuid.UUID("a11ce000-0000-0000-0000-000000000007")


class _Result:
    def scalars(self):
        return iter(())

    def scalar_one_or_none(self):
        return None


class _Session:
    def __init__(self, added):
        self.added = added

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    def begin(self):
        return self

    async def execute(self, stmt):
        return _Result()

    def add(self, row):
        self.added.append(row)


@pytest.fixture
def world(tmp_path, monkeypatch):
    added, uploads = [], []
    monkeypatch.setattr(app, "OUTPUT_DIR", str(tmp_path))
    monkeypatch.setattr(app, "BILLING_ENABLED", True)
    monkeypatch.setattr(app, "cloud", cloud)
    monkeypatch.setattr(app, "saas_jobs", {})
    monkeypatch.setattr(app, "concurrency_semaphore", asyncio.Semaphore(1))

    async def entitled(request):
        return SimpleNamespace(id=OWNER)

    async def owner(request):
        return OWNER

    monkeypatch.setattr(app, "require_managed_entitlement", entitled)
    monkeypatch.setattr(app, "_owner_id", owner)

    def render(script, config, out_dir, log):
        path = os.path.join(out_dir, "calm_response_final.mp4")
        with open(path, "wb") as f:
            f.write(b"finished short")
        return {"video_filename": "calm_response_final.mp4", "video_path": path,
                "duration": 20.9}

    monkeypatch.setattr(app, "generate_full_video", render)
    monkeypatch.setattr(type(videos.settings), "r2_configured", property(lambda s: True))
    monkeypatch.setattr(storage, "upload_file",
                        lambda local, key, content_type="video/mp4": uploads.append((local, key)))
    monkeypatch.setattr(database, "session", lambda: _Session(added))
    return tmp_path, added, uploads


async def _generate():
    req = app.SaaSGenerateRequest(script={"title": "Calm response to disrespect"})
    out = await app.saasshorts_generate(req, request=SimpleNamespace(),
                                        x_fal_key="fal", x_elevenlabs_key="el")
    job_id = out["job_id"]
    for _ in range(200):
        if app.saas_jobs[job_id]["status"] != "processing":
            break
        await asyncio.sleep(0.01)
    # The archive runs right after the status flips: let it finish.
    await asyncio.sleep(0.05)
    return job_id


def test_finished_ai_short_is_archived_to_history(world):
    root, added, uploads = world
    job_id = asyncio.run(_generate())

    assert app.saas_jobs[job_id]["status"] == "completed"
    key = f"users/{OWNER}/{job_id}/calm_response_final.mp4"
    assert uploads == [(str(root / f"saas_{job_id}" / "calm_response_final.mp4"), key)]
    rows = [r for r in added if isinstance(r, UserVideo)]
    assert len(rows) == 1
    assert rows[0].user_id == OWNER
    assert rows[0].job_id == job_id
    assert rows[0].r2_key == key
    assert rows[0].title == "Calm response to disrespect"


def test_self_host_does_not_archive(world, monkeypatch):
    _root, added, uploads = world
    monkeypatch.setattr(app, "BILLING_ENABLED", False)
    job_id = asyncio.run(_generate())

    assert app.saas_jobs[job_id]["status"] == "completed"
    assert uploads == [] and added == []


def test_archive_failure_keeps_the_job_completed(world, monkeypatch):
    _root, added, _uploads = world

    def broken(*a, **k):
        raise RuntimeError("r2 down")

    monkeypatch.setattr(storage, "upload_file", broken)
    job_id = asyncio.run(_generate())

    assert app.saas_jobs[job_id]["status"] == "completed"
    assert app.saas_jobs[job_id]["result"]["video_url"].endswith("calm_response_final.mp4")
    assert added == []
