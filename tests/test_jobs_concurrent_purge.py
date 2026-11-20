"""Walks over `jobs` that run on the event loop while an account erasure pops
records from it in a worker thread (``_purge_local_jobs_for_user`` runs under
``asyncio.to_thread``). A Python-level loop over the live dict can be switched
out mid-walk and then raises "dictionary changed size during iteration"; a
snapshot (``list(...)``, built in one C call) cannot.

The thread is simulated deterministically: one record pops another out of
`jobs` the moment it is read, as the purge would between two steps of the walk.
"""
import asyncio
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

app_module = pytest.importorskip("app")


class _PurgedMidWalk(dict):
    """A job record whose first read pops `victim` from `jobs`."""

    def __init__(self, victim, **kw):
        super().__init__(**kw)
        self._victim = victim

    def get(self, key, default=None):
        app_module.jobs.pop(self._victim, None)
        return super().get(key, default)


@pytest.fixture
def jobs(monkeypatch):
    table = {
        "a": _PurgedMidWalk("z", user_id="u1", status="queued"),
        "m": {"user_id": "u1", "status": "processing"},
        "z": {"user_id": "gone", "status": "queued"},
    }
    monkeypatch.setattr(app_module, "jobs", table)
    return table


def test_job_limit_count_survives_concurrent_purge(jobs, monkeypatch):
    user = SimpleNamespace(id="u1", plan="free")

    async def from_request(request):
        return user

    monkeypatch.setattr(app_module, "BILLING_ENABLED", True)
    monkeypatch.setattr(app_module, "_user_from_request", from_request)
    monkeypatch.setattr(app_module, "managed_keys",
                        SimpleNamespace(has_active_entitlement=lambda u: True))
    monkeypatch.setattr(app_module, "_submissions", {})
    monkeypatch.setattr(app_module, "_cloud_config", SimpleNamespace(
        PLAN_PRIORITY={"free": 2}, PLAN_JOB_LIMIT={"free": 2}))

    with pytest.raises(HTTPException) as exc:
        asyncio.run(app_module.reserve_process_minutes(
            SimpleNamespace(), None, "/tmp/x.mp4", "new-job"))
    assert exc.value.status_code == 429  # both of u1's jobs counted, no 500
    assert "z" not in app_module.jobs


def test_raise_users_jobs_survives_concurrent_purge(jobs, monkeypatch):
    monkeypatch.setattr(app_module, "job_queue", asyncio.PriorityQueue())
    monkeypatch.setattr(app_module, "_running_jobs", set())
    monkeypatch.setattr(app_module, "_rewrite_manifest_priorities",
                        lambda want, running, raised_at: set())

    assert asyncio.run(app_module._raise_users_jobs({"u1": 1})) == 0
    assert "z" not in app_module.jobs
