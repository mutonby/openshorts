"""The stuck-reservation sweeper must not refund videos that are still alive.

It used to release every ``reserved`` row older than STUCK_RESERVATION_HOURS.
With a queue longer than that, the reservation of a video still waiting (or
rendering) was refunded, the job delivered later and its commit found nothing
to commit: free minutes never counted. Now it skips reservations whose job is
queued, running or waiting in a resume manifest on the shared disk (both
instances of a deploy), and still frees a job alive past
LIVE_RESERVATION_MAX_HOURS (hung).
"""
import asyncio
import json
import os
import time

import pytest

app_module = pytest.importorskip("app")
metering = pytest.importorskip("cloud.metering")


@pytest.fixture
def out(tmp_path, monkeypatch):
    root = tmp_path / "output"
    root.mkdir()
    monkeypatch.setattr(app_module, "OUTPUT_DIR", str(root))
    monkeypatch.setattr(app_module, "INSTANCE_ID", "new")
    monkeypatch.setattr(app_module, "_running_jobs", set())
    monkeypatch.setattr(app_module, "jobs", {})
    return root


def _manifest(out, job_id, **extra):
    d = out / job_id
    d.mkdir(exist_ok=True)
    m = {"cmd": ["python", "main.py"], "priority": 2, "user_id": "u",
         "reservation_id": f"res-{job_id}", "watermark": False, "attempts": 0}
    m.update(extra)
    (d / app_module._RESUME_FILE).write_text(json.dumps(m))
    return d


class _Rows:
    def __init__(self, rows):
        self.rows = rows

    def all(self):
        return self.rows


class _Session:
    def __init__(self, rows):
        self.rows = rows

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def execute(self, stmt):
        from sqlalchemy.dialects import postgresql
        sql = str(stmt.compile(dialect=postgresql.dialect()))  # the real statement
        assert "FROM usage_ledger" in sql and "AS hung" in sql
        return _Rows(self.rows)


@pytest.fixture
def ledger(monkeypatch):
    """Rows the SQL returns (already older than the stuck window): id -> hung."""
    state = {"rows": [], "released": []}
    monkeypatch.setattr(metering.database, "session", lambda: _Session(state["rows"]))

    async def _release(lid):
        state["released"].append(lid)
    monkeypatch.setattr(metering, "release_reservation", _release)
    return state


def _sweep(live_ids=None):
    asyncio.run(metering._sweep_once(live_ids))


class TestLiveSet:
    def test_queued_running_and_pending_manifests_are_live(self, out):
        _manifest(out, "queued_here")
        app_module.jobs["queued_here"] = {"status": "queued", "reservation_id": "res-queued_here"}
        _manifest(out, "running_here", instance="new", heartbeat=time.time())
        app_module.jobs["running_here"] = {"status": "processing", "reservation_id": "res-running_here"}
        app_module._running_jobs.add("running_here")
        _manifest(out, "running_there", instance="old", heartbeat=time.time())
        _manifest(out, "left_by_drain")                       # queued, nobody holds it yet
        app_module.jobs["no_manifest"] = {"status": "queued", "reservation_id": "res-no_manifest"}
        app_module.jobs["done"] = {"status": "completed", "reservation_id": "res-done"}
        live = app_module._live_reservation_ids()
        assert live == {"res-queued_here", "res-running_here", "res-running_there",
                        "res-left_by_drain", "res-no_manifest"}

    def test_drain_cancelled_job_reading_failed_here_stays_live(self, out):
        # The old instance killed it at the drain timeout: its record may say
        # 'failed', but the manifest is kept and the next instance resumes it.
        _manifest(out, "cut", instance="old", heartbeat=time.time() - 600)
        app_module.jobs["cut"] = {"status": "failed", "reservation_id": "res-cut"}
        assert "res-cut" in app_module._live_reservation_ids()

    def test_manifest_past_active_job_max_is_not_live(self, out):
        d = _manifest(out, "ancient")
        old = time.time() - app_module.ACTIVE_JOB_MAX_SECONDS - 60
        os.utime(d / app_module._RESUME_FILE, (old, old))
        assert app_module._live_reservation_ids() == set()


class TestSweep:
    def test_live_reservations_are_kept_dead_ones_released(self, out, ledger):
        _manifest(out, "waiting")
        ledger["rows"] = [("res-waiting", False), ("res-finished", False)]
        _sweep(app_module._live_reservation_ids)
        assert ledger["released"] == ["res-finished"]

    def test_hung_job_is_released_even_if_alive(self, out, ledger):
        _manifest(out, "hung", instance="new", heartbeat=time.time())
        app_module._running_jobs.add("hung")
        ledger["rows"] = [("res-hung", True)]
        _sweep(app_module._live_reservation_ids)
        assert ledger["released"] == ["res-hung"]

    def test_unknown_live_set_releases_nothing(self, ledger):
        def boom():
            raise OSError("disk unavailable")
        ledger["rows"] = [("res-a", False), ("res-b", True)]
        _sweep(boom)
        assert ledger["released"] == []

    def test_without_live_set_old_behaviour(self, ledger):
        ledger["rows"] = [("res-a", False)]
        _sweep(None)
        assert ledger["released"] == ["res-a"]

    def test_handover_both_instances_keep_each_others_jobs(self, out, ledger, monkeypatch):
        # One disk, two processes. The new one queued a job of its own; the old
        # one is draining a running job and left a queued one on disk.
        _manifest(out, "new_queued")
        _manifest(out, "old_running", instance="old", heartbeat=time.time())
        _manifest(out, "old_left")
        rows = [("res-new_queued", False), ("res-old_running", False),
                ("res-old_left", False), ("res-gone", False)]

        # Seen from the old (draining) instance: only its own running job in memory.
        monkeypatch.setattr(app_module, "INSTANCE_ID", "old")
        monkeypatch.setattr(app_module, "jobs", {
            "old_running": {"status": "processing", "reservation_id": "res-old_running"},
            "old_left": {"status": "queued", "reservation_id": "res-old_left"},
        })
        monkeypatch.setattr(app_module, "_running_jobs", {"old_running"})
        ledger["rows"] = list(rows)
        _sweep(app_module._live_reservation_ids)
        assert ledger["released"] == ["res-gone"]

        # Seen from the new instance, which resumed old_left and holds new_queued.
        ledger["released"].clear()
        monkeypatch.setattr(app_module, "INSTANCE_ID", "new")
        monkeypatch.setattr(app_module, "jobs", {
            "new_queued": {"status": "queued", "reservation_id": "res-new_queued"},
            "old_left": {"status": "queued", "reservation_id": "res-old_left"},
        })
        monkeypatch.setattr(app_module, "_running_jobs", set())
        _sweep(app_module._live_reservation_ids)
        assert ledger["released"] == ["res-gone"]

