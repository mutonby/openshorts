"""The stuck-reservation sweeper must not refund videos that are still alive.

It used to release every ``reserved`` row older than STUCK_RESERVATION_HOURS.
With a queue longer than that, the reservation of a video still waiting (or
rendering) was refunded, the job delivered later and its commit found nothing
to commit: free minutes never counted. Now it skips reservations whose job is
queued, running or waiting in a resume manifest on the shared disk, and frees
a live one only when its JOB has not moved for LIVE_RESERVATION_MAX_HOURS (a
run that never ends, a manifest nobody takes): never by the reservation's age,
so a long wait in the queue does not count. A draining instance does not sweep.
"""
import asyncio
import json
import os
import time

import pytest

app_module = pytest.importorskip("app")
metering = pytest.importorskip("cloud.metering")

HOUR = 3600


@pytest.fixture
def out(tmp_path, monkeypatch):
    root = tmp_path / "output"
    root.mkdir()
    monkeypatch.setattr(app_module, "OUTPUT_DIR", str(root))
    monkeypatch.setattr(app_module, "INSTANCE_ID", "new")
    monkeypatch.setattr(app_module, "_running_jobs", set())
    monkeypatch.setattr(app_module, "jobs", {})
    monkeypatch.delenv("LIVE_RESERVATION_MAX_HOURS", raising=False)
    return root


def _manifest(out, job_id, age=0, **extra):
    d = out / job_id
    d.mkdir(exist_ok=True)
    m = {"cmd": ["python", "main.py"], "priority": 2, "user_id": "u",
         "reservation_id": f"res-{job_id}", "watermark": False, "attempts": 0}
    m.update(extra)
    path = d / app_module._RESUME_FILE
    path.write_text(json.dumps(m))
    if age:
        t = time.time() - age
        os.utime(path, (t, t))
    return d


def _live():
    return asyncio.run(app_module._live_reservation_ids())


class _Result:
    def __init__(self, ids):
        self.ids = ids

    def scalars(self):
        return iter(self.ids)


class _Session:
    def __init__(self, ids):
        self.ids = ids

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def execute(self, stmt):
        from sqlalchemy.dialects import postgresql
        sql = str(stmt.compile(dialect=postgresql.dialect()))  # the real statement
        assert "FROM usage_ledger" in sql and "usage_ledger.created_at <" in sql
        return _Result(self.ids)


@pytest.fixture
def ledger(monkeypatch):
    """Ids of the reserved rows the SQL returns (older than the stuck window)."""
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
        now = time.time()
        _manifest(out, "queued_here")
        app_module.jobs["queued_here"] = {"status": "queued", "reservation_id": "res-queued_here"}
        _manifest(out, "running_here", instance="new", heartbeat=now, run_started=now - 600)
        app_module.jobs["running_here"] = {"status": "processing", "run_started": now - 600,
                                           "reservation_id": "res-running_here"}
        app_module._running_jobs.add("running_here")
        _manifest(out, "running_there", instance="old", heartbeat=now, run_started=now - 900)
        _manifest(out, "left_by_drain", age=120)               # nobody holds it yet
        app_module.jobs["no_manifest"] = {"status": "queued", "reservation_id": "res-no_manifest"}
        app_module.jobs["done"] = {"status": "completed", "reservation_id": "res-done"}
        live = _live()
        assert set(live) == {"res-queued_here", "res-running_here", "res-running_there",
                             "res-left_by_drain", "res-no_manifest"}
        assert live["res-queued_here"] >= now                  # queued: moving now
        assert live["res-running_here"] == pytest.approx(now - 600)
        assert live["res-running_there"] == pytest.approx(now - 900)
        assert live["res-left_by_drain"] == pytest.approx(now - 120, abs=2)

    def test_queued_for_days_is_alive_now(self, out):
        _manifest(out, "patient", age=40 * HOUR)
        app_module.jobs["patient"] = {"status": "queued", "reservation_id": "res-patient"}
        assert _live()["res-patient"] >= time.time() - 5

    def test_drain_cancelled_job_reading_failed_here_stays_live(self, out):
        # The old instance killed it at the drain timeout: its record may say
        # 'failed', but the manifest is kept and the next instance resumes it.
        _manifest(out, "cut", instance="old", heartbeat=time.time() - 600)
        app_module.jobs["cut"] = {"status": "failed", "reservation_id": "res-cut"}
        assert "res-cut" in _live()

    def test_loose_files_do_not_spam_the_log(self, out, capsys):
        for name in (".instance", ".queue_watch.json", ".queue_watch.lock", ".priority_gen"):
            (out / name).write_text("x")
        _manifest(out, "j")
        assert set(_live()) == {"res-j"}
        assert "Bad resume manifest" not in capsys.readouterr().out


class TestSweep:
    def test_live_reservations_are_kept_dead_ones_released(self, out, ledger):
        _manifest(out, "waiting")
        ledger["rows"] = ["res-waiting", "res-finished"]
        _sweep(app_module._live_reservation_ids)
        assert ledger["released"] == ["res-finished"]

    def test_free_job_queued_longer_than_the_cap_keeps_its_reservation(self, out, ledger):
        # The free line waits 30 h: the reservation is 30 h old, the job is
        # still queued here. The cap is about the job not moving, not this.
        _manifest(out, "slow_line", age=30 * HOUR)
        app_module.jobs["slow_line"] = {"status": "queued", "reservation_id": "res-slow_line"}
        ledger["rows"] = ["res-slow_line"]
        _sweep(app_module._live_reservation_ids)
        assert ledger["released"] == []

    def test_run_that_never_ends_is_released_despite_its_heartbeat(self, out, ledger):
        now = time.time()
        _manifest(out, "hung", instance="new", heartbeat=now, run_started=now - 25 * HOUR)
        app_module.jobs["hung"] = {"status": "processing", "run_started": now - 25 * HOUR,
                                   "reservation_id": "res-hung"}
        app_module._running_jobs.add("hung")
        ledger["rows"] = ["res-hung"]
        _sweep(app_module._live_reservation_ids)
        assert ledger["released"] == ["res-hung"]

    def test_manifest_nobody_takes_for_a_day_is_released(self, out, ledger):
        _manifest(out, "orphan", age=25 * HOUR)
        ledger["rows"] = ["res-orphan"]
        _sweep(app_module._live_reservation_ids)
        assert ledger["released"] == ["res-orphan"]

    def test_unknown_live_set_releases_nothing(self, ledger):
        def boom():
            raise OSError("disk unavailable")
        ledger["rows"] = ["res-a", "res-b"]
        _sweep(boom)
        assert ledger["released"] == []

    def test_without_live_set_old_behaviour(self, ledger):
        ledger["rows"] = ["res-a"]
        _sweep(None)
        assert ledger["released"] == ["res-a"]

    def test_a_plain_set_counts_as_moving_now(self, ledger):
        ledger["rows"] = ["res-a", "res-b"]
        _sweep(lambda: {"res-a"})
        assert ledger["released"] == ["res-b"]

    def test_new_instance_keeps_the_draining_ones_running_jobs(self, out, ledger, monkeypatch):
        # One disk, two processes. The old one is draining a running job; the
        # new one resumed what the old one left queued and has its own.
        now = time.time()
        _manifest(out, "new_queued")
        _manifest(out, "old_running", instance="old", heartbeat=now, run_started=now - HOUR)
        _manifest(out, "old_left")
        monkeypatch.setattr(app_module, "jobs", {
            "new_queued": {"status": "queued", "reservation_id": "res-new_queued"},
            "old_left": {"status": "queued", "reservation_id": "res-old_left"},
        })
        ledger["rows"] = ["res-new_queued", "res-old_running", "res-old_left", "res-gone"]
        _sweep(app_module._live_reservation_ids)
        assert ledger["released"] == ["res-gone"]


def test_draining_instance_does_not_sweep(monkeypatch):
    calls = []

    async def fake_sweep(live_ids=None):
        calls.append(1)
    monkeypatch.setattr(metering, "_sweep_once", fake_sweep)
    monkeypatch.setattr(metering, "SWEEP_INTERVAL_SECONDS", 0)
    active = {"v": False}

    async def run():
        task = asyncio.create_task(metering._sweeper_loop(lambda: {}, lambda: active["v"]))
        for _ in range(5):
            await asyncio.sleep(0)
        assert calls == []
        active["v"] = True
        for _ in range(5):
            await asyncio.sleep(0)
        task.cancel()

    asyncio.run(run())
    assert calls


def test_dispatch_stamps_the_start_of_the_run(out):
    _manifest(out, "j")
    app_module._touch_manifest("j", now=1000.0, started=True)
    app_module._touch_manifest("j", now=1010.0)
    m = json.loads((out / "j" / app_module._RESUME_FILE).read_text())
    assert m["run_started"] == 1000.0 and m["heartbeat"] == 1010.0
