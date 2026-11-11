"""The manifest scans must not block the event loop (volume: hundreds of jobs).

The API, its health check and the job queue share one event loop. With the
queue backed up there are thousands of job dirs, and reading every manifest
(open + json.load) inside the loop stalled it for as long as that took on
every Stripe upsert, every resume scan and every metering sweep. The scans
now read and rewrite on a worker thread; only the heap is touched on the loop.

Each manifest read is made artificially slow here, so the test tells the two
apart on any machine: inside the loop the stall would be the whole scan.
"""
import asyncio
import itertools
import json
import time

import pytest

app_module = pytest.importorskip("app")

N = 800                  # manifests
READ_COST = 0.001        # seconds per manifest read -> ~0.8 s per full scan
MAX_STALL = 0.15         # what the loop may be held, in seconds


@pytest.fixture
def busy_disk(tmp_path, monkeypatch):
    root = tmp_path / "output"
    root.mkdir()
    for name in (".instance", ".queue_watch.json", ".queue_watch.lock"):
        (root / name).write_text("x")
    monkeypatch.setattr(app_module, "OUTPUT_DIR", str(root))
    monkeypatch.setattr(app_module, "INSTANCE_ID", "me")
    monkeypatch.setattr(app_module, "_running_jobs", set())
    monkeypatch.setattr(app_module, "_admitting", set())
    monkeypatch.setattr(app_module, "jobs", {})
    monkeypatch.setattr(app_module, "job_queue", asyncio.PriorityQueue())
    monkeypatch.setattr(app_module, "_job_seq", itertools.count())
    monkeypatch.setattr(app_module, "_seen_priority_gen", app_module._UNSEEN)
    for i in range(N):
        d = root / f"j{i}"
        d.mkdir()
        uid = f"u{i % 30}"
        (d / app_module._RESUME_FILE).write_text(json.dumps(
            {"priority": 2, "user_id": uid, "reservation_id": f"r{i}"}))
        app_module.jobs[f"j{i}"] = {"status": "queued", "user_id": uid,
                                    "reservation_id": f"r{i}", "logs": []}
        app_module._enqueue_job(f"j{i}", 2)
    real = app_module._read_manifest

    def slow_read(job_id):
        time.sleep(READ_COST)
        return real(job_id)
    monkeypatch.setattr(app_module, "_read_manifest", slow_read)
    return root


def _worst_stall(coro_fn):
    async def run():
        worst, stop = 0.0, False

        async def ticker():
            nonlocal worst
            last = time.perf_counter()
            while not stop:
                await asyncio.sleep(0.005)
                now = time.perf_counter()
                worst = max(worst, now - last - 0.005)
                last = now
        t = asyncio.create_task(ticker())
        await asyncio.sleep(0.02)
        started = time.perf_counter()
        result = await coro_fn()
        took = time.perf_counter() - started
        stop = True
        await t
        return worst, took, result
    return asyncio.run(run())


def test_reprioritize_keeps_the_loop_responsive(busy_disk):
    worst, took, moved = _worst_stall(lambda: app_module.reprioritize_user_jobs("u7", 1))
    assert moved == sum(1 for i in range(N) if i % 30 == 7)
    assert took > 0.5, "the slow reads did happen"
    assert worst < MAX_STALL, f"loop held {worst * 1000:.0f} ms"


def test_follow_keeps_the_loop_responsive(busy_disk):
    worst, took, _ = _worst_stall(app_module._follow_manifest_priorities)
    assert took > 0.5
    assert worst < MAX_STALL, f"loop held {worst * 1000:.0f} ms"


def test_live_reservations_keep_the_loop_responsive(busy_disk):
    worst, took, live = _worst_stall(app_module._live_reservation_ids)
    assert len(live) == N
    assert took > 0.5
    assert worst < MAX_STALL, f"loop held {worst * 1000:.0f} ms"
