"""Paying moves the videos a user already has in the queue up.

The priority used to be fixed at enqueue, so a free user who upgraded while
their video waited kept waiting behind every paid job. The webhook now raises
their queued entries (heap re-keyed, arrival order kept inside the new
priority), their resume manifests (so the job keeps the priority when it is
enqueued again from disk, by this or the other instance of a deploy) and with
them the position /api/status reports. Running jobs are not touched.
"""
import asyncio
import itertools
import json
import time

import pytest

app_module = pytest.importorskip("app")

ALICE = "a11ce000-0000-0000-0000-000000000000"   # free user who pays
BOB = "b0b00000-0000-0000-0000-000000000000"


@pytest.fixture
def out(tmp_path, monkeypatch):
    root = tmp_path / "output"
    root.mkdir()
    monkeypatch.setattr(app_module, "OUTPUT_DIR", str(root))
    monkeypatch.setattr(app_module, "INSTANCE_ID", "me")
    monkeypatch.setattr(app_module, "_draining", False)
    monkeypatch.setattr(app_module, "_running_jobs", set())
    monkeypatch.setattr(app_module, "_admitting", set())
    monkeypatch.setattr(app_module, "jobs", {})
    monkeypatch.setattr(app_module, "job_queue", asyncio.PriorityQueue())
    monkeypatch.setattr(app_module, "_job_seq", itertools.count())
    return root


def _manifest(out, job_id, **extra):
    d = out / job_id
    d.mkdir(exist_ok=True)
    m = {"cmd": ["python", "main.py"], "priority": 2, "user_id": None,
         "reservation_id": f"res-{job_id}", "watermark": False, "attempts": 0}
    m.update(extra)
    (d / app_module._RESUME_FILE).write_text(json.dumps(m))
    return d


def _queue(out, job_id, user, priority):
    """A job submitted here: manifest on disk, record in memory, heap entry."""
    _manifest(out, job_id, user_id=str(user), priority=priority)  # as written at submit
    app_module.jobs[job_id] = {"status": "queued", "user_id": user, "logs": []}
    app_module._enqueue_job(job_id, priority)


def _drain_order():
    """Pop the real heap: this is the order the worker would dispatch in."""
    order = []
    q = app_module.job_queue
    while not q.empty():
        order.append(q.get_nowait()[2])
    return order


def _disk_priority(out, job_id):
    return json.loads((out / job_id / app_module._RESUME_FILE).read_text())["priority"]


@pytest.fixture
def busy_queue(out):
    # Arrival order: free, starter, Alice x2 (free), starter, free, pro.
    _queue(out, "free1", BOB, 2)
    _queue(out, "starter1", BOB, 1)
    _queue(out, "alice1", ALICE, 2)
    _queue(out, "alice2", ALICE, 2)
    _queue(out, "starter2", BOB, 1)
    _queue(out, "free2", BOB, 2)
    _queue(out, "pro1", BOB, 0)
    # Alice also has a video rendering right now, heartbeated by us.
    _manifest(out, "alice_running", user_id=ALICE, priority=2,
              instance="me", heartbeat=time.time())
    app_module.jobs["alice_running"] = {"status": "processing", "user_id": ALICE, "logs": []}
    app_module._running_jobs.add("alice_running")
    return out


class TestReprioritize:
    def test_queue_order_after_upgrade_keeps_fifo(self, busy_queue):
        assert app_module.reprioritize_user_jobs(ALICE, 1) == 2
        # Alice's jobs join the starter line in their arrival order: after the
        # starter job that came before them, before the one that came after.
        assert _drain_order() == ["pro1", "starter1", "alice1", "alice2",
                                  "starter2", "free1", "free2"]

    def test_heap_stays_a_heap(self, busy_queue):
        app_module.reprioritize_user_jobs(ALICE, 0)
        heap = list(app_module.job_queue._queue)
        for i, e in enumerate(heap):
            for c in (2 * i + 1, 2 * i + 2):
                if c < len(heap):
                    assert e <= heap[c]
        assert _drain_order()[:3] == ["alice1", "alice2", "pro1"]

    def test_manifests_follow_running_job_untouched(self, busy_queue):
        app_module.reprioritize_user_jobs(ALICE, 1)
        assert _disk_priority(busy_queue, "alice1") == 1
        assert _disk_priority(busy_queue, "alice2") == 1
        # Running: same manifest, same record, not queued again.
        assert _disk_priority(busy_queue, "alice_running") == 2
        assert app_module._running_jobs == {"alice_running"}
        assert app_module.jobs["alice_running"]["status"] == "processing"
        assert "alice_running" not in [e[2] for e in app_module.job_queue._queue]
        # Someone else's jobs are not touched.
        assert _disk_priority(busy_queue, "free1") == 2

    def test_status_position_moves_up(self, busy_queue):
        before = app_module.queue_snapshot("alice1")
        app_module.reprioritize_user_jobs(ALICE, 1)
        after = app_module.queue_snapshot("alice1")
        assert before["position"] == 5   # behind pro, both starters and free1
        assert after["position"] == 3    # pro1, starter1, then alice1
        assert after["eta_seconds"] <= before["eta_seconds"]
        assert any("up the line" in line for line in app_module.jobs["alice1"]["logs"])

    def test_never_moves_a_job_down(self, busy_queue):
        app_module.reprioritize_user_jobs(ALICE, 0)
        assert app_module.reprioritize_user_jobs(ALICE, 1) == 0  # e.g. pro -> starter
        assert _disk_priority(busy_queue, "alice1") == 0
        assert _drain_order()[:2] == ["alice1", "alice2"]

    def test_uuid_and_str_owner_match(self, out):
        import uuid
        _queue(out, "j", uuid.UUID(ALICE), 2)
        assert app_module.reprioritize_user_jobs(ALICE, 1) == 1


class TestDeployHandover:
    def test_manifest_left_by_draining_instance_resumes_with_new_priority(self, out):
        # The old instance left Alice's queued job on disk (no heartbeat); the
        # webhook reaches this (new) instance before its resume scan took it.
        _manifest(out, "left", user_id=ALICE, priority=2)
        assert app_module.reprioritize_user_jobs(ALICE, 1) == 1
        _queue(out, "free_here", BOB, 2)
        app_module._resume_interrupted_jobs()
        assert app_module.jobs["left"]["status"] == "queued"
        assert _drain_order() == ["left", "free_here"]

    def test_job_running_on_other_instance_is_not_touched(self, out):
        _manifest(out, "theirs", user_id=ALICE, priority=2,
                  instance="other", heartbeat=time.time())
        assert app_module.reprioritize_user_jobs(ALICE, 1) == 0
        assert _disk_priority(out, "theirs") == 2

    def test_queue_follows_manifest_rewritten_by_other_instance(self, out):
        # The webhook reached the OTHER instance: it rewrote the manifest on
        # the shared disk but cannot reach our heap. The resume scan follows.
        _queue(out, "free1", BOB, 2)
        _queue(out, "alice1", ALICE, 2)
        _queue(out, "starter1", BOB, 1)
        _manifest(out, "alice1", user_id=ALICE, priority=1)
        assert app_module._follow_manifest_priorities() == ["alice1"]
        assert _drain_order() == ["alice1", "starter1", "free1"]

    def test_follow_never_moves_down_and_ignores_missing_manifests(self, out):
        _queue(out, "pro1", BOB, 0)
        _manifest(out, "pro1", user_id=BOB, priority=2)
        app_module.jobs["nomanifest"] = {"status": "queued", "user_id": BOB, "logs": []}
        app_module._enqueue_job("nomanifest", 2)
        assert app_module._follow_manifest_priorities() == []
        assert _drain_order() == ["pro1", "nomanifest"]


def test_auto_retry_enqueues_with_priority_read_at_retry_time(out, monkeypatch):
    monkeypatch.setattr(app_module, "AUTO_RETRY_DELAY_SECONDS", 0)
    d = _manifest(out, "retry", user_id=ALICE, priority=2)
    app_module.jobs["retry"] = {"status": "failed", "user_id": ALICE,
                                "logs": [], "output_dir": str(d)}

    async def run():
        app_module._schedule_auto_retry("retry", app_module.jobs["retry"])
        _manifest(out, "retry", user_id=ALICE, priority=1)  # upgraded meanwhile
        for _ in range(5):
            await asyncio.sleep(0)

    asyncio.run(run())
    assert list(app_module.job_queue._queue)[0][0] == 1
