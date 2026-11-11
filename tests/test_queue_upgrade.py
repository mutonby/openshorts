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
    monkeypatch.setattr(app_module, "_seen_priority_gen", app_module._UNSEEN)
    return root


def _reprio(user_id, priority):
    return asyncio.run(app_module.reprioritize_user_jobs(user_id, priority))


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
        assert _reprio(ALICE, 1) == 2
        # Alice's jobs join the starter line in their arrival order: after the
        # starter job that came before them, before the one that came after.
        assert _drain_order() == ["pro1", "starter1", "alice1", "alice2",
                                  "starter2", "free1", "free2"]

    def test_heap_stays_a_heap(self, busy_queue):
        _reprio(ALICE, 0)
        heap = list(app_module.job_queue._queue)
        for i, e in enumerate(heap):
            for c in (2 * i + 1, 2 * i + 2):
                if c < len(heap):
                    assert e <= heap[c]
        assert _drain_order()[:3] == ["alice1", "alice2", "pro1"]

    def test_manifests_follow_running_job_untouched(self, busy_queue):
        _reprio(ALICE, 1)
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
        _reprio(ALICE, 1)
        after = app_module.queue_snapshot("alice1")
        assert before["position"] == 5   # behind pro, both starters and free1
        assert after["position"] == 3    # pro1, starter1, then alice1
        assert after["eta_seconds"] <= before["eta_seconds"]
        assert any("up the line" in line for line in app_module.jobs["alice1"]["logs"])

    def test_never_moves_a_job_down(self, busy_queue):
        _reprio(ALICE, 0)
        assert _reprio(ALICE, 1) == 0  # e.g. pro -> starter
        assert _disk_priority(busy_queue, "alice1") == 0
        assert _drain_order()[:2] == ["alice1", "alice2"]

    def test_uuid_and_str_owner_match(self, out):
        import uuid
        _queue(out, "j", uuid.UUID(ALICE), 2)
        assert _reprio(ALICE, 1) == 1


class TestDeployHandover:
    def test_manifest_left_by_draining_instance_resumes_with_new_priority(self, out):
        # The old instance left Alice's queued job on disk (no heartbeat); the
        # webhook reaches this (new) instance before its resume scan took it.
        _manifest(out, "left", user_id=ALICE, priority=2)
        assert _reprio(ALICE, 1) == 1
        _queue(out, "free_here", BOB, 2)
        app_module._resume_interrupted_jobs()
        assert app_module.jobs["left"]["status"] == "queued"
        assert _drain_order() == ["left", "free_here"]

    def test_job_running_on_other_instance_is_not_touched(self, out):
        _manifest(out, "theirs", user_id=ALICE, priority=2,
                  instance="other", heartbeat=time.time())
        assert _reprio(ALICE, 1) == 0
        assert _disk_priority(out, "theirs") == 2

    def test_queue_follows_manifest_rewritten_by_other_instance(self, out):
        # The webhook reached the OTHER instance: it rewrote the manifest on
        # the shared disk but cannot reach our heap. The resume scan follows.
        _queue(out, "free1", BOB, 2)
        _queue(out, "alice1", ALICE, 2)
        _queue(out, "starter1", BOB, 1)
        assert asyncio.run(app_module._follow_manifest_priorities()) == []  # first look
        _manifest(out, "alice1", user_id=ALICE, priority=1)
        app_module._bump_priority_generation()   # what the other instance's rewrite does
        assert asyncio.run(app_module._follow_manifest_priorities()) == ["alice1"]
        assert _drain_order() == ["alice1", "starter1", "free1"]

    def test_follow_never_moves_down_and_ignores_missing_manifests(self, out):
        _queue(out, "pro1", BOB, 0)
        _manifest(out, "pro1", user_id=BOB, priority=2)
        app_module.jobs["nomanifest"] = {"status": "queued", "user_id": BOB, "logs": []}
        app_module._enqueue_job("nomanifest", 2)
        assert asyncio.run(app_module._follow_manifest_priorities()) == []
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


class TestResumeUsesCurrentPlan:
    """A job enqueued again from its manifest takes the better of the
    manifest's priority and its owner's plan now (one lookup for all)."""

    @pytest.fixture
    def plans(self, out, monkeypatch):
        from cloud import config as cloud_config
        monkeypatch.setattr(app_module, "BILLING_ENABLED", True)
        monkeypatch.setattr(app_module, "_cloud_config", cloud_config)
        monkeypatch.setattr(app_module, "_owner_plan_pending", set())
        state = {"live": {ALICE: "pro"}, "calls": []}

        async def lookup(owners):
            state["calls"].append(set(owners))
            if state["live"] is None:
                raise RuntimeError("db down")
            return {u: p for u, p in state["live"].items() if u in owners}
        monkeypatch.setattr(app_module, "_owner_live_plans", lookup)
        return state

    def test_deploy_resume_moves_owners_who_paid_since(self, out, plans):
        # Thousands in production; a few here. All written at free priority.
        for i in range(5):
            _manifest(out, f"alice{i}", user_id=ALICE, priority=2)
            _manifest(out, f"bob{i}", user_id=BOB, priority=2)
        _manifest(out, "bob_starter", user_id=BOB, priority=1)
        app_module._resume_interrupted_jobs()
        assert asyncio.run(app_module._raise_to_owner_plans()) == 5
        assert plans["calls"] == [{ALICE, BOB}], "one batched lookup, not one per job"
        order = _drain_order()
        assert sorted(order[:5]) == [f"alice{i}" for i in range(5)]
        assert order[5] == "bob_starter"
        assert _disk_priority(out, "alice0") == 0
        assert app_module._owner_plan_pending == set()

    def test_drain_cut_job_of_owner_who_paid_while_it_ran(self, out, plans):
        # It was running on the old instance when the webhook came, so its
        # manifest kept the free priority; the drain cut it and we resume it.
        _manifest(out, "cut", user_id=ALICE, priority=2,
                  instance="old", heartbeat=time.time() - 600)
        _manifest(out, "free1", user_id=BOB, priority=2)
        app_module._resume_interrupted_jobs()
        asyncio.run(app_module._raise_to_owner_plans())
        assert _drain_order()[0] == "cut"

    def test_lookup_failure_keeps_manifest_priority(self, out, plans):
        plans["live"] = None
        _manifest(out, "a", user_id=ALICE, priority=2)
        app_module._resume_interrupted_jobs()
        assert asyncio.run(app_module._raise_to_owner_plans()) == 0
        assert list(app_module.job_queue._queue)[0][0] == 2
        assert _disk_priority(out, "a") == 2

    def test_no_owner_no_lookup(self, out, plans):
        _manifest(out, "selfhost", user_id=None, priority=2)
        app_module._resume_interrupted_jobs()
        assert asyncio.run(app_module._raise_to_owner_plans()) == 0
        assert plans["calls"] == []

    def test_auto_retry_takes_current_plan(self, out, plans, monkeypatch):
        monkeypatch.setattr(app_module, "AUTO_RETRY_DELAY_SECONDS", 0)
        d = _manifest(out, "retry", user_id=ALICE, priority=2)
        app_module.jobs["retry"] = {"status": "failed", "user_id": ALICE,
                                    "logs": [], "output_dir": str(d)}

        async def run():
            app_module._schedule_auto_retry("retry", app_module.jobs["retry"])
            for _ in range(10):
                await asyncio.sleep(0)

        asyncio.run(run())
        assert list(app_module.job_queue._queue)[0][0] == 0


def test_live_plans_batches_and_compiles(monkeypatch):
    import uuid
    from sqlalchemy.dialects import postgresql
    from cloud import metering
    seen = []

    class _R:
        def __init__(self, rows):
            self.rows = rows

        def all(self):
            return self.rows

    class _S:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        async def execute(self, stmt):
            sql = str(stmt.compile(dialect=postgresql.dialect()))
            assert "FROM subscriptions" in sql and "current_period_end" in sql
            seen.append(sql)
            return _R([(uuid.UUID(ALICE), "creator")] if len(seen) == 1 else [])

    monkeypatch.setattr(metering.database, "session", lambda: _S())
    monkeypatch.setattr(metering, "LIVE_PLANS_CHUNK", 2)
    owners = [ALICE, BOB, str(uuid.uuid4()), str(uuid.uuid4()), str(uuid.uuid4()),
              "not-a-uuid", None]
    plans = asyncio.run(metering.live_plans(owners))
    assert len(seen) == 3            # 5 valid ids in chunks of 2
    assert plans == {ALICE: "creator"}


def test_follow_reads_manifests_only_when_the_generation_changes(out, monkeypatch):
    for i in range(3):
        _queue(out, f"j{i}", BOB, 2)
    reads = []
    real = app_module._manifest_priorities
    monkeypatch.setattr(app_module, "_manifest_priorities",
                        lambda ids: reads.append(len(ids)) or real(ids))
    asyncio.run(app_module._follow_manifest_priorities())     # first look reads
    asyncio.run(app_module._follow_manifest_priorities())     # nothing changed
    asyncio.run(app_module._follow_manifest_priorities())
    assert reads == [3]
    _reprio(BOB, 1)                                           # a rewrite bumps it
    asyncio.run(app_module._follow_manifest_priorities())
    assert reads == [3, 3]


def test_loose_files_in_output_dir_are_not_read_as_manifests(out, capsys):
    for name in (".instance", ".queue_watch.json", ".queue_watch.lock"):
        (out / name).write_text("x")
    _queue(out, "j1", ALICE, 2)
    _reprio(ALICE, 1)
    assert "Bad resume manifest" not in capsys.readouterr().out
