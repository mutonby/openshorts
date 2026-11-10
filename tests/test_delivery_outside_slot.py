"""Delivery outside the GPU slot.

Once main.py exits, the GPU slot goes to the next job and the delivery
(settle, R2 archive, unmark, webhook, emails) runs after it. The user must not
see the job finished before its clips are archived and unmarked; a deploy must
never lose a delivery, nor render the video again, nor deliver it twice.
"""
import asyncio
import json
import time

import pytest

app_module = pytest.importorskip("app")

STEPS = [name for name, _ in app_module._DELIVERY_STEPS]


@pytest.fixture
def env(tmp_path, monkeypatch):
    root = tmp_path / "output"
    root.mkdir()
    monkeypatch.setattr(app_module, "OUTPUT_DIR", str(root))
    monkeypatch.setattr(app_module, "INSTANCE_ID", "me")
    monkeypatch.setattr(app_module, "_draining", False)
    monkeypatch.setattr(app_module, "_stopping", False)
    monkeypatch.setattr(app_module, "_running_jobs", set())
    monkeypatch.setattr(app_module, "_delivering_jobs", set())
    monkeypatch.setattr(app_module, "_pending_deliveries", [])
    monkeypatch.setattr(app_module, "jobs", {})
    monkeypatch.setattr(app_module, "job_queue", asyncio.PriorityQueue())
    calls = []
    hooks = {}

    def fake(name):
        async def step(job_id, *a, **k):
            calls.append((name, job_id))
            if name in hooks:
                await hooks[name](job_id)
        return step

    for attr, name in (("_settle_reservation", "settle"), ("_archive_managed_job", "archive"),
                       ("_unmark_if_owner_paid", "unmark"),
                       ("_autopilot_job_finished", "autopilot"),
                       ("_notify_job_webhook", "webhook"), ("_record_job_alert", "alert"),
                       ("_track_proxy_usage", "proxy"), ("_notify_clips_ready", "email"),
                       ("_notify_clip_activity", "activity")):
        monkeypatch.setattr(app_module, attr, fake(name))
    return root, calls, hooks


def _job(root, job_id, rendered=True):
    d = root / job_id
    d.mkdir()
    (d / app_module._RESUME_FILE).write_text(json.dumps(
        {"cmd": ["python", "main.py"], "priority": 2, "user_id": "u",
         "reservation_id": f"res-{job_id}", "watermark": False, "attempts": 0}))
    (d / "V_metadata.json").write_text(json.dumps({"shorts": [{"title": "a"}]}))
    if rendered:
        (d / "V_clip_1.mp4").write_bytes(b"clip")
    job = {"status": "queued", "logs": [], "cmd": ["x"], "env": {},
           "output_dir": str(d), "user_id": "u", "reservation_id": f"res-{job_id}"}
    app_module.jobs[job_id] = job
    return d, job


def _run(monkeypatch, job_id, status="completed", cancel_in=None, hooks=None):
    """run_job_wrapper with a fake main.py; ``cancel_in`` cancels the task
    while that delivery step runs (a drain timeout taking the process)."""
    async def fake_run_job(jid, job):
        job["status"] = status
        if status == "completed":
            job["result"] = {"clips": [{"video_url": f"/videos/{jid}/V_clip_1.mp4"}]}
            job["ready_files"] = {0: "V_clip_1.mp4"}
    monkeypatch.setattr(app_module, "run_job", fake_run_job)

    async def go():
        sem = asyncio.Semaphore(1)
        await sem.acquire()
        monkeypatch.setattr(app_module, "concurrency_semaphore", sem)
        monkeypatch.setattr(app_module, "_delivery_semaphore", asyncio.Semaphore(4))
        q = asyncio.PriorityQueue()
        q.put_nowait((2, 0, job_id))
        await q.get()
        monkeypatch.setattr(app_module, "job_queue", q)
        app_module._running_jobs.add(job_id)
        task = asyncio.create_task(app_module.run_job_wrapper(job_id))
        if cancel_in is not None:
            async def cancel_me(_jid):
                task.cancel()
                await asyncio.sleep(1)
            hooks[cancel_in] = cancel_me
        try:
            await task
        except asyncio.CancelledError:
            pass
        return sem
    return asyncio.run(go())


class TestSlotGoesFirst:
    def test_slot_is_free_while_the_clips_upload(self, env, monkeypatch):
        root, calls, hooks = env
        d, job = _job(root, "j1")
        seen = {}

        async def during_archive(jid):
            seen["sem"] = app_module.concurrency_semaphore._value
            seen["running"] = set(app_module._running_jobs)
            seen["delivering"] = set(app_module._delivering_jobs)
            seen["status"] = app_module._presented_status(jid, job)
            seen["manifest"] = app_module._read_manifest(jid)
        hooks["archive"] = during_archive
        sem = _run(monkeypatch, "j1")
        assert seen["sem"] == 1, "the GPU slot must be released before the R2 upload"
        assert seen["running"] == set() and seen["delivering"] == {"j1"}
        assert seen["status"] == "processing", "not finished for the user mid-delivery"
        assert seen["manifest"]["stage"] == "delivering"
        assert seen["manifest"]["outcome"] == "completed"
        assert seen["manifest"]["delivered"] == ["settle"]
        assert [c[0] for c in calls] == STEPS        # every step, once, in order
        assert not (d / app_module._RESUME_FILE).exists()
        assert app_module._presented_status("j1", job) == "completed"
        assert app_module._delivering_jobs == set() and sem._value == 1

    def test_caller_is_told_only_once_it_reads_completed(self, env, monkeypatch):
        root, calls, hooks = env
        _d, job = _job(root, "j2")
        seen = {}

        def at(name):
            async def f(jid):
                seen[name] = app_module._presented_status(jid, job)
            return f
        for name in ("unmark", "webhook", "email"):
            hooks[name] = at(name)
        _run(monkeypatch, "j2")
        assert seen == {"unmark": "processing", "webhook": "completed", "email": "completed"}

    def test_failed_job_is_delivered_as_failed(self, env, monkeypatch):
        root, calls, _ = env
        d, job = _job(root, "j3", rendered=False)
        job["logs"].append("❌ Clip detection failed")
        _run(monkeypatch, "j3", status="failed")
        assert [c[0] for c in calls] == STEPS
        assert job["status"] == "failed" and not (d / app_module._RESUME_FILE).exists()

    def test_a_step_that_raises_does_not_stop_the_delivery(self, env, monkeypatch):
        root, calls, hooks = env
        d, _job_rec = _job(root, "j4")

        async def boom(jid):
            raise RuntimeError("R2 down")
        hooks["archive"] = boom
        _run(monkeypatch, "j4")
        assert [c[0] for c in calls] == STEPS
        assert not (d / app_module._RESUME_FILE).exists()

    def test_deliveries_are_capped(self, env, monkeypatch):
        root, _calls, hooks = env
        active = {"now": 0, "max": 0}

        async def slow(jid):
            active["now"] += 1
            active["max"] = max(active["max"], active["now"])
            await asyncio.sleep(0.05)
            active["now"] -= 1
        hooks["archive"] = slow
        recs = [_job(root, f"c{i}")[1] for i in range(4)]
        for i, rec in enumerate(recs):
            rec["status"] = "completed"

        async def go():
            monkeypatch.setattr(app_module, "_delivery_semaphore", asyncio.Semaphore(2))
            await asyncio.gather(*(app_module._deliver_job(f"c{i}", r)
                                   for i, r in enumerate(recs)))
        asyncio.run(go())
        assert active["max"] == 2


class TestDeployHandover:
    def test_drain_waits_for_deliveries(self, env):
        calls = []
        app_module._delivering_jobs.add("d1")

        async def scenario():
            async def finish_soon():
                await asyncio.sleep(0.2)
                app_module._delivering_jobs.discard("d1")
            asyncio.create_task(finish_soon())
            t = time.time()
            await app_module._drain_then_exit(lambda sig, frame: calls.append(sig),
                                              timeout=5, proxy_grace=0, hard_exit_after=0)
            return time.time() - t
        assert asyncio.run(scenario()) >= 0.2 and calls

    def test_cut_delivery_resumes_without_rendering_again(self, env, monkeypatch):
        """Drain timeout mid-upload: the manifest survives with the steps done;
        the next instance finishes from the first step not done, no main.py,
        no second settle, nothing enqueued."""
        root, calls, hooks = env
        d, job = _job(root, "k1")
        _run(monkeypatch, "k1", cancel_in="archive", hooks=hooks)
        m = app_module._read_manifest("k1")
        assert m is not None and m["stage"] == "delivering"
        assert m["delivered"] == ["settle"] and m["reservation_id"] == "res-k1"
        assert m["ready_files"] == {"0": "V_clip_1.mp4"}
        assert [c[0] for c in calls] == ["settle", "archive"]

        # The next instance: fresh memory, the old heartbeat gone stale.
        calls.clear()
        hooks.clear()
        app_module.jobs.clear()
        app_module._delivering_jobs.clear()
        monkeypatch.setattr(app_module, "INSTANCE_ID", "new")
        m["heartbeat"] = time.time() - 600
        (d / app_module._RESUME_FILE).write_text(json.dumps(m))
        app_module._recover_jobs_from_disk()
        assert "k1" not in app_module.jobs             # in flight: not "completed" yet
        keep = app_module._resume_interrupted_jobs()
        assert keep == {"res-k1"}                      # reservation kept for the settle
        assert app_module.job_queue.qsize() == 0       # main.py is not run again
        rec = app_module.jobs["k1"]
        assert app_module._presented_status("k1", rec) == "processing"
        assert rec["result"]["clips"][0]["video_url"] == "/videos/k1/V_clip_1.mp4"

        async def go():
            monkeypatch.setattr(app_module, "_delivery_semaphore", asyncio.Semaphore(4))
            app_module._start_pending_deliveries()
            for _ in range(50):
                await asyncio.sleep(0.01)
                if not app_module._delivering_jobs:
                    break
        asyncio.run(go())
        assert [c[0] for c in calls] == STEPS[1:]      # settle not repeated
        assert rec["status"] == "completed"
        assert app_module._presented_status("k1", rec) == "completed"
        assert not (d / app_module._RESUME_FILE).exists()

    def test_cut_delivery_is_never_dropped_as_poison(self, env):
        root, _calls, _ = env
        d, _job_rec = _job(root, "p1")
        app_module.jobs.clear()
        m = app_module._read_manifest("p1")
        now = time.time()
        m.update(stage="delivering", outcome="completed", delivered=["settle"],
                 attempts=app_module.MAX_RESUME_ATTEMPTS, instance="old",
                 heartbeat=now - 600, run_started=now - 900, resumed_at=now - 1000)
        (d / app_module._RESUME_FILE).write_text(json.dumps(m))
        assert app_module._resume_interrupted_jobs() == {"res-p1"}
        assert app_module._pending_deliveries == ["p1"]

    def test_live_delivery_elsewhere_is_left_alone_and_holds_no_gpu(self, env):
        root, _calls, _ = env
        d, _job_rec = _job(root, "e1")
        app_module.jobs.clear()
        m = app_module._read_manifest("e1")
        m.update(stage="delivering", outcome="completed", instance="old",
                 heartbeat=time.time())
        (d / app_module._RESUME_FILE).write_text(json.dumps(m))
        app_module._resume_interrupted_jobs()
        assert "e1" not in app_module.jobs and app_module._pending_deliveries == []
        assert app_module._jobs_busy_elsewhere() == 0
        assert app_module._job_view_from_disk("e1")["status"] == "processing"

    def test_delivering_manifest_is_not_reprioritized(self, env):
        root, _calls, _ = env
        d, _job_rec = _job(root, "r1")
        m = app_module._read_manifest("r1")
        m.update(stage="delivering", outcome="completed")
        (d / app_module._RESUME_FILE).write_text(json.dumps(m))
        assert app_module._rewrite_manifest_priorities({"u": 0}, set(), time.time()) == set()
        assert app_module._read_manifest("r1")["priority"] == 2

    def test_delivering_job_dir_is_never_swept(self, env):
        root, _calls, _ = env
        _job(root, "s1")
        app_module.jobs["s1"]["status"] = "completed"
        app_module._delivering_jobs.add("s1")
        (root / "s1" / app_module._RESUME_FILE).unlink()
        assert app_module._job_is_active("s1")
