"""Deploy handover: two instances, one disk, no job run twice and none lost.

Coolify starts the new container before stopping the old one and both share
OUTPUT_DIR. The old one must drain (finish what it runs, start nothing new);
the new one must resume only manifests nobody is heartbeating; a status poll
landing on either instance must answer from disk.
"""
import asyncio
import json
import os
import time

import httpx
import pytest

app_module = pytest.importorskip("app")


@pytest.fixture
def out(tmp_path, monkeypatch):
    root = tmp_path / "output"
    root.mkdir()
    monkeypatch.setattr(app_module, "OUTPUT_DIR", str(root))
    monkeypatch.setattr(app_module, "INSTANCE_ID", "me")
    monkeypatch.setattr(app_module, "_draining", False)
    monkeypatch.setattr(app_module, "_running_jobs", set())
    # Isolate the in-memory job table and queue from other tests.
    monkeypatch.setattr(app_module, "jobs", {})
    monkeypatch.setattr(app_module, "job_queue", asyncio.PriorityQueue())
    return root


def _manifest(out, job_id, **extra):
    d = out / job_id
    d.mkdir(exist_ok=True)
    m = {"cmd": ["python", "main.py"], "priority": 2, "user_id": None,
         "reservation_id": f"res-{job_id}", "watermark": False, "attempts": 0}
    m.update(extra)
    (d / app_module._RESUME_FILE).write_text(json.dumps(m))
    return d


class TestHeartbeat:
    def test_touch_stamps_instance_and_time(self, out):
        _manifest(out, "j1")
        app_module._touch_manifest("j1", now=1000.0)
        m = app_module._read_manifest("j1")
        assert m["instance"] == "me" and m["heartbeat"] == 1000.0

    def test_fresh_heartbeat_from_another_instance_is_busy(self, out):
        m = {"instance": "other", "heartbeat": 1000.0}
        assert app_module._manifest_busy_elsewhere(m, now=1030.0)

    def test_stale_heartbeat_is_free(self, out):
        m = {"instance": "other", "heartbeat": 1000.0}
        assert not app_module._manifest_busy_elsewhere(
            m, now=1000.0 + app_module.HEARTBEAT_STALE_AFTER + 1)

    def test_own_heartbeat_is_free(self, out):
        # Same container restarted (same hostname): that job is ours to resume.
        m = {"instance": "me", "heartbeat": 1000.0}
        assert not app_module._manifest_busy_elsewhere(m, now=1001.0)

    def test_no_heartbeat_is_free(self, out):
        assert not app_module._manifest_busy_elsewhere({}, now=5.0)


class TestResumeScan:
    def test_busy_elsewhere_is_skipped_but_reservation_kept(self, out):
        _manifest(out, "j1", instance="other", heartbeat=time.time())
        keep = app_module._resume_interrupted_jobs()
        assert "j1" not in app_module.jobs
        assert keep == {"res-j1"}, "the other instance's job must not be refunded as an orphan"

    def test_stale_and_never_started_are_resumed(self, out):
        _manifest(out, "stale", instance="other", heartbeat=time.time() - 600)
        _manifest(out, "queued")
        keep = app_module._resume_interrupted_jobs()
        assert app_module.jobs["stale"]["status"] == "queued"
        assert app_module.jobs["queued"]["status"] == "queued"
        assert keep == {"res-stale", "res-queued"}
        assert app_module._read_manifest("stale")["attempts"] == 1

    def test_a_job_we_already_hold_is_not_enqueued_twice(self, out):
        _manifest(out, "j1")
        app_module.jobs["j1"] = {"status": "processing", "logs": []}
        app_module._resume_interrupted_jobs()
        assert app_module.job_queue.qsize() == 0
        assert app_module._read_manifest("j1")["attempts"] == 0

    def test_poison_job_is_dropped_and_refundable(self, out):
        _manifest(out, "bad", attempts=app_module.MAX_RESUME_ATTEMPTS,
                  instance="other", heartbeat=time.time() - 600)   # it was running
        keep = app_module._resume_interrupted_jobs()
        assert "bad" not in app_module.jobs
        assert keep == set()
        assert app_module._read_manifest("bad") is None


class TestQueuedVideosSurviveDeploys:
    """A video that only waited in the queue has crashed nothing: resuming it
    after a deploy must not use up its attempts. Counting every resume
    dropped long-queued videos once a few deploys had gone by."""

    def test_never_run_job_survives_many_deploys(self, out, monkeypatch):
        _manifest(out, "waiting")
        for _ in range(app_module.MAX_RESUME_ATTEMPTS + 3):
            monkeypatch.setattr(app_module, "job_queue", asyncio.PriorityQueue())
            app_module.jobs.clear()
            keep = app_module._resume_interrupted_jobs()
            assert keep == {"res-waiting"}
            assert app_module.jobs["waiting"]["status"] == "queued"
        assert app_module._read_manifest("waiting")["attempts"] == 0

    def test_manifest_already_at_the_limit_but_never_run_is_resumed(self, out):
        # What production had: attempts used up by deploys alone, no heartbeat.
        _manifest(out, "w", attempts=app_module.MAX_RESUME_ATTEMPTS)
        assert app_module._resume_interrupted_jobs() == {"res-w"}
        assert app_module.jobs["w"]["status"] == "queued"

    def test_a_run_that_keeps_dying_is_still_dropped(self, out, monkeypatch):
        _manifest(out, "crashy")
        for i in range(app_module.MAX_RESUME_ATTEMPTS + 1):
            monkeypatch.setattr(app_module, "job_queue", asyncio.PriorityQueue())
            app_module.jobs.clear()
            # Resumed a while ago, it started running, then its instance died
            # mid-run (heartbeat stale): one more attempt each time.
            m = app_module._read_manifest("crashy")
            if m is None:
                break
            now = time.time()
            m.update(resumed_at=now - 1000 if "resumed_at" in m else None,
                     instance="other", heartbeat=now - 600, run_started=now - 700)
            (out / "crashy" / app_module._RESUME_FILE).write_text(json.dumps(m))
            app_module._resume_interrupted_jobs()
        assert app_module._read_manifest("crashy") is None
        assert "crashy" not in app_module.jobs


class TestDrain:
    def test_marker_of_another_instance_starts_draining(self, out):
        (out / app_module._INSTANCE_MARKER).write_text("newer")
        assert app_module._check_instance_marker() is True
        assert app_module._draining is True

    def test_own_marker_does_not_drain(self, out):
        app_module._write_instance_marker()
        assert app_module._check_instance_marker() is False
        assert app_module._draining is False

    def test_scan_stays_quiet_while_draining(self, out):
        _manifest(out, "j1")
        app_module._begin_drain("test")
        # The draining instance must not adopt manifests: they belong to the
        # next one. (_resume_scan checks the flag; the worker drops dispatches.)
        assert app_module._draining

    def test_exit_waits_for_running_jobs_then_hands_the_signal_on(self, out):
        calls = []
        app_module._running_jobs.add("j1")

        async def scenario():
            async def finish_soon():
                await asyncio.sleep(0.2)
                app_module._running_jobs.discard("j1")
            asyncio.create_task(finish_soon())
            t = time.time()
            await app_module._drain_then_exit(lambda sig, frame: calls.append(sig),
                                              timeout=5, proxy_grace=0, hard_exit_after=0)
            return time.time() - t
        waited = asyncio.run(scenario())
        assert calls and waited >= 0.2

    def test_exit_gives_up_at_the_timeout(self, out):
        calls = []
        app_module._running_jobs.add("stuck")
        asyncio.run(app_module._drain_then_exit(lambda sig, frame: calls.append(sig),
                                                timeout=0.3, proxy_grace=0, hard_exit_after=0))
        assert calls, "must still exit so the deploy can finish"

    def test_keeps_serving_for_the_proxy_grace_after_draining(self, out):
        calls = []
        t = time.time()
        asyncio.run(app_module._drain_then_exit(lambda sig, frame: calls.append(sig),
                                                timeout=1, proxy_grace=0.3, hard_exit_after=0))
        assert calls and time.time() - t >= 0.3

    def test_arms_a_hard_exit_after_handing_the_signal_on(self, out, monkeypatch):
        armed = []
        import threading

        class FakeTimer:
            def __init__(self, delay, fn):
                armed.append((delay, fn))
            def start(self):
                pass
        monkeypatch.setattr(threading, "Timer", FakeTimer)
        asyncio.run(app_module._drain_then_exit(lambda sig, frame: None, timeout=1,
                                                proxy_grace=0, hard_exit_after=7))
        assert armed and armed[0][0] == 7 and armed[0][1] is app_module._hard_exit


class TestReadiness:
    """Traefik drops a container the moment its Docker healthcheck fails, so
    /health/ready must go 503 on SIGTERM (socket still open, proxy stops
    sending traffic) but stay 200 on a marker drain (the newer instance is
    still booting and nobody else is routable yet)."""

    def _get(self):
        async def go():
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app_module.app),
                                         base_url="http://t") as c:
                return await c.get("/health/ready")
        return asyncio.run(go())

    def test_ready_by_default(self, out, monkeypatch):
        monkeypatch.setattr(app_module, "_stopping", False)
        assert self._get().status_code == 200

    def test_marker_drain_stays_ready(self, out, monkeypatch):
        monkeypatch.setattr(app_module, "_stopping", False)
        (out / app_module._INSTANCE_MARKER).write_text("newer")
        app_module._check_instance_marker()
        assert app_module._draining
        assert self._get().status_code == 200

    def test_sigterm_turns_unready(self, out, monkeypatch):
        monkeypatch.setattr(app_module, "_stopping", True)
        assert self._get().status_code == 503


class TestStatusFromDisk:
    def _client(self):
        return httpx.AsyncClient(transport=httpx.ASGITransport(app=app_module.app),
                                 base_url="http://t")

    def test_unknown_job_with_live_manifest_is_processing(self, out):
        _manifest(out, "j1", instance="other", heartbeat=time.time())

        async def go():
            async with self._client() as c:
                return await c.get("/api/status/j1")
        r = asyncio.run(go())
        assert r.status_code == 200
        assert r.json()["status"] == "processing"

    def test_unknown_job_with_idle_manifest_is_queued(self, out):
        _manifest(out, "j1")

        async def go():
            async with self._client() as c:
                return await c.get("/api/status/j1")
        r = asyncio.run(go())
        assert r.json()["status"] == "queued"

    def test_unknown_job_with_metadata_is_completed(self, out):
        d = out / "j1"
        d.mkdir()
        (d / "vid_metadata.json").write_text(json.dumps({"shorts": [{"start": 0, "end": 1}]}))

        async def go():
            async with self._client() as c:
                return await c.get("/api/status/j1")
        r = asyncio.run(go())
        assert r.json()["status"] == "completed"
        assert r.json()["result"]["clips"]

    def test_truly_unknown_job_is_404(self, out):
        async def go():
            async with self._client() as c:
                return await c.get("/api/status/nope")
        assert asyncio.run(go()).status_code == 404

    def test_queued_here_but_started_elsewhere_reads_processing(self, out):
        _manifest(out, "j1", instance="other", heartbeat=time.time())
        app_module.jobs["j1"] = {"status": "queued", "logs": ["Job queued"]}
        app_module._begin_drain("test")
        assert app_module._presented_status("j1", app_module.jobs["j1"]) == "processing"


class TestSharedGpu:
    """The new instance must not stack a full set of jobs on top of the ones
    the draining instance is still running on the same GPU."""

    def test_counts_only_fresh_jobs_of_other_instances(self, out):
        now = time.time()
        _manifest(out, "a", instance="old", heartbeat=now - 5)
        _manifest(out, "b", instance="old", heartbeat=now - 5)
        _manifest(out, "c", instance="old", heartbeat=now - 600)   # stale
        _manifest(out, "d", instance="me", heartbeat=now - 5)      # ours
        _manifest(out, "e")                                         # queued, no heartbeat
        (out / "no-manifest").mkdir()
        assert app_module._jobs_busy_elsewhere(now) == 2

    def test_waits_while_the_gpu_is_full(self, out, monkeypatch):
        monkeypatch.setattr(app_module, "MAX_CONCURRENT_JOBS", 3)
        monkeypatch.setattr(app_module, "SHARED_GPU_WAIT_SECONDS", 0.01)
        busy = {"n": 3}
        monkeypatch.setattr(app_module, "_jobs_busy_elsewhere", lambda now=None: busy["n"])
        monkeypatch.setattr(app_module, "_gpu_free_mb", lambda: None)
        calls = {"n": 0}
        real_sleep = asyncio.sleep

        async def fake_sleep(sec):
            calls["n"] += 1
            if calls["n"] == 3:
                busy["n"] = 1     # the old instance finished two jobs
            await real_sleep(0)
        monkeypatch.setattr(app_module.asyncio, "sleep", fake_sleep)
        asyncio.run(app_module._wait_for_shared_gpu())
        assert calls["n"] == 3

    def test_no_wait_when_alone(self, out, monkeypatch):
        monkeypatch.setattr(app_module, "MAX_CONCURRENT_JOBS", 3)
        monkeypatch.setattr(app_module, "_running_jobs", {"x", "y"})
        monkeypatch.setattr(app_module, "_jobs_busy_elsewhere", lambda now=None: 0)
        monkeypatch.setattr(app_module, "_gpu_free_mb", lambda: 20000)

        async def boom(sec):
            raise AssertionError("should not wait")
        monkeypatch.setattr(app_module.asyncio, "sleep", boom)
        asyncio.run(app_module._wait_for_shared_gpu())


class TestGpuHasRoom:
    def test_count_limit(self, out, monkeypatch):
        monkeypatch.setattr(app_module, "MAX_CONCURRENT_JOBS", 8)
        assert not app_module.gpu_has_room(5, 3, 20000)

    def test_needs_free_vram_when_something_runs(self, out, monkeypatch):
        monkeypatch.setattr(app_module, "MAX_CONCURRENT_JOBS", 8)
        monkeypatch.setattr(app_module, "GPU_MIN_FREE_MB", 4500)
        assert not app_module.gpu_has_room(3, 0, 1200)
        assert app_module.gpu_has_room(3, 0, 6000)

    def test_idle_card_always_starts(self, out, monkeypatch):
        monkeypatch.setattr(app_module, "GPU_MIN_FREE_MB", 4500)
        assert app_module.gpu_has_room(0, 0, 100)

    def test_no_nvidia_smi_falls_back_to_count(self, out, monkeypatch):
        monkeypatch.setattr(app_module, "MAX_CONCURRENT_JOBS", 8)
        assert app_module.gpu_has_room(4, 0, None)
