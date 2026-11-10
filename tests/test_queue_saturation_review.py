"""Saturation mode, second pass: country from a local database, expiry only of
jobs that never ran (claimed once across instances), top-up buyers, the kill
switch folding the slow lane back, and the queue watch under a busy lock.

Codes ``XA``/``XB`` are ISO user-assigned placeholders, not real countries.
"""
import asyncio
import hashlib
import json
import os
import time
from types import SimpleNamespace

import pytest

from test_queue_saturation import (  # noqa: F401  (fixtures)
    CLIENT, FREE, SLOW, H, _job, _order, _peek_order, _set_brake, app_module,
    env, geoip, out, qm, qw, sat)


@pytest.fixture
def hooks(monkeypatch):
    calls = {"settle": [], "webhook": [], "autopilot": []}

    async def settle(jid, job=None):
        calls["settle"].append(jid)

    async def webhook(jid):
        calls["webhook"].append(jid)

    async def autopilot(jid, job):
        calls["autopilot"].append(jid)
    monkeypatch.setattr(app_module, "_settle_reservation", settle)
    monkeypatch.setattr(app_module, "_notify_job_webhook", webhook)
    monkeypatch.setattr(app_module, "_autopilot_job_finished", autopilot)
    return calls




def _old(now):
    return now - app_module.ACTIVE_JOB_MAX_SECONDS - 60


def _manifest(out, jid, **m):
    d = out / jid
    d.mkdir(exist_ok=True)
    base = {"cmd": ["python", "main.py"], "priority": FREE, "user_id": "u",
            "reservation_id": f"r-{jid}", "attempts": 0}
    base.update(m)
    (d / app_module._RESUME_FILE).write_text(json.dumps(base))
    return d


# --- 1. Country: local database, header off by default -----------------------------

class TestGeoIP:
    def test_private_and_bad_addresses_are_unknown(self, env):
        for ip in ("10.1.2.3", "192.168.0.1", "127.0.0.1", "::1", "fe80::1",
                   "169.254.1.1", "100.64.0.1", "not-an-ip", "", None):
            assert geoip.country_of(ip) == ""

    def test_no_database_is_unknown_never_an_error(self, env, monkeypatch):
        monkeypatch.setattr(geoip, "_reader", None)
        assert geoip.country_of(CLIENT["XA"]) == ""
        assert qm.request_country({}, CLIENT["XA"]) == ""

    def test_reader_errors_are_unknown(self, env, monkeypatch):
        class Broken:
            def get(self, ip):
                raise ValueError("corrupt tree")
        monkeypatch.setattr(geoip, "_reader", Broken())
        assert geoip.country_of(CLIENT["XA"]) == ""

    def test_corrupt_file_is_not_loaded(self, env, tmp_path, monkeypatch):
        pytest.importorskip("maxminddb")
        monkeypatch.setattr(geoip, "_reader", None)
        p = tmp_path / "country.mmdb"
        p.write_bytes(b"\x00" * 4096)
        monkeypatch.setenv("GEOIP_COUNTRY_DB_PATH", str(p))
        assert geoip.reload_if_changed() is False
        assert not geoip.loaded() and geoip.country_of(CLIENT["XA"]) == ""

    def test_missing_file_keeps_unknown(self, env, tmp_path, monkeypatch):
        monkeypatch.setattr(geoip, "_reader", None)
        monkeypatch.setenv("GEOIP_COUNTRY_DB_PATH", str(tmp_path / "none.mmdb"))
        assert geoip.reload_if_changed() is False and not geoip.loaded()

    def test_refresh_checks_the_published_checksum(self, env, tmp_path, monkeypatch):
        p = tmp_path / "geo" / "country.mmdb"
        monkeypatch.setenv("GEOIP_COUNTRY_DB_PATH", str(p))
        monkeypatch.setenv("GEOIP_COUNTRY_DB_URL", "https://example.invalid/db.mmdb")
        monkeypatch.setenv("GEOIP_COUNTRY_DB_SHA256_URL", "https://example.invalid/db.sha256")
        payload = b"payload"

        def fetch(url):
            return payload if url.endswith(".mmdb") else b"0" * 64 + b"  db.mmdb\n"
        assert geoip.refresh(fetch=fetch) is False
        assert not p.exists()

    def test_refresh_refuses_an_unusable_file(self, env, tmp_path, monkeypatch):
        p = tmp_path / "geo" / "country.mmdb"
        monkeypatch.setenv("GEOIP_COUNTRY_DB_PATH", str(p))
        monkeypatch.setenv("GEOIP_COUNTRY_DB_URL", "https://example.invalid/db.mmdb")
        monkeypatch.setenv("GEOIP_COUNTRY_DB_SHA256_URL", "")
        data = b"not an mmdb"
        assert geoip.refresh(fetch=lambda url: data) is False
        assert not p.exists()
        assert [f for f in os.listdir(p.parent) if f.endswith(".part")] == []

    def test_fresh_file_is_not_downloaded_again(self, env, tmp_path, monkeypatch):
        p = tmp_path / "country.mmdb"
        p.write_bytes(b"x")
        monkeypatch.setenv("GEOIP_COUNTRY_DB_PATH", str(p))

        def fetch(url):
            raise AssertionError("no download for a fresh file")
        assert geoip.refresh(fetch=fetch) is False

    def test_no_url_no_download(self, env, tmp_path, monkeypatch):
        monkeypatch.setenv("GEOIP_COUNTRY_DB_PATH", str(tmp_path / "c.mmdb"))
        monkeypatch.setenv("GEOIP_COUNTRY_DB_URL", "")
        assert geoip.refresh(fetch=lambda u: 1 / 0) is False

    def test_real_database_when_available(self, env, tmp_path, monkeypatch):
        """With a real country MMDB on this machine (GEOIP_TEST_DB), the whole
        path: checksum, validation, atomic replace, reload, lookup."""
        src = os.environ.get("GEOIP_TEST_DB", "")
        if not src or not os.path.exists(src):
            pytest.skip("no GEOIP_TEST_DB")
        pytest.importorskip("maxminddb")
        data = open(src, "rb").read()
        p = tmp_path / "geo" / "country.mmdb"
        monkeypatch.setenv("GEOIP_COUNTRY_DB_PATH", str(p))
        monkeypatch.setenv("GEOIP_COUNTRY_DB_URL", "https://example.invalid/db.mmdb")
        monkeypatch.setenv("GEOIP_COUNTRY_DB_SHA256_URL", "https://example.invalid/db.sha256")
        digest = hashlib.sha256(data).hexdigest().encode()

        def fetch(url):
            return data if url.endswith(".mmdb") else digest + b"  db.mmdb\n"
        monkeypatch.setattr(geoip, "_reader", None)
        monkeypatch.setattr(geoip, "_loaded_mtime", None)
        assert geoip.refresh(fetch=fetch) is True
        assert geoip.reload_if_changed() is True
        assert len(geoip.country_of("8.8.8.8")) == 2

    def test_loop_does_nothing_unless_wanted(self, env, monkeypatch):
        calls = []
        monkeypatch.setattr(geoip, "refresh", lambda: calls.append(1))
        monkeypatch.setattr(geoip, "CHECK_EVERY", 0.01)

        async def run():
            t = asyncio.create_task(geoip._loop(lambda: False))
            await asyncio.sleep(0.05)
            t.cancel()
        asyncio.run(run())
        assert calls == []


# --- 2. Expiry only of jobs that never ran ---------------------------------------------

class TestExpiryOnlyUnrun:
    def test_drain_cut_job_is_resumed_not_expired(self, sat, out, hooks):
        now = time.time()
        d = _manifest(out, "ran", submitted_at=_old(now), run_started=now - 120,
                      heartbeat=now - 90, instance="old", resumed_at=now - 2000)
        (d / "x_metadata.json").write_text("{}")   # clips partly rendered
        app_module._resume_interrupted_jobs()
        assert app_module.jobs["ran"]["status"] == "queued"
        assert asyncio.run(app_module._expire_starved_jobs(now)) == []
        assert app_module.jobs["ran"]["status"] == "queued"
        assert hooks["settle"] == []

    def test_auto_retried_job_is_not_expired(self, sat, out, hooks):
        now = time.time()
        _job(out, "retry", FREE, _old(now))
        app_module.jobs["retry"]["auto_retries"] = 1
        assert asyncio.run(app_module._expire_starved_jobs(now)) == []

    def test_job_dispatched_here_is_not_expired(self, sat, out, hooks):
        now = time.time()
        _job(out, "started", FREE, _old(now))
        app_module.jobs["started"]["run_started"] = now - 10
        assert asyncio.run(app_module._expire_starved_jobs(now)) == []

    def test_recovery_respects_the_marker(self, sat, out, hooks):
        now = time.time()
        _job(out, "gone", FREE, _old(now))
        (out / "gone" / "x_metadata.json").write_text('{"shorts": []}')
        assert asyncio.run(app_module._expire_starved_jobs(now)) == ["gone"]
        app_module.jobs.clear()
        app_module._recover_jobs_from_disk()
        assert "gone" not in app_module.jobs                 # never "completed"
        assert app_module._job_view_from_disk("gone")["status"] == "failed"

    def test_real_settle_releases_the_reservation(self, sat, out, monkeypatch):
        released, committed = [], []

        async def release(rid):
            released.append(rid)

        async def commit(rid):
            committed.append(rid)

        async def noop(*a, **k):
            return None
        monkeypatch.setattr(app_module, "cloud", SimpleNamespace(
            metering=SimpleNamespace(release_reservation=release, commit_reservation=commit)))
        monkeypatch.setattr(app_module, "_notify_job_webhook", noop)
        monkeypatch.setattr(app_module, "_autopilot_job_finished", noop)
        now = time.time()
        _job(out, "old", SLOW, _old(now))
        assert asyncio.run(app_module._expire_starved_jobs(now)) == ["old"]
        assert released == ["r-old"] and committed == []


# --- 3. Top-up buyers ---------------------------------------------------------------------

class TestTopUp:
    def test_topup_job_never_expires_nor_reorders(self, sat, out, hooks):
        now = time.time()
        _job(out, "topup", qm.PAID_PRIORITY, _old(now), user="buyer")
        app_module.jobs["topup"]["topup"] = True
        _job(out, "free_new", FREE, now - 5)
        s = qm.current()
        s["lanes"]["normal"]["recent_first"] = True
        qm.adopt(s)
        assert asyncio.run(app_module._expire_starved_jobs(now)) == []
        assert _peek_order() == ["topup", "free_new"]

    def test_topup_flag_travels_in_the_manifest(self, sat, out):
        (out / "t1").mkdir()
        app_module._write_resume_manifest("t1", ["python"], qm.PAID_PRIORITY, "u", "r", True,
                                          submitted_at=1.0, topup=True)
        m = json.loads((out / "t1" / app_module._RESUME_FILE).read_text())
        assert m["topup"] is True and m["priority"] == qm.PAID_PRIORITY
        app_module._resume_interrupted_jobs()
        assert app_module.jobs["t1"]["topup"] is True
        assert app_module.job_queue._queue[0][0] == qm.PAID_PRIORITY


# --- 4. Kill switch: the slow lane folds back ---------------------------------------------

class TestKillSwitch:
    def test_resumed_slow_jobs_join_the_normal_lane(self, sat, out, hooks, monkeypatch):
        now = time.time()
        _manifest(out, "slow_old", priority=SLOW, lane=qm.SLOW, submitted_at=now - H)
        monkeypatch.setenv("QUEUE_SATURATION_DISABLED", "1")
        app_module._resume_interrupted_jobs()
        _job(out, "free_new", FREE, now)
        assert _peek_order() == ["slow_old", "free_new"]
        assert app_module.jobs["slow_old"]["lane"] == qm.NORMAL

    def test_empty_list_folds_at_enqueue(self, sat, out, monkeypatch):
        monkeypatch.setenv("FREE_SLOW_LANE_COUNTRIES", "")
        _job(out, "a", SLOW, 1.0)
        assert app_module.job_queue._queue[0][0] == FREE

    def test_live_fold_of_the_queue(self, sat, out, monkeypatch):
        _job(out, "slow1", SLOW, 1.0)
        _job(out, "free1", FREE, 2.0)
        _job(out, "slow2", SLOW, 3.0)
        assert _peek_order() == ["free1", "slow1", "slow2"]
        assert app_module._fold_slow_lane() == 0          # lane still on
        monkeypatch.setenv("QUEUE_SATURATION_DISABLED", "1")
        assert app_module._fold_slow_lane() == 2
        assert _order() == ["slow1", "free1", "slow2"]

    def test_kill_switch_turns_expiry_off(self, sat, out, hooks, monkeypatch):
        now = time.time()
        _job(out, "old", FREE, _old(now))
        monkeypatch.setenv("QUEUE_SATURATION_DISABLED", "1")
        assert asyncio.run(app_module._expire_starved_jobs(now)) == []


# --- 7. One expiry across instances ---------------------------------------------------------

class TestClaim:
    def test_claimed_elsewhere_settles_nothing_here(self, sat, out, hooks):
        now = time.time()
        _job(out, "both", FREE, _old(now))
        (out / "both" / app_module._EXPIRED_MARKER).write_text("1")   # the other one won
        assert asyncio.run(app_module._expire_starved_jobs(now)) == []
        assert app_module.jobs["both"]["status"] == "failed"
        assert hooks["settle"] == [] and hooks["webhook"] == []
        assert app_module.job_queue.qsize() == 0

    def test_two_expiries_one_release(self, sat, out, hooks):
        now = time.time()
        _job(out, "j", FREE, _old(now))
        first = asyncio.run(app_module._expire_starved_jobs(now))
        # The second instance had the same job queued.
        app_module.jobs["j"]["status"] = "queued"
        app_module._enqueue_job("j", FREE)
        second = asyncio.run(app_module._expire_starved_jobs(now))
        assert first == ["j"] and second == []
        assert hooks["settle"] == ["j"] and hooks["webhook"] == ["j"]

    def test_backs_off_when_another_instance_runs_it(self, sat, out, hooks):
        now = time.time()
        _job(out, "busy", FREE, _old(now))
        m = json.loads((out / "busy" / app_module._RESUME_FILE).read_text())
        m.update(instance="other", heartbeat=now, run_started=now)
        (out / "busy" / app_module._RESUME_FILE).write_text(json.dumps(m))
        assert asyncio.run(app_module._expire_starved_jobs(now)) == []
        assert not (out / "busy" / app_module._EXPIRED_MARKER).exists()
        assert app_module.jobs["busy"]["status"] == "queued"
        assert _peek_order() == ["busy"]

    def test_dispatch_refuses_a_job_claimed_elsewhere(self, sat, out, monkeypatch):
        _job(out, "claimed", FREE, 1.0)
        _job(out, "fine", FREE, 2.0)
        (out / "claimed" / app_module._EXPIRED_MARKER).write_text("1")
        started = []

        async def no_wait():
            return None

        async def fake_run(job_id):
            started.append(job_id)
        sem = asyncio.Semaphore(10)
        monkeypatch.setattr(app_module, "_wait_for_shared_gpu", no_wait)
        monkeypatch.setattr(app_module, "run_job_wrapper", fake_run)
        monkeypatch.setattr(app_module, "concurrency_semaphore", sem)

        async def run():
            worker = asyncio.create_task(app_module.process_queue())
            await asyncio.sleep(0.1)
            worker.cancel()
        asyncio.run(run())
        assert started == ["fine"]
        assert app_module.jobs["claimed"]["status"] == "failed"
        assert "claimed" not in app_module._running_jobs


# --- 5, 6. Queue watch under a busy lock; 8. ASR incident ---------------------------------

class Busy:
    """Holds the queue watch's lock like the other container would."""

    def __init__(self, d):
        import fcntl
        self.fh = open(os.path.join(d, qw.LOCK_FILE), "a")
        fcntl.flock(self.fh, fcntl.LOCK_EX | fcntl.LOCK_NB)

    def release(self):
        self.fh.close()


@pytest.fixture
def world(sat, tmp_path, monkeypatch):
    for k in ("QUEUE_ALERTS_DISABLED", "ADMIN_EMAIL"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "x")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "y")
    w = SimpleNamespace(active=True, sent=[], dir=str(tmp_path))
    w.snap = {"waiting": [], "running": 5, "slots": 5, "done_last_hour": 40,
              "eta_free": 0, "eta_paid": 0,
              "lanes": {"normal": {"queued": 10, "wait": 600},
                        "slow": {"queued": 300, "wait": 13 * H}}}
    from cloud import alerts

    async def fake_send(subject, body):
        w.sent.append((subject, body))
    monkeypatch.setattr(alerts, "send_admin_alert", fake_send)
    qw.configure(snapshot=lambda: dict(w.snap), busy_elsewhere=lambda: 0,
                 state_dir=w.dir, is_active=lambda: w.active, now=0)
    return w


class TestBusyLock:
    def test_boot_adopts_the_shared_modes_with_the_lock_taken(self, world):
        on, _ = qm.decide(qm.default_state(), {"slow": {"wait": 13 * H}}, 1.0)
        qm.save(world.dir, on)
        qm.reset()
        busy = Busy(world.dir)
        try:
            qw.follow_modes()
            assert qm.brake_active()
        finally:
            busy.release()

    def test_refusals_wait_for_the_lock_and_are_not_lost(self, world):
        world.active = False
        busy = Busy(world.dir)
        try:
            qm.note_rejected()
            qw.follow_modes()
            assert qm.has_pending()
        finally:
            busy.release()
        qw.follow_modes()
        assert not qm.has_pending()
        assert json.load(open(os.path.join(world.dir, qm.STATE_FILE)))["brake"]["rejected"] == 1

    def test_brake_alert_survives_a_busy_second_lock(self, world, monkeypatch):
        real = qw._try_lock
        calls = {"n": 0}

        def second_busy():
            calls["n"] += 1
            return real() if calls["n"] == 1 else None
        monkeypatch.setattr(qw, "_try_lock", second_busy)
        kind = asyncio.run(qw.tick(now=100_000.0))
        assert kind == "brake-on"
        assert any("Freno" in s for s, _ in world.sent)
        assert qm.brake_active()

    def test_busy_first_lock_decides_next_tick(self, world):
        busy = Busy(world.dir)
        try:
            asyncio.run(qw.tick(now=100_000.0))
            assert not qm.brake_active() and world.sent == []
        finally:
            busy.release()
        assert "brake-on" in asyncio.run(qw.tick(now=100_200.0))


class TestAsrIncident:
    def test_missing_gpu_reading_does_not_close_it(self, env):
        lim = qw.limits()
        snap = {"gpu": True, "asr_devices": ["cpu"] * 3}
        kind, sub = qw.asr_incident({}, snap, 0, lim)
        assert kind == "alert"
        for t in (1, 2, 3):
            kind, sub = qw.asr_incident(sub, {"gpu": False, "asr_devices": ["cpu"] * 3}, t, lim)
            assert kind != "recovered" and sub["active"]
        kind, sub = qw.asr_incident(sub, {"gpu": True, "asr_devices": ["cuda"] * 3}, 4, lim)
        kind, sub = qw.asr_incident(sub, {"gpu": True, "asr_devices": ["cuda"] * 3}, 5, lim)
        assert kind == "recovered"

    def test_cpu_on_purpose_still_closes_it(self, env, monkeypatch):
        lim = qw.limits()
        kind, sub = qw.asr_incident({}, {"gpu": True, "asr_devices": ["cpu"] * 3}, 0, lim)
        monkeypatch.setenv("PARAKEET_DEVICE", "cpu")
        kind, sub = qw.asr_incident(sub, {"gpu": True, "asr_devices": ["cpu"] * 3}, 1, lim)
        kind, sub = qw.asr_incident(sub, {"gpu": True, "asr_devices": ["cpu"] * 3}, 2, lim)
        assert kind == "recovered"


class TestStatusOfAnExpiredJob:
    """With the dashboard's poll (lib/jobPoll.js: a 404 means "gone", anything
    else is retried), an expired job must answer 200 + failed + the reason on
    every instance, so the panel stops polling and shows that reason, not the
    "no longer open" banner."""

    def test_other_instance_answers_failed_with_the_reason(self, sat, out, hooks, monkeypatch):
        from fastapi.testclient import TestClient
        now = time.time()
        _job(out, "gone1", SLOW, _old(now))
        assert asyncio.run(app_module._expire_starved_jobs(now)) == ["gone1"]
        app_module.jobs.clear()                         # the instance that never held it
        monkeypatch.setattr(app_module, "BILLING_ENABLED", False)  # no auth in this test
        r = TestClient(app_module.app).get("/api/status/gone1")
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "failed" and body["queue"] is None
        assert body["logs"][-1] == app_module.QUEUE_EXPIRED_LOG
