"""Render-queue watch: alert, reminder, recovery, hysteresis, kill switch, handover.

Fake clock and fake queue; nothing leaves the process (send_admin_alert is
replaced)."""
import asyncio
import json

import pytest

qw = pytest.importorskip("cloud.queue_watch")
alerts = pytest.importorskip("cloud.alerts")

MIN = 60
T0 = 100_000.0  # well past the boot hold of configure(now=0)


class World:
    """A queue the test edits by hand: id -> (paid, waiting since)."""

    def __init__(self):
        self.waiting = {}
        self.running = 5
        self.elsewhere = 0
        self.done = 40

    def add(self, n, *, paid, since, prefix=None):
        prefix = prefix or ("p" if paid else "f")
        start = len(self.waiting)
        for i in range(n):
            self.waiting[f"{prefix}{start + i}"] = (paid, since)

    def snapshot(self):
        return {
            "waiting": [{"id": k, "paid": v[0], "attempt": 0} for k, v in self.waiting.items()],
            "running": self.running, "slots": 5, "done_last_hour": self.done,
            "eta_free": 3600, "eta_paid": 600,
        }

    def wait_started(self, job_id):
        return self.waiting[job_id][1]


@pytest.fixture()
def world(monkeypatch, tmp_path):
    for k in ("QUEUE_ALERTS_DISABLED", "QUEUE_ALERT_PAID_WAIT_MINUTES",
              "QUEUE_ALERT_FREE_WAIT_MINUTES", "QUEUE_ALERT_MAX_QUEUED",
              "QUEUE_ALERT_REMIND_MINUTES", "ADMIN_EMAIL"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("BILLING_ENABLED", "1")
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "x")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "y")
    w = World()
    w.sent = []

    async def fake_send(subject, body):
        w.sent.append((subject, body))
    monkeypatch.setattr(alerts, "send_admin_alert", fake_send)
    w.active = True
    w.dir = str(tmp_path)
    _configure(w)
    return w


def _configure(w, now=0):
    qw.configure(snapshot=w.snapshot, wait_started=w.wait_started,
                 busy_elsewhere=lambda: w.elsewhere, state_dir=w.dir,
                 is_active=lambda: w.active, now=now)


def tick(now):
    return asyncio.run(qw.tick(now=now))


def test_healthy_queue_says_nothing(world):
    world.add(3, paid=True, since=T0 - 2 * MIN)
    world.add(20, paid=False, since=T0 - 10 * MIN)
    assert tick(T0) is None
    assert world.sent == []


def test_paid_job_waiting_too_long_alerts_once(world):
    world.add(30, paid=False, since=T0 - 20 * MIN)
    world.add(2, paid=True, since=T0 - 12 * MIN)
    assert tick(T0) == "alert"
    subject, body = world.sent[-1]
    assert "atascada" in subject
    assert "vídeos de pago esperando más de 10 min" in body
    assert "En cola: 32 vídeos (2 de pago, 2 esperan más de 10 min)." in body
    assert "de pago 12 min, gratis 20 min" in body
    assert "40 vídeos terminados en la última hora" in body
    # Same incident, a few minutes later: silent.
    assert tick(T0 + 3 * MIN) is None
    assert tick(T0 + 30 * MIN) is None
    assert len(world.sent) == 1


def test_reminder_every_hour_with_trend(world):
    world.add(2, paid=True, since=T0 - 15 * MIN)
    world.add(100, paid=False, since=T0 - 15 * MIN)
    assert tick(T0) == "alert"
    world.add(60, paid=False, since=T0 + 30 * MIN)
    assert tick(T0 + 59 * MIN) is None
    assert tick(T0 + 61 * MIN) == "remind"
    subject, body = world.sent[-1]
    assert "sigue atascada (desde hace 1 h 1 min)" in subject
    assert body.startswith("Va a peor: 102 → 162 vídeos en cola")
    # Next one only an hour after the reminder, and it says it got better.
    for k in list(world.waiting)[:100]:
        del world.waiting[k]
    assert tick(T0 + 100 * MIN) is None
    assert tick(T0 + 122 * MIN) == "remind"
    assert world.sent[-1][1].startswith("Va a mejor: 162 → 62")


def test_old_free_job_alerts(world):
    world.add(5, paid=False, since=T0 - 31 * MIN)
    assert tick(T0) == "alert"
    assert "el vídeo gratis más antiguo lleva más de 30 min" in world.sent[-1][1]
    assert "ninguno de pago" in world.sent[-1][1]


def test_absurd_queue_size_alerts(world, monkeypatch):
    monkeypatch.setenv("QUEUE_ALERT_MAX_QUEUED", "50")
    world.add(50, paid=False, since=T0 - 1 * MIN)
    assert tick(T0) == "alert"
    assert "más de 50 vídeos en cola" in world.sent[-1][1]


def test_recovery_needs_two_calm_ticks_and_clears_below_half(world):
    world.add(1, paid=True, since=T0 - 11 * MIN)
    assert tick(T0) == "alert"
    # The paid job starts; the free backlog sits between half and the
    # threshold (20 min vs 30): not an alert, not a recovery either.
    world.waiting.clear()
    world.add(10, paid=False, since=T0 - 20 * MIN)
    assert tick(T0 + 3 * MIN) is None
    assert tick(T0 + 6 * MIN) is None
    # Under half everywhere: the first calm tick is not enough...
    world.waiting.clear()
    world.add(2, paid=False, since=T0 + 5 * MIN, prefix="g")
    assert tick(T0 + 9 * MIN) is None
    # ...a blip back over half resets the streak...
    world.add(1, paid=True, since=T0 + 3 * MIN, prefix="q")
    assert tick(T0 + 12 * MIN) is None
    world.waiting.pop("q2")
    assert tick(T0 + 15 * MIN) is None
    # ...and two calm ticks in a row close it.
    assert tick(T0 + 18 * MIN) == "recovered"
    subject, body = world.sent[-1]
    assert subject == "✅ Cola de render recuperada"
    assert body.startswith("Ha estado atascada 18 min.")
    assert len(world.sent) == 2
    assert tick(T0 + 21 * MIN) is None


def test_hovering_at_the_threshold_does_not_flap(world):
    world.add(1, paid=False, since=T0 - 31 * MIN)
    assert tick(T0) == "alert"
    sent = 1
    for i in range(1, 10):
        # oldest free alternates just over / just under 30 min
        world.waiting["f0"] = (False, T0 + i * 3 * MIN - (31 if i % 2 else 29) * MIN)
        qw._since_cache.clear()
        tick(T0 + i * 3 * MIN)
    assert len(world.sent) == sent


def test_kill_switch(world, monkeypatch):
    monkeypatch.setenv("QUEUE_ALERTS_DISABLED", "1")
    world.add(1, paid=True, since=T0 - 60 * MIN)
    assert tick(T0) is None
    assert world.sent == []


def test_no_channel_or_no_billing_does_nothing(world, monkeypatch):
    world.add(1, paid=True, since=T0 - 60 * MIN)
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN")
    assert tick(T0) is None
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "x")
    monkeypatch.setenv("BILLING_ENABLED", "0")
    assert tick(T0) is None
    assert world.sent == []


def test_start_does_not_spawn_when_disabled(world, monkeypatch):
    monkeypatch.setenv("QUEUE_ALERTS_DISABLED", "1")
    qw._task = None
    qw.start(snapshot=world.snapshot)  # no running loop needed: returns first
    assert qw._task is None


def test_draining_instance_stays_quiet(world):
    world.add(1, paid=True, since=T0 - 60 * MIN)
    world.active = False
    assert tick(T0) is None
    assert world.sent == []


def test_handover_holds_then_judges(world):
    world.add(1, paid=True, since=T0 - 12 * MIN)
    world.elsewhere = 3  # the old container is still finishing its jobs
    assert tick(T0) is None
    world.elsewhere = 0
    assert tick(T0 + 3 * MIN) is None          # still within the grace window
    assert tick(T0 + 10 * MIN) == "alert"      # grace over, still waiting


def test_boot_hold(world):
    _configure(world, now=T0)
    world.add(1, paid=True, since=T0 - 60 * MIN)
    assert tick(T0 + 5 * MIN) is None
    assert tick(T0 + 11 * MIN) == "alert"


def test_new_instance_continues_the_incident(world):
    world.add(1, paid=True, since=T0 - 12 * MIN)
    assert tick(T0) == "alert"
    with open(f"{world.dir}/{qw.STATE_FILE}") as f:
        assert json.load(f)["active"] is True
    # Deploy: a new process with empty memory and the same shared dir.
    _configure(world, now=T0)
    assert tick(T0 + 20 * MIN) is None         # no second "atascada"
    assert tick(T0 + 61 * MIN) == "remind"     # reminder keeps the old clock
    world.waiting.clear()
    tick(T0 + 64 * MIN)
    assert tick(T0 + 67 * MIN) == "recovered"  # and the new one closes it
    assert [s[0].split()[0] for s in world.sent] == ["⚠️", "⚠️", "✅"]


def test_other_instance_holding_the_lock_skips(world):
    import fcntl
    world.add(1, paid=True, since=T0 - 12 * MIN)
    with open(f"{world.dir}/{qw.LOCK_FILE}", "a") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        assert tick(T0) is None
    assert tick(T0 + 3 * MIN) == "alert"


def test_retried_job_waits_from_its_retry(world):
    world.add(1, paid=True, since=T0 - 5 * MIN)
    assert tick(T0) is None
    # It ran, failed and was re-queued (attempt 1): its wait is read again.
    snap = world.snapshot

    def retried():
        s = snap()
        s["waiting"][0]["attempt"] = 1
        return s
    qw._snapshot = retried
    world.waiting["p0"] = (True, T0 + 8 * MIN)
    assert tick(T0 + 12 * MIN) is None  # 4 min since the retry, not 17


def test_fmt_duration():
    assert qw.fmt_duration(30) == "menos de 1 min"
    assert qw.fmt_duration(7 * MIN) == "7 min"
    assert qw.fmt_duration(72 * MIN) == "1 h 12 min"
    assert qw.fmt_duration(3 * 3600) == "3 h"
    assert qw.fmt_duration(8 * 86400 + 5 * 3600) == "8 d 5 h"


# --- the app.py side ----------------------------------------------------------

def test_app_wait_started_prefers_last_run(tmp_path, monkeypatch):
    app = pytest.importorskip("app")
    monkeypatch.setattr(app, "OUTPUT_DIR", str(tmp_path))
    d = tmp_path / "job1"
    d.mkdir()
    (d / ".owner").write_text("u")
    import os
    os.utime(d / ".owner", (1000, 1000))
    monkeypatch.setitem(app.jobs, "job1", {"status": "queued", "output_dir": str(d)})
    assert app._queue_wait_started("job1") == 1000
    (d / app._RESUME_FILE).write_text(json.dumps({"heartbeat": 5000}))
    assert app._queue_wait_started("job1") == 5000


def test_app_snapshot_counts_paid_and_admitting(monkeypatch):
    app = pytest.importorskip("app")

    class Plans:
        PLAN_PRIORITY = {"pro": 0, "creator": 1, "starter": 1, "free": 2}
    monkeypatch.setattr(app, "_cloud_config", Plans)
    q = asyncio.PriorityQueue()
    for p, jid in ((0, "a"), (2, "b"), (1, "gone")):
        q.put_nowait((p, 0, jid))
    monkeypatch.setattr(app, "job_queue", q)
    monkeypatch.setattr(app, "_admitting", {"c"})
    monkeypatch.setattr(app, "_running_jobs", {"r1", "r2"})
    monkeypatch.setattr(app, "jobs", {
        "a": {"status": "queued"}, "b": {"status": "queued"},
        "c": {"status": "queued", "user_plan": "creator", "output_dir": "/nonexistent"},
        "gone": {"status": "processing"},
        "d1": {"status": "completed", "logs": app._TimedLog(["x", "y"])},
        "d2": {"status": "completed", "logs": app._TimedLog(["♻️ recovered"])},
    })
    snap = app._queue_watch_snapshot()
    paid = {w["id"]: w["paid"] for w in snap["waiting"]}
    assert paid == {"a": True, "b": False, "c": True}
    assert snap["running"] == 2 and snap["done_last_hour"] == 1
    assert snap["eta_free"] >= snap["eta_paid"]


# --- integration with the paying-moves-up path (app.reprioritize_user_jobs) ---

def test_job_raised_to_paid_waits_from_the_raise(world):
    # 20 min on the free line (under its threshold); then its owner pays. As a
    # paid job its wait starts at the raise: keyed on (id, attempt) only, the
    # cached submission time made it "a paid job waiting 24 min" and paged.
    world.add(1, paid=False, since=T0 - 20 * MIN)
    assert tick(T0) is None
    world.waiting["f0"] = (True, T0 + 1 * MIN)   # raised_at, read from disk
    assert tick(T0 + 4 * MIN) is None
    assert [k for k in qw._since_cache if k[0] == "f0"] == [("f0", 0, True)]
    assert world.sent == []


def test_app_raise_counts_as_paid_and_restarts_the_wait(tmp_path, monkeypatch):
    import itertools
    import os
    app = pytest.importorskip("app")
    from cloud import config as cloud_config
    monkeypatch.setattr(app, "OUTPUT_DIR", str(tmp_path))
    monkeypatch.setattr(app, "_cloud_config", cloud_config)
    monkeypatch.setattr(app, "jobs", {})
    monkeypatch.setattr(app, "_running_jobs", set())
    monkeypatch.setattr(app, "_admitting", set())
    monkeypatch.setattr(app, "job_queue", asyncio.PriorityQueue())
    monkeypatch.setattr(app, "_job_seq", itertools.count())
    owner = "a11ce000-0000-0000-0000-000000000000"
    d = tmp_path / "j1"
    d.mkdir()
    (d / ".owner").write_text(owner)
    os.utime(d / ".owner", (1000, 1000))
    (d / app._RESUME_FILE).write_text(json.dumps({"priority": 2, "user_id": owner}))
    app.jobs["j1"] = {"status": "queued", "user_id": owner, "output_dir": str(d), "logs": []}
    app._enqueue_job("j1", 2)

    def paid():
        return {w["id"]: w["paid"] for w in app._queue_watch_snapshot()["waiting"]}

    assert paid() == {"j1": False}
    assert app._queue_wait_started("j1") == 1000
    app.reprioritize_user_jobs(owner, cloud_config.PLAN_PRIORITY["starter"])
    assert paid() == {"j1": True}            # told apart by its queue priority
    started = app._queue_wait_started("j1")
    assert started > 1000                    # the raise, not the submission
    m = json.loads((d / app._RESUME_FILE).read_text())
    assert m["raised_at"] == started and m["priority"] == 1
    assert not [n for n in os.listdir(d) if n.endswith(".tmp")]  # atomic write left no temp
