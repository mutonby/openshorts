"""Queue watch capacity faults: transcription on the CPU and output-rate drop.

Fake clock and fake queue; send_admin_alert is replaced, nothing leaves the
process. The wait alerts are pushed out of the way (huge thresholds) so each
test sees only the fault it is about."""
import asyncio
import itertools
import json

import pytest

qw = pytest.importorskip("cloud.queue_watch")
alerts = pytest.importorskip("cloud.alerts")

MIN = 60
H = 3600
T0 = 1_000_000.0
STEP = 3 * MIN


class World:
    def __init__(self):
        self.slots = 5
        self.running = 5
        self.queued = 1          # jobs waiting (fresh ids, so no wait builds up)
        self.done = 40
        self.elsewhere = 0
        self.gpu = True
        self.asr = []
        self.ids = itertools.count()
        self.now = T0

    def snapshot(self):
        return {
            "waiting": [{"id": f"j{next(self.ids)}", "paid": False, "attempt": 0}
                        for _ in range(self.queued)],
            "running": self.running, "slots": self.slots, "done_last_hour": self.done,
            "eta_free": 600, "eta_paid": 60, "gpu": self.gpu,
            "asr_devices": list(self.asr),
        }


@pytest.fixture()
def world(monkeypatch, tmp_path):
    for k in ("QUEUE_ALERTS_DISABLED", "QUEUE_ALERT_REMIND_MINUTES", "ADMIN_EMAIL",
              "PARAKEET_DEVICE", "QUEUE_ALERT_ASR_CPU_JOBS",
              "QUEUE_ALERT_THROUGHPUT_DROP_PERCENT", "QUEUE_ALERT_THROUGHPUT_WINDOW_MINUTES",
              "QUEUE_ALERT_THROUGHPUT_MIN_HOURS"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("BILLING_ENABLED", "1")
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "x")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "y")
    monkeypatch.setenv("QUEUE_ALERT_FREE_WAIT_MINUTES", "100000")
    monkeypatch.setenv("QUEUE_ALERT_PAID_WAIT_MINUTES", "10")
    monkeypatch.setenv("QUEUE_ALERT_MAX_QUEUED", "100000")
    w = World()
    w.sent = []

    async def fake_send(subject, body):
        w.sent.append((subject, body))
    monkeypatch.setattr(alerts, "send_admin_alert", fake_send)
    w.dir = str(tmp_path)
    configure(w, now=0)
    return w


def configure(w, now):
    qw.configure(snapshot=w.snapshot, busy_elsewhere=lambda: w.elsewhere,
                 state_dir=w.dir, now=now)


def tick(w, now=None):
    w.now = w.now if now is None else now
    return asyncio.run(qw.tick(now=w.now))


def run_for(w, seconds):
    """Tick every STEP for ``seconds``; return the subjects sent meanwhile."""
    before = len(w.sent)
    end = w.now + seconds
    while w.now < end:
        w.now += STEP
        tick(w)
    return [s for s, _ in w.sent[before:]]


def state(w):
    with open(f"{w.dir}/{qw.STATE_FILE}") as f:
        return json.load(f)


# --- transcription on the CPU ------------------------------------------------------

def test_cpu_transcription_alerts_at_once_even_right_after_a_deploy(world):
    configure(world, now=T0)               # fresh instance: the wait hold is on
    world.asr = ["cuda", "cpu", "cpu"]
    tick(world, T0 + MIN)
    assert world.sent == []                 # two in a row is not yet a pattern
    world.asr.append("cpu")
    tick(world, T0 + 2 * MIN)
    subject, body = world.sent[-1]
    assert subject == "⚠️ La transcripción va por CPU, no por GPU"
    assert body.startswith("Los últimos 3 vídeos se han transcrito en CPU")
    assert "avería del despliegue" in body and "Salida:" in body


def test_one_cpu_job_alone_is_quiet(world):
    world.asr = ["cuda", "cuda", "cpu", "cuda", "cpu"]
    run_for(world, 30 * MIN)
    assert world.sent == []


def test_cpu_on_purpose_or_no_gpu_never_alerts(world, monkeypatch):
    world.asr = ["cpu"] * 5
    world.gpu = False
    run_for(world, 10 * MIN)
    world.gpu = True
    monkeypatch.setenv("PARAKEET_DEVICE", "cpu")
    run_for(world, 10 * MIN)
    monkeypatch.delenv("PARAKEET_DEVICE")
    monkeypatch.setenv("QUEUE_ALERT_ASR_CPU_JOBS", "0")
    run_for(world, 10 * MIN)
    assert world.sent == []


def test_cpu_reminder_only_while_jobs_still_report_cpu_then_resolved(world):
    world.asr = ["cpu"] * 3
    tick(world, T0)
    assert len(world.sent) == 1
    world.asr += ["cpu"] * 3
    subjects = run_for(world, 61 * MIN)
    assert subjects == ["⚠️ La transcripción sigue yendo por CPU (desde hace 1 h)"]
    assert world.sent[-1][1].startswith("De los últimos 6 vídeos, 6 en CPU.")
    # The fix is deployed: the new instance has no reports yet -> no reminder.
    world.asr = []
    configure(world, now=world.now)
    assert run_for(world, 2 * H) == []
    world.asr = ["cuda"]
    assert run_for(world, 10 * MIN) == []   # one job is not enough
    world.asr = ["cuda", "cuda"]
    subjects = run_for(world, 10 * MIN)
    assert subjects == ["✅ La transcripción vuelve a ir por GPU"]
    assert "Los últimos 2 vídeos ya usan la GPU" in world.sent[-1][1]


# --- output rate drop -------------------------------------------------------------

def learn(w, hours=4, done=40):
    w.done = done
    assert run_for(w, hours * H) == []


def test_rate_is_learnt_from_full_queue_hours_and_persisted(world):
    learn(world, hours=4)
    st = state(world)
    assert len(st["rate_samples"]) == 3     # first hour fills the window, then 1/h
    assert st["rate_samples"][0] == pytest.approx(8.0)   # 40 per hour / 5 slots
    assert qw.usual_rate(st["rate_samples"], qw.limits()) == pytest.approx(8.0)


def test_rate_drop_alerts_after_the_window_reminds_and_recovers(world):
    learn(world)
    world.done = 15                          # 37 % of the usual 40
    subjects = run_for(world, 27 * MIN)
    assert subjects == []                    # not sustained yet
    subjects = run_for(world, 6 * MIN)
    assert subjects == ["⚠️ El ritmo de salida ha caído"]
    body = world.sent[-1][1]
    assert body.startswith("Con la cola llena salen 15 vídeos por hora; lo habitual es ~40 (38 %).")
    assert "Lleva así más de 30 min." in body
    world.done = 18
    subjects = run_for(world, 61 * MIN)
    assert subjects == ["⚠️ El ritmo de salida sigue bajo (desde hace 1 h)"]
    assert world.sent[-1][1].startswith("Ahora 18 vídeos por hora (en el último aviso 15)")
    # The low hours were not learnt as the new normal.
    assert qw.usual_rate(state(world)["rate_samples"], qw.limits()) == pytest.approx(8.0)
    # Hysteresis: back over the drop threshold but under halfway (75 %) is not over.
    world.done = 25
    assert run_for(world, 30 * MIN) == []
    world.done = 32
    subjects = run_for(world, 6 * MIN)
    assert subjects == ["✅ Ritmo de salida normal otra vez"]
    assert "Ahora 32 vídeos por hora; lo habitual es ~40." in world.sent[-1][1]


def test_queue_emptying_resolves_with_its_own_wording(world):
    learn(world)
    world.done = 10
    assert "⚠️ El ritmo de salida ha caído" in run_for(world, 40 * MIN)
    world.queued, world.running = 0, 2
    subjects = run_for(world, 6 * MIN)
    assert subjects == ["✅ Ritmo de salida: la cola ya no está llena"]


def test_too_few_samples_never_alert(world):
    learn(world, hours=3)                    # 2 samples, 3 needed
    world.done = 5
    assert run_for(world, 2 * H) == []
    assert len(state(world)["rate_samples"]) == 2


def test_night_with_an_empty_queue_is_not_a_drop(world):
    learn(world)
    world.queued, world.running, world.done = 0, 1, 3
    assert run_for(world, 5 * H) == []
    assert len(state(world)["rate_samples"]) == 3   # nothing learnt from it


def test_deploy_restarts_the_full_hour(world):
    learn(world)
    world.done = 10
    world.elsewhere = 3                      # old container still draining
    assert run_for(world, 15 * MIN) == []
    world.elsewhere = 0
    # New instance: its own done_last_hour starts from zero, so nothing is
    # judged until the queue has been full for a whole hour after the hold.
    configure(world, now=world.now)
    assert run_for(world, 60 * MIN) == []
    assert "⚠️ El ritmo de salida ha caído" in run_for(world, 2 * H)


def test_drop_percent_zero_turns_the_rate_check_off(world, monkeypatch):
    learn(world)
    monkeypatch.setenv("QUEUE_ALERT_THROUGHPUT_DROP_PERCENT", "0")
    world.done = 1
    assert run_for(world, 2 * H) == []


def test_kill_switch_covers_the_faults(world, monkeypatch):
    monkeypatch.setenv("QUEUE_ALERTS_DISABLED", "1")
    world.asr = ["cpu"] * 3
    assert run_for(world, 10 * MIN) == []


def test_queue_incident_and_faults_keep_separate_state(world):
    world.asr = ["cpu"] * 3
    tick(world, T0)
    st = state(world)
    assert st["asr"]["active"] is True and not st.get("active")


# --- the app.py side ----------------------------------------------------------------

def test_app_reads_the_asr_device_from_job_cpu(monkeypatch):
    app = pytest.importorskip("app")
    monkeypatch.setattr(app, "_recent_asr_devices", [])
    app._note_job_asr(json.dumps({"wall_s": 1, "asr_device": "cpu"}))
    app._note_job_asr(json.dumps({"wall_s": 1, "asr_device": None}))
    app._note_job_asr("not json")
    for _ in range(20):
        app._note_job_asr(json.dumps({"asr_device": "cuda"}))
    assert app._recent_asr_devices == ["cuda"] * 12
    monkeypatch.setitem(app._gpu_free_cache, "mb", 9000)
    snap = app._queue_watch_snapshot()
    assert snap["gpu"] is True and snap["asr_devices"] == ["cuda"] * 12
