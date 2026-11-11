"""Render-queue watch: page the admin when the job queue backs up.

The queue can saturate without a single job failing: every job still
succeeds, it just starts hours late, so neither the failure-rate alert nor the
proxy watcher says anything. This loop measures the queue every few minutes
and goes out through the same channel as every other operational alert
(``alerts.send_admin_alert``: the admin Telegram chat, plus ADMIN_EMAIL when
SMTP is set).

It alerts when any of these holds:

- a paid job has been waiting more than ``QUEUE_ALERT_PAID_WAIT_MINUTES``;
- the oldest free job has been waiting more than ``QUEUE_ALERT_FREE_WAIT_MINUTES``;
- more than ``QUEUE_ALERT_MAX_QUEUED`` jobs are waiting.

One message on entering, a reminder at most every ``QUEUE_ALERT_REMIND_MINUTES``
while it lasts (with the trend since the last message) and one "recovered" when
it is over. Hysteresis: it only clears once every figure is under half its
threshold for ``CLEAR_TICKS`` measurements in a row.

Deploys (see docs/architecture/jobs-and-deploys.md): the old and the new
container share ``output/`` and may both be up for a while.

- Only the instance that owns the ``.instance`` marker and is not draining
  measures (``is_active``); the incident state lives in a file in the shared
  output dir under a non-blocking flock, so a handover neither repeats the
  entry alert nor loses the recovery.
- A wait is measured from disk (``wait_started``): submission, or the last
  time any instance ran the job. A job resumed after a deploy or auto-retried
  waits from then, not from its submission.
- While another instance is still running jobs on the card (and for one paid
  threshold after that, or after boot) the watch holds: no new alert and no
  recovery, only reminders of an incident already open.

Two capacity faults ride the same loop, channel and shared state (sub-keys
``asr`` and ``rate``), each with its own alert / hourly reminder / resolved:

- **Transcription on the CPU** (``asr_incident``): the last
  ``QUEUE_ALERT_ASR_CPU_JOBS`` jobs that loaded Parakeet report
  ``asr_device: "cpu"`` in their JOB_CPU line on a host where nvidia-smi
  answers. A deploy fault (the image lost CUDA), so it alerts at once,
  deploy hold or not. Never with ``PARAKEET_DEVICE=cpu`` set on purpose, nor
  on a host without a GPU. Resolved after two jobs in a row on "cuda".
- **Output rate drop** (``rate_incident``): with the queue full (every slot
  busy and jobs waiting) for a whole hour, jobs finished in the last hour per
  slot fall under ``QUEUE_ALERT_THROUGHPUT_DROP_PERCENT`` of the usual for
  ``QUEUE_ALERT_THROUGHPUT_WINDOW_MINUTES``. "Usual" is learnt: one sample per
  full-queue hour, mean of the last ``RATE_SAMPLES`` healthy ones, kept in the
  shared state; nothing is judged before ``QUEUE_ALERT_THROUGHPUT_MIN_HOURS``
  samples. An empty queue (night), a handover or a fresh boot never count as
  a full hour. Resolved when the rate is back over halfway between the drop
  threshold and the usual, or when the queue is no longer full. An hour with
  braked jobs (fewer clips) in it is never learnt.

The saturation mode of the free queue (``cloud/queue_mode.py``) is decided
from the same snapshot on the same tick: every message carries one line per
free lane (order, brake, refusals) and the brake switching on or off is an
alert of its own.

``QUEUE_ALERTS_DISABLED=1`` turns the alerts off. They never start without
BILLING_ENABLED or without an alert channel configured; the loop still runs
for the saturation mode when that is configured.
"""
import asyncio
import json
import os
import time

from . import queue_mode
from .config import _env_int, _flag, is_enabled, settings

STATE_FILE = ".queue_watch.json"
LOCK_FILE = ".queue_watch.lock"
STARTUP_DELAY = 90          # let the instance boot and resume its jobs first
CLEAR_RATIO = 0.5           # recovered = every figure under half its threshold...
CLEAR_TICKS = 2             # ...this many measurements in a row

_snapshot = None            # () -> dict, this instance's queue (in memory)
_wait_started = None        # (job_id) -> epoch | None, from disk
_busy_elsewhere = None      # () -> int, jobs another instance is running
_is_active = None           # () -> bool, False while draining / not the newest
_state_dir = None
_hold_until = 0.0
_since_cache = {}           # (job_id, attempt, paid) -> epoch the current wait began
_full_since = None          # queue full (all slots busy + jobs waiting) since, this instance
_low_since = None           # output rate under the drop threshold since, this instance
_task = None


def disabled() -> bool:
    return _flag("QUEUE_ALERTS_DISABLED")


def channel_configured() -> bool:
    return bool(settings.telegram_configured
                or (settings.admin_email and settings.smtp_configured))


def enabled() -> bool:
    return is_enabled() and not disabled() and channel_configured()


def limits() -> dict:
    """Thresholds in seconds / jobs, read on every tick (env overrides)."""
    return {
        "paid_wait": max(1, _env_int("QUEUE_ALERT_PAID_WAIT_MINUTES", 10)) * 60,
        "free_wait": max(1, _env_int("QUEUE_ALERT_FREE_WAIT_MINUTES", 30)) * 60,
        "max_queued": max(1, _env_int("QUEUE_ALERT_MAX_QUEUED", 200)),
        "remind": max(1, _env_int("QUEUE_ALERT_REMIND_MINUTES", 60)) * 60,
        # 0 turns the check off.
        "asr_cpu_jobs": max(0, _env_int("QUEUE_ALERT_ASR_CPU_JOBS", 3)),
        "rate_drop": min(100, max(0, _env_int("QUEUE_ALERT_THROUGHPUT_DROP_PERCENT", 50))) / 100,
        "rate_window": max(1, _env_int("QUEUE_ALERT_THROUGHPUT_WINDOW_MINUTES", 30)) * 60,
        "rate_min_samples": max(1, _env_int("QUEUE_ALERT_THROUGHPUT_MIN_HOURS", 3)),
    }


def interval() -> int:
    return max(30, _env_int("QUEUE_ALERT_INTERVAL_SECONDS", 180))


# --- Measuring ----------------------------------------------------------------

def measure(snap: dict, since: dict, now: float, lim: dict) -> dict:
    """Figures for one tick. ``since`` maps job id -> epoch its wait began."""
    paid_ages, free_ages = [], []
    for w in snap.get("waiting") or []:
        age = max(0.0, now - since.get(w["id"], now))
        (paid_ages if w.get("paid") else free_ages).append(age)
    return {
        "queued": len(paid_ages) + len(free_ages),
        "paid_queued": len(paid_ages),
        "paid_late": sum(1 for a in paid_ages if a >= lim["paid_wait"]),
        "oldest_paid": max(paid_ages) if paid_ages else 0.0,
        "oldest_free": max(free_ages) if free_ages else 0.0,
        "running": int(snap.get("running") or 0),
        "running_elsewhere": int(snap.get("running_elsewhere") or 0),
        "slots": int(snap.get("slots") or 0),
        "done_last_hour": snap.get("done_last_hour"),
        "eta_free": snap.get("eta_free"),
        "eta_paid": snap.get("eta_paid"),
        # Saturation mode (cloud/queue_mode.py): per-lane figures and modes,
        # only when it is configured.
        "lanes": snap.get("lanes"),
        "modes": snap.get("modes"),
        # Braked jobs render fewer clips and finish sooner: an hour that
        # includes them is no baseline for the output rate.
        "reduced_clips": bool(snap.get("reduced_clips")),
    }


def breaches(m: dict, lim: dict) -> list:
    """Why the queue is in trouble, in plain words (empty = it is not)."""
    out = []
    if m["paid_late"]:
        out.append(f"hay vídeos de pago esperando más de {lim['paid_wait'] // 60} min")
    if m["oldest_free"] >= lim["free_wait"]:
        out.append(f"el vídeo gratis más antiguo lleva más de {lim['free_wait'] // 60} min")
    if m["queued"] >= lim["max_queued"]:
        out.append(f"hay más de {lim['max_queued']} vídeos en cola")
    return out


def calm(m: dict, lim: dict) -> bool:
    return (m["oldest_paid"] < lim["paid_wait"] * CLEAR_RATIO
            and m["oldest_free"] < lim["free_wait"] * CLEAR_RATIO
            and m["queued"] < lim["max_queued"] * CLEAR_RATIO)


# --- Deciding -----------------------------------------------------------------

def decide(state: dict, m: dict, now: float, lim: dict, hold: bool):
    """(kind, new_state): kind is None, "alert", "remind" or "recovered"."""
    state = dict(state or {})
    summary = {"queued": m["queued"], "oldest": max(m["oldest_paid"], m["oldest_free"])}
    if not state.get("active"):
        why = breaches(m, lim)
        if why and not hold:
            return "alert", {"active": True, "since": now, "last_sent": now,
                             "last": summary, "calm": 0, "why": why}
        return None, state
    state["calm"] = int(state.get("calm") or 0) + 1 if calm(m, lim) else 0
    if state["calm"] >= CLEAR_TICKS and not hold:
        return "recovered", {"active": False, "since": state.get("since"),
                             "recovered_at": now}
    if now - float(state.get("last_sent") or 0) >= lim["remind"]:
        state["last_sent"] = now
        state["prev"] = state.get("last")
        state["last"] = summary
        return "remind", state
    return None, state


# --- Wording ------------------------------------------------------------------

def fmt_duration(seconds) -> str:
    s = int(seconds or 0)
    if s < 60:
        return "menos de 1 min"
    if s < 3600:
        return f"{s // 60} min"
    if s < 2 * 86400:
        h, mins = s // 3600, (s % 3600) // 60
        return f"{h} h {mins} min" if mins else f"{h} h"
    d, h = s // 86400, (s % 86400) // 3600
    return f"{d} d {h} h" if h else f"{d} d"


def _eta(seconds) -> str:
    """queue_estimate's answer in words (it says 15 s when a slot is free)."""
    return "al momento" if (seconds or 0) < 60 else "~" + fmt_duration(seconds)


def _figures(m: dict, lim: dict) -> list:
    if m["paid_queued"]:
        paid = f"{m['paid_queued']} de pago"
        if m["paid_late"]:
            paid += f", {m['paid_late']} esperan más de {lim['paid_wait'] // 60} min"
    else:
        paid = "ninguno de pago"
    lines = [f"En cola: {m['queued']} vídeos ({paid})."]
    if m["queued"]:
        oldest = []
        if m["paid_queued"]:
            oldest.append(f"de pago {fmt_duration(m['oldest_paid'])}")
        if m["queued"] > m["paid_queued"]:
            oldest.append(f"gratis {fmt_duration(m['oldest_free'])}")
        lines.append("Espera del más antiguo: " + ", ".join(oldest) + ".")
    out = []
    if m["done_last_hour"] is not None:
        out.append(f"{m['done_last_hour']} vídeos terminados en la última hora")
    busy = f"{m['running']} de {m['slots']} huecos ocupados"
    if m["running_elsewhere"]:
        busy += f" (+{m['running_elsewhere']} en el contenedor viejo)"
    out.append(busy)
    lines.append("Salida: " + "; ".join(out) + ".")
    if m["queued"] and m["eta_free"] is not None:
        eta = f"Un vídeo nuevo esperaría {_eta(m['eta_free'])}"
        if m["eta_paid"] is not None:
            eta += f" (de pago: {_eta(m['eta_paid'])})"
        lines.append(eta + ".")
    return lines + mode_lines(m)


_LANE_NAMES = {"normal": "Carril gratis normal", "slow": "Carril gratis lento"}


def mode_lines(m: dict, now: float = None) -> list:
    """One line per free lane: queued, estimated wait, order, brake, refusals.
    Empty when the saturation mode is not configured."""
    lanes, modes = m.get("lanes"), m.get("modes")
    if not lanes or not modes:
        return []
    now = m.get("now") if now is None else now
    now = time.time() if now is None else now
    out = []
    for lane in ("normal", "slow"):
        fig = lanes.get(lane) or {}
        state = (modes.get("lanes") or {}).get(lane) or {}
        parts = [f"{int(fig.get('queued') or 0)} en cola",
                 f"espera estimada {_eta(fig.get('wait'))}",
                 "el más reciente primero" if state.get("recent_first") else "por orden de llegada"]
        if lane == "slow":
            b = modes.get("brake") or {}
            if b.get("active"):
                since = fmt_duration(now - float(b.get("since") or now))
                parts.append(f"FRENO activo desde hace {since} (menos clips, envíos nuevos "
                             f"rechazados: {int(b.get('rejected') or 0)})")
            elif b.get("off_at"):
                parts.append(f"sin freno (el último rechazó {int(b.get('rejected') or 0)} envíos)")
            else:
                parts.append("sin freno")
        out.append(f"{_LANE_NAMES[lane]}: " + ", ".join(parts) + ".")
    total = int(modes.get("rejected_total") or 0)
    if total:
        out.append(f"Envíos gratis rechazados por el freno, en total: {total}.")
    return out


def mode_message(event: str, modes: dict, m: dict, now: float, lim: dict):
    """(subject, body) when the brake engages or lets go."""
    b = modes.get("brake") or {}
    if event == "brake_on":
        subject = "🧯 Freno del carril gratis lento activado"
        head = ("La espera estimada del carril lento ha pasado el umbral: sus vídeos salen "
                "con menos clips y los envíos gratis nuevos de ese carril se rechazan "
                "con la oferta de planes.")
    else:
        lasted = fmt_duration(now - float(b.get("since") or now))
        subject = "✅ Freno del carril gratis lento desactivado"
        head = (f"Ha estado activo {lasted} y ha rechazado {int(b.get('rejected') or 0)} envíos. "
                "El carril vuelve a admitir envíos y a dar todos los clips.")
    return subject, "\n".join([head] + _figures(m, lim))


def _trend(prev: dict, m: dict) -> str:
    if not prev:
        return f"{m['queued']} vídeos en cola."
    before = int(prev.get("queued") or 0)
    delta = m["queued"] - before
    if abs(delta) <= max(3, before * 0.05):
        return f"Sigue igual que en el último aviso ({before} → {m['queued']} en cola)."
    word = "peor" if delta > 0 else "mejor"
    return f"Va a {word}: {before} → {m['queued']} vídeos en cola desde el último aviso."


def message(kind: str, state: dict, m: dict, now: float, lim: dict):
    """(subject, body) for send_admin_alert."""
    if kind == "alert":
        subject = "⚠️ Cola de render atascada"
        body = ["Motivo: " + "; ".join(state.get("why") or breaches(m, lim)) + "."]
    elif kind == "remind":
        lasted = fmt_duration(now - float(state.get("since") or now))
        subject = f"⚠️ La cola de render sigue atascada (desde hace {lasted})"
        body = [_trend(state.get("prev"), m)]
    else:
        lasted = fmt_duration(now - float(state.get("since") or now))
        subject = "✅ Cola de render recuperada"
        body = [f"Ha estado atascada {lasted}."]
    return subject, "\n".join(body + _figures(m, lim))


# --- Capacity faults: transcription on the CPU, output rate drop -----------------

RATE_SAMPLES = 24           # full-queue hours the "usual" rate is the mean of
RATE_SAMPLE_EVERY = 3600    # one sample per hour (done_last_hour is an hour)
SUB_KEYS = ("asr", "rate", "rate_samples", "rate_sampled_at")


def asr_on_purpose() -> bool:
    return os.environ.get("PARAKEET_DEVICE", "").strip().lower() == "cpu"


def _incident(sub: dict, breach: bool, clear: bool, still: bool, now: float,
              lim: dict, hold: bool, data: dict):
    """The alert / reminder / resolved machine shared by the capacity faults.

    ``breach`` opens it (unless ``hold``), ``clear`` for CLEAR_TICKS ticks in a
    row closes it (never during ``hold``), ``still`` gates the hourly reminder
    (evidence that it is still happening). ``data`` is kept for the wording."""
    sub = dict(sub or {})
    if not sub.get("active"):
        if breach and not hold:
            return "alert", {"active": True, "since": now, "last_sent": now,
                             "calm": 0, "last": data}
        return None, sub
    sub["calm"] = int(sub.get("calm") or 0) + 1 if clear else 0
    if sub["calm"] >= CLEAR_TICKS and not hold:
        return "recovered", {"active": False, "since": sub.get("since"),
                             "recovered_at": now, "last": data}
    if still and now - float(sub.get("last_sent") or 0) >= lim["remind"]:
        sub["last_sent"] = now
        sub["prev"] = sub.get("last")
        sub["last"] = data
        return "remind", sub
    return None, sub


def asr_incident(sub: dict, snap: dict, now: float, lim: dict):
    """Jobs transcribing on the CPU on a GPU host. Not held by a deploy: the
    jobs that report are the new image's own, and the sooner the better."""
    n = lim["asr_cpu_jobs"]
    devices = list(snap.get("asr_devices") or [])
    watch = n > 0 and bool(snap.get("gpu")) and not asr_on_purpose()
    recent = devices[-n:] if n else []
    breach = watch and len(recent) == n and all(d == "cpu" for d in recent)
    # Closed by evidence (jobs on "cuda") or by configuration (the check off,
    # CPU on purpose), never by a missing figure: nvidia-smi not answering on
    # one tick says nothing about where the next jobs transcribe.
    configured_off = n <= 0 or asr_on_purpose()
    clear = configured_off or (len(devices) >= CLEAR_TICKS
                               and all(d == "cuda" for d in devices[-CLEAR_TICKS:]))
    data = {"cpu": devices.count("cpu"), "seen": len(devices)}
    still = bool(devices) and devices[-1] == "cpu"
    return _incident(sub, breach, clear, still, now, lim, False, data)


def usual_rate(samples, lim):
    """Mean finished-per-slot-per-hour of the learnt full-queue hours, or None
    while there are too few of them to judge."""
    samples = [float(x) for x in (samples or [])]
    if len(samples) < lim["rate_min_samples"]:
        return None
    return sum(samples) / len(samples)


def rate_incident(state: dict, m: dict, now: float, lim: dict, hold: bool):
    """(kind, sub, samples, sampled_at) for the output-rate check."""
    global _full_since, _low_since
    sub = state.get("rate") or {}
    samples = list(state.get("rate_samples") or [])
    sampled_at = float(state.get("rate_sampled_at") or 0)
    done, slots = m.get("done_last_hour"), m["slots"]
    full = (done is not None and slots > 0 and m["running"] >= slots
            and m["queued"] > 0 and not m["running_elsewhere"] and not hold)
    _full_since = (_full_since or now) if full else None
    # done_last_hour looks back one hour: only a queue full for all of it is
    # a fair comparison (and this instance must have been up for all of it).
    comparable = full and now - _full_since >= 3600
    per_slot = (done / slots) if comparable else None
    usual = usual_rate(samples, lim)
    drop = lim["rate_drop"]
    low = bool(comparable and usual and drop > 0 and per_slot < usual * drop)
    _low_since = (_low_since or now) if low else None
    breach = low and now - _low_since >= lim["rate_window"]
    back = drop + (1 - drop) / 2
    clear = (not full and not hold) or bool(comparable and usual and per_slot >= usual * back)
    # Learn only healthy hours: never during an incident, and never an hour
    # under the drop threshold of what has been learnt so far (even before
    # there are enough samples to alert), or a fault would become the usual.
    partial = sum(samples) / len(samples) if samples else None
    # Nor an hour with braked jobs in it (fewer clips each, so they finish
    # sooner): that would lift the usual and make a normal hour read as a drop.
    healthy = not m.get("reduced_clips") and not (
        partial and drop > 0 and per_slot is not None and per_slot < partial * drop)
    if (comparable and healthy and not sub.get("active")
            and now - sampled_at >= RATE_SAMPLE_EVERY):
        samples = (samples + [round(per_slot, 3)])[-RATE_SAMPLES:]
        sampled_at = now
    data = {"done": done, "usual": round(usual * slots, 1) if usual else None,
            "full": full}
    kind, sub = _incident(sub, breach, clear, low, now, lim, hold, data)
    return kind, sub, samples, sampled_at


def _pct(done, usual):
    return f"{round(100 * done / usual)} %" if usual else "?"


def fault_message(fault: str, kind: str, sub: dict, m: dict, now: float, lim: dict):
    """(subject, body) for the capacity faults."""
    lasted = fmt_duration(now - float(sub.get("since") or now))
    d = sub.get("last") or {}
    salida = next(ln for ln in _figures(m, lim) if ln.startswith("Salida:"))
    if fault == "asr":
        n = lim["asr_cpu_jobs"]
        if kind == "alert":
            return ("⚠️ La transcripción va por CPU, no por GPU",
                    f"Los últimos {n} vídeos se han transcrito en CPU en una máquina con GPU.\n"
                    "Es una avería del despliegue (la imagen no ve CUDA), no un atasco: "
                    "cada vídeo tarda más y cabe menos en la cola.\n" + salida)
        if kind == "remind":
            return (f"⚠️ La transcripción sigue yendo por CPU (desde hace {lasted})",
                    f"De los últimos {d.get('seen', 0)} vídeos, {d.get('cpu', 0)} en CPU.\n" + salida)
        return ("✅ La transcripción vuelve a ir por GPU",
                f"Ha ido por CPU durante {lasted}. Los últimos {CLEAR_TICKS} vídeos ya usan la GPU.")
    done, usual = d.get("done"), d.get("usual")
    if kind == "alert":
        return ("⚠️ El ritmo de salida ha caído",
                f"Con la cola llena salen {done} vídeos por hora; lo habitual es ~{usual:g} "
                f"({_pct(done, usual)}). Lleva así más de {lim['rate_window'] // 60} min.\n"
                + "\n".join(_figures(m, lim)))
    if kind == "remind":
        before = (sub.get("prev") or {}).get("done")
        was = f" (en el último aviso {before})" if before is not None else ""
        return (f"⚠️ El ritmo de salida sigue bajo (desde hace {lasted})",
                f"Ahora {done} vídeos por hora{was}; lo habitual es ~{usual:g}.\n"
                + "\n".join(_figures(m, lim)))
    if not d.get("full"):
        return ("✅ Ritmo de salida: la cola ya no está llena",
                f"Ha estado bajo {lasted}. Sin cola llena no hay con qué compararlo; "
                "si sigue mal, volverá a avisar.")
    return ("✅ Ritmo de salida normal otra vez",
            f"Ha estado bajo {lasted}. Ahora {done} vídeos por hora; lo habitual es ~{usual:g}.")


# --- Shared state (both containers of a deploy) ----------------------------------

def _path(name):
    return os.path.join(_state_dir, name) if _state_dir else None


# The last state this instance decided, kept in memory: when the shared file
# cannot be written (disk full, read-only mount) or read, ticks go on from it
# instead of from an empty state, which re-opened the incident and re-sent
# the alert on every tick.
_mem_state = None
_save_failed = False


def load_state() -> dict:
    global _mem_state
    p = _path(STATE_FILE)
    if not p:
        return dict(_mem_state or {})
    if _save_failed and _mem_state is not None:
        return dict(_mem_state)  # the file is older than what we decided
    try:
        with open(p) as f:
            data = json.load(f)
    except FileNotFoundError:
        return dict(_mem_state or {})
    except (OSError, ValueError) as e:
        if _mem_state is not None:
            return dict(_mem_state)
        print(f"⚠️  Queue watch could not read its state: {e}")
        return {}
    data = data if isinstance(data, dict) else {}
    _mem_state = dict(data)
    return data


def save_state(state: dict):
    global _mem_state, _save_failed
    _mem_state = dict(state)
    p = _path(STATE_FILE)
    if not p:
        return
    tmp = p + ".tmp"
    try:
        with open(tmp, "w") as f:
            json.dump(state, f)
        os.replace(tmp, p)
        _save_failed = False
    except OSError as e:
        if not _save_failed:
            print(f"⚠️  Queue watch could not save its state, keeping it in memory: {e}")
        _save_failed = True


def _try_lock():
    """An exclusive non-blocking flock on the shared dir: the open file, None
    when the other instance holds it, False when locking is unavailable (no
    dir, no fcntl), in which case the tick goes ahead unlocked."""
    p = _path(LOCK_FILE)
    if not p:
        return False
    try:
        import fcntl
        fh = open(p, "a")
    except (ImportError, OSError):
        return False
    try:
        fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        fh.close()
        return None
    return fh


# --- Saturation mode (cloud/queue_mode.py) ---------------------------------------

def lane_figures(snap: dict) -> dict:
    """The snapshot's per-lane figures, or zeros."""
    lanes = snap.get("lanes") or {}
    return {lane: {"queued": int((lanes.get(lane) or {}).get("queued") or 0),
                   "wait": float((lanes.get(lane) or {}).get("wait") or 0)}
            for lane in queue_mode.LANES}


def decide_modes(snap: dict, now: float) -> list:
    """Measuring instance: decide the lanes' modes from this tick's snapshot,
    fold in the refusals made here, share the result and apply it. Returns the
    switch events. Under the same lock as the incident state; if the other
    instance holds it right now, nothing changes this tick."""
    lock = _try_lock()
    if lock is None:
        return []
    try:
        state = queue_mode.fold(queue_mode.load(_state_dir), queue_mode.take_pending())
        new, events = queue_mode.decide(state, lane_figures(snap), now)
        queue_mode.save(_state_dir, new)
        queue_mode.adopt(new)
        for e in events:
            print(f"🚦 Queue mode: {e} (normal wait {int(new['figures']['normal'].get('wait') or 0)} s, "
                  f"slow wait {int(new['figures']['slow'].get('wait') or 0)} s)")
        return events
    finally:
        if lock:
            lock.close()


def follow_modes():
    """Non-measuring instance (and every instance at boot): adopt the shared
    modes, hand over refusals. The file is replaced atomically, so it is read
    without the lock; the lock is only for adding the refusals made here, and
    when the other instance holds it they wait for the next tick."""
    queue_mode.adopt(queue_mode.load(_state_dir))
    if not queue_mode.has_pending():
        return
    lock = _try_lock()
    if lock is None:
        return
    try:
        state = queue_mode.fold(queue_mode.load(_state_dir), queue_mode.take_pending())
        queue_mode.save(_state_dir, state)
        queue_mode.adopt(state)
    finally:
        if lock:
            lock.close()


async def _send_mode_alerts(events, m, now, lim) -> list:
    """The brake's on / off alerts, sent from the decision itself so a busy
    incident lock cannot lose them. Returns their kinds."""
    out = []
    for event in events:
        if event in ("brake_on", "brake_off"):
            out.append((event.replace("_", "-"),
                        mode_message(event, m.get("modes") or {}, m, now, lim)))
    if out:
        from .alerts import send_admin_alert
        for _name, (subject, body) in out:
            await send_admin_alert(subject, body)
        print(f"📣 Queue watch: {', '.join(n for n, _ in out)}")
    return [n for n, _ in out]


# --- The tick -----------------------------------------------------------------

async def tick(now=None):
    """Measure once and send whatever the state machine says. Returns the kind
    of message sent (or None), for tests and logs."""
    global _hold_until
    alerting, moding = enabled(), queue_mode.enabled()
    if not (alerting or moding) or _snapshot is None:
        return None
    if _is_active is not None and not _is_active():
        # The other container decides; this one only follows the shared modes
        # (a submission landing here is braked too) and hands over its refusals.
        if moding:
            follow_modes()
        return None
    loop = asyncio.get_running_loop()
    now = time.time() if now is None else now
    snap = _snapshot()
    mode_events = decide_modes(snap, now) if moding else []
    if moding:
        snap["modes"] = queue_mode.current()
        snap["reduced_clips"] = queue_mode.reduced_clips_recently(now)
    if not alerting:
        return None
    elsewhere = 0
    if _busy_elsewhere is not None:
        elsewhere = int(await loop.run_in_executor(None, _busy_elsewhere) or 0)
    snap["running_elsewhere"] = elsewhere

    waiting = snap.get("waiting") or []
    # Paid-ness is part of the key: a free job raised to a paid priority (its
    # owner paid) starts a new wait, read again from disk (its ``raised_at``).
    keys = {(w["id"], w.get("attempt", 0), bool(w.get("paid"))) for w in waiting}
    for k in [k for k in _since_cache if k not in keys]:
        del _since_cache[k]
    new = [k for k in keys if k not in _since_cache]
    if new:
        starts = {}
        if _wait_started is not None:
            starts = await loop.run_in_executor(
                None, lambda: {k: _wait_started(k[0]) for k in new})
        for k in new:
            got = starts.get(k)
            _since_cache[k] = min(float(got), now) if got else now
    since = {w["id"]: _since_cache[(w["id"], w.get("attempt", 0), bool(w.get("paid")))]
             for w in waiting}

    lim = limits()
    m = measure(snap, since, now, lim)
    m["now"] = now
    if elsewhere:
        _hold_until = max(_hold_until, now + lim["paid_wait"])
    hold = now < _hold_until

    mode_sent = await _send_mode_alerts(mode_events, m, now, lim)
    lock = _try_lock()
    if lock is None:
        # The other instance is ticking right now; the brake's alerts are out.
        return ",".join(mode_sent) or None
    try:
        state = load_state()
        queue_state = {k: v for k, v in state.items() if k not in SUB_KEYS}
        kind, new_state = decide(queue_state, m, now, lim, hold)
        asr_kind, asr_sub = asr_incident(state.get("asr"), snap, now, lim)
        rate_kind, rate_sub, samples, sampled_at = rate_incident(state, m, now, lim, hold)
        new_state = dict(new_state, asr=asr_sub, rate=rate_sub,
                         rate_samples=samples, rate_sampled_at=sampled_at)
        if new_state != state:
            save_state(new_state)
        out = []
        if kind:
            out.append((kind, message(kind, new_state, m, now, lim)))
        if asr_kind:
            out.append(("asr-" + asr_kind, fault_message("asr", asr_kind, asr_sub, m, now, lim)))
        if rate_kind:
            out.append(("rate-" + rate_kind,
                        fault_message("rate", rate_kind, rate_sub, m, now, lim)))
        if out:
            from .alerts import send_admin_alert
            for name, (subject, body) in out:
                await send_admin_alert(subject, body)
            print(f"📣 Queue watch: {', '.join(n for n, _ in out)} ({m['queued']} queued, "
                  f"oldest paid {int(m['oldest_paid'])} s, oldest free "
                  f"{int(m['oldest_free'])} s, done/h {m.get('done_last_hour')})")
        # The queue incident's kind, as before; with a fault or a brake switch
        # too, every kind sent, comma-separated.
        names = [n for n, _ in out] + mode_sent
        if any(n != kind for n in names):
            return ",".join(names)
        return kind
    finally:
        if lock:
            lock.close()


async def _loop():
    await asyncio.sleep(STARTUP_DELAY)
    while True:
        try:
            await tick()
        except asyncio.CancelledError:
            return
        except Exception as e:
            print(f"⚠️  Queue watch error: {e}")
        await asyncio.sleep(interval())


def configure(*, snapshot, wait_started=None, busy_elsewhere=None, state_dir=None,
              is_active=None, now=None):
    """Wire the app's callbacks in (cloud/ cannot import app.py)."""
    global _snapshot, _wait_started, _busy_elsewhere, _state_dir, _is_active, _hold_until
    global _mem_state, _save_failed, _full_since, _low_since
    _snapshot, _wait_started, _busy_elsewhere = snapshot, wait_started, busy_elsewhere
    _full_since = _low_since = None
    _state_dir, _is_active = state_dir, is_active
    _since_cache.clear()
    _mem_state, _save_failed = None, False
    # A fresh instance (restart or deploy) gives the jobs it just resumed one
    # paid threshold to start before judging the queue.
    _hold_until = (time.time() if now is None else now) + limits()["paid_wait"]


def start(**kwargs):
    """Start the loop from the app lifespan (cloud mode only). The saturation
    mode rides the same loop, so it runs even with the alerts off."""
    global _task
    if not queue_mode.enabled():
        if disabled():
            print("📣 Queue watch disabled (QUEUE_ALERTS_DISABLED).")
            return
        if not enabled():
            print("📣 Queue watch off: no alert channel configured (TELEGRAM_* or ADMIN_EMAIL + SMTP_*).")
            return
    elif not enabled():
        print("📣 Queue alerts off; the queue is still measured for the saturation mode.")
    configure(**kwargs)
    if queue_mode.enabled():
        # Pick up the modes the previous container left at once, not after
        # STARTUP_DELAY: a deploy must not open the brake or reorder the queue.
        follow_modes()
    _task = asyncio.create_task(_loop())
