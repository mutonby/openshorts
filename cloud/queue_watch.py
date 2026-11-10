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

``QUEUE_ALERTS_DISABLED=1`` turns it off. It never starts without
BILLING_ENABLED or without an alert channel configured.
"""
import asyncio
import json
import os
import time

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
_since_cache = {}           # (job_id, attempt) -> epoch the current wait began
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
    return lines


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


# --- Shared state (both containers of a deploy) ----------------------------------

def _path(name):
    return os.path.join(_state_dir, name) if _state_dir else None


def load_state() -> dict:
    p = _path(STATE_FILE)
    if not p:
        return {}
    try:
        with open(p) as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def save_state(state: dict):
    p = _path(STATE_FILE)
    if not p:
        return
    tmp = p + ".tmp"
    try:
        with open(tmp, "w") as f:
            json.dump(state, f)
        os.replace(tmp, p)
    except OSError as e:
        print(f"⚠️  Queue watch could not save its state: {e}")


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


# --- The tick -----------------------------------------------------------------

async def tick(now=None):
    """Measure once and send whatever the state machine says. Returns the kind
    of message sent (or None), for tests and logs."""
    global _hold_until
    if not enabled() or _snapshot is None:
        return None
    if _is_active is not None and not _is_active():
        return None
    loop = asyncio.get_running_loop()
    snap = _snapshot()
    elsewhere = 0
    if _busy_elsewhere is not None:
        elsewhere = int(await loop.run_in_executor(None, _busy_elsewhere) or 0)
    snap["running_elsewhere"] = elsewhere
    now = time.time() if now is None else now

    waiting = snap.get("waiting") or []
    keys = {(w["id"], w.get("attempt", 0)) for w in waiting}
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
    since = {w["id"]: _since_cache[(w["id"], w.get("attempt", 0))] for w in waiting}

    lim = limits()
    m = measure(snap, since, now, lim)
    if elsewhere:
        _hold_until = max(_hold_until, now + lim["paid_wait"])
    hold = now < _hold_until

    lock = _try_lock()
    if lock is None:
        return None  # the other instance is ticking right now
    try:
        state = load_state()
        kind, new_state = decide(state, m, now, lim, hold)
        if new_state != state:
            save_state(new_state)
        if kind:
            from .alerts import send_admin_alert
            subject, body = message(kind, new_state, m, now, lim)
            await send_admin_alert(subject, body)
            print(f"📣 Queue watch: {kind} ({m['queued']} queued, oldest paid "
                  f"{int(m['oldest_paid'])} s, oldest free {int(m['oldest_free'])} s)")
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
    _snapshot, _wait_started, _busy_elsewhere = snapshot, wait_started, busy_elsewhere
    _state_dir, _is_active = state_dir, is_active
    _since_cache.clear()
    # A fresh instance (restart or deploy) gives the jobs it just resumed one
    # paid threshold to start before judging the queue.
    _hold_until = (time.time() if now is None else now) + limits()["paid_wait"]


def start(**kwargs):
    """Start the loop from the app lifespan (cloud mode only)."""
    global _task
    if disabled():
        print("📣 Queue watch disabled (QUEUE_ALERTS_DISABLED).")
        return
    if not enabled():
        print("📣 Queue watch off: no alert channel configured (TELEGRAM_* or ADMIN_EMAIL + SMTP_*).")
        return
    configure(**kwargs)
    _task = asyncio.create_task(_loop())
