"""Saturation mode for the free queue: two lanes, newest first, and a brake.

The free plan is one video per sign-up, and when more of them arrive than the
card can render, a single FIFO makes everybody wait the same hours, including
the person who signed up a minute ago and is deciding right now whether to pay.
This module holds the policy; ``app.py`` applies it to its heap and the queue
watch (``cloud/queue_watch.py``) feeds it the measurement it already takes.

- **Lanes.** A free job goes to the slow lane when the client's country (its
  address in the local country database, ``cloud/geoip.py``; or a trusted
  proxy's header, ``QUEUE_COUNTRY_HEADER``, off by default) is in
  ``FREE_SLOW_LANE_COUNTRIES`` at submit; anything else, an unknown country
  included, goes to the normal lane. The slow lane is one priority below the
  free one (``SLOW_PRIORITY``), so it moves only when nothing is in front of
  it. Paid plans never go there, and paying moves a job up as usual.
- **Newest first.** While a lane's estimated wait (what a job joining at its
  back would wait in FIFO) is at least ``QUEUE_RECENT_FIRST_MINUTES``, that
  lane serves its newest job first; under ``QUEUE_RECENT_FIRST_RELEASE_MINUTES``
  it is FIFO again, so the old ones go first as soon as the backlog shrinks.
- **Brake.** While the slow lane's estimated wait is at least
  ``SLOW_LANE_BRAKE_HOURS``, its jobs render at most ``SLOW_LANE_BRAKE_CLIPS``
  clips and new free submissions for it are refused before anything is
  reserved; under ``SLOW_LANE_BRAKE_RELEASE_HOURS`` the brake lets go.

Nothing happens unless something is configured: with no slow-lane country and
``QUEUE_RECENT_FIRST_MINUTES`` unset the queue is the plain priority FIFO it
always was. ``QUEUE_SATURATION_DISABLED=1`` turns all of it off at once.

The modes are decided by the instance that measures the queue and shared with
the other container of a deploy through ``.queue_mode.json`` in the output dir
(under the queue watch's lock), so a submission that lands on the old
container is braked like one on the new container.
"""
import json
import os
import re
import time

from .config import _flag, is_enabled

NORMAL = "normal"
SLOW = "slow"
LANES = (NORMAL, SLOW)
PAID_PRIORITY = 1            # config.PLAN_PRIORITY["starter"]: top-up buyers too
FREE_PRIORITY = 2            # config.PLAN_PRIORITY["free"]
SLOW_PRIORITY = 3            # after every free job of the normal lane
STATE_FILE = ".queue_mode.json"
# Codes a CDN sends when it does not know (Cloudflare: XX unknown, T1 Tor).
UNKNOWN_COUNTRIES = frozenset({"XX", "T1"})
# How long after the brake lets go the jobs finished in the last hour still
# include reduced ones (the queue watch's output-rate baseline skips them).
REDUCED_TAIL_SECONDS = 3600

_CODE = re.compile(r"^[A-Z]{2}$")


# --- Configuration (read on every call: env overrides need no restart in tests) --

def killed() -> bool:
    return _flag("QUEUE_SATURATION_DISABLED")


def slow_countries() -> frozenset:
    """ISO 3166-1 alpha-2 codes of the slow lane (empty by default)."""
    raw = os.environ.get("FREE_SLOW_LANE_COUNTRIES", "")
    codes = {c.strip().upper() for c in re.split(r"[\s,;]+", raw) if c.strip()}
    return frozenset(c for c in codes if _CODE.match(c) and c not in UNKNOWN_COUNTRIES)


def country_header() -> str:
    """A geolocation header to trust, or "" (the default: none). Only for an
    API behind a proxy/CDN that sets it and strips any copy from clients;
    otherwise anyone can send it."""
    return os.environ.get("QUEUE_COUNTRY_HEADER", "").strip().lower()


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, "") or default)
    except ValueError:
        return default


def limits() -> dict:
    """Thresholds in seconds; 0 turns that part off."""
    recent = max(0.0, _env_float("QUEUE_RECENT_FIRST_MINUTES", 0)) * 60
    recent_off = _env_float("QUEUE_RECENT_FIRST_RELEASE_MINUTES", recent / 120) * 60
    brake = max(0.0, _env_float("SLOW_LANE_BRAKE_HOURS", 12)) * 3600
    brake_off = _env_float("SLOW_LANE_BRAKE_RELEASE_HOURS", brake / 7200) * 3600
    try:
        clips = max(1, int(os.environ.get("SLOW_LANE_BRAKE_CLIPS", "") or 3))
    except ValueError:
        clips = 3
    return {
        "recent_on": recent,
        "recent_off": min(max(0.0, recent_off), recent),
        "brake_on": brake,
        "brake_off": min(max(0.0, brake_off), brake),
        "brake_clips": clips,
    }


def enabled() -> bool:
    """Cloud only, not killed, and something to do."""
    if not is_enabled() or killed():
        return False
    return bool(slow_countries()) or limits()["recent_on"] > 0


# --- Lanes --------------------------------------------------------------------

def _code(raw) -> str:
    code = str(raw or "").strip().upper()
    return code if _CODE.match(code) and code not in UNKNOWN_COUNTRIES else ""


def request_country(headers, client_ip=None) -> str:
    """The client's country, or "" when unknown. The trusted header when one
    is configured (``QUEUE_COUNTRY_HEADER``), else the client address looked
    up in the local country database (``cloud/geoip.py``: in memory, no call).
    Never raises."""
    name = country_header()
    if name:
        try:
            code = _code(headers.get(name))
        except Exception:
            code = ""
        if code:
            return code
    try:
        from . import geoip
        return _code(geoip.country_of(client_ip))
    except Exception:
        return ""


def lane_for(plan, country) -> str:
    """The lane of a new job: slow only for the free plan from a listed country."""
    if killed() or plan != "free":
        return NORMAL
    code = str(country or "").strip().upper()
    return SLOW if code and code in slow_countries() else NORMAL


def priority_for(plan_priority: int, lane: str) -> int:
    """Queue priority of a new job: the plan's, pushed behind the free lane for
    the slow lane. Paid priorities are never touched."""
    if lane == SLOW and int(plan_priority) >= FREE_PRIORITY:
        return SLOW_PRIORITY
    return int(plan_priority)


def lane_of_priority(priority) -> str:
    return SLOW if int(priority) >= SLOW_PRIORITY else NORMAL


def slow_lane_on() -> bool:
    """Whether the slow lane exists now (Cloud, not killed, a country set).
    When it does not, a job at SLOW_PRIORITY belongs to the normal lane."""
    return is_enabled() and not killed() and bool(slow_countries())


def effective_priority(priority) -> int:
    """The priority to queue at now: the slow lane folds into the normal one
    while it is switched off (kill switch, empty list)."""
    priority = int(priority)
    if priority >= SLOW_PRIORITY and not slow_lane_on():
        return FREE_PRIORITY
    return priority


# --- State ----------------------------------------------------------------------

def default_state() -> dict:
    return {
        "lanes": {lane: {"recent_first": False, "since": None} for lane in LANES},
        "brake": {"active": False, "since": None, "off_at": None, "rejected": 0},
        "rejected_total": 0,
        "figures": {},
        # When the mode was switched on (first decision with it configured):
        # the starved-job expiry counts from here, so the jobs already
        # waiting at activation get the whole window, not a wave of expiries.
        "enabled_since": None,
    }


def _normalized(state) -> dict:
    out = default_state()
    if not isinstance(state, dict):
        return out
    for lane in LANES:
        got = ((state.get("lanes") or {}).get(lane) or {})
        out["lanes"][lane].update({k: got[k] for k in ("recent_first", "since") if k in got})
    out["brake"].update({k: v for k, v in (state.get("brake") or {}).items()
                         if k in out["brake"]})
    out["rejected_total"] = int(state.get("rejected_total") or 0)
    out["figures"] = dict(state.get("figures") or {})
    if state.get("enabled_since") is not None:
        try:
            out["enabled_since"] = float(state["enabled_since"])
        except (TypeError, ValueError):
            pass
    return out


_state = default_state()     # what this instance applies now
_pending_rejected = 0        # refusals here not yet folded into the shared file
_on_change = None            # app callback: the lanes' order changed


def register_on_change(callback):
    global _on_change
    _on_change = callback


def reset():
    """Tests and a fresh configure(): back to the plain queue."""
    global _state, _pending_rejected
    _state, _pending_rejected = default_state(), 0


def current() -> dict:
    s = _normalized(_state)
    s["brake"]["rejected"] += _pending_rejected
    s["rejected_total"] += _pending_rejected
    return s


def recent_first(lane) -> bool:
    if killed():
        return False
    return bool(_state["lanes"].get(lane, {}).get("recent_first"))


def brake_active() -> bool:
    if killed() or not slow_countries():
        return False
    return bool(_state["brake"].get("active"))


def enabled_since():
    """When the mode was switched on, as this instance knows it (None until
    the first decision or the shared state was adopted: then nothing expires)."""
    return _state.get("enabled_since")


def forget_activation(state_dir):
    """Boot with the mode off: drop the activation time, so switching it on
    again later starts a fresh window instead of expiring the queue at once."""
    s = load(state_dir)
    if s.get("enabled_since") is None:
        return
    s["enabled_since"] = None
    save(state_dir, s)


def reduced_clips_recently(now=None) -> bool:
    """Brake on, or off for less than an hour: the last hour's finished jobs
    include braked ones, which render fewer clips and finish sooner."""
    b = _state["brake"]
    if b.get("active"):
        return True
    off = b.get("off_at")
    now = time.time() if now is None else now
    return bool(off) and now - float(off) < REDUCED_TAIL_SECONDS


def note_rejected():
    global _pending_rejected
    _pending_rejected += 1


def has_pending() -> bool:
    return _pending_rejected > 0


def take_pending() -> int:
    global _pending_rejected
    n, _pending_rejected = _pending_rejected, 0
    return n


def fold(state: dict, pending: int) -> dict:
    s = _normalized(state)
    if pending:
        s["brake"]["rejected"] += pending
        s["rejected_total"] += pending
    return s


def adopt(state: dict):
    """Apply a decided (or shared) state here; tell the app if an order changed."""
    global _state
    new = _normalized(state)
    old = _state
    _state = new
    changed = [lane for lane in LANES
               if bool(old["lanes"][lane]["recent_first"]) != bool(new["lanes"][lane]["recent_first"])]
    if changed and _on_change is not None:
        try:
            _on_change(changed)
        except Exception as e:
            print(f"⚠️  Queue mode: re-ordering the queue failed: {e}")
    return changed


# --- Deciding -------------------------------------------------------------------

def decide(state: dict, figures: dict, now: float, lim: dict = None):
    """(new_state, events) from the lanes' estimated waits.

    ``figures`` = ``{"normal": {"queued", "wait"}, "slow": {"queued", "wait"}}``;
    ``wait`` is the seconds a job joining the back of that lane would wait in
    FIFO. Each switch has its own on / off threshold (hysteresis). Events:
    ``recent_on:<lane>``, ``recent_off:<lane>``, ``brake_on``, ``brake_off``.
    """
    lim = lim or limits()
    s = _normalized(state)
    events = []
    if s["enabled_since"] is None and not killed():
        s["enabled_since"] = now
    on_ok = not killed()
    for lane in LANES:
        cur = s["lanes"][lane]
        wait = float(((figures or {}).get(lane) or {}).get("wait") or 0)
        want = bool(cur["recent_first"])
        if not on_ok or lim["recent_on"] <= 0:
            want = False
        elif not want and wait >= lim["recent_on"]:
            want = True
        elif want and wait < lim["recent_off"]:
            want = False
        if want != bool(cur["recent_first"]):
            cur["recent_first"], cur["since"] = want, now
            events.append(("recent_on:" if want else "recent_off:") + lane)
    b = s["brake"]
    wait = float(((figures or {}).get(SLOW) or {}).get("wait") or 0)
    want = bool(b["active"])
    if not on_ok or lim["brake_on"] <= 0 or not slow_countries():
        want = False
    elif not want and wait >= lim["brake_on"]:
        want = True
    elif want and wait < lim["brake_off"]:
        want = False
    if want != bool(b["active"]):
        if want:
            b.update(active=True, since=now, off_at=None, rejected=0)
            events.append("brake_on")
        else:
            b.update(active=False, off_at=now)
            events.append("brake_off")
    s["figures"] = {lane: dict((figures or {}).get(lane) or {}) for lane in LANES}
    return s, events


# --- Shared file (both containers of a deploy) ------------------------------------

def load(state_dir) -> dict:
    if not state_dir:
        return current()
    try:
        with open(os.path.join(state_dir, STATE_FILE)) as f:
            return _normalized(json.load(f))
    except FileNotFoundError:
        return _normalized(_state)
    except (OSError, ValueError) as e:
        print(f"⚠️  Queue mode could not read its state, keeping the last one: {e}")
        return _normalized(_state)


def save(state_dir, state: dict) -> bool:
    if not state_dir:
        return True
    p = os.path.join(state_dir, STATE_FILE)
    tmp = p + ".tmp"
    try:
        with open(tmp, "w") as f:
            json.dump(state, f)
        os.replace(tmp, p)
        return True
    except OSError as e:
        print(f"⚠️  Queue mode could not save its state, keeping it in memory: {e}")
        return False
