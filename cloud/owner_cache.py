"""Who the caller of a job poll is, remembered per credential.

/api/status is polled many times a second, and the full user load behind
each poll is several queries. With the host short of CPU those polls held the
whole Postgres pool and every endpoint (login, Stripe webhooks) answered 500.
So the owner check of a GET remembers the caller's id for OWNER_ID_TTL, and at
most OWNER_LOOKUP_CONCURRENCY of those lookups run at once: the rest of the
pool stays free for everything else. When the gate is full a caller seen
recently gets its last answer, and an unknown one a 503 to retry.

A ban, a revoked key or an erased account calls ``forget_user``, so a
credential that stops working stops passing at once, not at the end of the TTL.
No dependencies beyond the standard library: importable with billing off.
"""
import asyncio
import hashlib
import os
import time


def _env_float(name, default):
    try:
        return max(0.0, float(os.environ.get(name, default)))
    except ValueError:
        return float(default)


OWNER_ID_TTL = _env_float("OWNER_ID_TTL", 600)
# How long past its TTL an answer may still stand in when the gate is full.
OWNER_ID_STALE = _env_float("OWNER_ID_STALE", 3600)
OWNER_LOOKUP_CONCURRENCY = max(1, int(_env_float("OWNER_LOOKUP_CONCURRENCY", 8)))
OWNER_LOOKUP_WAIT = _env_float("OWNER_LOOKUP_WAIT", 3)
_MAX = 20000

# key -> (fresh_until monotonic, user id as str)
_entries: dict = {}
_gate = None


class Busy(Exception):
    """Every lookup slot was taken for OWNER_LOOKUP_WAIT and the caller is unknown."""


def key_for(credential: str) -> str:
    return hashlib.sha256(credential.encode()).hexdigest()


def _gate_now():
    global _gate
    if _gate is None:
        _gate = asyncio.Semaphore(OWNER_LOOKUP_CONCURRENCY)
    return _gate


def _remember(key, user_id, now):
    if len(_entries) >= _MAX:
        stale = now - OWNER_ID_STALE
        for k in [k for k, v in _entries.items() if v[0] <= stale] or list(_entries):
            _entries.pop(k, None)
    _entries[key] = (now + OWNER_ID_TTL, user_id)


async def owner_id(credential: str, load):
    """The caller's user id (str) or None. ``load`` is an async callable that
    returns the user (with ``.id``) or None; it runs only when needed."""
    sync()
    key = key_for(credential)
    now = time.monotonic()
    hit = _entries.get(key)
    if hit and hit[0] > now:
        return hit[1]
    gate = _gate_now()
    try:
        await asyncio.wait_for(gate.acquire(), OWNER_LOOKUP_WAIT)
    except asyncio.TimeoutError:
        if hit and hit[0] + OWNER_ID_STALE > now:
            return hit[1]
        raise Busy()
    try:
        user = await load()
    finally:
        gate.release()
    if user is None:
        _entries.pop(key, None)
        return None
    _remember(key, str(user.id), time.monotonic())
    return str(user.id)


# Other per-user memories to clear with this one (cloud.auth registers its
# signed-in user snapshot): one call site for ban, revoke and erase.
_forget_hooks: list = []


def on_forget(hook) -> None:
    if hook not in _forget_hooks:
        _forget_hooks.append(hook)


def _forget_here(uid) -> None:
    """Drop this process's memories of one user, or of everyone (None)."""
    if uid is None:
        _entries.clear()
    else:
        for k in [k for k, v in _entries.items() if v[1] == uid]:
            _entries.pop(k, None)
    for hook in list(_forget_hooks):
        try:
            hook(uid)
        except Exception:
            pass


def forget_user(user_id) -> None:
    """Drop every remembered credential of this user (ban, revoke, erase), in
    this process and in every other API process (``bump``)."""
    _forget_here(str(user_id))
    bump()


# --- Several API processes ----------------------------------------------------
# A split API (serve.py) runs the routes in several processes, each with its
# own copy of these memories, and the forget comes from one of them (a ban in
# the support API, a revoke, a Stripe event). CACHE_EPOCH_FILE, on the disk
# they all share (both containers of a deploy too), is touched on every
# forget; a process that sees it change drops everything it remembers, so a
# credential that stops working stops passing everywhere within
# EPOCH_CHECK_SECONDS. Unset (one process): nothing to tell anyone.
EPOCH_CHECK_SECONDS = 0.5
_epoch = {"file": None, "seen": None, "checked": 0.0}


def _epoch_file():
    return os.environ.get("CACHE_EPOCH_FILE") or None


def bump() -> None:
    """Tell the other API processes to drop what they remember."""
    path = _epoch_file()
    if not path:
        return
    try:
        with open(path, "a"):
            pass
        os.utime(path, None)
    except OSError:
        pass
    # Our own clear already happened; do not clear again on our own bump.
    _epoch["seen"] = _stamp(path)


def _stamp(path):
    try:
        st = os.stat(path)
        return (st.st_mtime_ns, st.st_size, st.st_ino)
    except OSError:
        return None


def sync() -> None:
    """Drop everything remembered here if another process forgot someone.
    Cheap: one stat every EPOCH_CHECK_SECONDS at most."""
    path = _epoch_file()
    if not path:
        return
    now = time.monotonic()
    if _epoch["file"] == path and now - _epoch["checked"] < EPOCH_CHECK_SECONDS:
        return
    stamp = _stamp(path)
    first = _epoch["file"] != path
    _epoch.update(file=path, checked=now)
    if first:
        _epoch["seen"] = stamp
        return
    if stamp != _epoch["seen"]:
        _epoch["seen"] = stamp
        _forget_here(None)


def clear() -> None:
    global _gate
    _entries.clear()
    _gate = None
    _epoch.update(file=None, seen=None, checked=0.0)
