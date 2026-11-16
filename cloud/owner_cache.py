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


def forget_user(user_id) -> None:
    """Drop every remembered credential of this user (ban, revoke, erase)."""
    uid = str(user_id)
    for k in [k for k, v in _entries.items() if v[1] == uid]:
        _entries.pop(k, None)


def clear() -> None:
    global _gate
    _entries.clear()
    _gate = None
