"""Durable trail + Telegram alert for every byte sent through the paid proxy.

Two producers:
- ``main.py`` prints ``PROXY_ROUTE=<json>`` after a download (winner label,
  paid bytes summed over every attempt that used the per-GB proxy, and the
  error text of each free attempt that failed first). ``app.py``'s log reader
  parses it and calls :func:`record_download` when the job ends.
- ``cloud.metering.probe_url_minutes`` appends an event when the duration
  probe had to reach the paid proxy; ``app.py`` drains those right after the
  probe with :func:`drain_probe_events`.

Both persist a ``proxy_usage`` row (best effort) and page the admin. The
alert has a short cooldown that folds a burst into one message: on a day like
28-aug-2026, when every job fell to the paid proxy, one Telegram line per
job would have been 180 messages saying the same thing.
"""
from __future__ import annotations

import asyncio
import datetime as _dt
import json
import os
import time
from typing import Optional
from urllib.parse import urlparse

PAID_LABELS = {"HD", "fallback"}       # download attempts billed per GB (main.plan_download_attempts)
ALERT_COOLDOWN_SECONDS = 300

# Hard ceiling on paid-proxy traffic per UTC day. Above it the paid proxy is
# taken out of every chain until midnight: jobs that need it fail with a clear
# message instead of bleeding money. 28-aug-2026 was $14 in one day because
# nothing capped the fallback while the static IPs were being refused by
# YouTube for hours. 0 disables the cap.
DAILY_BUDGET_MB = float(os.environ.get("PAID_PROXY_DAILY_MB", "500"))

_budget_cache = {"at": 0.0, "bytes": -1}
_budget_alerted = {"day": None}

_pending = {"count": 0, "bytes": 0, "lines": [], "last_sent": 0.0}
_lock = asyncio.Lock()


def host_of(url: Optional[str]) -> Optional[str]:
    try:
        return urlparse(url or "").hostname or None
    except Exception:
        return None


def parse_route_line(line: str) -> Optional[dict]:
    """``PROXY_ROUTE={...}`` -> dict, or None for anything malformed."""
    if not line.startswith("PROXY_ROUTE="):
        return None
    try:
        data = json.loads(line.split("=", 1)[1])
    except ValueError:
        return None
    return data if isinstance(data, dict) else None


def paid_bytes_of(route: Optional[dict]) -> int:
    try:
        return int((route or {}).get("paid_bytes") or 0)
    except (TypeError, ValueError):
        return 0


def used_paid(route: Optional[dict]) -> bool:
    """True when the paid proxy carried bytes or won the download."""
    if not route:
        return False
    return paid_bytes_of(route) > 0 or str(route.get("winner") or "") in PAID_LABELS


def describe_failures(route: Optional[dict], limit: int = 3) -> str:
    """The free attempts that failed before the paid one, one per line."""
    out = []
    for a in (route or {}).get("attempts") or []:
        if a.get("ok") or a.get("label") in PAID_LABELS:
            continue
        err = str(a.get("error") or "").strip().replace("\n", " ")
        if err:
            out.append(f"  {a.get('label')}: {err[:140]}")
        if len(out) >= limit:
            break
    return "\n".join(out)


async def _persist(source: str, job_id: Optional[str], url_host: Optional[str],
                   route_label: Optional[str], paid_bytes: int, detail: Optional[dict]):
    try:
        from . import database as _db
        from .models import ProxyUsage
        async with _db.session() as s:
            s.add(ProxyUsage(source=source, job_id=job_id, url_host=url_host,
                             route=route_label, paid_bytes=int(paid_bytes or 0),
                             detail=detail))
            await s.commit()
    except Exception as e:  # accounting must never break a job
        print(f"⚠️ proxy_usage row not written: {e}")


async def _alert(line: str, nbytes: int):
    """Page the admin, folding events inside the cooldown into one message."""
    try:
        from . import alerts
    except Exception:
        return
    async with _lock:
        _pending["count"] += 1
        _pending["bytes"] += int(nbytes or 0)
        if len(_pending["lines"]) < 5:
            _pending["lines"].append(line)
        now = time.time()
        if now - _pending["last_sent"] < ALERT_COOLDOWN_SECONDS:
            return
        count, total = _pending["count"], _pending["bytes"]
        lines = list(_pending["lines"])
        _pending.update(count=0, bytes=0, lines=[], last_sent=now)
    head = ("💸 Paid proxy (DataImpulse) used"
            + (f": {count} events, {total / 1e6:.1f} MB in the last {ALERT_COOLDOWN_SECONDS // 60} min"
               if count > 1 else ""))
    await alerts.notify(head + "\n" + "\n".join(lines))


async def flush_alerts():
    """Send whatever the cooldown is still holding (called by a periodic tick)."""
    async with _lock:
        if not _pending["count"] or time.time() - _pending["last_sent"] < ALERT_COOLDOWN_SECONDS:
            return
        count, total = _pending["count"], _pending["bytes"]
        lines = list(_pending["lines"])
        _pending.update(count=0, bytes=0, lines=[], last_sent=time.time())
    try:
        from . import alerts
        await alerts.notify(
            f"💸 Paid proxy (DataImpulse) used: {count} events, {total / 1e6:.1f} MB\n" + "\n".join(lines))
    except Exception:
        pass


async def paid_bytes_today() -> int:
    """Paid bytes recorded since UTC midnight (60 s cache; -1 -> unknown)."""
    now = time.time()
    if now - _budget_cache["at"] < 60 and _budget_cache["bytes"] >= 0:
        return _budget_cache["bytes"]
    try:
        from sqlalchemy import func, select
        from . import database as _db
        from .models import ProxyUsage
        day_start = _dt.datetime.now(_dt.timezone.utc).replace(hour=0, minute=0,
                                                               second=0, microsecond=0)
        async with _db.session() as sess:
            total = (await sess.execute(
                select(func.coalesce(func.sum(ProxyUsage.paid_bytes), 0))
                .where(ProxyUsage.created_at >= day_start))).scalar_one()
        _budget_cache.update(at=now, bytes=int(total or 0))
        return _budget_cache["bytes"]
    except Exception as e:
        print(f"⚠️ paid_bytes_today failed ({e}); treating budget as not exceeded")
        return 0


async def budget_exceeded() -> bool:
    """True when today's paid traffic is over PAID_PROXY_DAILY_MB.

    Fails open (False) when the DB is unreachable: refusing every YouTube job
    because the accounting query broke would be the worse failure. Alerts
    once per UTC day when the cap trips.
    """
    if DAILY_BUDGET_MB <= 0:
        return False
    used = await paid_bytes_today()
    if used < DAILY_BUDGET_MB * 1e6:
        return False
    day = _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%d")
    if _budget_alerted["day"] != day:
        _budget_alerted["day"] = day
        try:
            from . import alerts
            await alerts.notify(
                f"⛔ Paid proxy daily budget hit: {used / 1e6:.0f} MB of "
                f"{DAILY_BUDGET_MB:.0f} MB used today. DataImpulse is disabled "
                "until UTC midnight; jobs that need it will fail with a clear error. "
                "Raise PAID_PROXY_DAILY_MB if this is expected.", urgent=True)
        except Exception:
            pass
    return True


def budget_exceeded_sync() -> bool:
    """Non-blocking view for sync callers (the resume scan): last cached
    value only, unknown counts as not exceeded."""
    if DAILY_BUDGET_MB <= 0:
        return False
    cached = _budget_cache["bytes"]
    return cached >= 0 and cached >= DAILY_BUDGET_MB * 1e6


async def record_download(job_id: str, route: Optional[dict], source_url: Optional[str] = None):
    """Persist a download's route; alert when the paid proxy was involved."""
    if not route:
        return
    nbytes = paid_bytes_of(route)
    paid = used_paid(route)
    await _persist("download", job_id, host_of(source_url), str(route.get("winner") or "none"),
                   nbytes if paid else 0, route)
    if paid:
        why = describe_failures(route)
        await _alert(f"download job {job_id[:8]}: {nbytes / 1e6:.1f} MB via '{route.get('winner')}'"
                     + (f"\n{why}" if why else ""), nbytes)


async def record_probe(event: dict):
    """A duration probe that reached the paid proxy (see cloud.metering)."""
    url_host = host_of(event.get("url"))
    await _persist("probe", event.get("job_id"), url_host, "paid-probe",
                   int(event.get("bytes_estimate") or 0), event)
    fails = "\n".join(f"  {k}: {str(v)[:140]}" for k, v in (event.get("static_errors") or {}).items())
    paid_failed = event.get("paid_failed")
    outcome = (f"paid proxy failed too: {str(paid_failed)[:140]}" if paid_failed
               else "paid proxy answered")
    await _alert(f"probe {url_host or '?'}: statics failed, {outcome}"
                 + (f"\n{fails}" if fails else ""), int(event.get("bytes_estimate") or 0))


async def drain_probe_events():
    """Record every paid probe the metering module queued since the last call."""
    try:
        from . import metering
        events = metering.pop_paid_probe_events()
    except Exception:
        return
    for ev in events:
        await record_probe(ev)
