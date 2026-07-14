"""Minute metering: quota reservation, commit and release — atomic and restart-safe.

Accounting model (all serialized by a per-user ``SELECT ... FOR UPDATE`` lock):

* A subscription grants ``minutes_per_period`` for the window
  ``[current_period_start, current_period_end)``. Usage against the plan is the
  sum of ledger rows tagged with the current ``period_end`` and status
  reserved|committed. Rows from previous periods carry a different ``period_end``
  and stop counting automatically → renewal needs no cron.
* Top-ups form a FIFO pool (``minutes_total - minutes_consumed``) that persists
  across periods and survives cancellation.
* A reservation consumes IMMEDIATELY (not at commit): plan first, then top-ups
  FIFO. Because the whole split happens while holding the user lock, concurrent
  reservations can never oversell. ``commit`` just flips the row to committed;
  ``release`` refunds the exact top-up allocation recorded on the row.

Probing input duration (ffprobe / yt-dlp metadata) lives here too.
"""
import asyncio
import json
import math
import os
import subprocess
from datetime import datetime, timezone, timedelta
from decimal import Decimal

from sqlalchemy import select, update, func, and_

from . import database
from .models import User, Subscription, CreditTopup, UsageLedger


SWEEP_INTERVAL_SECONDS = 15 * 60
STUCK_RESERVATION_HOURS = 3


class QuotaExceeded(Exception):
    def __init__(self, remaining: float, required: float):
        self.remaining = remaining
        self.required = required
        super().__init__(f"Quota exceeded: need {required} min, {remaining} remaining")


def _now():
    return datetime.now(timezone.utc)


def _D(x) -> Decimal:
    return Decimal(str(x))


# --------------------------------------------------------------------------- #
# Duration probing
# --------------------------------------------------------------------------- #
def probe_file_minutes(path: str) -> float:
    """Return the media duration in minutes via ffprobe. Raises on failure."""
    out = subprocess.check_output(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", path],
        stderr=subprocess.STDOUT,
    )
    seconds = float(out.decode().strip())
    return seconds / 60.0


def probe_url_minutes(url: str) -> float:
    """Return the video duration in minutes from yt-dlp metadata (no download).

    Raises ValueError if the duration is unknown (e.g. live streams) — the caller
    turns that into a 400 for managed users.
    """
    import yt_dlp
    with yt_dlp.YoutubeDL({"skip_download": True, "quiet": True, "no_warnings": True}) as ydl:
        info = ydl.extract_info(url, download=False)
    duration = info.get("duration")
    if not duration:
        raise ValueError("Could not determine video duration")
    return float(duration) / 60.0


# --------------------------------------------------------------------------- #
# Balance computation (assumes caller holds the user lock when mutating)
# --------------------------------------------------------------------------- #
async def _active_subscription(session, user_id):
    row = (await session.execute(
        select(Subscription).where(Subscription.user_id == user_id)
    )).scalar_one_or_none()
    if row and row.status in ("active", "trialing") and row.current_period_end > _now():
        return row
    return None


async def _plan_used_this_period(session, user_id, period_end) -> Decimal:
    total = (await session.execute(
        select(func.coalesce(func.sum(UsageLedger.minutes_from_plan), 0)).where(and_(
            UsageLedger.user_id == user_id,
            UsageLedger.period_end == period_end,
            UsageLedger.status.in_(("reserved", "committed")),
        ))
    )).scalar_one()
    return _D(total)


async def _topups_fifo(session, user_id):
    return list((await session.execute(
        select(CreditTopup).where(CreditTopup.user_id == user_id)
        .order_by(CreditTopup.created_at.asc())
    )).scalars())


async def _balance(session, user_id):
    """Return a dict of the user's current minute balance (read-only)."""
    sub = await _active_subscription(session, user_id)
    plan_allowance = _D(sub.minutes_per_period) if sub else _D(0)
    period_end = sub.current_period_end if sub else None
    plan_used = await _plan_used_this_period(session, user_id, period_end) if sub else _D(0)
    plan_remaining = max(_D(0), plan_allowance - plan_used)

    topups = await _topups_fifo(session, user_id)
    topup_remaining = sum((_D(t.minutes_total) - _D(t.minutes_consumed) for t in topups), _D(0))

    return {
        "plan": sub.plan if sub else None,
        "plan_allowance": float(plan_allowance),
        "plan_used": float(plan_used),
        "plan_remaining": float(plan_remaining),
        "topup_remaining": float(topup_remaining),
        "remaining": float(plan_remaining + topup_remaining),
        "period_end": period_end,
        "_sub": sub,
        "_plan_remaining_d": plan_remaining,
        "_topups": topups,
    }


async def get_balance(user_id) -> dict:
    """Public read-only balance for /api/me."""
    async with database.session() as session:
        b = await _balance(session, user_id)
    return {k: v for k, v in b.items() if not k.startswith("_")}


def has_topup_credit_sync(remaining: float) -> bool:
    return remaining > 0


# --------------------------------------------------------------------------- #
# Reserve / commit / release
# --------------------------------------------------------------------------- #
async def reserve_minutes(user_id, minutes: float, job_id: str, job_type: str = "process"):
    """Atomically reserve ``minutes``. Returns the ledger row id.

    Raises ``QuotaExceeded`` if the user lacks the minutes. Consumes plan first,
    then top-ups FIFO, all under the per-user lock.
    """
    minutes = _D(minutes)
    async with database.session() as session:
        async with session.begin():
            # Serialize all of this user's reservations.
            await session.execute(
                select(User.id).where(User.id == user_id).with_for_update()
            )
            b = await _balance(session, user_id)
            plan_remaining = b["_plan_remaining_d"]
            remaining_total = _D(b["remaining"])
            if minutes > remaining_total:
                raise QuotaExceeded(remaining=float(remaining_total), required=float(minutes))

            from_plan = min(minutes, plan_remaining)
            from_topup = minutes - from_plan

            allocations = []
            need = from_topup
            if need > 0:
                for t in b["_topups"]:
                    if need <= 0:
                        break
                    avail = _D(t.minutes_total) - _D(t.minutes_consumed)
                    if avail <= 0:
                        continue
                    take = min(avail, need)
                    t.minutes_consumed = _D(t.minutes_consumed) + take
                    allocations.append({"topup_id": str(t.id), "minutes": float(take)})
                    need -= take

            row = UsageLedger(
                user_id=user_id,
                job_id=job_id,
                job_type=job_type,
                minutes=minutes,
                minutes_from_plan=from_plan,
                minutes_from_topup=from_topup,
                topup_allocations=allocations or None,
                status="reserved",
                period_end=b["period_end"],
            )
            session.add(row)
            await session.flush()
            return str(row.id)


async def commit_reservation(ledger_id: str):
    """Flip a reservation to committed. Consumption already happened at reserve."""
    async with database.session() as session:
        async with session.begin():
            row = await session.get(UsageLedger, ledger_id)
            if row and row.status == "reserved":
                row.status = "committed"


async def release_reservation(ledger_id: str):
    """Release a reservation and refund its exact top-up allocation."""
    async with database.session() as session:
        async with session.begin():
            row = await session.get(UsageLedger, ledger_id)
            if not row or row.status != "reserved":
                return
            await session.execute(
                select(User.id).where(User.id == row.user_id).with_for_update()
            )
            for alloc in (row.topup_allocations or []):
                t = await session.get(CreditTopup, alloc["topup_id"])
                if t is not None:
                    t.minutes_consumed = max(_D(0), _D(t.minutes_consumed) - _D(alloc["minutes"]))
            row.status = "released"


async def release_orphaned_reservations():
    """At startup, release every still-``reserved`` row.

    Jobs live only in memory, so a restart loses all in-flight jobs; their
    reservations must be refunded or they would leak quota forever.
    """
    async with database.session() as session:
        ids = list((await session.execute(
            select(UsageLedger.id).where(UsageLedger.status == "reserved")
        )).scalars())
    for lid in ids:
        await release_reservation(str(lid))
    if ids:
        print(f"☁️  Released {len(ids)} orphaned reservation(s) at startup.")


async def _sweep_once():
    cutoff = _now() - timedelta(hours=STUCK_RESERVATION_HOURS)
    async with database.session() as session:
        ids = list((await session.execute(
            select(UsageLedger.id).where(and_(
                UsageLedger.status == "reserved",
                UsageLedger.created_at < cutoff,
            ))
        )).scalars())
    for lid in ids:
        await release_reservation(str(lid))
    if ids:
        print(f"☁️  Swept {len(ids)} stuck reservation(s).")


async def _sweeper_loop():
    while True:
        try:
            await asyncio.sleep(SWEEP_INTERVAL_SECONDS)
            await _sweep_once()
        except asyncio.CancelledError:
            break
        except Exception as e:  # never let the sweeper die
            print(f"⚠️  Metering sweeper error: {e}")


def start_sweeper():
    asyncio.create_task(_sweeper_loop())
