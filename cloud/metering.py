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

from . import config, database
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

    Uses the same proxy + extractor settings as the actual download (main.py) so
    the probe behaves consistently with it. Metadata only — negligible bandwidth.
    Raises ValueError if the duration is unknown (e.g. live streams).
    """
    import yt_dlp
    # SSRF guard: reject non-http(s) / private / metadata hosts before probing.
    from security_utils import assert_public_url
    assert_public_url(url)

    bgutil_http = os.environ.get("BGUTIL_BASE_URL", "").strip()
    bgutil_script = os.environ.get("BGUTIL_SCRIPT_PATH", "").strip()
    proxy = os.environ.get("PROXY_URL", "").strip()
    conservative = {"youtube": {"player_client": ["tv_embed", "android", "mweb", "web"],
                                "player_skip": ["webpage", "configs"]}}
    # Try the bgutil/HD extractor first (http or baked-in script), then the
    # conservative one — mirrors the download's HD→fallback logic.
    if bgutil_http:
        hd = [{"youtubepot-bgutilhttp": {"base_url": [bgutil_http]}}]
    elif bgutil_script:
        hd = [{"youtubepot-bgutilscript": {"script_path": [bgutil_script]}}]
    else:
        hd = []
    strategies = hd + [conservative]

    last_err = None
    for extractor_args in strategies:
        opts = {"skip_download": True, "quiet": True, "no_warnings": True,
                "extractor_args": extractor_args}
        if proxy:
            opts["proxy"] = proxy
        try:
            with yt_dlp.YoutubeDL(opts) as ydl:
                info = ydl.extract_info(url, download=False)
            duration = info.get("duration")
            if duration:
                return float(duration) / 60.0
            last_err = ValueError("no duration in metadata")
        except Exception as e:
            last_err = e
    raise ValueError(f"Could not determine video duration ({last_err})")


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


def free_plan_eligible(user) -> bool:
    """True if this ``User`` row qualifies for the free monthly allowance.

    Free minutes are only for Google-authenticated accounts: magic-link-only
    sign-ins accept disposable-email domains and would make the free plan a
    multi-account faucet. FREE_PLAN_MINUTES = 0 disables the free plan.
    """
    if user is None or config.FREE_PLAN_MINUTES <= 0:
        return False
    if config.FREE_REQUIRES_GOOGLE and not user.google_sub:
        return False
    return True


def free_period_end(now: datetime | None = None) -> datetime:
    """First instant of the next UTC calendar month.

    Free-plan ledger rows are tagged with this synthetic ``period_end`` so the
    existing period accounting resets them monthly with no cron, exactly like
    Stripe periods. A real subscription's ``current_period_end`` is anchored to
    the checkout instant, so it can only collide with this value if it lands on
    a month start to the second — accepted as negligible.
    """
    now = now or _now()
    if now.month == 12:
        return datetime(now.year + 1, 1, 1, tzinfo=timezone.utc)
    return datetime(now.year, now.month + 1, 1, tzinfo=timezone.utc)


async def is_free_user(session, user_id) -> bool:
    """Google-authed user currently on the free plan (no active/trialing sub)."""
    if await _active_subscription(session, user_id):
        return False
    return free_plan_eligible(await session.get(User, user_id))


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


async def _balance(session, user_id, user=None):
    """Return a dict of the user's current minute balance (read-only).

    ``user`` may be a pre-fetched ``User`` row to save a lookup; it is only
    consulted on the subscription-less path (free-plan eligibility).
    """
    sub = await _active_subscription(session, user_id)
    if sub:
        plan_name = sub.plan
        plan_allowance = _D(sub.minutes_per_period)
        # During the trial, cap the allowance so a cancel-before-charge account
        # can't burn a whole plan's worth of managed minutes. Kept for
        # grandfathered 'trialing' subscriptions.
        if sub.status == "trialing":
            plan_allowance = min(plan_allowance, _D(config.TRIAL_MINUTE_CAP))
        period_end = sub.current_period_end
        plan_used = await _plan_used_this_period(session, user_id, period_end)
    else:
        if user is None:
            user = await session.get(User, user_id)
        if free_plan_eligible(user):
            plan_name = "free"
            plan_allowance = _D(config.FREE_PLAN_MINUTES)
            period_end = free_period_end()
            plan_used = await _plan_used_this_period(session, user_id, period_end)
        else:
            plan_name = None
            plan_allowance = _D(0)
            period_end = None
            plan_used = _D(0)
    plan_remaining = max(_D(0), plan_allowance - plan_used)

    topups = await _topups_fifo(session, user_id)
    topup_remaining = sum((_D(t.minutes_total) - _D(t.minutes_consumed) for t in topups), _D(0))

    return {
        "plan": plan_name,
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
            # Serialize all of this user's reservations. Selecting the full row
            # (same lock semantics) lets _balance skip a second User lookup.
            locked_user = (await session.execute(
                select(User).where(User.id == user_id).with_for_update()
            )).scalar_one_or_none()
            b = await _balance(session, user_id, user=locked_user)
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
