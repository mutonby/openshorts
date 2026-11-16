"""Support-ops API v1: the support back-office looks up and fixes one account.

A small HTTP API under ``/api/support-ops/v1`` with the same shape in every
product, so the back-office keeps one client. It is not part of the product:
no UI links to it, it is left out of the OpenAPI schema and it sends no CORS
headers (``CorsGuard``).

Switches and auth
- ``SUPPORT_OPS_READ_KEY`` (GET only) and ``SUPPORT_OPS_WRITE_KEY`` (GET and
  POST), sent as ``Authorization: Bearer <key>`` and compared in constant
  time. With neither set every route answers 404, as if it did not exist. A
  key shorter than ``MIN_KEY_LEN`` keeps the whole feature off (warning at
  startup). Wrong or missing key: 401. The read key on a POST: 403.

Writes
- ``POST /users/{id}/actions/{action}`` with ``{"params", "dry_run", "ticket",
  "actor"}``. ``dry_run`` defaults to true: the action runs inside a
  transaction that is rolled back, so ``after`` is what the real write would
  leave. ``before``/``after`` hold only the fields the action touches, and
  ``restore`` takes a ``before`` back.
- A real write needs a ticket and leaves one structured ``support-ops audit``
  log line (ticket, actor, action, user id, before, after).
- These routes never email the customer, never call Stripe (the back-office
  handles Stripe itself) and only ever touch the one account in the path.

Plans, minutes and top-ups follow cloud/metering.py and cloud/config.py; this
module adds no plan rules of its own.
"""
import asyncio
import hmac
import json
import math
import os
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from decimal import Decimal, ROUND_HALF_UP
from typing import Optional

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse
from sqlalchemy import select, func, delete

from . import config, database, email_policy, metering
from .models import (
    AccountDeletion, ApiKey, AutopilotRun, AutopilotSettings, CancellationFeedback,
    CreditTopup, EvidenceEvent, MagicLinkToken, Project, Subscription, UploadPostProfile,
    UsageLedger, User,
)

PREFIX = "/api/support-ops/v1"
READ_KEY_ENV = "SUPPORT_OPS_READ_KEY"
WRITE_KEY_ENV = "SUPPORT_OPS_WRITE_KEY"
MIN_KEY_LEN = 32
MAX_LOOKUP = 5
ACTIVITY_DEFAULT = 50
ACTIVITY_MAX = 200
ERROR_TEXT_MAX = 500
MAX_CREDIT_CHANGE = 100_000
# Ledger rows written by reset_usage / restore. Committed rows whose
# minutes_from_plan moves the period's usage to the requested value; every
# other reader of the ledger filters on job_type == "process".
ADJUST_JOB_TYPE = "support_adjust"
PAID_PLANS = tuple(config.PLAN_MINUTES)
PLANS = ("free",) + PAID_PLANS
INTERVALS = ("month", "year")
SUB_STATUSES = ("active", "trialing", "past_due", "unpaid", "canceled", "incomplete",
                "incomplete_expired", "paused", config.ON_HOLD)
# update_field may write these users columns and nothing else.
FIELD_ALLOWLIST = {
    "marketing_opt_out": "bool: the account opted out of commercial email",
    "free_plan_denied": "string|null: why the account gets no free monthly minutes "
                        "(e.g. 'disposable_mx', 'recreated_after_deletion'); null restores the free plan",
}
_SLUG_RE = re.compile(r"^[a-z0-9_]{1,32}$")

router = APIRouter(prefix=PREFIX, include_in_schema=False)

_keys = {"read": None, "write": None}


# --------------------------------------------------------------------------- #
# Switches and auth
# --------------------------------------------------------------------------- #
def load_keys(environ=None) -> bool:
    """Read the two keys from the environment (startup). Returns enabled()."""
    env = os.environ if environ is None else environ
    read = (env.get(READ_KEY_ENV) or "").strip() or None
    write = (env.get(WRITE_KEY_ENV) or "").strip() or None
    short = [name for name, v in ((READ_KEY_ENV, read), (WRITE_KEY_ENV, write))
             if v is not None and len(v) < MIN_KEY_LEN]
    if short:
        print(f"⚠️  {', '.join(short)} shorter than {MIN_KEY_LEN} characters: "
              "the support-ops API stays off.")
        read = write = None
    if read is not None and read == write:
        print(f"⚠️  {READ_KEY_ENV} equals {WRITE_KEY_ENV}: it is treated as the write key.")
        read = None
    _keys["read"], _keys["write"] = read, write
    return enabled()


def enabled() -> bool:
    return bool(_keys["read"] or _keys["write"])


def _not_found():
    return HTTPException(status_code=404, detail="Not Found")


def _authorize(request: Request, write: bool):
    if not enabled():
        raise _not_found()
    header = request.headers.get("authorization", "")
    token = header[7:].strip() if header[:7].lower() == "bearer " else ""
    given = token.encode()
    # Both comparisons always run: the timing says nothing about which key matched.
    is_write = bool(_keys["write"]) & hmac.compare_digest(given, (_keys["write"] or "\0").encode())
    is_read = bool(_keys["read"]) & hmac.compare_digest(given, (_keys["read"] or "\0").encode())
    if not (is_write or is_read):
        raise HTTPException(status_code=401, detail="Invalid support-ops key.")
    if write and not is_write:
        raise HTTPException(status_code=403, detail="This key is read-only.")


class CorsGuard:
    """ASGI middleware, installed outside the CORS middleware (app.py): these
    routes never carry CORS headers and never answer a preflight, and with the
    feature off every path under the prefix is a plain 404, whatever the
    method."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope.get("type") != "http" or not scope.get("path", "").startswith(PREFIX):
            return await self.app(scope, receive, send)
        if not enabled() or scope.get("method") == "OPTIONS":
            response = JSONResponse({"detail": "Not Found"}, status_code=404)
            return await response(scope, receive, send)

        async def send_without_cors(message):
            if message.get("type") == "http.response.start":
                message = dict(message)
                message["headers"] = [
                    (k, v) for k, v in message.get("headers", [])
                    if not k.lower().startswith(b"access-control-")
                ]
            await send(message)

        # Drop the Origin header too, so the CORS middleware treats the call
        # as a same-origin one and adds nothing.
        headers = [(k, v) for k, v in scope.get("headers", []) if k.lower() != b"origin"]
        await self.app(dict(scope, headers=headers), receive, send_without_cors)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _now():
    return datetime.now(timezone.utc)


def _iso(dt) -> Optional[str]:
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).isoformat()


def _num(value) -> float:
    return float(Decimal(str(value)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


def _parse_uuid(value) -> Optional[uuid.UUID]:
    try:
        return uuid.UUID(str(value))
    except (ValueError, TypeError, AttributeError):
        return None


def _parse_instant(value, name) -> datetime:
    if not isinstance(value, str) or not value.strip():
        raise _Bad(f"'{name}' must be an ISO-8601 date or datetime string.")
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(text)
    except ValueError:
        raise _Bad(f"'{name}' is not an ISO-8601 date or datetime: {value!r}.")
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _add_interval(start: datetime, interval: str) -> datetime:
    """One billing interval after ``start`` (same day of month, clamped)."""
    months = 12 if interval == "year" else 1
    y, m = divmod(start.month - 1 + months, 12)
    year, month = start.year + y, m + 1
    for day in (start.day, 30, 29, 28):
        try:
            return start.replace(year=year, month=month, day=day)
        except ValueError:
            continue
    return start + timedelta(days=30 * months)


class _Bad(Exception):
    """Invalid params: answered as 400 with this message."""


class _Conflict(Exception):
    """A write that would clash with another account or the account's state: 409."""


def _http(e: Exception) -> HTTPException:
    if isinstance(e, _Conflict):
        return HTTPException(status_code=409, detail=str(e))
    return HTTPException(status_code=400, detail=str(e))


# --------------------------------------------------------------------------- #
# UserView
# --------------------------------------------------------------------------- #
def _sub_snapshot(row) -> Optional[dict]:
    if row is None:
        return None
    return {
        "stripe_subscription_id": row.stripe_subscription_id,
        "plan": row.plan,
        "interval": row.interval,
        "status": row.status,
        "minutes_per_period": int(row.minutes_per_period),
        "current_period_start": _iso(row.current_period_start),
        "current_period_end": _iso(row.current_period_end),
    }


async def _credits(session, user_id) -> float:
    topups = await metering._topups_fifo(session, user_id)
    return _num(sum((Decimal(t.minutes_total) - Decimal(t.minutes_consumed) for t in topups),
                    Decimal(0)))


async def user_view(session, user) -> dict:
    """The account as the back-office sees it. Never a secret: no token, key,
    hash or third-party credential is read here."""
    b = await metering._balance(session, user.id, user=user)
    sub_row, sub = b["_sub_row"], b["_sub"]
    if sub is not None:
        period_start = sub.current_period_start
    elif b["plan"] == "free":
        period_start = _now().replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    else:
        period_start = None
    keys_active, keys_last_used = (await session.execute(
        select(func.count(ApiKey.id), func.max(ApiKey.last_used_at))
        .where(ApiKey.user_id == user.id, ApiKey.revoked_at.is_(None))
    )).one()
    topup_count = (await session.execute(
        select(func.count(CreditTopup.id)).where(CreditTopup.user_id == user.id)
    )).scalar_one()
    profile = await session.get(UploadPostProfile, user.id)
    autopilot = await session.get(AutopilotSettings, user.id)
    return {
        "id": str(user.id),
        "email": user.email,
        "name": None,
        "created_at": _iso(user.created_at),
        "last_login_at": _iso(user.last_login_at),
        "auth_provider": "google" if user.google_sub else "magic_link",
        "plan": b["plan"],
        "plan_status": sub_row.status if sub_row is not None else None,
        "billing_interval": sub_row.interval if sub_row is not None else None,
        "stripe_customer_id": user.stripe_customer_id,
        "stripe_subscription_id": sub_row.stripe_subscription_id if sub_row is not None else None,
        "usage": {
            "used": _num(b["plan_used"]),
            "limit": _num(b["plan_allowance"]),
            "unit": "minutes",
            "period_start": _iso(period_start),
            "period_end": _iso(b["period_end"]),
        },
        "credits": _num(b["topup_remaining"]),
        "banned": bool(user.ban_reason),
        "ban_reason": user.ban_reason,
        "deleted": False,
        "extra": {
            "credits_unit": "minutes",
            "remaining_minutes": _num(b["remaining"]),
            "plan_remaining_minutes": _num(b["plan_remaining"]),
            "banned_at": _iso(user.banned_at),
            "free_plan_denied": user.free_plan_denied,
            "marketing_opt_out": bool(user.marketing_opt_out),
            "google_linked": bool(user.google_sub),
            "cancel_at_period_end": bool(sub_row.cancel_at_period_end) if sub_row is not None else None,
            "hold_from": _iso(sub_row.hold_from) if sub_row is not None else None,
            "hold_until": _iso(sub_row.hold_until) if sub_row is not None else None,
            "subscription_period_end": _iso(sub_row.current_period_end) if sub_row is not None else None,
            "topups": int(topup_count or 0),
            "api_keys_active": int(keys_active or 0),
            "api_keys_last_used_at": _iso(keys_last_used),
            "social_profile_linked": profile is not None,
            "autopilot_enabled": bool(autopilot and autopilot.enabled),
        },
    }


def _deleted_view(rec: AccountDeletion, email=None) -> dict:
    """An erased account: only what its erasure record keeps."""
    return {
        "id": rec.former_user_id, "email": email, "name": None,
        "created_at": None, "last_login_at": None, "auth_provider": None,
        "plan": None, "plan_status": None, "billing_interval": None,
        "stripe_customer_id": rec.stripe_customer_id, "stripe_subscription_id": None,
        "usage": {"used": None, "limit": None, "unit": "minutes",
                  "period_start": None, "period_end": None},
        "credits": None, "banned": False, "ban_reason": None, "deleted": True,
        "extra": {"deleted_at": _iso(rec.deleted_at), "plan_at_deletion": rec.plan_at_deletion,
                  "deletion_reason": rec.reason},
    }


# --------------------------------------------------------------------------- #
# Read routes
# --------------------------------------------------------------------------- #
@router.get("/users")
async def find_users(request: Request):
    _authorize(request, write=False)
    q = request.query_params
    given = [k for k in ("email", "id", "stripe_customer") if q.get(k)]
    if len(given) != 1:
        raise HTTPException(status_code=400, detail=(
            "Pass exactly one of ?email=, ?id= or ?stripe_customer=."))
    key = given[0]
    value = q[key].strip()
    from .account import email_fingerprint
    async with database.session() as session:
        deleted = []
        if key == "email":
            candidates = {value.lower(), email_policy.normalize_email(value)}
            users = (await session.execute(
                select(User).where(User.email.in_(candidates)).limit(MAX_LOOKUP)
            )).scalars().all()
            if not users:
                deleted = [(r, value) for r in (await session.execute(
                    select(AccountDeletion)
                    .where(AccountDeletion.email_sha256 == email_fingerprint(value))
                    .order_by(AccountDeletion.deleted_at.desc()).limit(MAX_LOOKUP)
                )).scalars().all()]
        elif key == "id":
            uid = _parse_uuid(value)
            users = [] if uid is None else [u for u in [await session.get(User, uid)] if u]
            if not users:
                deleted = [(r, None) for r in (await session.execute(
                    select(AccountDeletion).where(AccountDeletion.former_user_id == value)
                    .limit(MAX_LOOKUP)
                )).scalars().all()]
        else:
            users = (await session.execute(
                select(User).where(User.stripe_customer_id == value).limit(MAX_LOOKUP)
            )).scalars().all()
            if not users:
                deleted = [(r, None) for r in (await session.execute(
                    select(AccountDeletion).where(AccountDeletion.stripe_customer_id == value)
                    .order_by(AccountDeletion.deleted_at.desc()).limit(MAX_LOOKUP)
                )).scalars().all()]
        views = [await user_view(session, u) for u in users]
    views += [_deleted_view(r, email) for r, email in deleted]
    return {"users": views[:MAX_LOOKUP]}


@router.get("/users/{user_id}")
async def get_user(user_id: str, request: Request):
    _authorize(request, write=False)
    uid = _parse_uuid(user_id)
    async with database.session() as session:
        user = await session.get(User, uid) if uid else None
        if user is not None:
            return {"user": await user_view(session, user)}
        rec = (await session.execute(
            select(AccountDeletion).where(AccountDeletion.former_user_id == user_id).limit(1)
        )).scalar_one_or_none()
    if rec is None:
        raise HTTPException(status_code=404, detail="User not found.")
    return {"user": _deleted_view(rec)}


# app.py hands us ``(user_id) -> {job_id: {...}}``: the jobs this process holds
# in memory for that user (live status, error text, source). Without it the
# activity comes from the database alone.
_local_jobs = None


def register_local_jobs(fn):
    global _local_jobs
    _local_jobs = fn


def _item(type_, id_, created_at, status=None, title=None, error=None, minutes=None,
          source=None, detail=None) -> dict:
    return {"type": type_, "id": id_, "created_at": _iso(created_at), "status": status,
            "title": title, "error": (error or None) and str(error)[:ERROR_TEXT_MAX],
            "minutes": None if minutes is None else _num(minutes), "source": source,
            "detail": detail or {}}


_LEDGER_STATUS = {"reserved": "in_progress", "committed": "charged", "released": "refunded"}


async def activity(session, user_id, limit: int) -> list:
    """The account's recent activity, newest first: jobs (from the minutes
    ledger, with titles, Autopilot origin and, for jobs this process still
    holds, live status and error text), sign-ins and checkouts, top-ups,
    support adjustments, and cancel-flow outcomes (closed-list reason only)."""
    items = []
    ledger = (await session.execute(
        select(UsageLedger).where(UsageLedger.user_id == user_id)
        .order_by(UsageLedger.created_at.desc()).limit(limit)
    )).scalars().all()
    job_ids = {r.job_id for r in ledger}
    titles = {}
    if job_ids:
        titles = dict((await session.execute(
            select(Project.job_id, Project.title).where(Project.job_id.in_(job_ids))
        )).all())
    runs = (await session.execute(
        select(AutopilotRun).where(AutopilotRun.user_id == user_id)
        .order_by(AutopilotRun.created_at.desc()).limit(limit)
    )).scalars().all()
    autopilot_jobs = {r.job_id for r in runs if r.job_id}
    local = {}
    if _local_jobs is not None:
        try:
            local = _local_jobs(user_id) or {}
        except Exception as e:
            print(f"⚠️  support-ops: local job lookup failed: {e}")
    for r in ledger:
        if r.job_type == ADJUST_JOB_TYPE:
            items.append(_item("usage_adjustment", str(r.id), r.created_at, status="applied",
                               minutes=r.minutes_from_plan, source="support",
                               detail={"period_end": _iso(r.period_end)}))
            continue
        live = local.get(r.job_id) or {}
        charged = Decimal(0) if r.status == "released" else Decimal(r.minutes)
        items.append(_item(
            "job", r.job_id, r.created_at,
            status=live.get("status") or _LEDGER_STATUS.get(r.status, r.status),
            title=titles.get(r.job_id), error=live.get("error"), minutes=charged,
            source=("autopilot" if r.job_id in autopilot_jobs else live.get("source")),
            detail={"job_type": r.job_type, "ledger_status": r.status,
                    "minutes_reserved": _num(r.minutes),
                    "minutes_from_plan": _num(r.minutes_from_plan),
                    "minutes_from_topup": _num(r.minutes_from_topup),
                    "period_end": _iso(r.period_end)}))
    for job_id, live in local.items():
        if job_id in job_ids:
            continue
        created = live.get("created_at")
        if isinstance(created, (int, float)):
            created = datetime.fromtimestamp(created, tz=timezone.utc)
        items.append(_item("job", job_id, created, status=live.get("status"),
                           error=live.get("error"), source=live.get("source"),
                           detail={"ledger_status": None}))
    for r in runs:
        items.append(_item("autopilot", str(r.id), r.created_at, status=r.status,
                           title=r.video_title, error=r.reason, source=r.trigger,
                           detail={"job_id": r.job_id, "video_id": r.video_id,
                                   "clips": r.clips_count, "posted": r.posted_count}))
    for e in (await session.execute(
            select(EvidenceEvent).where(EvidenceEvent.user_id == user_id)
            .order_by(EvidenceEvent.created_at.desc()).limit(limit))).scalars().all():
        meta = e.meta or {}
        items.append(_item(e.kind, str(e.id), e.created_at, status="ok", source=e.method,
                           detail={k: meta[k] for k in ("plan", "kind", "minutes") if k in meta}))
    for t in (await session.execute(
            select(CreditTopup).where(CreditTopup.user_id == user_id)
            .order_by(CreditTopup.created_at.desc()).limit(limit))).scalars().all():
        items.append(_item("topup", str(t.id), t.created_at, status="credited",
                           minutes=t.minutes_total,
                           source="stripe" if t.stripe_session_id else "support",
                           detail={"minutes_consumed": _num(t.minutes_consumed),
                                   "minutes_left": _num(Decimal(t.minutes_total)
                                                        - Decimal(t.minutes_consumed))}))
    for c in (await session.execute(
            select(CancellationFeedback).where(CancellationFeedback.user_id == user_id)
            .order_by(CancellationFeedback.created_at.desc()).limit(limit))).scalars().all():
        items.append(_item("cancellation", str(c.id), c.created_at, status=c.outcome,
                           source="cancel_flow",
                           detail={"reason": c.reason, "rating": c.rating, "plan": c.plan,
                                   "interval": c.interval}))
    epoch = datetime.min.replace(tzinfo=timezone.utc).isoformat()
    items.sort(key=lambda i: i["created_at"] or epoch, reverse=True)
    return items[:limit]


@router.get("/users/{user_id}/activity")
async def get_activity(user_id: str, request: Request):
    _authorize(request, write=False)
    raw = request.query_params.get("limit")
    try:
        limit = ACTIVITY_DEFAULT if raw is None else int(raw)
    except ValueError:
        raise HTTPException(status_code=400, detail="'limit' must be an integer.")
    if limit < 1:
        raise HTTPException(status_code=400, detail="'limit' must be at least 1.")
    limit = min(limit, ACTIVITY_MAX)
    uid = _parse_uuid(user_id)
    async with database.session() as session:
        if uid is None or await session.get(User, uid) is None:
            raise HTTPException(status_code=404, detail="User not found.")
        return {"items": await activity(session, uid, limit)}


# --------------------------------------------------------------------------- #
# Actions
# --------------------------------------------------------------------------- #
@dataclass
class _Ctx:
    session: object
    user: User
    params: dict
    apply: bool
    ticket: Optional[str]
    after_commit: list = field(default_factory=list)


def _only(params, allowed, required=()):
    extra = set(params) - set(allowed)
    if extra:
        raise _Bad(f"Unknown params: {', '.join(sorted(extra))}.")
    missing = [k for k in required if k not in params]
    if missing:
        raise _Bad(f"Missing params: {', '.join(missing)}.")


def _number(value, name) -> Decimal:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise _Bad(f"'{name}' must be a number.")
    return Decimal(str(value)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)


def _email(value, name) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > 254:
        raise _Bad(f"'{name}' must be an email address.")
    try:
        from email_validator import validate_email, EmailNotValidError
    except ImportError:  # pragma: no cover - email-validator ships with billing deps
        validate_email = None
    if validate_email is not None:
        try:
            validate_email(value.strip(), check_deliverability=False)
        except EmailNotValidError as e:
            raise _Bad(f"'{name}' is not a valid email address: {e}")
    elif not re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", value.strip()):
        raise _Bad(f"'{name}' is not a valid email address.")
    return email_policy.normalize_email(value)


# --- primitives: each reads/writes one restorable piece of the account ----- #
async def _set_credits(ctx, target: Decimal):
    """Top-up balance to exactly ``target`` minutes (never below 0). Up: one new
    top-up row (whole minutes, the fraction pre-consumed). Down: consumed from
    the oldest top-ups first, the order metering spends them in."""
    if target < 0:
        raise _Bad("Credits cannot go below 0.")
    topups = await metering._topups_fifo(ctx.session, ctx.user.id)
    current = sum((Decimal(t.minutes_total) - Decimal(t.minutes_consumed) for t in topups),
                  Decimal(0))
    delta = target - current
    if delta > 0:
        whole = math.ceil(delta)
        ctx.session.add(CreditTopup(user_id=ctx.user.id, stripe_session_id=None,
                                    minutes_total=whole,
                                    minutes_consumed=Decimal(whole) - delta))
    elif delta < 0:
        need = -delta
        for t in topups:
            if need <= 0:
                break
            avail = Decimal(t.minutes_total) - Decimal(t.minutes_consumed)
            if avail <= 0:
                continue
            take = min(avail, need)
            t.minutes_consumed = Decimal(t.minutes_consumed) + take
            need -= take
    await ctx.session.flush()


async def _usage_snapshot(ctx) -> dict:
    b = await metering._balance(ctx.session, ctx.user.id, user=ctx.user)
    return {"used": _num(b["plan_used"]), "period_end": _iso(b["period_end"])}


async def _set_usage(ctx, target: Decimal, period_end_iso=None):
    """Plan minutes used this period to exactly ``target``, through one
    committed adjustment row in the ledger (nothing is rewritten)."""
    if target < 0:
        raise _Bad("Usage cannot go below 0.")
    b = await metering._balance(ctx.session, ctx.user.id, user=ctx.user)
    if period_end_iso is not None and period_end_iso != _iso(b["period_end"]):
        raise _Bad("The billing period changed since that snapshot; it cannot be restored.")
    used = Decimal(str(b["plan_used"]))
    delta = target - used
    if delta == 0:
        return
    if b["period_end"] is None:
        raise _Bad("This account has no plan period (no subscription and no free plan).")
    ctx.session.add(UsageLedger(
        user_id=ctx.user.id,
        job_id=f"support:{ctx.ticket or 'dry-run'}:{uuid.uuid4().hex[:12]}",
        job_type=ADJUST_JOB_TYPE, minutes=delta, minutes_from_plan=delta,
        minutes_from_topup=0, topup_allocations=None, status="committed",
        period_end=b["period_end"]))
    await ctx.session.flush()


async def _set_email(ctx, new_email: str):
    if new_email == ctx.user.email:
        return
    taken = (await ctx.session.execute(
        select(User.id).where(User.email == new_email, User.id != ctx.user.id).limit(1)
    )).scalar_one_or_none()
    if taken is not None:
        raise _Conflict("Another account already uses that email.")
    old = ctx.user.email
    ctx.user.email = new_email
    # Unused sign-in links still point at the old address, which no longer
    # names this account: one of them would open a new, empty account there.
    await ctx.session.execute(delete(MagicLinkToken).where(
        MagicLinkToken.email == old, MagicLinkToken.used_at.is_(None)))
    await ctx.session.flush()


def _ban_snapshot(user) -> dict:
    return {"ban_reason": user.ban_reason, "banned_at": _iso(user.banned_at)}


def _set_ban(ctx, reason, banned_at):
    ctx.user.ban_reason = reason
    ctx.user.banned_at = banned_at if reason else None
    if reason:
        from . import owner_cache
        owner_cache.forget_user(ctx.user.id)


def _check_field(name, value):
    if name not in FIELD_ALLOWLIST:
        raise _Bad(f"Field '{name}' is not editable. Editable: {', '.join(FIELD_ALLOWLIST)}.")
    if name == "marketing_opt_out" and not isinstance(value, bool):
        raise _Bad("'marketing_opt_out' must be true or false.")
    if name == "free_plan_denied" and value is not None and not (
            isinstance(value, str) and _SLUG_RE.match(value)):
        raise _Bad("'free_plan_denied' must be null or a lowercase label of up to 32 characters.")
    return value


async def _subscription(ctx):
    return (await ctx.session.execute(
        select(Subscription).where(Subscription.user_id == ctx.user.id).with_for_update()
    )).scalar_one_or_none()


async def _check_sub_id(ctx, sub_id):
    other = (await ctx.session.execute(
        select(Subscription.user_id).where(Subscription.stripe_subscription_id == sub_id,
                                           Subscription.user_id != ctx.user.id)
    )).scalar_one_or_none()
    if other is not None:
        raise _Conflict("That Stripe subscription id belongs to another account.")


def _watch_live(ctx, row, was_live):
    """After a real write that turns the plan live: the same follow-ups the
    Stripe webhook runs (unmark the free-plan clips, move queued jobs up)."""
    if ctx.apply and row is not None and metering._entitles(row):
        ctx.after_commit.append(lambda uid=ctx.user.id, plan=row.plan, became=not was_live:
                                _on_plan_live(uid, plan, became))


async def _on_plan_live(user_id, plan, became_live):
    try:
        from . import billing
    except Exception as e:  # pragma: no cover - billing deps missing
        print(f"⚠️  support-ops: plan follow-ups skipped: {e}")
        return
    if became_live:
        asyncio.create_task(billing._unmark_after_upgrade(user_id))
    await billing._reprioritize_waiting_jobs(user_id, plan)


async def _restore_subscription(ctx, target):
    row = await _subscription(ctx)
    was_live = metering._entitles(row)
    if target is None:
        if row is not None:
            await ctx.session.delete(row)
            await ctx.session.flush()
        return
    if not isinstance(target, dict) or set(target) != set(_SUB_FIELDS):
        raise _Bad(f"'subscription' must be null or an object with: {', '.join(_SUB_FIELDS)}.")
    sid = target["stripe_subscription_id"]
    if not isinstance(sid, str) or not sid:
        raise _Bad("'subscription.stripe_subscription_id' must be a string.")
    if target["plan"] not in PAID_PLANS:
        raise _Bad(f"'subscription.plan' must be one of {', '.join(PAID_PLANS)}.")
    if target["interval"] not in INTERVALS:
        raise _Bad("'subscription.interval' must be 'month' or 'year'.")
    if target["status"] not in SUB_STATUSES:
        raise _Bad(f"'subscription.status' must be one of {', '.join(SUB_STATUSES)}.")
    minutes = target["minutes_per_period"]
    if isinstance(minutes, bool) or not isinstance(minutes, int) or minutes < 0:
        raise _Bad("'subscription.minutes_per_period' must be a non-negative integer.")
    values = dict(
        stripe_subscription_id=sid, plan=target["plan"], interval=target["interval"],
        status=target["status"], minutes_per_period=minutes,
        current_period_start=_parse_instant(target["current_period_start"],
                                            "subscription.current_period_start"),
        current_period_end=_parse_instant(target["current_period_end"],
                                          "subscription.current_period_end"),
    )
    await _check_sub_id(ctx, sid)
    if row is None:
        row = Subscription(user_id=ctx.user.id, cancel_at_period_end=False, **values)
        ctx.session.add(row)
    else:
        for k, v in values.items():
            setattr(row, k, v)
    await ctx.session.flush()
    _watch_live(ctx, row, was_live)


_SUB_FIELDS = ("stripe_subscription_id", "plan", "interval", "status", "minutes_per_period",
               "current_period_start", "current_period_end")


# --- validators (pure: run before any database access) --------------------- #
def _v_set_plan(p):
    _only(p, ("plan", "until", "interval", "stripe_subscription_id"), ("plan",))
    if p["plan"] not in PLANS:
        raise _Bad(f"'plan' must be one of {', '.join(PLANS)}.")
    out = {"plan": p["plan"]}
    if p.get("interval") is not None:
        if p["interval"] not in INTERVALS:
            raise _Bad("'interval' must be 'month' or 'year'.")
        out["interval"] = p["interval"]
    if p.get("until") is not None:
        if p["plan"] == "free":
            raise _Bad("'until' does not apply to the free plan.")
        until = _parse_instant(p["until"], "until")
        if until <= _now():
            raise _Bad("'until' must be in the future.")
        out["until"] = until
    sid = p.get("stripe_subscription_id")
    if sid is not None:
        if not (isinstance(sid, str) and sid.startswith("sub_") and len(sid) <= 255):
            raise _Bad("'stripe_subscription_id' must look like 'sub_...'.")
        out["stripe_subscription_id"] = sid
    return out


def _v_add_credits(p):
    _only(p, ("amount",), ("amount",))
    amount = _number(p["amount"], "amount")
    if amount == 0:
        raise _Bad("'amount' must not be 0.")
    if abs(amount) > MAX_CREDIT_CHANGE:
        raise _Bad(f"'amount' must be within ±{MAX_CREDIT_CHANGE} minutes.")
    return {"amount": amount}


def _v_none(p):
    _only(p, ())
    return {}


def _v_change_email(p):
    _only(p, ("new_email",), ("new_email",))
    return {"new_email": _email(p["new_email"], "new_email")}


def _v_ban(p):
    _only(p, ("reason",), ("reason",))
    reason = p["reason"]
    if not isinstance(reason, str) or not reason.strip() or len(reason) > 500:
        raise _Bad("'reason' must be a non-empty string of up to 500 characters.")
    return {"reason": reason.strip()}


def _v_delete(p):
    _only(p, ("confirm_email",), ("confirm_email",))
    if not isinstance(p["confirm_email"], str) or not p["confirm_email"].strip():
        raise _Bad("'confirm_email' must be the account's email.")
    return {"confirm_email": p["confirm_email"]}


def _v_update_field(p):
    _only(p, ("field", "value"), ("field", "value"))
    if not isinstance(p["field"], str):
        raise _Bad("'field' must be a string.")
    return {"field": p["field"], "value": _check_field(p["field"], p["value"])}


_RESTORABLE = ("subscription", "credits", "usage", "email", "ban_reason", "banned_at") + tuple(
    FIELD_ALLOWLIST)


def _v_restore(p):
    _only(p, ("before",), ("before",))
    before = p["before"]
    if not isinstance(before, dict) or not before:
        raise _Bad("'before' must be the non-empty 'before' object of an earlier action.")
    unknown = set(before) - set(_RESTORABLE)
    if unknown:
        raise _Bad(f"'before' has keys no action writes: {', '.join(sorted(unknown))}.")
    if ("ban_reason" in before) != ("banned_at" in before):
        raise _Bad("'ban_reason' and 'banned_at' are restored together.")
    out = {}
    if "credits" in before:
        out["credits"] = _number(before["credits"], "before.credits")
    if "usage" in before:
        u = before["usage"]
        if not isinstance(u, dict) or set(u) != {"used", "period_end"}:
            raise _Bad("'before.usage' must be {used, period_end}.")
        out["usage"] = {"used": _number(u["used"], "before.usage.used"),
                        "period_end": u["period_end"]}
    if "email" in before:
        out["email"] = _email(before["email"], "before.email")
    if "ban_reason" in before:
        reason, at = before["ban_reason"], before["banned_at"]
        if reason is not None and (not isinstance(reason, str) or not reason.strip()):
            raise _Bad("'before.ban_reason' must be null or a non-empty string.")
        out["ban_reason"] = reason
        out["banned_at"] = None if at is None else _parse_instant(at, "before.banned_at")
    for name in FIELD_ALLOWLIST:
        if name in before:
            out[name] = _check_field(name, before[name])
    if "subscription" in before:
        out["subscription"] = before["subscription"]   # validated against the row in the action
    return out


# --- the actions ------------------------------------------------------------ #
async def _a_set_plan(ctx):
    p = ctx.params
    row = await _subscription(ctx)
    was_live = metering._entitles(row)
    before = {"subscription": _sub_snapshot(row)}
    plan, now = p["plan"], _now()
    if plan == "free":
        # App-side only: the row stops granting its plan; Stripe is untouched
        # and its next webhook for this subscription writes the row again.
        if was_live:
            row.status = "canceled"
    else:
        interval = p.get("interval") or (row.interval if row is not None else "month")
        if row is None:
            sid = p.get("stripe_subscription_id")
            if not sid:
                raise _Bad("This account has no subscription row: pass "
                           "'stripe_subscription_id' (the Stripe subscription it pays), "
                           "or use add_credits.")
            await _check_sub_id(ctx, sid)
            row = Subscription(
                user_id=ctx.user.id, stripe_subscription_id=sid, plan=plan, interval=interval,
                status="active", minutes_per_period=config.PLAN_MINUTES[plan],
                current_period_start=now,
                current_period_end=p.get("until") or _add_interval(now, interval),
                cancel_at_period_end=False)
            ctx.session.add(row)
        else:
            if p.get("stripe_subscription_id"):
                await _check_sub_id(ctx, p["stripe_subscription_id"])
                row.stripe_subscription_id = p["stripe_subscription_id"]
            row.plan = plan
            row.interval = interval
            row.minutes_per_period = config.PLAN_MINUTES[plan]
            if not was_live:
                row.status = "active"
                row.current_period_start = now
                row.current_period_end = p.get("until") or _add_interval(now, interval)
            elif p.get("until"):
                row.current_period_end = p["until"]
    await ctx.session.flush()
    _watch_live(ctx, row, was_live)
    return before, {"subscription": _sub_snapshot(row)}


async def _a_add_credits(ctx):
    before = await _credits(ctx.session, ctx.user.id)
    target = Decimal(str(before)) + ctx.params["amount"]
    if target < 0:
        raise _Bad(f"The account has {before} top-up minutes; removing "
                   f"{-ctx.params['amount']} would go below 0.")
    await _set_credits(ctx, target)
    return {"credits": before}, {"credits": await _credits(ctx.session, ctx.user.id)}


async def _a_reset_usage(ctx):
    before = await _usage_snapshot(ctx)
    if before["used"] != 0:
        await _set_usage(ctx, Decimal(0))
    return {"usage": before}, {"usage": await _usage_snapshot(ctx)}


async def _a_change_email(ctx):
    before = {"email": ctx.user.email}
    await _set_email(ctx, ctx.params["new_email"])
    return before, {"email": ctx.user.email}


async def _a_ban(ctx):
    before = _ban_snapshot(ctx.user)
    _set_ban(ctx, ctx.params["reason"], ctx.user.banned_at or _now())
    return before, _ban_snapshot(ctx.user)


async def _a_unban(ctx):
    before = _ban_snapshot(ctx.user)
    _set_ban(ctx, None, None)
    return before, _ban_snapshot(ctx.user)


async def _a_update_field(ctx):
    name = ctx.params["field"]
    before = {name: getattr(ctx.user, name)}
    setattr(ctx.user, name, ctx.params["value"])
    return before, {name: getattr(ctx.user, name)}


async def _a_restore(ctx):
    t = ctx.params
    before, after = {}, {}
    if "subscription" in t:
        before["subscription"] = _sub_snapshot(await _subscription(ctx))
        await _restore_subscription(ctx, t["subscription"])
        after["subscription"] = _sub_snapshot(await _subscription(ctx))
    if "credits" in t:
        before["credits"] = await _credits(ctx.session, ctx.user.id)
        await _set_credits(ctx, t["credits"])
        after["credits"] = await _credits(ctx.session, ctx.user.id)
    if "usage" in t:
        before["usage"] = await _usage_snapshot(ctx)
        await _set_usage(ctx, t["usage"]["used"], t["usage"]["period_end"])
        after["usage"] = await _usage_snapshot(ctx)
    if "email" in t:
        before["email"] = ctx.user.email
        await _set_email(ctx, t["email"])
        after["email"] = ctx.user.email
    if "ban_reason" in t:
        before.update(_ban_snapshot(ctx.user))
        _set_ban(ctx, t["ban_reason"], t["banned_at"])
        after.update(_ban_snapshot(ctx.user))
    for name in FIELD_ALLOWLIST:
        if name in t:
            before[name] = getattr(ctx.user, name)
            setattr(ctx.user, name, t[name])
            after[name] = getattr(ctx.user, name)
    return before, after


@dataclass(frozen=True)
class _Action:
    params: dict
    description: str
    validate: object
    run: object = None


ACTIONS = {
    "set_plan": _Action(
        {"plan": f"string: one of {', '.join(PLANS)}",
         "until": "string (ISO-8601), optional: end of the current period (paid plans)",
         "interval": "string, optional: 'month' or 'year' (default: the current one, else month)",
         "stripe_subscription_id": "string, optional: 'sub_...'; required when the account "
                                   "has no subscription row yet"},
        "App-side plan: rewrites the account's subscription row (plan, minutes per period "
        "from the plan catalog, status, period). Never calls Stripe, whose next webhook for "
        "that subscription overwrites the row. 'free' marks a live row canceled. Moving the "
        "period end starts a new usage period.",
        _v_set_plan, _a_set_plan),
    "add_credits": _Action(
        {"amount": "number: top-up minutes to add; negative removes (never below 0)"},
        "Top-up minutes (they do not expire while the plan is active). Positive adds a "
        "top-up row; negative consumes the oldest top-ups first.",
        _v_add_credits, _a_add_credits),
    "reset_usage": _Action(
        {}, "Plan minutes used in the current period back to 0 (one adjustment row in the "
            "minutes ledger; job history is untouched).",
        _v_none, _a_reset_usage),
    "change_email": _Action(
        {"new_email": "string: the new address (stored normalized, like at sign-up)"},
        "Sign-in email. 409 if another account has it. Magic links go to the new address; "
        "a linked Google account keeps signing in. Pending links to the old address are "
        "voided. Stripe's customer email is not touched.",
        _v_change_email, _a_change_email),
    "ban": _Action(
        {"reason": "string: why (kept on the account)"},
        "Blocks sign-in (magic link and Google), session tokens, API keys, OAuth tokens and "
        "Autopilot.", _v_ban, _a_ban),
    "unban": _Action({}, "Lifts a ban.", _v_none, _a_unban),
    "delete_account": _Action(
        {"confirm_email": "string: must equal the account's email"},
        "Erases the account through the same path as the user's own 'delete account' "
        "(storage purge, one transaction over the user's tables, erasure record), but "
        "without Stripe and without the goodbye email. 409 while a job is processing or "
        "while the subscription row is live: cancel it in Stripe first (or set_plan free). "
        "Cannot be restored; a dry run only checks.",
        _v_delete),
    "update_field": _Action(
        {"field": "string: one of " + ", ".join(FIELD_ALLOWLIST),
         "value": "the new value; " + "; ".join(f"{k}: {v}" for k, v in FIELD_ALLOWLIST.items())},
        "Writes one allowlisted users column.", _v_update_field, _a_update_field),
    "restore": _Action(
        {"before": "object: the 'before' of an earlier action (keys: "
                   + ", ".join(_RESTORABLE) + ")"},
        "Writes a 'before' back. Usage restores only within the same billing period.",
        _v_restore, _a_restore),
}


@router.get("/actions")
async def list_actions(request: Request):
    _authorize(request, write=False)
    return {"actions": [{"name": name, "params": a.params, "description": a.description}
                        for name, a in ACTIONS.items()]}


def _audit(**entry):
    print("support-ops audit " + json.dumps(entry, default=str, sort_keys=True))


async def _parse_body(request: Request) -> dict:
    raw = await request.body()
    if not raw.strip():
        return {}
    try:
        body = json.loads(raw)
    except ValueError:
        raise HTTPException(status_code=400, detail="The body must be JSON.")
    if not isinstance(body, dict):
        raise HTTPException(status_code=400, detail="The body must be a JSON object.")
    return body


def _text(value, name, limit=64) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (str, int)):
        raise HTTPException(status_code=400, detail=f"'{name}' must be a string.")
    text = str(value).strip()
    if len(text) > limit:
        raise HTTPException(status_code=400, detail=f"'{name}' is longer than {limit} characters.")
    return text or None


async def _delete_account(uid, params, dry_run, ticket, actor):
    from . import account
    async with database.session() as session:
        user = await session.get(User, uid)
        if user is None:
            raise HTTPException(status_code=404, detail="User not found.")
        if email_policy.normalize_email(params["confirm_email"]) != \
                email_policy.normalize_email(user.email):
            raise HTTPException(status_code=400, detail=(
                "'confirm_email' does not match the account's email."))
        sub = (await session.execute(
            select(Subscription).where(Subscription.user_id == uid))).scalar_one_or_none()
        in_flight = (await session.execute(
            select(func.count(UsageLedger.id)).where(
                UsageLedger.user_id == uid, UsageLedger.status == "reserved"))).scalar_one()
        before = {"account": {"id": str(uid), "email": user.email,
                              "stripe_customer_id": user.stripe_customer_id,
                              "subscription": _sub_snapshot(sub)}}
    if sub is not None and sub.status not in ("canceled", "incomplete_expired"):
        raise HTTPException(status_code=409, detail=(
            f"The subscription row is '{sub.status}'. Cancel the subscription in Stripe first "
            "(or set_plan free if Stripe already shows it cancelled), then delete."))
    if in_flight:
        raise HTTPException(status_code=409, detail="A job is still processing; try again later.")
    if dry_run:
        return before, {"account": None}
    _audit(phase="start", action="delete_account", ticket=ticket, actor=actor,
           user_id=str(uid), before=before)
    await account.erase_account(uid, reason=None, cancel_stripe=False, notify_user=False)
    return before, {"account": None}


@router.post("/users/{user_id}/actions/{action}")
async def run_action(user_id: str, action: str, request: Request):
    _authorize(request, write=True)
    spec = ACTIONS.get(action)
    if spec is None:
        raise HTTPException(status_code=404, detail=f"Unknown action '{action}'.")
    body = await _parse_body(request)
    unknown = set(body) - {"params", "dry_run", "ticket", "actor"}
    if unknown:
        raise HTTPException(status_code=400, detail=f"Unknown body keys: {', '.join(sorted(unknown))}.")
    dry_run = body.get("dry_run", True)
    if not isinstance(dry_run, bool):
        raise HTTPException(status_code=400, detail="'dry_run' must be true or false.")
    ticket = _text(body.get("ticket"), "ticket")
    actor = _text(body.get("actor"), "actor") or "support"
    if not dry_run and not ticket:
        raise HTTPException(status_code=400, detail="'ticket' is required for a real write.")
    params = body.get("params")
    if params is None:
        params = {}
    if not isinstance(params, dict):
        raise HTTPException(status_code=400, detail="'params' must be an object.")
    try:
        params = spec.validate(params)
    except (_Bad, _Conflict) as e:
        raise _http(e)
    uid = _parse_uuid(user_id)
    if uid is None:
        raise HTTPException(status_code=404, detail="User not found.")

    if action == "delete_account":
        before, after = await _delete_account(uid, params, dry_run, ticket, actor)
    else:
        ctx = None
        async with database.session() as session:
            tx = await session.begin()
            try:
                # The same row lock reserve_minutes and the billing webhook
                # take: nothing else moves this account's minutes meanwhile.
                user = (await session.execute(
                    select(User).where(User.id == uid).with_for_update()
                )).scalar_one_or_none()
                if user is None:
                    raise HTTPException(status_code=404, detail="User not found.")
                ctx = _Ctx(session=session, user=user, params=params, apply=not dry_run,
                           ticket=ticket)
                try:
                    before, after = await spec.run(ctx)
                except (_Bad, _Conflict) as e:
                    raise _http(e)
                if dry_run:
                    await tx.rollback()
                else:
                    await tx.commit()
            except BaseException:
                if tx.is_active:
                    await tx.rollback()
                raise
        for hook in (ctx.after_commit if ctx and not dry_run else ()):
            try:
                await hook()
            except Exception as e:
                print(f"⚠️  support-ops: follow-up after {action} failed: {e}")
    changed = before != after
    if not dry_run:
        _audit(action=action, ticket=ticket, actor=actor, user_id=str(uid), changed=changed,
               params=params, before=before, after=after)
    return {"action": action, "dry_run": dry_run, "changed": changed,
            "before": before, "after": after}
