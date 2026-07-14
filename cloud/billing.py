"""Stripe billing: catalog, checkout, customer portal and webhooks.

Design (matches the Stripe implementation planner for this use case):
- Stripe-hosted Checkout: subscription mode for plans, payment mode for top-ups.
- Stripe-hosted Customer Portal for self-service plan management.
- Prices resolved by stable lookup_key at runtime (no price IDs in env).
- Webhooks are idempotent (stripe_events dedupe) and order-safe (last_event_at);
  every handler upserts the full object rather than applying deltas.
"""
import asyncio
from datetime import datetime, timezone

import stripe
from fastapi import APIRouter, Request, HTTPException
from pydantic import BaseModel
from sqlalchemy import select

from .config import settings, PLAN_MINUTES, TRIAL_DAYS, SUBSCRIPTION_LOOKUP_KEYS, TOPUP_LOOKUP_KEYS
from . import database
from .models import User, Subscription, CreditTopup, StripeEvent
from .auth import get_current_user_required

router = APIRouter()

_catalog = None


def _init_stripe():
    stripe.api_key = settings.stripe_secret_key


def _ts(unix) -> datetime:
    return datetime.fromtimestamp(unix, tz=timezone.utc)


# --------------------------------------------------------------------------- #
# Catalog (resolved once by lookup_key)
# --------------------------------------------------------------------------- #
async def get_catalog(force=False):
    global _catalog
    if _catalog is not None and not force:
        return _catalog
    _init_stripe()
    prices = await asyncio.to_thread(
        lambda: stripe.Price.list(
            lookup_keys=SUBSCRIPTION_LOOKUP_KEYS + TOPUP_LOOKUP_KEYS,
            expand=["data.product"], limit=100, active=True,
        )
    )
    plans, topups, by_id = [], [], {}
    for p in prices.data:
        md = p.metadata or {}
        if p.recurring:
            plan = md.get("plan")
            interval = md.get("interval") or p.recurring.get("interval")
            if plan not in PLAN_MINUTES:
                continue
            entry = {
                "kind": "subscription", "price_id": p.id, "lookup_key": p.lookup_key,
                "plan": plan, "interval": interval, "minutes": PLAN_MINUTES[plan],
                "amount": p.unit_amount, "currency": p.currency,
            }
            plans.append(entry)
        else:
            minutes = int(md.get("topup_minutes", 0))
            entry = {
                "kind": "topup", "price_id": p.id, "lookup_key": p.lookup_key,
                "minutes": minutes, "amount": p.unit_amount, "currency": p.currency,
            }
            topups.append(entry)
        by_id[p.id] = entry
    _catalog = {"plans": plans, "topups": topups, "by_id": by_id}
    return _catalog


def plan_info_for_price(price_id: str):
    """Sync lookup into the cached catalog (webhook path). Returns entry or None."""
    if _catalog is None:
        return None
    return _catalog["by_id"].get(price_id)


# --------------------------------------------------------------------------- #
# Customer helper
# --------------------------------------------------------------------------- #
async def ensure_stripe_customer(user_id, email) -> str:
    async with database.session() as s:
        async with s.begin():
            u = await s.get(User, user_id)
            if u.stripe_customer_id:
                return u.stripe_customer_id
            cust = await asyncio.to_thread(
                lambda: stripe.Customer.create(email=email, metadata={"user_id": str(user_id)})
            )
            u.stripe_customer_id = cust.id
            return cust.id


async def _active_sub_exists(user_id) -> bool:
    async with database.session() as s:
        sub = (await s.execute(
            select(Subscription).where(Subscription.user_id == user_id)
        )).scalar_one_or_none()
    return bool(sub and sub.status in ("active", "trialing"))


# --------------------------------------------------------------------------- #
# Endpoints
# --------------------------------------------------------------------------- #
@router.get("/api/billing/plans")
async def list_plans():
    cat = await get_catalog()
    return {"plans": cat["plans"], "topups": cat["topups"]}


class CheckoutRequest(BaseModel):
    price_id: str


@router.post("/api/billing/checkout")
async def create_checkout(body: CheckoutRequest, request: Request):
    user = await get_current_user_required(request)
    cat = await get_catalog()
    entry = cat["by_id"].get(body.price_id)
    if not entry:
        raise HTTPException(status_code=400, detail="Unknown price")

    mode = "subscription" if entry["kind"] == "subscription" else "payment"
    if mode == "subscription" and await _active_sub_exists(user.id):
        raise HTTPException(status_code=409,
                            detail="You already have an active plan. Manage it from your account.")

    customer_id = await ensure_stripe_customer(user.id, user.email)
    fe = settings.frontend_url
    kwargs = dict(
        mode=mode,
        customer=customer_id,
        line_items=[{"price": body.price_id, "quantity": 1}],
        client_reference_id=str(user.id),
        success_url=f"{fe}/#/account?checkout=success",
        cancel_url=f"{fe}/#/pricing?checkout=cancel",
        allow_promotion_codes=True,
        metadata={
            "user_id": str(user.id),
            "kind": entry["kind"],
            "minutes": str(entry.get("minutes", "")),
        },
    )
    # Subscriptions start with a card-required free trial that auto-charges.
    if mode == "subscription" and TRIAL_DAYS > 0:
        kwargs["subscription_data"] = {"trial_period_days": TRIAL_DAYS}
    session = await asyncio.to_thread(lambda: stripe.checkout.Session.create(**kwargs))
    return {"url": session.url}


@router.post("/api/billing/portal")
async def create_portal(request: Request):
    user = await get_current_user_required(request)
    customer_id = await ensure_stripe_customer(user.id, user.email)
    session = await asyncio.to_thread(lambda: stripe.billing_portal.Session.create(
        customer=customer_id, return_url=f"{settings.frontend_url}/#/account",
    ))
    return {"url": session.url}


# --------------------------------------------------------------------------- #
# Webhook
# --------------------------------------------------------------------------- #
@router.post("/api/stripe/webhook")
async def stripe_webhook(request: Request):
    payload = await request.body()
    sig = request.headers.get("stripe-signature", "")
    try:
        event = stripe.Webhook.construct_event(payload, sig, settings.stripe_webhook_secret)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid signature")

    # Dedupe: if we've already recorded this event, ack and skip.
    async with database.session() as s:
        if await s.get(StripeEvent, event["id"]):
            return {"ok": True, "dedup": True}

    await handle_event(event)

    async with database.session() as s:
        async with s.begin():
            from sqlalchemy.dialects.postgresql import insert as pg_insert
            await s.execute(pg_insert(StripeEvent).values(
                id=event["id"], type=event["type"], created=_ts(event["created"]),
            ).on_conflict_do_nothing(index_elements=["id"]))
    return {"ok": True}


async def handle_event(event: dict):
    """Idempotent event dispatch. Public so tests can call it directly."""
    etype = event["type"]
    obj = event["data"]["object"]
    created = _ts(event["created"])

    if etype == "checkout.session.completed" and obj.get("mode") == "payment":
        await _apply_topup(obj)
    elif etype in ("customer.subscription.created", "customer.subscription.updated"):
        await _upsert_subscription(obj, created)
    elif etype == "customer.subscription.deleted":
        await _set_subscription_status(obj, "canceled", created)
    elif etype == "invoice.payment_failed":
        await _set_subscription_status_by_invoice(obj, "past_due", created)
    elif etype == "invoice.paid":
        await _set_subscription_status_by_invoice(obj, "active", created)


async def _user_id_for_customer(session, customer_id):
    return (await session.execute(
        select(User.id).where(User.stripe_customer_id == customer_id)
    )).scalar_one_or_none()


async def _apply_topup(session_obj: dict):
    """Credit a top-up from a completed one-off Checkout (idempotent by session id)."""
    session_id = session_obj["id"]
    minutes = int((session_obj.get("metadata") or {}).get("minutes") or 0)
    if minutes <= 0:
        return
    async with database.session() as s:
        async with s.begin():
            existing = (await s.execute(
                select(CreditTopup).where(CreditTopup.stripe_session_id == session_id)
            )).scalar_one_or_none()
            if existing:
                return
            user_id = (session_obj.get("metadata") or {}).get("user_id")
            if not user_id:
                user_id = await _user_id_for_customer(s, session_obj.get("customer"))
            if not user_id:
                return
            s.add(CreditTopup(
                user_id=user_id, stripe_session_id=session_id,
                minutes_total=minutes, minutes_consumed=0,
            ))


def _sub_item(sub_obj: dict):
    try:
        return sub_obj["items"]["data"][0]
    except Exception:
        return {}


def _sub_price_id(sub_obj: dict):
    try:
        return _sub_item(sub_obj)["price"]["id"]
    except Exception:
        return None


def _sub_period(sub_obj: dict):
    """Return (start, end) unix timestamps, from the subscription or its item.

    Newer Stripe API versions expose current_period_* on the item, older ones on
    the subscription itself — support both.
    """
    item = _sub_item(sub_obj)
    start = sub_obj.get("current_period_start") or item.get("current_period_start")
    end = sub_obj.get("current_period_end") or item.get("current_period_end")
    return start, end


async def _upsert_subscription(sub_obj: dict, event_created: datetime):
    price_id = _sub_price_id(sub_obj)
    info = plan_info_for_price(price_id)
    if info is None:
        await get_catalog(force=True)
        info = plan_info_for_price(price_id)
    if info is None:
        return  # unknown price — ignore
    plan = info["plan"]
    interval = info["interval"]
    minutes = PLAN_MINUTES[plan]

    async with database.session() as s:
        async with s.begin():
            user_id = await _user_id_for_customer(s, sub_obj.get("customer"))
            if not user_id:
                return
            row = (await s.execute(
                select(Subscription).where(Subscription.user_id == user_id).with_for_update()
            )).scalar_one_or_none()
            # Order guard: ignore events older than what we've already applied.
            if row and row.last_event_at and event_created < row.last_event_at:
                return
            start, end = _sub_period(sub_obj)
            values = dict(
                stripe_subscription_id=sub_obj["id"],
                stripe_price_id=price_id,
                plan=plan, interval=interval, status=sub_obj["status"],
                minutes_per_period=minutes,
                current_period_start=_ts(start),
                current_period_end=_ts(end),
                cancel_at_period_end=bool(sub_obj.get("cancel_at_period_end")),
                last_event_at=event_created,
            )
            if row is None:
                s.add(Subscription(user_id=user_id, **values))
            else:
                for k, v in values.items():
                    setattr(row, k, v)


async def _set_subscription_status(sub_obj: dict, status: str, event_created: datetime):
    async with database.session() as s:
        async with s.begin():
            row = (await s.execute(
                select(Subscription).where(
                    Subscription.stripe_subscription_id == sub_obj["id"]
                ).with_for_update()
            )).scalar_one_or_none()
            if row is None:
                return
            if row.last_event_at and event_created < row.last_event_at:
                return
            row.status = status
            row.last_event_at = event_created


async def _set_subscription_status_by_invoice(invoice_obj: dict, status: str, event_created: datetime):
    sub_id = invoice_obj.get("subscription")
    if not sub_id:
        return
    await _set_subscription_status({"id": sub_id}, status, event_created)
