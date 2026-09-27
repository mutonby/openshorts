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

import httpx
import stripe
from fastapi import APIRouter, Request, HTTPException, BackgroundTasks
from typing import Literal
from pydantic import BaseModel, Field, ConfigDict
from sqlalchemy import select

from .config import (settings, PLAN_MINUTES, TRIAL_DAYS, SUBSCRIPTION_LOOKUP_KEYS,
                     TOPUP_LOOKUP_KEYS, new_subscriber_label)
from . import analytics, config, database
from .models import User, Subscription, CreditTopup, StripeEvent, SignupAttribution, CancellationFeedback
from .auth import get_current_user_required

router = APIRouter()

_catalog = None


def _init_stripe():
    stripe.api_key = settings.stripe_secret_key


def _ts(unix) -> datetime:
    return datetime.fromtimestamp(unix, tz=timezone.utc)


def _money(cents, currency="usd") -> str:
    """'$59.00' from Stripe's integer-cents + currency."""
    sym = {"usd": "$", "eur": "€", "gbp": "£"}.get((currency or "usd").lower(), "")
    return f"{sym}{(cents or 0) / 100:,.2f} {(currency or 'usd').upper()}"


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


async def _blocking_sub_status(user_id):
    """Return the status of a subscription that must block a new checkout, else None."""
    async with database.session() as s:
        sub = (await s.execute(
            select(Subscription).where(Subscription.user_id == user_id)
        )).scalar_one_or_none()
    # Block a second checkout while a subscription is live or actively dunning:
    # past_due / unpaid / paused all have Stripe still retrying a charge, so a
    # fresh subscription would double-bill (real case: annual trial -> end-trial
    # -> card declined -> past_due, then a fresh monthly sub).
    #
    # 'incomplete' is deliberately NOT here. It means a checkout was started and
    # the payment never completed (3DS abandoned, card declined at confirm) —
    # nothing is being collected, and Stripe expires the row within 24h. Blocking
    # it only stops the retry, exactly when the user is most willing to pay.
    # Terminal states (canceled, incomplete_expired) do NOT block. Free-plan
    # users have no Subscription row at all, so their upgrade checkout passes.
    return sub.status if (sub and sub.status in config.CHECKOUT_BLOCKING_STATES) else None


# --------------------------------------------------------------------------- #
# Endpoints
# --------------------------------------------------------------------------- #
@router.get("/api/billing/plans")
async def list_plans():
    cat = await get_catalog()
    return {"plans": cat["plans"], "topups": cat["topups"]}


# Shown next to Stripe's terms checkbox. Two jobs in one tick: it makes the
# terms opposable to someone who signed up with a magic link and never opened
# them, and it is the express request for immediate performance that art. 16(m)
# of the consumer-rights directive wants before we start spending minutes
# inside the 14-day withdrawal window. Without it a withdrawal on day 13 is
# a full refund of a plan that was already used.
CONSENT_MESSAGE = (
    "I accept the Terms of Service and the Privacy Policy, and I ask OpenShorts "
    "to start the service immediately. EU consumers: you keep your 14-day right "
    "of withdrawal, but you accept that we may charge for the minutes already "
    "processed when you withdraw."
)


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
    if mode == "subscription":
        blocked = await _blocking_sub_status(user.id)
        if blocked in ("past_due", "unpaid"):
            # Their card failed. Telling them "you already have a plan" is both
            # false (they get no minutes) and a dead end — point at the portal.
            raise HTTPException(status_code=409, detail=(
                "Your last payment didn't go through, so we can't start a new plan "
                "yet. Update your card under Manage billing and it resumes right away."))
        if blocked:
            raise HTTPException(
                status_code=409,
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
        # Business customers need their legal name and VAT on the invoice to
        # deduct the tax, and this is the only place they can ever type them:
        # nothing in the dashboard asks for fiscal data. Without it every
        # company is invoiced as an anonymous e-mail address and writes to
        # support asking for a corrected invoice (ticket #4520), which cannot
        # be done in place — the invoice is already issued and its customer
        # data frozen, so it takes a rectificativa in the accounting.
        # ``customer_update.name`` is required for Checkout to save the name
        # onto an EXISTING customer (we always pass one); ``address`` saves the
        # billing address the same way. Not made ``required``: it is shown only
        # in supported countries and forcing it would add a field to a funnel
        # that already converts at ~1%.
        tax_id_collection={"enabled": True},
        customer_update={"name": "auto", "address": "auto"},
        metadata={
            "user_id": str(user.id),
            "kind": entry["kind"],
            "minutes": str(entry.get("minutes", "")),
        },
    )
    # Dead while TRIAL_DAYS == 0 (trials retired in favor of the free plan);
    # kept as the documented grandfathering mechanism.
    if mode == "subscription" and TRIAL_DAYS > 0:
        kwargs["subscription_data"] = {"trial_period_days": TRIAL_DAYS}
    consent = dict(
        consent_collection={"terms_of_service": "required"},
        custom_text={"terms_of_service_acceptance": {"message": CONSENT_MESSAGE}},
    )
    try:
        session = await asyncio.to_thread(
            lambda: stripe.checkout.Session.create(**kwargs, **consent))
    except Exception as exc:
        # The checkbox needs a Terms of service URL under Stripe > Settings >
        # Checkout; without it Stripe rejects the whole session. Losing the
        # consent record is bad, refusing the sale is worse, so fall back and
        # let it switch itself on the moment the URL is configured.
        #
        # Caught by message rather than by exception class on purpose: which
        # subclass Stripe raises is not worth betting the checkout button on.
        # Anything that is not about the terms URL re-raises untouched.
        # Match both prose and the parameter name: Stripe has phrased this as
        # "terms of service" and as `terms_of_service_url` depending on where
        # it fails, and a miss here means a 500 on the checkout button.
        msg = str(exc).lower()
        if "terms of service" not in msg and "terms_of_service" not in msg:
            raise
        print(f"[billing] checkout consent box disabled: {exc}", flush=True)
        session = await asyncio.to_thread(lambda: stripe.checkout.Session.create(**kwargs))
    return {"url": session.url}


@router.post("/api/billing/end-trial")
async def end_trial(request: Request):
    """End the free trial immediately: charge the card now and unlock full plan
    minutes. Kept for grandfathered trialing subscriptions — removable once
    ``SELECT count(*) FROM subscriptions WHERE status='trialing'`` reaches 0.
    Used when a trialing user hits the trial minute cap and chooses to
    activate their plan right away. The subscription webhook flips status→active
    (and thus the full ``minutes_per_period`` allowance) once Stripe confirms."""
    user = await get_current_user_required(request)
    async with database.session() as session:
        sub = (await session.execute(
            select(Subscription).where(Subscription.user_id == user.id)
        )).scalar_one_or_none()
    if not sub or sub.status != "trialing":
        raise HTTPException(status_code=409, detail="No active trial to convert.")
    try:
        updated = await asyncio.to_thread(lambda: stripe.Subscription.modify(
            sub.stripe_subscription_id, trial_end="now",
        ))
    except Exception:
        raise HTTPException(status_code=502, detail="Could not activate your plan. Try again.")
    return {"status": updated.get("status", "active")}


class CancellationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    reason: Literal["too_expensive", "not_using", "quality", "missing_features",
                    "technical_issues", "switching", "other"] | None = None
    comment: str | None = Field(default=None, max_length=2000)
    rating: int | None = Field(default=None, ge=1, le=5, strict=True)


@router.post("/api/billing/cancel")
async def cancellation_portal(body: CancellationRequest, request: Request, background_tasks: BackgroundTasks):
    """Optional private feedback; only Stripe webhooks establish billing state."""
    user = await get_current_user_required(request)
    async with database.session() as s:
        sub = (await s.execute(select(Subscription).where(
            Subscription.user_id == user.id))).scalar_one_or_none()
        owner = await s.get(User, user.id)
        customer_id = owner.stripe_customer_id if owner else None
    if not sub or sub.status in ("canceled", "incomplete_expired"):
        raise HTTPException(status_code=409, detail="No subscription to cancel. Use Manage billing.")
    if not customer_id:
        raise HTTPException(status_code=409, detail="No billing account. Use Manage billing.")
    _init_stripe()
    kwargs = dict(customer=customer_id,
                  return_url=f"{settings.frontend_url}/#/account")
    try:
        portal = await asyncio.to_thread(lambda: stripe.billing_portal.Session.create(
            **kwargs, flow_data={"type": "subscription_cancel", "subscription_cancel": {
                "subscription": sub.stripe_subscription_id}}))
    except Exception:
        # Some portal configurations cannot deep link: preserve the normal portal.
        try:
            portal = await asyncio.to_thread(
                lambda: stripe.billing_portal.Session.create(**kwargs))
        except Exception:
            raise HTTPException(status_code=502, detail="Could not open billing portal. Please try again.")
    try:
        async with database.session() as s:
            s.add(CancellationFeedback(
                user_id=user.id, stripe_subscription_id=sub.stripe_subscription_id,
                reason=body.reason, comment=(body.comment or "").strip() or None,
                rating=body.rating, status="intent"))
            await s.commit()
    except Exception:
        # SQL exceptions may contain private text. Never log them or block billing.
        pass
    comment = (body.comment or "").strip()
    if body.reason or comment or body.rating is not None:
        from .cancellation_notifications import notify_cancellation_feedback
        background_tasks.add_task(
            notify_cancellation_feedback, user_id=user.id,
            reason=body.reason or "Not provided", comment=comment,
            rating=body.rating, plan=getattr(sub, "plan", None),
        )
    return {"url": portal.url}


@router.post("/api/billing/portal")
async def create_portal(request: Request):
    user = await get_current_user_required(request)
    customer_id = await ensure_stripe_customer(user.id, user.email)
    session = await asyncio.to_thread(lambda: stripe.billing_portal.Session.create(
        customer=customer_id, return_url=f"{settings.frontend_url}/#/account",
    ))
    return {"url": session.url}


# --------------------------------------------------------------------------- #
# Legal invoices (AgentLedger)
# --------------------------------------------------------------------------- #
# Stripe stays the payment processor; the legally valid Spanish invoice is
# issued by AgentLedger (aikount.com) and surfaced here with signed public
# links, so the customer never needs the Stripe-hosted PDF. Same flow as
# Upload-Post: read by stripe_customer_id, and when AgentLedger has nothing yet
# (just-subscribed user) ask it to import this customer's history on the spot.
_AGENTLEDGER_TIMEOUT = 8.0
_AGENTLEDGER_BACKFILL_TIMEOUT = 20.0
_AGENTLEDGER_BACKFILL_DEBOUNCE_S = 90
_agentledger_backfill_at: dict[str, float] = {}   # stripe_customer_id -> unix ts


def _agentledger_headers() -> dict:
    return {"Authorization": f"Bearer {settings.agentledger_api_key}"}


async def _agentledger_fetch_invoices(client: httpx.AsyncClient, customer_id: str) -> list[dict]:
    """Invoices for one Stripe customer. 404 = no contact yet → []."""
    r = await client.get(f"{settings.agentledger_api_url}/contacts/{customer_id}/invoices",
                         headers=_agentledger_headers(), timeout=_AGENTLEDGER_TIMEOUT)
    if r.status_code == 404:
        return []
    if r.status_code != 200:
        raise HTTPException(status_code=502, detail="Could not load invoices")
    raw = (r.json() or {}).get("invoices") or []
    # Project to what the UI needs; tokens stay inside the pre-built URLs.
    return [{
        "doc_number": i.get("doc_number") or "",
        "doc_date": i.get("doc_date") or "",
        "total": i.get("total") or "0.00",
        "currency": i.get("currency") or "EUR",
        "status": i.get("status") or "",
        "public_url": i.get("public_url") or "",
        "pdf_url": i.get("pdf_url") or "",
    } for i in raw]


async def _agentledger_backfill(client: httpx.AsyncClient, customer_id: str) -> None:
    """Best-effort: import this customer's whole paid-invoice history now."""
    try:
        await client.post(f"{settings.agentledger_api_url}/integrations/stripe/sync-customer",
                          json={"treasury_id": settings.agentledger_treasury_id,
                                "stripe_customer_id": customer_id},
                          headers=_agentledger_headers(), timeout=_AGENTLEDGER_BACKFILL_TIMEOUT)
    except httpx.HTTPError:
        pass


@router.get("/api/billing/invoices")
async def list_invoices(request: Request):
    """Legal invoices for the signed-in user: ``{"invoices": [{doc_number,
    doc_date, total, currency, status, public_url, pdf_url}]}``. Free users
    (no Stripe customer yet) get an empty list, not an error."""
    user = await get_current_user_required(request)
    if not settings.agentledger_api_key:
        raise HTTPException(status_code=503, detail="Billing backend not configured")
    async with database.session() as session:
        u = await session.get(User, user.id)
        customer_id = u.stripe_customer_id if u else None
    if not customer_id:
        return {"invoices": []}

    async with httpx.AsyncClient() as client:
        try:
            invoices = await _agentledger_fetch_invoices(client, customer_id)
        except httpx.HTTPError:
            raise HTTPException(status_code=502, detail="Invoice service unreachable")
        if not invoices:
            import time as _time
            last = _agentledger_backfill_at.get(customer_id, 0.0)
            if _time.time() - last > _AGENTLEDGER_BACKFILL_DEBOUNCE_S:
                _agentledger_backfill_at[customer_id] = _time.time()
                await _agentledger_backfill(client, customer_id)
                try:
                    invoices = await _agentledger_fetch_invoices(client, customer_id)
                except httpx.HTTPError:
                    invoices = []
    return {"invoices": invoices}


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
        await _notify_invoice_paid(obj)
        await _track_invoice_revenue(obj)
    elif etype == "charge.refunded":
        await _track_refund(obj)


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
    buyer_email = None
    acquisition = {}
    async with database.session() as s:
        async with s.begin():
            existing = (await s.execute(
                select(CreditTopup).where(CreditTopup.stripe_session_id == session_id)
            )).scalar_one_or_none()
            if existing:
                return
            # The id rides in Stripe's metadata, so it can name an account that
            # no longer exists: a checkout completed moments before the user
            # erased themselves, or any webhook Stripe retries afterwards.
            # Inserting it blind trips the foreign key, and because the event is
            # only recorded after handle_event returns, the 500 makes Stripe
            # retry the same doomed insert for three days.
            user_id = (session_obj.get("metadata") or {}).get("user_id")
            if user_id:
                try:
                    user_id = (await s.execute(
                        select(User.id).where(User.id == user_id)
                    )).scalar_one_or_none()
                except Exception:
                    user_id = None  # not a uuid at all
            if not user_id:
                user_id = await _user_id_for_customer(s, session_obj.get("customer"))
            if not user_id:
                return
            s.add(CreditTopup(
                user_id=user_id, stripe_session_id=session_id,
                minutes_total=minutes, minutes_consumed=0,
            ))
            buyer_email = (await s.execute(
                select(User.email).where(User.id == user_id)
            )).scalar_one_or_none()
            acquisition = await _acquisition_properties(s, user_id)

    amount_txt = ""
    total = session_obj.get("amount_total")
    if total:
        amount_txt = f" ({_money(total, session_obj.get('currency'))})"
    from .alerts import send_admin_alert
    await send_admin_alert(
        f"💰 Top-up purchased{amount_txt}",
        f"{buyer_email or 'A user'} bought +{minutes} minutes.",
    )
    # Mirror the sale into OpenPanel, on the same profile the browser identifies.
    analytics.track_revenue(
        user_id, total, session_obj.get("currency"),
        type="topup", source="openshorts", minutes=minutes,
        stripe_session_id=session_id, email=buyer_email, **acquisition,
    )


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
            # Cross-subscription guard: the table keeps one row per user, so an
            # event for a *different* subscription of the same user must not clobber
            # the row that represents their live plan. This is exactly what happens
            # when a user has a dangling sub (e.g. a past_due annual that gets
            # canceled) alongside a live one (a trialing monthly): the canceled
            # sub's update would otherwise overwrite the good row. Only let another
            # subscription take over when the stored one is no longer live.
            if row and row.stripe_subscription_id != sub_obj["id"]:
                stored_live = row.status in ("active", "trialing", "past_due", "unpaid")
                incoming_live = sub_obj.get("status") in ("active", "trialing", "past_due")
                if stored_live and not incoming_live:
                    return
            start, end = _sub_period(sub_obj)
            now_canceling = bool(sub_obj.get("cancel_at_period_end"))
            # Detect the moment the user hits "cancel" (False -> True).
            was_canceling = bool(row.cancel_at_period_end) if row else False
            just_canceled = now_canceling and not was_canceling
            # Purchase signals: brand-new subscription, and trial -> paid conversion.
            is_new_sub = row is None
            prev_status = row.status if row else None
            buyer_email = (await s.execute(
                select(User.email).where(User.id == user_id)
            )).scalar_one_or_none()
            end_dt = _ts(end)
            values = dict(
                stripe_subscription_id=sub_obj["id"],
                stripe_price_id=price_id,
                plan=plan, interval=interval, status=sub_obj["status"],
                minutes_per_period=minutes,
                current_period_start=_ts(start),
                current_period_end=end_dt,
                cancel_at_period_end=now_canceling,
                last_event_at=event_created,
            )
            if row is None:
                s.add(Subscription(user_id=user_id, **values))
            else:
                for k, v in values.items():
                    setattr(row, k, v)

    # Purchase alert: someone just subscribed (trial started or paid outright).
    # Not for 'incomplete': Checkout creates the subscription the moment the
    # user hits pay, before 3DS / the card answer, so that status only says
    # someone reached the button. About half of them expire unpaid (24 of 53
    # in the 30 days to 16-sep-2026), and the ones that do pay are announced
    # by the 'Payment received' alert on invoice.paid, which carries the
    # amount; a second message here would just be noise.
    now_status = sub_obj["status"]
    label = new_subscriber_label(now_status) if is_new_sub else None
    if label:
        from .alerts import send_admin_alert
        await send_admin_alert(
            "🎉 New subscriber",
            f"{buyer_email or 'A user'} started the {plan} ({interval}) plan.\nStatus: {label}.",
        )
    elif prev_status == "trialing" and now_status == "active":
        from .alerts import send_admin_alert
        await send_admin_alert(
            "💳 Trial converted to paid",
            f"{buyer_email or 'A user'} is now a paying {plan} ({interval}) subscriber. 🎉",
        )

    # Churn alert to the admin the moment a subscription is set to cancel.
    if just_canceled:
        from .alerts import send_admin_alert
        from .config import VIDEO_RETENTION_GRACE_DAYS
        await send_admin_alert(
            "🔻 Subscription canceled",
            f"A {plan} subscriber just canceled.\n"
            f"Access continues until {end_dt:%Y-%m-%d}. Google-authed users then "
            f"drop to the free plan (clips expire after 7 days); others keep their "
            f"videos {VIDEO_RETENTION_GRACE_DAYS} more days before deletion.",
        )


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


async def _notify_invoice_paid(invoice_obj: dict):
    """Real money-in alert: every paid invoice — first charge AND every monthly
    renewal (renewals were previously silent). Includes the amount."""
    amount = invoice_obj.get("amount_paid", 0)
    if amount <= 0:                       # $0 invoices (e.g. legacy trial start)
        return
    money = _money(amount, invoice_obj.get("currency"))
    reason = invoice_obj.get("billing_reason") or ""
    kind = {
        "subscription_create": "new subscription",
        "subscription_cycle": "renewal",
        "subscription_update": "plan change",
    }.get(reason, reason or "payment")

    email = None
    async with database.session() as s:
        cust = invoice_obj.get("customer")
        if cust:
            email = (await s.execute(
                select(User.email).where(User.stripe_customer_id == cust)
            )).scalar_one_or_none()

    from .alerts import send_admin_alert
    await send_admin_alert(
        f"💵 Payment received: {money}",
        f"{email or 'A customer'} — {kind}.",
    )


# --------------------------------------------------------------------------- #
# OpenPanel revenue mirror
# --------------------------------------------------------------------------- #
# The browser fires ``Subscribed`` with the list price when the user comes back
# from Stripe, but that is an intent signal: it misses renewals entirely, is
# eaten by ad-blockers and can never see a refund. The webhook is the only
# place money is known for certain, so every paid invoice, top-up and refund is
# mirrored here as OpenPanel's native ``revenue`` event. ``profileId`` is the
# user's uuid — the same id the browser and the job events already use — so
# the sale lands on the profile that carries their pageviews and ``Signup``.

_INVOICE_REVENUE_TYPE = {
    "subscription_create": "new",
    "subscription_cycle": "renewal",
    "subscription_update": "upgrade",
    "subscription_threshold": "renewal",
}


async def _acquisition_properties(session, user_id) -> dict:
    """Never raises: a missing row simply reads as ``channel=direct``."""
    try:
        row = await session.get(SignupAttribution, user_id)
    except Exception:
        row = None
    return analytics.acquisition_properties(row)


async def _track_invoice_revenue(invoice_obj: dict):
    try:
        amount = int(invoice_obj.get("amount_paid") or 0)
    except (TypeError, ValueError):
        return
    if amount <= 0:
        return
    cust = invoice_obj.get("customer")
    if not cust:
        return
    async with database.session() as s:
        user = (await s.execute(
            select(User.id, User.email).where(User.stripe_customer_id == cust)
        )).first()
        if user is None:
            return
        user_id, email = user
        plan = interval = None
        sub_id = invoice_obj.get("subscription")
        if sub_id:
            sub = (await s.execute(
                select(Subscription.plan, Subscription.interval).where(
                    Subscription.stripe_subscription_id == sub_id
                )
            )).first()
            if sub is not None:
                plan, interval = sub
        acquisition = await _acquisition_properties(s, user_id)
    reason = invoice_obj.get("billing_reason") or ""
    analytics.track_revenue(
        user_id, amount, invoice_obj.get("currency"),
        type=_INVOICE_REVENUE_TYPE.get(reason, "other"), source="openshorts",
        plan=plan, interval=interval, billing_reason=reason or None,
        stripe_invoice_id=invoice_obj.get("id"), email=email, **acquisition,
    )


async def _track_refund(charge_obj: dict):
    """Negative revenue. ``amount_refunded`` is cumulative across partial
    refunds, so prefer the newest entry of ``refunds.data`` when Stripe
    includes it (it is not expanded by default on recent API versions)."""
    refunds = ((charge_obj.get("refunds") or {}).get("data") or [])
    raw = refunds[0].get("amount") if refunds else charge_obj.get("amount_refunded")
    try:
        amount = int(raw or 0)
    except (TypeError, ValueError):
        return
    if amount <= 0:
        return
    cust = charge_obj.get("customer")
    if not cust:
        return
    async with database.session() as s:
        user = (await s.execute(
            select(User.id, User.email).where(User.stripe_customer_id == cust)
        )).first()
        if user is None:
            return
        user_id, email = user
        acquisition = await _acquisition_properties(s, user_id)
    analytics.track_revenue(
        user_id, -amount, charge_obj.get("currency"),
        type="refund", source="openshorts",
        stripe_charge_id=charge_obj.get("id"), email=email, **acquisition,
    )
