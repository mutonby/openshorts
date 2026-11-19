"""Daily ops digest to Telegram: one message per day with the pulse of the
product (signups, processing activity, clips, revenue events, the queue, paid
proxy use) and every alert that was not urgent enough to go out on its own
(cloud/alerts.hold), counted by kind. It replaces the per-event pings that
turned into spam as usage grew.

The figures are computed from the DB at send time and the held alerts live in
a file of the shared output dir, so redeploys lose no data. During a deploy
handover only the active instance sends (``is_active``).
The send is scheduled by wall clock (sleep until the next 06:00 UTC): a deploy
in the middle of the day just re-arms the same next send, so there is no
duplicate-send window to guard with state. The one accepted loss: a deploy
that lands exactly across the send minute skips that day's message (the data
still shows up in DB queries; only the ping is lost). Transient failures at
send time are retried in-process rather than skipped.
"""
import asyncio
from datetime import datetime, timedelta, timezone

from sqlalchemy import distinct, func, select

from . import database
from .models import ProxyUsage, StripeEvent, UsageLedger, User, UserVideo


def _send_hour_utc() -> int:
    """DIGEST_HOUR_UTC (default 7: 08:00 in Madrid in winter, 09:00 in summer)."""
    import os
    try:
        return min(23, max(0, int(os.environ.get("DIGEST_HOUR_UTC", "7"))))
    except ValueError:
        return 7

# Stripe event types worth a line in the digest, with a human label.
# invoice.payment_succeeded is deliberately absent: it is a near-alias of
# invoice.paid and counting both would double-report every payment.
_STRIPE_LABELS = {
    "checkout.session.completed": "checkout completed",
    "invoice.paid": "payment",
    "invoice.payment_failed": "PAYMENT FAILED",
    "customer.subscription.created": "new subscription",
    "customer.subscription.deleted": "subscription ended",
}


async def _window_stats(start, end):
    async with database.session() as s:
        signups = (await s.execute(
            select(func.count(User.id))
            .where(User.created_at >= start, User.created_at < end)
        )).scalar_one()

        jobs, processors, refunded, minutes = (await s.execute(
            select(
                func.count(UsageLedger.id),
                func.count(distinct(UsageLedger.user_id)),
                func.count(UsageLedger.id).filter(UsageLedger.status == "released"),
                func.coalesce(func.sum(UsageLedger.minutes), 0),
            ).where(UsageLedger.job_type == "process",
                    UsageLedger.created_at >= start,
                    UsageLedger.created_at < end)
        )).one()

        clips = (await s.execute(
            select(func.count(UserVideo.id))
            .where(UserVideo.created_at >= start, UserVideo.created_at < end)
        )).scalar_one()

        # processed_at (server-side), NOT StripeEvent.created: Stripe keeps the
        # original event timestamp across delivery retries, so an event
        # delivered late after an outage would land inside an already-digested
        # window and never be reported. New subscriptions are counted from the
        # customer.subscription.created event rather than Subscription rows:
        # the DB upserts re-subscribing churned users into their existing row,
        # whose created_at never refreshes.
        stripe_rows = (await s.execute(
            select(StripeEvent.type, func.count(StripeEvent.id))
            .where(StripeEvent.processed_at >= start,
                   StripeEvent.processed_at < end)
            .group_by(StripeEvent.type)
        )).all()

        paid_proxy = (await s.execute(
            select(func.coalesce(func.sum(ProxyUsage.paid_bytes), 0))
            .where(ProxyUsage.created_at >= start, ProxyUsage.created_at < end)
        )).scalar_one()

    return {
        "signups": signups,
        "jobs": jobs,
        "processors": processors,
        "refunded": refunded,
        "minutes": float(minutes or 0),
        "clips": clips,
        "stripe": {t: n for t, n in stripe_rows if t in _STRIPE_LABELS},
        "paid_proxy_mb": int(paid_proxy or 0) / 1e6,
    }


def _format(stats, held=(), queue=()) -> str:
    lines = [
        "📊 Resumen del día (últimas 24 h)",
        f"Altas: {stats['signups']}",
        f"Procesado: {stats['processors']} usuario(s), {stats['jobs']} vídeo(s), "
        f"{stats['minutes']:.0f} min"
        + (f"; {stats['refunded']} devueltos (fallo o caducados)" if stats['refunded'] else ""),
        f"Clips creados: {stats['clips']}",
    ]
    for etype, n in sorted(stats["stripe"].items()):
        lines.append(f"Stripe: {n}× {_STRIPE_LABELS[etype]}")
    if stats.get("paid_proxy_mb"):
        lines.append(f"Proxy de pago: {stats['paid_proxy_mb']:.0f} MB")
    if stats["signups"] == 0 and stats["jobs"] == 0 and stats["clips"] == 0:
        lines.append("(día tranquilo)")
    if queue:
        lines += ["", "Cola ahora:"] + list(queue)
    if held:
        lines += ["", "Avisos del día (sin urgencia):"] + list(held)
    return "\n".join(lines)


def _queue_lines(now: float) -> list:
    """The queue watch's last measurement, if it is recent (this instance
    measures: it is the active one)."""
    try:
        from . import queue_watch
        if now - queue_watch.latest_figures_at[0] <= 900:
            return list(queue_watch.latest_figures)
    except Exception:
        pass
    return []


async def send_daily_digest():
    from . import alerts
    end = datetime.now(timezone.utc)
    start = end - timedelta(hours=24)
    stats = await _window_stats(start, end)
    taken = alerts.take_held(end.timestamp())
    try:
        await alerts.send_telegram(
            _format(stats, alerts.summarize_held(taken), _queue_lines(end.timestamp())),
            raise_errors=True)
    except Exception:
        alerts.give_back(taken)
        raise
    print(f"📊 Daily digest sent ({len(taken)} held alert(s)).")


_SEND_RETRIES = 6           # transient DB/Telegram trouble at send time...
_RETRY_DELAY_SECONDS = 300  # ...retries for up to 30 min instead of skipping


async def _digest_loop(is_active=None):
    while True:
        now = datetime.now(timezone.utc)
        target = now.replace(hour=_send_hour_utc(), minute=0, second=0,
                             microsecond=0)
        if target <= now:
            target += timedelta(days=1)
        await asyncio.sleep((target - now).total_seconds())
        # Retry in place rather than falling through to tomorrow's target —
        # otherwise one DB blip at the send hour silently swallows the whole day.
        for attempt in range(_SEND_RETRIES):
            if is_active is not None and not is_active():
                break  # a draining container: the new one sends it
            try:
                await send_daily_digest()
                break
            except asyncio.CancelledError:
                return
            except Exception as e:
                print(f"⚠️  Daily digest error (attempt {attempt + 1}): {e}")
                await asyncio.sleep(_RETRY_DELAY_SECONDS)


_task = None  # keep a strong reference: the loop only holds weak task refs


def start_digest(is_active=None):
    global _task
    _task = asyncio.create_task(_digest_loop(is_active))
