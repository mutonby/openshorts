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
import contextlib
import os
from urllib.parse import urlparse
import random
import time
import subprocess

import yt_account
from datetime import datetime, timezone, timedelta
from decimal import Decimal

from sqlalchemy import select, update, func, and_

from . import config, database
from .models import User, Subscription, CreditTopup, UsageLedger, FirstVideoGrant


SWEEP_INTERVAL_SECONDS = 15 * 60
STUCK_RESERVATION_HOURS = 3
# A reservation whose job is still queued or running outlives the stuck window;
# one whose job has not MOVED for this long (env LIVE_RESERVATION_MAX_HOURS: a
# run that never ends, a manifest nobody resumes) is released as hung.
LIVE_RESERVATION_MAX_HOURS = 24


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


# Errors on a free (static) route that a different IP could fix. Only these
# justify spending the per-GB proxy on the same URL; a private, removed or
# members-only video fails the same on every IP, and a live stream has no
# duration anywhere. Before this list every one of those went to the paid
# proxy twice (both extractors), ~1.7 MB a time — the "1.76 MB, 2 requests"
# rows that filled the DataImpulse panel on 31-aug-2026.
_IP_SPECIFIC_HINTS = (
    "sign in to confirm you", "not a bot", "http error 403", "http error 429",
    "http error 407", "http error 502", "http error 503", "proxyerror",
    "tunnel connection failed", "connection reset", "timed out", "timeout",
    "unable to download webpage", "unable to download api page",
    # "Video unavailable" looks like a content error but is NOT reliable on
    # the static pool: on 1-sep-2026 five videos "unavailable" on all three
    # Decodo IPs downloaded fine through the residential proxy (proxy_usage
    # rows 19:33-07:07). YouTube serves a fake unavailable to IPs it dislikes
    # — the same symptom as the 19-aug server-IP ban. A truly dead link costs
    # ~3 MB to re-check; a real video wrongly refused costs the job.
    "video unavailable",
    "remote end closed", "connection refused", "network is unreachable",
    "name or service not known", "eof occurred",
)
_CONTENT_HINTS = (
    "private video", "has been removed", "members-only",
    "join this channel", "not available on this app", "is not a valid url",
    "unsupported url", "no video formats found", "premieres in", "will begin in",
    "this live event", "no duration in metadata", "requested format is not available",
    "account has been terminated", "video is age", "confirm your age",
    # An uploader's country block: the residential pool exits from the same
    # blocked countries, so the paid probe failed with the identical error in
    # 5 of the 6 cases between 3-sep and 5-sep-2026 (3.6 MB each, and the
    # Telegram alert claimed the paid proxy "answered").
    "available in your country", "in your country", "geo-restricted", "geoblock",
    "blocked it in your country",
)


def static_failure_warrants_paid(err) -> bool:
    """Does this failure on a free route justify retrying through the paid proxy?"""
    e = str(err or "").lower()
    if any(h in e for h in _CONTENT_HINTS):
        return False
    return any(h in e for h in _IP_SPECIFIC_HINTS)


# Paid-probe events queued for app.py (thread-safe enough: appended from the
# executor thread that runs the probe, drained on the event loop).
_paid_probe_events: list = []

# URLs whose probe saw EVERY static IP answer the bot-check and the paid proxy
# answer the video. Since 30-sep-2026 that verdict is per video and identical
# on every one of our IPs (measured: 9 of 10 videos, 3 statics + the server's
# own IP, twice each), so the download repeating the four anonymous attempts
# costs ~8 s and four more hits on IPs YouTube is already scoring, for the
# same answer. app.py pops the verdict when it builds the job.
_static_bot_verdicts: set = set()


def pop_statics_bot_checked(url) -> bool:
    """True (once) when the probe of ``url`` found the static pool bot-checked
    for this video and had to use the paid proxy."""
    try:
        _static_bot_verdicts.remove(url)
        return True
    except KeyError:
        return False


def _all_bot_checked(static_errors) -> bool:
    return bool(static_errors) and all(
        "not a bot" in (e or "") for e in static_errors.values())


def pop_paid_probe_events() -> list:
    out, _paid_probe_events[:] = list(_paid_probe_events), []
    return out


def plan_probe_proxies(direct_first, statics, paid):
    """Ordered proxies for a metadata probe — pure, unit-tested.

    Same cheapest-first order as the download plan (``main.plan_download_attempts``):
    the server's own IP when the operator allows it, then the flat-rate static ISP
    pool, then the per-GB paid proxy. ``None`` means "no proxy" (direct).

    This probe used to read ``PROXY_URL`` only, which it kept doing after the
    static pool landed (dfa3124, 19-aug-2026) because that commit never touched
    this file. Every managed YouTube submission then paid the per-GB proxy for
    ~0.2-1 MB of metadata while the download itself went out free over the
    statics — and invisibly, since ``PROXY_BYTES`` (main.py) only counts the
    bytes of the winning *download* attempt.
    """
    order = []
    if direct_first or not (statics or paid):
        order.append(None)
    order.extend(statics)
    if paid:
        order.append(paid)
    return order


def is_youtube_url(url: str) -> bool:
    host = (urlparse(url or "").hostname or "").lower()
    return host in ("youtube.com", "www.youtube.com", "m.youtube.com", "youtu.be",
                    "music.youtube.com") or host.endswith(".youtube.com")


def probe_url_minutes(url: str, allow_paid: bool = True) -> float:
    """Return the video duration in minutes from yt-dlp metadata (no download).

    Uses the same proxy order + extractor settings as the actual download
    (main.py) so the probe behaves consistently with it and bills the same way.
    Raises ValueError if the duration is unknown (e.g. live streams).
    """
    # SSRF guard: reject non-http(s) / private / metadata hosts before probing.
    from security_utils import assert_public_url
    assert_public_url(url)
    # Throwaway hosts (tmpfiles.org) sign links with a short-lived stamp; take
    # the live one so the probe sees the file and not an HTML page.
    import file_hosts
    url = file_hosts.resolve(url)
    # Free and first: a search / playlist / channel page is walked entry by
    # entry by yt-dlp and never yields a duration (yt_clients.youtube_non_video_reason).
    from yt_clients import NotASingleVideo, youtube_non_video_reason
    reason = youtube_non_video_reason(url)
    if reason:
        raise NotASingleVideo(f"This link is {reason}.")

    bgutil_http = os.environ.get("BGUTIL_BASE_URL", "").strip()
    bgutil_script = os.environ.get("BGUTIL_SCRIPT_PATH", "").strip()
    # Same client lists as the download (yt_clients.py): the authed defaults
    # alone answer "Video unavailable" for a share of videos on every IP, and
    # each of those probes then paid the per-GB proxy for nothing.
    from yt_clients import hd_extractor_args, fallback_extractor_args
    hd_args = hd_extractor_args(bgutil_http, bgutil_script)
    # (extractor args, send the account cookies). The second attempt on each
    # route drops the cookies, exactly like the download's 'fallback-static'
    # step (main.py): with the cookies attached YouTube answers UNPLAYABLE for
    # every client — web_embedded, tv_downgraded, web AND mweb — on a share of
    # videos, and yt-dlp reports that as "Video unavailable" (measured in prod
    # 9-sep-2026, all three statics, same video anonymous → 1080p 137+140).
    # Without this step the probe read that as an IP problem and escalated to
    # the per-GB proxy, which carries the same cookies and fails identically,
    # while the download quietly recovered anonymously on the same static.
    # With no HD path at all (self-host, no PO token provider) the fallback is
    # the only attempt, so it keeps the cookies the operator configured.
    strategies = ([(hd_args, True)] if hd_args else []) + [
        (fallback_extractor_args(bgutil_http, bgutil_script), not hd_args)]

    # Rotated per probe to spread load across the pool, like the download does.
    statics = [p.strip() for p in
               os.environ.get("STATIC_PROXY_URLS", "").split(",") if p.strip()]
    if statics:
        k = random.randrange(len(statics))
        statics = statics[k:] + statics[:k]
    paid = os.environ.get("PROXY_URL", "").strip()
    if not allow_paid:
        paid = ""  # daily budget hit (cloud/proxy_ledger.budget_exceeded)
    # Same rule as the download plan: a non-YouTube URL never touches the
    # per-GB proxy (main.plan_download_attempts, youtube=False). Twitch, Kick,
    # Rumble, product pages and drive links were all reaching it here.
    if not is_youtube_url(url):
        paid = ""
    proxies = plan_probe_proxies(
        os.environ.get("DIRECT_FIRST", "").strip() == "1",
        statics,
        paid,
    )
    static_errors: dict = {}

    # Same cookies + PO token the download uses: an anonymous probe gets
    # "Sign in to confirm you're not a bot" from the static IPs (4-sep-2026,
    # all three, ~10 probes/day paying 1.8 MB each on the per-GB proxy)
    # while the authenticated download sails through the same IPs.
    # One of the configured accounts (yt_account.Session), as a private file.
    session = yt_account.Session()
    try:
        return _probe_with_proxies(url, proxies, strategies, static_errors,
                                   paid, session.path)
    finally:
        session.close()


# What the last probes learned about each URL, for the quality gate that runs
# right after the metering probe (app._probe_youtube_quality): asking YouTube
# a second time for the same video spent the account's budget twice per
# submission. Small and short-lived: one entry per recent submission.
_PROBE_FACTS_TTL = 600
_probe_facts: dict = {}


def _remember_probe(url, info):
    try:
        heights = [f.get("height") or 0 for f in (info.get("formats") or [])
                   if f.get("vcodec", "none") != "none" and f.get("protocol") != "mhtml"
                   and f.get("ext") != "mhtml"]
        height = max(heights, default=0)
        if not height:
            return            # nothing to say about quality: let the gate ask
        now = time.time()
        for k in [k for k, v in _probe_facts.items() if now - v["at"] > _PROBE_FACTS_TTL]:
            _probe_facts.pop(k, None)
        _probe_facts[url] = {"at": now, "max_height": int(height),
                             "duration": int(info.get("duration") or 0)}
    except Exception:
        pass


def probe_facts(url):
    """``{max_height, duration, mode, cookies_invalid}`` from a probe of this
    URL in the last few minutes, or None."""
    f = _probe_facts.get(url)
    if not f or time.time() - f["at"] > _PROBE_FACTS_TTL:
        return None
    return {"max_height": f["max_height"], "duration": f["duration"],
            "mode": "metering-probe", "cookies_invalid": False}


def _probe_with_proxies(url, proxies, strategies, static_errors, paid, ck_path):
    """Walk the proxy chain (outer) x extractor strategies (inner) until one
    reports a duration. Split out of ``probe_url_minutes`` so the temporary
    cookie file has one obvious lifetime.

    A dead route in the chain is routine (the proxy watcher is what reports it,
    on Telegram); yt-dlp printing a full ERROR block per failed proxy per
    strategy would just flood the API log, hence the quiet logger. The reason
    still reaches the caller through ``last_err``.
    """
    import yt_dlp

    class _QuietLogger:
        def debug(self, msg): pass
        def info(self, msg): pass
        def warning(self, msg): pass
        def error(self, msg): pass

    last_err = None
    session = yt_account.Session.of(ck_path)
    for proxy in proxies:
        is_paid = bool(paid) and proxy == paid
        # Every attempt's error on this route, not just the last one. The
        # anonymous retry (see probe_url_minutes) on a static IP routinely
        # ends in "Sign in to confirm you're not a bot"; if the attempt with
        # the cookies had already said "confirm your age" / "Private video",
        # that is the verdict, and the bot-check must not turn it into an IP
        # problem worth the paid proxy (17-sep-2026, 3QFAEqE9-Kk, twice).
        route_errors = []
        if is_paid:
            # Only spend the per-GB proxy when a free route failed for a
            # reason another IP can fix. Content errors and "no duration"
            # (live streams) are the same on every IP.
            if not static_errors or not any(static_failure_warrants_paid(e)
                                            for e in static_errors.values()):
                break
        for step, (extractor_args, use_cookies) in enumerate(strategies):
            # noplaylist: a URL pasted from a playing playlist or a mix
            # (`watch?v=X&list=...`) is that ONE video. Without this yt-dlp
            # walks the whole list and fails on its first private / age-gated
            # / bot-checked entry, a video the user never asked for; the probe
            # then escalated to the per-GB proxy, which walked the same list
            # and failed the same way, and the user got a 400 for a valid
            # link. 26 of the 37 paid probes between 7 and 17-sep-2026 were
            # exactly this (proxy_usage: `list=` and `results?search_query`).
            opts = {"skip_download": True, "quiet": True, "no_warnings": True,
                    "noplaylist": True,
                    "logger": _QuietLogger(), "extractor_args": extractor_args}
            authed = bool(ck_path and use_cookies)
            if authed and session.spent:
                # YouTube said twice for this video (or, during a cooldown,
                # to everybody) that the account is limited: asking again
                # with its cookies gets the same answer and counts against
                # it. (Once is often one exit IP only: the next route is
                # still asked.) The anonymous step of this route still runs.
                route_errors.append("skipped: the YouTube account is limited")
                continue
            if authed:
                opts["cookiefile"] = ck_path
            if proxy:
                opts["proxy"] = proxy
            try:
                import ssrf_guard
                # With the cookies: one of the account's slots, spaced from
                # the previous request (yt_account), like the download.
                with (session.request() if authed else contextlib.nullcontext()):
                    with ssrf_guard.guarded(), yt_dlp.YoutubeDL(opts) as ydl:
                        info = ydl.extract_info(url, download=False)
                if authed:
                    session.ok()
                duration = info.get("duration")
                if duration:
                    _remember_probe(url, info)
                    if is_paid:
                        _paid_probe_events.append({
                            "url": url, "static_errors": dict(static_errors),
                            "bytes_estimate": 1_800_000 * (1 + step)})
                        if _all_bot_checked(static_errors):
                            _static_bot_verdicts.add(url)
                    return float(duration) / 60.0
                last_err = ValueError("no duration in metadata")
                if info.get("extractor") == "generic":
                    # A direct media file: yt-dlp's generic extractor never
                    # reports a duration, so stop trying proxies and read the
                    # container header over HTTP instead.
                    break
            except Exception as e:
                last_err = e
                route_errors.append(str(e)[:300])
                if authed and yt_account.is_rate_limit(str(e)):
                    session.limited()   # moves to another account if one is awake
        else:
            if not is_paid:
                static_errors[_route_name(proxy, proxies)] = (
                    " || ".join(route_errors) if route_errors else str(last_err)[:300])
            continue
        break
    if paid and any(p == paid for p in proxies) and static_errors and \
            any(static_failure_warrants_paid(e) for e in static_errors.values()) and last_err is not None \
            and not isinstance(last_err, ValueError):
        # The paid route was tried and failed too: still worth a trail line.
        _paid_probe_events.append({"url": url, "static_errors": dict(static_errors),
                                   "bytes_estimate": 3_600_000, "paid_failed": str(last_err)[:200]})
    # Direct file URLs (agent uploads on tmpfiles/uguu/R2, a CDN mp4): ffprobe
    # fetches just the moov atom via range requests. Also the last resort for
    # any URL yt-dlp could not size.
    try:
        seconds = _ffprobe_url_seconds(url)
        if seconds > 0:
            return seconds / 60.0
    except Exception as e:
        last_err = e
    raise ValueError(f"Could not determine video duration ({last_err})")


def _route_name(proxy, proxies) -> str:
    if proxy is None:
        return "direct"
    statics = [p for p in proxies if p is not None]
    try:
        return f"static{statics.index(proxy) + 1}"
    except ValueError:
        return "proxy"


def _ffprobe_url_seconds(url: str, timeout: int = 30) -> float:
    out = subprocess.check_output(
        ["ffprobe", "-v", "error", "-rw_timeout", str(timeout * 1_000_000),
         "-user_agent", "Mozilla/5.0 (OpenShorts probe)",
         "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", url],
        stderr=subprocess.STDOUT, timeout=timeout + 5,
    )
    return float(out.decode().strip() or 0)


# --------------------------------------------------------------------------- #
# Balance computation (assumes caller holds the user lock when mutating)
# --------------------------------------------------------------------------- #
async def _subscription_row(session, user_id):
    """The user's subscription row whatever its status (None if never subscribed).

    Distinct from ``_active_subscription``: a row in a state that grants no
    minutes (past_due, incomplete, ...) still has to be visible to the UI, or a
    customer whose card failed silently reads as a plain free account with no
    way to fix it. Entitlement decisions must use ``_active_subscription``.
    """
    return (await session.execute(
        select(Subscription).where(Subscription.user_id == user_id)
    )).scalar_one_or_none()


def _entitles(row) -> bool:
    """True if this subscription row grants plan minutes right now."""
    return bool(row and row.status in ("active", "trialing")
                and row.current_period_end > _now())


async def _active_subscription(session, user_id):
    row = await _subscription_row(session, user_id)
    return row if _entitles(row) else None


LIVE_PLANS_CHUNK = 500


async def live_plans(user_ids) -> dict:
    """``{str(user_id): plan}`` for the users among ``user_ids`` whose
    subscription grants its plan right now (the rule of ``_entitles``). For
    the job queue, which may hold thousands of jobs: one query per
    LIVE_PLANS_CHUNK owners, never one per job. Ids that are not uuids
    (self-host, test doubles) are skipped."""
    import uuid as _uuid
    ids = []
    for u in {str(x) for x in (user_ids or ()) if x is not None}:
        try:
            ids.append(_uuid.UUID(u))
        except ValueError:
            continue
    plans = {}
    if not ids:
        return plans
    async with database.session() as session:
        for i in range(0, len(ids), LIVE_PLANS_CHUNK):
            rows = (await session.execute(
                select(Subscription.user_id, Subscription.plan).where(and_(
                    Subscription.user_id.in_(ids[i:i + LIVE_PLANS_CHUNK]),
                    Subscription.status.in_(("active", "trialing")),
                    Subscription.current_period_end > _now(),
                ))
            )).all()
            plans.update({str(uid): plan for uid, plan in rows})
    return plans


async def topup_owners(user_ids) -> set:
    """``{str(user_id)}`` of the users among ``user_ids`` with top-up minutes
    left (the free queue's saturation mode queues their jobs as paid). One
    query per LIVE_PLANS_CHUNK owners; non-uuid ids are skipped."""
    import uuid as _uuid
    ids = []
    for u in {str(x) for x in (user_ids or ()) if x is not None}:
        try:
            ids.append(_uuid.UUID(u))
        except ValueError:
            continue
    out = set()
    if not ids:
        return out
    async with database.session() as session:
        for i in range(0, len(ids), LIVE_PLANS_CHUNK):
            rows = (await session.execute(
                select(CreditTopup.user_id).where(and_(
                    CreditTopup.user_id.in_(ids[i:i + LIVE_PLANS_CHUNK]),
                    CreditTopup.minutes_total > CreditTopup.minutes_consumed,
                )).distinct()
            )).scalars().all()
            out.update(str(u) for u in rows)
    return out


def free_plan_eligible(user) -> bool:
    """True if this ``User`` row qualifies for the free monthly allowance.

    Google accounts always qualify. Email (magic-link) accounts qualify too,
    as long as the address isn't a disposable/temp-mail domain — sign-up already
    blocks those (cloud/email_policy), and this is the defense-in-depth check so
    an old disposable account can't slip through. FREE_PLAN_MINUTES = 0 disables
    the free plan entirely.
    """
    if user is None or config.FREE_PLAN_MINUTES <= 0:
        return False
    # Denied accounts (re-registered after an erasure, temp-mail MX): no free
    # minutes whatever the sign-in method. getattr: test doubles and rows
    # loaded before the column existed.
    if getattr(user, "free_plan_denied", None):
        return False
    if user.google_sub:
        return True
    # Email account: eligible unless the domain is disposable.
    from . import email_policy
    return not email_policy.is_disposable(user.email or "")


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
    """User currently on the free tier (no active/trialing sub).

    Used by the retention sweeps: an account whose free minutes were withdrawn
    (``free_plan_denied``) still keeps its library on the free 7-day expiry,
    not forever.
    """
    if await _active_subscription(session, user_id):
        return False
    # A month on hold spends free minutes but is still a paying customer's
    # library: it must not expire like a free one while they are away.
    row = await _subscription_row(session, user_id)
    if row is not None and row.status == config.ON_HOLD:
        return False
    user = await session.get(User, user_id)
    if user is not None and getattr(user, "free_plan_denied", None):
        return True
    return free_plan_eligible(user)


async def _plan_used_this_period(session, user_id, period_end) -> Decimal:
    total = (await session.execute(
        select(func.coalesce(func.sum(UsageLedger.minutes_from_plan), 0)).where(and_(
            UsageLedger.user_id == user_id,
            UsageLedger.period_end == period_end,
            UsageLedger.status.in_(("reserved", "committed")),
        ))
    )).scalar_one()
    return _D(total)


async def has_processed_before(user_id) -> bool:
    """True once the account has any live (reserved/committed) process job.

    Released reservations (a job that failed or never started) don't count, so
    a first video that errors out keeps the first-video grant for the retry.
    """
    async with database.session() as session:
        row = (await session.execute(
            select(UsageLedger.id).where(and_(
                UsageLedger.user_id == user_id,
                UsageLedger.job_type == "process",
                UsageLedger.status.in_(("reserved", "committed")),
            )).limit(1)
        )).first()
    return row is not None


def ip_fingerprint(ip) -> str:
    """HMAC of a client IP (keyed on the app secret), or "" when unknown."""
    import hashlib
    import hmac
    if not ip or ip == "unknown":
        return ""
    key = (config.settings.jwt_secret or "openshorts").encode()
    return hmac.new(key, f"ip:{ip}".encode(), hashlib.sha256).hexdigest()


async def ip_had_first_video(ip_hash: str) -> bool:
    """True when this network already got a live first-video grant inside
    ``FIRST_VIDEO_IP_WINDOW_DAYS``."""
    if not ip_hash:
        return False
    since = _now() - timedelta(days=config.FIRST_VIDEO_IP_WINDOW_DAYS)
    async with database.session() as session:
        row = (await session.execute(
            select(FirstVideoGrant.id)
            .join(UsageLedger, UsageLedger.job_id == FirstVideoGrant.job_id)
            .where(and_(
                FirstVideoGrant.ip_hash == ip_hash,
                FirstVideoGrant.created_at >= since,
                UsageLedger.status.in_(("reserved", "committed")),
            )).limit(1)
        )).first()
    return row is not None


async def record_first_video(ip_hash: str, job_id: str):
    """Remember a first-video grant for this network; prune expired rows."""
    if not ip_hash:
        return
    from sqlalchemy import delete
    since = _now() - timedelta(days=config.FIRST_VIDEO_IP_WINDOW_DAYS)
    async with database.session() as session:
        async with session.begin():
            await session.execute(delete(FirstVideoGrant).where(FirstVideoGrant.created_at < since))
            session.add(FirstVideoGrant(ip_hash=ip_hash, job_id=job_id))


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
    sub_row = await _subscription_row(session, user_id)
    sub = sub_row if _entitles(sub_row) else None
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
        "_sub_row": sub_row,
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


async def free_actions_today(user_id, job_type: str) -> int:
    """How many zero-minute ``job_type`` rows this user has today (UTC)."""
    day_start = _now().replace(hour=0, minute=0, second=0, microsecond=0)
    async with database.session() as session:
        return int((await session.execute(
            select(func.count(UsageLedger.id)).where(and_(
                UsageLedger.user_id == user_id,
                UsageLedger.job_type == job_type,
                UsageLedger.created_at >= day_start,
            ))
        )).scalar_one() or 0)


async def record_free_action(user_id, job_id: str, job_type: str):
    """A committed zero-minute row: counts a free daily allowance (AI edits)
    across restarts without touching the balance."""
    async with database.session() as session:
        async with session.begin():
            session.add(UsageLedger(
                user_id=user_id, job_id=str(job_id), job_type=job_type,
                minutes=_D(0), minutes_from_plan=_D(0), minutes_from_topup=_D(0),
                status="committed",
            ))


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


# A reservation this recent at startup may belong to a job the other instance
# of a deploy accepted while this one booted, whose manifest the startup scan
# could not have seen yet: leave it to the sweeper, which checks liveness.
ORPHAN_GRACE_SECONDS = 120


async def release_orphaned_reservations(keep_ids=None, created_before=None):
    """At startup, release every still-``reserved`` row.

    Jobs live only in memory, so a restart loses all in-flight jobs; their
    reservations must be refunded or they would leak quota forever.

    ``keep_ids`` are reservations for jobs being *resumed* after the restart —
    those keep running and will settle themselves, so they must not be refunded.
    ``created_before`` (the boot instant) limits it to rows older than that
    minus ORPHAN_GRACE_SECONDS: during a deploy the old instance keeps taking
    jobs, and one reserved while this instance booted has a manifest the keep
    set was computed too early to contain.
    """
    keep = {str(k) for k in (keep_ids or ())}
    q = select(UsageLedger.id).where(UsageLedger.status == "reserved")
    if created_before is not None:
        q = q.where(UsageLedger.created_at
                    < created_before - timedelta(seconds=ORPHAN_GRACE_SECONDS))
    async with database.session() as session:
        ids = list((await session.execute(q)).scalars())
    released = 0
    for lid in ids:
        if str(lid) in keep:
            continue
        await release_reservation(str(lid))
        released += 1
    if released:
        print(f"☁️  Released {released} orphaned reservation(s) at startup.")


def _live_cap_hours() -> float:
    try:
        return float(os.environ.get("LIVE_RESERVATION_MAX_HOURS") or LIVE_RESERVATION_MAX_HOURS)
    except ValueError:
        return float(LIVE_RESERVATION_MAX_HOURS)


async def _sweep_once(live_ids=None):
    """Release reservations older than STUCK_RESERVATION_HOURS whose job is over.

    ``live_ids()`` (app.py; may be a coroutine) maps the reservations whose job
    is still queued, running or waiting in a resume manifest on the shared disk
    to the moment that job was last known to be moving (``alive_since``, epoch
    seconds; a plain set counts as "moving now"). A queue longer than the stuck
    window must not refund a video that will still be delivered: its commit
    would then find nothing to commit. It is read BEFORE the rows, so a job
    that ends in between is either committed already or still counted live. If
    it fails the pass is skipped: not knowing what is alive is no reason to
    refund everything. A live reservation whose job has not moved for
    LIVE_RESERVATION_MAX_HOURS is released as hung; that clock is the job's
    (a run that never ends, a manifest nobody takes), never the reservation's
    age, so a long wait in the queue does not count.
    """
    live = None
    if live_ids is not None:
        try:
            got = live_ids()
            if asyncio.iscoroutine(got) or isinstance(got, asyncio.Future):
                got = await got
            if isinstance(got, dict):
                live = {str(k): float(v) for k, v in got.items()}
            else:
                stamp = _now().timestamp()
                live = {str(k): stamp for k in got}
        except Exception as e:
            print(f"⚠️  Metering sweep skipped, live jobs unknown: {e}")
            return
    now = _now()
    cutoff = now - timedelta(hours=STUCK_RESERVATION_HOURS)
    hung_before = now.timestamp() - _live_cap_hours() * 3600
    async with database.session() as session:
        rows = list((await session.execute(
            select(UsageLedger.id).where(and_(
                UsageLedger.status == "reserved",
                UsageLedger.created_at < cutoff,
            ))
        )).scalars())
    ids = [lid for lid in rows
           if live is None or str(lid) not in live or live[str(lid)] < hung_before]
    for lid in ids:
        await release_reservation(str(lid))
    if ids:
        print(f"☁️  Swept {len(ids)} stuck reservation(s).")
    kept = len(rows) - len(ids)
    if kept:
        print(f"☁️  Kept {kept} reservation(s) older than {STUCK_RESERVATION_HOURS} h: "
              f"their job is still queued or running.")


async def _sweeper_loop(live_ids=None, is_active=None):
    while True:
        try:
            await asyncio.sleep(SWEEP_INTERVAL_SECONDS)
            # A draining instance leaves the sweep to the new one: from here
            # the new one's queue looks like manifests nobody has taken.
            if is_active is not None and not is_active():
                continue
            await _sweep_once(live_ids)
        except asyncio.CancelledError:
            break
        except Exception as e:  # never let the sweeper die
            print(f"⚠️  Metering sweeper error: {e}")


def start_sweeper(live_ids=None, is_active=None):
    """``live_ids``: callable (or coroutine function) returning the live
    reservations, see ``_sweep_once``. ``is_active``: False while draining."""
    asyncio.create_task(_sweeper_loop(live_ids, is_active))
