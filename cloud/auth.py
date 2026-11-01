"""Authentication for cloud mode: magic link + JWT sessions + current-user loading.

- Magic link: POST /api/auth/magic-link -> emailed single-use token (15 min).
  POST /api/auth/magic-link/verify consumes it (POST, not GET, so email scanners
  that prefetch links can't burn them) and returns a 30-day JWT.
- get_current_user_optional never raises: a missing/expired token degrades to the
  anonymous BYOK flow rather than breaking it.
- /api/me returns the user's plan + minute balance (no Stripe call needed).
"""
import hashlib
import hmac
import re
import secrets
from datetime import datetime, timezone, timedelta
from typing import Optional

import jwt
from fastapi import APIRouter, Request, HTTPException
from pydantic import BaseModel, EmailStr
from sqlalchemy import select, func

from .config import settings
from . import config, database, metering, email_policy, scopes
from .models import User, MagicLinkToken, UploadPostProfile, SignupAttribution, LoginCode

router = APIRouter()

JWT_ALGO = "HS256"
JWT_TTL_DAYS = 30
MAGIC_TTL_MINUTES = 15
MAGIC_RATE_LIMIT = 3          # per email
MAGIC_RATE_LIMIT_IP = 10      # per IP, any email — bots rotate addresses, not IPs
MAGIC_RATE_WINDOW_MIN = 15
# The Google sign-in code: traded once, by POST, within this window.
LOGIN_CODE_TTL_MINUTES = 5
_CHALLENGE_RE = re.compile(r"^[A-Za-z0-9_-]{43}$")
# Only accept attribution for an account this young. Both sign-in flows post it
# right after the redirect, so anything older is a returning user whose "source"
# would be their re-entry, not their sign-up.
ATTRIBUTION_MAX_AGE_MIN = 60
ATTRIBUTION_FIELD_MAX = 500   # client-supplied strings are truncated to this


def _now():
    return datetime.now(timezone.utc)


# --------------------------------------------------------------------------- #
# JWT + current user
# --------------------------------------------------------------------------- #
def issue_jwt(user_id, email) -> str:
    payload = {
        "sub": str(user_id),
        "email": email,
        "exp": _now() + timedelta(days=JWT_TTL_DAYS),
        "iat": _now(),
    }
    return jwt.encode(payload, settings.jwt_secret, algorithm=JWT_ALGO)


def _decode_jwt(token: str) -> Optional[dict]:
    try:
        return jwt.decode(token, settings.jwt_secret, algorithms=[JWT_ALGO])
    except Exception:
        return None


class CurrentUser:
    """Lightweight per-request user snapshot carried through endpoints.

    Kept small and eager (no lazy DB access) so sync helpers like
    ``managed_keys.has_active_entitlement`` can read it without awaiting.
    """

    def __init__(self, id, email, entitled=False, plan=None, upload_post_profile=None):
        self.id = id
        self.email = email
        self.entitled = entitled
        self.plan = plan
        self.upload_post_profile = upload_post_profile


async def _load_current_user(session, user_id) -> Optional["CurrentUser"]:
    user = await session.get(User, user_id)
    if user is None:
        return None
    sub = await metering._active_subscription(session, user_id)
    topups = await metering._topups_fifo(session, user_id)
    topup_remaining = sum(
        (float(t.minutes_total) - float(t.minutes_consumed) for t in topups), 0.0
    )
    profile = await session.get(UploadPostProfile, user_id)
    # Entitled to managed keys with an active plan, any top-up credit, or the
    # free monthly allowance (Google-authenticated accounts only).
    free = sub is None and metering.free_plan_eligible(user)
    entitled = bool(sub is not None or topup_remaining > 0 or free)
    return CurrentUser(
        id=user.id,
        email=user.email,
        entitled=entitled,
        plan=sub.plan if sub else ("free" if free else None),
        upload_post_profile=profile.profile_username if profile else None,
    )


async def get_current_user_optional(request: Request) -> Optional[CurrentUser]:
    """Return the authenticated user, or None for anonymous / BYOK requests.

    Accepts either a session JWT (``Bearer <jwt>``) or a user API key
    (``Bearer osk_...`` / ``X-API-Key``). A key authenticates as its owner, so
    quota, entitlement and job ownership apply to agents exactly as to the
    dashboard — no parallel auth path to keep in sync.
    """
    from . import api_keys

    auth = request.headers.get("Authorization", "")
    token = auth[7:].strip() if auth.lower().startswith("bearer ") else ""
    api_token = token if api_keys.looks_like_key(token) else request.headers.get("X-API-Key", "")

    if api_keys.looks_like_key(api_token):
        principal = await api_keys.key_principal(api_token)
        if principal is None:
            return None
        user_id, scope = principal
        if not scopes.allows(scope, request.method, request.url.path):
            # An OAuth token only reaches what its consent covered. Loud
            # rather than anonymous: a None here would read as "sign in".
            print(f"🔒 Scoped key refused: {request.method} {request.url.path} (scope {scope})")
            raise HTTPException(status_code=403, detail=(
                f"This token was issued to an MCP client (scope '{scope}') and cannot "
                "call this endpoint. Use the MCP tools, or create an API key in the "
                "dashboard for full API access."))
        async with database.session() as session:
            return await _load_current_user(session, user_id)

    if not token:
        return None
    payload = _decode_jwt(token)
    if not payload:
        return None
    try:
        import uuid
        user_id = uuid.UUID(payload["sub"])
    except Exception:
        return None
    async with database.session() as session:
        return await _load_current_user(session, user_id)


async def get_current_user_required(request: Request) -> CurrentUser:
    user = await get_current_user_optional(request)
    if user is None:
        raise HTTPException(status_code=401, detail="Authentication required")
    return user


async def get_session_user_required(request: Request,
                                    detail: str = "Sign in to the dashboard to do this.") -> CurrentUser:
    """The signed-in user, refusing API-key auth (``osk_``, OAuth tokens too).

    For what an agent credential must never reach: money (cancel, resume,
    end a trial, the Stripe portal, invoices) on top of keys and erasure,
    which have their own copies of this check."""
    from . import api_keys
    auth = request.headers.get("Authorization", "")
    if api_keys.looks_like_key(auth[7:].strip()) or api_keys.looks_like_key(
            request.headers.get("X-API-Key", "")):
        raise HTTPException(status_code=403, detail=detail)
    return await get_current_user_required(request)


# --------------------------------------------------------------------------- #
# Magic link
# --------------------------------------------------------------------------- #
class MagicLinkRequest(BaseModel):
    email: EmailStr


class MagicVerifyRequest(BaseModel):
    token: str


@router.post("/api/auth/magic-link")
async def request_magic_link(payload: MagicLinkRequest, request: Request):
    from .emails import send_magic_link_email

    disposable_msg = (
        "That email provider isn't supported. Please use a permanent "
        "address (Gmail, Outlook, iCloud…) or sign in with Google.")

    # Block temp-mail before doing anything else — the free plan is open to
    # email accounts now, so disposable domains would be a free-minute farm.
    if email_policy.is_disposable(payload.email):
        raise HTTPException(status_code=400, detail=disposable_msg)

    # Normalize (strip +tags / Gmail dots) so aliases can't mint extra accounts.
    email = email_policy.normalize_email(payload.email)

    if review_login_enabled() and email == _review_email():
        return {"ok": True, "password_required": True}

    # Best-effort: uvicorn resolves this from X-Forwarded-For (trusted "*" in
    # the Dockerfile), which a client can forge — good enough against the bots
    # we actually see, but not a security boundary.
    client_ip = request.client.host if request.client else None

    raw_token = secrets.token_urlsafe(32)
    token_hash = hashlib.sha256(raw_token.encode()).hexdigest()
    async with database.session() as session:
        async with session.begin():
            window_start = _now() - timedelta(minutes=MAGIC_RATE_WINDOW_MIN)
            recent, recent_ip = (await session.execute(
                select(
                    func.count(MagicLinkToken.id).filter(MagicLinkToken.email == email),
                    func.count(MagicLinkToken.id).filter(MagicLinkToken.request_ip == client_ip),
                ).where(MagicLinkToken.created_at >= window_start)
            )).one()
            if recent >= MAGIC_RATE_LIMIT:
                raise HTTPException(status_code=429, detail="Too many login attempts. Try again in a few minutes.")
            if client_ip and recent_ip >= MAGIC_RATE_LIMIT_IP:
                raise HTTPException(status_code=429, detail=(
                    "Too many sign-in attempts from your network. Try again in a few minutes."))

            # Recorded BEFORE the MX gate so rejected attempts still count
            # toward both limits — otherwise a junk-domain flood never trips
            # them. The raw token is only ever emailed after the gate passes.
            session.add(MagicLinkToken(
                email=email,
                token_hash=token_hash,
                expires_at=_now() + timedelta(minutes=MAGIC_TTL_MINUTES),
                request_ip=client_ip,
            ))
            account_exists = (await session.execute(
                select(User.id).where(User.email == email)
            )).scalar_one_or_none() is not None

    # The static list can't keep up with rotating temp-mail domains, and bots
    # sign up with domains that don't receive mail at all — both come back as
    # bounces that flood the support inbox and burn sender reputation. The
    # gate protects sign-UP only: an existing account must keep working even
    # if its domain's MX later matches the disposable list or resolves badly.
    if not account_exists:
        verdict = await email_policy.mx_verdict(email)
        if verdict == email_policy.MX_DISPOSABLE:
            raise HTTPException(status_code=400, detail=disposable_msg)
        if verdict == email_policy.MX_NONE:
            raise HTTPException(status_code=400, detail=(
                "That email domain can't receive mail. Please double-check the address."))
    else:
        # The account keeps signing in, but eligibility for the free minutes
        # is re-checked: a temp-mail front whose MX joined disposable_mx.txt
        # after the account was made would otherwise keep its free plan
        # forever (568 such accounts in sep-2026).
        await _deny_free_if_disposable_mx(email)

    link = f"{settings.frontend_url}/#/auth/verify?ml={raw_token}"
    await send_magic_link_email(email, link)
    return {"ok": True}


async def _deny_free_if_disposable_mx(email: str):
    """Mark an existing account's free plan withdrawn when its mail domain's
    MX is a known temp-mail host. Best effort: never blocks the sign-in."""
    try:
        if await email_policy.mx_verdict(email) != email_policy.MX_DISPOSABLE:
            return
        from sqlalchemy import update
        async with database.session() as session:
            async with session.begin():
                await session.execute(
                    update(User)
                    .where(User.email == email, User.free_plan_denied.is_(None))
                    .values(free_plan_denied="disposable_mx"))
    except Exception as e:
        print(f"⚠️ free-plan MX re-check failed: {e}")


async def purge_stale_magic_tokens():
    """Drop token rows too old to matter for auth or rate limiting.

    Called from the retention sweeper: without it the table grows with every
    sign-in forever and the rate-limit counts scan an ever-larger window.
    """
    from sqlalchemy import delete

    cutoff = _now() - timedelta(days=30)
    async with database.session() as session:
        async with session.begin():
            await session.execute(
                delete(MagicLinkToken).where(MagicLinkToken.created_at < cutoff))
            await session.execute(
                delete(LoginCode).where(LoginCode.expires_at < _now() - timedelta(days=1)))


@router.post("/api/auth/magic-link/verify")
async def verify_magic_link(payload: MagicVerifyRequest):
    token_hash = hashlib.sha256(payload.token.encode()).hexdigest()
    async with database.session() as session:
        async with session.begin():
            row = (await session.execute(
                select(MagicLinkToken).where(MagicLinkToken.token_hash == token_hash)
                .with_for_update()
            )).scalar_one_or_none()
            if row is None or row.used_at is not None or row.expires_at < _now():
                raise HTTPException(status_code=400, detail="This sign-in link is invalid or expired.")
            row.used_at = _now()
            email = row.email.lower()

            user = (await session.execute(
                select(User).where(User.email == email)
            )).scalar_one_or_none()
            if user is None:
                from .account import free_plan_denial_for_signup
                user = User(email=email, last_login_at=_now(),
                            free_plan_denied=await free_plan_denial_for_signup(session, email))
                session.add(user)
                await session.flush()
            else:
                user.last_login_at = _now()
            user_id, user_email = user.id, user.email

        token = issue_jwt(user_id, user_email)
        cu = await _load_current_user(session, user_id)
    return {"token": token, "user": {"id": str(user_id), "email": user_email, "entitled": cu.entitled}}


# --------------------------------------------------------------------------- #
# Google sign-in code exchange
# --------------------------------------------------------------------------- #
def valid_challenge(value) -> bool:
    return isinstance(value, str) and bool(_CHALLENGE_RE.match(value))


def challenge_for(verifier: str) -> str:
    import base64
    return base64.urlsafe_b64encode(
        hashlib.sha256(verifier.encode()).digest()).rstrip(b"=").decode()


async def create_login_code(user_id, challenge: str) -> str:
    """Mint the single-use code the OAuth callback redirects with."""
    raw = secrets.token_urlsafe(32)
    async with database.session() as session:
        async with session.begin():
            session.add(LoginCode(
                code_hash=hashlib.sha256(raw.encode()).hexdigest(), user_id=user_id,
                challenge=challenge,
                expires_at=_now() + timedelta(minutes=LOGIN_CODE_TTL_MINUTES)))
    return raw


class LoginExchangeRequest(BaseModel):
    code: str
    verifier: str


@router.post("/api/auth/exchange")
async def exchange_login_code(payload: LoginExchangeRequest):
    """Trade the Google callback's code for a session JWT. Needs the verifier
    the dashboard kept in this browser's sessionStorage when it started the
    sign-in, so a code someone else obtained cannot sign this browser in."""
    bad = HTTPException(status_code=400, detail="This sign-in has expired. Please sign in again.")
    if not payload.code or len(payload.code) > 200 or not (43 <= len(payload.verifier or "") <= 128):
        raise bad
    code_hash = hashlib.sha256(payload.code.encode()).hexdigest()
    async with database.session() as session:
        async with session.begin():
            row = (await session.execute(
                select(LoginCode).where(LoginCode.code_hash == code_hash).with_for_update()
            )).scalar_one_or_none()
            if row is None or row.used_at is not None or row.expires_at < _now():
                raise bad
            if not hmac.compare_digest(challenge_for(payload.verifier), row.challenge):
                raise bad
            row.used_at = _now()
            user = await session.get(User, row.user_id)
            if user is None:
                raise bad
            user_id, user_email = user.id, user.email
        token = issue_jwt(user_id, user_email)
        cu = await _load_current_user(session, user_id)
    return {"token": token, "user": {"id": str(user_id), "email": user_email, "entitled": cu.entitled}}


# --------------------------------------------------------------------------- #
# Reviewer sign-in (ChatGPT app directory)
# --------------------------------------------------------------------------- #
class ReviewLoginRequest(BaseModel):
    email: EmailStr
    password: str


def review_login_enabled() -> bool:
    return bool(settings.review_login_email and settings.review_login_password)


def _review_email() -> str:
    return email_policy.normalize_email(settings.review_login_email)


def _is_review_email(email: str) -> bool:
    # compare_digest only takes ASCII str; EmailStr admits non-ASCII addresses.
    return hmac.compare_digest(email.encode(), _review_email().encode())


class LoginMethodRequest(BaseModel):
    email: EmailStr


@router.post("/api/auth/login-method")
async def login_method(payload: LoginMethodRequest):
    """Tell the sign-in form, as the email is typed, whether it signs in with a
    password. Only the review account does; everyone else gets the magic link.
    Reviewers are handed a username and a password, so the password field has
    to be on screen before they press anything that reads "email me a link"."""
    email = email_policy.normalize_email(payload.email)
    return {"password": review_login_enabled() and _is_review_email(email)}


@router.post("/api/auth/review-login")
async def review_login(payload: ReviewLoginRequest):
    """Password sign-in for the single account named by REVIEW_LOGIN_EMAIL.

    Directory reviewers test the OAuth connection from their own browser and
    cannot open a magic link sent to our test inbox. Only that existing account
    can sign in this way, and only while both env vars are set."""
    email = email_policy.normalize_email(payload.email)
    ok = (review_login_enabled()
          and _is_review_email(email)
          and hmac.compare_digest(payload.password.encode(), settings.review_login_password.encode()))
    if not ok:
        raise HTTPException(status_code=401, detail="Wrong email or password.")
    async with database.session() as session:
        async with session.begin():
            user = (await session.execute(
                select(User).where(User.email == email)
            )).scalar_one_or_none()
            if user is None:
                raise HTTPException(status_code=401, detail="Wrong email or password.")
            user.last_login_at = _now()
            user_id, user_email = user.id, user.email
        token = issue_jwt(user_id, user_email)
        cu = await _load_current_user(session, user_id)
    return {"token": token, "user": {"id": str(user_id), "email": user_email, "entitled": cu.entitled}}


# --------------------------------------------------------------------------- #
# Signup attribution
# --------------------------------------------------------------------------- #
class AttributionPayload(BaseModel):
    referrer: Optional[str] = None
    landing_path: Optional[str] = None
    utm_source: Optional[str] = None
    utm_medium: Optional[str] = None
    utm_campaign: Optional[str] = None


def _clip(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    value = value.strip()
    return value[:ATTRIBUTION_FIELD_MAX] or None


def _referrer_host(referrer: Optional[str]) -> Optional[str]:
    """Bare host of the referrer, the key we actually group by."""
    if not referrer:
        return None
    try:
        from urllib.parse import urlparse
        host = (urlparse(referrer).hostname or "").lower()
    except Exception:
        return None
    return host or None


@router.post("/api/auth/attribution")
async def record_attribution(payload: AttributionPayload, request: Request):
    """Record where a brand-new account came from. First touch wins.

    Fire-and-forget from the client right after sign-in: it never fails the
    sign-up, it just returns ``recorded: false`` when the row already exists or
    the account is too old to be a genuine sign-up.
    """
    from sqlalchemy.dialects.postgresql import insert as pg_insert

    user = await get_current_user_required(request)
    async with database.session() as session:
        async with session.begin():
            row = await session.get(User, user.id)
            if row is None or row.created_at is None:
                return {"ok": True, "recorded": False}
            if _now() - row.created_at > timedelta(minutes=ATTRIBUTION_MAX_AGE_MIN):
                return {"ok": True, "recorded": False}
            referrer = _clip(payload.referrer)
            # ON CONFLICT DO NOTHING keeps first touch even if the client posts
            # twice concurrently (two tabs finishing the same redirect).
            result = await session.execute(
                pg_insert(SignupAttribution).values(
                    user_id=user.id,
                    referrer=referrer,
                    referrer_host=_referrer_host(referrer),
                    landing_path=_clip(payload.landing_path),
                    utm_source=_clip(payload.utm_source),
                    utm_medium=_clip(payload.utm_medium),
                    utm_campaign=_clip(payload.utm_campaign),
                ).on_conflict_do_nothing(index_elements=["user_id"])
            )
    return {"ok": True, "recorded": result.rowcount > 0}


# --------------------------------------------------------------------------- #
# /api/me
# --------------------------------------------------------------------------- #
@router.get("/api/me")
async def get_me(request: Request):
    user = await get_current_user_required(request)
    async with database.session() as session:
        bal = await metering._balance(session, user.id)
        sub = bal["_sub"]
        from .onboarding import is_pending as _survey_pending
        survey_pending = await _survey_pending(session, user)
    # Report the entitling subscription when there is one; otherwise fall back to
    # a row that needs the customer's attention (card declined, payment never
    # finished). Without this a past_due customer sees a plain free account and
    # never learns their card failed. `plan` stays as computed — the entitlement
    # really is free — so the UI shows "free plan" plus the warning badge.
    attention = bal["_sub_row"]
    if attention is not None and attention.status not in (*config.BILLING_ATTENTION_STATES,
                                                          config.ON_HOLD):
        attention = None
    shown = sub or attention
    return {
        "user": {"id": str(user.id), "email": user.email},
        "entitled": user.entitled,
        "plan": bal["plan"],
        "status": shown.status if shown else None,
        "interval": shown.interval if shown else None,
        "period_end": bal["period_end"].isoformat() if bal["period_end"] else None,
        "cancel_at_period_end": shown.cancel_at_period_end if shown else False,
        # Month on hold (cancel flow): when the plan and its billing come back.
        "hold_until": (shown.current_period_end.isoformat()
                       if shown and shown.status == config.ON_HOLD else None),
        # A hold already booked for the next period, still undoable.
        "hold_scheduled": bool(shown and shown.status == "active" and shown.hold_from
                               and shown.hold_from > datetime.now(timezone.utc)),
        # Drives the "Manage billing" route in the UI: any Stripe relationship,
        # live or broken, must keep the portal reachable.
        "has_billing_account": bool(sub or attention),
        "minutes": {
            "plan_allowance": bal["plan_allowance"],
            "plan_used": bal["plan_used"],
            "plan_remaining": bal["plan_remaining"],
            "topup_remaining": bal["topup_remaining"],
            "remaining": bal["remaining"],
        },
        "upload_post_profile": user.upload_post_profile,
        "onboarding_survey_pending": survey_pending,
    }
