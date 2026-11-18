"""Central configuration for the optional cloud (paid / managed-keys) mode.

Everything here is read lazily from environment variables so that importing the
`cloud` package never has side effects. Nothing in this module is loaded unless
``BILLING_ENABLED`` is truthy (see ``cloud.is_enabled``).
"""
import os
import math
from functools import lru_cache


def _flag(name: str, default: str = "") -> bool:
    return os.environ.get(name, default).lower() in ("1", "true", "yes")


def is_enabled() -> bool:
    """Master switch. When False the whole cloud package stays dormant."""
    return _flag("BILLING_ENABLED")


# --- Plan catalog -----------------------------------------------------------
# minutes granted per billing period, keyed by internal plan name. This is the
# AUTHORITATIVE quota source (not Stripe price metadata, which a dashboard edit
# could change). Plan name is resolved from the Stripe price; minutes from here.
PLAN_MINUTES = {
    "starter": 100,
    "creator": 300,
    "pro": 750,
}

# Free plan: monthly allowance for Google-authenticated users with no
# subscription. No Stripe object exists for it — it is resolved entirely in
# cloud/metering.py against a synthetic calendar-month period. Setting
# FREE_PLAN_MINUTES = 0 disables the free plan.
FREE_PLAN_MINUTES = 20
# An address erased (DELETE /api/account) within this many days gets no free
# minutes when it signs up again: deleting and re-registering was a way to
# reset the monthly allowance (15 re-registrations by 6 accounts, sep-2026).
FREE_REDO_BLOCK_DAYS = 90
# Free is open to Google accounts AND permanent email accounts; disposable /
# temp-mail domains are blocked at sign-up (cloud/email_policy) and aliases are
# normalized, so the plan isn't a multi-account faucet.
# The 20-minute monthly quota is the only bound on free usage — no daily job
# cap (it confused users and the minute ledger already caps real cost).
# Free users' clips expire from R2 after this many days (paid libraries are
# durable). Also an upgrade lever, mirroring OpusClip's 3-day free exports.
FREE_CLIP_RETENTION_DAYS = 7

# Free trial length (days) for new subscriptions. Trials are retired in favor
# of the free plan; the checkout only injects trial_period_days when > 0, and
# existing 'trialing' subscriptions are still honored until they convert or
# cancel (grandfathering).
TRIAL_DAYS = 0

# Subscription states that grant no minutes but still need the customer to act
# (fix a card, finish a payment). They must stay VISIBLE in /api/me and keep the
# billing-portal route reachable — otherwise the account silently reads as free
# and the user has no way out. Terminal states (canceled, incomplete_expired)
# are deliberately absent: nothing to fix, so nothing to nag about.
BILLING_ATTENTION_STATES = ("past_due", "unpaid", "incomplete", "paused")

# Subscription states that must block a NEW subscription checkout. Narrower than
# BILLING_ATTENTION_STATES on purpose: Stripe is still trying to collect on
# past_due/unpaid/paused, so a second subscription would double-bill. It is NOT
# collecting on 'incomplete' (payment abandoned at 3DS/confirm, expired by Stripe
# within 24h), so blocking that would only stop the retry.
CHECKOUT_BLOCKING_STATES = ("active", "trialing", "past_due", "unpaid", "paused", "on_hold")

# "Pause for a month" from the cancel flow (cloud/cancellation). Stripe keeps
# the subscription 'active' and voids the invoice of the skipped month
# (pause_collection), so our row derives its own state from the hold window
# stored in the subscription's metadata: a period that STARTS inside
# [hold_from, hold_until) is the skipped one. 'on_hold' grants no plan minutes
# (the account falls back to the free plan) but keeps the library durable, and
# is distinct from Stripe's own 'paused' (a trial that ended without a card).
ON_HOLD = "on_hold"
# hold_until = hold_from + HOLD_DAYS: shorter than any monthly period, so the
# period after the skipped one starts past the window and is charged again.
HOLD_DAYS = 27


def subscription_status(sub_obj: dict, period_start) -> str:
    """Our status for a Stripe subscription whose current period starts at
    ``period_start`` (unix): Stripe's own, or ON_HOLD inside the hold window.

    Lives here (not in billing.py) so the test runs without the stripe SDK.
    """
    status = sub_obj.get("status")
    meta = sub_obj.get("metadata") or {}
    try:
        hold_from, hold_until = int(meta.get("hold_from") or 0), int(meta.get("hold_until") or 0)
    except (TypeError, ValueError):
        return status
    if status == "active" and period_start and hold_from and hold_from <= int(period_start) < hold_until:
        return ON_HOLD
    return status

def new_subscriber_label(status: str):
    """Admin-alert text for a brand-new subscription row, or None to stay silent.

    Only states where a payment method actually went through count. Stripe
    creates the row as 'incomplete' seconds after the Checkout page opens,
    before the user types anything; announcing that is announcing a click.
    Lives here (not in billing.py) so the test runs without the stripe SDK.
    """
    if status == "trialing":
        return "trial started — card on file"
    if status == "active":
        return status
    return None


# Smallest slice of a long video worth offering instead of the quota wall.
# A user whose remaining minutes cannot cover the whole source is offered its
# first N minutes (N = what they have left) so they see clips before paying:
# 93 of the 99 walls sampled to 16-sep-2026 were shown to accounts with their
# 20 free minutes untouched, pasting a 21-90 min video, and most left Stripe
# without paying. Below this many minutes the slice cannot hold a few 15-60 s
# clips, so the wall stays as it was.
PARTIAL_MIN_MINUTES = 5

# A free account's FIRST video is clipped whole when it runs up to this many
# minutes, even past the 20-minute balance: the balance goes to zero and the
# user sees every clip of the video they came with. Once per account (any
# reserved/committed process job disqualifies it), and the free plan itself is
# already gated (Google or permanent email, re-registration blocked 90 days).
# Longer first videos, and every later one past the balance, are clipped to
# the first N minutes automatically instead of hitting the wall.
# 0 disables the grant.
FIRST_VIDEO_MAX_MINUTES = 60
# ...and once per client IP in this many days, whatever the account: a pile of
# Google accounts from one network gets one whole first video, and the rest
# fall back to the first-N-minutes cut. The IP is kept only as an HMAC
# (cloud/metering.ip_fingerprint) and rows older than the window are deleted.
FIRST_VIDEO_IP_WINDOW_DAYS = 30

# Minute cap DURING the trial (across all plans). Kept for grandfathered
# 'trialing' subscriptions; removable once no subscription has status
# 'trialing'.
TRIAL_MINUTE_CAP = 20

# Gemini IMAGE generation (thumbnails) is the one expensive managed Gemini call
# (~$0.04/image, batch of ~3). It isn't naturally minute-metered, so each
# thumbnail generation batch consumes this many minutes from the plan quota —
# roughly matching its cost to the per-minute economics. Text (titles/desc) is free.
THUMBNAIL_MINUTES = 3

# Other managed Gemini calls that upload a video for context (AI effect/filter
# generation, thumbnail analysis, SaaS-shorts analysis). Cheaper than image gen
# but not free — meter a small fixed cost so an entitled user can't loop them to
# burn the operator's managed Gemini budget. Pure-text calls (titles/desc) stay free.
MANAGED_ANALYSIS_MINUTES = 1

# Post-processing FFmpeg re-encodes (subtitle burn, hook overlay) and the
# Remotion render proxy do real server compute per call. Meter a small fixed
# cost so an entitled user can't loop them for free off-quota.
SUBTITLE_MINUTES = 2


def subtitle_minutes_for(filename: str) -> int:
    """Minutes charged for burning captions onto ``filename``.

    Zero for the normal path: the SRT comes from the transcript already stored
    in metadata.json and the burn is one short FFmpeg pass, so there is nothing
    to recover. Captions are also table stakes for short-form — charging 2 min
    (10% of the free monthly quota) per clip priced them out of the product and
    only 9% of delivered clips ever had them (prod audit, 25-jul-2026).

    Dubbed clips are the exception: subtitling them re-runs Whisper over the
    translated audio, which is real compute, so they keep the charge.
    """
    return SUBTITLE_MINUTES if str(filename).startswith("translated_") else 0
RENDER_MINUTES = 3

# Editing a clip that already exists is free: AI edit and effects, the hook
# overlay, recut/rerender and reframe. Metering them cost ~1% of all minutes
# (prod, 24-sep to 24-oct-2026) and made the editor feel like a meter running,
# when the minutes a user buys are meant to pay for the video they import.
# What bounds the compute instead is EDITOR_ACTIONS_PER_HOUR per user (p99 of
# the same month: 12 such actions in an hour, the busiest user 46).
EDITOR_ACTION_MINUTES = 0
EDITOR_ACTIONS_PER_HOUR = 60
# Except the two that call Gemini with the clip (AI edit, AI effects): those
# spend the managed Google key, so only this many a day per user are free and
# each one after costs MANAGED_ANALYSIS_MINUTES again. Real use since 1-aug-2026:
# p99 is 4 a day, the busiest user-day 8, so 5 covers 99.4% of user-days, while
# the hourly cap alone would let one account run ~1,400 Gemini calls a day.
AI_EDITS_FREE_PER_DAY = 5


def billable_minutes(minutes: float) -> int:
    """Whole minutes charged for ``minutes`` of source: rounded down, at least 1.

    A 4.6-minute video costs 4, not 5, the way the per-minute clippers bill
    (Opus Clip rounds partial minutes down too). Anything that must cover the
    whole file (the download's safety cap) uses the duration rounded UP, not
    this.
    """
    try:
        m = float(minutes)
    except (TypeError, ValueError):
        return 1
    return max(1, int(math.floor(m + 1e-9)))

# The thumbnail-studio upload kicks off a (possibly YouTube) download + a full
# Whisper transcription in the background — expensive and proxy-bandwidth-heavy.
# Charge a fixed guard cost up front, settled when the background job finishes.
TRANSCRIBE_MINUTES = 2

# Stripe prices are resolved at runtime by these stable lookup_keys, so no price
# IDs need to be copied into env vars (they differ between test and live anyway).
SUBSCRIPTION_LOOKUP_KEYS = [
    "starter_monthly", "starter_yearly",
    "creator_monthly", "creator_yearly",
    "pro_monthly", "pro_yearly",
]
TOPUP_LOOKUP_KEYS = ["topup_60", "topup_200"]

# Queue priority per plan (lower dispatches first). BYOK / anonymous = 2.
PLAN_PRIORITY = {
    "pro": 0,
    "creator": 1,
    "starter": 1,
    "free": 2,
}

# Max simultaneous managed jobs per user, by plan.
PLAN_JOB_LIMIT = {
    "pro": 3,
    "creator": 2,
    "starter": 2,
    "free": 1,
}


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, "") or default)
    except ValueError:
        return default


# --- Usage limits ------------------------------------------------------------
# Each one sits well above what real accounts do; they only bound a runaway
# client or a script. All overridable by env.

# Thumbnail Studio sources: the longest video it accepts, and how its
# transcription is billed (one minute per this many source minutes, never less
# than TRANSCRIBE_MINUTES, so a typical video keeps its flat price).
THUMBNAIL_MAX_SOURCE_MINUTES = _env_int("THUMBNAIL_MAX_SOURCE_MINUTES", 180)
THUMBNAIL_SOURCE_MINUTES_PER_CHARGED = _env_int("THUMBNAIL_SOURCE_MINUTES_PER_CHARGED", 10)
# Title refinement and description calls (Gemini text) per user and hour, and
# refinements per Thumbnail Studio session.
THUMBNAIL_TEXT_CALLS_PER_HOUR = _env_int("THUMBNAIL_TEXT_CALLS_PER_HOUR", 60)
THUMBNAIL_REFINES_PER_SESSION = _env_int("THUMBNAIL_REFINES_PER_SESSION", 40)
THUMBNAIL_MESSAGE_MAX_CHARS = _env_int("THUMBNAIL_MESSAGE_MAX_CHARS", 2000)
THUMBNAIL_TITLE_MAX_CHARS = _env_int("THUMBNAIL_TITLE_MAX_CHARS", 300)
# Turns of the refinement chat sent back to the model (the oldest are dropped).
THUMBNAIL_CONVERSATION_MAX_TURNS = _env_int("THUMBNAIL_CONVERSATION_MAX_TURNS", 20)
# Transcript characters a description request sends (about 4 h of speech).
DESCRIBE_TRANSCRIPT_MAX_CHARS = _env_int("DESCRIBE_TRANSCRIPT_MAX_CHARS", 200000)

# Social posting through the managed Upload-Post account: posts per user and
# hour (a scheduled week is one post per clip). Posting is part of the paid
# plans; a free account gets this many posts in total to try it.
SOCIAL_POSTS_PER_HOUR = _env_int("SOCIAL_POSTS_PER_HOUR", 60)
FREE_SOCIAL_POSTS = _env_int("FREE_SOCIAL_POSTS", 3)

# Agent upload slots (POST /api/uploads) a user may hold open at once.
MAX_PENDING_UPLOADS_PER_USER = _env_int("MAX_PENDING_UPLOADS_PER_USER", 5)

# OAuth dynamic client registrations per client IP and hour. A client
# registers once per connection; this only stops one stuck in a loop.
OAUTH_REGISTRATIONS_PER_HOUR = _env_int("OAUTH_REGISTRATIONS_PER_HOUR", 60)

# SaaSShorts analysis: scripts per request and manual description length.
SAAS_MAX_SCRIPTS = _env_int("SAAS_MAX_SCRIPTS", 10)
SAAS_DESCRIPTION_MAX_CHARS = _env_int("SAAS_DESCRIPTION_MAX_CHARS", 10000)


class Settings:
    """Lazily-evaluated env-backed settings. Access attributes, not the class."""

    # Core
    @property
    def database_url(self) -> str:
        return os.environ.get("DATABASE_URL", "")

    @property
    def jwt_secret(self) -> str:
        return os.environ.get("JWT_SECRET", "")

    @property
    def frontend_url(self) -> str:
        return os.environ.get("FRONTEND_URL", "https://openshorts.app").rstrip("/")

    @property
    def public_api_url(self) -> str:
        """Absolute base of this API as the internet sees it.

        Needed by anything that has to put a working API link inside a message
        that leaves the browser — the unsubscribe link in an email, for one.
        Same variable app.py uses for the webhook payload's absolute URLs.
        """
        return os.environ.get("PUBLIC_API_URL", "").rstrip("/")

    @property
    def allowed_origins(self) -> list:
        raw = os.environ.get("ALLOWED_ORIGINS", "")
        origins = [o.strip() for o in raw.split(",") if o.strip()]
        return origins or [self.frontend_url]

    # Email (SMTP — e.g. Namecheap Private Email)
    @property
    def smtp_host(self) -> str:
        return os.environ.get("SMTP_HOST", "mail.privateemail.com")

    @property
    def smtp_port(self) -> int:
        return int(os.environ.get("SMTP_PORT", "465"))

    @property
    def smtp_user(self) -> str:
        return os.environ.get("SMTP_USER", "")

    @property
    def smtp_password(self) -> str:
        return os.environ.get("SMTP_PASSWORD", "")

    @property
    def smtp_configured(self) -> bool:
        return bool(self.smtp_user and self.smtp_password)

    @property
    def email_from(self) -> str:
        # Namecheap requires the From to be the authenticated mailbox.
        return os.environ.get("EMAIL_FROM") or (f"OpenShorts <{self.smtp_user}>" if self.smtp_user else "OpenShorts")

    @property
    def admin_email(self) -> str:
        # Where operational alerts (proxy out of credits, high failure rate) go.
        return os.environ.get("ADMIN_EMAIL", "")

    @property
    def openai_apps_challenge(self) -> str:
        # Domain-verification token from the ChatGPT app portal, served at
        # /.well-known/openai-apps-challenge (404 while unset).
        return os.environ.get("OPENAI_APPS_CHALLENGE", "").strip()

    @property
    def review_login_email(self) -> str:
        # App-directory reviewers cannot receive a magic link: this one account
        # signs in with REVIEW_LOGIN_PASSWORD instead. Off while either is unset.
        return os.environ.get("REVIEW_LOGIN_EMAIL", "").strip().lower()

    @property
    def review_login_password(self) -> str:
        return os.environ.get("REVIEW_LOGIN_PASSWORD", "")

    # Telegram (optional) — real-time admin alerts (purchases, churn, outages)
    @property
    def telegram_bot_token(self) -> str:
        return os.environ.get("TELEGRAM_BOT_TOKEN", "")

    @property
    def telegram_chat_id(self) -> str:
        return os.environ.get("TELEGRAM_CHAT_ID", "")

    @property
    def telegram_configured(self) -> bool:
        return bool(self.telegram_bot_token and self.telegram_chat_id)

    # Google OAuth
    @property
    def google_client_id(self) -> str:
        return os.environ.get("GOOGLE_CLIENT_ID", "")

    @property
    def google_client_secret(self) -> str:
        return os.environ.get("GOOGLE_CLIENT_SECRET", "")

    @property
    def google_auth_enabled(self) -> bool:
        return bool(self.google_client_id and self.google_client_secret)

    # Stripe
    @property
    def stripe_secret_key(self) -> str:
        return os.environ.get("STRIPE_SECRET_KEY", "")

    @property
    def stripe_webhook_secret(self) -> str:
        return os.environ.get("STRIPE_WEBHOOK_SECRET", "")

    # AgentLedger (aikount.com) — issues the legally valid Spanish invoice for
    # every Stripe charge; the account page lists them via /api/billing/invoices.
    @property
    def agentledger_api_url(self) -> str:
        return os.environ.get("AGENTLEDGER_API_URL", "https://api.aikount.com/api/v1").rstrip("/")

    @property
    def agentledger_api_key(self) -> str:
        return os.environ.get("AGENTLEDGER_API_KEY", "")

    @property
    def agentledger_treasury_id(self) -> str:
        # The Stripe Connect treasury in AgentLedger that holds OpenShorts' customers.
        return os.environ.get("AGENTLEDGER_TREASURY_ID", "720d3b70-3806-4c59-8729-2495b489a771")

    # Managed provider keys (server-owned, only handed to entitled users)
    @property
    def managed_gemini_key(self) -> str:
        return os.environ.get("MANAGED_GEMINI_API_KEY", "")

    @property
    def managed_upload_post_key(self) -> str:
        return os.environ.get("MANAGED_UPLOAD_POST_API_KEY", "")

    @property
    def openshorts_logo_url(self) -> str:
        return os.environ.get("OPENSHORTS_LOGO_URL", "https://openshorts.app/logo.png")

    # Cloudflare R2 (S3-compatible) — durable video library storage
    @property
    def r2_endpoint(self) -> str:
        return os.environ.get("R2_ENDPOINT", "")

    @property
    def r2_bucket(self) -> str:
        return os.environ.get("R2_BUCKET", "")

    @property
    def r2_access_key_id(self) -> str:
        return os.environ.get("R2_ACCESS_KEY_ID", "")

    @property
    def r2_secret_access_key(self) -> str:
        return os.environ.get("R2_SECRET_ACCESS_KEY", "")

    @property
    def r2_public_base(self) -> str:
        """Base URL of a custom domain bound to the R2 bucket, e.g.
        https://cdn.openshorts.app — unset means keep signing S3 URLs.

        The S3 endpoint (*.r2.cloudflarestorage.com) is not a browser-facing
        endpoint: measured 22-ago-2026, a presigned GET from Chrome on a Spanish
        residential line returns 503 on every attempt while the same URL with the
        same headers returns 206 at 11 MB/s from a datacenter. A custom domain
        serves the same objects through the normal Cloudflare edge, which both
        fixes that and puts the bytes on Cloudflare's network instead of a single
        long path out of Hetzner (measured to the same client: 90 KB/s).

        Objects under it are public: an unguessable key is the capability.
        /videos/{job_id}/... no longer works that way (signed links or the
        owner, app._authorize_media); doing the same here needs a Worker
        checking a signed token.
        """
        return os.environ.get("R2_PUBLIC_BASE", "").strip().rstrip("/")

    @property
    def r2_configured(self) -> bool:
        return bool(self.r2_endpoint and self.r2_bucket and self.r2_access_key_id
                    and self.r2_secret_access_key)


# Days a user's videos survive after their subscription ends (grace period).
VIDEO_RETENTION_GRACE_DAYS = 7

# Days a paid job's source video stays in R2 after its upload
# (cloud/source_archive.py), so the editor can still extend a cut or reframe and
# the source preview still plays once the server copy has aged out. 0 turns the
# archive off, and the next sweep then deletes whatever is still stored.
SOURCE_ARCHIVE_DAYS = max(0, _env_int("SOURCE_ARCHIVE_DAYS", 30))

# How long the erasure log (cloud/models.AccountDeletion) is kept. Matches the
# "rights declarations and related logs: up to 5 years" line in the privacy
# policy, which in turn matches the statute of limitations for civil claims.
DELETION_LOG_RETENTION_DAYS = 5 * 365


settings = Settings()


def validate_required():
    """Raise a clear error if mandatory cloud settings are missing.

    Called from ``cloud.setup`` at startup so a misconfigured deploy fails fast
    instead of erroring on the first paid request.
    """
    missing = []
    if not settings.database_url:
        missing.append("DATABASE_URL")
    if not settings.jwt_secret:
        missing.append("JWT_SECRET")
    if missing:
        raise RuntimeError(
            "BILLING_ENABLED is set but required settings are missing: "
            + ", ".join(missing)
            + ". Set them (see docker-compose.cloud.yml) or unset BILLING_ENABLED "
            "to run in self-hosted BYOK mode."
        )
