"""Central configuration for the optional cloud (paid / managed-keys) mode.

Everything here is read lazily from environment variables so that importing the
`cloud` package never has side effects. Nothing in this module is loaded unless
``BILLING_ENABLED`` is truthy (see ``cloud.is_enabled``).
"""
import os
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
# Gate the free allowance behind Google sign-in; magic-link-only accounts are
# disposable-email fodder and stay unentitled.
FREE_REQUIRES_GOOGLE = True
# Max processing jobs per free account per UTC day (abuse bound on GPU/proxy).
FREE_DAILY_JOBS = 3
# Same bound per client IP across free accounts. Generous on purpose: much of
# the audience is behind CGNAT (India/Indonesia share IPs among thousands).
FREE_DAILY_JOBS_PER_IP = 5
# Free users' clips expire from R2 after this many days (paid libraries are
# durable). Also an upgrade lever, mirroring OpusClip's 3-day free exports.
FREE_CLIP_RETENTION_DAYS = 7

# Free trial length (days) for new subscriptions. Trials are retired in favor
# of the free plan; the checkout only injects trial_period_days when > 0, and
# existing 'trialing' subscriptions are still honored until they convert or
# cancel (grandfathering).
TRIAL_DAYS = 0

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
HOOK_MINUTES = 1
RENDER_MINUTES = 3

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
    def r2_configured(self) -> bool:
        return bool(self.r2_endpoint and self.r2_bucket and self.r2_access_key_id
                    and self.r2_secret_access_key)


# Days a user's videos survive after their subscription ends (grace period).
VIDEO_RETENTION_GRACE_DAYS = 7


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
