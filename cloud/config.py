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

# Free trial length (days) for new subscriptions. Card required; auto-charges at
# the end. No minute cap during the trial — the card + Stripe Radar deter abuse.
TRIAL_DAYS = 3

# Gemini IMAGE generation (thumbnails) is the one expensive managed Gemini call
# (~$0.04/image, batch of ~3). It isn't naturally minute-metered, so each
# thumbnail generation batch consumes this many minutes from the plan quota —
# roughly matching its cost to the per-minute economics. Text (titles/desc) is free.
THUMBNAIL_MINUTES = 3

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
}

# Max simultaneous managed jobs per user, by plan.
PLAN_JOB_LIMIT = {
    "pro": 3,
    "creator": 2,
    "starter": 2,
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

    # Email (Resend)
    @property
    def resend_api_key(self) -> str:
        return os.environ.get("RESEND_API_KEY", "")

    @property
    def email_from(self) -> str:
        return os.environ.get("EMAIL_FROM", "OpenShorts <login@openshorts.app>")

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
