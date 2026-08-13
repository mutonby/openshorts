"""SQLAlchemy models for cloud mode.

Money and quota live here, so the accounting tables (subscriptions, credit_topups,
usage_ledger) are designed for atomic, restart-safe metering — see cloud/metering.py.
"""
import uuid
from sqlalchemy import (
    Column, String, Integer, Numeric, Boolean, DateTime, ForeignKey, Text, func, Index,
)
from sqlalchemy.dialects.postgresql import UUID, CITEXT, JSONB

from .database import Base


def _uuid():
    return uuid.uuid4()


class User(Base):
    __tablename__ = "users"
    id = Column(UUID(as_uuid=True), primary_key=True, default=_uuid)
    email = Column(CITEXT, unique=True, nullable=False)
    google_sub = Column(Text, unique=True, nullable=True)
    stripe_customer_id = Column(Text, unique=True, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    last_login_at = Column(DateTime(timezone=True), nullable=True)


class MagicLinkToken(Base):
    __tablename__ = "magic_link_tokens"
    id = Column(UUID(as_uuid=True), primary_key=True, default=_uuid)
    email = Column(CITEXT, nullable=False)
    token_hash = Column(Text, unique=True, nullable=False)  # sha256 of the raw token
    expires_at = Column(DateTime(timezone=True), nullable=False)
    used_at = Column(DateTime(timezone=True), nullable=True)
    request_ip = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    __table_args__ = (
        Index("ix_magic_email_created", "email", "created_at"),
        # NOTE: schema is create_all-only; on an existing DB this index must be
        # applied by hand: CREATE INDEX IF NOT EXISTS ix_magic_ip_created
        #   ON magic_link_tokens (request_ip, created_at);
        Index("ix_magic_ip_created", "request_ip", "created_at"),
    )


class Subscription(Base):
    __tablename__ = "subscriptions"
    id = Column(UUID(as_uuid=True), primary_key=True, default=_uuid)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"),
                     unique=True, nullable=False)  # one active sub per user
    stripe_subscription_id = Column(Text, unique=True, nullable=False)
    stripe_price_id = Column(Text, nullable=True)
    plan = Column(String(20), nullable=False)       # starter | creator | pro
    interval = Column(String(10), nullable=False)   # month | year
    status = Column(String(20), nullable=False)     # active | trialing | past_due | canceled | incomplete
    minutes_per_period = Column(Integer, nullable=False)
    current_period_start = Column(DateTime(timezone=True), nullable=False)
    current_period_end = Column(DateTime(timezone=True), nullable=False)
    cancel_at_period_end = Column(Boolean, nullable=False, default=False)
    last_event_at = Column(DateTime(timezone=True), nullable=True)  # ordering guard for webhooks
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())


class CreditTopup(Base):
    __tablename__ = "credit_topups"
    id = Column(UUID(as_uuid=True), primary_key=True, default=_uuid)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"),
                     nullable=False, index=True)
    stripe_session_id = Column(Text, unique=True, nullable=True)  # idempotency for the webhook
    minutes_total = Column(Integer, nullable=False)
    minutes_consumed = Column(Numeric(10, 2), nullable=False, default=0)  # FIFO drain target
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class UsageLedger(Base):
    __tablename__ = "usage_ledger"
    id = Column(UUID(as_uuid=True), primary_key=True, default=_uuid)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    job_id = Column(Text, nullable=False)
    job_type = Column(String(20), nullable=False, default="process")
    minutes = Column(Numeric(10, 2), nullable=False)             # total reserved
    minutes_from_plan = Column(Numeric(10, 2), nullable=False, default=0)
    minutes_from_topup = Column(Numeric(10, 2), nullable=False, default=0)
    # [{topup_id, minutes}] — exact FIFO allocation, so release can refund precisely.
    topup_allocations = Column(JSONB, nullable=True)
    status = Column(String(12), nullable=False, default="reserved")  # reserved | committed | released
    period_end = Column(DateTime(timezone=True), nullable=True)   # sub period this counts against
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    __table_args__ = (
        Index("ix_usage_user_status", "user_id", "status"),
        Index("ix_usage_user_period_status", "user_id", "period_end", "status"),
    )


class SignupAttribution(Base):
    """Where a user came from, captured once at sign-up (first touch wins).

    Its own table rather than columns on ``users`` because the schema bootstrap
    is ``create_all`` (see cloud/database.py), which creates missing tables but
    never ALTERs an existing one — a new table lands on deploy with no migration.

    ``referrer_host`` is the grouping key ("github.com", "www.youtube.com",
    "google"); the full ``referrer`` is kept for the long tail. Rows are only
    written for users whose account is minutes old, so returning users from
    before this shipped never get a misleading "signup" source.
    """
    __tablename__ = "signup_attribution"
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"),
                     primary_key=True)
    referrer = Column(Text, nullable=True)
    referrer_host = Column(Text, nullable=True)
    landing_path = Column(Text, nullable=True)
    utm_source = Column(Text, nullable=True)
    utm_medium = Column(Text, nullable=True)
    utm_campaign = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    __table_args__ = (
        Index("ix_attrib_host", "referrer_host"),
        Index("ix_attrib_utm_source", "utm_source"),
    )


class ApiKey(Base):
    """A user-issued ``osk_...`` token for programmatic access (MCP, scripts, CI).

    Only the sha256 of the raw token is stored — the raw value is shown once at
    creation and never again. ``prefix`` keeps the first characters so the UI
    can tell keys apart. Revocation is a timestamp rather than a delete so a
    leaked-then-revoked key stays visible in the user's list with its history.

    Its own table (not columns on ``users``) because the schema bootstrap is
    ``create_all`` — see [SignupAttribution] above for the same reasoning.
    """
    __tablename__ = "api_keys"
    id = Column(UUID(as_uuid=True), primary_key=True, default=_uuid)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"),
                     nullable=False, index=True)
    name = Column(Text, nullable=False)
    key_hash = Column(Text, unique=True, nullable=False)  # sha256 of the raw osk_ token
    prefix = Column(Text, nullable=False)                 # e.g. "osk_a1b2c3" (display only)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    last_used_at = Column(DateTime(timezone=True), nullable=True)
    revoked_at = Column(DateTime(timezone=True), nullable=True)


class UploadPostProfile(Base):
    __tablename__ = "upload_post_profiles"
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"),
                     primary_key=True)
    profile_username = Column(Text, unique=True, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class UserVideo(Base):
    __tablename__ = "user_videos"
    id = Column(UUID(as_uuid=True), primary_key=True, default=_uuid)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"),
                     nullable=False, index=True)
    job_id = Column(Text, nullable=False)
    clip_index = Column(Integer, nullable=True)
    r2_key = Column(Text, nullable=False)
    title = Column(Text, nullable=True)
    size_bytes = Column(Integer, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class ClipExpiryWarning(Base):
    """One row per clip already covered by an expiry-warning email.

    De-duplication for ``videos.warn_free_expiring`` used to be a process-local
    set, which meant every API restart re-armed the warning for clips that had
    already been warned about. The warning window is a full day and the sweep
    runs every six hours, so any deploy inside that window sent the same user a
    second "your clips will be deleted tomorrow" email. Persisting the state
    fixes that: a restart no longer forgets who has been told.

    Its own table rather than a column on ``user_videos`` because the schema
    bootstrap is ``create_all`` (see cloud/database.py), which creates missing
    tables but never ALTERs an existing one — see [SignupAttribution] above for
    the same reasoning. The CASCADE means rows disappear on their own when
    ``purge_free_expired`` deletes the clip they refer to, so this never needs
    its own cleanup pass.
    """
    __tablename__ = "clip_expiry_warnings"
    video_id = Column(UUID(as_uuid=True),
                      ForeignKey("user_videos.id", ondelete="CASCADE"),
                      primary_key=True)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"),
                     nullable=False, index=True)
    warned_at = Column(DateTime(timezone=True), server_default=func.now())


class Project(Base):
    """One re-openable project per completed job.

    The metadata JSON in R2 (``metadata_r2_key``) is the source of truth for the
    clips + transcript; ``state`` holds only what lives outside that file: the
    browser-side Remotion layers and the current server file per clip.
    ``state`` schema: {"v": 1, "clips": [{"index", "original_file",
    "server_file", "active_layers"}]}.
    """
    __tablename__ = "projects"
    id = Column(UUID(as_uuid=True), primary_key=True, default=_uuid)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"),
                     nullable=False, index=True)
    job_id = Column(Text, unique=True, nullable=False)
    title = Column(Text, nullable=True)
    metadata_r2_key = Column(Text, nullable=False)
    state = Column(JSONB, nullable=True)
    size_bytes = Column(Integer, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())


class StripeEvent(Base):
    __tablename__ = "stripe_events"
    id = Column(Text, primary_key=True)  # Stripe event.id — dedupe key
    type = Column(Text, nullable=True)
    created = Column(DateTime(timezone=True), nullable=True)
    processed_at = Column(DateTime(timezone=True), server_default=func.now())
