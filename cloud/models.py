"""SQLAlchemy models for cloud mode.

Money and quota live here, so the accounting tables (subscriptions, credit_topups,
usage_ledger) are designed for atomic, restart-safe metering — see cloud/metering.py.
"""
import uuid
from sqlalchemy import (
    Column, String, Integer, Numeric, Boolean, DateTime, ForeignKey, Text, func, Index,
    UniqueConstraint,
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
    # Set by the unsubscribe link in the only email we send that is a
    # commercial communication rather than a service notice (the out-of-minutes
    # upsell). LSSI art. 21.2 lets us mail our own customers about a similar
    # product, but only if every message carries a way out — so this column is
    # what that link writes, and cloud/emails.py refuses to send when it is set.
    # Schema bootstrap is create_all, which never ALTERs: see
    # cloud/database.init_engine for the additive statement that adds it to an
    # existing database.
    marketing_opt_out = Column(Boolean, nullable=False, server_default="false",
                               default=False)
    # Why this account gets no free monthly minutes, or NULL (the normal case).
    # Set at sign-up for an address erased in the last FREE_REDO_BLOCK_DAYS
    # ("recreated_after_deletion": deleting and re-registering handed out a
    # fresh 20 minutes), and for accounts whose mail domain turns out to be a
    # temp-mail front ("disposable_mx"), which the domain list alone cannot
    # see. Paid plans and top-ups are untouched; clearing it restores the free
    # plan. Added to existing databases by cloud/database._ADDITIVE_COLUMNS.
    free_plan_denied = Column(String(32), nullable=True)


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


class CancellationFeedback(Base):
    """Private cancellation intent, NOT evidence of a completed cancellation."""
    __tablename__ = "cancellation_feedback"
    id = Column(UUID(as_uuid=True), primary_key=True, default=_uuid)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"),
                     nullable=False, index=True)
    # Snapshot: subscriptions can be replaced later for this same user.
    stripe_subscription_id = Column(Text, nullable=False, index=True)
    reason = Column(String(32), nullable=True)
    comment = Column(String(2000), nullable=True)
    rating = Column(Integer, nullable=True)
    status = Column(String(20), nullable=False, default="intent")
    created_at = Column(DateTime(timezone=True), server_default=func.now())


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


class AccountDeletion(Base):
    """Proof that an account was erased, kept after the user row is gone.

    GDPR Art. 17 erasure has an accountability twin (Art. 5.2): if a former user
    later claims we never deleted their account, the only way to answer is a
    record that outlives the deletion. The identifying field is therefore a
    sha256 of the account email, which confirms "yes, this address was deleted
    on this date" without storing the address itself.

    ``reason`` is one label from ``account.DELETION_REASONS``, never free text:
    anything the user could type would land in a row that deliberately outlives
    their account, which is the opposite of what this row is for.

    ``stripe_customer_id`` is the one exception and it is deliberate: the
    invoices behind it must be kept for six years under Spanish commercial law,
    so the reference that lets us find them survives too (privacy policy §5).
    There is no FK to ``users`` — the whole point is that the row it would
    reference no longer exists.

    Rows are dropped after DELETION_LOG_RETENTION_DAYS by the retention sweeper,
    matching the "rights declarations and related logs: up to 5 years" line in
    the privacy policy.
    """
    __tablename__ = "account_deletions"
    id = Column(UUID(as_uuid=True), primary_key=True, default=_uuid)
    former_user_id = Column(Text, nullable=False)        # the old uuid, resolves to nothing now
    email_sha256 = Column(Text, nullable=False, index=True)
    stripe_customer_id = Column(Text, nullable=True)
    plan_at_deletion = Column(String(20), nullable=True)
    r2_objects_deleted = Column(Integer, nullable=True)
    reason = Column(String(32), nullable=True)           # one of account.DELETION_REASONS
    deleted_at = Column(DateTime(timezone=True), server_default=func.now())


class OAuthClient(Base):
    """An MCP client registered through OAuth dynamic client registration
    (RFC 7591): claude.ai, ChatGPT, Cursor... They are public clients (no
    secret): PKCE is what ties the authorization code to the party that
    started the flow. Rows are not user-owned — one registration serves every
    user of that client — so they survive account erasure."""
    __tablename__ = "oauth_clients"
    id = Column(Text, primary_key=True)               # client_id
    client_name = Column(Text, nullable=False)
    redirect_uris = Column(Text, nullable=False)      # JSON list
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class OAuthCode(Base):
    """A short-lived authorization code (10 min, single use). Only its sha256
    is stored; the code itself travels once, in the redirect back to the
    client. Redeeming it mints an ``osk_`` API key for the user, which is what
    the client keeps as its access token: the key shows up in the account
    page like any other, and revoking it there disconnects the client."""
    __tablename__ = "oauth_codes"
    code_hash = Column(Text, primary_key=True)
    client_id = Column(Text, ForeignKey("oauth_clients.id", ondelete="CASCADE"), nullable=False)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"),
                     nullable=False, index=True)
    redirect_uri = Column(Text, nullable=False)
    code_challenge = Column(Text, nullable=False)
    scope = Column(Text, nullable=True)
    expires_at = Column(DateTime(timezone=True), nullable=False)
    used_at = Column(DateTime(timezone=True), nullable=True)


class ProxyUsage(Base):
    """One row per event that put bytes through the per-GB paid proxy, or per
    download route decision worth keeping (the winner and why the free routes
    failed).

    The monthly proxy counter in app.py is in-memory and the container log
    rotates within the hour, so on 28-aug-2026 a $14 DataImpulse day could not
    be reconstructed at all. This is the durable trail: which job, which route
    won, how many paid bytes, and the failure text of every free attempt that
    was tried first. Not user-owned data (no FK to users): it is operational
    accounting and survives account erasure on purpose.
    """
    __tablename__ = "proxy_usage"
    id = Column(UUID(as_uuid=True), primary_key=True, default=_uuid)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), index=True)
    source = Column(String(16), nullable=False)          # download | probe
    job_id = Column(String(64), nullable=True, index=True)
    url_host = Column(String(120), nullable=True)
    route = Column(String(32), nullable=True)            # winning attempt label, or "none"
    paid_bytes = Column(Integer, nullable=False, default=0)
    detail = Column(JSONB, nullable=True)                # attempts: [{label, ok, bytes, error}]


class AutopilotSettings(Base):
    """Per-user switch for Autopilot: clip every new video on the user's
    connected YouTube channel, and optionally publish the best clips.

    The channel itself is not stored here. It is whatever YouTube account the
    user connected to their Upload-Post profile, read at poll time, so
    reconnecting a different channel just works and disconnecting stops it.

    ``enabled_at`` is the baseline: only videos published after it are clipped
    automatically, so switching Autopilot on never burns the month on the back
    catalogue. Older videos can still be clipped by hand from the same page.
    """
    __tablename__ = "autopilot_settings"
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"),
                     primary_key=True)
    enabled = Column(Boolean, nullable=False, default=False)
    enabled_at = Column(DateTime(timezone=True), nullable=True)
    # The user confirmed they own (or have rights to) the channel's content.
    # /api/process requires that attestation per job; Autopilot records it once.
    rights_ack_at = Column(DateTime(timezone=True), nullable=True)
    autopublish = Column(Boolean, nullable=False, default=False)
    publish_platforms = Column(JSONB, nullable=True)      # ["tiktok", "instagram", "youtube"]
    clips_to_publish = Column(Integer, nullable=False, default=3)
    max_minutes = Column(Integer, nullable=False, default=30)
    # Autopublish slot: one clip a day at this local hour, in this IANA zone.
    publish_hour = Column(Integer, nullable=False, default=17)
    timezone = Column(Text, nullable=True)
    last_checked_at = Column(DateTime(timezone=True), nullable=True)
    last_error = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())


class AutopilotRun(Base):
    """One channel video Autopilot picked up, and what happened to it.

    The (user_id, video_id) unique constraint is the dedupe: during a deploy two
    API containers run the poller at once, and only the one whose INSERT wins
    submits the job.
    """
    __tablename__ = "autopilot_runs"
    id = Column(UUID(as_uuid=True), primary_key=True, default=_uuid)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"),
                     nullable=False, index=True)
    video_id = Column(Text, nullable=False)
    video_url = Column(Text, nullable=True)
    video_title = Column(Text, nullable=True)
    thumbnail_url = Column(Text, nullable=True)
    published_at = Column(DateTime(timezone=True), nullable=True)
    trigger = Column(String(12), nullable=False, default="auto")   # auto | manual
    status = Column(String(16), nullable=False, default="queued")  # queued | processing | completed | failed | skipped
    reason = Column(Text, nullable=True)
    job_id = Column(Text, nullable=True, index=True)
    clips_count = Column(Integer, nullable=True)
    posted_count = Column(Integer, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    __table_args__ = (
        UniqueConstraint("user_id", "video_id", name="uq_autopilot_user_video"),
    )
