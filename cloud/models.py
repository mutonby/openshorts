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
    __table_args__ = (Index("ix_magic_email_created", "email", "created_at"),)


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


class UploadPostProfile(Base):
    __tablename__ = "upload_post_profiles"
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"),
                     primary_key=True)
    profile_username = Column(Text, unique=True, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class StripeEvent(Base):
    __tablename__ = "stripe_events"
    id = Column(Text, primary_key=True)  # Stripe event.id — dedupe key
    type = Column(Text, nullable=True)
    created = Column(DateTime(timezone=True), nullable=True)
    processed_at = Column(DateTime(timezone=True), server_default=func.now())
