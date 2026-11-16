"""Async SQLAlchemy engine + session management for cloud mode."""
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from sqlalchemy.orm import declarative_base

from .config import settings

Base = declarative_base()

_engine = None
_sessionmaker = None

# Columns added to tables that already exist in production. ADD COLUMN IF NOT
# EXISTS is a no-op on a database that already has them and on a fresh one that
# create_all just built.
_ADDITIVE_COLUMNS = (
    "ALTER TABLE users ADD COLUMN IF NOT EXISTS "
    "marketing_opt_out BOOLEAN NOT NULL DEFAULT false",
    # Nullable, no default: a catalog-only change, no table rewrite.
    "ALTER TABLE users ADD COLUMN IF NOT EXISTS free_plan_denied VARCHAR(32)",
    # Account ban set by the support back-office (cloud/support_ops.py).
    "ALTER TABLE users ADD COLUMN IF NOT EXISTS ban_reason TEXT",
    "ALTER TABLE users ADD COLUMN IF NOT EXISTS banned_at TIMESTAMPTZ",
    # The per-network first-video check joins first_video_grants on job_id.
    "CREATE INDEX IF NOT EXISTS ix_usage_job_id ON usage_ledger (job_id)",
    # cancellation_feedback shipped without it; rows before it are all cancels.
    "ALTER TABLE cancellation_feedback ADD COLUMN IF NOT EXISTS outcome VARCHAR(16)",
    # Scope of an OAuth-issued API key (cloud/scopes.py); NULL = hand-made key.
    "ALTER TABLE api_keys ADD COLUMN IF NOT EXISTS scope TEXT",
    # Month on hold from the cancel flow (cloud/cancellation, config.ON_HOLD).
    "ALTER TABLE subscriptions ADD COLUMN IF NOT EXISTS hold_from TIMESTAMPTZ",
    "ALTER TABLE subscriptions ADD COLUMN IF NOT EXISTS hold_until TIMESTAMPTZ",
)


def _pool_options(url: str) -> dict:
    """Postgres pool sizing. SQLAlchemy's default (5 + 10 overflow, 30 s wait)
    ran full on 16-nov with a busy host, and each request then waited 30 s for
    a connection before answering 500. A bigger pool and a short wait: even
    three instances overlapping in quick deploys (30 each) fit in Postgres' 100.
    Env: DB_POOL_SIZE, DB_MAX_OVERFLOW, DB_POOL_TIMEOUT."""
    if not url.startswith("postgresql"):
        return {}
    import os

    def _env(name, default):
        try:
            return max(0, int(os.environ.get(name, default)))
        except ValueError:
            return default
    return {"pool_size": _env("DB_POOL_SIZE", 15),
            "max_overflow": _env("DB_MAX_OVERFLOW", 15),
            "pool_timeout": _env("DB_POOL_TIMEOUT", 10)}


async def init_engine():
    """Create the async engine + sessionmaker and ensure the schema exists.

    Uses ``create_all`` for zero-friction boot (additive-only schema in v1).
    Alembic (see ``alembic/``) is available for controlled migrations later.
    """
    global _engine, _sessionmaker
    if _engine is not None:
        return
    # Import models so their tables register on Base.metadata before create_all.
    from . import models  # noqa: F401

    _engine = create_async_engine(settings.database_url, pool_pre_ping=True,
                                  **_pool_options(settings.database_url))
    _sessionmaker = async_sessionmaker(_engine, expire_on_commit=False, class_=AsyncSession)

    async with _engine.begin() as conn:
        # Case-insensitive email uniqueness.
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS citext"))
        await conn.run_sync(Base.metadata.create_all)
        # create_all creates missing TABLES and never ALTERs an existing one, so
        # a column added to a model that already shipped exists in the code and
        # not in the database. Each statement here is additive and idempotent;
        # keep them that way, and prefer Alembic once one of them is not.
        for statement in _ADDITIVE_COLUMNS:
            try:
                await conn.execute(text(statement))
            except Exception as e:  # pragma: no cover - depends on the server
                print(f"⚠️  Additive schema step failed ({statement}): {e}")


def get_sessionmaker():
    if _sessionmaker is None:
        raise RuntimeError("Cloud DB engine not initialized — call init_engine() first.")
    return _sessionmaker


def session():
    """Return a new AsyncSession context manager (for use outside FastAPI deps)."""
    return get_sessionmaker()()


async def get_db():
    """FastAPI dependency yielding an AsyncSession."""
    async with get_sessionmaker()() as s:
        yield s
