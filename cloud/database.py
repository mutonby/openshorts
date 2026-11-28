"""Async SQLAlchemy engine + session management for cloud mode."""
import re

from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from sqlalchemy.orm import declarative_base

from .config import DEFAULT_TENANT_BRAND, DEFAULT_TENANT_ID, settings

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
    # Firebase identity, used by Phase 2 ID-token verification and by tenant
    # membership/erasure once an account has signed in that way.
    "ALTER TABLE users ADD COLUMN IF NOT EXISTS firebase_uid TEXT",
    "CREATE UNIQUE INDEX IF NOT EXISTS ux_users_firebase_uid ON users (firebase_uid)",
    # The per-network first-video check joins first_video_grants on job_id.
    "CREATE INDEX IF NOT EXISTS ix_usage_job_id ON usage_ledger (job_id)",
    # cancellation_feedback shipped without it; rows before it are all cancels.
    "ALTER TABLE cancellation_feedback ADD COLUMN IF NOT EXISTS outcome VARCHAR(16)",
    # Scope of an OAuth-issued API key (cloud/scopes.py); NULL = hand-made key.
    "ALTER TABLE api_keys ADD COLUMN IF NOT EXISTS scope TEXT",
    # Month on hold from the cancel flow (cloud/cancellation, config.ON_HOLD).
    "ALTER TABLE subscriptions ADD COLUMN IF NOT EXISTS hold_from TIMESTAMPTZ",
    "ALTER TABLE subscriptions ADD COLUMN IF NOT EXISTS hold_until TIMESTAMPTZ",
    # Tenant scope. Added nullable first: existing rows are backfilled by
    # _tenant_schema_steps below before the columns become mandatory.
    "ALTER TABLE user_videos ADD COLUMN IF NOT EXISTS tenant_id UUID",
    "ALTER TABLE projects ADD COLUMN IF NOT EXISTS tenant_id UUID",
    "ALTER TABLE upload_post_profiles ADD COLUMN IF NOT EXISTS tenant_id UUID",
    "ALTER TABLE usage_ledger ADD COLUMN IF NOT EXISTS tenant_reservation_id TEXT",
)

# Existing media tables whose rows must belong to a tenant. The FK is RESTRICT
# so deleting a tenant with data fails loudly instead of orphaning clips.
_TENANT_SCOPED_TABLES = ("user_videos", "projects", "upload_post_profiles")


def _pool_options(url: str) -> dict:
    """Postgres pool sizing. SQLAlchemy's default (5 + 10 overflow, 30 s wait)
    ran full on 16-nov with a busy host, and each request then waited 30 s for
    a connection before answering 500. A bigger pool and a short wait: even
    three instances overlapping in quick deploys (30 each) fit in Postgres' 100.
    Env: DB_POOL_SIZE, DB_MAX_OVERFLOW, DB_POOL_TIMEOUT.

    A front process of a split API (api_role, serve.py) is one of several, so
    it keeps a small pool of its own: FRONT_DB_POOL_SIZE (3) kept open and up
    to FRONT_DB_MAX_OVERFLOW (7) more while busy, closed again when returned.
    Size Postgres' max_connections for both containers of a deploy:
    2 x (engine pool + API_FRONT_WORKERS x front pool)."""
    if not url.startswith("postgresql"):
        return {}
    import os
    import api_role

    def _env(name, default):
        try:
            return max(0, int(os.environ.get(name, default)))
        except ValueError:
            return default
    if api_role.is_front():
        return {"pool_size": _env("FRONT_DB_POOL_SIZE", 3),
                "max_overflow": _env("FRONT_DB_MAX_OVERFLOW", 7),
                "pool_timeout": _env("DB_POOL_TIMEOUT", 10)}
    return {"pool_size": _env("DB_POOL_SIZE", 15),
            "max_overflow": _env("DB_MAX_OVERFLOW", 15),
            "pool_timeout": _env("DB_POOL_TIMEOUT", 10)}


async def init_engine(create_schema: bool = True):
    """Create the async engine + sessionmaker and ensure the schema exists.

    Uses ``create_all`` for zero-friction boot (additive-only schema in v1).
    Alembic (see ``alembic/``) is available for controlled migrations later.
    ``create_schema=False`` (the front processes of a split API) only opens
    the pool: the engine of the same container builds the schema, and N
    processes racing on CREATE TABLE at boot is what that avoids.
    """
    global _engine, _sessionmaker
    if _engine is not None:
        return
    # Import models so their tables register on Base.metadata before create_all.
    from . import models  # noqa: F401

    _engine = create_async_engine(settings.database_url, pool_pre_ping=True,
                                  **_pool_options(settings.database_url))
    _sessionmaker = async_sessionmaker(_engine, expire_on_commit=False, class_=AsyncSession)
    if not create_schema:
        return

    async with _engine.begin() as conn:
        # Case-insensitive email uniqueness.
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS citext"))
        await conn.run_sync(Base.metadata.create_all)
        # create_all creates missing TABLES and never ALTERs an existing one, so
        # a column added to a model that already shipped exists in the code and
        # not in the database. Each statement here is additive and idempotent;
        # keep them that way, and prefer Alembic once one of them is not.
        await _additive_steps(conn)
        await _tenant_schema_steps(conn)


# An ALTER TABLE takes the table's exclusive lock even when IF NOT EXISTS
# makes it a no-op, and every instance runs these at boot: during a deploy the
# new container's ALTER queued behind the old one's live reads, every read
# after it queued behind the ALTER, and the ones holding a lock on another
# table deadlocked with it (500s on /api/me, measured 18-nov-2026 in a local
# handover under load). So a step whose column or index is already there is
# skipped after a catalog look (no lock), and one that is really needed waits
# at most ADDITIVE_LOCK_TIMEOUT for its lock instead of stalling traffic; it
# is tried again at the next boot.
ADDITIVE_LOCK_TIMEOUT = "5s"
_ADD_COLUMN = re.compile(r"ALTER TABLE (\w+) ADD COLUMN IF NOT EXISTS (\w+)", re.I)
_ADD_INDEX = re.compile(r"CREATE (UNIQUE )?INDEX IF NOT EXISTS (\w+) ON (\w+)", re.I)


async def _already_there(conn, statement) -> bool:
    m = _ADD_COLUMN.match(statement)
    if m:
        row = (await conn.execute(text(
            "SELECT 1 FROM information_schema.columns WHERE table_schema = current_schema() "
            "AND table_name = :t AND column_name = :c"), {"t": m[1], "c": m[2]})).first()
        return row is not None
    m = _ADD_INDEX.match(statement)
    if m:
        row = (await conn.execute(text(
            "SELECT 1 FROM pg_indexes WHERE schemaname = current_schema() AND indexname = :i"),
            {"i": m[2]})).first()
        return row is not None
    return False


async def _additive_steps(conn):
    postgres = conn.dialect.name == "postgresql"
    for statement in _ADDITIVE_COLUMNS:
        try:
            if postgres:
                if await _already_there(conn, statement):
                    continue
                async with conn.begin_nested():
                    await conn.execute(text(f"SET LOCAL lock_timeout = '{ADDITIVE_LOCK_TIMEOUT}'"))
                    await conn.execute(text(statement))
                    await conn.execute(text("SET LOCAL lock_timeout = 0"))
            else:
                await conn.execute(text(statement))
        except Exception as e:  # pragma: no cover - depends on the server
            print(f"⚠️  Additive schema step failed ({statement}): {e}")


async def _constraint_exists(conn, table: str, name: str) -> bool:
    row = (await conn.execute(text(
        "SELECT 1 FROM pg_constraint WHERE conname = :n AND conrelid = to_regclass(:t)"),
        {"n": name, "t": table})).first()
    return row is not None


async def _column_nullable(conn, table: str, column: str) -> bool:
    row = (await conn.execute(text(
        "SELECT is_nullable FROM information_schema.columns "
        "WHERE table_schema = current_schema() AND table_name = :t AND column_name = :c"),
        {"t": table, "c": column})).first()
    return row is None or row[0] == "YES"


async def _tenant_schema_steps(conn):
    """Finish tenant scoping on databases that predate the white-label schema.

    ``create_all`` adds the new tables and the nullable columns; this backfills
    them with the default tenant, adds the FK/index, then makes the columns
    mandatory. Each check queries the catalog first so a healthy boot takes no
    table lock at all (see the lock note above ``ADDITIVE_LOCK_TIMEOUT``).
    """
    if conn.dialect.name != "postgresql":
        return
    await conn.execute(text(
        "INSERT INTO tenants (id, brand_name, credit_balance) "
        "VALUES (:id, :brand, 0) ON CONFLICT (id) DO NOTHING"),
        {"id": DEFAULT_TENANT_ID, "brand": DEFAULT_TENANT_BRAND})
    for table in _TENANT_SCOPED_TABLES:
        fk_name = f"fk_{table}_tenant_id_tenants"
        index_stmt = f"CREATE INDEX IF NOT EXISTS ix_{table}_tenant_id ON {table} (tenant_id)"
        try:
            if (await _column_nullable(conn, table, "tenant_id")) is False:
                if (await _constraint_exists(conn, table, fk_name)) and \
                        (await _already_there(conn, index_stmt)):
                    continue
            async with conn.begin_nested():
                await conn.execute(text(f"SET LOCAL lock_timeout = '{ADDITIVE_LOCK_TIMEOUT}'"))
                await conn.execute(text(
                    f"UPDATE {table} SET tenant_id = :id WHERE tenant_id IS NULL"),
                    {"id": DEFAULT_TENANT_ID})
                if not await _constraint_exists(conn, table, fk_name):
                    await conn.execute(text(
                        f"ALTER TABLE {table} ADD CONSTRAINT {fk_name} "
                        "FOREIGN KEY (tenant_id) REFERENCES tenants(id) ON DELETE RESTRICT"))
                if not await _already_there(conn, index_stmt):
                    await conn.execute(text(index_stmt))
                if await _column_nullable(conn, table, "tenant_id"):
                    await conn.execute(text(
                        f"ALTER TABLE {table} ALTER COLUMN tenant_id SET NOT NULL"))
                await conn.execute(text("SET LOCAL lock_timeout = 0"))
        except Exception as e:  # pragma: no cover - depends on the server
            print(f"⚠️  Tenant schema step failed for {table}: {e}")


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
