"""Async SQLAlchemy engine + session management for cloud mode."""
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from sqlalchemy.orm import declarative_base

from .config import settings

Base = declarative_base()

_engine = None
_sessionmaker = None


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

    _engine = create_async_engine(settings.database_url, pool_pre_ping=True)
    _sessionmaker = async_sessionmaker(_engine, expire_on_commit=False, class_=AsyncSession)

    async with _engine.begin() as conn:
        # Case-insensitive email uniqueness.
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS citext"))
        await conn.run_sync(Base.metadata.create_all)


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
