"""Minutes held by a queued or running video are shown apart from the spent ones.

A free video can wait a long time in a saturated queue with its minutes
reserved. The balance counts the reservation (it is not spendable twice), but
the account page and the MCP ``get_quota`` used to report it as "used", so a
customer whose video never started read their minutes as gone.

The ledger half runs against a real Postgres (``TEST_DATABASE_URL``, which CI
provides; it drops and recreates every table there).
"""
import asyncio
import os
import uuid
from datetime import datetime, timezone

import pytest

import mcp_server


# --------------------------------------------------------------------------- #
# MCP get_quota
# --------------------------------------------------------------------------- #
class _Resp:
    def __init__(self, data, status=200):
        self._data, self.status_code = data, status

    def json(self):
        return self._data


class _Client:
    def __init__(self, data):
        self.data = data

    async def get(self, path):
        assert path == "/api/me"
        return _Resp(self.data)


def _me(held):
    return {"plan": "free", "entitled": True, "upload_post_profile": None,
            "minutes": {"plan_allowance": 20.0, "plan_used": 17.0, "plan_held": held,
                        "held": held, "plan_remaining": 3.0, "topup_remaining": 0.0,
                        "remaining": 3.0}}


def test_get_quota_says_held_minutes_are_not_spent():
    out, is_error = asyncio.run(mcp_server._tool_get_quota(_Client(_me(17.0)), {}))
    assert not is_error
    assert out["minutes"]["held"] == 17.0
    assert "17 of these minutes are held" in out["held_note"]
    assert "come back" in out["held_note"]


def test_get_quota_without_held_minutes_has_no_note():
    out, _ = asyncio.run(mcp_server._tool_get_quota(_Client(_me(0.0)), {}))
    assert "held_note" not in out


# --------------------------------------------------------------------------- #
# Against Postgres
# --------------------------------------------------------------------------- #
DB_URL = os.environ.get("TEST_DATABASE_URL")
try:
    import asyncpg  # noqa: F401
    _HAVE_ASYNCPG = True
except ImportError:
    _HAVE_ASYNCPG = False

needs_db = pytest.mark.skipif(not (DB_URL and _HAVE_ASYNCPG),
                              reason="TEST_DATABASE_URL (Postgres) and asyncpg needed")


@pytest.fixture
def db():
    from sqlalchemy import text
    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
    from sqlalchemy.pool import NullPool
    from cloud import database, models  # noqa: F401

    engine = create_async_engine(DB_URL, poolclass=NullPool)

    async def init():
        async with engine.begin() as conn:
            await conn.execute(text("CREATE EXTENSION IF NOT EXISTS citext"))
            await conn.run_sync(database.Base.metadata.drop_all)
            await conn.run_sync(database.Base.metadata.create_all)
    asyncio.run(init())
    saved = database._engine, database._sessionmaker
    database._engine = engine
    database._sessionmaker = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)
    yield database
    database._engine, database._sessionmaker = saved
    asyncio.run(engine.dispose())


async def _free_user_with(rows):
    """A free (Google) account with ledger rows ``(minutes, status)``."""
    from cloud import database, metering
    from cloud.models import UsageLedger, User
    async with database.session() as s:
        async with s.begin():
            u = User(email=f"{uuid.uuid4().hex[:8]}@gmail.com", google_sub="g-" + uuid.uuid4().hex,
                     last_login_at=datetime.now(timezone.utc))
            s.add(u)
            await s.flush()
            for i, (minutes, status) in enumerate(rows):
                s.add(UsageLedger(user_id=u.id, job_id=f"job-{i}", job_type="process",
                                  minutes=minutes, minutes_from_plan=minutes,
                                  minutes_from_topup=0, status=status,
                                  period_end=metering.free_period_end()))
            return u.id


@needs_db
def test_balance_reports_the_queued_reservation_apart(db):
    from cloud import metering

    async def go():
        uid = await _free_user_with([(17, "reserved"), (2, "committed"), (5, "released")])
        return await metering.get_balance(uid)
    b = asyncio.run(go())
    # The reservation still counts against the balance...
    assert b["plan_used"] == 19.0
    assert b["remaining"] == 1.0
    # ...and is reported as held, not spent.
    assert b["plan_held"] == 17.0
    assert b["held"] == 17.0


@needs_db
def test_released_reservation_is_no_longer_held(db):
    from cloud import database, metering
    from cloud.models import UsageLedger
    from sqlalchemy import select

    async def go():
        uid = await _free_user_with([(17, "reserved")])
        async with database.session() as s:
            rid = (await s.execute(select(UsageLedger.id).where(
                UsageLedger.user_id == uid))).scalar_one()
        await metering.release_reservation(str(rid))
        return await metering.get_balance(uid)
    b = asyncio.run(go())
    assert b["held"] == 0.0 and b["plan_used"] == 0.0 and b["remaining"] == 20.0
