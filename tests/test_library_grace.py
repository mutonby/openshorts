"""A plan that ended keeps its library 7 more days (cloud/videos.py sweeps).

The Terms (section 13), the privacy policy and the History tab all say the
same thing: a paid library is kept while the subscription is active and
deleted 7 days after it ends. The code only did that for accounts that cannot
fall back to the free plan. Everybody else read as a plain free account the
moment the plan ended, and the free sweep (every clip older than 7 days) took
the whole library on its next pass: hours after the end, not 7 days.

The first tests pin the rule with stand-ins; the last one runs the real sweep
against Postgres when ``TEST_DATABASE_URL`` is set (CI).
"""
import asyncio
import os
import uuid
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

metering = pytest.importorskip("cloud.metering")
videos = pytest.importorskip("cloud.videos")
from cloud import config  # noqa: E402
from cloud.models import Project, UserVideo  # noqa: E402

UTC = timezone.utc
NOW = datetime(2026, 12, 22, 16, 0, tzinfo=UTC)
GRACE = config.VIDEO_RETENTION_GRACE_DAYS


def _sub(status, *, end, last_event=None):
    return SimpleNamespace(status=status, current_period_end=end, last_event_at=last_event)


# --------------------------------------------------------------------------- #
# When a plan ended
# --------------------------------------------------------------------------- #
class TestPlanEndedAt:
    def test_a_live_plan_has_not_ended(self):
        row = _sub("active", end=NOW + timedelta(days=10), last_event=NOW - timedelta(days=20))
        assert metering.plan_ended_at(row, NOW) is None
        assert metering.plan_ended_at(_sub("trialing", end=NOW + timedelta(days=1)), NOW) is None

    def test_cancelled_at_period_end_ends_with_the_period(self):
        end = NOW - timedelta(hours=5)
        row = _sub("canceled", end=end, last_event=end + timedelta(seconds=3))
        assert metering.plan_ended_at(row, NOW) == end

    def test_cancelled_on_the_spot_ends_then_not_at_the_paid_period(self):
        # A refund: Stripe cancels now, the period it had paid for runs on.
        cancelled = NOW - timedelta(days=2)
        row = _sub("canceled", end=NOW + timedelta(days=24), last_event=cancelled)
        assert metering.plan_ended_at(row, NOW) == cancelled

    def test_a_renewal_that_failed_ends_at_the_failure(self):
        failed = NOW - timedelta(days=1)
        row = _sub("past_due", end=NOW + timedelta(days=29), last_event=failed)
        assert metering.plan_ended_at(row, NOW) == failed

    def test_a_live_row_whose_period_is_over_ended_with_it(self):
        # The webhook that should have closed it never came.
        end = NOW - timedelta(days=3)
        assert metering.plan_ended_at(_sub("active", end=end), NOW) == end

    def test_a_checkout_that_never_charged_was_never_a_plan(self):
        for status in ("incomplete", "incomplete_expired"):
            row = _sub(status, end=NOW + timedelta(days=30), last_event=NOW - timedelta(hours=1))
            assert metering.plan_ended_at(row, NOW) is None
        assert metering.plan_ended_at(None, NOW) is None

    def test_a_month_on_hold_is_not_an_end(self):
        row = _sub(config.ON_HOLD, end=NOW + timedelta(days=5), last_event=NOW - timedelta(days=9))
        assert metering.plan_ended_at(row, NOW) is None


def _grace(row, now):
    async def run():
        async def _row(_session, _uid):
            return row
        saved = metering._subscription_row
        metering._subscription_row = _row
        try:
            return await metering.in_paid_grace(None, "u", GRACE, now)
        finally:
            metering._subscription_row = saved
    return asyncio.run(run())


def test_the_grace_is_seven_days_from_the_end_to_the_second():
    assert GRACE == 7                      # what the Terms and the privacy policy say
    end = datetime(2026, 12, 22, 10, 0, 30, tzinfo=UTC)
    row = _sub("canceled", end=end, last_event=end)
    assert _grace(row, end + timedelta(hours=6))
    assert _grace(row, end + timedelta(days=7) - timedelta(seconds=1))
    assert not _grace(row, end + timedelta(days=7))
    assert not _grace(None, end)           # never subscribed: the free rule alone
    assert not _grace(_sub("incomplete_expired", end=end, last_event=end), end)


# --------------------------------------------------------------------------- #
# The sweeps, with a stand-in database and bucket
# --------------------------------------------------------------------------- #
class _Result:
    def __init__(self, rows):
        self._rows = list(rows)

    def scalars(self):
        return iter(self._rows)


class _Library:
    """One user's clips and projects, the bucket behind them and what the
    sweeps did to them."""

    def __init__(self, user_id, clip_dates):
        self.user_id = user_id
        self.videos = [SimpleNamespace(id=uuid.uuid4(), user_id=user_id, created_at=d,
                                       r2_key=f"users/{user_id}/job-{i}/clip.mp4")
                       for i, d in enumerate(clip_dates)]
        self.projects = [SimpleNamespace(id=uuid.uuid4(), user_id=user_id, job_id=f"job-{i}",
                                         updated_at=d, metadata_r2_key=f"users/{user_id}/job-{i}/metadata.json")
                         for i, d in enumerate(clip_dates)]
        self.bucket = {v.r2_key for v in self.videos} | {p.metadata_r2_key for p in self.projects}
        self.deleted_tables = []
        self.warned = []

    def session(self):
        lib = self

        class _Session:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *_exc):
                return False

            def begin(self):
                return self

            def add(self, row):
                lib.warned.append(row)

            async def get(self, _model, _key):
                return SimpleNamespace(email="k@example.com")

            async def execute(self, stmt):
                if stmt.is_delete:
                    lib.deleted_tables.append(stmt.table.name)
                    return _Result([])
                entity = stmt.column_descriptions[0]["entity"]
                if entity is UserVideo:
                    return _Result(lib.videos)
                if entity is Project:
                    return _Result(lib.projects)
                return _Result([])           # ClipExpiryWarning.video_id: none yet
        return _Session()


def _sweep(monkeypatch, lib, row, now, sweep=None):
    async def _free(_session, _uid):
        return True

    async def _row(_session, _uid):
        return row

    monkeypatch.setattr(metering, "_now", lambda: now)
    monkeypatch.setattr(metering, "is_free_user", _free)
    monkeypatch.setattr(metering, "_subscription_row", _row)
    monkeypatch.setattr(videos.database, "session", lib.session)
    monkeypatch.setattr(videos.storage, "delete_key", lib.bucket.discard)
    monkeypatch.setattr(videos.storage, "list_keys",
                        lambda prefix: [k for k in sorted(lib.bucket) if k.startswith(prefix)])
    asyncio.run((sweep or videos.purge_free_expired)())


# The usual shape: clips made in the first days of a monthly plan that was
# cancelled at period end. A month later every one of them is over 7 days old.
PLAN_END = datetime(2026, 12, 22, 10, 0, tzinfo=UTC)
CLIPS = [datetime(2026, 11, 22, 12, 0, tzinfo=UTC), datetime(2026, 11, 23, 9, 30, tzinfo=UTC)]
ENDED = _sub("canceled", end=PLAN_END, last_event=PLAN_END + timedelta(seconds=2))


def test_the_first_sweep_after_the_plan_ends_deletes_nothing(monkeypatch):
    lib = _Library("u-1", CLIPS)
    kept = set(lib.bucket)
    _sweep(monkeypatch, lib, ENDED, PLAN_END + timedelta(hours=5, minutes=40))
    assert lib.bucket == kept and lib.deleted_tables == []


def test_the_library_is_still_there_on_the_seventh_day(monkeypatch):
    lib = _Library("u-1", CLIPS)
    kept = set(lib.bucket)
    _sweep(monkeypatch, lib, ENDED, PLAN_END + timedelta(days=6, hours=23))
    assert lib.bucket == kept and lib.deleted_tables == []


def test_seven_days_after_the_end_it_is_deleted(monkeypatch):
    lib = _Library("u-1", CLIPS)
    _sweep(monkeypatch, lib, ENDED, PLAN_END + timedelta(days=7, hours=1))
    assert lib.bucket == set()
    assert sorted(lib.deleted_tables) == ["projects", "user_videos"]


def test_a_free_account_keeps_its_seven_days_per_clip(monkeypatch):
    # No plan ever, or a checkout that never charged: nothing changes for them.
    for row in (None, _sub("incomplete_expired", end=PLAN_END, last_event=NOW - timedelta(hours=2))):
        lib = _Library("u-2", CLIPS)
        _sweep(monkeypatch, lib, row, NOW)
        assert lib.bucket == set(), row


def test_no_deleted_tomorrow_email_while_the_grace_holds(monkeypatch):
    """The warning mail says "will be deleted tomorrow". A clip in its seventh
    day is not when its owner's plan ended yesterday."""
    sent = []

    async def _mail(email, count):
        sent.append((email, count))

    from cloud import emails
    monkeypatch.setattr(emails, "send_clips_expiring_email", _mail)
    now = PLAN_END + timedelta(days=1)
    lib = _Library("u-1", [now - timedelta(days=6, hours=12)])
    _sweep(monkeypatch, lib, ENDED, now, sweep=videos.warn_free_expiring)
    assert sent == [] and lib.warned == []
    # Same clip, an account that never had a plan: warned as always.
    lib = _Library("u-2", [now - timedelta(days=6, hours=12)])
    _sweep(monkeypatch, lib, None, now, sweep=videos.warn_free_expiring)
    assert sent == [("k@example.com", 1)] and len(lib.warned) == 1


# --------------------------------------------------------------------------- #
# The real sweep against Postgres (CI)
# --------------------------------------------------------------------------- #
DB_URL = os.environ.get("TEST_DATABASE_URL")
try:
    import asyncpg  # noqa: F401
    _HAVE_ASYNCPG = True
except ImportError:
    _HAVE_ASYNCPG = False


@pytest.mark.skipif(not (DB_URL and _HAVE_ASYNCPG),
                    reason="TEST_DATABASE_URL (Postgres) and asyncpg needed")
def test_the_real_sweep_waits_for_the_grace(monkeypatch):
    from sqlalchemy import select, text
    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
    from sqlalchemy.pool import NullPool
    from cloud import database, models
    from cloud.models import Subscription, User

    bucket = set()
    monkeypatch.setattr(videos.storage, "delete_key", bucket.discard)
    monkeypatch.setattr(videos.storage, "list_keys",
                        lambda prefix: [k for k in sorted(bucket) if k.startswith(prefix)])
    now = datetime.now(UTC)
    old = now - timedelta(days=30)

    async def run():
        engine = create_async_engine(DB_URL, poolclass=NullPool)
        saved = database._engine, database._sessionmaker
        database._engine = engine
        database._sessionmaker = async_sessionmaker(engine, expire_on_commit=False,
                                                    class_=AsyncSession)
        try:
            async with engine.begin() as conn:
                await conn.execute(text("CREATE EXTENSION IF NOT EXISTS citext"))
                await conn.run_sync(models.Base.metadata.create_all)
            users = {}
            async with database.session() as s:
                async with s.begin():
                    for name, ended_days_ago in (("grace", 2), ("past", 8), ("free", None)):
                        tag = uuid.uuid4().hex[:10]
                        u = User(email=f"{name}-{tag}@example.com", google_sub=f"g-{tag}")
                        s.add(u)
                        await s.flush()
                        users[name] = u.id
                        if ended_days_ago is not None:
                            ended = now - timedelta(days=ended_days_ago)
                            s.add(Subscription(
                                user_id=u.id, stripe_subscription_id=f"sub_{tag}", plan="creator",
                                interval="month", status="canceled", minutes_per_period=300,
                                current_period_start=ended - timedelta(days=30),
                                current_period_end=ended, cancel_at_period_end=True,
                                last_event_at=ended))
                        key = f"users/{u.id}/job-{tag}/clip.mp4"
                        bucket.add(key)
                        s.add(UserVideo(user_id=u.id, job_id=f"job-{tag}", clip_index=0,
                                        r2_key=key, created_at=old))
            await videos.purge_free_expired()
            async with database.session() as s:
                left = set((await s.execute(
                    select(UserVideo.user_id).where(UserVideo.user_id.in_(users.values()))
                )).scalars())
            return users, left
        finally:
            database._engine, database._sessionmaker = saved
            await engine.dispose()

    users, left = asyncio.run(run())
    assert left == {users["grace"]}
    assert len(bucket) == 1 and str(users["grace"]) in next(iter(bucket))
