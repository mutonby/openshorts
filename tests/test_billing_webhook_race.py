"""Two subscription webhooks for a user who has no subscription row yet.

Checkout sends customer.subscription.created and .updated within milliseconds
of each other. FOR UPDATE on a subscription row that does not exist locks
nothing, so both handlers used to INSERT and the second one hit the UNIQUE
constraint on subscriptions.user_id (a 500 and a Stripe retry). The handler
must serialize webhooks of the same user and end with exactly one row.

The fake session below mimics the Postgres behaviour that matters here: a
SELECT ... FOR [NO KEY] UPDATE on the user waits until the other transaction
ends, a plain SELECT does not, and a commit with a second row for the same
user raises.
"""
import asyncio

import pytest

billing = pytest.importorskip("cloud.billing")  # needs the stripe SDK

from sqlalchemy.dialects import postgresql  # noqa: E402

USER_ID = "3f9a1c2b-0000-0000-0000-000000000000"


class UniqueViolation(Exception):
    pass


class _Result:
    def __init__(self, value):
        self.value = value

    def scalar_one_or_none(self):
        return self.value


class _Db:
    def __init__(self):
        self.subs = {}  # committed rows: user_id -> Subscription
        self.user_lock = asyncio.Lock()
        self.user_lock_sql = []

    def session(self):
        return _Session(self)


class _Session:
    def __init__(self, db):
        self.db, self.pending, self.holds_lock = db, [], False

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    def begin(self):
        return _Tx(self)

    async def execute(self, stmt):
        sql = str(stmt.compile(dialect=postgresql.dialect()))
        await asyncio.sleep(0)  # a round trip to the database: let the other webhook run
        if "FROM users" in sql:
            if "FOR" in sql and "UPDATE" in sql and not self.holds_lock:
                self.db.user_lock_sql.append(sql)
                await self.db.user_lock.acquire()
                self.holds_lock = True
            return _Result(USER_ID)
        if "FROM subscriptions" in sql:
            row = self.db.subs.get(USER_ID)
            await asyncio.sleep(0)
            return _Result(row)
        raise AssertionError(f"unexpected statement: {sql}")

    def add(self, obj):
        self.pending.append(obj)


class _Tx:
    def __init__(self, session):
        self.s = session

    async def __aenter__(self):
        return self.s

    async def __aexit__(self, exc_type, *exc):
        try:
            if exc_type is None:
                for obj in self.s.pending:
                    if obj.user_id in self.s.db.subs:
                        raise UniqueViolation("subscriptions_user_id_key")
                    self.s.db.subs[obj.user_id] = obj
        finally:
            if self.s.holds_lock:
                self.s.holds_lock = False
                self.s.db.user_lock.release()
        return False


def _sub(status, created):
    return {
        "id": "sub_new", "customer": "cus_new", "status": status,
        "cancel_at_period_end": False, "metadata": {},
        "items": {"data": [{"price": {"id": "price_starter_month"},
                            "current_period_start": created,
                            "current_period_end": created + 30 * 86400}]},
    }


@pytest.fixture
def db(monkeypatch):
    db = _Db()

    async def _catalog(force=False):
        return None

    monkeypatch.setattr(billing.database, "session", db.session)
    monkeypatch.setattr(billing, "plan_info_for_price",
                        lambda pid: {"plan": "starter", "interval": "month"})
    monkeypatch.setattr(billing, "get_catalog", _catalog)
    return db


def test_created_and_updated_together_leave_one_row(db):
    t0 = 1794154000

    async def both():
        await asyncio.gather(
            billing._upsert_subscription(_sub("incomplete", t0), billing._ts(t0)),
            billing._upsert_subscription(_sub("incomplete", t0 + 1), billing._ts(t0 + 1)),
        )

    asyncio.run(both())  # raised UniqueViolation before the user lock

    assert list(db.subs) == [USER_ID]
    row = db.subs[USER_ID]
    assert row.stripe_subscription_id == "sub_new"
    assert row.last_event_at == billing._ts(t0 + 1)


def test_user_lock_does_not_block_foreign_key_inserts(db):
    # FOR UPDATE on users would also wait for every insert elsewhere that
    # references the user (FK checks take KEY SHARE); NO KEY UPDATE does not.
    t0 = 1794154000
    asyncio.run(billing._upsert_subscription(_sub("incomplete", t0), billing._ts(t0)))
    assert db.user_lock_sql and all("FOR NO KEY UPDATE" in s for s in db.user_lock_sql)
