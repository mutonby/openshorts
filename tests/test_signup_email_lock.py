"""Two sign-ins for a brand-new address must not both INSERT the user.

A Google callback reloaded or double-clicked (or a magic link opened next to a
Google login) ran two transactions that both found no user and both INSERTed;
the second one hit users_email_key and answered 500. Both sign-in paths now
take a transaction-scoped advisory lock keyed on the email before the lookup.

The fake session mimics what matters: pg_advisory_xact_lock waits until the
other transaction ends, a plain SELECT does not see uncommitted rows, and a
commit with an email that already exists raises.
"""
import asyncio
from types import SimpleNamespace

import pytest
from sqlalchemy.dialects import postgresql

from cloud import auth, oauth


class UniqueViolation(Exception):
    pass


class _Result:
    def __init__(self, value):
        self.value = value

    def scalar_one_or_none(self):
        return self.value


class _Db:
    def __init__(self):
        self.users = {}  # committed rows: email -> user
        self.locks = {}
        self.lock_sql = []

    def session(self):
        return _Session(self)


class _Session:
    def __init__(self, db):
        self.db, self.pending, self.held = db, [], []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    def begin(self):
        return _Tx(self)

    async def execute(self, stmt):
        sql = str(stmt.compile(dialect=postgresql.dialect()))
        await asyncio.sleep(0)  # a round trip: let the other sign-in run
        if "pg_advisory_xact_lock" in sql:
            self.db.lock_sql.append(sql)
            lock = self.db.locks.setdefault("email", asyncio.Lock())
            await lock.acquire()
            self.held.append(lock)
            return _Result(None)
        if "FROM users" in sql:
            return _Result(self.db.users.get("new@example.com"))
        raise AssertionError(f"unexpected statement: {sql}")

    def add(self, obj):
        self.pending.append(obj)

    async def flush(self):
        pass


class _Tx:
    def __init__(self, session):
        self.s = session

    async def __aenter__(self):
        return self.s

    async def __aexit__(self, exc_type, *rest):
        try:
            if exc_type is None:
                for obj in self.s.pending:
                    if obj.email in self.s.db.users:
                        raise UniqueViolation(obj.email)
                    self.s.db.users[obj.email] = obj
        finally:
            for lock in self.s.held:
                lock.release()
        return False


def test_lock_statement_is_an_advisory_xact_lock_on_the_email():
    class _Capture:
        async def execute(self, stmt):
            self.sql = str(stmt.compile(dialect=postgresql.dialect()))

    cap = _Capture()
    asyncio.run(auth.lock_signup_email(cap, "new@example.com"))
    assert "pg_advisory_xact_lock" in cap.sql
    assert "hashtextextended" in cap.sql


def test_two_concurrent_google_callbacks_create_one_user(monkeypatch):
    db = _Db()
    monkeypatch.setattr(oauth.database, "session", db.session)

    async def _denial(session, email):
        return None

    import cloud.account as account
    monkeypatch.setattr(account, "free_plan_denial_for_signup", _denial)
    monkeypatch.setattr(oauth.evidence, "record_event", lambda **kw: None)
    monkeypatch.setattr(oauth, "valid_challenge", lambda c: False)
    monkeypatch.setattr(oauth, "_legacy_token_redirect_open", lambda: False)

    class _Google:
        async def authorize_access_token(self, request):
            return {"userinfo": {"email": "new@example.com", "sub": "g-1", "email_verified": True}}

    monkeypatch.setattr(oauth, "oauth", SimpleNamespace(google=_Google()))
    request = SimpleNamespace(session={})

    async def both():
        return await asyncio.gather(oauth.google_callback(request), oauth.google_callback(request))

    asyncio.run(both())  # the second one used to raise UniqueViolation
    assert list(db.users) == ["new@example.com"]
    assert len(db.lock_sql) == 2
