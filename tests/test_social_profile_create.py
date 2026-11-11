"""Creating the Upload-Post profile is idempotent.

Two near-simultaneous "connect" requests for the same user both try to create
the profile; the loser gets a 409 "Username already in use". That is success
(the profile exists), not a 500.
"""
import asyncio

import httpx
import pytest
from fastapi import HTTPException

from cloud import social_profiles


def _run_with_reply(monkeypatch, status, body):
    def handler(request):
        return httpx.Response(status, text=body)

    real = httpx.AsyncClient

    def client(*a, **kw):
        kw["transport"] = httpx.MockTransport(handler)
        return real(*a, **kw)

    monkeypatch.setattr(social_profiles.httpx, "AsyncClient", client)
    return asyncio.run(social_profiles._create_remote_profile("os_abc"))


@pytest.mark.parametrize("status,body", [
    (200, "{}"),
    (201, "{}"),
    (409, '{"success": false, "message": "Username already in use"}'),
    (400, '{"message": "user already exists"}'),
])
def test_existing_or_created_profile_is_success(monkeypatch, status, body):
    assert _run_with_reply(monkeypatch, status, body) is None


def test_profile_limit_is_a_503(monkeypatch):
    with pytest.raises(HTTPException) as e:
        _run_with_reply(monkeypatch, 403, '{"error_code": "PROFILE_LIMIT_REACHED"}')
    assert e.value.status_code == 503


def test_other_client_errors_still_raise(monkeypatch):
    with pytest.raises(httpx.HTTPStatusError):
        _run_with_reply(monkeypatch, 401, '{"message": "bad key"}')


# --- the local row: two concurrent connects must not 500 on the primary key ---

class _FakeResult:
    def __init__(self, rowcount):
        self.rowcount = rowcount


class _FakeSession:
    """Stands in for the DB. `lost_race` = another request inserted first."""

    def __init__(self, lost_race, existing):
        self.lost_race, self.existing, self.statements = lost_race, existing, []
        self._first_get = True

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    def begin(self):
        return self

    async def get(self, model, key):
        # First get (outside the txn): no row yet. Later get (after losing): the winner's.
        if self._first_get:
            self._first_get = False
            return None
        return self.existing

    async def execute(self, stmt):
        self.statements.append(stmt)
        return _FakeResult(0 if self.lost_race else 1)


def _ensure(monkeypatch, lost_race, existing=None):
    import types
    import uuid

    sess = _FakeSession(lost_race, existing)
    monkeypatch.setattr(social_profiles.database, "session", lambda: sess)

    async def _noop(username):
        return None

    monkeypatch.setattr(social_profiles, "_create_remote_profile", _noop)
    user = types.SimpleNamespace(id=uuid.uuid4())
    return asyncio.run(social_profiles.ensure_profile(user)), sess, user


def test_ensure_profile_inserts_with_on_conflict_do_nothing(monkeypatch):
    from sqlalchemy.dialects import postgresql

    name, sess, user = _ensure(monkeypatch, lost_race=False)
    assert name == social_profiles.profile_username_for(user.id)
    sql = str(sess.statements[0].compile(dialect=postgresql.dialect()))
    assert "ON CONFLICT DO NOTHING" in sql


def test_ensure_profile_losing_the_race_returns_the_winners_row(monkeypatch):
    import types

    winner = types.SimpleNamespace(profile_username="os_winner")
    name, _, _ = _ensure(monkeypatch, lost_race=True, existing=winner)
    assert name == "os_winner"
