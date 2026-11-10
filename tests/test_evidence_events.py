"""Evidence for payment disputes (cloud/evidence.py): sign-ins and checkouts
with the client IP and user agent, written in the background.

The contract that matters most is the negative one: recording evidence can
never fail or change a sign-in or a checkout. The rest pins that each flow
leaves its row and that the rows go with the account.
"""
import asyncio
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from cloud import evidence
from cloud.models import EvidenceEvent

IP = "203.0.113.7"   # TEST-NET-3, documentation range


def _request(ua="Mozilla/5.0 (test)", ip=IP):
    return SimpleNamespace(client=SimpleNamespace(host=ip),
                           headers={"user-agent": ua} if ua is not None else {})


class _FakeSession:
    """Async session double: records add()/execute() and commits nothing."""

    def __init__(self, sink, fail=False, rowcount=1):
        self.sink, self.fail, self.rowcount = sink, fail, rowcount

    async def __aenter__(self):
        if self.fail:
            raise RuntimeError(f"connection refused for user@example.com from {IP}")
        return self

    async def __aexit__(self, *a):
        return False

    def begin(self):
        return self

    def add(self, obj):
        self.sink.append(obj)

    async def execute(self, stmt):
        self.sink.append(stmt)
        return SimpleNamespace(rowcount=self.rowcount)


async def _drain():
    if evidence._pending:
        await asyncio.gather(*list(evidence._pending))


# --------------------------------------------------------------------------- #
# The helper
# --------------------------------------------------------------------------- #
class TestRecordEvent:
    def test_writes_one_row_with_ip_and_user_agent(self):
        rows = []

        async def main():
            task = evidence.record_event(
                user_id="u1", email="a@example.com", kind="login", method="magic_link",
                request=_request(), session_factory=lambda: _FakeSession(rows))
            await task

        asyncio.run(main())
        [row] = rows
        assert isinstance(row, EvidenceEvent)
        assert (row.user_id, row.email, row.kind, row.method) == \
            ("u1", "a@example.com", "login", "magic_link")
        assert row.ip == IP and row.user_agent == "Mozilla/5.0 (test)"
        assert row.meta == {}

    def test_user_agent_is_truncated(self):
        assert len(evidence._client(_request(ua="x" * 5000))[1]) == evidence.USER_AGENT_MAX

    def test_missing_request_parts_are_null(self):
        assert evidence._client(None) == (None, None)
        assert evidence._client(SimpleNamespace(client=None, headers={})) == (None, None)
        assert evidence._client(object()) == (None, None)

    def test_database_failure_is_swallowed_and_logged_without_pii(self, capsys):
        async def main():
            task = evidence.record_event(
                user_id="u1", email="user@example.com", kind="checkout",
                method="stripe_checkout", request=_request(),
                session_factory=lambda: _FakeSession([], fail=True))
            await task   # completes normally: the error never reaches the caller

        asyncio.run(main())
        out = capsys.readouterr().out
        assert "not recorded" in out
        assert "user@example.com" not in out and IP not in out

    def test_without_an_event_loop_it_returns_none(self, capsys):
        assert evidence.record_event(user_id="u1", email="a@example.com", kind="login",
                                     method="google", request=_request()) is None
        assert "user@example.com" not in capsys.readouterr().out

    def test_never_raises_on_bad_input(self):
        async def main():
            # meta that cannot become a dict
            return evidence.record_event(user_id="u1", email="a@example.com", kind="login",
                                         method="google", request=_request(), meta=42)

        assert asyncio.run(main()) is None


# --------------------------------------------------------------------------- #
# Call sites
# --------------------------------------------------------------------------- #
class TestCheckout:
    @pytest.fixture(autouse=True)
    def _billing(self, monkeypatch):
        self.billing = billing = pytest.importorskip("cloud.billing")  # needs the stripe SDK
        entry = {"price_id": "price_s", "kind": "subscription", "plan": "starter",
                 "interval": "month", "minutes": 100}

        async def _user(request):
            return SimpleNamespace(id="u1", email="a@example.com")

        async def _catalog():
            return {"by_id": {"price_s": entry}}

        async def _none(*a, **k):
            return None

        async def _customer(*a, **k):
            return "cus_1"

        monkeypatch.setattr(billing, "get_current_user_required", _user)
        monkeypatch.setattr(billing, "get_catalog", _catalog)
        monkeypatch.setattr(billing, "_blocking_sub_status", _none)
        monkeypatch.setattr(billing, "ensure_stripe_customer", _customer)
        monkeypatch.setattr(billing.stripe.checkout.Session, "create",
                            lambda **k: SimpleNamespace(url="https://checkout.stripe.com/c/x"))

    def _checkout(self, factory):
        evidence_db = []

        async def main():
            body = self.billing.CheckoutRequest(price_id="price_s", source="wall")
            out = await self.billing.create_checkout(body, request=_request())
            await _drain()
            return out

        mp = pytest.MonkeyPatch()
        mp.setattr(evidence.database, "session", lambda: factory(evidence_db))
        try:
            return asyncio.run(main()), evidence_db
        finally:
            mp.undo()

    def test_records_a_checkout_event(self):
        out, rows = self._checkout(lambda sink: _FakeSession(sink))
        assert out == {"url": "https://checkout.stripe.com/c/x"}
        [row] = rows
        assert (row.kind, row.method, row.user_id, row.email) == \
            ("checkout", "stripe_checkout", "u1", "a@example.com")
        assert row.ip == IP
        assert row.meta == {"plan": "starter", "kind": "subscription", "minutes": "100"}

    def test_a_broken_evidence_store_does_not_change_the_checkout(self):
        out, rows = self._checkout(lambda sink: _FakeSession(sink, fail=True))
        assert out == {"url": "https://checkout.stripe.com/c/x"}
        assert rows == []


class TestMagicLinkLogin:
    def test_records_a_login_event(self, monkeypatch):
        from cloud import auth

        token_row = SimpleNamespace(used_at=None, email="a@example.com",
                                    expires_at=datetime.now(timezone.utc) + timedelta(minutes=5))
        user = SimpleNamespace(id="u1", email="a@example.com", last_login_at=None)
        results = iter([token_row, user])

        class _AuthSession(_FakeSession):
            async def execute(self, stmt):
                return SimpleNamespace(scalar_one_or_none=lambda v=next(results): v)

        async def _load(session, uid):
            return SimpleNamespace(entitled=False)

        rows = []
        monkeypatch.setattr(auth.database, "session", lambda: _AuthSession(rows))
        monkeypatch.setattr(auth, "_load_current_user", _load)
        monkeypatch.setattr(auth, "issue_jwt", lambda uid, email: "jwt")

        async def main():
            out = await auth.verify_magic_link(auth.MagicVerifyRequest(token="t"),
                                               request=_request())
            await _drain()
            return out

        out = asyncio.run(main())
        assert out["token"] == "jwt" and out["user"]["id"] == "u1"
        [row] = [r for r in rows if isinstance(r, EvidenceEvent)]
        assert (row.kind, row.method, row.user_id, row.email) == \
            ("login", "magic_link", "u1", "a@example.com")
        assert row.ip == IP and row.user_agent == "Mozilla/5.0 (test)"


# --------------------------------------------------------------------------- #
# Erasure and retention
# --------------------------------------------------------------------------- #
class TestErasureAndRetention:
    def test_is_erased_with_the_account(self, monkeypatch):
        from cloud import account
        assert EvidenceEvent in account.USER_OWNED_TABLES

        stmts = []
        monkeypatch.setattr(account.database, "session", lambda: _FakeSession(stmts))
        assert asyncio.run(account._erase_rows(
            "u1", "a@example.com", None, None, 0, None)) is True
        evidence_deletes = [s for s in stmts if getattr(getattr(s, "table", None), "name", None)
                            == "evidence_events"]
        # One by user id (USER_OWNED_TABLES), one by email for rows without it.
        assert len(evidence_deletes) == 2

    def test_rows_age_out(self, monkeypatch):
        assert evidence.RETENTION_DAYS == 400
        stmts = []
        monkeypatch.setattr(evidence.database, "session", lambda: _FakeSession(stmts))
        asyncio.run(evidence.purge_stale_events())
        [stmt] = stmts
        assert stmt.table.name == "evidence_events"

    def test_the_sweeper_runs_the_purge(self):
        import inspect
        from cloud import videos
        assert "purge_stale_events" in inspect.getsource(videos._sweeper_loop)
