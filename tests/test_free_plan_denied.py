"""Free minutes withdrawn from abusive sign-ups (``User.free_plan_denied``).

Two leaks, both measured in prod on 22-sep-2026:
- a temp-mail farm (~100 rotating front domains, all MX'd to
  wabblywabble.com / wallywatts.com): 568 accounts, 4.828 free minutes;
- delete-and-re-register: the ledger is keyed by user id, so a new row for the
  same address started with a fresh 20 minutes (15 times, 6 addresses).
"""
import asyncio
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from cloud import account, config, email_policy, metering


def _user(**kw):
    base = dict(google_sub=None, email="real@gmail.com", free_plan_denied=None)
    base.update(kw)
    return SimpleNamespace(**base)


class TestEligibility:
    def test_denied_account_gets_no_free_minutes(self):
        assert metering.free_plan_eligible(_user(free_plan_denied="disposable_mx")) is False

    def test_denied_google_account_gets_no_free_minutes_either(self):
        assert metering.free_plan_eligible(
            _user(google_sub="g-1", free_plan_denied="recreated_after_deletion")) is False

    def test_normal_accounts_are_unchanged(self):
        assert metering.free_plan_eligible(_user()) is True
        assert metering.free_plan_eligible(_user(google_sub="g-1")) is True

    def test_rows_without_the_attribute_still_work(self):
        assert metering.free_plan_eligible(SimpleNamespace(google_sub="g", email="a@gmail.com")) is True


class TestFarmMx:
    @pytest.mark.parametrize("hosts", [
        ["mail.wabblywabble.com", "mail.wallywatts.com"],
        ["mail.wallywatts.com"],
        ["mx2.den.yt"],
        ["temp-mail-pro.com"],
        ["prd-smtp.10minutemail.com"],
    ])
    def test_farm_mx_hosts_are_disposable(self, hosts):
        assert email_policy.classify_mx_hosts(hosts) == email_policy.MX_DISPOSABLE

    def test_real_mx_hosts_are_fine(self):
        assert email_policy.classify_mx_hosts(["gmail-smtp-in.l.google.com"]) == email_policy.MX_OK
        assert email_policy.classify_mx_hosts(["mx.yandex.net"]) == email_policy.MX_OK


class _Result:
    def __init__(self, value):
        self._value = value

    def scalar_one_or_none(self):
        return self._value


class _Session:
    def __init__(self, value):
        self.value = value
        self.statements = []

    async def execute(self, stmt):
        self.statements.append(stmt)
        return _Result(self.value)


class TestSignupAfterDeletion:
    def test_recent_erasure_denies_the_free_plan(self):
        s = _Session(value="some-id")
        assert asyncio.run(account.free_plan_denial_for_signup(s, "Foo.Bar@gmail.com")) \
            == "recreated_after_deletion"
        params = s.statements[0].compile().params
        # Same fingerprint the erasure stored (normalised address)...
        assert account.email_fingerprint("foobar@gmail.com") in params.values()
        # ...and only erasures inside the window count.
        cutoffs = [v for v in params.values() if isinstance(v, datetime)]
        assert cutoffs
        expected = datetime.now(timezone.utc) - timedelta(days=config.FREE_REDO_BLOCK_DAYS)
        assert abs((cutoffs[0] - expected).total_seconds()) < 60

    def test_no_erasure_keeps_the_free_plan(self):
        assert asyncio.run(account.free_plan_denial_for_signup(_Session(None), "a@gmail.com")) is None

    def test_window_zero_disables_the_check(self, monkeypatch):
        monkeypatch.setattr(config, "FREE_REDO_BLOCK_DAYS", 0)
        s = _Session("some-id")
        assert asyncio.run(account.free_plan_denial_for_signup(s, "a@gmail.com")) is None
        assert s.statements == []


class TestMagicLinkSignup:
    """verify_magic_link stamps the new row; an existing row is untouched."""

    def _run(self, monkeypatch, existing_user, deleted_before):
        from cloud import auth

        token_row = SimpleNamespace(used_at=None, email="again@gmail.com",
                                    expires_at=datetime.now(timezone.utc) + timedelta(minutes=5))
        added = []

        class _Res:
            def __init__(self, v):
                self.v = v

            def scalar_one_or_none(self):
                return self.v

        class _Sess:
            calls = 0

            async def execute(self, stmt):
                _Sess.calls += 1
                return _Res(token_row if _Sess.calls == 1 else existing_user)

            def add(self, obj):
                obj.id = "new-id"
                added.append(obj)

            async def flush(self):
                pass

            def begin(self):
                return self

            async def __aenter__(self):
                return self

            async def __aexit__(self, *a):
                return False

        async def _denial(session, email):
            return "recreated_after_deletion" if deleted_before else None

        async def _load(session, uid):
            return SimpleNamespace(entitled=False)

        monkeypatch.setattr(auth.database, "session", lambda: _Sess())
        monkeypatch.setattr(account, "free_plan_denial_for_signup", _denial)
        monkeypatch.setattr(auth, "_load_current_user", _load)
        monkeypatch.setattr(auth, "issue_jwt", lambda uid, email: "jwt")
        # The login evidence row would land in this fake session too; it has
        # its own tests (test_evidence_events.py).
        monkeypatch.setattr(auth.evidence, "record_event", lambda **k: None)
        asyncio.run(auth.verify_magic_link(auth.MagicVerifyRequest(token="t")))
        return added

    def test_new_account_after_deletion_is_denied(self, monkeypatch):
        added = self._run(monkeypatch, existing_user=None, deleted_before=True)
        assert added[0].free_plan_denied == "recreated_after_deletion"

    def test_new_account_without_deletion_is_eligible(self, monkeypatch):
        added = self._run(monkeypatch, existing_user=None, deleted_before=False)
        assert added[0].free_plan_denied is None

    def test_existing_account_is_not_restamped(self, monkeypatch):
        existing = SimpleNamespace(id="u1", email="again@gmail.com", free_plan_denied=None,
                                   last_login_at=None)
        added = self._run(monkeypatch, existing_user=existing, deleted_before=True)
        assert added == []
        assert existing.free_plan_denied is None
