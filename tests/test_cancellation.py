"""The dashboard cancel flow (cloud/cancellation.py): reason + review, then a
cancel at period end.

What must hold: nothing is recorded unless Stripe accepted the cancel, the
reason reaches Stripe in its own vocabulary, the free text never reaches the
Telegram alert, and the UI offers exactly the reasons the server accepts.
"""
import asyncio
import os
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from fastapi import HTTPException

from cloud import alerts, cancellation

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class _Result:
    def __init__(self, value):
        self.value = value

    def scalar_one_or_none(self):
        return self.value


class _Session:
    def __init__(self, sub, added):
        self.sub, self.added = sub, added

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    def begin(self):
        return self

    async def execute(self, _stmt):
        return _Result(self.sub)

    def add(self, obj):
        self.added.append(obj)


@pytest.fixture
def env(monkeypatch):
    sub = SimpleNamespace(
        id="sub-row", stripe_subscription_id="sub_123", plan="creator", interval="month",
        status="active", cancel_at_period_end=False, hold_from=None, hold_until=None,
        current_period_end=datetime.now(timezone.utc) + timedelta(days=12),
    )
    state = SimpleNamespace(sub=sub, added=[], stripe_calls=[], alerts=[], stripe_error=None)
    user = SimpleNamespace(id="3f9a1c2b-0000-0000-0000-000000000000")

    async def _user(_request):
        return user

    def _modify(sid, feedback, comment):
        if state.stripe_error:
            raise state.stripe_error
        state.stripe_calls.append((sid, feedback, comment))

    async def _alert(subject, body, **kw):
        state.alerts.append((subject, body))

    async def _session_user(request, detail=None):
        return await _user(request)

    monkeypatch.setattr(cancellation, "get_current_user_required", _user)
    monkeypatch.setattr(cancellation, "get_session_user_required", _session_user)
    monkeypatch.setattr(cancellation.database, "session", lambda: _Session(state.sub, state.added))
    monkeypatch.setattr(cancellation, "_stripe_cancel_at_period_end", _modify)
    monkeypatch.setattr(cancellation, "RETENTION_COUPON", "coupon_test")
    state.offer_blocked = False
    state.applied = []
    state.resumed = []
    monkeypatch.setattr(cancellation, "_stripe_offer_blocked", lambda sid: state.offer_blocked)
    monkeypatch.setattr(cancellation, "_stripe_coupon_terms",
                        lambda cid: {"percent_off": 50, "months": 3})
    monkeypatch.setattr(cancellation, "_stripe_apply_retention",
                        lambda sid, cid: state.applied.append((sid, cid)))
    monkeypatch.setattr(cancellation, "_stripe_resume", lambda sid: state.resumed.append(sid))
    state.holds = []
    state.unholds = []
    monkeypatch.setattr(cancellation, "_stripe_hold",
                        lambda sid, f, u: state.holds.append((sid, f, u)))
    monkeypatch.setattr(cancellation, "_stripe_unhold",
                        lambda sid, started, now_ts: state.unholds.append((sid, started)))
    monkeypatch.setattr(cancellation.analytics, "track", lambda *a, **k: None)
    monkeypatch.setattr(alerts, "send_admin_alert", _alert)
    return state


def _cancel(**body):
    body.setdefault("reason", "too_expensive")
    return asyncio.run(cancellation.cancel_subscription(
        cancellation.CancelRequest(**body), request=None))


class TestCancel:
    def test_cancels_at_period_end_with_the_reason_in_stripe_terms(self, env):
        out = _cancel(reason="clip_quality", details="blurry", rating=2, review="meh")
        assert out["ok"]
        assert env.stripe_calls == [("sub_123", "low_quality", "blurry | meh")]
        assert env.sub.cancel_at_period_end is True

    def test_feedback_row_is_stored(self, env):
        _cancel(reason="missing_feature", details="  ", rating=4, review="good",
                review_public_ok=True)
        (row,) = env.added
        assert row.reason == "missing_feature"
        assert row.details is None            # whitespace is not feedback
        assert row.rating == 4 and row.review == "good" and row.review_public_ok

    def test_public_ok_without_a_review_is_not_recorded(self, env):
        _cancel(rating=5, review_public_ok=True)
        assert env.added[0].review_public_ok is False

    def test_stripe_failure_records_nothing(self, env):
        env.stripe_error = RuntimeError("stripe down")
        with pytest.raises(HTTPException) as e:
            _cancel()
        assert e.value.status_code == 502
        assert env.added == [] and env.alerts == []
        assert env.sub.cancel_at_period_end is False

    def test_unknown_reason_is_refused(self, env):
        with pytest.raises(HTTPException) as e:
            _cancel(reason="i hate it")
        assert e.value.status_code == 422
        assert env.stripe_calls == []

    def test_already_canceling_is_refused(self, env):
        env.sub.cancel_at_period_end = True
        with pytest.raises(HTTPException) as e:
            _cancel()
        assert e.value.status_code == 409

    @pytest.mark.parametrize("status", ["canceled", "incomplete", "incomplete_expired"])
    def test_nothing_live_to_cancel(self, env, status):
        env.sub.status = status
        with pytest.raises(HTTPException) as e:
            _cancel()
        assert e.value.status_code == 409

    def test_alert_never_carries_the_written_text(self, env):
        _cancel(details="my name is Alice Smith", review="call me on 555-1234", rating=1)
        (subject, body), = env.alerts
        assert "Alice" not in body and "555" not in body
        assert "3f9a1c2b" in body and "too_expensive" in body and "1/5" in body


def _offer():
    return asyncio.run(cancellation.retention_offer(request=None))


def _accept(**body):
    body.setdefault("reason", "too_expensive")
    return asyncio.run(cancellation.accept_retention_offer(
        cancellation.CancelRequest(**body), request=None))


class TestRetentionOffer:
    def test_a_live_monthly_plan_gets_the_offer(self, env):
        out = _offer()
        assert (out["eligible"], out["percent_off"], out["months"]) == (True, 50, 3)

    @pytest.mark.parametrize("field,value", [
        ("interval", "year"),        # would halve a whole year
        ("status", "trialing"),
        ("status", "past_due"),      # a discount does not fix a failed card
    ])
    def test_no_offer_outside_a_live_monthly_plan(self, env, field, value):
        setattr(env.sub, field, value)
        assert _offer() == {"eligible": False}

    def test_no_second_offer_or_stacked_discount(self, env):
        env.offer_blocked = True
        assert _offer()["eligible"] is False
        with pytest.raises(HTTPException) as e:
            _accept()
        assert e.value.status_code == 409
        assert env.applied == []

    def test_a_stripe_hiccup_means_no_offer_not_a_broken_flow(self, env, monkeypatch):
        def boom(_sid):
            raise RuntimeError("stripe down")
        monkeypatch.setattr(cancellation, "_stripe_offer_blocked", boom)
        assert _offer()["eligible"] is False

    def test_taking_it_applies_the_coupon_and_keeps_the_feedback(self, env):
        out = _accept(reason="not_using_it", details="busy month", rating=3)
        assert out["ok"] and out["percent_off"] == 50
        assert env.applied == [("sub_123", cancellation.RETENTION_COUPON)]
        assert env.stripe_calls == []                  # nothing was cancelled
        assert env.sub.cancel_at_period_end is False
        (row,) = env.added
        assert row.outcome == "retained" and row.reason == "not_using_it"
        (subject, body), = env.alerts
        assert "busy month" not in body and "50% off" in body

    def test_no_coupon_configured_no_offer(self, env, monkeypatch):
        monkeypatch.setattr(cancellation, "RETENTION_COUPON", "")
        assert _offer()["eligible"] is False

    def test_a_cancel_row_says_so(self, env):
        _cancel()
        assert env.added[0].outcome == "canceled"


def _pause(**body):
    body.setdefault("reason", "not_using_it")
    return asyncio.run(cancellation.pause_subscription(
        cancellation.CancelRequest(**body), request=None))


class TestPause:
    def test_offered_on_a_live_monthly_plan_from_its_period_end(self, env):
        out = _offer()
        assert out["pause_eligible"] is True
        assert out["pause_from"] == env.sub.current_period_end.isoformat()
        assert set(out["pause_reasons"]) == set(cancellation.PAUSE_REASONS)

    @pytest.mark.parametrize("field,value", [
        ("interval", "year"), ("status", "trialing"), ("status", "past_due"),
        ("hold_from", datetime(2026, 1, 1, tzinfo=timezone.utc)),   # one per subscription
    ])
    def test_not_offered_otherwise(self, env, field, value):
        setattr(env.sub, field, value)
        assert "pause_eligible" not in _offer()

    def test_holds_the_month_after_the_paid_one(self, env):
        out = _pause(details="exams", rating=4)
        assert out["ok"]
        ((sid, hold_from, hold_until),) = env.holds
        end = env.sub.current_period_end
        assert sid == "sub_123" and hold_from == int(end.timestamp())
        assert hold_until - hold_from == cancellation.config.HOLD_DAYS * 86400
        assert env.sub.hold_from == end and env.stripe_calls == []   # not cancelled
        (row,) = env.added
        assert row.outcome == "paused" and row.reason == "not_using_it"
        (subject, body), = env.alerts
        assert "exams" not in body

    def test_only_for_the_pause_reasons(self, env):
        with pytest.raises(HTTPException) as e:
            _pause(reason="clip_quality")
        assert e.value.status_code == 422
        assert env.holds == []

    def test_stripe_failure_records_nothing(self, env, monkeypatch):
        def boom(*a):
            raise RuntimeError("stripe down")
        monkeypatch.setattr(cancellation, "_stripe_hold", boom)
        with pytest.raises(HTTPException) as e:
            _pause()
        assert e.value.status_code == 502
        assert env.added == [] and env.sub.hold_from is None

    def test_unpause_during_the_hold_charges_a_new_period(self, env):
        env.sub.status = cancellation.config.ON_HOLD
        out = asyncio.run(cancellation.unpause_subscription(request=None))
        assert out == {"ok": True, "charged": True}
        assert env.unholds == [("sub_123", True)]

    def test_unpause_before_it_starts_forgets_it(self, env):
        env.sub.hold_from = env.sub.current_period_end
        env.sub.hold_until = env.sub.current_period_end + timedelta(days=27)
        out = asyncio.run(cancellation.unpause_subscription(request=None))
        assert out == {"ok": True, "charged": False}
        assert env.unholds == [("sub_123", False)]
        assert env.sub.hold_from is None and env.sub.hold_until is None

    def test_nothing_to_unpause(self, env):
        with pytest.raises(HTTPException) as e:
            asyncio.run(cancellation.unpause_subscription(request=None))
        assert e.value.status_code == 409

    def test_the_ui_knows_the_pause_reasons(self):
        modal = open(os.path.join(REPO, "dashboard/src/components/CancelPlanModal.jsx")).read()
        assert "pause_reasons" in modal and "/api/billing/pause" in modal


class TestHoldStatus:
    def _sub(self, start, **meta):
        return {"status": "active", "metadata": {k: str(v) for k, v in meta.items()}}

    def test_period_inside_the_window_is_on_hold(self):
        sub = self._sub(0, hold_from=1000, hold_until=1000 + 27 * 86400)
        assert cancellation.config.subscription_status(sub, 1000) == "on_hold"

    def test_paid_period_before_and_renewal_after_are_active(self):
        sub = self._sub(0, hold_from=1000, hold_until=1000 + 27 * 86400)
        assert cancellation.config.subscription_status(sub, 1000 - 30 * 86400) == "active"
        assert cancellation.config.subscription_status(sub, 1000 + 28 * 86400) == "active"

    def test_stripe_states_win(self):
        sub = {"status": "past_due", "metadata": {"hold_from": "1000", "hold_until": "9999"}}
        assert cancellation.config.subscription_status(sub, 1000) == "past_due"
        assert cancellation.config.subscription_status({"status": "active"}, 1000) == "active"
        bad = {"status": "active", "metadata": {"hold_from": "x"}}
        assert cancellation.config.subscription_status(bad, 1000) == "active"

    def test_on_hold_blocks_a_second_checkout(self):
        assert cancellation.config.ON_HOLD in cancellation.config.CHECKOUT_BLOCKING_STATES


class TestResume:
    def test_undoes_a_scheduled_cancel(self, env):
        env.sub.cancel_at_period_end = True
        assert asyncio.run(cancellation.resume_subscription(request=None))["ok"]
        assert env.resumed == ["sub_123"]
        assert env.sub.cancel_at_period_end is False

    def test_nothing_to_undo(self, env):
        with pytest.raises(HTTPException) as e:
            asyncio.run(cancellation.resume_subscription(request=None))
        assert e.value.status_code == 409
        assert env.resumed == []


class TestReasons:
    def test_every_reason_maps_to_a_stripe_feedback_value(self):
        stripe_enum = {"customer_service", "low_quality", "missing_features", "other",
                       "switched_service", "too_complex", "too_expensive", "unused"}
        assert set(cancellation.CANCEL_REASONS.values()) <= stripe_enum

    def test_the_ui_offers_exactly_the_reasons_the_server_accepts(self):
        modal = open(os.path.join(REPO, "dashboard/src/components/CancelPlanModal.jsx")).read()
        for value in cancellation.CANCEL_REASONS:
            assert f"'{value}'" in modal, f"{value} is accepted but never offered"
