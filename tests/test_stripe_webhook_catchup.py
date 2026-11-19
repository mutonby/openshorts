"""A payment whose webhook delivery failed is applied by the catch-up.

Stripe waits ~20 s for the webhook. When the API stalls (saturated, or
restarting) the delivery times out and Stripe retries it only 30 min to hours
later, so a customer who has just paid stays on the free plan meanwhile. The
engine asks Stripe for the events whose delivery has not succeeded and applies
the unrecorded ones through the webhook's own path, once.
"""
import asyncio

import pytest

billing = pytest.importorskip("cloud.billing")  # needs the stripe SDK

from sqlalchemy.sql.dml import Insert  # noqa: E402


class _Scalars:
    def __init__(self, values):
        self.values = values

    def all(self):
        return list(self.values)


class _Result:
    def __init__(self, values):
        self.values = values

    def scalars(self):
        return _Scalars(self.values)


class _Db:
    def __init__(self, recorded=()):
        self.recorded = set(recorded)

    def session(self):
        return _Session(self)


class _Tx:
    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


class _Session:
    def __init__(self, db):
        self.db = db

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    def begin(self):
        return _Tx()

    async def get(self, _model, key):
        await asyncio.sleep(0)
        return object() if key in self.db.recorded else None

    async def execute(self, stmt):
        await asyncio.sleep(0)
        if isinstance(stmt, Insert):
            self.db.recorded.add(stmt.compile().params["id"])
            return _Result([])
        # SELECT stripe_events.id WHERE id IN (...): every recorded id is
        # enough here, the caller only tests membership.
        return _Result(sorted(self.db.recorded))


def _event(eid, etype, created):
    return {"id": eid, "type": etype, "created": created, "data": {"object": {}}}


@pytest.fixture
def env(monkeypatch):
    db = _Db()
    handled = []
    monkeypatch.setattr(billing, "database", db)
    import cloud.auth
    monkeypatch.setattr(cloud.auth, "forget_current_user", lambda *a: None)

    async def handle_event(event):
        handled.append(event["id"])

    monkeypatch.setattr(billing, "handle_event", handle_event)
    return db, handled


def test_catchup_applies_the_paid_checkout_whose_delivery_failed(env, monkeypatch):
    db, handled = env
    db.recorded.add("evt_seen")  # its first delivery worked; Stripe just has not heard back
    events = [
        _event("evt_seen", "customer.subscription.updated", 1_000),
        _event("evt_sub", "customer.subscription.created", 1_001),
        _event("evt_cs", "checkout.session.completed", 1_002),
    ]
    monkeypatch.setattr(billing, "_list_undelivered", lambda now: list(events))

    applied = asyncio.run(billing.reconcile_undelivered_events(now=2_000))

    assert applied == 2
    assert handled == ["evt_sub", "evt_cs"]
    assert {"evt_sub", "evt_cs"} <= db.recorded
    # Stripe's own retry arrives later: recorded, so it is a dedup, not a re-run.
    assert asyncio.run(billing.apply_event(events[2])) is False
    assert handled == ["evt_sub", "evt_cs"]


def test_one_failing_event_does_not_hold_back_the_others(env, monkeypatch):
    db, handled = env

    async def handle_event(event):
        if event["id"] == "evt_bad":
            raise RuntimeError("QueuePool limit reached")
        handled.append(event["id"])

    monkeypatch.setattr(billing, "handle_event", handle_event)
    monkeypatch.setattr(billing, "_list_undelivered", lambda now: [
        _event("evt_bad", "invoice.paid", 1), _event("evt_ok", "invoice.paid", 2)])

    assert asyncio.run(billing.reconcile_undelivered_events(now=2_000)) == 1
    assert handled == ["evt_ok"]
    assert "evt_bad" not in db.recorded  # left for the next pass / Stripe's retry


def test_webhook_and_catchup_holding_the_same_event_apply_it_once(env, monkeypatch):
    db, handled = env
    release = None

    async def slow_handle_event(event):
        handled.append(event["id"])
        await release.wait()  # still applying when the second caller arrives

    monkeypatch.setattr(billing, "handle_event", slow_handle_event)
    ev = _event("evt_race", "customer.subscription.created", 1)

    async def both():
        nonlocal release
        release = asyncio.Event()
        first = asyncio.create_task(billing.apply_event(ev))
        second = asyncio.create_task(billing.apply_event(ev))
        await asyncio.sleep(0.01)
        release.set()
        return await asyncio.gather(first, second)

    assert asyncio.run(both()) == [True, True]
    assert handled == ["evt_race"]
    assert not billing._applying


def test_asks_stripe_only_for_undelivered_events_of_the_handled_types(monkeypatch):
    seen = {}

    class _Page:
        def auto_paging_iter(self):
            return iter([_event("b", "invoice.paid", 20), _event("a", "invoice.paid", 10)])

    def fake_list(**kwargs):
        seen.update(kwargs)
        return _Page()

    monkeypatch.setattr(billing.stripe.Event, "list", fake_list)
    events = billing._list_undelivered(now=100_000)

    assert seen["delivery_success"] is False
    assert set(seen["types"]) == set(billing.HANDLED_EVENT_TYPES)
    assert len(seen["types"]) <= 20  # Stripe's limit for the filter
    assert seen["created"]["lte"] == 100_000 - billing.RECONCILE_MIN_AGE_S
    assert seen["created"]["gte"] == 100_000 - billing.RECONCILE_LOOKBACK_S
    assert [e["id"] for e in events] == ["a", "b"]  # oldest first


def test_every_type_the_webhook_handles_is_caught_up():
    import inspect
    src = inspect.getsource(billing.handle_event)
    for etype in billing.HANDLED_EVENT_TYPES:
        assert f'"{etype}"' in src
