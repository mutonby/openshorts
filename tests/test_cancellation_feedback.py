"""Cancellation handoff only: Stripe and database are isolated, never contacted."""
from contextlib import asynccontextmanager
from types import SimpleNamespace
from uuid import uuid4
from unittest.mock import Mock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from cloud import billing
from cloud.auth import CurrentUser


@pytest.fixture
def setup(monkeypatch):
    user = CurrentUser(id=uuid4(), email='owner@example.com')
    sub = SimpleNamespace(stripe_subscription_id='sub_owner', status='active')
    rows = []
    queries = []
    class Session:
        async def get(self, model, user_id):
            assert user_id == user.id
            return SimpleNamespace(stripe_customer_id='cus_owner')
        async def execute(self, query):
            queries.append(query)
            return SimpleNamespace(scalar_one_or_none=lambda: sub)
        def add(self, row):
            rows.append(row)
        async def commit(self):
            pass
    @asynccontextmanager
    async def session():
        yield Session()
    async def auth(request):
        return user
    monkeypatch.setattr(billing, 'get_current_user_required', auth)
    monkeypatch.setattr(billing.database, 'session', session)
    portal = Mock(return_value=SimpleNamespace(url='https://billing.stripe.test/session'))
    monkeypatch.setattr(billing.stripe.billing_portal.Session, 'create', portal)
    app = FastAPI()
    app.include_router(billing.router)
    return SimpleNamespace(client=TestClient(app), user=user, sub=sub, rows=rows,
                           portal=portal, queries=queries, session=session)


def test_feedback_is_private_intent_and_portal_targets_owned_subscription(setup):
    s = setup
    response = s.client.post('/api/billing/cancel', json={
        'reason': 'too_expensive', 'comment': 'My private experience', 'rating': 2})
    assert response.status_code == 200
    assert response.json()['url'] == 'https://billing.stripe.test/session'
    row, = s.rows
    assert row.user_id == s.user.id
    assert row.stripe_subscription_id == 'sub_owner'
    assert row.status == 'intent'
    assert row.comment == 'My private experience'
    kwargs = s.portal.call_args.kwargs
    assert kwargs['customer'] == 'cus_owner'
    assert kwargs['flow_data']['subscription_cancel']['subscription'] == 'sub_owner'
    assert 'private' not in repr(kwargs)
    assert s.user.id in s.queries[0].compile().params.values()


@pytest.mark.parametrize('body', [{}, {'comment': '  '}])
def test_skip_is_allowed(setup, body):
    assert setup.client.post('/api/billing/cancel', json=body).status_code == 200
    assert setup.rows[0].comment is None


@pytest.mark.parametrize('body', [
    {'reason': 'invalid'}, {'comment': 'x' * 2001}, {'rating': 0},
    {'rating': 6}, {'rating': True}, {'user_id': 'another-user'},
    {'stripe_subscription_id': 'sub_someone_else'},
])
def test_invalid_input_never_reaches_stripe(setup, body):
    assert setup.client.post('/api/billing/cancel', json=body).status_code == 422
    setup.portal.assert_not_called()


def test_feedback_storage_failure_cannot_block_cancellation(setup, monkeypatch):
    calls = 0
    @asynccontextmanager
    async def session():
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError('private database error')
        async with setup.session() as s:
            yield s
    monkeypatch.setattr(billing.database, 'session', session)
    assert setup.client.post('/api/billing/cancel', json={'reason': 'other'}).status_code == 200


def test_deep_link_failure_falls_back_to_normal_portal(setup):
    setup.portal.side_effect = [RuntimeError('unsupported flow'), SimpleNamespace(url='fallback')]
    assert setup.client.post('/api/billing/cancel', json={}).json()['url'] == 'fallback'
    assert 'flow_data' not in setup.portal.call_args.kwargs


def test_stripe_failure_is_retryable_and_not_a_cancellation(setup):
    setup.portal.side_effect = RuntimeError('stripe down')
    assert setup.client.post('/api/billing/cancel', json={}).status_code == 502
    assert setup.rows == []


@pytest.mark.parametrize('status', ['canceled', 'incomplete_expired'])
def test_terminal_subscription_rejected(setup, status):
    setup.sub.status = status
    assert setup.client.post('/api/billing/cancel', json={}).status_code == 409
    setup.portal.assert_not_called()


def test_authentication_is_required(setup, monkeypatch):
    from cloud.auth import get_current_user_required
    monkeypatch.setattr(billing, 'get_current_user_required', get_current_user_required)
    assert setup.client.post('/api/billing/cancel', json={}).status_code == 401
    setup.portal.assert_not_called()


def test_account_has_optional_accessible_feedback_form():
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    component = root / 'dashboard/src/components/CancellationFeedback.jsx'
    assert component.exists()
    source = component.read_text()
    assert 'showModal()' in source
    assert 'Skip feedback and continue' in source
    assert 'Send feedback and continue' in source
    assert 'maxLength={2000}' in source
    assert 'CancellationFeedback' in (root / 'dashboard/src/components/AccountPage.jsx').read_text()

