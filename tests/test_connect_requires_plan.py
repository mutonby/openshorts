"""Connecting social accounts is part of the paid plans.

18-nov-2026: free accounts had connected thousands of YouTube channels
through the managed Upload-Post account (most of them free, posting nothing),
filling its platform quota. /api/social/connect now answers a free account
with a 402 the dashboard turns into the plans modal.
"""
import asyncio
import types

import pytest

social = pytest.importorskip("cloud.social_profiles")
from fastapi import HTTPException  # noqa: E402


def _user(plan, entitled=True):
    return types.SimpleNamespace(id="u-1", email="a@example.com", plan=plan, entitled=entitled)


@pytest.mark.parametrize("plan,entitled", [("starter", True), ("creator", True), ("pro", True),
                                           (None, True)])     # bought minutes, no plan
def test_paid_accounts_may_connect(plan, entitled):
    social.require_paid_plan_to_connect(_user(plan, entitled))


@pytest.mark.parametrize("plan,entitled", [("free", True), (None, False)])
def test_free_accounts_get_the_plans_wall(plan, entitled):
    with pytest.raises(HTTPException) as e:
        social.require_paid_plan_to_connect(_user(plan, entitled))
    assert e.value.status_code == 402
    assert e.value.detail["error"] == "connect_requires_plan"
    assert "src=connect_wall" in e.value.detail["upgrade_url"]


def test_the_endpoint_refuses_before_touching_upload_post(monkeypatch):
    calls = []

    async def current_user(_request):
        return _user("free")

    async def ensure(_user):
        calls.append("ensure")
        return "os_x"

    async def url(*_a, **_k):
        calls.append("url")
        return "https://connect"
    import cloud.auth
    monkeypatch.setattr(cloud.auth, "get_current_user_required", current_user)
    monkeypatch.setattr(social, "ensure_profile", ensure)
    monkeypatch.setattr(social, "get_connect_url", url)
    request = types.SimpleNamespace(query_params={})
    with pytest.raises(HTTPException) as e:
        asyncio.run(social.social_connect(request))
    assert e.value.status_code == 402 and calls == []

    async def paid_user(_request):
        return _user("starter")
    monkeypatch.setattr(cloud.auth, "get_current_user_required", paid_user)
    assert asyncio.run(social.social_connect(request)) == {"access_url": "https://connect"}
    assert calls == ["ensure", "url"]
