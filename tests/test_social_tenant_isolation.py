"""Multi-tenant isolation for the social surface (posting, scheduling, analytics).

Cloud runs every user's connected accounts inside ONE Upload-Post account, as
profiles named ``os_<hex>``. That means the profile string is the only thing
standing between "schedule to my TikTok" and "schedule to a stranger's", and it
must never come from the request body.

Two things are pinned here:

1. ``resolve_post_profile`` ignores the client-supplied profile in cloud mode
   and refuses when the server cannot resolve one, rather than falling back to
   the client's value. The call sites used to read ``forced_profile or
   req.user_id``; that is the exact shape that turns a resolver bug into a
   cross-tenant write.
2. The vendor-facing list helper filters by profile even if the provider
   returns additional profiles. Listing and mutations share that check.
"""
import asyncio
import json
from datetime import datetime, timedelta, timezone

import httpx
import pytest
from fastapi import HTTPException

import app as app_module


OTHER_TENANT = "os_deadbeefcafe"


class TestResolvePostProfileInCloud:
    @pytest.fixture(autouse=True)
    def _cloud(self, monkeypatch):
        monkeypatch.setattr(app_module, "BILLING_ENABLED", True)

    def test_uses_the_server_resolved_profile(self):
        assert app_module.resolve_post_profile("os_mine", None) == "os_mine"

    def test_ignores_a_profile_supplied_by_the_caller(self):
        assert app_module.resolve_post_profile("os_mine", OTHER_TENANT) == "os_mine"

    def test_refuses_instead_of_falling_back_to_the_caller_value(self):
        # The whole point: no server profile must NOT mean "use theirs".
        with pytest.raises(HTTPException) as exc:
            app_module.resolve_post_profile(None, OTHER_TENANT)
        assert exc.value.status_code == 409  # "connect a social account first"

    @pytest.mark.parametrize("empty", [None, ""])
    def test_empty_server_profile_is_also_refused(self, empty):
        with pytest.raises(HTTPException):
            app_module.resolve_post_profile(empty, OTHER_TENANT)


class TestResolvePostProfileSelfHosted:
    """Self-host has no user model: the caller owns the Upload-Post account
    whose key resolved the request, so it may name its own profile."""

    @pytest.fixture(autouse=True)
    def _self_host(self, monkeypatch):
        monkeypatch.setattr(app_module, "BILLING_ENABLED", False)

    def test_client_profile_is_honoured(self):
        assert app_module.resolve_post_profile(None, "my-profile") == "my-profile"

    def test_missing_profile_is_a_client_error(self):
        with pytest.raises(HTTPException) as exc:
            app_module.resolve_post_profile(None, None)
        assert exc.value.status_code == 400


class TestScheduledPostsAreFilteredByProfile:
    def test_other_tenants_rows_are_dropped(self, monkeypatch):
        vendor_rows = {
            "scheduled_posts": [
                {"job_id": "mine-1", "profile_username": "os_mine"},
                {"job_id": "theirs", "profile_username": OTHER_TENANT},
                {"job_id": "mine-2", "profile_username": "os_mine"},
                {"job_id": "nameless"},  # no profile at all: not ours
            ]
        }

        async def fake_get(api_key, url, params):
            assert params == {"profile_username": "os_mine"}
            return vendor_rows

        monkeypatch.setattr(app_module, "_upload_post_get", fake_get)
        rows = asyncio.run(app_module._scheduled_posts_for("key", "os_mine"))

        assert [r["job_id"] for r in rows] == ["mine-1", "mine-2"]
        assert all(r.get("profile_username") == "os_mine" for r in rows)


def test_reschedule_changes_the_time_only_for_a_post_owned_by_the_profile(monkeypatch):
    calls = []
    when = datetime.now(timezone(timedelta(hours=2))) + timedelta(days=2)
    utc_when = when.astimezone(timezone.utc).isoformat()

    async def managed_profile(request, byok_profile):
        assert byok_profile == OTHER_TENANT
        return "managed-key", "os_mine"

    def vendor(request):
        assert request.headers["Authorization"] == "Apikey managed-key"
        calls.append((request.method, request.url.path))
        if request.method == "GET":
            assert request.url.params["profile_username"] == "os_mine"
            return httpx.Response(200, json={"scheduled_posts": [
                {"job_id": "mine", "profile_username": "os_mine"},
                {"job_id": "theirs", "profile_username": OTHER_TENANT},
            ]})
        assert request.method == "PATCH"
        assert json.loads(request.content) == {"scheduled_date": utc_when}
        return httpx.Response(200, json={"success": True, "scheduled_date": utc_when})

    monkeypatch.setattr(app_module, "_social_analytics_auth", managed_profile)
    monkeypatch.setattr(app_module.social_platforms, "upload_post_client",
                        lambda timeout: httpx.AsyncClient(transport=httpx.MockTransport(vendor)))

    async def exercise_route():
        transport = httpx.ASGITransport(app=app_module.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            base = "/api/social/scheduled"
            params = {"user": OTHER_TENANT}
            body = {"scheduled_date": when.isoformat()}
            denied = await client.patch(f"{base}/theirs", json=body, params=params)
            assert denied.status_code == 404
            result = await client.patch(f"{base}/mine", json=body, params=params)
            assert result.status_code == 200
            assert result.json() == {"success": True, "job_id": "mine", "scheduled_date": utc_when}

    asyncio.run(exercise_route())
    assert calls == [
        ("GET", "/api/uploadposts/schedule"),
        ("GET", "/api/uploadposts/schedule"),
        ("PATCH", "/api/uploadposts/schedule/mine"),
    ]


class TestImpressionsWithoutAProfile:
    """The profile is only created on "connect socials", so most cloud users
    have none: the account page's views card must get an empty answer, not a
    409, and the vendor must not be called."""

    def test_returns_the_empty_card(self, monkeypatch):
        monkeypatch.setattr(app_module, "BILLING_ENABLED", True)

        async def no_profile(request, body_key=None):
            return "managed-key", None

        async def vendor_must_not_be_called(*args, **kwargs):
            raise AssertionError("vendor called without a profile")

        monkeypatch.setattr(app_module, "resolve_upload_post", no_profile)
        monkeypatch.setattr(app_module, "_upload_post_get", vendor_must_not_be_called)
        body = asyncio.run(app_module.social_total_impressions(
            request=None, period="last_month", start_date=None, end_date=None,
            platform=None, breakdown=None, user=OTHER_TENANT))

        assert body["profile_username"] is None
        assert not body["posts_count"]
        assert body["refreshing"] is False
