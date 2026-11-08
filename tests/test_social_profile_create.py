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
