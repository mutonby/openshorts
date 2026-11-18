"""cloud/usage_events.py: which requests become events, which request fields
ride along (never free text), and that the response is untouched."""
import asyncio
import json
from types import SimpleNamespace

import pytest

from cloud import usage_events as ue


def test_event_routing():
    assert ue._event_for("POST", "/api/hook") == ("ClipEdited", {"tool": "hook"})
    assert ue._event_for("POST", "/api/social/post") == ("SocialPostSent", {})
    assert ue._event_for("GET", "/api/jobs/abc/download-all") == ("ClipsDownloadedAll", {})
    assert ue._event_for("GET", "/api/hook") is None
    assert ue._event_for("POST", "/api/process") is None


def test_body_props_keep_ids_and_drop_free_text():
    body = json.dumps({"job_id": "j", "clip_index": 2, "text": "my secret hook",
                       "mode": "pro", "preset": "cine", "target_language": "pt-BR"}).encode()
    props = ue._body_props("ClipEdited", body)
    assert props == {"clip_index": 2, "mode": "pro", "preset": "cine", "target_language": "pt-BR"}
    social = ue._body_props("SocialPostSent", json.dumps(
        {"clip_index": 0, "platforms": ["tiktok", "instagram", "x; drop"], "title": "t"}).encode())
    assert social == {"clip_index": 0, "platforms": "instagram,tiktok"}
    assert ue._body_props("ClipEdited", b"not json") == {}
    assert ue._body_props("ClipEdited", json.dumps({"mode": "a b c"}).encode()) == {}


def _run(app, path, method="POST", body=b"{}", headers=None):
    sent = []

    async def receive():
        return {"type": "http.request", "body": body, "more_body": False}

    async def send(message):
        sent.append(message)

    scope = {"type": "http", "method": method, "path": path,
             "headers": [(k.encode(), v.encode()) for k, v in (headers or {}).items()]}

    async def go():
        await ue.UsageEventsMiddleware(app)(scope, receive, send)
        await asyncio.sleep(0)
        await asyncio.sleep(0)

    asyncio.run(go())
    return sent


def _endpoint(status):
    async def app(scope, receive, send):
        await receive()
        await send({"type": "http.response.start", "status": status, "headers": []})
        await send({"type": "http.response.body", "body": b"ok"})
    return app


@pytest.fixture
def tracked(monkeypatch):
    box = []
    monkeypatch.setenv("OPENPANEL_CLIENT_ID", "cid")

    async def fake_user(request):
        return SimpleNamespace(id="u1", plan="starter")

    import cloud.auth as auth
    monkeypatch.setattr(auth, "get_current_user_optional", fake_user)
    monkeypatch.setattr(ue.analytics, "track",
                        lambda name, user_id=None, **p: box.append((name, user_id, p)))
    return box


def test_tool_call_is_reported_after_the_response(tracked):
    sent = _run(_endpoint(200), "/api/subtitle",
                body=json.dumps({"clip_index": 1, "preset": "cine"}).encode(),
                headers={"authorization": "Bearer osk_abc"})
    assert [m["type"] for m in sent] == ["http.response.start", "http.response.body"]
    assert tracked == [("ClipEdited", "u1", {"tool": "subtitles", "clip_index": 1,
                                             "preset": "cine", "via_api": True,
                                             "ok": True, "status": 200, "plan": "starter"})]


def test_failure_status_is_reported_as_not_ok(tracked):
    _run(_endpoint(402), "/api/hook")
    assert tracked[0][2]["ok"] is False and tracked[0][2]["status"] == 402


def test_other_paths_and_unconfigured_analytics_are_untouched(tracked, monkeypatch):
    _run(_endpoint(200), "/api/process")
    assert tracked == []
    monkeypatch.delenv("OPENPANEL_CLIENT_ID")
    _run(_endpoint(200), "/api/hook")
    assert tracked == []


def test_anonymous_caller_sends_nothing(tracked, monkeypatch):
    async def nobody(request):
        return None
    import cloud.auth as auth
    monkeypatch.setattr(auth, "get_current_user_optional", nobody)
    _run(_endpoint(200), "/api/hook")
    assert tracked == []
