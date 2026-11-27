"""Pinterest board discovery, settings and upload plumbing without provider writes."""
import asyncio
from contextlib import asynccontextmanager
import json
from types import SimpleNamespace

import httpx
import pytest
from fastapi import HTTPException

import app as app_module
import mcp_server
import social_platforms as social
from cloud import auth, autopilot, scopes


PROFILE = "owner"
BOARD = "987654321098765432"
MISSING = "876543210987654321"


def test_pinterest_discovery_and_posting_use_the_resolved_profile(monkeypatch, tmp_path):
    lookups, uploads = [], []
    job_id = "10000000-0000-4000-8000-000000000001"
    clip_file = tmp_path / job_id / "clip.mp4"
    clip_file.parent.mkdir()
    clip_file.touch()
    job = {"user_id": PROFILE, "result": {"clips": [{"clip_index": 4, "video_url": f"/videos/{job_id}/clip.mp4"}]}}

    async def resolve(request, body_key=None):
        return "managed-key", PROFILE

    async def user(request):
        return SimpleNamespace(id=PROFILE)

    async def no_op(*args):
        pass

    async def no_trial(request, job):
        return None, None

    def vendor(request):
        assert request.method == "GET" and request.url.path == "/api/uploadposts/pinterest/boards"
        assert request.headers["Authorization"] == "Apikey managed-key"
        lookups.append(dict(request.url.params))
        return httpx.Response(200, json={"boards": [
            {"id": BOARD, "name": "Creator board"},
            {"id": MISSING, "name": "Secret board", "privacy": "SECRET"},
        ]})

    def upload(url, headers, data, path, filename, timeout):
        assert path == str(clip_file)
        uploads.append(data)
        return httpx.Response(202, json={"success": True})

    monkeypatch.setattr(app_module, "BILLING_ENABLED", True)
    monkeypatch.setattr(app_module, "OUTPUT_DIR", str(tmp_path))
    monkeypatch.setattr(app_module, "jobs", {job_id: job})
    monkeypatch.setattr(app_module, "resolve_upload_post", resolve)
    monkeypatch.setattr(app_module, "_user_from_request", user)
    monkeypatch.setattr(app_module, "_ensure_job_files", no_op)
    monkeypatch.setattr(app_module, "_assert_job_owner", no_op)
    monkeypatch.setattr(app_module, "_check_social_post_rate", no_op)
    monkeypatch.setattr(app_module, "_claim_free_post", no_trial)
    monkeypatch.setattr(app_module, "_post_video_blocking", upload)
    monkeypatch.setattr(social, "upload_post_client", lambda timeout: httpx.AsyncClient(
        transport=httpx.MockTransport(vendor)))

    async def exercise():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app_module.app), base_url="http://test") as client:
            fetched = await client.get("/api/social/pinterest-boards", params={"user": "another-profile"})
            assert fetched.json() == {"boards": [{"id": BOARD, "name": "Creator board"}]}
            body = {"job_id": job_id, "clip_index": 4, "platforms": ["pinterest"], "user_id": "another-profile"}
            assert (await client.post("/api/social/post", json=body)).status_code == 400
            assert (await client.post("/api/social/post", json={**body, "pinterest_board_id": MISSING})).status_code == 400
            assert not uploads
            body["pinterest_board_id"] = BOARD
            assert (await client.post("/api/social/post", json=body)).json() == {"success": True}
            assert (await client.post("/api/social/post", json={**body, "scheduled_date": "2027-01-01T12:00:00"})).status_code == 200

    asyncio.run(exercise())
    assert lookups == [{"profile": PROFILE}] * 4
    assert len(uploads) == 2
    assert all(row["pinterest_board_id"] == BOARD and row["user"] == PROFILE for row in uploads)
    assert "scheduled_date" not in uploads[0]
    assert uploads[1]["scheduled_date"] == "2027-01-01T12:00:00"


def test_autopilot_persists_a_verified_board_and_skips_it_when_removed(monkeypatch):
    prefs = autopilot.AutopilotSettings(
        user_id=PROFILE, autopublish=True, publish_platforms=["youtube"], timezone="UTC")
    lookups, uploads, states = [], [], []
    available = {BOARD}

    class FakeSession:
        @asynccontextmanager
        async def begin(self):
            yield

        async def get(self, model, owner):
            assert owner == PROFILE
            return prefs

    @asynccontextmanager
    async def session():
        yield FakeSession()

    async def owner(request):
        return SimpleNamespace(id=PROFILE)

    async def profile(user_id):
        assert user_id == PROFILE
        return PROFILE

    async def validate(_key, profile, board):
        lookups.append((profile, board))
        if board not in available:
            raise social.PinterestBoardError("Unavailable", 400)

    async def upload(profile, path, clip, platforms, when, timezone):
        uploads.append((platforms, clip["pinterest_board_id"]))
        return True

    async def update(run_id, **fields):
        states.append(fields)

    monkeypatch.setattr(auth, "get_current_user_required", owner)
    monkeypatch.setattr(autopilot.database, "session", session)
    monkeypatch.setattr(autopilot, "_profile_for", profile)
    monkeypatch.setattr(social, "validate_pinterest_board", validate)
    monkeypatch.setattr(autopilot, "_upload_clip", upload)
    monkeypatch.setattr(autopilot, "_update_run", update)

    async def exercise():
        with pytest.raises(HTTPException) as error:
            await autopilot.put_autopilot(autopilot.AutopilotUpdate(pinterest_board_id=MISSING), None)
        assert error.value.status_code == 400
        selected = autopilot.AutopilotUpdate(pinterest_board_id=BOARD, publish_platforms=["pinterest", "youtube"])
        assert (await autopilot.put_autopilot(selected, None))["settings"]["pinterest_board_id"] == BOARD
        items = [({}, "mock-clip.mp4")]
        await autopilot._publish("run", PROFILE, items, ["youtube", "pinterest"], prefs)
        available.clear()
        await autopilot._publish("run", PROFILE, items, ["youtube", "pinterest"], prefs)
        await autopilot._publish("run", PROFILE, items, ["pinterest"], prefs)
        cleared = autopilot.AutopilotUpdate(pinterest_board_id=None, publish_platforms=["youtube"])
        assert (await autopilot.put_autopilot(cleared, None))["settings"]["pinterest_board_id"] is None

    asyncio.run(exercise())
    assert lookups == [(PROFILE, MISSING)] + [(PROFILE, BOARD)] * 4
    assert uploads == [(["youtube", "pinterest"], BOARD), (["youtube"], BOARD)]
    assert states == [{"posted_count": 1, "reason": None},
                      {"posted_count": 1, "reason": "pinterest_board_unavailable"},
                      {"posted_count": 0, "reason": "pinterest_board_unavailable"}]


def test_mcp_uses_board_discovery_and_passes_the_selected_id():
    calls = []

    def handler(request):
        calls.append(request.url.path)
        if request.method == "GET":
            assert request.url.params["user"] == PROFILE
            return httpx.Response(200, json={"boards": [{"id": BOARD, "name": "Creator board"}]})
        body = json.loads(request.content)
        assert body["pinterest_board_id"] == BOARD and body["user_id"] == PROFILE
        return httpx.Response(200, json={"success": True})

    async def exercise():
        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(base_url="http://test", transport=transport) as client:
            listed, error = await mcp_server._tool_list_pinterest_boards(client, {"profile": PROFILE})
            assert not error and listed["boards"][0]["id"] == BOARD
            posted, error = await mcp_server._tool_publish_clip(client, {
                "job_id": "job", "clip_index": 0, "platforms": ["pinterest"],
                "pinterest_board_id": BOARD, "profile": PROFILE,
            })
            assert not error and posted["success"]

    asyncio.run(exercise())
    assert calls == ["/api/social/pinterest-boards", "/api/social/post"]
    assert scopes.allows(scopes.MCP_SCOPE, "GET", calls[0])
    assert not scopes.allows(scopes.MCP_SCOPE, "POST", calls[0])


def test_cli_board_list_and_publishing_forward_self_host_profile(monkeypatch, capsys):
    from cli import openshorts_cli as cli

    def request(method, path, body=None):
        if path == "/api/status/job":
            return 200, {"status": "completed", "result": {"clips": [{"clip_index": 4, "title": "Last clip"}]}}
        if method == "GET":
            assert path == "/api/social/pinterest-boards?user=owner"
            return 200, {"boards": [{"id": BOARD, "name": "Creator board"}]}
        assert path == "/api/social/post"
        assert body["user_id"] == PROFILE and body["pinterest_board_id"] == BOARD and body["clip_index"] == 4
        return 202, {"success": True}

    monkeypatch.setattr(cli, "_request", request)
    cli.main(["pinterest-boards", "--profile", PROFILE])
    assert BOARD in capsys.readouterr().out
    cli.main(["clips", "job"])
    assert "[4] Last clip" in capsys.readouterr().out
    cli.main(["publish", "job", "4", "--platforms", "pinterest", "--profile", PROFILE,
              "--pinterest-board-id", BOARD])
    assert "publishing:" in capsys.readouterr().out


def test_mcp_clip_summaries_preserve_sparse_indices():
    clips = [{"clip_index": 3, "title": "Recovered clip"}, {"clip_index": 8, "title": "Later clip"}]
    assert [summary["index"] for summary in mcp_server._clip_summaries("job", {"clips": clips})] == [3, 8]
