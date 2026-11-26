"""GET /api/jobs lists live jobs so an agent/CLI can recover a lost job_id.

Self-host has no project history (that is cloud-only, behind BILLING_ENABLED),
and the MCP server had no way to enumerate jobs at all.
"""
import pytest

app_module = pytest.importorskip("app")
from starlette.testclient import TestClient


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(app_module, "jobs", {})
    return TestClient(app_module.app)


def test_lists_live_jobs_newest_first_with_clip_counts(client):
    app_module.jobs["old"] = {"status": "completed", "result": {"clips": [1, 2]}}
    app_module.jobs["new"] = {"status": "processing", "result": {"clips": []}}
    data = client.get("/api/jobs").json()
    assert [j["job_id"] for j in data["jobs"]] == ["new", "old"]  # newest inserted first
    assert data["count"] == 2
    by_id = {j["job_id"]: j for j in data["jobs"]}
    assert by_id["old"]["clips"] == 2 and by_id["old"]["status"] == "completed"
    assert by_id["new"]["clips"] == 0


def test_limit_caps_the_list(client):
    for i in range(5):
        app_module.jobs[f"j{i}"] = {"status": "queued"}
    assert client.get("/api/jobs?limit=2").json()["count"] == 2


def test_cloud_mode_shows_only_the_callers_own_jobs(client, monkeypatch):
    monkeypatch.setattr(app_module, "BILLING_ENABLED", True)

    async def _caller(_request):
        return "user-A"

    monkeypatch.setattr(app_module, "_owner_id_from_request", _caller)
    app_module.jobs["mine"] = {"status": "completed", "user_id": "user-A"}
    app_module.jobs["theirs"] = {"status": "completed", "user_id": "user-B"}
    app_module.jobs["ownerless"] = {"status": "completed"}
    ids = {j["job_id"] for j in client.get("/api/jobs").json()["jobs"]}
    assert ids == {"mine"}  # never another owner's job, never an ownerless one


def test_cloud_mode_anonymous_caller_sees_nothing(client, monkeypatch):
    monkeypatch.setattr(app_module, "BILLING_ENABLED", True)

    async def _anon(_request):
        return None

    monkeypatch.setattr(app_module, "_owner_id_from_request", _anon)
    app_module.jobs["mine"] = {"status": "completed", "user_id": "user-A"}
    app_module.jobs["ownerless"] = {"status": "completed"}
    assert client.get("/api/jobs").json()["count"] == 0


def test_mcp_oauth_scope_allows_the_listing():
    from cloud.scopes import allows, MCP_SCOPE
    assert allows(MCP_SCOPE, "GET", "/api/jobs")
    # and the tool is registered for the MCP server
    import mcp_server
    assert "list_jobs" in mcp_server._TOOL_IMPLS
    assert any(t["name"] == "list_jobs" for t in mcp_server.TOOLS)
