"""MCP protocol layer (mcp_server.handle_message).

The handler is deliberately pure — tool execution is injected — so the whole
JSON-RPC surface an MCP client depends on (initialize negotiation, silent
notifications, tools/list, tools/call wrapping, error codes) is testable
without the FastAPI app or a database.
"""
import asyncio

from mcp_server import PROTOCOL_VERSIONS, TOOLS, _TOOL_IMPLS, handle_message


async def _ok_tool(name, args):
    return {"echo": {"name": name, "args": args}}, False


async def _err_tool(name, args):
    return {"error": "boom"}, True


def _run(msg, tool_caller=_ok_tool):
    return asyncio.run(handle_message(msg, tool_caller))


class TestInitialize:
    def test_echoes_a_supported_client_version(self):
        resp = _run({"jsonrpc": "2.0", "id": 1, "method": "initialize",
                     "params": {"protocolVersion": PROTOCOL_VERSIONS[-1]}})
        assert resp["result"]["protocolVersion"] == PROTOCOL_VERSIONS[-1]

    def test_falls_back_to_latest_for_unknown_versions(self):
        resp = _run({"jsonrpc": "2.0", "id": 1, "method": "initialize",
                     "params": {"protocolVersion": "1999-01-01"}})
        assert resp["result"]["protocolVersion"] == PROTOCOL_VERSIONS[0]

    def test_advertises_tools_capability(self):
        resp = _run({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}})
        assert "tools" in resp["result"]["capabilities"]
        assert resp["result"]["serverInfo"]["name"] == "openshorts"


class TestNotificationsAndErrors:
    def test_notification_returns_none(self):
        assert _run({"jsonrpc": "2.0", "method": "notifications/initialized"}) is None

    def test_unknown_method_is_32601(self):
        resp = _run({"jsonrpc": "2.0", "id": 7, "method": "does/not-exist"})
        assert resp["error"]["code"] == -32601
        assert resp["id"] == 7

    def test_non_jsonrpc_message_is_32600(self):
        resp = _run({"not": "jsonrpc"})
        assert resp["error"]["code"] == -32600

    def test_ping_answers_empty_result(self):
        assert _run({"jsonrpc": "2.0", "id": 2, "method": "ping"})["result"] == {}


class TestTools:
    def test_list_matches_the_implementations(self):
        resp = _run({"jsonrpc": "2.0", "id": 3, "method": "tools/list"})
        listed = {t["name"] for t in resp["result"]["tools"]}
        assert listed == set(_TOOL_IMPLS)

    def test_every_tool_declares_an_object_schema(self):
        for tool in TOOLS:
            assert tool["inputSchema"]["type"] == "object", tool["name"]
            for req in tool["inputSchema"].get("required", []):
                assert req in tool["inputSchema"]["properties"], tool["name"]

    def test_every_tool_declares_behaviour_hints(self):
        # The ChatGPT app review rejects a tool missing any of these three.
        for tool in TOOLS:
            hints = tool["annotations"]
            for key in ("readOnlyHint", "destructiveHint", "openWorldHint"):
                assert isinstance(hints[key], bool), (tool["name"], key)
        by_name = {t["name"]: t["annotations"] for t in TOOLS}
        assert by_name["publish_clip"]["openWorldHint"] is True
        assert by_name["get_quota"]["readOnlyHint"] is True

    def test_call_wraps_result_as_text_and_structured(self):
        resp = _run({"jsonrpc": "2.0", "id": 4, "method": "tools/call",
                     "params": {"name": "get_quota", "arguments": {"a": 1}}})
        result = resp["result"]
        assert result["isError"] is False
        assert result["structuredContent"]["echo"]["args"] == {"a": 1}
        assert result["content"][0]["type"] == "text"

    def test_call_marks_tool_failures_as_is_error(self):
        resp = _run({"jsonrpc": "2.0", "id": 5, "method": "tools/call",
                     "params": {"name": "get_quota", "arguments": {}}},
                    tool_caller=_err_tool)
        assert resp["result"]["isError"] is True


class TestClipSummaries:
    def test_source_segments_tell_recut_where_the_clip_is(self):
        from mcp_server import _clip_summaries
        clips = _clip_summaries("j", {"clips": [
            {"start": 10.0, "end": 40.5, "video_url": "/videos/j/a.mp4"},
            {"start": 5, "end": 30, "recipe": {"segments": [
                {"start": 5, "end": 12, "text": "x"}, {"start": 20, "end": 30}]}},
            {"video_url": "/videos/j/c.mp4"},
        ]})
        assert clips[0]["source_segments"] == [{"start": 10.0, "end": 40.5}]
        assert clips[1]["source_segments"] == [{"start": 5, "end": 12}, {"start": 20, "end": 30}]
        assert clips[2]["source_segments"] == []


class TestExpiredJobs:
    def _client(self, on_disk):
        import httpx
        calls = []

        def handler(req):
            calls.append((req.method, req.url.path))
            if req.url.path.endswith("/restore"):
                on_disk.append(True)
                return httpx.Response(200, json={"status": "completed"})
            if not on_disk:
                return httpx.Response(404, json={"detail": "Job not found"})
            return httpx.Response(200, json={"status": "completed", "result": {"clips": []}})
        return httpx.AsyncClient(transport=httpx.MockTransport(handler),
                                 base_url="http://api"), calls

    def test_a_job_gone_from_disk_is_restored_from_the_archive(self):
        from mcp_server import _tool_get_job_status
        client, calls = self._client([])
        out, is_error = asyncio.run(_tool_get_job_status(client, {"job_id": "j"}))
        assert not is_error and out["status"] == "completed"
        assert calls == [("GET", "/api/status/j"), ("POST", "/api/projects/j/restore"),
                         ("GET", "/api/status/j")]

    def test_a_failed_restore_is_reported_not_hidden_as_job_not_found(self):
        import httpx
        from mcp_server import _tool_get_job_status

        def handler(req):
            if req.url.path.endswith("/restore"):
                return httpx.Response(502, json={"detail": "Restore download failed: x"})
            return httpx.Response(404, json={"detail": "Job not found"})
        client = httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="http://api")
        out, is_error = asyncio.run(_tool_get_job_status(client, {"job_id": "j"}))
        assert is_error and out["http_status"] == 502
        assert out["error"].startswith("Restore download failed")

    def test_a_job_nobody_archived_stays_not_found(self):
        import httpx
        from mcp_server import _tool_get_job_status

        def handler(req):
            if req.url.path.endswith("/restore"):
                return httpx.Response(404, json={"detail": "Project not found"})
            return httpx.Response(404, json={"detail": "Job not found"})
        client = httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="http://api")
        out, is_error = asyncio.run(_tool_get_job_status(client, {"job_id": "j"}))
        assert is_error and out["error"] == "Job not found"

    def test_a_live_job_is_not_restored(self):
        from mcp_server import _tool_get_job_status
        client, calls = self._client([True])
        asyncio.run(_tool_get_job_status(client, {"job_id": "j"}))
        assert calls == [("GET", "/api/status/j")]


def test_ids_that_are_not_ids_never_reach_an_api_path():
    """job_id / upload_id are spliced into REST paths; "../" or "?" in one
    would address another endpoint with the caller's credentials."""
    import mcp_server

    called = []

    async def impl(client, args):
        called.append(args)
        return {}, False

    mcp_server._TOOL_IMPLS["__probe"] = impl
    try:
        for bad in ("../../api/billing/end-trial?", "a/b", "x?y=1", "", 5):
            out, is_error = asyncio.run(mcp_server.call_tool(None, "__probe", {"job_id": bad}))
            assert is_error and "Invalid job_id" in out["error"]
        out, is_error = asyncio.run(mcp_server.call_tool(None, "__probe", {"upload_id": "../x"}))
        assert is_error
        assert called == []
    finally:
        mcp_server._TOOL_IMPLS.pop("__probe", None)
