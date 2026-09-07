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

    def test_add_subtitles_offers_every_caption_preset(self):
        # The enum is what lets an agent pick "hormozi" by name; it must not
        # drift from the server-side list.
        from subtitles import CAPTION_PRESETS
        tool = next(t for t in TOOLS if t["name"] == "add_subtitles")
        assert tool["inputSchema"]["properties"]["preset"]["enum"] == list(CAPTION_PRESETS)

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
