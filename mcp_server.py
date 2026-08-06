"""Remote MCP server: the whole pipeline as agent-callable tools at ``/mcp``.

Stateless Streamable-HTTP transport (plain JSON responses, no SSE, no session
ids) implemented directly over FastAPI — the protocol surface an MCP client
actually needs for tools is three methods (initialize / tools/list / tools/call)
plus notifications, and owning those ~200 lines beats carrying the full SDK as
a dependency for them.

Tools don't reimplement anything: each one is an in-process HTTP call back into
this same app (httpx ASGITransport) with the caller's auth headers forwarded.
Cloud mode therefore meters, gates and scopes agent calls exactly like
dashboard calls — an ``osk_`` API key IS the user (see cloud/api_keys.py).
Self-host keeps its BYOK semantics: no key required, ``X-Gemini-Key`` /
env fallbacks apply unchanged.

Connect with any MCP client, e.g.:
    claude mcp add --transport http openshorts https://mcp.openshorts.app/mcp \
        --header "Authorization: Bearer osk_..."

(mcp.openshorts.app is a domain alias of the API app; api.openshorts.app/mcp
serves the identical endpoint.)
"""
import json
import os
from typing import Optional

import httpx
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, Response

import mcp_ui

router = APIRouter()

PROTOCOL_VERSIONS = ("2025-06-18", "2025-03-26", "2024-11-05")
SERVER_INFO = {"name": "openshorts", "title": "OpenShorts", "version": "1.0.0"}
INSTRUCTIONS = (
    "OpenShorts turns long videos (YouTube URLs) into viral-ready vertical "
    "clips. Typical flow: process_video -> poll get_job_status until "
    "'completed' (a job takes minutes; poll every 30-60s or pass webhook_url) "
    "-> list_clips -> optionally add_subtitles / publish_clip. Check "
    "get_quota before large jobs."
)

# Headers an MCP caller may use to authenticate / bring their own keys; they are
# forwarded verbatim to the internal endpoints so every existing auth path works.
_FORWARD_HEADERS = ("authorization", "x-api-key", "x-gemini-key", "x-upload-post-key")

_LOG_TAIL = 10  # status logs are for humans; agents only need the tail


TOOLS = [
    {
        "name": "process_video",
        "title": "Process a video into short clips",
        "description": (
            "Start clipping a video: downloads the source, transcribes it, finds "
            "the most viral moments with AI and renders vertical (9:16) clips "
            "with captions. Returns a job_id immediately — the work takes "
            "minutes; poll get_job_status or pass webhook_url to be called back. "
            "The caller must own the content or hold the rights to process it "
            "(confirm_rights)."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "source_url": {
                    "type": "string",
                    "description": "Public video URL (YouTube or a direct video file URL).",
                },
                "confirm_rights": {
                    "type": "boolean",
                    "description": "Must be true: the user owns the content or has rights to process it.",
                },
                "layouts": {
                    "type": "array",
                    "items": {"type": "string",
                              "enum": ["auto", "split", "screencast", "speaker_cut", "punch_in"]},
                    "description": "Optional extra reframe layouts. 'auto' lets AI pick per video.",
                },
                "output_format": {
                    "type": "string",
                    "enum": ["auto", "vertical", "horizontal", "square"],
                    "description": "Clip aspect. Default auto (vertical).",
                },
                "webhook_url": {
                    "type": "string",
                    "description": "Optional public HTTPS URL POSTed once when the job finishes or fails.",
                },
                "webhook_secret": {
                    "type": "string",
                    "description": "Optional secret; the webhook body is then HMAC-SHA256 signed (X-OpenShorts-Signature).",
                },
                "force_low_quality": {
                    "type": "boolean",
                    "description": "Set true to proceed after a needs_confirmation low-resolution warning.",
                },
            },
            "required": ["source_url", "confirm_rights"],
        },
    },
    {
        "name": "get_job_status",
        "title": "Get processing job status",
        "description": (
            "Status of a processing job: 'queued', 'processing', 'completed' or "
            "'failed', with recent log lines and, once completed, the clips."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {"job_id": {"type": "string"}},
            "required": ["job_id"],
        },
    },
    {
        "name": "list_clips",
        "title": "List a job's clips",
        "description": (
            "The clips of a completed job, with titles, platform-ready "
            "descriptions and download URLs. In MCP Apps-capable clients the "
            "result also renders as an interactive clip picker."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {"job_id": {"type": "string"}},
            "required": ["job_id"],
        },
        # Both spellings of the tool->template link: MCP Apps hosts read
        # _meta.ui.resourceUri, the ChatGPT Apps SDK reads openai/outputTemplate.
        "_meta": {
            "ui": {"resourceUri": mcp_ui.CLIP_PICKER_URI},
            "openai/outputTemplate": mcp_ui.CLIP_PICKER_URI,
        },
    },
    {
        "name": "get_quota",
        "title": "Get plan and remaining minutes",
        "description": (
            "The authenticated user's plan and remaining processing minutes. "
            "Call before large jobs; process_video fails with quota_exceeded "
            "when the balance is insufficient."
        ),
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "add_subtitles",
        "title": "Burn styled captions onto a clip",
        "description": (
            "Re-style the captions of one clip (clips already ship with default "
            "captions). style 'karaoke' highlights the active word."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "job_id": {"type": "string"},
                "clip_index": {"type": "integer", "description": "0-based index from list_clips."},
                "style": {"type": "string", "enum": ["classic", "karaoke"]},
                "position": {"type": "string", "enum": ["top", "middle", "bottom"]},
                "font_size": {"type": "integer"},
                "font_name": {"type": "string"},
                "font_color": {"type": "string", "description": "Hex color, e.g. #FFFFFF."},
                "highlight_color": {"type": "string", "description": "Karaoke active-word color."},
                "uppercase": {"type": "boolean"},
            },
            "required": ["job_id", "clip_index"],
        },
    },
    {
        "name": "recut_clip",
        "title": "Re-cut a clip from an edited segment list",
        "description": (
            "Re-render one clip from a new list of source-video segments (the "
            "same engine behind the dashboard's clip editor). Times are seconds "
            "in the ORIGINAL source video; segments are concatenated in the "
            "given order, so you can trim, extend, drop a dead moment in the "
            "middle, or reorder. Segments inside the clip's original range "
            "re-render in seconds; going outside needs the retained source "
            "video and re-runs the reframe engine."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "job_id": {"type": "string"},
                "clip_index": {"type": "integer", "description": "0-based index from list_clips."},
                "segments": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "start": {"type": "number", "description": "Seconds in the source video."},
                            "end": {"type": "number", "description": "Seconds in the source video."},
                        },
                        "required": ["start", "end"],
                    },
                    "minItems": 1,
                    "description": "Ordered source segments the new clip is made of.",
                },
                "snap_to_words": {
                    "type": "boolean",
                    "description": "Snap each boundary onto transcript word boundaries (recommended).",
                },
                "reapply_captions": {
                    "type": "boolean",
                    "description": "Burn default captions back on after the recut (default true).",
                },
            },
            "required": ["job_id", "clip_index", "segments"],
        },
    },
    {
        "name": "publish_clip",
        "title": "Publish a clip to social platforms",
        "description": (
            "Post one clip to the user's connected accounts (TikTok lands as a "
            "draft in the app; Instagram and YouTube publish directly). Requires "
            "a connected social profile (cloud) or an Upload-Post key (self-host). "
            "Optionally schedule with an ISO-8601 scheduled_date."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "job_id": {"type": "string"},
                "clip_index": {"type": "integer"},
                "platforms": {
                    "type": "array",
                    "items": {"type": "string", "enum": ["tiktok", "instagram", "youtube"]},
                },
                "title": {"type": "string"},
                "description": {"type": "string"},
                "scheduled_date": {"type": "string", "description": "ISO-8601; omit to post now."},
                "timezone": {"type": "string"},
            },
            "required": ["job_id", "clip_index", "platforms"],
        },
    },
]


# --------------------------------------------------------------------------- #
# Internal dispatch: each tool is an in-process call to the existing REST API.
# --------------------------------------------------------------------------- #
def _client(request: Request) -> httpx.AsyncClient:
    headers = {k: v for k, v in request.headers.items() if k.lower() in _FORWARD_HEADERS}
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=request.app, raise_app_exceptions=False),
        base_url="http://openshorts.internal",
        headers=headers,
        timeout=300.0,
    )


def _api_error(resp: httpx.Response) -> dict:
    try:
        detail = resp.json().get("detail")
    except Exception:
        detail = resp.text[:300]
    return {"error": detail or f"HTTP {resp.status_code}", "http_status": resp.status_code}


async def _tool_process_video(client, args):
    if not args.get("confirm_rights"):
        return {"error": "confirm_rights must be true: the user must own the "
                         "content or hold the rights to process it."}, True
    body = {
        "url": args["source_url"],
        "acknowledged": True,
        "layouts": args.get("layouts") or [],
        "output_format": args.get("output_format"),
        "force_low_quality": bool(args.get("force_low_quality")),
        "webhook_url": args.get("webhook_url"),
        "webhook_secret": args.get("webhook_secret"),
    }
    resp = await client.post("/api/process", json=body)
    if resp.status_code >= 400:
        return _api_error(resp), True
    data = resp.json()
    if data.get("needs_confirmation"):
        data["hint"] = ("Source resolution is below the quality gate. Ask the "
                        "user, then retry with force_low_quality=true to proceed.")
        return data, False
    data["hint"] = ("Processing takes minutes. Poll get_job_status every 30-60s"
                    + ("" if body["webhook_url"] else " (or re-run with webhook_url for a callback)") + ".")
    return data, False


async def _tool_get_job_status(client, args):
    resp = await client.get(f"/api/status/{args['job_id']}")
    if resp.status_code >= 400:
        return _api_error(resp), True
    data = resp.json()
    out = {"job_id": args["job_id"], "status": data.get("status"),
           "recent_logs": (data.get("logs") or [])[-_LOG_TAIL:]}
    if data.get("status") == "completed":
        out["clips"] = _clip_summaries(args["job_id"], data.get("result") or {})
    return out, data.get("status") == "failed"


def _clip_summaries(job_id, result):
    base = os.environ.get("PUBLIC_API_URL", "").rstrip("/")
    out = []
    for i, clip in enumerate(result.get("clips") or []):
        rel = clip.get("video_url") or ""
        out.append({
            "index": i,
            "title": clip.get("title") or clip.get("video_title_for_youtube_short"),
            "duration_seconds": (round(clip["end"] - clip["start"], 1)
                                 if isinstance(clip.get("start"), (int, float))
                                 and isinstance(clip.get("end"), (int, float)) else None),
            "video_url": f"{base}{rel}" if base and rel.startswith("/") else rel,
            "youtube_title": clip.get("video_title_for_youtube_short"),
            "tiktok_description": clip.get("video_description_for_tiktok"),
            "instagram_description": clip.get("video_description_for_instagram"),
        })
    return out


async def _tool_list_clips(client, args):
    out, is_error = await _tool_get_job_status(client, args)
    if is_error:
        return out, True
    if out.get("status") != "completed":
        return {"error": f"Job is {out.get('status')}, clips are not ready yet.",
                "status": out.get("status")}, True
    return {"job_id": args["job_id"], "clips": out.get("clips") or []}, False


async def _tool_get_quota(client, args):
    resp = await client.get("/api/me")
    # 401: anonymous. 404: self-host, where /api/me isn't even mounted (the
    # cloud router only registers under BILLING_ENABLED). Neither is an error
    # from the agent's point of view — there is simply no quota to report.
    if resp.status_code in (401, 404):
        return {"self_host_or_anonymous": True,
                "note": "No authenticated cloud user; if this is a self-hosted "
                        "instance there is no minute quota."}, False
    if resp.status_code >= 400:
        return _api_error(resp), True
    data = resp.json()
    return {"plan": data.get("plan"), "entitled": data.get("entitled"),
            "minutes": data.get("minutes"),
            "upload_post_profile": data.get("upload_post_profile")}, False


async def _tool_add_subtitles(client, args):
    body = {"job_id": args["job_id"], "clip_index": args["clip_index"]}
    for k in ("style", "position", "font_size", "font_name", "font_color",
              "highlight_color", "uppercase"):
        if args.get(k) is not None:
            body[k] = args[k]
    resp = await client.post("/api/subtitle", json=body)
    if resp.status_code >= 400:
        return _api_error(resp), True
    return resp.json(), False


async def _tool_recut_clip(client, args):
    body = {"job_id": args["job_id"], "clip_index": args["clip_index"],
            "segments": args["segments"]}
    for k in ("snap_to_words", "reapply_captions"):
        if args.get(k) is not None:
            body[k] = args[k]
    resp = await client.post("/api/clip/rerender", json=body)
    if resp.status_code >= 400:
        return _api_error(resp), True
    return resp.json(), False


async def _tool_publish_clip(client, args):
    body = {"job_id": args["job_id"], "clip_index": args["clip_index"],
            "platforms": args["platforms"]}
    for k in ("title", "description", "scheduled_date", "timezone"):
        if args.get(k) is not None:
            body[k] = args[k]
    resp = await client.post("/api/social/post", json=body)
    if resp.status_code >= 400:
        return _api_error(resp), True
    return resp.json(), False


_TOOL_IMPLS = {
    "process_video": _tool_process_video,
    "get_job_status": _tool_get_job_status,
    "list_clips": _tool_list_clips,
    "get_quota": _tool_get_quota,
    "add_subtitles": _tool_add_subtitles,
    "recut_clip": _tool_recut_clip,
    "publish_clip": _tool_publish_clip,
}


async def call_tool(request: Request, name: str, args: dict) -> tuple[dict, bool]:
    impl = _TOOL_IMPLS.get(name)
    if impl is None:
        return {"error": f"Unknown tool: {name}"}, True
    try:
        async with _client(request) as client:
            return await impl(client, args or {})
    except KeyError as e:
        return {"error": f"Missing required argument: {e}"}, True
    except Exception as e:
        return {"error": f"Tool failed: {e}"}, True


# --------------------------------------------------------------------------- #
# JSON-RPC / MCP protocol layer (pure: testable without the app)
# --------------------------------------------------------------------------- #
def _rpc_error(msg_id, code, message):
    return {"jsonrpc": "2.0", "id": msg_id, "error": {"code": code, "message": message}}


def _rpc_result(msg_id, result):
    return {"jsonrpc": "2.0", "id": msg_id, "result": result}


async def handle_message(msg, tool_caller) -> Optional[dict]:
    """One JSON-RPC message in, one response dict out (None for notifications).

    ``tool_caller(name, args) -> (result_dict, is_error)`` is injected so this
    layer stays free of HTTP and app state.
    """
    if not isinstance(msg, dict) or msg.get("jsonrpc") != "2.0":
        return _rpc_error(None, -32600, "Expected a JSON-RPC 2.0 message")
    method = msg.get("method")
    msg_id = msg.get("id")

    if method is None:
        # A response from the client (has 'result'/'error') — nothing to do.
        return None
    if msg_id is None:
        return None  # notification (e.g. notifications/initialized): accept silently

    if method == "initialize":
        client_version = (msg.get("params") or {}).get("protocolVersion")
        version = client_version if client_version in PROTOCOL_VERSIONS else PROTOCOL_VERSIONS[0]
        return _rpc_result(msg_id, {
            "protocolVersion": version,
            "capabilities": {"tools": {"listChanged": False},
                             "resources": {"listChanged": False}},
            "serverInfo": SERVER_INFO,
            "instructions": INSTRUCTIONS,
        })
    if method == "ping":
        return _rpc_result(msg_id, {})
    if method == "tools/list":
        return _rpc_result(msg_id, {"tools": TOOLS})
    if method == "resources/list":
        return _rpc_result(msg_id, {"resources": mcp_ui.RESOURCES})
    if method == "resources/templates/list":
        return _rpc_result(msg_id, {"resourceTemplates": []})
    if method == "resources/read":
        uri = (msg.get("params") or {}).get("uri") or ""
        # Per-call URIs (ui://openshorts/clip-picker/<job>) resolve to the same
        # template; the data those carried was baked into the tool result.
        if uri == mcp_ui.CLIP_PICKER_URI or uri.startswith(mcp_ui.CLIP_PICKER_URI + "/"):
            return _rpc_result(msg_id, {"contents": [{
                "uri": uri,
                "mimeType": mcp_ui.MIME_TYPE,
                "text": mcp_ui.clip_picker_html(),
            }]})
        return _rpc_error(msg_id, -32002, f"Resource not found: {uri}")
    if method == "tools/call":
        params = msg.get("params") or {}
        name = params.get("name")
        result, is_error = await tool_caller(name, params.get("arguments") or {})
        content = [{"type": "text", "text": json.dumps(result, ensure_ascii=False)}]
        # A successful list_clips additionally ships the picker with its data
        # baked in, so hosts that render embedded resources need no bridge.
        # Non-UI clients ignore extra content entries.
        if name == "list_clips" and not is_error and result.get("clips"):
            content.append({"type": "resource", "resource": {
                "uri": f"{mcp_ui.CLIP_PICKER_URI}/{result.get('job_id', 'result')}",
                "mimeType": mcp_ui.MIME_TYPE,
                "text": mcp_ui.clip_picker_html(result),
            }})
        return _rpc_result(msg_id, {
            "content": content,
            "structuredContent": result,
            "isError": is_error,
        })
    return _rpc_error(msg_id, -32601, f"Method not found: {method}")


# --------------------------------------------------------------------------- #
# Transport endpoints
# --------------------------------------------------------------------------- #
def _billing_enabled() -> bool:
    return os.environ.get("BILLING_ENABLED", "").lower() in ("1", "true", "yes")


async def _authorized(request: Request) -> bool:
    """Cloud mode requires a resolvable user (API key or JWT) before any RPC.

    The internal endpoints would each reject anonymous calls anyway; failing
    once here with a clear 401 is what lets MCP clients surface 'add your API
    key' instead of a per-tool 402. Self-host stays open (BYOK)."""
    if not _billing_enabled():
        return True
    from cloud.auth import get_current_user_optional
    return (await get_current_user_optional(request)) is not None


@router.post("/mcp")
async def mcp_endpoint(request: Request):
    if not await _authorized(request):
        return JSONResponse(
            {"error": "Authentication required. Pass an OpenShorts API key: "
                      "Authorization: Bearer osk_... (create one in the dashboard)."},
            status_code=401,
            headers={"WWW-Authenticate": "Bearer"},
        )
    try:
        msg = json.loads(await request.body())
    except Exception:
        return JSONResponse(_rpc_error(None, -32700, "Parse error"), status_code=400)
    if isinstance(msg, list):
        return JSONResponse(_rpc_error(None, -32600, "Batching is not supported"),
                            status_code=400)

    async def tool_caller(name, args):
        return await call_tool(request, name, args)

    response = await handle_message(msg, tool_caller)
    if response is None:
        return Response(status_code=202)
    return JSONResponse(response)


@router.get("/mcp")
async def mcp_get():
    # Stateless server: no server-initiated SSE stream to offer.
    return Response(status_code=405, headers={"Allow": "POST"})
