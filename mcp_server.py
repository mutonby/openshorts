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
import re
from typing import Optional

import httpx
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, Response

import mcp_ui
from subtitles import CAPTION_PRESETS, line_budget

router = APIRouter()

PROTOCOL_VERSIONS = ("2025-06-18", "2025-03-26", "2024-11-05")
SERVER_INFO = {"name": "openshorts", "title": "OpenShorts", "version": "1.0.0"}
# Kept descriptive: app-directory review flags instructions that read as
# commands to the model (capitals, "do not use other tools", upsell steps).
INSTRUCTIONS = (
    "OpenShorts turns long videos (YouTube URLs or direct video files) into "
    "short vertical clips. process_video takes the video URL as the user gave "
    "it; OpenShorts downloads, transcribes and analyses the video on its own "
    "servers, so the video does not need to be opened or transcribed first. "
    "Typical flow: process_video, then get_job_status until the job is "
    "'completed' (a job takes a few minutes; checking every 30-60 seconds is "
    "enough, or pass webhook_url), then list_clips, and optionally "
    "add_subtitles, recut_clip or publish_clip. get_quota shows the minutes "
    "left. If a tool reports quota_exceeded or no_plan, it also returns an "
    "upgrade_url the user can open to add minutes. Processing requires the "
    "user to own the video or hold the rights to it: confirm this with the "
    "user, then pass confirm_rights=true. publish_clip posts only to the "
    "accounts the user connected, and only when the user asks for it."
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
            "Start clipping a video from its URL. OpenShorts downloads the "
            "source itself, transcribes it, finds the most viral moments with AI "
            "and renders vertical (9:16) clips. Captions and the AI hook line are "
            "burned by default; pass captions=false or auto_hook=false to skip either. "
            "blur_sensitive defaults to false and blurs faces, license plates and "
            "on-screen personal details — set it only when the user asks. "
            "Takes the URL as the user gave it; the video does not need to be "
            "opened or transcribed beforehand. Returns a job_id immediately — the work takes minutes; "
            "poll get_job_status or pass webhook_url to be called back. The "
            "caller must own the content or hold the rights to process it "
            "(confirm_rights)."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "source_url": {
                    "type": "string",
                    "description": "Public video URL, passed through exactly as the user gave it "
                                   "(YouTube watch/short/live URL or a direct video file URL). "
                                   "The server does the downloading. Omit when using upload_id.",
                },
                "upload_id": {
                    "type": "string",
                    "description": "Instead of source_url: the id from create_upload after the "
                                   "file was PUT to its upload_url. Use when the user gave you a "
                                   "video file rather than a link.",
                },
                "auto_hook": {
                    "type": "boolean",
                    "description": "Burn the AI-written hook line (the clip's title) over the first "
                                   "seconds of each clip, as the dashboard does. Default true; set "
                                   "false for clean clips.",
                },
                "hook_style": {
                    "type": "string",
                    "enum": ["pill", "classic", "dark", "yellow", "red", "outline", "outline_yellow"],
                    "description": "Look of the hook text (with auto_hook). Default pill.",
                },
                "captions": {
                    "type": "boolean",
                    "description": "Default true: burn word-level captions on every clip. Set false "
                                   "when the source already has subtitles burned in (they would "
                                   "stack) or the user wants clean clips; add_subtitles can still "
                                   "caption a clip later.",
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
                "target_clips": {
                    "type": "integer", "minimum": 1, "maximum": 15,
                    "description": "How many clips to aim for. A target, not a guarantee: "
                                   "fewer come back when the material doesn't hold them. "
                                   "Default: the AI decides (usually 2-6).",
                },
                "clip_min_seconds": {
                    "type": "number", "minimum": 5, "maximum": 175,
                    "description": "Minimum clip length in seconds (default 15).",
                },
                "clip_max_seconds": {
                    "type": "number", "minimum": 10, "maximum": 180,
                    "description": "Maximum clip length in seconds (default 60). Must be ≥ 5s above the minimum.",
                },
                "blur_sensitive": {
                    "type": "boolean",
                    "description": "Default false. Blur faces, license plates, QR codes and "
                                   "on-screen emails, phone numbers and card numbers in each "
                                   "clip. Leave false unless the user asked: talking-head "
                                   "clips need the faces. Detection stays on the server.",
                },
            },
            "required": ["confirm_rights"],
        },
    },
    {
        "name": "create_upload",
        "title": "Reserve an upload slot for a local video file",
        "description": (
            "Use when the user hands you a video FILE instead of a link. Returns "
            "an upload_url: send the file's raw bytes to it with an HTTP PUT "
            "(for example `curl -T video.mp4 <upload_url>`), then call "
            "process_video with the returned upload_id. Unused slots are "
            "deleted after expires_in seconds (6 h); max size is max_mb. "
            "If the upload_url cannot be reached from where you run, the user "
            "can run the curl command themselves or share a public link to "
            "the video instead."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "filename": {"type": "string", "description": "Original file name (optional, for the extension)."},
            },
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
            "Re-style the captions of one clip (clips already ship with the "
            "'default' preset). Easiest: pass a preset; any other field you send "
            "overrides that part of it. Without a preset, style 'karaoke' "
            "highlights the active word."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "job_id": {"type": "string"},
                "clip_index": {"type": "integer", "description": "0-based index from list_clips."},
                "preset": {"type": "string", "enum": list(CAPTION_PRESETS),
                           "description": (
                               "default: Anton caps, yellow active word. "
                               "hormozi: words appear as spoken, yellow active word. "
                               "pill: purple box behind the active word. "
                               "lime: lime box behind the active word. "
                               "oneword: one big word at a time. "
                               "clean: no outline, soft shadow, sentence case.")},
                "style": {"type": "string", "enum": ["classic", "karaoke"]},
                "position": {"type": "string", "enum": ["top", "middle", "bottom"]},
                "font_size": {"type": "integer",
                              "description": "44 is the default (M); 34 = S, 56 = L, 70 = XL."},
                "font_name": {"type": "string",
                              "description": "Anton, Montserrat ExtraBold, Verdana, Arial, Impact, Georgia..."},
                "font_color": {"type": "string", "description": "Hex color, e.g. #FFFFFF."},
                "highlight_color": {"type": "string", "description": "Karaoke active-word color (or box color with effect 'highlight')."},
                "effect": {"type": "string", "enum": ["none", "pop", "glow", "box", "highlight"],
                           "description": "Karaoke active-word effect; 'highlight' = solid box behind it."},
                "reveal": {"type": "boolean", "description": "Karaoke: words appear as they are spoken."},
                "one_word": {"type": "boolean", "description": "Show one word at a time."},
                "shadow": {"type": "integer", "description": "Drop shadow depth 0-6."},
                "border_width": {"type": "integer", "description": "Outline 0-10 (0 needs a shadow to stay readable)."},
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
            "in the ORIGINAL source video: start from the clip's source_segments "
            "in list_clips (e.g. end 5 s earlier = lower the last end by 5). "
            "Segments are concatenated in the "
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
                "framing": {
                    "type": "string", "enum": ["auto", "full", "track"],
                    "description": "Layout override: 'full' shows the whole source frame "
                                   "(no side-cropping), 'track' forces the subject-tracking "
                                   "crop, 'auto' resets to the AI classifier. Omit to keep "
                                   "the clip's current framing. Non-auto values re-run the "
                                   "reframe engine and need the retained source video.",
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

# Behaviour hints hosts use to decide what needs the user's confirmation; the
# ChatGPT app review requires all three on every tool. openWorld = reaches past
# the user's OpenShorts account (fetches an outside URL, posts publicly).
_READ = {"readOnlyHint": True, "destructiveHint": False, "openWorldHint": False, "idempotentHint": True}
_ANNOTATIONS = {
    "process_video": {"readOnlyHint": False, "destructiveHint": False, "openWorldHint": True},
    "create_upload": {"readOnlyHint": False, "destructiveHint": False, "openWorldHint": False},
    "get_job_status": _READ,
    "list_clips": _READ,
    "get_quota": _READ,
    "add_subtitles": {"readOnlyHint": False, "destructiveHint": False, "openWorldHint": False},
    # Replaces the clip's current render with the new cut.
    "recut_clip": {"readOnlyHint": False, "destructiveHint": True, "openWorldHint": False},
    "publish_clip": {"readOnlyHint": False, "destructiveHint": False, "openWorldHint": True},
}
for _tool in TOOLS:
    _tool["annotations"] = {"title": _tool["title"], **_ANNOTATIONS[_tool["name"]]}


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
    if not args.get("source_url") and not args.get("upload_id"):
        return {"error": "Give source_url (a public video link) or upload_id (from create_upload)."}, True
    body = {
        "url": args.get("source_url"),
        "upload_id": args.get("upload_id"),
        "acknowledged": True,
        "layouts": args.get("layouts") or [],
        "output_format": args.get("output_format"),
        "force_low_quality": bool(args.get("force_low_quality")),
        "webhook_url": args.get("webhook_url"),
        "webhook_secret": args.get("webhook_secret"),
    }
    for k in ("target_clips", "clip_min_seconds", "clip_max_seconds", "captions"):
        if args.get(k) is not None:
            body[k] = args[k]
    # Same default as the dashboard: hook on unless the caller opts out. The
    # REST endpoint keeps "absent = off" for older integrations; the MCP
    # tool is newer than the hook and its users expect the dashboard output.
    if args.get("auto_hook", True):
        body["auto_hook"] = True
        if args.get("hook_style"):
            body["auto_hook_style"] = args["hook_style"]
    # Off unless the caller opts in. Unlike the hook, this must not default
    # to the dashboard's "remembered" choice: an agent turns it on only when
    # the user asked for faces and plates to be hidden.
    if args.get("blur_sensitive") is True:
        body["blur_sensitive"] = True
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


async def _tool_create_upload(client, args):
    resp = await client.post("/api/uploads", json={"filename": args.get("filename") or "video.mp4"})
    if resp.status_code >= 400:
        return _api_error(resp), True
    return resp.json(), False


async def _job_call(client, method, path, job_id, **kw):
    # Cloud jobs leave local disk an hour after they finish (their clips live
    # on in R2), so an agent coming back later gets 404 "Job not found". Bring
    # the project back the way the dashboard reopens one, then retry once.
    send = getattr(client, method.lower())
    resp = await send(path, **kw)
    if resp.status_code == 404:
        restored = await client.post(f"/api/projects/{job_id}/restore")
        if restored.status_code < 400:
            resp = await send(path, **kw)
    return resp


async def _tool_get_job_status(client, args):
    resp = await _job_call(client, "GET", f"/api/status/{args['job_id']}", args["job_id"])
    if resp.status_code >= 400:
        return _api_error(resp), True
    data = resp.json()
    out = {"job_id": args["job_id"], "status": data.get("status"),
           "recent_logs": (data.get("logs") or [])[-_LOG_TAIL:]}
    if data.get("status") == "completed":
        out["clips"] = _clip_summaries(args["job_id"], data.get("result") or {})
    return out, data.get("status") == "failed"


def _source_segments(clip):
    # Where the clip sits in the ORIGINAL video: what recut_clip edits. A clip
    # that was already recut carries its exact segment list in its recipe.
    recipe = (clip.get("recipe") or {}).get("segments")
    if recipe:
        return [{"start": s["start"], "end": s["end"]} for s in recipe]
    if isinstance(clip.get("start"), (int, float)) and isinstance(clip.get("end"), (int, float)):
        return [{"start": clip["start"], "end": clip["end"]}]
    return []


def _media_secret() -> str:
    if not _billing_enabled():
        return ""
    try:
        from cloud.config import settings
        return settings.jwt_secret or ""
    except Exception:
        return ""


def _clip_url(base, rel):
    """Absolute and, in cloud mode, signed (media_auth): the agent and whatever
    it hands the link to (a downloader, Upload-Post) have no session."""
    if not rel.startswith("/"):
        return rel
    secret = _media_secret()
    if secret:
        import media_auth
        return media_auth.sign_absolute_url(base, rel, secret)
    return f"{base}{rel}" if base else rel


def _with_download_url(data):
    """A fetchable link next to the relative ``new_video_url`` an edit returns."""
    rel = data.get("new_video_url") if isinstance(data, dict) else None
    if isinstance(rel, str) and rel.startswith("/"):
        data["download_url"] = _clip_url(os.environ.get("PUBLIC_API_URL", "").rstrip("/"), rel)
    return data


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
            "video_url": _clip_url(base, rel),
            "source_segments": _source_segments(clip),
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
    out = {"plan": data.get("plan"), "entitled": data.get("entitled"),
           "minutes": data.get("minutes"),
           "upload_post_profile": data.get("upload_post_profile")}
    if data.get("plan") in (None, "free") or not data.get("entitled"):
        # The link an agent hands the user when the free minutes run short.
        from cloud.config import settings
        out["upgrade_url"] = f"{settings.frontend_url}/#/pricing?plan=starter&src=api"
    return out, False


async def _tool_add_subtitles(client, args):
    body = {"job_id": args["job_id"], "clip_index": args["clip_index"]}
    for k in ("preset", "style", "position", "font_size", "font_name", "font_color",
              "highlight_color", "effect", "reveal", "shadow", "border_width",
              "uppercase"):
        if args.get(k) is not None:
            body[k] = args[k]
    if args.get("one_word"):
        body["max_chars"] = 1
    elif ("font_size" in body or "font_name" in body) and \
            CAPTION_PRESETS.get(str(body.get("preset") or "").lower(), {}).get("max_chars") != 1:
        # A resized or refonted line gets the budget that keeps it on one
        # line, as the dashboard does; the preset's font fills the gap.
        base = CAPTION_PRESETS.get(str(body.get("preset") or "default").lower(),
                                   CAPTION_PRESETS["default"])
        body["max_chars"] = line_budget(body.get("font_name", base["font_name"]),
                                        body.get("font_size", base["font_size"]))
    resp = await _job_call(client, "POST", "/api/subtitle", args["job_id"], json=body)
    if resp.status_code >= 400:
        return _api_error(resp), True
    return _with_download_url(resp.json()), False


async def _tool_recut_clip(client, args):
    body = {"job_id": args["job_id"], "clip_index": args["clip_index"],
            "segments": args["segments"]}
    for k in ("snap_to_words", "reapply_captions", "framing"):
        if args.get(k) is not None:
            body[k] = args[k]
    resp = await _job_call(client, "POST", "/api/clip/rerender", args["job_id"], json=body)
    if resp.status_code >= 400:
        return _api_error(resp), True
    return _with_download_url(resp.json()), False


async def _tool_publish_clip(client, args):
    body = {"job_id": args["job_id"], "clip_index": args["clip_index"],
            "platforms": args["platforms"]}
    for k in ("title", "description", "scheduled_date", "timezone"):
        if args.get(k) is not None:
            body[k] = args[k]
    resp = await _job_call(client, "POST", "/api/social/post", args["job_id"], json=body)
    if resp.status_code >= 400:
        return _api_error(resp), True
    return resp.json(), False


_TOOL_IMPLS = {
    "process_video": _tool_process_video,
    "create_upload": _tool_create_upload,
    "get_job_status": _tool_get_job_status,
    "list_clips": _tool_list_clips,
    "get_quota": _tool_get_quota,
    "add_subtitles": _tool_add_subtitles,
    "recut_clip": _tool_recut_clip,
    "publish_clip": _tool_publish_clip,
}


_ID_ARGS = ("job_id", "upload_id")
_ID_RE = re.compile(r"[A-Za-z0-9-]{1,64}")


async def call_tool(request: Request, name: str, args: dict) -> tuple[dict, bool]:
    impl = _TOOL_IMPLS.get(name)
    if impl is None:
        return {"error": f"Unknown tool: {name}"}, True
    # Ids are spliced into API paths: anything but an id shape could walk
    # the path ("../") or add a query to another endpoint.
    for key in _ID_ARGS:
        value = (args or {}).get(key)
        if value is not None and not (isinstance(value, str) and _ID_RE.fullmatch(value)):
            return {"error": f"Invalid {key}."}, True
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
        return _rpc_result(msg_id, {"resources": [
            {**r, "_meta": mcp_ui.widget_meta()} for r in mcp_ui.RESOURCES]})
    if method == "resources/templates/list":
        return _rpc_result(msg_id, {"resourceTemplates": []})
    if method == "resources/read":
        uri = (msg.get("params") or {}).get("uri") or ""
        # Per-call URIs (ui://openshorts/clip-picker/<job>) resolve to the same
        # template; the data those carried was baked into the tool result.
        if uri.startswith(mcp_ui.CLIP_PICKER_BASE):  # any version, incl. per-call /<job>
            return _rpc_result(msg_id, {"contents": [{
                "uri": uri,
                "mimeType": mcp_ui.MIME_TYPE,
                "text": mcp_ui.clip_picker_html(),
                "_meta": mcp_ui.widget_meta(),
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
                "_meta": mcp_ui.widget_meta(),
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
        # OAuth-capable clients (claude.ai, ChatGPT) read resource_metadata off
        # this header and run the login flow themselves; everyone else gets the
        # API-key hint in the body.
        from cloud import mcp_oauth
        u = request.base_url
        return JSONResponse(
            {"error": "Authentication required. Connect with OAuth (claude.ai, ChatGPT) "
                      "or pass an OpenShorts API key: Authorization: Bearer osk_... "
                      "(create one in the dashboard)."},
            status_code=401,
            headers={"WWW-Authenticate": mcp_oauth.www_authenticate(f"{u.scheme}://{u.netloc}")},
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
