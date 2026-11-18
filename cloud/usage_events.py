"""Server-side events for what people do with their clips after delivery.

``ClipsDelivered`` says a job produced clips; nothing said what happened next.
The browser can report some of it, but only with analytics consent and past
ad-blockers. Every edit tool and the social post are API calls anyway, so this
middleware reports them from here, complete:

* ``ClipEdited``      — one of the clip tools answered (``tool``, ``ok``, ``status``)
* ``SocialPostSent``  — /api/social/post answered (``ok``, ``platforms``)
* ``ClipsDownloadedAll`` — the "download all" zip was served

Pure ASGI, so a streamed response is never buffered. The event is sent after
the response has finished, in the background, and nothing here can fail or
slow the request it rides on (analytics.track already swallows everything).
"""
import asyncio
import json
import re

from . import analytics

# POST path -> tool name. Only the endpoints a person uses to change a clip.
TOOL_PATHS = {
    "/api/edit": "ai_edit",
    "/api/clip/rerender": "rerender",
    "/api/clip/reframe": "reframe",
    "/api/clip/mode": "mode",
    "/api/subtitle": "subtitles",
    "/api/subtitle/remove": "subtitles_remove",
    "/api/hook": "hook",
    "/api/translate": "translate",
    "/api/effects/generate": "effects",
}
SOCIAL_POST_PATH = "/api/social/post"
_DOWNLOAD_ALL_RE = re.compile(r"^/api/jobs/[^/]+/download-all$")
_MAX_BODY = 64 * 1024
_TOKEN_RE = re.compile(r"^[A-Za-z0-9_-]{1,32}$")


def _event_for(method: str, path: str):
    """(event name, base props) for a request worth reporting, else None."""
    if method == "POST" and path in TOOL_PATHS:
        return "ClipEdited", {"tool": TOOL_PATHS[path]}
    if method == "POST" and path == SOCIAL_POST_PATH:
        return "SocialPostSent", {}
    if method == "GET" and _DOWNLOAD_ALL_RE.match(path):
        return "ClipsDownloadedAll", {}
    return None


def _body_props(event: str, body: bytes) -> dict:
    """The few request fields worth keeping. Never free text: no prompts,
    captions, titles or descriptions."""
    if not body:
        return {}
    try:
        data = json.loads(body)
    except Exception:
        return {}
    if not isinstance(data, dict):
        return {}
    props = {}
    if isinstance(data.get("clip_index"), int):
        props["clip_index"] = data["clip_index"]
    if event == "SocialPostSent" and isinstance(data.get("platforms"), list):
        names = [p for p in data["platforms"] if isinstance(p, str) and p.isalnum()]
        props["platforms"] = ",".join(sorted(names)[:20])
    if event == "ClipEdited":
        for key in ("mode", "preset", "target_language"):
            value = data.get(key)
            if isinstance(value, str) and _TOKEN_RE.match(value):
                props[key] = value
    return props


def _uses_api_key(headers: dict) -> bool:
    auth = headers.get("authorization", "")
    token = auth[7:].strip() if auth.lower().startswith("bearer ") else ""
    return token.startswith("osk_") or headers.get("x-api-key", "").startswith("osk_")


async def _report(scope, event, props, status):
    from starlette.requests import Request
    from .auth import get_current_user_optional
    try:
        user = await get_current_user_optional(Request(scope))
    except Exception:
        user = None
    if user is None:
        return
    analytics.track(event, user_id=user.id, ok=status < 400, status=status,
                    plan=getattr(user, "plan", None), **props)


class UsageEventsMiddleware:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope.get("type") != "http":
            return await self.app(scope, receive, send)
        hit = _event_for(scope.get("method", ""), scope.get("path", ""))
        if hit is None or not analytics._config():
            return await self.app(scope, receive, send)

        event, props = hit
        body = bytearray()
        status = {"code": 500}

        async def receive_wrapper():
            message = await receive()
            if message.get("type") == "http.request" and len(body) < _MAX_BODY:
                body.extend(message.get("body", b"")[: _MAX_BODY - len(body)])
            return message

        async def send_wrapper(message):
            if message.get("type") == "http.response.start":
                status["code"] = int(message.get("status", 500))
            await send(message)

        try:
            await self.app(scope, receive_wrapper, send_wrapper)
        finally:
            try:
                headers = {k.decode("latin-1").lower(): v.decode("latin-1")
                           for k, v in scope.get("headers", [])}
                extra = dict(props, **_body_props(event, bytes(body)))
                extra["via_api"] = _uses_api_key(headers)
                asyncio.get_running_loop().create_task(
                    _report(scope, event, extra, status["code"]))
            except Exception:
                pass
