"""FastAPI app — JSON API + HTMX web UI (dashboard, composer)."""

from __future__ import annotations

import html
import json
import logging
import os
import secrets
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from opendispatch.adapters import ADAPTERS
from opendispatch.ai import AdaptError, adapt_caption_async
from opendispatch.api.queue import get_queue
from opendispatch.api.schema import ContentUnit, parse_target, validate
from opendispatch.media import PLATFORM_IMAGE_SPECS, TranscodeError, transcode_image_bytes
from opendispatch.profiles import (
    CRED_LABELS,
    PASSWORD_FIELDS,
    PLATFORM_CRED_MAP,
    PLATFORM_EMOJI,
    PRESET_COLORS,
    Profile,
    ProfileStore,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("open-dispatch.api")

try:
    from importlib.metadata import version as _pkg_version

    VERSION = _pkg_version("open-dispatch")
except Exception:  # not installed as a package (dev checkout) — keep in sync w/ pyproject
    VERSION = "0.4.0"

# Captured once at import — used to compute server uptime on /healthz.
_START_TIME = datetime.now(tz=timezone.utc)

REPO_ROOT = Path(__file__).resolve().parent.parent
TEMPLATES_DIR = REPO_ROOT / "web" / "templates"
STATIC_DIR = REPO_ROOT / "web" / "static"

app = FastAPI(
    title="Open-Dispatch",
    version=VERSION,
    description="One API to dispatch content anywhere — open-source.",
    docs_url="/docs",
    redoc_url=None,
)

templates = Jinja2Templates(directory=str(TEMPLATES_DIR))
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

API_TOKEN = os.getenv("OPEN_DISPATCH_API_TOKEN", "").strip()


def _authorized(request: Request) -> bool:
    if not API_TOKEN:
        return True
    auth = request.headers.get("authorization", "")
    supplied = auth.removeprefix("Bearer ").strip()
    return bool(supplied) and secrets.compare_digest(supplied, API_TOKEN)


@app.middleware("http")
async def security_middleware(request: Request, call_next):
    if request.url.path != "/healthz" and not _authorized(request):
        return JSONResponse({"detail": "authentication required"}, status_code=401,
                            headers={"WWW-Authenticate": "Bearer"})
    if request.method in {"POST", "PUT", "PATCH", "DELETE"} and request.headers.get("origin"):
        origin = request.headers["origin"].rstrip("/")
        if origin != str(request.base_url).rstrip("/"):
            return JSONResponse({"detail": "origin not allowed"}, status_code=403)
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "same-origin"
    if request.method in {"POST", "PUT", "PATCH", "DELETE"} and not request.headers.get("authorization"):
        response.headers["Cache-Control"] = "no-store"
    return response


def _platforms_configured() -> set[str]:
    """Which platforms have at least the canonical credential env var set?"""
    checks = {
        "telegram":  ("TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID"),
        "twitter":   ("TWITTER_API_KEY", "TWITTER_ACCESS_TOKEN"),
        "instagram": ("IG_USER_ID", "IG_TOKEN"),
        "bluesky":   ("BLUESKY_HANDLE", "BLUESKY_APP_PASSWORD"),
        "linkedin":  ("LINKEDIN_ACCESS_TOKEN", "LINKEDIN_AUTHOR_URN"),
        "threads":   ("THREADS_USER_ID", "THREADS_ACCESS_TOKEN"),
        "youtube":   ("YOUTUBE_CLIENT_ID", "YOUTUBE_CLIENT_SECRET", "YOUTUBE_REFRESH_TOKEN"),
        "tiktok":    ("TIKTOK_ACCESS_TOKEN",),
        "facebook":  ("FACEBOOK_PAGE_ID", "FACEBOOK_ACCESS_TOKEN"),
        "discord":   ("DISCORD_WEBHOOK_URL",),
    }
    return {
        platform for platform, envs in checks.items()
        if all(os.getenv(e) for e in envs)
    }


def _build_formats(text: str, formats_json: str | None) -> dict[str, Any]:
    """Build per-platform formats dict from either raw JSON or a single text field."""
    if formats_json and formats_json.strip():
        try:
            parsed = json.loads(formats_json)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError as e:
            raise ValueError(f"formats_json: {e}") from e
    # Default: use the text across the common format keys for every selected platform
    return {
        "telegram_message":  {"text": text},
        "twitter_thread":    {"tweets": [text]},
        "bluesky_post":      {"text": text},
        "instagram_post":    {"caption": text},
        "linkedin_post":     {"text": text},
        "threads_post":      {"text": text},
        "facebook_post":     {"text": text},
        "discord_message":   {"content": text},
        # youtube_short omitted on purpose — needs a video_path the form
        # composer can't supply yet. Use AI adapter + Advanced JSON for now.
    }


def _wants_html(request: Request) -> bool:
    """Heuristic — does this request expect HTML (browser) vs JSON (API)?"""
    accept = request.headers.get("accept", "")
    return "text/html" in accept or "hx-request" in {k.lower() for k in request.headers}


def _fmt_uptime(seconds: float) -> str:
    """Human-friendly uptime, e.g. '3d 14h 22m' or '47s'."""
    secs = int(seconds)
    days, rem = divmod(secs, 86400)
    hours, rem = divmod(rem, 3600)
    mins, sec = divmod(rem, 60)
    parts = []
    if days:
        parts.append(f"{days}d")
    if hours:
        parts.append(f"{hours}h")
    if mins:
        parts.append(f"{mins}m")
    if not parts:  # < 1 min
        parts.append(f"{sec}s")
    return " ".join(parts[:3])


def _rel_time(iso: str | None) -> str:
    """ISO timestamp → 'just now' / '14 min ago' / '2h ago' / '3d ago'."""
    if not iso:
        return "never"
    try:
        then = datetime.fromisoformat(iso.replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return iso
    delta = (datetime.now(tz=timezone.utc) - then).total_seconds()
    if delta < 60:
        return "just now"
    if delta < 3600:
        return f"{int(delta // 60)} min ago"
    if delta < 86400:
        return f"{int(delta // 3600)}h ago"
    return f"{int(delta // 86400)}d ago"


def _queue_backend_name() -> str:
    """Which queue backend is active (mirrors get_queue() selection logic)."""
    if os.getenv("DATABASE_URL"):
        return "Postgres"
    if os.getenv("REDIS_URL"):
        return "Redis"
    return "JSONL on disk"


def _worker_status() -> dict[str, Any]:
    """Read the worker heartbeat file and report liveness.

    The worker writes REPO_ROOT/data/.worker_heartbeat (ISO timestamp)
    on every poll loop.  If the file is missing or stale (> 3× poll
    interval), the worker is not running.
    """
    heartbeat_path = REPO_ROOT / "data" / ".worker_heartbeat"
    poll_interval = int(os.getenv("WORKER_POLL_INTERVAL", "5"))
    stale_threshold = poll_interval * 3  # ≈ 15s by default

    if not heartbeat_path.exists():
        return {"running": False, "label": "not running", "last_beat": None}
    try:
        ts = heartbeat_path.read_text(encoding="utf-8").strip()
        then = datetime.fromisoformat(ts)
        age = (datetime.now(tz=timezone.utc) - then).total_seconds()
        if age <= stale_threshold:
            return {"running": True, "label": "running", "last_beat": _rel_time(ts)}
        return {"running": False, "label": "stale", "last_beat": _rel_time(ts)}
    except Exception:  # noqa: BLE001
        return {"running": False, "label": "unknown", "last_beat": None}


def _healthz_context() -> dict[str, Any]:
    """Build the health-dashboard stats from the queue. Cheap — one list_all()."""
    rows = get_queue().list_all()
    counts = _row_counts(rows)

    published = [r for r in rows if r.get("status") == "published"]
    last_dispatch = max(
        published,
        key=lambda r: r.get("updated_at", ""),
        default=None,
    )

    # A row "errored" if it carries a last_error and hasn't since succeeded.
    # The worker re-queues transient failures (status="queued" + last_error),
    # and marks exhausted ones "dead" — surface both so retries are visible.
    errored_rows = [
        r for r in rows
        if r.get("last_error") and r.get("status") not in {"published", "canceled"}
    ]
    errored_rows.sort(key=lambda r: r.get("updated_at", ""), reverse=True)
    recent_errors = [
        {
            "platform": r.get("platform", "?"),
            "error": (r.get("last_error") or "unknown error"),
            "when": _rel_time(r.get("updated_at")),
            "status": r.get("status", "?"),
            "attempts": r.get("attempts", 0),
        }
        for r in errored_rows[:5]
    ]

    top_platforms = Counter(r.get("platform", "?") for r in rows).most_common(5)

    uptime_secs = (datetime.now(tz=timezone.utc) - _START_TIME).total_seconds()

    return {
        "active": "health",
        "version": VERSION,
        "now": datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "uptime": _fmt_uptime(uptime_secs),
        "backend": _queue_backend_name(),
        "worker": _worker_status(),
        "counts": counts,
        "total": len(rows),
        "configured_platforms": sorted(_platforms_configured()),
        "last_dispatch": (
            {
                "platform": last_dispatch.get("platform", "?"),
                "when": _rel_time(last_dispatch.get("updated_at")),
                "post_id": last_dispatch.get("post_id"),
            }
            if last_dispatch
            else None
        ),
        "top_platforms": top_platforms,
        "recent_errors": recent_errors,
        "failed_count": len(errored_rows),
        "platform_emoji": PLATFORM_EMOJI,
    }


# ─── JSON API + health ─────────────────────────────────────────────────────

@app.get("/healthz")
def healthz(request: Request) -> Any:
    """JSON for monitors/curl; HTML status dashboard for browsers."""
    if _wants_html(request):
        return templates.TemplateResponse(request, "health.html", _healthz_context())
    return JSONResponse({"status": "ok", "version": VERSION})


@app.post("/dispatch", status_code=202)
async def dispatch(request: Request) -> JSONResponse:
    try:
        body = await request.json()
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=f"invalid JSON body: {e}") from e
    if not isinstance(body, dict):
        raise HTTPException(status_code=400, detail="JSON body must be an object")
    unit = ContentUnit.from_dict(body)
    if len(json.dumps(unit.to_dict(), ensure_ascii=False)) > 100_000:
        raise HTTPException(status_code=413, detail="dispatch payload is too large")
    errs = validate(unit)
    if errs:
        raise HTTPException(status_code=400, detail={"errors": errs})

    q = get_queue()
    sf = unit.scheduled_for or datetime.now(tz=timezone.utc).isoformat()
    enqueued = []
    for target in unit.targets:
        platform, account = parse_target(target)
        key = f"{platform}:{account or 'default'}"
        row_id = q.enqueue(unit.to_dict(), key, sf)
        enqueued.append({"id": row_id, "target": target, "scheduled_for": sf})

    return JSONResponse(
        status_code=202,
        content={"unit_id": unit.id, "enqueued": enqueued},
    )


@app.get("/queue")
def list_queue(status: str | None = None, limit: int = 100, offset: int = 0) -> dict[str, Any]:
    if limit < 1 or limit > 500 or offset < 0:
        raise HTTPException(status_code=400, detail="limit must be 1..500 and offset must be non-negative")
    rows = get_queue().list_all(status=status)
    return {"count": len(rows), "offset": offset, "limit": limit, "has_more": offset + limit < len(rows),
            "rows": rows[offset:offset + limit]}


@app.get("/campaign/{unit_id}")
def campaign_status(unit_id: str, limit: int = 100, offset: int = 0) -> dict[str, Any]:
    """Return a bounded page of rows belonging to one dispatched content unit."""
    if limit < 1 or limit > 500 or offset < 0:
        raise HTTPException(status_code=400, detail="limit must be 1..500 and offset must be non-negative")
    rows = [
        row for row in get_queue().list_all()
        if (row.get("unit") or {}).get("id") == unit_id
    ]
    if not rows:
        raise HTTPException(status_code=404, detail="campaign not found")
    return {"unit_id": unit_id, "count": len(rows), "offset": offset, "limit": limit,
            "has_more": offset + limit < len(rows), "rows": rows[offset:offset + limit]}


@app.post("/campaign/{unit_id}/cancel")
def cancel_campaign(unit_id: str) -> dict[str, Any]:
    """Cancel queued rows for a campaign; rows already in flight are unchanged."""
    q = get_queue()
    existing = [
        row for row in q.list_all()
        if (row.get("unit") or {}).get("id") == unit_id
    ]
    if not existing:
        raise HTTPException(status_code=404, detail="campaign not found")
    canceled = q.cancel_campaign(unit_id)
    return {
        "unit_id": unit_id,
        "canceled": len(canceled),
        "rows": canceled,
    }


@app.get("/queue/{row_id}/json")
def get_row_json(row_id: str) -> dict[str, Any]:
    row = get_queue().get(row_id)
    if not row:
        raise HTTPException(status_code=404, detail="not found")
    return row


@app.post("/queue/{row_id}/retry")
async def retry_row(request: Request, row_id: str) -> Any:
    q = get_queue()
    row = q.get(row_id)
    if not row:
        raise HTTPException(status_code=404, detail="not found")
    if row.get("status") == "canceled":
        raise HTTPException(status_code=409, detail="canceled rows cannot be retried")
    if not q.retry(row_id):
        raise HTTPException(status_code=409, detail="row cannot be retried in its current state")
    if _wants_html(request):
        # HTMX caller: re-render the queue fragment
        return await _render_queue_fragment(request, status=None)
    return {"id": row_id, "status": "queued"}


@app.post("/_retry-all")
async def retry_all(request: Request) -> Any:
    """Re-queue every failed/dead row. Used by the health dashboard."""
    q = get_queue()
    requeued = 0
    for row in q.list_all():
        if row.get("last_error") and row.get("status") not in {"published", "canceled"}:
            if q.retry(row["id"]):
                requeued += 1
    if _wants_html(request):
        # Browser form submit: bounce back to the refreshed health dashboard.
        return RedirectResponse(url="/healthz", status_code=303)
    return {"requeued": requeued}


@app.delete("/queue/{row_id}")
async def delete_row(request: Request, row_id: str) -> Any:
    """Delete a single queue row permanently."""
    q = get_queue()
    row = q.get(row_id)
    if not row:
        raise HTTPException(status_code=404, detail="not found")
    q.delete(row_id)
    if _wants_html(request):
        return RedirectResponse(url="/", status_code=303)
    return {"deleted": row_id}


@app.post("/queue/{row_id}/delete")
async def delete_row_post(request: Request, row_id: str) -> Any:
    """HTML-form-friendly alias for DELETE /queue/{id}.

    When called via HTMX from the queue table, returns the refreshed
    queue fragment instead of a redirect.
    """
    q = get_queue()
    row = q.get(row_id)
    if not row:
        raise HTTPException(status_code=404, detail="not found")
    q.delete(row_id)
    # HTMX inline call from dashboard table
    if "hx-request" in {k.lower() for k in request.headers}:
        return await _render_queue_fragment(request, status=None)
    if _wants_html(request):
        return RedirectResponse(url="/", status_code=303)
    return {"deleted": row_id}


@app.post("/_purge")
async def purge_queue(request: Request, status: str = "published") -> Any:
    """Delete all rows with a given status. Default: clear published rows."""
    allowed = {"published", "dead"}
    if status not in allowed:
        raise HTTPException(status_code=400, detail=f"can only purge: {', '.join(sorted(allowed))}")
    q = get_queue()
    deleted = 0
    for row in q.list_all(status=status):
        q.delete(row["id"])
        deleted += 1
    if _wants_html(request):
        return RedirectResponse(url="/", status_code=303)
    return {"purged": deleted, "status": status}


# ─── Web UI ────────────────────────────────────────────────────────────────

def _row_counts(rows: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = Counter(r.get("status", "") for r in rows)
    counts["all"] = len(rows)
    return counts


async def _render_queue_fragment(request: Request, status: str | None) -> HTMLResponse:
    rows = get_queue().list_all(status=status)
    rows.sort(key=lambda r: r.get("created_at", ""), reverse=True)
    return templates.TemplateResponse(
        request, "_queue_table.html",
        {"rows": rows, "filter_status": status},
    )


@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request, status: str | None = None) -> HTMLResponse:
    all_rows = get_queue().list_all()
    rows = [r for r in all_rows if not status or r.get("status") == status]
    rows.sort(key=lambda r: r.get("created_at", ""), reverse=True)
    return templates.TemplateResponse(
        request, "dashboard.html",
        {
            "rows": rows,
            "counts": _row_counts(all_rows),
            "filter_status": status,
            "active": "dashboard",
            "version": VERSION,
            "now": datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        },
    )


@app.get("/_queue-fragment", response_class=HTMLResponse)
async def queue_fragment(request: Request, status: str | None = None) -> HTMLResponse:
    return await _render_queue_fragment(request, status)


@app.get("/compose", response_class=HTMLResponse)
async def compose_page(request: Request) -> HTMLResponse:
    platforms = sorted(ADAPTERS.keys())
    configured = _platforms_configured()
    profiles = ProfileStore().list()
    # Build per-profile configured-platforms map for client-side JS
    import json as _json
    profiles_js = _json.dumps([
        {"id": p.id, "name": p.name, "emoji": p.emoji,
         "configured": p.configured_platforms()}
        for p in profiles
    ]).replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
    return templates.TemplateResponse(
        request, "compose.html",
        {
            "platforms": platforms,
            "configured": configured,
            "profiles": profiles,
            "profiles_js": profiles_js,
            "active": "compose",
            "version": VERSION,
            "now": datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        },
    )


@app.post("/_compose", response_class=HTMLResponse)
async def compose_submit(
    request: Request,
    text: str = Form(""),
    platforms: list[str] = Form(default=[]),
    profile_id: str = Form(""),
    scheduled_for: str = Form(""),
    webhook_url: str = Form(""),
    formats_json: str = Form(""),
) -> HTMLResponse:
    errors: list[str] = []
    targets: list[str] = [p for p in platforms if p]

    if not targets:
        errors.append("pick at least one platform")
    if not text and not formats_json:
        errors.append("text or formats_json is required")

    try:
        formats = _build_formats(text, formats_json or None)
    except ValueError as e:
        errors.append(str(e))
        formats = {}

    if errors:
        return templates.TemplateResponse(
            request, "_compose_result.html",
            {"ok": False, "errors": errors},
        )

    unit = ContentUnit.from_dict({
        "category": "web",
        "targets": targets,
        "scheduled_for": scheduled_for.strip() or None,
        "formats": formats,
        "webhook_url": webhook_url.strip() or None,
        "profile_id": profile_id.strip() or None,
    })
    errs = validate(unit)
    if errs:
        return templates.TemplateResponse(
            request, "_compose_result.html",
            {"ok": False, "errors": errs},
        )

    # Resolve profile label for the result panel
    profile = ProfileStore().get(profile_id.strip()) if profile_id.strip() else None
    profile_label = f"{profile.emoji} {profile.name}" if profile else ".env defaults"

    q = get_queue()
    sf = unit.scheduled_for or datetime.now(tz=timezone.utc).isoformat()
    enqueued = []
    for target in unit.targets:
        platform, account = parse_target(target)
        key = f"{platform}:{account or 'default'}"
        row_id = q.enqueue(unit.to_dict(), key, sf)
        enqueued.append({"id": row_id, "target": target, "scheduled_for": sf})

    return templates.TemplateResponse(
        request, "_compose_result.html",
        {"ok": True, "unit_id": unit.id, "enqueued": enqueued,
         "profile_label": profile_label},
    )


@app.get("/queue/{row_id}")
async def row_detail(request: Request, row_id: str) -> Any:
    """Content-negotiated: HTML for browsers, JSON for API consumers.

    Browsers send `Accept: text/html,...`, curl/SDKs send `*/*` or
    `application/json`. We return HTML only when the client clearly
    prefers it so we don't break existing API consumers.
    """
    row = get_queue().get(row_id)
    wants_html = _wants_html(request)
    if not row:
        if wants_html:
            return HTMLResponse(
                "<h1>404 — row not found</h1>"
                f"<p>No queue row with id <code>{row_id}</code>.</p>"
                "<p><a href='/'>Back to dashboard</a></p>",
                status_code=404,
            )
        raise HTTPException(status_code=404, detail="not found")
    if wants_html:
        return templates.TemplateResponse(
            request, "row_detail.html",
            {
                "row": row,
                "unit_json": json.dumps(row.get("unit", {}), indent=2, ensure_ascii=False),
                "active": "dashboard",
                "version": VERSION,
                "now": datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
            },
        )
    # Default to JSON for API consumers (back-compat with the documented contract)
    return row


# Redirect bare /queue (GET) to dashboard when called from a browser
@app.get("/queue/", response_class=HTMLResponse, include_in_schema=False)
async def queue_redirect() -> RedirectResponse:
    return RedirectResponse(url="/", status_code=307)


# ─── AI caption adaptation ────────────────────────────────────────────────

@app.post("/ai/adapt")
async def ai_adapt(request: Request) -> dict[str, Any]:
    """Adapt a source caption into per-platform formats.

    Request body:
      { "text": "...", "platforms": ["twitter","bluesky",...], "provider": "openrouter"|"ollama"|"heuristic" }

    Response:
      { "ok": true, "formats": {...}, "provider": "openrouter" }

    Provider defaults: ollama if OLLAMA_HOST set, else openrouter if
    OPENROUTER_API_KEY set, else heuristic (no LLM).
    """
    try:
        body = await request.json()
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=f"invalid JSON body: {e}") from e
    if not isinstance(body, dict):
        raise HTTPException(status_code=400, detail="JSON body must be an object")
    text = (body.get("text") or "").strip()
    platforms = body.get("platforms") or []
    provider = body.get("provider")
    if not text:
        raise HTTPException(status_code=400, detail="text is required")
    if not isinstance(platforms, list) or not platforms:
        raise HTTPException(status_code=400, detail="platforms must be a non-empty list")
    if len(text) > 10_000 or len(platforms) > 20:
        raise HTTPException(status_code=413, detail="caption or platform list is too large")
    try:
        formats = await adapt_caption_async(text, platforms, provider=provider)
    except AdaptError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    return {"ok": True, "formats": formats}


@app.post("/_compose-adapt", response_class=HTMLResponse)
async def compose_adapt_htmx(
    request: Request,
    text: str = Form(""),
    platforms: list[str] = Form(default=[]),
) -> HTMLResponse:
    """HTMX endpoint: takes the in-progress composer text + selected platforms,
    returns a JSON formats blob to drop into the Advanced textarea.
    """
    if not text.strip() or not platforms:
        return HTMLResponse(
            '<div class="alert alert-error">'
            "<strong>Need text and at least one platform.</strong>"
            "</div>",
        )
    try:
        formats = await adapt_caption_async(text, platforms)
    except AdaptError as e:
        return HTMLResponse(
            f'<div class="alert alert-error"><strong>Adapt failed:</strong> {html.escape(str(e))}</div>',
        )
    pretty = html.escape(json.dumps(formats, indent=2, ensure_ascii=False))
    return HTMLResponse(
        '<div class="alert alert-success">'
        "<strong>Adapted.</strong> Paste this into the Advanced JSON box "
        "(or just submit — the adapter ran on the server side):"
        f'<pre class="mono small" style="margin-top:8px;max-height:300px;overflow:auto">{pretty}</pre>'
        "</div>",
    )


# ─── Profiles ─────────────────────────────────────────────────────────────────

def _profile_ctx(version: str, now: str, active: str = "profiles") -> dict:
    return {"active": active, "version": version, "now": now,
            "platform_cred_map": PLATFORM_CRED_MAP, "cred_labels": CRED_LABELS,
            "password_fields": PASSWORD_FIELDS, "platform_emoji": PLATFORM_EMOJI,
            "preset_colors": PRESET_COLORS}


@app.get("/profiles", response_class=HTMLResponse)
async def profiles_list(request: Request) -> HTMLResponse:
    profiles = ProfileStore().list()
    ctx = _profile_ctx(VERSION, datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC"))
    return templates.TemplateResponse(request, "profiles.html",
                                      {**ctx, "profiles": profiles})


@app.get("/profiles/new", response_class=HTMLResponse)
async def profile_new(request: Request) -> HTMLResponse:
    ctx = _profile_ctx(VERSION, datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC"))
    return templates.TemplateResponse(request, "profile_form.html",
                                      {**ctx, "profile": None, "title": "New Profile"})


@app.get("/profiles/{profile_id}/edit", response_class=HTMLResponse)
async def profile_edit(request: Request, profile_id: str) -> HTMLResponse:
    profile = ProfileStore().get(profile_id)
    if not profile:
        raise HTTPException(status_code=404, detail="profile not found")
    ctx = _profile_ctx(VERSION, datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC"))
    return templates.TemplateResponse(request, "profile_form.html",
                                      {**ctx, "profile": profile,
                                       "title": f"Edit — {profile.name}"})


@app.post("/profiles", response_class=HTMLResponse)
async def profile_create(request: Request) -> HTMLResponse:
    form = await request.form()
    profile = _profile_from_form(form)
    ProfileStore().save(profile)
    return RedirectResponse(url="/profiles", status_code=303)


@app.post("/profiles/{profile_id}", response_class=HTMLResponse)
async def profile_update(request: Request, profile_id: str) -> HTMLResponse:
    store = ProfileStore()
    if not store.get(profile_id):
        raise HTTPException(status_code=404, detail="profile not found")
    form = await request.form()
    profile = _profile_from_form(form, profile_id=profile_id)
    store.save(profile)
    return RedirectResponse(url="/profiles", status_code=303)


@app.post("/profiles/{profile_id}/delete", response_class=HTMLResponse)
async def profile_delete(request: Request, profile_id: str) -> HTMLResponse:
    ProfileStore().delete(profile_id)
    return RedirectResponse(url="/profiles", status_code=303)


@app.get("/api/profiles")
def api_profiles_list() -> list[dict]:
    """JSON list — used by the compose page JS."""
    return [
        {"id": p.id, "name": p.name, "emoji": p.emoji,
         "color": p.color, "configured": p.configured_platforms()}
        for p in ProfileStore().list()
    ]


def _profile_from_form(form: Any, profile_id: str | None = None) -> Profile:
    """Build a Profile from a submitted HTML form."""
    pid = profile_id or str(form.get("id", "")).strip() or None
    existing = ProfileStore().get(profile_id) if profile_id else None
    platforms: dict[str, dict[str, str]] = {
        p: dict(creds) for p, creds in (existing.platforms.items() if existing else [])
    }
    for platform, fields in PLATFORM_CRED_MAP.items():
        creds = platforms.setdefault(platform, {})
        for field_name in fields:
            # The edit form renders credentials with value="" (secrets are never
            # sent to the browser), so an empty submit can only mean "unchanged".
            # A checkbox submits "<field>__clear" to explicitly remove a stored
            # credential — otherwise a rotated, leaked token could never be
            # deleted through the UI.
            if str(form.get(f"{platform}__{field_name}__clear", "")).strip():
                creds.pop(field_name, None)
                continue
            submitted = str(form.get(f"{platform}__{field_name}", "")).strip()
            if submitted:
                creds[field_name] = submitted
        if not any(creds.values()):
            platforms.pop(platform, None)
    return Profile(
        id=pid if pid else Profile().id,
        name=str(form.get("name", "")).strip() or "Unnamed",
        emoji=str(form.get("emoji", "🎭")).strip() or "🎭",
        color=str(form.get("color", "#4ade80")).strip() or "#4ade80",
        platforms=platforms,
    )


# ─── Media transcoding ────────────────────────────────────────────────────

@app.get("/media/specs")
def media_specs() -> dict[str, Any]:
    """Return the per-platform image-spec table so clients can introspect
    what we'll do to their images before they upload."""
    return {
        name: {
            "max_width": spec.max_width,
            "max_height": spec.max_height,
            "aspect_min": spec.aspect_min,
            "aspect_max": spec.aspect_max,
            "format": spec.format,
            "quality": spec.quality,
            "description": spec.description,
        }
        for name, spec in PLATFORM_IMAGE_SPECS.items()
    }


@app.post("/media/transcode")
async def media_transcode(request: Request) -> Any:
    """Transcode raw image bytes for one platform spec.

    Two ways to call:
    1. Raw body: `Content-Type: image/<anything>`, body = image bytes,
       query param `?platform=instagram` — returns the transcoded JPEG bytes
       directly with Content-Type: image/jpeg.
    2. multipart/form-data with fields `image` (file) and `platform` — same response.
    """
    from fastapi.responses import Response
    platform = request.query_params.get("platform", "")
    blob: bytes | None = None

    content_type = (request.headers.get("content-type") or "").lower()
    if content_type.startswith("multipart/form-data"):
        form = await request.form()
        platform = platform or str(form.get("platform", ""))
        upload = form.get("image")
        if upload is None:
            raise HTTPException(status_code=400, detail="multipart: missing 'image' field")
        blob = await upload.read()  # type: ignore[union-attr]
    else:
        blob = await request.body()

    if not platform:
        raise HTTPException(status_code=400, detail="platform query param required")
    if not blob:
        raise HTTPException(status_code=400, detail="empty image body")
    if len(blob) > 10 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="image exceeds 10 MiB limit")

    try:
        out = transcode_image_bytes(blob, platform)
    except TranscodeError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    spec = PLATFORM_IMAGE_SPECS[platform]
    mime = "image/jpeg" if spec.format == "JPEG" else "image/png"
    return Response(content=out, media_type=mime)


# ─── Bulk CSV import ──────────────────────────────────────────────────────

# Canonical text field per format key — used to auto-wrap plain-text CSV
# cells into the payload shape each adapter reads. Kept explicit so adding
# an adapter means adding one row here, not another elif branch.
_TEXT_FIELD_BY_FORMAT: dict[str, str] = {
    "twitter_thread":   "tweets",
    "bluesky_post":     "text",
    "telegram_message": "text",
    "linkedin_post":    "text",
    "threads_post":     "text",
    "facebook_post":    "text",
    "discord_message":  "content",
    "instagram_post":   "caption",
    "tiktok_post":      "caption",
    "youtube_short":    "title",
}


@app.post("/dispatch/bulk", status_code=202)
async def dispatch_bulk(request: Request) -> JSONResponse:
    """Bulk-enqueue posts from a CSV upload.

    Accepts multipart/form-data with a single field named `file` (text/csv).

    CSV format — one row per post, columns (order matters):
      targets, format_key, text_or_json, scheduled_for
      - targets:       comma-separated platform targets, e.g. "twitter,bluesky"
      - format_key:    e.g. "twitter_thread" or "bluesky_post"
      - text_or_json:  either plain text (auto-wrapped) or raw JSON object
      - scheduled_for: ISO-8601 datetime or empty (post immediately)

    Example row:
      "twitter,bluesky","twitter_thread","Hello world!","2026-07-01T09:00:00+00:00"

    Returns list of enqueued unit IDs and row IDs.
    """
    import csv
    import io

    content_type = request.headers.get("content-type", "")
    max_bytes = 1_000_000
    content_length = request.headers.get("content-length")
    if content_length:
        try:
            declared_bytes = int(content_length)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail="invalid Content-Length header") from exc
        if declared_bytes > max_bytes:
            raise HTTPException(status_code=413, detail="CSV exceeds 1 MiB limit")
    if "multipart" in content_type:
        form = await request.form()
        upload = form.get("file")
        if upload is None:
            raise HTTPException(status_code=400, detail="multipart: missing 'file' field")
        raw = (await upload.read()).decode("utf-8")  # type: ignore[union-attr]
    else:
        raw = (await request.body()).decode("utf-8")

    if len(raw.encode("utf-8")) > max_bytes:
        raise HTTPException(status_code=413, detail="CSV exceeds 1 MiB limit")
    if not raw.strip():
        raise HTTPException(status_code=400, detail="CSV body is empty")

    reader = csv.reader(io.StringIO(raw))
    q = get_queue()
    results: list[dict] = []
    errors: list[dict] = []

    for i, row in enumerate(reader, start=1):
        if i > 1000:
            raise HTTPException(status_code=413, detail="CSV exceeds 1000-row limit")
        # skip blank lines and header rows starting with '#' or 'targets'
        if not row or not row[0].strip() or row[0].strip().lower() in {"targets", "#"}:
            continue
        if len(row) < 3:
            errors.append({"row": i, "error": "need at least 3 columns: targets, format_key, text_or_json"})
            continue

        raw_targets = [t.strip() for t in row[0].split(",") if t.strip()]
        format_key = row[1].strip()
        text_or_json = row[2].strip()
        scheduled_for = row[3].strip() if len(row) > 3 else ""

        # Parse the payload — JSON object or plain text
        try:
            payload = json.loads(text_or_json)
            if not isinstance(payload, dict):
                raise ValueError("not a dict")
        except (json.JSONDecodeError, ValueError):
            # Treat as plain text — wrap using the canonical text field for
            # each format key (matches what the adapters actually read).
            text_field = _TEXT_FIELD_BY_FORMAT.get(format_key, "text")
            payload = {text_field: [text_or_json] if text_field == "tweets" else text_or_json}

        unit = ContentUnit(
            targets=raw_targets,
            formats={format_key: payload},
            scheduled_for=scheduled_for or None,
        )
        errs = validate(unit)
        if errs:
            errors.append({"row": i, "errors": errs})
            continue

        sf = unit.scheduled_for or datetime.now(tz=timezone.utc).isoformat()
        enqueued = []
        for target in unit.targets:
            platform, account = parse_target(target)
            key = f"{platform}:{account or 'default'}"
            row_id = q.enqueue(unit.to_dict(), key, sf)
            enqueued.append({"id": row_id, "target": target, "scheduled_for": sf})

        results.append({"unit_id": unit.id, "enqueued": enqueued})

    return JSONResponse(
        status_code=202,
        content={
            "imported": len(results),
            "errors": len(errors),
            "results": results,
            "error_details": errors,
        },
    )
