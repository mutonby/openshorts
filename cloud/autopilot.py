"""Autopilot: clip every new video on the user's YouTube channel, by itself.

The user connects YouTube once (the same Upload-Post profile the dashboard
already uses for posting) and switches Autopilot on. From then on:

1. A poller lists the channel's latest uploads through Upload-Post
   (``GET /api/uploadposts/media?platform=youtube``). That endpoint reads the
   channel's uploads playlist with the user's own OAuth token, so it sees
   videos uploaded straight to YouTube, not just ones posted through us, and
   no scraping is involved.
2. A video published after Autopilot was switched on becomes a job on the
   normal pipeline: an in-process ``POST /api/process`` with the video URL,
   authenticated as the user. Metering, the minute reservation, the probe, the
   per-plan job limit and the download proxies all apply exactly as when the
   user pastes the link themselves, so nothing here has to be kept in sync.
3. When the job finishes, ``on_job_finished`` (called from app.py's job
   wrapper) records the outcome, emails the user, and, if they asked for it,
   schedules the best clips on their connected accounts, one a day.

Why this exists: most subscribers used the product on one day and never came
back (6 of 33 retained subscribers had used it on 3+ distinct days, sep-2026).
Value that arrives every week without the user doing anything is the lever.

Guard rails, each one a cost that would otherwise scale with the channel:
- only videos published after ``enabled_at`` and at most ``MAX_VIDEO_AGE`` old,
  so switching it on never processes the back catalogue;
- ``DAILY_AUTO_LIMIT`` automatic jobs per user per 24 h (downloads go through
  the shared proxy pool);
- ``max_minutes`` per video (the first N minutes of a long stream), so one
  two-hour live does not empty the month;
- YouTube Shorts are skipped: they already are short-form.

Twitch (``AUTOPILOT_TWITCH=1``, once Upload-Post lists Twitch broadcasts):
the same loop also reads the past broadcasts of a Twitch account connected to
the profile (``platform=twitch``). A broadcast still live is left for a later
tick, and instead of the first ``max_minutes`` of a stream (usually the
"starting soon" screen) the job clips the stretch its viewers clipped most
(``twitch_signals.best_stretch``), when they clipped anything. Twitch video
ids are stored as ``twitch:<id>`` so they can never collide with YouTube ids.
"""
import asyncio
import os
import uuid
from datetime import datetime, timedelta, timezone
from typing import Iterable, List, Optional

import httpx
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel
from sqlalchemy import and_, delete, func, select, update
from sqlalchemy.exc import IntegrityError

from . import database
from .config import settings
from .models import AutopilotRun, AutopilotSettings, UploadPostProfile, User

router = APIRouter()

API_BASE = "https://api.upload-post.com/api"

PAID_PLANS = ("starter", "creator", "pro")
PUBLISH_PLATFORMS = ("tiktok", "instagram", "youtube")
TWITCH_PREFIX = "twitch:"


def twitch_enabled() -> bool:
    """Twitch as a source is off until Upload-Post serves ``platform=twitch``."""
    return os.environ.get("AUTOPILOT_TWITCH", "").strip() == "1"


def channel_sources(accounts: dict) -> List[str]:
    """The source platforms to read for a profile, in a stable order.

    ``accounts`` is ``fetch_social_accounts``; empty means it could not be read,
    and then only YouTube is tried, exactly as before Twitch existed.
    """
    if not accounts:
        return ["youtube"]
    out = [p for p in ("youtube",) if p in accounts]
    if twitch_enabled() and "twitch" in accounts:
        out.append("twitch")
    return out
MAX_MINUTES_CHOICES = (10, 20, 30, 45, 60, 90)
DEFAULT_MAX_MINUTES = 30
DEFAULT_CLIPS_TO_PUBLISH = 3
MAX_CLIPS_TO_PUBLISH = 5
DEFAULT_PUBLISH_HOUR = 17

TICK_SECONDS = 300                  # how often the loop wakes up
CHECK_EVERY = timedelta(minutes=55)  # per-user channel check interval
MAX_VIDEO_AGE = timedelta(days=3)    # older uploads are never picked automatically
DAILY_AUTO_LIMIT = 1                 # automatic jobs per user per rolling 24 h
MANUAL_DAILY_LIMIT = 5               # "clip this video" clicks per user per 24 h
STALE_RUN_AFTER = timedelta(hours=6)  # a job that never reported back
CHANNEL_LIST_LIMIT = 12

# Run statuses. "queued" only exists between the INSERT that claims a video and
# the /api/process answer; "processing" means a job id exists.
QUEUED, PROCESSING, COMPLETED, FAILED, SKIPPED = (
    "queued", "processing", "completed", "failed", "skipped")

_app = None          # the FastAPI app, for in-process calls (set by start())
_is_active = None    # callable: False while this instance drains for a deploy
_task = None         # strong ref: the loop only keeps weak references to tasks
_bg_tasks = set()


# --------------------------------------------------------------------------- #
# Pure helpers (unit-tested without a database)
# --------------------------------------------------------------------------- #
def parse_ts(value) -> Optional[datetime]:
    """An Upload-Post/YouTube timestamp ('2026-09-22T10:01:35Z') as aware UTC."""
    if not value:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    try:
        dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def pick_new_videos(media: Iterable[dict], baseline: Optional[datetime],
                    seen_ids: set, now: datetime,
                    max_age: timedelta = MAX_VIDEO_AGE) -> List[dict]:
    """Channel uploads Autopilot should clip now, oldest first.

    A video qualifies when it was published after the baseline (the moment
    Autopilot was switched on), is not older than ``max_age``, and has not been
    picked before. Items without an id or a parseable timestamp are ignored:
    better to miss one than to clip the back catalogue.
    """
    if baseline is None:
        return []
    out = []
    for item in media or []:
        vid = (item or {}).get("id")
        published = parse_ts((item or {}).get("timestamp"))
        if not vid or published is None or vid in seen_ids:
            continue
        if item.get("is_live"):
            continue  # a broadcast still running: its VOD is not finished yet
        if published <= baseline or now - published > max_age:
            continue
        out.append(item)
    out.sort(key=lambda m: parse_ts(m.get("timestamp")))
    return out


def rank_clips(clips: List[dict], n: int) -> List[int]:
    """Indices of the ``n`` clips worth publishing, best first.

    ``predicted_score`` is the second-pass virality score the selector already
    computes; clips without one keep their pipeline order after the scored ones.
    """
    if n <= 0 or not clips:
        return []

    def score(i):
        try:
            return float(clips[i].get("predicted_score"))
        except (TypeError, ValueError):
            return float("-inf")

    order = sorted(range(len(clips)), key=lambda i: (-score(i), i))
    return order[:n]


def schedule_slots(now: datetime, n: int, hour: int = DEFAULT_PUBLISH_HOUR,
                   tz_name: Optional[str] = None) -> List[str]:
    """``n`` local datetimes (ISO, no offset) at ``hour``, one per day.

    Starts today if the slot is still at least an hour away, else tomorrow.
    Upload-Post takes the local wall time plus an IANA ``timezone`` parameter.
    """
    tz = _zone(tz_name)
    local_now = now.astimezone(tz)
    first = local_now.replace(hour=hour % 24, minute=0, second=0, microsecond=0)
    if first - local_now < timedelta(hours=1):
        first += timedelta(days=1)
    return [(first + timedelta(days=k)).strftime("%Y-%m-%dT%H:%M:%S") for k in range(n)]


def _zone(tz_name: Optional[str]):
    try:
        from zoneinfo import ZoneInfo
        return ZoneInfo(tz_name) if tz_name else timezone.utc
    except Exception:
        return timezone.utc


def outcome_from_process(status_code: int, body) -> dict:
    """Map an /api/process answer to what the run becomes.

    Returns ``{"status", "reason", "job_id", "retry"}``. ``retry`` means the
    claim should be dropped and the video tried again on a later tick (the
    user is at their simultaneous-job limit, or the server hiccuped).
    """
    body = body if isinstance(body, dict) else {}
    detail = body.get("detail")
    if status_code == 200:
        if body.get("needs_confirmation"):
            return {"status": SKIPPED, "reason": "low_quality", "job_id": None, "retry": False}
        if body.get("job_id"):
            return {"status": PROCESSING, "reason": None, "job_id": body["job_id"], "retry": False}
        return {"status": FAILED, "reason": "no_job_id", "job_id": None, "retry": False}
    if status_code == 402:
        return {"status": SKIPPED, "reason": "out_of_minutes", "job_id": None, "retry": False}
    if status_code == 429:
        return {"status": QUEUED, "reason": "busy", "job_id": None, "retry": True}
    if status_code == 400:
        text = str(detail or "").lower()
        if "short-form" in text or "only" in text and "long" in text:
            return {"status": SKIPPED, "reason": "too_short", "job_id": None, "retry": False}
        return {"status": SKIPPED, "reason": "unavailable", "job_id": None, "retry": False}
    if status_code == 403:
        return {"status": SKIPPED, "reason": "url_ingest_disabled", "job_id": None, "retry": False}
    if status_code >= 500:
        return {"status": QUEUED, "reason": "server_error", "job_id": None, "retry": True}
    return {"status": FAILED, "reason": f"http_{status_code}", "job_id": None, "retry": False}


def clean_platforms(values) -> List[str]:
    """Keep the platforms Autopilot can publish to, in a stable order."""
    wanted = {str(v).lower() for v in (values or [])}
    return [p for p in PUBLISH_PLATFORMS if p in wanted]


def eligible(user) -> bool:
    """Autopilot is part of the paid plans (it spends minutes on its own)."""
    return bool(user is not None and getattr(user, "entitled", False)
                and getattr(user, "plan", None) in PAID_PLANS)


# --------------------------------------------------------------------------- #
# Upload-Post
# --------------------------------------------------------------------------- #
class ChannelError(Exception):
    """The channel could not be listed; ``code`` is shown to the user."""

    def __init__(self, code: str):
        super().__init__(code)
        self.code = code


def _headers():
    return {"Authorization": f"Apikey {settings.managed_upload_post_key}"}


def normalise_media(items, platform: str) -> List[dict]:
    """Tag each listed video with its ``source``; Twitch ids get their prefix.

    Upload-Post returns a Twitch broadcast's title as ``title`` and a YouTube
    upload's as ``caption``; both end up in ``caption``.
    """
    out = []
    for m in items or []:
        if not isinstance(m, dict) or not m.get("id"):
            continue
        m = dict(m)
        m["source"] = platform
        if platform == "twitch":
            raw = str(m["id"])
            m["source_id"] = raw[len(TWITCH_PREFIX):] if raw.startswith(TWITCH_PREFIX) else raw
            m["id"] = TWITCH_PREFIX + m["source_id"]
            m["permalink"] = m.get("permalink") or f"https://www.twitch.tv/videos/{m['source_id']}"
        if not m.get("caption") and m.get("title"):
            m["caption"] = m["title"]
        out.append(m)
    return out


async def fetch_channel_videos(profile: str, limit: int = CHANNEL_LIST_LIMIT,
                               platform: str = "youtube") -> List[dict]:
    """The latest uploads (YouTube) or past broadcasts (Twitch) of the account
    connected to ``profile``, each tagged with its ``source``."""
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.get(
            f"{API_BASE}/uploadposts/media", headers=_headers(),
            params={"platform": platform, "user": profile, "limit": limit})
    try:
        data = resp.json()
    except ValueError:
        data = {}
    if resp.status_code == 200 and data.get("success") is not False:
        return normalise_media(data.get("media"), platform)
    message = str(data.get("error") or data.get("message") or "").lower()
    if f"not have a {platform}" in message or "not associated" in message:
        raise ChannelError(f"{platform}_not_connected")
    if "reauth" in message or "expired" in message or "revoked" in message \
            or resp.status_code in (401, 403):
        raise ChannelError(f"{platform}_reauth_required")
    raise ChannelError("channel_unavailable")


async def fetch_all_videos(profile: str, limit: int = CHANNEL_LIST_LIMIT,
                           accounts: Optional[dict] = None) -> List[dict]:
    """Videos from every connected source. Raises the first source's error only
    when no source could be read (one broken connection must not stop the
    other)."""
    if accounts is None:
        try:
            accounts = await fetch_social_accounts(profile)
        except Exception:
            accounts = {}
    sources = channel_sources(accounts)
    if not sources:
        raise ChannelError("youtube_not_connected")
    media, errors = [], []
    for platform in sources:
        try:
            media.extend(await fetch_channel_videos(profile, limit, platform))
        except ChannelError as e:
            errors.append(e)
    if errors and len(errors) == len(sources):
        raise errors[0]
    media.sort(key=lambda m: parse_ts(m.get("timestamp")) or datetime.min.replace(
        tzinfo=timezone.utc), reverse=True)
    return media


async def fetch_social_accounts(profile: str) -> dict:
    """``{platform: {handle, display_name, reauth_required}}`` for a profile."""
    async with httpx.AsyncClient(timeout=20) as client:
        resp = await client.get(f"{API_BASE}/uploadposts/users/{profile}",
                                headers=_headers())
    if resp.status_code != 200:
        return {}
    try:
        accounts = ((resp.json() or {}).get("profile") or {}).get("social_accounts") or {}
    except ValueError:
        return {}
    out = {}
    for platform, info in accounts.items():
        if isinstance(info, dict) and info:
            out[platform] = {
                "handle": info.get("handle"),
                "display_name": info.get("display_name"),
                "reauth_required": bool(info.get("reauth_required")),
            }
    return out


async def is_youtube_private(video_id: str) -> bool:
    """True when the video is private on YouTube.

    oEmbed answers 401/403 for a private video and 200 for a public or
    unlisted one. It has to run BEFORE is_youtube_short: /shorts/<id> of a
    private video also answers 200, so a private upload read as "already a
    short". A private video cannot be clipped at all: yt-dlp gets "Private
    video", and the YouTube Data API behind Upload-Post lists the owner's
    uploads but never serves the media file.
    """
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get("https://www.youtube.com/oembed", params={
                "url": f"https://www.youtube.com/watch?v={video_id}", "format": "json"})
        return resp.status_code in (401, 403)
    except Exception:
        return False


async def is_youtube_short(video_id: str) -> bool:
    """True when YouTube serves this id as a Short.

    ``/shorts/<id>`` answers 200 for a Short and redirects to ``/watch`` for a
    regular video. From the prod servers (EU) YouTube first redirects every
    request to consent.youtube.com, which read as "not a Short" and let a Short
    through (22-sep-2026); the SOCS/CONSENT cookies skip that page. Any other
    answer (a bot check) still reads as "not a Short" and the normal
    pipeline's own too-short gate applies.
    """
    try:
        async with httpx.AsyncClient(timeout=10, follow_redirects=False,
                                     cookies={"SOCS": "CAI", "CONSENT": "YES+"}) as client:
            resp = await client.head(f"https://www.youtube.com/shorts/{video_id}")
        return resp.status_code == 200
    except Exception:
        return False


async def _upload_clip(profile: str, file_path: str, clip: dict, platforms: List[str],
                       when: str, tz_name: str) -> bool:
    """Schedule one clip on Upload-Post. Mirrors app.post_to_socials' payload."""
    loop = asyncio.get_event_loop()
    content = await loop.run_in_executor(None, _read_bytes, file_path)
    title = clip.get("video_title_for_youtube_short") or clip.get("title") or "New short"
    description = (clip.get("video_description_for_instagram")
                   or clip.get("video_description_for_tiktok") or title)
    data = {
        "user": profile,
        "title": title,
        "platform[]": platforms,
        "async_upload": "true",
        "scheduled_date": when,
        "timezone": tz_name or "UTC",
    }
    if "tiktok" in platforms:
        data["tiktok_title"] = description
        data["post_mode"] = os.environ.get("TIKTOK_POST_MODE", "MEDIA_UPLOAD").strip()
    if "instagram" in platforms:
        data["instagram_title"] = description
        data["media_type"] = "REELS"
    if "youtube" in platforms:
        data["youtube_title"] = title
        data["youtube_description"] = description
        data["privacyStatus"] = "public"
    files = {"video": (os.path.basename(file_path), content, "video/mp4")}
    async with httpx.AsyncClient(timeout=300) as client:
        resp = await client.post(f"{API_BASE}/upload", headers=_headers(),
                                 data=data, files=files)
    if resp.status_code not in (200, 201, 202):
        print(f"⚠️  Autopilot publish failed ({resp.status_code}): {resp.text[:300]}")
        return False
    return True


def _read_bytes(path):
    with open(path, "rb") as f:
        return f.read()


# --------------------------------------------------------------------------- #
# Database helpers
# --------------------------------------------------------------------------- #
async def _load_user(user_id):
    from .auth import _load_current_user
    async with database.session() as s:
        return await _load_current_user(s, user_id)


async def _profile_for(user_id) -> Optional[str]:
    async with database.session() as s:
        prof = await s.get(UploadPostProfile, user_id)
        return prof.profile_username if prof else None


async def _get_settings(user_id) -> Optional[AutopilotSettings]:
    async with database.session() as s:
        return await s.get(AutopilotSettings, user_id)


async def _set_check_result(user_id, error: Optional[str]):
    async with database.session() as s:
        async with s.begin():
            await s.execute(update(AutopilotSettings)
                            .where(AutopilotSettings.user_id == user_id)
                            .values(last_checked_at=func.now(), last_error=error))


async def _count_recent(user_id, trigger: str, since: datetime) -> int:
    async with database.session() as s:
        return (await s.execute(
            select(func.count(AutopilotRun.id)).where(
                AutopilotRun.user_id == user_id,
                AutopilotRun.trigger == trigger,
                AutopilotRun.created_at >= since,
                AutopilotRun.status.in_((QUEUED, PROCESSING, COMPLETED, FAILED)),
            ))).scalar_one()


async def _seen_ids(user_id, video_ids) -> set:
    if not video_ids:
        return set()
    async with database.session() as s:
        rows = await s.execute(select(AutopilotRun.video_id).where(
            AutopilotRun.user_id == user_id, AutopilotRun.video_id.in_(list(video_ids))))
        return {r[0] for r in rows}


async def _claim(user_id, video: dict, trigger: str) -> Optional[uuid.UUID]:
    """INSERT the run row; the unique constraint makes the claim exclusive."""
    run_id = uuid.uuid4()
    try:
        async with database.session() as s:
            async with s.begin():
                s.add(AutopilotRun(
                    id=run_id, user_id=user_id, video_id=video["id"],
                    video_url=video_url(video),
                    video_title=(video.get("caption") or "")[:300] or None,
                    thumbnail_url=video.get("thumbnail_url"),
                    published_at=parse_ts(video.get("timestamp")),
                    trigger=trigger, status=QUEUED,
                ))
    except IntegrityError:
        return None
    return run_id


def video_url(video: dict) -> str:
    if video.get("permalink"):
        return video["permalink"]
    if str(video.get("id", "")).startswith(TWITCH_PREFIX):
        return f"https://www.twitch.tv/videos/{video['id'][len(TWITCH_PREFIX):]}"
    return f"https://www.youtube.com/watch?v={video['id']}"


async def _update_run(run_id, **values):
    async with database.session() as s:
        async with s.begin():
            await s.execute(update(AutopilotRun).where(AutopilotRun.id == run_id).values(**values))


async def _drop_run(run_id):
    async with database.session() as s:
        async with s.begin():
            await s.execute(delete(AutopilotRun).where(AutopilotRun.id == run_id))


# --------------------------------------------------------------------------- #
# Submitting a job through the normal pipeline
# --------------------------------------------------------------------------- #
async def _submit(user_id, email, run_id, video_url: str, max_minutes: int,
                  source_range=None) -> dict:
    """``POST /api/process`` in-process, as the user, and record the outcome."""
    from .auth import issue_jwt
    if _app is None:
        raise RuntimeError("autopilot not started")
    token = issue_jwt(user_id, email)
    base = os.environ.get("PUBLIC_API_URL", "").rstrip("/") or "http://localhost"
    body = {
        "url": video_url,
        # Recorded once when the user switched Autopilot on for their own
        # channel (rights_ack_at); the pipeline wants it per job.
        "acknowledged": True,
        "max_minutes": max_minutes,
        "auto_hook": "1",
    }
    if source_range:
        # The stretch viewers clipped most, billed alone (plan_source_range).
        body.pop("max_minutes", None)
        body["source_start"], body["source_end"] = source_range
    try:
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=_app),
                                     base_url=base, timeout=600) as client:
            resp = await client.post(
                "/api/process", json=body,
                headers={"Authorization": f"Bearer {token}",
                         "User-Agent": "OpenShorts-Autopilot/1.0"})
        try:
            payload = resp.json()
        except ValueError:
            payload = {}
        outcome = outcome_from_process(resp.status_code, payload)
    except Exception as e:
        print(f"⚠️  Autopilot submit error for {video_url}: {e}")
        outcome = {"status": QUEUED, "reason": "server_error", "job_id": None, "retry": True}

    if outcome["retry"]:
        await _drop_run(run_id)          # try this video again on a later tick
    else:
        await _update_run(run_id, status=outcome["status"], reason=outcome["reason"],
                          job_id=outcome["job_id"])
    print(f"🤖 Autopilot {video_url} -> {outcome['status']}"
          f"{' (' + outcome['reason'] + ')' if outcome['reason'] else ''}")
    return outcome


async def _start_video(user, video: dict, trigger: str, max_minutes: int) -> dict:
    """Claim a channel video and push it into the pipeline."""
    run_id = await _claim(user.id, video, trigger)
    if run_id is None:
        return {"status": "duplicate", "reason": "already_picked", "job_id": None}
    if video.get("source") == "twitch":
        if video.get("is_live"):
            await _drop_run(run_id)  # not finished: a later tick takes it
            return {"status": SKIPPED, "reason": "still_live", "job_id": None}
        stretch = await twitch_stretch(video, max_minutes)
        return await _submit(user.id, user.email, run_id, video_url(video), max_minutes,
                             source_range=stretch)
    if await is_youtube_private(video["id"]):
        if trigger != "manual":
            # Scheduled uploads are often private until they go live: let the
            # next poll pick the video up again instead of skipping it for good.
            async with database.session() as s:
                async with s.begin():
                    await s.execute(delete(AutopilotRun).where(AutopilotRun.id == run_id))
            return {"status": SKIPPED, "reason": "private_video", "job_id": None}
        await _update_run(run_id, status=SKIPPED, reason="private_video")
        return {"status": SKIPPED, "reason": "private_video", "job_id": None}
    if await is_youtube_short(video["id"]):
        await _update_run(run_id, status=SKIPPED, reason="youtube_short")
        return {"status": SKIPPED, "reason": "youtube_short", "job_id": None}
    url = video.get("permalink") or f"https://www.youtube.com/watch?v={video['id']}"
    return await _submit(user.id, user.email, run_id, url, max_minutes)


async def twitch_stretch(video: dict, max_minutes: int):
    """``(start, end)`` of the ``max_minutes`` the stream's viewers clipped most,
    or None (no signal, no Twitch credentials, or the broadcast fits)."""
    import twitch_signals
    total = float(video.get("duration_seconds") or 0)
    length = float(max_minutes or DEFAULT_MAX_MINUTES) * 60.0
    if total and total <= length:
        return None
    try:
        clips = await asyncio.to_thread(twitch_signals.fetch_viewer_clips,
                                        str(video.get("source_id") or ""))
    except Exception:
        return None
    stretch = twitch_signals.best_stretch(clips, total, length)
    if stretch:
        print(f"🤖 Autopilot Twitch {video.get('source_id')}: clipping "
              f"{stretch[0] / 60:.0f}-{stretch[1] / 60:.0f} min ({len(clips)} viewer clips)")
    return stretch


# --------------------------------------------------------------------------- #
# The poller
# --------------------------------------------------------------------------- #
async def check_user(user_id) -> Optional[str]:
    """One channel check for one user. Returns an error code or None."""
    prefs = await _get_settings(user_id)
    if prefs is None or not prefs.enabled:
        return None
    user = await _load_user(user_id)
    if not eligible(user):
        await _set_check_result(user_id, "plan_required")
        return "plan_required"
    profile = await _profile_for(user_id)
    if not profile:
        await _set_check_result(user_id, "youtube_not_connected")
        return "youtube_not_connected"
    try:
        media = await fetch_all_videos(profile)
    except ChannelError as e:
        await _set_check_result(user_id, e.code)
        return e.code

    now = datetime.now(timezone.utc)
    seen = await _seen_ids(user_id, [m.get("id") for m in media if m.get("id")])
    fresh = pick_new_videos(media, parse_ts(prefs.enabled_at), seen, now)
    budget = DAILY_AUTO_LIMIT - await _count_recent(user_id, "auto", now - timedelta(days=1))
    for video in fresh:
        if budget <= 0:
            break  # the rest stay unclaimed and are picked up on a later day
        result = await _start_video(user, video, "auto", prefs.max_minutes or DEFAULT_MAX_MINUTES)
        if result["status"] in (PROCESSING, FAILED):
            budget -= 1
        if result.get("reason") in ("out_of_minutes", "busy"):
            break  # nothing else will fit right now
    await _set_check_result(user_id, None)
    return None


async def _expire_stale_runs():
    """A job that never reported back (container killed past its resume
    attempts) must not show 'processing' forever."""
    cutoff = datetime.now(timezone.utc) - STALE_RUN_AFTER
    async with database.session() as s:
        async with s.begin():
            await s.execute(update(AutopilotRun).where(
                AutopilotRun.status.in_((QUEUED, PROCESSING)),
                AutopilotRun.updated_at < cutoff,
            ).values(status=FAILED, reason="timeout"))


async def poll_once():
    await _expire_stale_runs()
    due_before = datetime.now(timezone.utc) - CHECK_EVERY
    async with database.session() as s:
        user_ids = [r[0] for r in await s.execute(
            select(AutopilotSettings.user_id).where(
                AutopilotSettings.enabled.is_(True),
                (AutopilotSettings.last_checked_at.is_(None))
                | (AutopilotSettings.last_checked_at < due_before),
            ).order_by(AutopilotSettings.last_checked_at.asc().nullsfirst()))]
    for uid in user_ids:
        if _is_active is not None and not _is_active():
            return  # draining for a deploy: the new instance takes over
        try:
            await check_user(uid)
        except Exception as e:
            print(f"⚠️  Autopilot check failed for {uid}: {e}")
        await asyncio.sleep(1)  # be gentle with Upload-Post and YouTube


async def _loop():
    await asyncio.sleep(60)  # let the instance finish booting and resuming jobs
    while True:
        try:
            if _is_active is None or _is_active():
                await poll_once()
        except asyncio.CancelledError:
            return
        except Exception as e:
            print(f"⚠️  Autopilot loop error: {e}")
        await asyncio.sleep(TICK_SECONDS)


def start(app, is_active=None):
    """Start the poller. ``is_active`` returns False while the instance drains."""
    global _app, _is_active, _task
    _app = app
    _is_active = is_active
    if os.environ.get("AUTOPILOT_DISABLED", "").lower() in ("1", "true", "yes"):
        print("🤖 Autopilot poller disabled (AUTOPILOT_DISABLED).")
        return
    _task = asyncio.create_task(_loop())


# --------------------------------------------------------------------------- #
# Job completion (called from app.run_job_wrapper)
# --------------------------------------------------------------------------- #
async def on_job_finished(job_id: str, job: dict, failure_reason: Optional[str] = None):
    """Settle the run for ``job_id`` if Autopilot started it; no-op otherwise.

    Runs on the instance that ran the job. The status flip is a guarded
    UPDATE, so a second call for the same job does nothing.
    """
    if not job_id or not isinstance(job, dict):
        return
    ok = job.get("status") == "completed"
    clips = ((job.get("result") or {}).get("clips")) or []
    async with database.session() as s:
        async with s.begin():
            row = (await s.execute(
                update(AutopilotRun)
                .where(and_(AutopilotRun.job_id == job_id,
                            AutopilotRun.status == PROCESSING))
                .values(status=COMPLETED if ok else FAILED,
                        reason=None if ok else (failure_reason or "processing_failed"),
                        clips_count=len(clips) if ok else 0)
                .returning(AutopilotRun.id, AutopilotRun.user_id, AutopilotRun.video_title)
            )).first()
    if row is None or not ok or not clips:
        return
    run_id, user_id, video_title = row

    prefs = await _get_settings(user_id)
    platforms = clean_platforms(prefs.publish_platforms if prefs else [])
    to_post = []
    if prefs and prefs.autopublish and platforms:
        n = max(0, min(MAX_CLIPS_TO_PUBLISH, prefs.clips_to_publish or DEFAULT_CLIPS_TO_PUBLISH))
        to_post = rank_clips(clips, n)

    # Our own email replaces the generic "clips ready" one for this job.
    job["email_sent"] = True
    try:
        from .emails import send_autopilot_clips_email
        async with database.session() as s:
            user = await s.get(User, user_id)
        if user and user.email:
            await send_autopilot_clips_email(
                user.email, video_title, len(clips), len(to_post),
                f"{settings.frontend_url}/#app?tab=autopilot")
    except Exception as e:
        print(f"⚠️  Autopilot email failed for {job_id}: {e}")

    if to_post:
        profile = await _profile_for(user_id)
        output_dir = job.get("output_dir") or ""
        items = []
        for i in to_post:
            rel = clips[i].get("video_url") or ""
            path = os.path.join(output_dir, os.path.basename(rel))
            if rel and os.path.exists(path):
                items.append((clips[i], path))
        if profile and items:
            task = asyncio.create_task(_publish(run_id, profile, items, platforms, prefs))
            _bg_tasks.add(task)
            task.add_done_callback(_bg_tasks.discard)


async def _publish(run_id, profile, items, platforms, prefs):
    """Schedule the chosen clips, one a day. Runs in the background so the
    job slot is not held while the files upload."""
    slots = schedule_slots(datetime.now(timezone.utc), len(items),
                           prefs.publish_hour if prefs.publish_hour is not None else DEFAULT_PUBLISH_HOUR,
                           prefs.timezone)
    posted = 0
    for (clip, path), when in zip(items, slots):
        try:
            if await _upload_clip(profile, path, clip, platforms, when, prefs.timezone or "UTC"):
                posted += 1
        except Exception as e:
            print(f"⚠️  Autopilot publish error: {e}")
    await _update_run(run_id, posted_count=posted,
                      reason=None if posted == len(items) else "publish_partial")


# --------------------------------------------------------------------------- #
# API
# --------------------------------------------------------------------------- #
class AutopilotUpdate(BaseModel):
    enabled: Optional[bool] = None
    rights_ack: Optional[bool] = None
    autopublish: Optional[bool] = None
    publish_platforms: Optional[List[str]] = None
    clips_to_publish: Optional[int] = None
    max_minutes: Optional[int] = None
    publish_hour: Optional[int] = None
    timezone: Optional[str] = None


class ManualRun(BaseModel):
    video_id: str


def _settings_view(prefs: Optional[AutopilotSettings]) -> dict:
    return {
        "enabled": bool(prefs and prefs.enabled),
        "enabled_at": prefs.enabled_at.isoformat() if prefs and prefs.enabled_at else None,
        "rights_ack": bool(prefs and prefs.rights_ack_at),
        "autopublish": bool(prefs and prefs.autopublish),
        "publish_platforms": clean_platforms(prefs.publish_platforms) if prefs else [],
        "clips_to_publish": (prefs.clips_to_publish if prefs else None) or DEFAULT_CLIPS_TO_PUBLISH,
        "max_minutes": (prefs.max_minutes if prefs else None) or DEFAULT_MAX_MINUTES,
        "publish_hour": prefs.publish_hour if prefs and prefs.publish_hour is not None
        else DEFAULT_PUBLISH_HOUR,
        "timezone": prefs.timezone if prefs else None,
        "last_checked_at": prefs.last_checked_at.isoformat() if prefs and prefs.last_checked_at else None,
        "last_error": prefs.last_error if prefs else None,
    }


def _run_view(r: AutopilotRun) -> dict:
    return {
        "id": str(r.id),
        "video_id": r.video_id,
        "video_url": r.video_url,
        "video_title": r.video_title,
        "thumbnail_url": r.thumbnail_url,
        "published_at": r.published_at.isoformat() if r.published_at else None,
        "trigger": r.trigger,
        "status": r.status,
        "reason": r.reason,
        "job_id": r.job_id,
        "clips_count": r.clips_count,
        "posted_count": r.posted_count,
        "created_at": r.created_at.isoformat() if r.created_at else None,
    }


async def _recent_runs(user_id, limit=20):
    async with database.session() as s:
        return list((await s.execute(
            select(AutopilotRun).where(AutopilotRun.user_id == user_id)
            .order_by(AutopilotRun.created_at.desc()).limit(limit))).scalars())


@router.get("/api/autopilot")
async def get_autopilot(request: Request):
    from .auth import get_current_user_required
    user = await get_current_user_required(request)
    prefs = await _get_settings(user.id)
    profile = await _profile_for(user.id)
    accounts = {}
    if profile:
        try:
            accounts = await fetch_social_accounts(profile)
        except Exception:
            accounts = {}
    return {
        "eligible": eligible(user),
        "plan": user.plan,
        "settings": _settings_view(prefs),
        "accounts": accounts,
        "youtube": accounts.get("youtube"),
        "twitch": accounts.get("twitch") if twitch_enabled() else None,
        "twitch_available": twitch_enabled(),
        "limits": {
            "daily_auto": DAILY_AUTO_LIMIT,
            "max_minutes_choices": list(MAX_MINUTES_CHOICES),
            "max_clips_to_publish": MAX_CLIPS_TO_PUBLISH,
            "platforms": list(PUBLISH_PLATFORMS),
        },
        "runs": [_run_view(r) for r in await _recent_runs(user.id)],
    }


@router.put("/api/autopilot")
async def put_autopilot(payload: AutopilotUpdate, request: Request):
    from .auth import get_current_user_required
    user = await get_current_user_required(request)
    turning_on = payload.enabled is True
    if turning_on and not eligible(user):
        raise HTTPException(status_code=403, detail={
            "error": "plan_required",
            "message": "Autopilot is included in the Starter, Creator and Pro plans."})

    async with database.session() as s:
        async with s.begin():
            prefs = await s.get(AutopilotSettings, user.id)
            if prefs is None:
                prefs = AutopilotSettings(user_id=user.id, enabled=False, autopublish=False,
                                          clips_to_publish=DEFAULT_CLIPS_TO_PUBLISH,
                                          max_minutes=DEFAULT_MAX_MINUTES,
                                          publish_hour=DEFAULT_PUBLISH_HOUR)
                s.add(prefs)
            # Connecting the channel through YouTube's own OAuth already proves it
            # is theirs, so switching Autopilot on records the attestation
            # instead of asking for a checkbox.
            if payload.rights_ack or turning_on:
                prefs.rights_ack_at = prefs.rights_ack_at or datetime.now(timezone.utc)
            if turning_on:
                if not prefs.enabled:
                    # New baseline: only videos published from now on.
                    prefs.enabled_at = datetime.now(timezone.utc)
                    prefs.last_checked_at = None
                    prefs.last_error = None
                prefs.enabled = True
            elif payload.enabled is False:
                prefs.enabled = False
            if payload.autopublish is not None:
                prefs.autopublish = bool(payload.autopublish)
            if payload.publish_platforms is not None:
                prefs.publish_platforms = clean_platforms(payload.publish_platforms)
            if payload.clips_to_publish is not None:
                prefs.clips_to_publish = max(1, min(MAX_CLIPS_TO_PUBLISH, int(payload.clips_to_publish)))
            if payload.max_minutes is not None:
                if int(payload.max_minutes) not in MAX_MINUTES_CHOICES:
                    raise HTTPException(status_code=400, detail="Unsupported max_minutes value.")
                prefs.max_minutes = int(payload.max_minutes)
            if payload.publish_hour is not None:
                prefs.publish_hour = max(0, min(23, int(payload.publish_hour)))
            if payload.timezone is not None:
                tz = payload.timezone.strip()[:64]
                prefs.timezone = tz if (tz and _zone(tz) is not timezone.utc) or tz == "UTC" else prefs.timezone
    return {"settings": _settings_view(await _get_settings(user.id))}


@router.get("/api/autopilot/videos")
async def list_channel_videos(request: Request):
    """The channel's latest uploads, each with what Autopilot did with it."""
    from .auth import get_current_user_required
    user = await get_current_user_required(request)
    profile = await _profile_for(user.id)
    if not profile:
        return {"videos": [], "error": "youtube_not_connected"}
    try:
        media = await fetch_all_videos(profile)
    except ChannelError as e:
        return {"videos": [], "error": e.code}
    ids = [m.get("id") for m in media if m.get("id")]
    runs = {}
    if ids:
        async with database.session() as s:
            for r in (await s.execute(select(AutopilotRun).where(
                    AutopilotRun.user_id == user.id,
                    AutopilotRun.video_id.in_(ids)))).scalars():
                runs[r.video_id] = _run_view(r)
    return {"videos": [{
        "id": m.get("id"),
        "title": m.get("caption"),
        "url": m.get("permalink"),
        "thumbnail_url": m.get("thumbnail_url"),
        "published_at": m.get("timestamp"),
        "source": m.get("source") or "youtube",
        "is_live": bool(m.get("is_live")),
        "run": runs.get(m.get("id")),
    } for m in media if m.get("id")], "error": None}


@router.post("/api/autopilot/run")
async def run_now(payload: ManualRun, request: Request):
    """Clip one channel video now ("clip my latest video"), outside the
    automatic schedule. The video must belong to the user's own channel."""
    from .auth import get_current_user_required
    user = await get_current_user_required(request)
    if not eligible(user):
        raise HTTPException(status_code=403, detail={
            "error": "plan_required",
            "message": "Autopilot is included in the Starter, Creator and Pro plans."})
    prefs = await _get_settings(user.id)
    now = datetime.now(timezone.utc)
    if await _count_recent(user.id, "manual", now - timedelta(days=1)) >= MANUAL_DAILY_LIMIT:
        raise HTTPException(status_code=429, detail="Daily limit reached. Try again tomorrow.")
    profile = await _profile_for(user.id)
    if not profile:
        raise HTTPException(status_code=400, detail={"error": "youtube_not_connected",
                                                     "message": "Connect your YouTube channel first."})
    try:
        media = await fetch_all_videos(profile, limit=50)
    except ChannelError as e:
        raise HTTPException(status_code=400, detail={"error": e.code, "message": "Could not read your channel."})
    video = next((m for m in media if m.get("id") == payload.video_id), None)
    if video is None:
        raise HTTPException(status_code=404, detail="That video is not on your connected channel.")
    # A video that failed or was skipped earlier (out of minutes, a hiccup) can
    # be retried by hand; one that is running or done cannot be doubled.
    async with database.session() as s:
        async with s.begin():
            await s.execute(delete(AutopilotRun).where(
                AutopilotRun.user_id == user.id,
                AutopilotRun.video_id == payload.video_id,
                AutopilotRun.status.in_((FAILED, SKIPPED))))
    result = await _start_video(user, video, "manual",
                                (prefs.max_minutes if prefs else None) or DEFAULT_MAX_MINUTES)
    if result["status"] == "duplicate":
        raise HTTPException(status_code=409, detail="This video was already clipped by Autopilot.")
    if result.get("retry") or result["status"] == QUEUED:
        raise HTTPException(status_code=429, detail=(
            "You already have the maximum number of jobs running. Try again in a few minutes."
            if result.get("reason") == "busy" else "Could not start the job. Try again shortly."))
    return {"status": result["status"], "reason": result.get("reason"), "job_id": result.get("job_id")}
