import os
import llm_backend
import re
import sys
import uuid
import subprocess
import threading
import json
import shutil
import glob
import hashlib
import hmac
import time
import zipfile
import math
import itertools
import functools
import asyncio
import signal
import socket
from datetime import datetime, timezone, timedelta
from dotenv import load_dotenv
from typing import Any, Dict, Optional, List
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request, Header, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from starlette.background import BackgroundTask
from pydantic import BaseModel
from s3_uploader import upload_job_artifacts, list_all_clips, upload_actor_to_s3, list_actor_gallery, upload_video_to_gallery, list_video_gallery
import recut
import layout_ranges

load_dotenv()

# Constants
UPLOAD_DIR = "uploads"
OUTPUT_DIR = "output"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Configuration
# Default to 1 if not set, but user can set higher for powerful servers
MAX_CONCURRENT_JOBS = int(os.environ.get("MAX_CONCURRENT_JOBS", "5"))
MAX_FILE_SIZE_MB = 2048  # 2GB limit

# How TikTok receives our uploads. MEDIA_UPLOAD lands the video in the user's
# TikTok drafts so they finish the post inside TikTok's own editor; DIRECT_POST
# publishes straight to their feed, which is Upload-Post's default.
#
# Drafts are the safer default for an automated pipeline: nothing reaches an
# audience without the account owner seeing it first, and TikTok's own editor is
# where covers, sounds and hashtags actually get chosen. The UI must say so —
# a user who expects a published post and finds a draft will read it as a bug.
TIKTOK_POST_MODE = os.environ.get("TIKTOK_POST_MODE", "MEDIA_UPLOAD").strip()
# Ceiling for the working directory once it lives on a persistent volume: the
# age-based sweep alone can't stop a burst of long videos from filling the disk.
# 0 disables the cap.
OUTPUT_MAX_GB = int(os.environ.get("OUTPUT_MAX_GB", "25"))
# Same idea for source uploads, which are the biggest single files on disk.
UPLOADS_MAX_GB = int(os.environ.get("UPLOADS_MAX_GB", "15"))
# Pre-flight quality gate: warn before processing a YouTube source below this
# height (0 disables). Only applies to URLs; uploads are whatever the user gave.
QUALITY_GATE_MIN_HEIGHT = int(os.environ.get("QUALITY_GATE_MIN_HEIGHT", "720"))
# Reject sources shorter than this before starting (0 disables). A 24s YouTube
# Short cannot yield 15-60s clips: Gemini returns nothing, the job burns
# managed minutes and dies with "no usable clips" (prod 20-ago: 3 of 5 recent
# failures were exactly this, one user retrying the same 24s video).
MIN_SOURCE_SECONDS = int(os.environ.get("MIN_SOURCE_SECONDS", "45"))
QUALITY_PROBE_SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "quality_probe.py")
DISABLE_YOUTUBE_URL = os.environ.get("DISABLE_YOUTUBE_URL", "false").lower() in ("1", "true", "yes")

# Every log line in this module is emoji-prefixed, and a Windows console is
# cp1252 by default. _recover_jobs_from_disk() prints one during startup, so
# without this the server dies before it ever listens:
#
#   UnicodeEncodeError: 'charmap' codec can't encode characters in position 0-1
#   ERROR:    Application startup failed. Exiting.
#
# subtitles._configure_stdio solved this for the transcription path; the server
# needs it too, and needs it before the first print.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

# ---- Cloud billing (paid / managed-keys) integration --------------------------
# All paid-mode code lives in the optional `cloud/` package and is imported ONLY
# when BILLING_ENABLED is set. With the flag off, the app behaves exactly as the
# self-hosted BYOK app does today (no extra dependencies required).
BILLING_ENABLED = os.environ.get("BILLING_ENABLED", "").lower() in ("1", "true", "yes")

# Job/file retention (issue #46). Self-host defaults to 24h: the 1h sweep kept
# deleting finished projects under users who never touched their env, and the
# OUTPUT_MAX_GB / UPLOADS_MAX_GB caps below already bound the disk. Cloud keeps
# the tight default because clips are archived to R2 as soon as a job finishes.
JOB_RETENTION_SECONDS = int(
    os.environ.get("JOB_RETENTION_SECONDS", "3600" if BILLING_ENABLED else "86400")
)
# The retained download of a URL job (--keep-original) is the one artifact that
# is a full copy of someone else's video rather than something we made, so it
# can be aged out ahead of the clips it produced. Defaults to the job clock,
# i.e. no change: dropping it earlier costs the clip editor, whose rerender,
# reframe, scenes and EDL endpoints all read that file and answer 409 once it
# is gone. Lower it only if you would rather lose in-session re-edits than
# keep the original around. Uploads are deliberately untouched: that file is
# the user's own content, which they attested to owning.
SOURCE_RETENTION_SECONDS = int(
    os.environ.get("SOURCE_RETENTION_SECONDS", str(JOB_RETENTION_SECONDS))
)
# Force full pipeline logs to the client even under billing (local debugging).
DEBUG_LOGS = os.environ.get("DEBUG_LOGS", "").lower() in ("1", "true", "yes")

if BILLING_ENABLED:
    import cloud
    from cloud import managed_keys, metering as _metering, config as _cloud_config, alerts as _alerts
    from cloud.auth import get_current_user_optional
else:
    cloud = None
    managed_keys = None
    _metering = None
    _cloud_config = None
    _alerts = None

    async def get_current_user_optional(request: Request):
        # No-op dependency in self-host mode: every request is anonymous / BYOK.
        return None


async def _user_from_request(request: Request):
    """Load the authenticated cloud user (or None). Cheap indexed lookup."""
    return await get_current_user_optional(request)


async def resolve_gemini(request: Request) -> Optional[str]:
    """Resolve the Gemini API key for a request.

    Cloud (hosted) is PAID-ONLY: there is no BYOK for the core pipeline, so the
    ``X-Gemini-Key`` header is ignored — an entitled user (active plan or trial)
    gets the managed server key, everyone else gets ``None`` (→ 402, start trial).
    Self-host keeps BYOK: header wins, else the env fallback.
    """
    if BILLING_ENABLED:
        user = await _user_from_request(request)
        if managed_keys.has_active_entitlement(user):
            return managed_keys.gemini_key()
        return None
    header = request.headers.get("X-Gemini-Key")
    if header:
        return header
    return os.environ.get("GEMINI_API_KEY")


async def resolve_upload_post(request: Request, body_key: Optional[str] = None):
    """Resolve the Upload-Post key and the profile to post as.

    Returns ``(api_key, forced_profile_username_or_None)``. Cloud is paid-only:
    an entitled user gets the managed key + their own forced profile (body key /
    user_id ignored); a non-entitled user gets ``(None, None)``. Self-host keeps
    BYOK: header, then body key, then env.
    """
    if BILLING_ENABLED:
        user = await _user_from_request(request)
        if managed_keys.has_active_entitlement(user):
            profile = await cloud.social_profiles.ensure_profile(user)
            return managed_keys.upload_post_key(), profile
        return None, None
    header = request.headers.get("X-Upload-Post-Key")
    key = header or body_key or os.environ.get("UPLOAD_POST_API_KEY")
    return key, None


def resolve_post_profile(forced_profile: Optional[str], client_profile: Optional[str]) -> str:
    """The Upload-Post profile to act as, for posting/scheduling/analytics.

    Fails closed on purpose. Every call site used to read
    ``forced_profile or client_profile``, which quietly honours whatever
    profile the *client* asked for if the server ever failed to resolve its
    own — one refactor of ``resolve_upload_post`` away from letting a cloud
    user schedule into someone else's connected accounts. In cloud mode the
    client value is never consulted: either the server knows the caller's
    profile or the request is refused.
    """
    if BILLING_ENABLED:
        if not forced_profile:
            raise HTTPException(
                status_code=503,
                detail="Could not resolve your social profile. Please try again.")
        return forced_profile
    # Self-host: no user model, the caller owns the Upload-Post account whose
    # key resolved above, so it picks its own profile.
    profile = forced_profile or client_profile
    if not profile:
        raise HTTPException(status_code=400, detail="Missing Upload-Post user profile")
    return profile


def gemini_missing_error():
    """The right 4xx when no Gemini key could be resolved.

    402 for a signed-in-but-not-entitled cloud user (needs a plan); 400 otherwise
    (BYOK header simply missing).
    """
    if BILLING_ENABLED:
        return HTTPException(status_code=402, detail={
            "error": "no_plan",
            "message": "This action needs an active plan. Choose a plan or add your own API key.",
        })
    return HTTPException(status_code=400, detail="Missing X-Gemini-Key header")


# Probe rate limiter. In-memory, resets on restart by design — the hard monthly
# quota lives in the metering ledger; this only stops someone hammering the
# proxy with metadata probes.
_probe_times: dict = {}  # user_id -> [monotonic timestamps]
PROBES_PER_HOUR = 15

# Out-of-minutes upsell email: at most one per user per day (a client may
# retry the same 402 many times).
_last_quota_email: dict = {}
_QUOTA_EMAIL_COOLDOWN = 24 * 3600


def _maybe_send_quota_email(user):
    if user is None or user.plan != "free" or not user.email:
        return
    now = time.monotonic()
    last = _last_quota_email.get(str(user.id))
    if last is not None and now - last < _QUOTA_EMAIL_COOLDOWN:
        return
    _last_quota_email[str(user.id)] = now
    from cloud.emails import send_out_of_minutes_email
    upgrade_url = f"{_cloud_config.settings.frontend_url}/#/pricing"
    asyncio.create_task(send_out_of_minutes_email(user.email, upgrade_url))


def _check_probe_rate(user_id):
    now = time.monotonic()
    times = _probe_times.setdefault(str(user_id), [])
    times[:] = [t for t in times if now - t < 3600]
    if len(times) >= PROBES_PER_HOUR:
        raise HTTPException(status_code=429,
                            detail="Too many requests this hour. Please slow down.")
    times.append(now)


async def reserve_process_minutes(request, url, input_path, job_id):
    """Meter a managed /api/process request.

    Returns (user_id, priority, reservation_id, plan).

    BYOK / self-host requests don't consume minutes (priority 2, no reservation).
    For a managed (entitled, no BYOK header) request this probes the input
    duration, enforces the per-user concurrent-job limit, and reserves minutes —
    raising 402 (quota) or 429 (too many jobs) as needed.

    NOTE: in cloud mode ``resolve_gemini`` ignores ``X-Gemini-Key`` (paid-only,
    no BYOK), so we must NOT skip metering just because that header is present —
    otherwise a client could send a dummy header and run unlimited managed jobs
    on the operator's key for free. Only skip metering when billing is off.
    """
    if not BILLING_ENABLED:
        return None, 2, None, None
    user = await _user_from_request(request)
    if not managed_keys.has_active_entitlement(user):
        return None, 2, None, None  # shouldn't happen (resolve_gemini would have 402'd)

    priority = _cloud_config.PLAN_PRIORITY.get(user.plan, 1)

    # Per-user simultaneous job cap.
    limit = _cloud_config.PLAN_JOB_LIMIT.get(user.plan, 2)
    active = sum(1 for j in jobs.values()
                 if j.get('user_id') == user.id and j.get('status') in ('queued', 'processing'))
    if active >= limit:
        raise HTTPException(status_code=429,
                            detail="You already have the maximum number of jobs running. Please wait.")

    # Out of minutes -> 402 before probing. The probe is a real yt-dlp metadata
    # fetch through the download proxies, and every job costs at least one
    # minute, so a user at zero can be turned away without spending bandwidth on
    # a duration we are about to reject anyway (4 of 12 submissions in the
    # 21-aug-2026 sample were quota 402s that had already paid for their probe).
    balance = await _metering.get_balance(user.id)
    if balance["remaining"] < 1:
        _maybe_send_quota_email(user)
        raise HTTPException(status_code=402, detail={
            "error": "quota_exceeded",
            "minutes_required": 1,
            "minutes_remaining": balance["remaining"],
        })

    # Probe rate limit: probing costs a (cheap) proxied metadata call. The
    # 20-minute monthly quota is the real bound on free usage; there is no daily
    # job cap.
    _check_probe_rate(user.id)

    # Probe input duration (blocking → run in a thread). When today's paid
    # traffic is over budget, the probe (and below, the job itself) runs
    # without the per-GB proxy: statics or nothing.
    try:
        from cloud import proxy_ledger as _pl
        paid_allowed = not await _pl.budget_exceeded()
    except Exception:
        pass
    loop = asyncio.get_event_loop()
    try:
        if url:
            minutes = await loop.run_in_executor(
                None, functools.partial(_metering.probe_url_minutes, url,
                                        allow_paid=paid_allowed))
        else:
            minutes = await loop.run_in_executor(None, _metering.probe_file_minutes, input_path)
    except Exception:
        raise HTTPException(status_code=400,
                            detail="Could not determine the video duration. Try a different source.")
    finally:
        # A probe that had to reach the paid proxy leaves an event behind;
        # record it (DB row + Telegram) whether or not the probe succeeded.
        try:
            from cloud import proxy_ledger as _pl
            await _pl.drain_probe_events()
        except Exception:
            pass
    minutes = max(1, math.ceil(minutes))

    try:
        reservation_id = await _metering.reserve_minutes(user.id, minutes, job_id)
    except _metering.QuotaExceeded as e:
        _maybe_send_quota_email(user)
        raise HTTPException(status_code=402, detail={
            "error": "quota_exceeded",
            "minutes_required": e.required,
            "minutes_remaining": e.remaining,
        })

    return user.id, priority, reservation_id, user.plan


async def reserve_managed_action(request, minutes, job_id, job_type):
    """Reserve quota for a synchronous managed action (e.g. thumbnail image gen).

    Returns a reservation_id to commit/release around the work, or None for
    BYOK / self-host. Raises 402 when the user is out of minutes.
    """
    if not BILLING_ENABLED:
        return None
    if minutes <= 0:
        # Free action (e.g. burning captions). Skip the ledger entirely rather
        # than writing a 0-minute row on every call — the endpoint's own
        # entitlement gate is what bounds it.
        return None
    user = await _user_from_request(request)
    if not managed_keys.has_active_entitlement(user):
        return None  # BYOK header path (self-host) — not metered
    try:
        return await _metering.reserve_minutes(user.id, minutes, job_id, job_type)
    except _metering.QuotaExceeded as e:
        _maybe_send_quota_email(user)
        raise HTTPException(status_code=402, detail={
            "error": "quota_exceeded",
            "minutes_required": e.required,
            "minutes_remaining": e.remaining,
        })


async def require_managed_entitlement(request):
    """Gate a managed compute endpoint that doesn't resolve a Gemini key itself.

    Some endpoints (subtitle/hook FFmpeg re-encodes, render proxy, the thumbnail
    upload that kicks off a YouTube download + Whisper) do expensive server work
    without ever calling ``resolve_gemini``, so nothing was stopping an anonymous
    or non-entitled caller from driving unbounded compute in cloud mode. In cloud
    mode this rejects them with 402; it's a no-op for self-host (BILLING off).
    """
    if not BILLING_ENABLED:
        return None
    user = await _user_from_request(request)
    if not managed_keys.has_active_entitlement(user):
        raise gemini_missing_error()
    return user


async def _owner_id(request):
    """The authenticated cloud user's id to stamp on a new job/session, or None
    for self-host / BYOK / anonymous (BILLING off → nothing to scope)."""
    if not BILLING_ENABLED:
        return None
    user = await _user_from_request(request)
    return user.id if user else None


async def _assert_job_owner(request, record):
    """Cloud multi-tenant guard: reject unless the caller owns this in-memory
    job/session record.

    No-op for self-host (BILLING off) and for records with no owner stamped
    (BYOK / self-host jobs never set ``user_id``). Returns 404 rather than 403 so
    a non-owner can't even confirm the id exists. UUID ids already make these
    stores hard to enumerate; this closes the gap for a shared/leaked id.
    """
    if not BILLING_ENABLED:
        return
    owner = record.get("user_id") if isinstance(record, dict) else None
    if owner is None:
        return
    user = await _user_from_request(request)
    # Compare as strings: live jobs store a uuid.UUID, but jobs recovered from
    # the .owner sidecar store its string form — UUID != str is always True.
    if user is None or str(user.id) != str(owner):
        raise HTTPException(status_code=404, detail="Not found")

# Application State
# PriorityQueue holds (priority, seq, job_id). Lower priority dispatches first:
# pro=0, starter/creator=1, BYOK/anonymous/self-host=2. The seq counter keeps
# FIFO order within a priority and makes the tuples always comparable. With
# BILLING disabled every job enqueues at priority 2 → plain FIFO as before.
job_queue = asyncio.PriorityQueue()
_job_seq = itertools.count()
jobs: Dict[str, Dict] = {}
thumbnail_sessions: Dict[str, Dict] = {}
publish_jobs: Dict[str, Dict] = {}  # {publish_id: {status, result, error}}
# Semester to limit concurrency to MAX_CONCURRENT_JOBS
concurrency_semaphore = asyncio.Semaphore(MAX_CONCURRENT_JOBS)


def _enqueue_job(job_id: str, priority: int = 2):
    job_queue.put_nowait((priority, next(_job_seq), job_id))

def _relocate_root_job_artifacts(job_id: str, job_output_dir: str) -> bool:
    """
    Backward-compat rescue:
    If main.py accidentally wrote metadata/clips into OUTPUT_DIR root (e.g. output/<jobid>_...),
    move them into output/<job_id>/ so the API can find and serve them.
    """
    try:
        os.makedirs(job_output_dir, exist_ok=True)
        root = OUTPUT_DIR
        pattern = os.path.join(root, f"{job_id}_*_metadata.json")
        meta_candidates = sorted(glob.glob(pattern), key=lambda p: os.path.getmtime(p), reverse=True)
        if not meta_candidates:
            return False

        # Move the newest metadata and its associated clips.
        metadata_path = meta_candidates[0]
        base_name = os.path.basename(metadata_path).replace("_metadata.json", "")

        # Move metadata
        dest_metadata = os.path.join(job_output_dir, os.path.basename(metadata_path))
        if os.path.abspath(metadata_path) != os.path.abspath(dest_metadata):
            shutil.move(metadata_path, dest_metadata)

        # Move any clips that match the same base_name into the job folder
        clip_pattern = os.path.join(root, f"{base_name}_clip_*.mp4")
        for clip_path in glob.glob(clip_pattern):
            dest_clip = os.path.join(job_output_dir, os.path.basename(clip_path))
            if os.path.abspath(clip_path) != os.path.abspath(dest_clip):
                shutil.move(clip_path, dest_clip)

        # Also move any temp_ clips that might remain
        temp_clip_pattern = os.path.join(root, f"temp_{base_name}_clip_*.mp4")
        for clip_path in glob.glob(temp_clip_pattern):
            dest_clip = os.path.join(job_output_dir, os.path.basename(clip_path))
            if os.path.abspath(clip_path) != os.path.abspath(dest_clip):
                shutil.move(clip_path, dest_clip)

        return True
    except Exception:
        return False

def _canonical_clip_file(output_dir, base_name, index):
    """The file to serve for clip ``index``, preferring a derived version.

    The pipeline writes the clean reframe as ``<base>_clip_<n>.mp4`` and any
    post-processing (auto-captions, /api/subtitle re-styles, and clip-editor
    recuts) as ``subtitled_<ts>_<clean>.mp4`` / ``recut_<ts>_<clean>.mp4``,
    keeping the original for re-styling. Every place that rebuilds the
    canonical name from disk — restore after a restart, the R2 upload, the
    download bundle — must therefore resolve to the newest derived file, or
    clips silently lose their captions (or their recut) on a redeploy.
    """
    clean = f"{base_name}_clip_{index + 1}.mp4"
    try:
        # subtitled_*_{clean} also matches subtitled_<ts>_recut_<ts>_{clean}
        # and subtitled_<ts>_hooked_<ts>_{clean}, i.e. captioned recuts and
        # captioned hooks; the bare recut_/hooked_/hook_ patterns cover
        # derivations that shipped uncaptioned (hook_ is the legacy manual-
        # hook prefix, kept so old jobs still resolve).
        derived = (glob.glob(os.path.join(output_dir, f"subtitled_*_{clean}"))
                   + glob.glob(os.path.join(output_dir, f"recut_*_{clean}"))
                   + glob.glob(os.path.join(output_dir, f"hooked_*_{clean}"))
                   + glob.glob(os.path.join(output_dir, f"hook_{clean}")))
    except Exception:
        derived = []
    if not derived:
        return clean
    # Highest timestamp wins — that's the most recent styling.
    return os.path.basename(max(derived, key=os.path.getmtime))


def _strip_burned_captions(output_dir, filename):
    """Walk ``subtitled_<ts>_`` prefixes back to the file without burned captions.

    Returns the name unchanged when there is nothing to strip (or when the
    underlying file is gone, e.g. a library restore that only kept the current
    version).
    """
    while True:
        m = re.match(r'^subtitled_\d+_(.+)$', filename)
        if not m or not os.path.exists(os.path.join(output_dir, m.group(1))):
            return filename
        filename = m.group(1)


def _strip_burned_hook(output_dir, filename):
    """Walk ``hooked_<ts>_`` (and legacy ``hook_``) prefixes back to the file
    without a burned hook. Same fail-safe contract as _strip_burned_captions:
    the name is returned unchanged when there is nothing to strip or the
    underlying file is gone."""
    while True:
        m = re.match(r'^(?:hooked_\d+_|hook_)(.+)$', filename)
        if not m or not os.path.exists(os.path.join(output_dir, m.group(1))):
            return filename
        filename = m.group(1)


def _reapply_captions(job_id, clip_index, video_path):
    """Re-burn the default captions onto a freshly derived file.

    Captions must always be the LAST layer. Editing or hooking a clip that
    already had them burned in produced `edited_subtitled_<...>`, and the next
    subtitle pass then stacked a second caption layer on top of the first —
    visibly doubled and unreadable in real user clips (26-jul-2026). So the
    derivation runs on the clean file and captions go back on afterwards.

    Returns the captioned path, or None if there was nothing to caption.
    """
    try:
        meta_files = glob.glob(os.path.join(OUTPUT_DIR, job_id, "*_metadata.json"))
        if not meta_files:
            return None
        with open(meta_files[0], 'r') as f:
            data = json.load(f)
        transcript = data.get('transcript')
        clips = data.get('shorts', [])
        if not transcript or clip_index >= len(clips):
            return None
        clip = clips[clip_index]
        import main as _main
        # The look this job was generated with (main.py records the job's
        # AUTO_CAPTION_PRESET in the metadata); the server process itself does
        # not carry the job's env.
        preset = data.get('caption_preset')
        # A recut clip is a concatenation of source segments, so the flat
        # start..end window is wrong for it — caption against the clip-relative
        # remapped transcript instead (same trick /api/subtitle uses).
        recipe_segments = (clip.get('recipe') or {}).get('segments')
        if recipe_segments:
            v_transcript = recut.virtual_transcript(transcript, recipe_segments)
            return _main.auto_caption_clip(
                video_path, v_transcript, 0.0,
                recut.total_duration(recipe_segments), preset=preset)
        return _main.auto_caption_clip(video_path, transcript,
                                       clip['start'], clip['end'], preset=preset)
    except Exception as e:
        print(f"⚠️  Could not re-apply captions to {video_path}: {e}")
        return None


def _recover_jobs_from_disk():
    """Rebuild completed jobs from OUTPUT_DIR after a restart (issue #46 / #18).

    Jobs live in memory, so a restart used to orphan finished clips that are
    still on disk: the frontend restores the job_id from localStorage but every
    endpoint answers 404 "Job not found". Rebuild a minimal completed record
    for each job directory that has a metadata JSON.
    """
    recovered = 0
    try:
        entries = os.listdir(OUTPUT_DIR)
    except FileNotFoundError:
        return
    for job_id in entries:
        job_path = os.path.join(OUTPUT_DIR, job_id)
        if not os.path.isdir(job_path) or job_id in jobs:
            continue
        json_files = glob.glob(os.path.join(job_path, "*_metadata.json"))
        if not json_files:
            continue
        try:
            with open(json_files[0], 'r') as f:
                data = json.load(f)
            base_name = os.path.basename(json_files[0]).replace('_metadata.json', '')
            clips = data.get('shorts', [])
            for i, clip in enumerate(clips):
                if not clip.get('video_url'):
                    clip['video_url'] = (
                        f"/videos/{job_id}/"
                        f"{_canonical_clip_file(job_path, base_name, i)}")
            owner = None
            owner_path = os.path.join(job_path, ".owner")
            if os.path.exists(owner_path):
                with open(owner_path) as f:
                    raw = f.read().strip()
                owner = int(raw) if raw.isdigit() else (raw or None)
            jobs[job_id] = {
                'status': 'completed',
                'logs': ["♻️ Job recovered from disk after server restart."],
                'output_dir': job_path,
                'user_id': owner,
                'result': {'clips': clips, 'cost_analysis': data.get('cost_analysis')},
            }
            recovered += 1
        except Exception as e:
            print(f"⚠️ Could not recover job {job_id}: {e}")
    if recovered:
        print(f"♻️  Recovered {recovered} completed job(s) from disk.")


# --- Mid-flight job resume (survive a redeploy without losing work) ----------
# A job lives only in memory, so killing the container mid-processing used to
# lose it: the user's clip just stops. We persist a tiny manifest per job and,
# on startup, re-enqueue any that were interrupted — the user sees it resume
# instead of vanish. Bounded by MAX_RESUME_ATTEMPTS so a video that reliably
# crashes the worker can't crashloop the service.
_RESUME_FILE = ".resume.json"
MAX_RESUME_ATTEMPTS = 2

# --- Deploy handover (two instances, one disk) -------------------------------
# Coolify starts the NEW container before it stops the old one (rolling
# update), and both see the same OUTPUT_DIR. Without coordination the new one
# re-enqueued, at startup, the very jobs the old one was still rendering —
# and the old one had 30 s to live anyway. Now:
#   * every instance stamps OUTPUT_DIR/.instance with its id at startup; an
#     instance that sees another id there knows it is the OLD one and DRAINS:
#     it finishes what it is running, starts nothing new, and leaves queued
#     manifests for the new instance to pick up;
#   * a running job writes a heartbeat into its manifest every few seconds,
#     so the new instance skips manifests that are alive elsewhere and resumes
#     only the stale ones (an instance killed mid-job stops heartbeating);
#   * SIGTERM (docker stop) also drains, up to DRAIN_TIMEOUT_SECONDS — keep it
#     under the Coolify stop grace period — before letting uvicorn exit.
INSTANCE_ID = os.environ.get("INSTANCE_ID") or socket.gethostname()
_INSTANCE_MARKER = ".instance"
HEARTBEAT_EVERY = 10                 # seconds between manifest heartbeats
HEARTBEAT_STALE_AFTER = 60           # no heartbeat for this long = nobody has it
RESUME_SCAN_INTERVAL = 30            # seconds between looks for stale manifests
HANDOVER_CHECK_INTERVAL = 5          # seconds between looks at the marker
DRAIN_TIMEOUT_SECONDS = int(os.environ.get("DRAIN_TIMEOUT_SECONDS", "840"))
# After the jobs are drained, keep SERVING this long with /health/ready at 503
# before closing the socket: the proxy only drops a container once its Docker
# healthcheck has failed interval*retries times (15 s with the Coolify
# settings), and closing the socket earlier sends that many seconds of
# requests to a dead port. Measured 2026-08-25: ~60 s of alternating 502/200
# per deploy with retries=12 and no grace at all.
PROXY_DRAIN_SECONDS = float(os.environ.get("PROXY_DRAIN_SECONDS", "20"))
# Once uvicorn has the signal it closes within --timeout-graceful-shutdown
# (15 s), but the interpreter then waits for non-daemon threads, and a
# request cancelled mid-flight can leave an executor thread stuck in a
# network probe (yt-dlp) for as long as that takes. Seen 2026-08-25: "Finished
# server process" printed, container alive until the 900 s SIGKILL, deploy
# stuck in "Removing old containers". So the process is ended outright a
# little after uvicorn was told to stop. Jobs are already drained by then.
HARD_EXIT_SECONDS = float(os.environ.get("HARD_EXIT_SECONDS", "30"))
_draining = False
_stopping = False                    # SIGTERM received: report not-ready so the
                                     # proxy stops routing here before the
                                     # listening socket closes
_running_jobs: set = set()           # job ids with a live subprocess here


def _manifest_path(job_id):
    return os.path.join(OUTPUT_DIR, job_id, _RESUME_FILE)


def _read_manifest(job_id):
    try:
        with open(_manifest_path(job_id)) as f:
            return json.load(f)
    except FileNotFoundError:
        return None
    except Exception as e:
        print(f"⚠️ Bad resume manifest for {job_id}: {e}")
        return None


def _touch_manifest(job_id, now=None):
    """Stamp 'this instance is running it, as of now' into the manifest."""
    m = _read_manifest(job_id)
    if m is None:
        return
    m["heartbeat"] = now if now is not None else time.time()
    m["instance"] = INSTANCE_ID
    try:
        with open(_manifest_path(job_id), "w") as f:
            json.dump(m, f)
    except Exception as e:
        print(f"⚠️ Could not heartbeat manifest for {job_id}: {e}")


def _manifest_busy_elsewhere(m, now=None):
    """True when ANOTHER instance heartbeated this job recently. Our own id
    with a fresh heartbeat is a job WE were running before a restart of this
    same container (same hostname) — that one must be resumed, not skipped."""
    now = time.time() if now is None else now
    return (m.get("instance") not in (None, INSTANCE_ID)
            and now - float(m.get("heartbeat") or 0) < HEARTBEAT_STALE_AFTER)


def _write_instance_marker():
    try:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        with open(os.path.join(OUTPUT_DIR, _INSTANCE_MARKER), "w") as f:
            f.write(INSTANCE_ID)
    except Exception as e:
        print(f"⚠️ Could not write instance marker: {e}")


def _read_instance_marker():
    try:
        with open(os.path.join(OUTPUT_DIR, _INSTANCE_MARKER)) as f:
            return f.read().strip()
    except Exception:
        return None


def _begin_drain(reason):
    global _draining
    if not _draining:
        _draining = True
        print(f"⏸️ Draining ({reason}): finishing {len(_running_jobs)} running job(s), "
              f"starting none.")


def _check_instance_marker():
    """A different id in the marker means a newer instance is up: drain."""
    other = _read_instance_marker()
    if other and other != INSTANCE_ID:
        _begin_drain(f"newer instance {other} is up")
        return True
    return False


async def _handover_watch():
    while not _draining:
        await asyncio.sleep(HANDOVER_CHECK_INTERVAL)
        _check_instance_marker()


async def _resume_scan():
    """Keep looking for manifests nobody is running (the old instance left
    them queued, or was killed at the end of its grace period)."""
    while True:
        await asyncio.sleep(RESUME_SCAN_INTERVAL)
        if _draining:
            continue
        try:
            _resume_interrupted_jobs()
        except Exception as e:
            print(f"⚠️ Resume scan failed: {e}")


async def _drain_then_exit(previous_handler, timeout=None, proxy_grace=None,
                           hard_exit_after=None):
    """Wait for running jobs (bounded), let the proxy notice we are not ready,
    then hand the signal to uvicorn."""
    timeout = DRAIN_TIMEOUT_SECONDS if timeout is None else timeout
    proxy_grace = PROXY_DRAIN_SECONDS if proxy_grace is None else proxy_grace
    hard_exit_after = HARD_EXIT_SECONDS if hard_exit_after is None else hard_exit_after
    deadline = time.time() + timeout
    while _running_jobs and time.time() < deadline:
        await asyncio.sleep(1)
    if _running_jobs:
        print(f"⏱️ Drain timeout after {timeout}s with {len(_running_jobs)} job(s) still "
              f"running — they will resume on the next instance.")
    else:
        print("✅ Drained: no running jobs.")
    if proxy_grace > 0:
        print(f"⏳ Serving {proxy_grace:.0f}s more while the proxy drops this instance.")
        await asyncio.sleep(proxy_grace)
    print("👋 Shutting down.")
    if hard_exit_after > 0:
        import threading
        threading.Timer(hard_exit_after, _hard_exit).start()
    if callable(previous_handler):
        previous_handler(signal.SIGTERM, None)
    else:
        os._exit(0)


def _hard_exit():
    print(f"⛔ Still alive {HARD_EXIT_SECONDS:.0f}s after the stop signal (a thread "
          f"is hanging) — exiting now.", flush=True)
    os._exit(0)


def _install_drain_signal_handler():
    """Replace uvicorn's SIGTERM handler with drain-first. uvicorn's own
    handler closes the listening socket at once, which would make Traefik
    send half the traffic to a refused port for the whole drain."""
    previous = signal.getsignal(signal.SIGTERM)
    loop = asyncio.get_running_loop()

    def on_sigterm():
        global _stopping
        _stopping = True
        _begin_drain("SIGTERM")
        asyncio.ensure_future(_drain_then_exit(previous))

    try:
        loop.add_signal_handler(signal.SIGTERM, on_sigterm)
    except (NotImplementedError, RuntimeError, ValueError) as e:
        print(f"⚠️ Drain-on-SIGTERM unavailable ({e}); jobs will resume on restart instead.")


def _write_resume_manifest(job_id, cmd, priority, user_id, reservation_id, watermark,
                           webhook_url=None, webhook_secret=None, base_url=None):
    try:
        path = os.path.join(OUTPUT_DIR, job_id, _RESUME_FILE)
        with open(path, "w") as f:
            json.dump({
                "cmd": cmd, "priority": priority,
                "user_id": None if user_id is None else str(user_id),
                "reservation_id": reservation_id,
                "watermark": bool(watermark), "attempts": 0,
                # The caller's webhook must survive a redeploy: a pipeline that
                # relies on the callback would otherwise hang forever on a job
                # that resumed fine. The secret is the caller's own HMAC value,
                # stored next to their video on the same disk — not a server
                # credential (those are rebuilt from os.environ on resume).
                "webhook_url": webhook_url,
                "webhook_secret": webhook_secret,
                "base_url": base_url,
            }, f)
    except Exception as e:
        print(f"⚠️ Could not write resume manifest for {job_id}: {e}")


def _clear_resume_manifest(job_id):
    """Drop the manifest once a job reaches a terminal state, so it is never
    re-run on a later restart. Only an interrupted (still-running) job keeps it."""
    try:
        os.remove(os.path.join(OUTPUT_DIR, job_id, _RESUME_FILE))
    except FileNotFoundError:
        pass
    except Exception as e:
        print(f"⚠️ Could not clear resume manifest for {job_id}: {e}")


def _resume_interrupted_jobs() -> set:
    """Re-enqueue jobs that were mid-processing when the server last stopped.

    Runs after _recover_jobs_from_disk: a job whose clips already finished has a
    metadata JSON and is recovered as 'completed', so we only resume manifests
    with no metadata yet (analysis never finished). Also called periodically
    (_resume_scan): a manifest another instance is heartbeating is left alone
    until that heartbeat goes stale, and a job this instance already holds is
    never enqueued twice.

    Returns the set of reservation ids for every manifest still on disk —
    resumed here or alive on the other instance — so the caller can keep them
    out of the orphaned-reservation refund. Does NO DB work — the DB engine
    isn't up yet at this point in startup. A poison job (too many attempts) is
    simply not resumed; its reservation is then refunded as a normal orphan.
    """
    keep_reservations: set = set()
    try:
        entries = os.listdir(OUTPUT_DIR)
    except FileNotFoundError:
        return keep_reservations
    resumed = 0
    for job_id in entries:
        job_path = os.path.join(OUTPUT_DIR, job_id)
        manifest_path = os.path.join(job_path, _RESUME_FILE)
        if not os.path.isfile(manifest_path):
            continue
        if glob.glob(os.path.join(job_path, "*_metadata.json")):
            # Finished after all — recovered as completed already.
            _clear_resume_manifest(job_id)
            continue
        try:
            with open(manifest_path) as f:
                m = json.load(f)
        except Exception as e:
            print(f"⚠️ Bad resume manifest for {job_id}: {e}")
            continue

        if m.get("reservation_id"):
            keep_reservations.add(str(m["reservation_id"]))
        if job_id in jobs:
            continue  # already ours (queued, running or recovered)
        if _manifest_busy_elsewhere(m):
            continue  # the other instance is on it; we take over if it goes stale

        attempts = int(m.get("attempts", 0)) + 1
        user_id = m.get("user_id")
        reservation_id = m.get("reservation_id")
        if attempts > MAX_RESUME_ATTEMPTS:
            # Poison job: don't resume. Leaving its reservation out of the keep
            # set lets the orphan sweep refund it, and the user can retry by hand.
            print(f"🛑 Job {job_id} exceeded {MAX_RESUME_ATTEMPTS} resume attempts — giving up.")
            _clear_resume_manifest(job_id)
            if reservation_id:
                keep_reservations.discard(str(reservation_id))  # let the sweep refund it
            continue

        # Rebuild env from scratch — the manifest holds no secrets. Managed
        # (cloud) jobs get the server key; self-host falls back to its env key.
        env = os.environ.copy()
        try:
            from cloud import proxy_ledger as _pl
            if BILLING_ENABLED and _pl.budget_exceeded_sync():
                env.pop("PROXY_URL", None)  # daily paid-proxy budget hit
        except Exception:
            pass
        if BILLING_ENABLED and user_id is not None:
            try:
                env["GEMINI_API_KEY"] = managed_keys.gemini_key()
            except Exception:
                pass
        if m.get("watermark"):
            env["WATERMARK"] = "1"
        else:
            env.pop("WATERMARK", None)

        m["attempts"] = attempts
        try:
            with open(manifest_path, "w") as f:
                json.dump(m, f)
        except Exception:
            pass

        jobs[job_id] = {
            'status': 'queued',
            'logs': [f"♻️ Resuming your video after a server update (attempt {attempts})."],
            'cmd': m.get("cmd"),
            'env': env,
            'output_dir': job_path,
            'user_id': None if user_id is None else user_id,
            'reservation_id': reservation_id,
            'watermark': bool(m.get("watermark")),
            'webhook_url': m.get("webhook_url"),
            'webhook_secret': m.get("webhook_secret"),
            'base_url': m.get("base_url"),
        }
        _enqueue_job(job_id, int(m.get("priority", 2)))
        resumed += 1
    if resumed:
        print(f"♻️  Re-enqueued {resumed} interrupted job(s) after restart.")
    return keep_reservations


def _dir_size(path: str) -> int:
    total = 0
    for root, _dirs, files in os.walk(path):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    return total


def _enforce_uploads_size_cap():
    """Delete the oldest source uploads while UPLOAD_DIR is over UPLOADS_MAX_GB.

    Sources are only needed while a job runs (and for the preview afterwards),
    but they're the biggest files on disk — up to MAX_FILE_SIZE_MB each.
    """
    cap = UPLOADS_MAX_GB * 1024 ** 3
    if cap <= 0:
        return
    used = _dir_size(UPLOAD_DIR)
    if used <= cap:
        return
    files = []
    for name in os.listdir(UPLOAD_DIR):
        p = os.path.join(UPLOAD_DIR, name)
        if os.path.isfile(p):
            try:
                files.append((os.path.getmtime(p), p, os.path.getsize(p)))
            except OSError:
                pass
    files.sort()
    print(f"🧹 Uploads at {used / 1024**3:.1f} GB (cap {UPLOADS_MAX_GB} GB) — trimming.")
    for _mtime, path, size in files:
        if used <= cap:
            break
        try:
            os.remove(path)
            used -= size
            print(f"🧹 Size cap: removed upload {os.path.basename(path)}")
        except OSError:
            pass


def _enforce_output_size_cap():
    """Delete the oldest job dirs while OUTPUT_DIR is over OUTPUT_MAX_GB."""
    cap = OUTPUT_MAX_GB * 1024 ** 3
    if cap <= 0:
        return
    used = _dir_size(OUTPUT_DIR)
    if used <= cap:
        return
    thumbs = os.path.basename(THUMBNAILS_DIR)
    candidates = []
    for job_id in os.listdir(OUTPUT_DIR):
        if job_id == thumbs:
            continue
        p = os.path.join(OUTPUT_DIR, job_id)
        if os.path.isdir(p):
            try:
                candidates.append((os.path.getmtime(p), p, job_id))
            except OSError:
                pass
    candidates.sort()  # oldest first
    print(f"🧹 Output dir at {used / 1024**3:.1f} GB (cap {OUTPUT_MAX_GB} GB) — trimming.")
    for _mtime, path, job_id in candidates:
        if used <= cap:
            break
        size = _dir_size(path)
        shutil.rmtree(path, ignore_errors=True)
        jobs.pop(job_id, None)
        used -= size
        print(f"🧹 Size cap: purged {job_id} ({size / 1024**2:.0f} MB)")


def _sweep_retained_sources(now=None):
    """Delete retained downloads older than SOURCE_RETENTION_SECONDS.

    Only the ``source_video`` a URL job kept in its own directory: an upload is
    the user's own file and ages out with the job like everything else. Yields
    the job ids it emptied. A no-op unless the knob is set below the job clock.
    """
    if SOURCE_RETENTION_SECONDS >= JOB_RETENTION_SECONDS:
        return
    now = time.time() if now is None else now
    for job_id in os.listdir(OUTPUT_DIR):
        if job_id == os.path.basename(THUMBNAILS_DIR):
            continue
        try:
            metas = glob.glob(os.path.join(OUTPUT_DIR, job_id, "*_metadata.json"))
            if not metas:
                continue
            with open(metas[0]) as f:
                name = json.load(f).get('source_video')
            if not name:
                continue
            src = os.path.join(OUTPUT_DIR, job_id, os.path.basename(name))
            if (os.path.exists(src)
                    and now - os.path.getmtime(src) > SOURCE_RETENTION_SECONDS):
                os.remove(src)
                yield job_id
        except Exception:
            continue


async def cleanup_jobs():
    """Background task to remove old jobs and files."""
    import time
    print("🧹 Cleanup task started.")
    while True:
        try:
            await asyncio.sleep(300) # Check every 5 minutes
            now = time.time()
            
            # Simple directory cleanup based on modification time
            # Check OUTPUT_DIR
            for job_id in os.listdir(OUTPUT_DIR):
                # Not a job: the thumbnails dir backs a StaticFiles mount, so
                # deleting it would 500 every /thumbnails request until reboot.
                if job_id == os.path.basename(THUMBNAILS_DIR):
                    continue
                job_path = os.path.join(OUTPUT_DIR, job_id)
                if os.path.isdir(job_path):
                    if now - os.path.getmtime(job_path) > JOB_RETENTION_SECONDS:
                        print(f"🧹 Purging old job: {job_id}")
                        shutil.rmtree(job_path, ignore_errors=True)
                        if job_id in jobs:
                            del jobs[job_id]

            for job_id in _sweep_retained_sources(now):
                print(f"🧹 Dropped retained source for job {job_id}")

            # Hard disk cap. The time-based sweep above bounds the *age* of what
            # we keep, not its size: a burst of long videos can fill the volume
            # inside one retention window. Drop the oldest jobs until we're back
            # under the cap — clips are already archived to R2 and get restored
            # on demand, so this only costs a re-download.
            _enforce_output_size_cap()
            _enforce_uploads_size_cap()

            # Cleanup SaaSShorts jobs from memory
            try:
                saas_expired = [
                    jid for jid, jdata in list(saas_jobs.items())
                    if jdata.get("status") in ("completed", "failed")
                    and jdata.get("output_dir")
                    and os.path.isdir(jdata["output_dir"])
                    and now - os.path.getmtime(jdata["output_dir"]) > JOB_RETENTION_SECONDS
                ]
                for jid in saas_expired:
                    del saas_jobs[jid]
            except NameError:
                pass

            # Agent upload slots: expire with their file (the file sweep below
            # removes it; a slot whose file is gone or too old is dropped).
            for uid in _sweep_pending_uploads(now):
                print(f"🧹 Expired agent upload slot {uid}")

            # Cleanup Uploads
            for filename in os.listdir(UPLOAD_DIR):
                file_path = os.path.join(UPLOAD_DIR, filename)
                try:
                    if now - os.path.getmtime(file_path) > JOB_RETENTION_SECONDS:
                         os.remove(file_path)
                except Exception: pass

        except Exception as e:
            print(f"⚠️ Cleanup error: {e}")

async def process_queue():
    """Background worker to process jobs from the queue with concurrency limit."""
    print(f"🚀 Job Queue Worker started with {MAX_CONCURRENT_JOBS} concurrent slots.")
    while True:
        try:
            # Wait for a job (priority, seq, job_id) — lowest priority first.
            _priority, _seq, job_id = await job_queue.get()

            if _draining:
                # Leave it on disk for the next instance (manifest, no heartbeat).
                print(f"⏸️ Draining — leaving {job_id} for the next instance.")
                job_queue.task_done()
                continue

            # Acquire semaphore slot (waits if max jobs are running)
            await concurrency_semaphore.acquire()
            if _draining:
                concurrency_semaphore.release()
                job_queue.task_done()
                print(f"⏸️ Draining — leaving {job_id} for the next instance.")
                continue
            print(f"🔄 Acquired slot for job: {job_id}")
            _running_jobs.add(job_id)
            _touch_manifest(job_id)

            # Process in background task to not block the loop (allowing other slots to fill)
            asyncio.create_task(run_job_wrapper(job_id))
            
        except Exception as e:
            print(f"❌ Queue dispatch error: {e}")
            await asyncio.sleep(1)

# Monthly proxy bandwidth counter (in-memory; an alert threshold, not a bill —
# losing it on a deploy just means the alert re-arms from 0 mid-month).
_proxy_month = {"month": None, "bytes": 0, "alerted": False}
PROXY_ALERT_GB = 100


def _job_source_url(job) -> Optional[str]:
    """The ``-u`` argument of the job's main.py command, if it was a URL job."""
    cmd = list((job or {}).get('cmd') or [])
    # The command is ``python -u main.py -u <url>``: skip past the script
    # name first or the interpreter's own ``-u`` flag is what gets matched.
    start = next((i + 1 for i, a in enumerate(cmd) if str(a).endswith('main.py')), 0)
    for i in range(start, len(cmd) - 1):
        if cmd[i] in ('-u', '--url'):
            return cmd[i + 1]
    return None


async def _track_proxy_usage(job_id):
    job = jobs.get(job_id) or {}
    # Durable trail + Telegram page whenever the per-GB proxy carried bytes
    # (cloud mode only: self-host has no DB and pays nobody per GB).
    if BILLING_ENABLED and job.get('proxy_route'):
        try:
            from cloud import proxy_ledger as _pl
            await _pl.record_download(job_id, job.get('proxy_route'), _job_source_url(job))
        except Exception as e:
            print(f"⚠️ proxy ledger failed for {job_id}: {e}")
    nbytes = job.get('proxy_bytes') or 0
    if not nbytes:
        return
    month = datetime.now(timezone.utc).strftime("%Y-%m")
    if _proxy_month["month"] != month:
        _proxy_month.update(month=month, bytes=0, alerted=False)
    _proxy_month["bytes"] += nbytes
    gb = _proxy_month["bytes"] / 1e9
    if gb >= PROXY_ALERT_GB and not _proxy_month["alerted"] and _alerts:
        _proxy_month["alerted"] = True
        try:
            await _alerts.send_admin_alert(
                "Proxy bandwidth threshold",
                f"Managed downloads have used {gb:.1f} GB of proxy bandwidth in {month} "
                f"(threshold {PROXY_ALERT_GB} GB). Review free-plan usage.",
            )
        except Exception as e:
            print(f"⚠️ Proxy alert failed: {e}")


async def run_job_wrapper(job_id):
    """Wrapper to run job and release semaphore"""
    try:
        job = jobs.get(job_id)
        if job:
            await run_job(job_id, job)
    except Exception as e:
         print(f"❌ Job wrapper error {job_id}: {e}")
    finally:
        # The subprocess returned (success or genuine failure) — a terminal
        # state, so drop the resume manifest. It only survives if the container
        # was killed mid-run, which is exactly when we want to resume.
        _clear_resume_manifest(job_id)
        # Settle the minute reservation (managed jobs only): commit on success,
        # release otherwise so the minutes go back to the user.
        await _settle_reservation(job_id)
        # Archive the completed clips to the user's durable R2 library (history).
        await _archive_managed_job(job_id)
        # Fire the caller's webhook (after archive, so durable links exist).
        await _notify_job_webhook(job_id)
        # Operational alerting for managed jobs (proxy out of credits / failures).
        await _record_job_alert(job_id)
        # Accumulate proxy bandwidth for the monthly cost alert.
        await _track_proxy_usage(job_id)
        # Tell the owner their clips are ready (managed jobs, once per job).
        await _notify_clips_ready(job_id)
        # Telegram pulse for high-signal activity (first clip / paid user).
        await _notify_clip_activity(job_id)
        # Always release semaphore and mark queue task done
        _running_jobs.discard(job_id)
        concurrency_semaphore.release()
        job_queue.task_done()
        print(f"✅ Released slot for job: {job_id}")


async def _archive_managed_job(job_id):
    if not BILLING_ENABLED:
        return
    job = jobs.get(job_id) or {}
    if not job.get('user_id') or job.get('status') != 'completed':
        return
    clips = (job.get('result') or {}).get('clips') or []
    if not clips:
        return
    try:
        await cloud.videos.archive_job(job['user_id'], job_id, clips, job['output_dir'])
    except Exception as e:
        print(f"⚠️  R2 archive error for {job_id}: {e}")


def _archive_clip_edit_bg(job_id: str, clip_index: int, filename: str):
    """Fire-and-forget R2 re-archive of an edited clip (managed jobs only).

    Keeps the user's durable library (history/projects) pointing at the current
    version of each clip without blocking the edit response."""
    if not BILLING_ENABLED:
        return
    user_id = (jobs.get(job_id) or {}).get('user_id')
    if not user_id:
        return
    output_dir = os.path.join(OUTPUT_DIR, job_id)

    async def _run():
        try:
            await cloud.videos.archive_clip_edit(user_id, job_id, clip_index, output_dir, filename)
        except Exception as e:
            print(f"⚠️  R2 edit archive error for {job_id}: {e}")

    asyncio.create_task(_run())


async def _notify_clips_ready(job_id):
    """Email the owner when their clips finish — processing takes minutes, so
    this lets them close the tab. Once per job (email_sent flag)."""
    if not BILLING_ENABLED:
        return
    job = jobs.get(job_id) or {}
    if not job.get('user_id') or job.get('status') != 'completed' or job.get('email_sent'):
        return
    clips = (job.get('result') or {}).get('clips') or []
    if not clips:
        return
    job['email_sent'] = True
    try:
        from cloud.database import session as cloud_session
        from cloud.models import User
        from cloud.emails import send_clips_ready_email
        async with cloud_session() as s:
            user = await s.get(User, job['user_id'])
        if not user or not user.email:
            return
        title = clips[0].get('video_title_for_youtube_short') or clips[0].get('title') or "Your video"
        # #app opens the app itself. The bare frontend URL showed the marketing
        # landing to anyone whose browser had not already set the skip flag,
        # which is the wrong page for someone clicking "View my clips".
        await send_clips_ready_email(user.email, title, len(clips),
                                     f"{_cloud_config.settings.frontend_url}/#app")
    except Exception as e:
        print(f"⚠️  Clips-ready email error for {job_id}: {e}")


async def _notify_clip_activity(job_id):
    """Telegram pulse when a PAID user's clips are created. Free-tier activity
    (including first clips) is deliberately silent: at current signup volume it
    drowned the ops channel without being actionable.
    Telegram-only (best effort, no email)."""
    if not BILLING_ENABLED:
        return
    job = jobs.get(job_id) or {}
    if not job.get('user_id') or job.get('status') != 'completed':
        return
    clips = (job.get('result') or {}).get('clips') or []
    if not clips:
        return
    try:
        from cloud.database import session as cloud_session
        from cloud.models import User
        from cloud import metering
        async with cloud_session() as s:
            user = await s.get(User, job['user_id'])
            if not user:
                return
            sub = await metering._active_subscription(s, user.id)
        if sub is None:
            return
        title = clips[0].get('video_title_for_youtube_short') or clips[0].get('title') or "video"
        n = len(clips)
        await _alerts.send_telegram(
            f"🎬 Clips created\n{user.email} ({sub.plan}) — “{title}” ({n} clip{'s' if n != 1 else ''})")
    except Exception as e:
        print(f"⚠️  Clip-activity notify error for {job_id}: {e}")


# Markers that identify a line as an actual error rather than progress noise.
# "Error:" (capital E) catches raised exception lines — RuntimeError:,
# DownloadError:, GeminiBlockedError: — which "ERROR:" alone missed, leaving
# alerts with a bare "Traceback ... exit code 1" and no cause (prod 20-ago).
_ERROR_MARKERS = ("❌", "ERROR:", "Error:", "Traceback", "FATAL", "Exception",
                  "Process failed with exit code", "No metadata file generated",
                  "Execution error:")


def _job_error_text(logs) -> str:
    """The lines that explain WHY a job failed, for the alert's classifier.

    The tail of the log is usually progress noise (scene detection, ffmpeg
    banners), which made alerts blame whatever word happened to be nearby —
    a silent upload got reported as a broken download path, and a Gemini blip
    as an ffmpeg problem. Pick the error-bearing lines instead, newest last.
    """
    # Per-attempt download warnings are only noise when a later attempt
    # RECOVERED: HD-direct fails on every job (banned server IP) and a static
    # proxy takes over, yet its "Video unavailable" made whole alerts read as
    # download outages when the job actually died in Gemini. When no attempt
    # succeeded, those lines carry the real cause and must stay.
    recovered = any("Download succeeded" in ln for ln in logs)
    hits = [ln for ln in logs
            if any(m in ln for m in _ERROR_MARKERS)
            and not (recovered and "Download attempt" in ln)]
    if not hits:
        return " ".join(logs[-10:])  # nothing recognisable — fall back to the tail
    return " ".join(hits[-6:])


async def _record_job_alert(job_id):
    if not BILLING_ENABLED:
        return
    job = jobs.get(job_id) or {}
    if not job.get('user_id'):
        return  # only track managed jobs
    ok = job.get('status') == 'completed'
    err = "" if ok else _job_error_text(job.get('logs', []))
    try:
        await _alerts.record_job_outcome(ok, err)
    except Exception as e:
        print(f"⚠️  Alert recording error for {job_id}: {e}")
    await _track_job_outcome(job, ok, err)


async def _track_job_outcome(job, ok, err):
    """Report the job to OpenPanel, with the user's job index.

    The index is what makes the retention question answerable: on 26-jul-2026,
    491 of 564 users who ever processed a video did it exactly once. Counting
    distinct users at index 1 versus index >= 2 measures whether the clip
    quality work moved that, which nothing in the stack could do before.

    Client-side analytics cannot cover this: a render finishes minutes later,
    often after the tab is closed, and ad-blockers eat a share of the rest.
    """
    try:
        from cloud import analytics as _an
        from sqlalchemy import text as _sa_text
        from cloud import database as _db
        user_id = job.get('user_id')
        job_index = None
        try:
            async with _db.session() as s:
                job_index = (await s.execute(_sa_text(
                    "select count(*) from usage_ledger "
                    "where user_id = :uid and job_type = 'process'"),
                    {"uid": user_id})).scalar()
        except Exception:
            pass  # an index we cannot read is not worth failing a job over
        clips = len(((job.get('result') or {}).get('clips')) or [])
        _an.track(
            "ClipsDelivered" if ok else "JobFailed",
            user_id=user_id,
            job_index=job_index,
            clips=clips if ok else None,
            plan=job.get('user_plan'),
            source="url" if job.get('url') else "upload",
            reason=(_alerts._classify_failure(err) if not ok and err else None),
        )
    except Exception as e:
        print(f"⚠️  Analytics error: {e}")


# --- Job completion webhooks --------------------------------------------------
# Agents and pipelines (n8n, cron, MCP clients) need push, not poll: a caller
# passes webhook_url on /api/process and gets one POST when the job reaches a
# terminal state. The URL goes through assert_public_url both at submit and at
# delivery time — the second check is what defeats DNS rebinding between them.
WEBHOOK_TIMEOUT = 10.0
WEBHOOK_RETRY_DELAYS = (0, 10, 60)  # seconds before each attempt


def _sign_webhook(body: bytes, secret: str) -> str:
    import hmac as _hmac
    import hashlib as _hashlib
    return "sha256=" + _hmac.new(secret.encode(), body, _hashlib.sha256).hexdigest()


async def _webhook_clip_entries(job_id, job):
    """The payload's clip list: absolute URLs, plus durable R2 links when the
    job was archived (a webhook consumer usually fetches later, after the
    1-hour local retention would have expired the /videos path)."""
    base = (job.get('base_url') or os.environ.get("PUBLIC_API_URL", "")).rstrip("/")
    clips = (job.get('result') or {}).get('clips') or []
    entries = []
    for i, clip in enumerate(clips):
        rel = clip.get('video_url') or ""
        entries.append({
            "index": i,
            "title": clip.get('title') or clip.get('video_title_for_youtube_short'),
            "video_url": f"{base}{rel}" if rel.startswith("/") and base else rel,
        })
    if BILLING_ENABLED and job.get('user_id'):
        try:
            from sqlalchemy import select as _select
            from cloud.database import session as cloud_session
            from cloud.models import UserVideo
            from cloud import storage as _storage
            async with cloud_session() as s:
                vids = list((await s.execute(
                    _select(UserVideo).where(UserVideo.job_id == job_id)
                )).scalars())
            for v in vids:
                if v.clip_index is not None and v.clip_index < len(entries):
                    entries[v.clip_index]["download_url"] = _storage.presigned_get(
                        v.r2_key, expires=24 * 3600)
        except Exception as e:
            print(f"⚠️ Webhook R2 links failed for {job_id}: {e}")
    return entries


async def _deliver_webhook(url, body: bytes, secret):
    headers = {"Content-Type": "application/json", "User-Agent": "OpenShorts-Webhook/1.0"}
    if secret:
        headers["X-OpenShorts-Signature"] = _sign_webhook(body, secret)
    from security_utils import assert_public_url, UnsafeURLError
    loop = asyncio.get_event_loop()
    for attempt, delay in enumerate(WEBHOOK_RETRY_DELAYS, 1):
        if delay:
            await asyncio.sleep(delay)
        try:
            # Re-resolve on every attempt: the submit-time check is stale by now.
            await loop.run_in_executor(None, assert_public_url, url)
            async with httpx.AsyncClient(timeout=WEBHOOK_TIMEOUT,
                                         follow_redirects=False) as client:
                resp = await client.post(url, content=body, headers=headers)
            if resp.status_code < 300:
                print(f"🪝 Webhook delivered to {url} (attempt {attempt})")
                return
            print(f"⚠️ Webhook attempt {attempt} to {url}: HTTP {resp.status_code}")
        except UnsafeURLError as e:
            print(f"🛑 Webhook URL no longer safe, dropping: {e}")
            return
        except Exception as e:
            print(f"⚠️ Webhook attempt {attempt} to {url} failed: {e}")
    print(f"❌ Webhook to {url} gave up after {len(WEBHOOK_RETRY_DELAYS)} attempts.")


async def _notify_job_webhook(job_id):
    """Fire the caller's webhook for a terminal job. Runs inside run_job_wrapper's
    finally AFTER the R2 archive, so durable links exist; the actual delivery
    (with its retry sleeps) is detached so the worker slot frees immediately."""
    job = jobs.get(job_id) or {}
    url = job.get('webhook_url')
    if not url or job.get('webhook_sent'):
        return
    job['webhook_sent'] = True
    completed = job.get('status') == 'completed'
    payload = {
        "event": "job.completed" if completed else "job.failed",
        "job_id": job_id,
        "status": job.get('status'),
        "clips": (await _webhook_clip_entries(job_id, job)) if completed else [],
    }
    if not completed:
        payload["error"] = _job_error_text(job.get('logs', []))[-500:]
    body = json.dumps(payload).encode()
    asyncio.create_task(_deliver_webhook(url, body, job.get('webhook_secret')))


async def _settle_reservation(job_id):
    if not BILLING_ENABLED:
        return
    job = jobs.get(job_id) or {}
    reservation_id = job.get('reservation_id')
    if not reservation_id:
        return
    try:
        if job.get('status') == 'completed':
            await cloud.metering.commit_reservation(reservation_id)
        else:
            await cloud.metering.release_reservation(reservation_id)
    except Exception as e:
        print(f"⚠️  Reservation settle error for {job_id}: {e}")

def _owned_by(record, uid: str) -> bool:
    owner = record.get('user_id') if isinstance(record, dict) else None
    return owner is not None and str(owner) == uid


def _rm_under(base_dir: str, relative: str):
    """Remove a file or directory, refusing anything that escapes ``base_dir``."""
    target = _safe_under(base_dir, relative)
    if not target or target == os.path.realpath(base_dir):
        return
    if os.path.isdir(target):
        shutil.rmtree(target, ignore_errors=True)
    else:
        try:
            os.remove(target)
        except OSError:
            pass


def _purge_local_jobs_for_user(user_id) -> int:
    """Delete this user's working files and in-memory records from local disk.

    Called by cloud/account.py when an account is erased. The durable copies
    live on R2 and are deleted there; these are the working files on the API's
    own disk, which would otherwise sit around until the one-hour cleanup sweep
    — and thumbnails not even then, because that sweep skips their directory.

    Three stores, because each records ownership differently:
      - clip jobs: the ``.owner`` file every managed job writes, so jobs
        recovered from disk after a restart (no in-memory record) count too;
      - SaaSShorts jobs (``output/saas_<id>``): ``saas_jobs`` only, no marker
        file, so a restart loses the link and those age out on the sweep;
      - thumbnail sessions (``output/thumbnails/<id>`` plus the source video in
        ``uploads/``): likewise in-memory only.

    Blocking: rmtree over gigabytes of video. Callers must run it in a thread.
    """
    uid = str(user_id)
    removed = 0

    job_ids = {jid for jid, job in list(jobs.items()) if _owned_by(job, uid)}
    thumbs_dir_name = os.path.basename(THUMBNAILS_DIR)
    try:
        entries = os.listdir(OUTPUT_DIR)
    except OSError:
        entries = []
    for job_id in entries:
        # Never a job, and it backs a StaticFiles mount: deleting the directory
        # itself 500s every /thumbnails request until the process restarts.
        if job_id == thumbs_dir_name:
            continue
        try:
            with open(os.path.join(OUTPUT_DIR, job_id, ".owner")) as f:
                if f.read().strip() == uid:
                    job_ids.add(job_id)
        except OSError:
            continue

    for job_id in job_ids:
        _rm_under(OUTPUT_DIR, job_id)
        jobs.pop(job_id, None)
        # Source uploads are named "<job_id>_<filename>" (see /api/process).
        for path in glob.glob(os.path.join(UPLOAD_DIR, f"{glob.escape(job_id)}_*")):
            try:
                os.remove(path)
            except OSError:
                pass
        removed += 1

    for jid, job in list(saas_jobs.items()):
        if not _owned_by(job, uid):
            continue
        out = job.get('output_dir')
        if out:
            _rm_under(OUTPUT_DIR, os.path.basename(out))
        saas_jobs.pop(jid, None)
        removed += 1

    for sid, sess in list(thumbnail_sessions.items()):
        if not _owned_by(sess, uid):
            continue
        # Generated thumbnails are served publicly at /thumbnails/<id>/... and
        # nothing else ever deletes them.
        _rm_under(THUMBNAILS_DIR, sid)
        video_path = sess.get('video_path')
        if video_path:
            _rm_under(UPLOAD_DIR, os.path.basename(video_path))
        thumbnail_sessions.pop(sid, None)
        removed += 1

    if removed:
        print(f"🗑️  Purged {removed} local work item(s) for erased user {uid}.")
    return removed


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Rehydrate finished jobs from disk before serving (survives restarts).
    _recover_jobs_from_disk()
    # Re-enqueue jobs that were mid-processing when we stopped (redeploy). Their
    # reservations must survive the orphan sweep so the resumed run can settle them.
    _resumed_reservation_ids = _resume_interrupted_jobs()
    # Deploy handover: claim the marker (any older instance sees it and drains),
    # keep watching it in case a newer one appears, keep looking for manifests
    # left behind, and drain instead of dying on docker stop.
    _write_instance_marker()
    _install_drain_signal_handler()
    asyncio.create_task(_handover_watch())
    asyncio.create_task(_resume_scan())
    # Start worker and cleanup
    worker_task = asyncio.create_task(process_queue())
    cleanup_task = asyncio.create_task(cleanup_jobs())
    if BILLING_ENABLED:
        await cloud.setup_async(app, keep_reservation_ids=_resumed_reservation_ids)
        # Account erasure lives in cloud/, which can't import app.py; hand it the
        # one thing only this module can do — wipe the local working files.
        cloud.account.register_local_purge(_purge_local_jobs_for_user)
        # Nag on Telegram while the residential proxy is down/out of credits —
        # a single job-failure alert is easy to miss and ingest stays broken
        # until someone tops the balance up.
        asyncio.create_task(_alerts.proxy_watch_loop())
    yield
    # Cleanup (optional: cancel worker)

app = FastAPI(lifespan=lifespan)

# Cloud mode: attach middleware + routers at import time (before the app serves).
if BILLING_ENABLED:
    cloud.setup_sync(app)

# MCP server (/mcp): the pipeline as agent-callable tools. Works in both modes —
# cloud requires an osk_ API key, self-host keeps BYOK (see mcp_server.py).
import mcp_server as _mcp_server
app.include_router(_mcp_server.router)

# Enable CORS for frontend. Cloud mode locks this down to the configured origins;
# self-host keeps the permissive wildcard it has always used.
app.add_middleware(
    CORSMiddleware,
    allow_origins=cloud.settings.allowed_origins if BILLING_ENABLED else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    # Without this the browser hides them from fetch(), so the clip download had
    # no total to measure against and could not show progress. Safelisted or not,
    # they only become readable to JS once they are named here.
    expose_headers=["Content-Length", "Content-Range", "Accept-Ranges"],
)

# Mount static files for serving videos. A miss under /videos/<job_id>/ first
# tries to bring the job back from R2 (see restoring_static.py): the players
# of a reopened project used to 404 while the transcript requests were still
# restoring it. Late-bound lambda: the restorer is defined further down.
from restoring_static import RestoringStaticFiles
app.mount("/videos", RestoringStaticFiles(
    directory=OUTPUT_DIR,
    restorer=lambda job_id: _restore_for_public_path(job_id)), name="videos")

# Mount static files for serving thumbnails
THUMBNAILS_DIR = os.path.join(OUTPUT_DIR, "thumbnails")
os.makedirs(THUMBNAILS_DIR, exist_ok=True)
app.mount("/thumbnails", StaticFiles(directory=THUMBNAILS_DIR), name="thumbnails")


def _safe_under(base_dir: str, user_rel_path: str) -> Optional[str]:
    """Resolve ``user_rel_path`` under ``base_dir`` and reject path traversal.

    Returns the absolute path only if it stays inside ``base_dir`` (after
    following ``..``); otherwise None. Used to sanitize client-supplied file
    references so ``../../.env`` can't escape the output directories.
    """
    base = os.path.realpath(base_dir)
    target = os.path.realpath(os.path.join(base, user_rel_path))
    if target == base or target.startswith(base + os.sep):
        return target
    return None

class ProcessRequest(BaseModel):
    url: str

# Masks user:password credentials embedded in any URL (e.g. the residential
# proxy URL that yt-dlp echoes in its verbose debug output) before the line is
# ever printed to the server console or stored in the job log.
_CREDENTIAL_URL_RE = re.compile(r'(\w+://)[^:/@\s]+:[^@/\s]+@')


def _scrub_secrets(line: str) -> str:
    return _CREDENTIAL_URL_RE.sub(r'\1***:***@', line)


# Cloud users don't need (and shouldn't see) implementation details: the ingest
# plumbing (proxy / downloader / cookies) OR which AI model powers it, token
# usage and cost. These are dropped from the client view even when the line is
# emoji-prefixed. Never applied under DEBUG_LOGS (local dev sees everything).
_SENSITIVE_LOG_RE = re.compile(
    # Ingest plumbing
    r'proxy|yt[-_ ]?dlp|youtube-?dl|cookie|residential|po[_ ]?token'
    r'|player_client|extractor|\bdownload|descarg'
    # AI model / provider / cost / pipeline internals
    r'|gemini|openai|anthropic|\bflash\b|\bmodel\b|token|thinking'
    r'|\bcost\b|\$\s*[0-9]|scoring window|shortlist',
    re.IGNORECASE,
)


def _visible_logs(logs):
    """Logs to surface to the client.

    Self-host (BILLING off) shows the full pipeline output so people running
    their own instance can debug. Cloud shows a curated whitelist view
    (log_view.friendly_logs): plain progress for normal users — transcription
    percentage, clip counters — with no file paths, model names or pipeline
    internals.

    DEBUG_LOGS=true forces the full output even under billing — for local dev
    where you run in paid mode but still want the raw logs.
    """
    if not BILLING_ENABLED or DEBUG_LOGS:
        return logs
    from log_view import friendly_logs
    return friendly_logs(logs)


def enqueue_output(out, job_id):
    """Reads output from a subprocess and appends it to jobs logs."""
    try:
        for line in iter(out.readline, b''):
            decoded_line = _scrub_secrets(line.decode('utf-8').strip())
            if decoded_line:
                # Internal marker from main.py's downloader, not a log line.
                # Internal marker: a clip finished its whole chain and this is
                # the file to serve for it. Consumed here like PROXY_BYTES so it
                # never reaches the user's log.
                if decoded_line.startswith("CLIP_READY "):
                    try:
                        _, index, filename = decoded_line.split(" ", 2)
                        if job_id in jobs:
                            jobs[job_id].setdefault('ready_files', {})[int(index)] = filename
                    except ValueError:
                        pass
                    continue
                if decoded_line.startswith("PROXY_BYTES="):
                    try:
                        if job_id in jobs:
                            jobs[job_id]['proxy_bytes'] = int(decoded_line.split("=", 1)[1])
                    except ValueError:
                        pass
                    continue
                if decoded_line.startswith("PROXY_ROUTE="):
                    # Which download attempt won and why the free ones failed;
                    # persisted at job end (cloud/proxy_ledger). Not shown to clients.
                    try:
                        from cloud import proxy_ledger as _pl
                        route = _pl.parse_route_line(decoded_line)
                        if route is not None and job_id in jobs:
                            jobs[job_id]['proxy_route'] = route
                    except Exception:
                        pass
                    continue
                print(f"📝 [Job Output] {decoded_line}")
                if job_id in jobs:
                    jobs[job_id]['logs'].append(decoded_line)
    except Exception as e:
        print(f"Error reading output for job {job_id}: {e}")
    finally:
        out.close()

async def run_job(job_id, job_data):
    """Executes the subprocess for a specific job."""
    
    cmd = job_data['cmd']
    env = job_data['env']
    output_dir = job_data['output_dir']
    
    jobs[job_id]['status'] = 'processing'
    jobs[job_id]['logs'].append("Job started by worker.")
    print(f"🎬 [run_job] Executing command for {job_id}: {' '.join(cmd)}")
    
    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, # Merge stderr to stdout
            env=env,
            cwd=os.getcwd()
        )
        
        # We need to capture logs in a thread because Popen isn't async
        t_log = threading.Thread(target=enqueue_output, args=(process.stdout, job_id))
        t_log.daemon = True
        t_log.start()
        
        # Async wait for process with incremental updates
        start_wait = time.time()
        last_heartbeat = time.time()
        while process.poll() is None:
            await asyncio.sleep(2)
            if time.time() - last_heartbeat >= HEARTBEAT_EVERY:
                _touch_manifest(job_id)
                last_heartbeat = time.time()
            
            # Check for partial results every 2 seconds
            # Look for metadata file
            try:
                json_files = glob.glob(os.path.join(output_dir, "*_metadata.json"))
                if json_files:
                    target_json = json_files[0]
                    # Read metadata (it might be being written to, so simple try/except or just read)
                    # Use a lock or just robust read? json.load might fail if file is partial.
                    # Usually main.py writes it once at start (based on my review).
                    if os.path.getsize(target_json) > 0:
                        with open(target_json, 'r') as f:
                            data = json.load(f)
                            
                        base_name = os.path.basename(target_json).replace('_metadata.json', '')
                        clips = data.get('shorts', [])
                        cost_analysis = data.get('cost_analysis')
                        
                        # Check which clips actually exist on disk
                        # Only clips main.py has announced as finished. It names
                        # the file itself, so a clip shows up WITH its hook and
                        # captions instead of as the bare reframe, and it is never
                        # served while ffmpeg is still writing it. A clip whose
                        # marker never arrives (it failed) simply stays hidden
                        # until the job ends and the result is rebuilt from disk.
                        ready_files = (jobs.get(job_id) or {}).get('ready_files') or {}
                        ready_clips = []
                        for i, clip in enumerate(clips):
                             clip_filename = ready_files.get(i)
                             if not clip_filename:
                                 continue
                             clip_path = os.path.join(output_dir, clip_filename)
                             if os.path.exists(clip_path) and os.path.getsize(clip_path) > 0:
                                 clip['video_url'] = f"/videos/{job_id}/{clip_filename}"
                                 ready_clips.append(clip)
                        
                        if ready_clips:
                             jobs[job_id]['result'] = {'clips': ready_clips, 'cost_analysis': cost_analysis}
            except Exception as e:
                # Ignore read errors during processing
                pass

        returncode = process.returncode
        
        if returncode == 0:
            jobs[job_id]['status'] = 'completed'
            jobs[job_id]['logs'].append("Process finished successfully.")
            
            # Self-host: silent AWS S3 backup. Cloud mode stores to R2 instead
            # (see _archive_managed_job), so skip the redundant/paid AWS upload.
            if not BILLING_ENABLED:
                loop = asyncio.get_event_loop()
                loop.run_in_executor(None, upload_job_artifacts, output_dir, job_id)
            
            # Find result JSON
            json_files = glob.glob(os.path.join(output_dir, "*_metadata.json"))
            if not json_files:
                # Backward-compat rescue if outputs were written to OUTPUT_DIR root
                if _relocate_root_job_artifacts(job_id, output_dir):
                    json_files = glob.glob(os.path.join(output_dir, "*_metadata.json"))
            if json_files:
                target_json = json_files[0] 
                with open(target_json, 'r') as f:
                    data = json.load(f)
                
                # Enhance result with video URLs
                base_name = os.path.basename(target_json).replace('_metadata.json', '')
                clips = data.get('shorts', [])
                cost_analysis = data.get('cost_analysis')

                for i, clip in enumerate(clips):
                     clip_filename = _canonical_clip_file(output_dir, base_name, i)
                     clip['video_url'] = f"/videos/{job_id}/{clip_filename}"
                
                jobs[job_id]['result'] = {'clips': clips, 'cost_analysis': cost_analysis}
            else:
                 jobs[job_id]['status'] = 'failed'
                 jobs[job_id]['logs'].append("No metadata file generated.")
        else:
            jobs[job_id]['status'] = 'failed'
            jobs[job_id]['logs'].append(_scrub_secrets(f"Process failed with exit code {returncode}"))
            
    except Exception as e:
        jobs[job_id]['status'] = 'failed'
        # Exception text can embed URLs with credentials (e.g. the proxy URL
        # inside a yt-dlp/httpx error) — scrub before it reaches client logs.
        jobs[job_id]['logs'].append(_scrub_secrets(f"Execution error: {str(e)}"))

@app.get("/health")
async def health():
    """Lightweight liveness probe for uptime monitoring."""
    return {"status": "ok"}


@app.get("/health/ready")
async def health_ready():
    """Readiness probe for the Docker HEALTHCHECK (Dockerfile). Traefik's docker
    provider drops a container from the load balancer as soon as it turns
    unhealthy, so answering 503 from the moment SIGTERM arrives pulls this
    instance out of rotation while it can still serve, instead of after its
    socket is gone. Only SIGTERM flips it: a drain triggered by the instance
    marker starts while the new container is still booting, and going
    unready then would leave nobody routable."""
    if _stopping:
        return JSONResponse({"status": "stopping"}, status_code=503)
    return {"status": "ready"}

@app.get("/api/config")
async def get_config():
    return {
        "youtubeUrlEnabled": not DISABLE_YOUTUBE_URL,
        "billingEnabled": BILLING_ENABLED,
        "googleAuthEnabled": bool(BILLING_ENABLED and cloud.settings.google_auth_enabled),
        "jobRetentionSeconds": JOB_RETENTION_SECONDS,
        # Self-host only: tells the dashboard the Gemini key is optional
        # because the moment picker runs on an OpenAI-compatible server.
        "localLlm": None if BILLING_ENABLED else llm_backend.describe(),
        # The caption looks the subtitle modal and the advanced options offer,
        # in display order. Defined once, server-side (subtitles.py).
        "captionPresets": caption_presets_for_api(),
    }

async def _probe_youtube_quality(url: str) -> dict:
    """Run quality_probe.py in a worker thread; {} on any failure (fail-open)."""
    def _run():
        try:
            proc = subprocess.run(
                [sys.executable, QUALITY_PROBE_SCRIPT, "--url", url],
                capture_output=True, timeout=75,
            )
            return json.loads(proc.stdout.decode(errors="replace").strip() or "{}")
        except Exception as e:
            print(f"⚠️ Quality probe failed ({e}); starting job without gate.")
            return {}

    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _run)


def _media_duration_seconds(path: str) -> float:
    """Container duration via ffprobe; 0.0 on any failure (fail-open)."""
    try:
        proc = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "csv=p=0", path],
            capture_output=True, timeout=30)
        return float(proc.stdout.decode().strip() or 0)
    except Exception:
        return 0.0


def _reject_short_source(duration: float):
    raise HTTPException(status_code=400, detail=(
        f"This video is only {int(duration)}s long — clip generation needs at "
        f"least {MIN_SOURCE_SECONDS}s of material to cut from. It already is "
        f"short-form content."))


# Layouts the caller can let the renderer choose from, mapped to the env var
# each one is gated on. The renderer only ever picks between layouts that are
# switched on here.
#
# This is opt-in per job, not a detector running on every video, because the
# detection is not good enough to be trusted unprompted: measured over the
# 48-clip corpus, routing every video through the on-screen-content check fixed
# 13 clips and spoiled 13 others (talking heads and corner tickers demoted to a
# layout they do not need). Asking the person who knows what they uploaded costs
# them one click and removes that whole class of error. It is also what OpusClip
# does — its "applicable auto layout" panel lets the user pick which layouts the
# AI may apply.
LAYOUT_ENV = {
    "split": "SPLIT_LAYOUT",          # two speakers stacked
    "screencast": "SCREENCAST_LAYOUT",  # slides/screen share over the speaker
    "speaker_cut": "SPEAKER_CUT",     # hard cuts to whoever is talking
    "punch_in": "PUNCH_IN",           # small push on the clip's beats
}

# Stacking and cutting both need to know who is speaking.
LAYOUT_IMPLIES = {
    "split": ["SPEAKER_SIGNAL"],
    "speaker_cut": ["SPEAKER_SIGNAL"],
}


# --------------------------------------------------------------------------- #
# Agent uploads: a two-step path for callers that hold a video FILE, not a URL
# (an MCP client handed the file by the user). POST reserves an id and returns
# a PUT URL; the client streams the raw bytes there with no auth beyond the
# unguessable id (so `curl -T` works from any agent runtime); /api/process then
# takes the upload_id. Files live in UPLOAD_DIR under the same retention sweep
# as every other source upload, and the owner recorded at POST is checked at
# process time so a leaked id cannot start a job on someone else's account.
# --------------------------------------------------------------------------- #
pending_uploads: Dict[str, Dict] = {}
# Unconsumed slots are gone after this; a consumed one becomes the job's
# input and follows the job's own retention instead.
UPLOAD_TTL_SECONDS = int(os.environ.get("UPLOAD_TTL_SECONDS", str(6 * 3600)))


def _upload_url_base(request):
    return os.environ.get("PUBLIC_API_URL", "").rstrip("/") or str(request.base_url).rstrip("/")


@app.post("/api/uploads")
async def create_upload(request: Request):
    """Reserve an upload slot. Body (JSON, optional): {"filename": "..."}."""
    user_id = await _owner_id(request)
    if BILLING_ENABLED and user_id is None:
        raise HTTPException(status_code=401, detail="Sign in or use an API key to upload")
    try:
        body = await request.json()
    except Exception:
        body = {}
    filename = os.path.basename(str((body or {}).get("filename") or "video.mp4")) or "video.mp4"
    upload_id = str(uuid.uuid4())
    pending_uploads[upload_id] = {
        "user_id": user_id,
        "filename": filename,
        "path": os.path.join(UPLOAD_DIR, f"pending_{upload_id}_{filename}"),
        "created": time.time(),
        "bytes": 0,
        "complete": False,
    }
    return {
        "upload_id": upload_id,
        "upload_url": f"{_upload_url_base(request)}/api/uploads/{upload_id}",
        "method": "PUT",
        "max_mb": MAX_FILE_SIZE_MB,
        "expires_in": UPLOAD_TTL_SECONDS,
        "hint": "PUT the raw video bytes to upload_url (e.g. curl -T video.mp4 <upload_url>), "
                "then call /api/process (or the process_video tool) with this upload_id. "
                "The slot and file are deleted after expires_in seconds if unused, or "
                "DELETE this URL to drop them sooner.",
    }


@app.put("/api/uploads/{upload_id}")
async def put_upload(upload_id: str, request: Request):
    """Receive the raw video body for a reserved slot. Streams to disk, capped
    at MAX_FILE_SIZE_MB; a second PUT replaces the first."""
    slot = pending_uploads.get(upload_id)
    if not slot or time.time() - slot["created"] > UPLOAD_TTL_SECONDS:
        pending_uploads.pop(upload_id, None)
        raise HTTPException(status_code=404, detail="Unknown or expired upload_id")
    limit_bytes = MAX_FILE_SIZE_MB * 1024 * 1024
    size = 0
    with open(slot["path"], "wb") as out:
        async for chunk in request.stream():
            size += len(chunk)
            if size > limit_bytes:
                out.close()
                os.remove(slot["path"])
                raise HTTPException(status_code=413, detail=f"File too large. Max size {MAX_FILE_SIZE_MB}MB")
            out.write(chunk)
    if size == 0:
        os.remove(slot["path"])
        raise HTTPException(status_code=400, detail="Empty body")
    slot.update({"bytes": size, "complete": True})
    duration = await asyncio.get_event_loop().run_in_executor(None, _media_duration_seconds, slot["path"])
    if duration <= 0:
        os.remove(slot["path"])
        slot["complete"] = False
        raise HTTPException(status_code=400, detail="The body is not a readable video file")
    return {"upload_id": upload_id, "bytes": size, "duration_seconds": round(duration, 1),
            "hint": "Now call /api/process with upload_id."}


@app.delete("/api/uploads/{upload_id}")
async def delete_upload(upload_id: str, request: Request):
    """Drop a slot and its file before it expires (owner only in cloud mode)."""
    slot = pending_uploads.get(upload_id)
    if not slot or (BILLING_ENABLED and slot.get("user_id") != await _owner_id(request)):
        raise HTTPException(status_code=404, detail="Unknown or expired upload_id")
    pending_uploads.pop(upload_id, None)
    try:
        os.remove(slot["path"])
    except OSError:
        pass
    return {"deleted": upload_id}


def _sweep_pending_uploads(now=None):
    """Expire agent upload slots older than UPLOAD_TTL_SECONDS (file included).
    Returns the ids removed. Called from the cleanup loop; pure enough to test."""
    now = now or time.time()
    gone = []
    for uid, slot in list(pending_uploads.items()):
        if now - slot["created"] > UPLOAD_TTL_SECONDS:
            pending_uploads.pop(uid, None)
            try:
                os.remove(slot["path"])
            except OSError:
                pass
            gone.append(uid)
    return gone


def _take_pending_upload(upload_id, user_id):
    """The completed upload for this caller, or an HTTPException."""
    slot = pending_uploads.get(upload_id)
    if not slot or (BILLING_ENABLED and slot.get("user_id") != user_id):
        raise HTTPException(status_code=404, detail="Unknown or expired upload_id")
    if not slot.get("complete") or not os.path.exists(slot["path"]):
        raise HTTPException(status_code=409, detail="Upload not received yet: PUT the video to upload_url first")
    return slot


def layout_env(requested):
    """Env overrides for the layouts this job allows. Unknown names are ignored
    rather than rejected: a newer dashboard must not break an older API.

    The special value "auto" hands the choice to Gemini (one call per video).
    It composes with explicit picks: layout_picker only ever adds, so asking for
    "auto,punch_in" means "decide the layout yourself, and punch in regardless".
    "none" is the opposite: it switches the picker OFF for this job even when
    the deployment runs with AUTO_LAYOUT=1, for the user who wants the plain
    single crop and nothing clever.
    """
    env = {}
    for name in requested or []:
        key = str(name).strip().lower()
        if key == "auto":
            env["AUTO_LAYOUT"] = "1"
            continue
        if key == "none":
            env["AUTO_LAYOUT"] = "0"
            continue
        var = LAYOUT_ENV.get(key)
        if not var:
            continue
        env[var] = "1"
        for extra in LAYOUT_IMPLIES.get(key, []):
            env[extra] = "1"
    return env


@app.post("/api/process")
async def process_endpoint(
    request: Request,
    file: Optional[UploadFile] = File(None),
    url: Optional[str] = Form(None),
    acknowledged: Optional[str] = Form(None),
    output_format: Optional[str] = Form(None),
    layouts: Optional[str] = Form(None),
    force_low_quality: Optional[str] = Form(None),
    webhook_url: Optional[str] = Form(None),
    webhook_secret: Optional[str] = Form(None),
    target_clips: Optional[str] = Form(None),
    clip_min_seconds: Optional[str] = Form(None),
    clip_max_seconds: Optional[str] = Form(None),
    auto_hook: Optional[str] = Form(None),
    auto_hook_style: Optional[str] = Form(None),
    thumbnail_session_id: Optional[str] = Form(None),
    captions: Optional[str] = Form(None),
    caption_preset: Optional[str] = Form(None),
    upload_id: Optional[str] = Form(None),
):
    api_key = await resolve_gemini(request)
    if not api_key and not (llm_backend.active() and not BILLING_ENABLED):
        # Self-host with an OpenAI-compatible server configured needs no
        # Google key for the core pipeline: the moment picker runs there and
        # the frame-based stages degrade on their own (layout_picker returns
        # "none", silent videos fail with a message that says why).
        raise gemini_missing_error()

    ack_flag = str(acknowledged).lower() in ("1", "true", "yes")
    # May be lowered by the paid-proxy budget check in the metering block
    # below; self-host never runs that block, so it must default here.
    paid_allowed = True
    force_low = str(force_low_quality).lower() in ("1", "true", "yes")

    # Handle JSON body manually for URL payload
    content_type = request.headers.get("content-type", "")
    if "application/json" in content_type:
        body = await request.json()
        url = body.get("url")
        ack_flag = bool(body.get("acknowledged"))
        force_low = bool(body.get("force_low_quality"))
        output_format = body.get("output_format")
        layouts = body.get("layouts")
        webhook_url = body.get("webhook_url")
        webhook_secret = body.get("webhook_secret")
        target_clips = body.get("target_clips")
        clip_min_seconds = body.get("clip_min_seconds")
        clip_max_seconds = body.get("clip_max_seconds")
        auto_hook = body.get("auto_hook")
        auto_hook_style = body.get("auto_hook_style")
        thumbnail_session_id = body.get("thumbnail_session_id")
        captions = body.get("captions")
        caption_preset = body.get("caption_preset")
        upload_id = body.get("upload_id")

    # Normalize output format (auto = keep pipeline default).
    if output_format not in ("vertical", "horizontal", "square"):
        output_format = "auto"

    # Accepts a JSON list or a comma-separated form field.
    if isinstance(layouts, str):
        layouts = [p for p in layouts.split(",") if p.strip()]
    elif not isinstance(layouts, list):
        layouts = []

    # caption_preset: a subtitles.CAPTION_PRESETS name for the auto-caption
    # pass of this job. A bad name 400s here, before any download or probe,
    # rather than silently shipping a look the user did not ask for.
    caption_preset = caption_preset or None
    if caption_preset is not None and caption_preset not in CAPTION_PRESETS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown caption_preset '{caption_preset}'. "
                   f"One of: {', '.join(CAPTION_PRESETS)}.")

    # Module handover (issue #68): reuse the Thumbnail Studio source video and
    # its transcript so publishing to YouTube can flow straight into clip
    # generation without re-uploading or re-transcribing.
    thumb_session = None
    if thumbnail_session_id and not url and not file:
        thumb_session = thumbnail_sessions.get(thumbnail_session_id)
        if not thumb_session:
            raise HTTPException(status_code=404, detail="Thumbnail session not found or expired")
        await _assert_job_owner(request, thumb_session)
        src = thumb_session.get("video_path")
        if not src or not os.path.exists(src):
            raise HTTPException(status_code=404, detail="Source video for this session is no longer on disk")

    # Agent upload (POST /api/uploads + PUT): the file is already on disk.
    upload_slot = None
    if upload_id and not url and not file and not thumb_session:
        upload_slot = _take_pending_upload(upload_id, await _owner_id(request))

    if not url and not file and not thumb_session and not upload_slot:
        raise HTTPException(status_code=400, detail="Must provide URL, File or upload_id")

    # Completion callback: reject unsafe targets NOW (clear 400) — delivery
    # re-validates anyway, but failing at submit is the debuggable behavior.
    if webhook_url:
        from security_utils import assert_public_url, UnsafeURLError
        try:
            await asyncio.get_event_loop().run_in_executor(
                None, assert_public_url, webhook_url)
        except UnsafeURLError as e:
            raise HTTPException(status_code=400, detail=f"Invalid webhook_url: {e}")

    if not ack_flag:
        raise HTTPException(status_code=400, detail="You must confirm you own the content or have rights to process it.")

    if url and DISABLE_YOUTUBE_URL:
        raise HTTPException(status_code=403, detail="YouTube URL ingest is disabled on this deployment. Please upload a file you own.")

    # Pre-flight quality gate: probe the offered resolution BEFORE starting, so
    # the user can abort (refresh cookies / update yt-dlp) instead of burning
    # 20 min on a 360p-only source. Fail-open: any probe error starts normally.
    # The probe also runs under force_low_quality so the short-source check
    # can't be bypassed through the quality-gate confirm.
    if url and (QUALITY_GATE_MIN_HEIGHT > 0 or MIN_SOURCE_SECONDS > 0):
        probe = await _probe_youtube_quality(url)
        # Hard reject, no confirm-and-retry: a too-short source fails the same
        # way on every retry, so letting the user force it just burns the job.
        source_duration = int(probe.get("duration") or 0)
        if MIN_SOURCE_SECONDS > 0 and 0 < source_duration < MIN_SOURCE_SECONDS:
            _reject_short_source(source_duration)
        max_height = int(probe.get("max_height") or 0)
        if not force_low and QUALITY_GATE_MIN_HEIGHT > 0 \
                and 0 < max_height < QUALITY_GATE_MIN_HEIGHT:
            print(f"⚠️ Quality gate: only {max_height}p available for {url} — asking user first.")
            return JSONResponse({
                "needs_confirmation": True,
                "quality_check": {
                    "max_height": max_height,
                    "min_height": QUALITY_GATE_MIN_HEIGHT,
                    "cookies_invalid": bool(probe.get("cookies_invalid")),
                },
            })

    # Capture attestation context for legal record (IP + timestamp + UA)
    client_ip = request.client.host if request.client else "unknown"
    fwd = request.headers.get("x-forwarded-for")
    if fwd:
        client_ip = fwd.split(",")[0].strip()
    user_agent = request.headers.get("user-agent", "")
    attestation = {
        "acknowledged": True,
        "ip": client_ip,
        "user_agent": user_agent,
        "timestamp": time.time(),
        "source": ("thumbnail_session" if thumb_session else "upload_id" if upload_slot
                   else "url" if url else "file"),
    }

    job_id = str(uuid.uuid4())
    job_output_dir = os.path.join(OUTPUT_DIR, job_id)
    os.makedirs(job_output_dir, exist_ok=True)

    # Prepare Command
    # sys.executable, not "python": bare "python" resolves against PATH, which
    # outside Docker is whatever interpreter happens to be first — not the venv
    # running this server. Every job then dies on `import cv2`. The quality
    # probe above already gets this right.
    cmd = [sys.executable, "-u", "main.py"] # -u for unbuffered
    env = os.environ.copy()
    if not paid_allowed:
        # Daily paid-proxy budget hit: this job runs on the free routes only.
        env.pop("PROXY_URL", None)
    if api_key:
        env["GEMINI_API_KEY"] = api_key # Override with key from request
    else:
        env.pop("GEMINI_API_KEY", None)  # local-LLM job: main.py must not find a stale key
    # The stdio fix above only covers this process. main.py prints an emoji on
    # its first line and configures nothing, so on a cp1252 console the child
    # still dies before it renders anything -- the server starts and every job
    # fails instead. setdefault, so an explicit PYTHONIOENCODING still wins.
    env.setdefault("PYTHONIOENCODING", "utf-8")

    # Optional layouts are per job. The renderer reads these at import time in
    # the subprocess, so they must be set before Popen — same path WATERMARK
    # already takes.
    chosen = layout_env(layouts)
    env.update(chosen)
    if chosen:
        print(f"[layouts] job={job_id} enabled={sorted(chosen)}")

    # Auto-hook: burn each clip's Gemini hook text during the render. Off when
    # the field is absent, so API/MCP/webhook callers keep their old output
    # byte-for-byte; the dashboard sends an explicit value either way.
    if str(auto_hook).lower() in ("1", "true", "yes"):
        env["AUTO_HOOK"] = "1"
        from hooks import HOOK_STYLES
        if auto_hook_style in HOOK_STYLES:
            env["AUTO_HOOK_STYLE"] = auto_hook_style
        print(f"[auto-hook] job={job_id} style={env.get('AUTO_HOOK_STYLE', 'classic')}")

    # Manual generation controls (discussion #65): optional clip-count target
    # and duration band, forwarded to the selection prompts via the same env
    # overrides the A/B harness already reads (clip_selection.py). All three
    # are honest TARGETS, not guarantees — the model may return fewer clips
    # when the material doesn't hold them. Bad values 400 instead of silently
    # producing something the user didn't ask for.
    def _gen_control(raw, name, lo, hi, integer=False):
        if raw in (None, ""):
            return None
        try:
            val = float(raw)
        except (TypeError, ValueError):
            raise HTTPException(status_code=400, detail=f"{name} must be a number")
        if integer and val != int(val):
            raise HTTPException(status_code=400, detail=f"{name} must be an integer")
        if not (lo <= val <= hi):
            raise HTTPException(status_code=400,
                                detail=f"{name} must be between {lo:g} and {hi:g}")
        return int(val) if integer else val

    n_clips = _gen_control(target_clips, "target_clips", 1, 15, integer=True)
    min_secs = _gen_control(clip_min_seconds, "clip_min_seconds", 5, 175)
    max_secs = _gen_control(clip_max_seconds, "clip_max_seconds", 10, 180)
    if min_secs is not None and max_secs is not None and max_secs < min_secs + 5:
        raise HTTPException(status_code=400,
                            detail="clip_max_seconds must be at least 5s above clip_min_seconds")
    if n_clips is not None:
        env["CLIP_TARGET_MIN"] = env["CLIP_TARGET_MAX"] = str(n_clips)
    if min_secs is not None:
        env["CLIP_MIN_SECONDS"] = str(min_secs)
    if max_secs is not None:
        env["CLIP_MAX_SECONDS"] = str(max_secs)
    if n_clips is not None or min_secs is not None or max_secs is not None:
        print(f"[gen-controls] job={job_id} clips={n_clips} band={min_secs}-{max_secs}")

    # captions=false: the source already carries burned-in subtitles (or the
    # caller adds its own later), so skip the free auto-caption pass instead
    # of stacking a second layer. Absent → the deployment default (on).
    if captions is not None and str(captions).lower() in ("0", "false", "no"):
        env["AUTO_CAPTIONS"] = "0"
        print(f"[captions] job={job_id} auto-captions off")

    # caption_preset: the look the auto pass burns on every clip of THIS job.
    # Absent → the deployment default, so API and MCP callers keep their old
    # output. Validated up front, next to the other request fields.
    if caption_preset:
        env["AUTO_CAPTION_PRESET"] = caption_preset
        print(f"[captions] job={job_id} preset={caption_preset}")

    input_path = None
    if url:
        # Keep the downloaded source inside the job dir: the clip editor's
        # re-render path cuts new segments from it, and it ages out with the
        # rest of the job (retention window + OUTPUT_MAX_GB cap) either way.
        cmd.extend(["-u", url, "--keep-original"])
    elif thumb_session:
        # Hardlink (or copy) the session's video under the job's name so source
        # lookup, the clip editor and the preview treat it exactly like a normal
        # upload; the transcript rides along so the pipeline skips Whisper.
        src = thumb_session["video_path"]
        src_duration = _media_duration_seconds(src)
        if MIN_SOURCE_SECONDS > 0 and 0 < src_duration < MIN_SOURCE_SECONDS:
            shutil.rmtree(job_output_dir, ignore_errors=True)
            _reject_short_source(src_duration)
        input_path = os.path.join(UPLOAD_DIR, f"{job_id}_{os.path.basename(src)}")
        try:
            os.link(src, input_path)
        except OSError:
            shutil.copyfile(src, input_path)
        cmd.extend(["-i", input_path])
        # An empty transcript (e.g. a silent or music-only source) is not worth
        # forwarding: main.py would reject it and retranscribe anyway.
        if thumb_session.get("transcript_ready") and (thumb_session.get("transcript") or {}).get("segments"):
            transcript_path = os.path.join(job_output_dir, "source_transcript.json")
            with open(transcript_path, "w") as f:
                json.dump(thumb_session["transcript"], f)
            cmd.extend(["--transcript", transcript_path])
    elif upload_slot:
        # Move the pre-uploaded file under the job's name so it is cleaned up
        # with the job like any other upload; the slot is consumed.
        src = upload_slot["path"]
        src_duration = _media_duration_seconds(src)
        if MIN_SOURCE_SECONDS > 0 and 0 < src_duration < MIN_SOURCE_SECONDS:
            shutil.rmtree(job_output_dir, ignore_errors=True)
            _reject_short_source(src_duration)
        input_path = os.path.join(UPLOAD_DIR, f"{job_id}_{upload_slot['filename']}")
        os.replace(src, input_path)
        pending_uploads.pop(upload_id, None)
        cmd.extend(["-i", input_path])
    else:
        # Save uploaded file with size limit check.
        # basename() strips any path components from the client-supplied
        # filename so a name like "../../main.py" can't escape UPLOAD_DIR.
        safe_name = os.path.basename(file.filename or "upload") or "upload"
        input_path = os.path.join(UPLOAD_DIR, f"{job_id}_{safe_name}")

        # Read file in chunks to check size
        size = 0
        limit_bytes = MAX_FILE_SIZE_MB * 1024 * 1024

        with open(input_path, "wb") as buffer:
            while content := await file.read(1024 * 1024): # Read 1MB chunks
                size += len(content)
                if size > limit_bytes:
                    os.remove(input_path)
                    shutil.rmtree(job_output_dir)
                    raise HTTPException(status_code=413, detail=f"File too large. Max size {MAX_FILE_SIZE_MB}MB")
                buffer.write(content)

        upload_duration = _media_duration_seconds(input_path)
        if MIN_SOURCE_SECONDS > 0 and 0 < upload_duration < MIN_SOURCE_SECONDS:
            os.remove(input_path)
            shutil.rmtree(job_output_dir, ignore_errors=True)
            _reject_short_source(upload_duration)

        cmd.extend(["-i", input_path])

    cmd.extend(["-o", job_output_dir])
    if output_format and output_format != "auto":
        cmd.extend(["--format", output_format])

    print(f"[attestation] job={job_id} ip={attestation['ip']} source={attestation['source']} ack=true")

    # Meter + reserve minutes for managed users (no-op for BYOK / self-host).
    user_id, priority, reservation_id, user_plan = await reserve_process_minutes(request, url, input_path, job_id)
    if user_plan == "free":
        # Free-plan clips carry a burned-in watermark (applied by the main.py
        # subprocess after each clip renders).
        env["WATERMARK"] = "1"

    # Absolute-URL base for the webhook payload: explicit env wins (the API may
    # sit behind a proxy whose forwarded headers we can't trust), else what the
    # caller connected to.
    api_base = os.environ.get("PUBLIC_API_URL", "").rstrip("/") or str(request.base_url).rstrip("/")

    # Enqueue Job
    jobs[job_id] = {
        'status': 'queued',
        'logs': [f"Job {job_id} queued."],
        'cmd': cmd,
        'env': env,
        'output_dir': job_output_dir,
        'attestation': attestation,
        'user_id': user_id,
        'reservation_id': reservation_id,
        'watermark': env.get("WATERMARK") == "1",
        'webhook_url': webhook_url,
        'webhook_secret': webhook_secret,
        'base_url': api_base,
    }

    # Persist the owner so recovered jobs keep their multi-tenant guard after a
    # restart (see _recover_jobs_from_disk).
    if user_id is not None:
        try:
            os.makedirs(job_output_dir, exist_ok=True)
            with open(os.path.join(job_output_dir, ".owner"), "w") as f:
                f.write(str(user_id))
        except Exception as e:
            print(f"⚠️ Could not persist job owner for {job_id}: {e}")

    # Resume manifest: enough to re-run this job if the container dies mid-flight
    # (a redeploy). No secrets — the env is rebuilt from os.environ on resume.
    _write_resume_manifest(job_id, cmd, priority, user_id, reservation_id,
                           watermark=jobs[job_id]['watermark'],
                           webhook_url=webhook_url, webhook_secret=webhook_secret,
                           base_url=api_base)

    _enqueue_job(job_id, priority)

    return {"job_id": job_id, "status": "queued"}

def _job_view_from_disk(job_id):
    """What the disk says about a job this instance does not hold in memory.

    During a deploy handover a poll can land on either instance; the one that
    did not accept the job must still answer from the shared directory: a
    metadata JSON means completed (recovered like at startup), a manifest
    means the other instance has it (heartbeat) or will pick it up (queued).
    """
    job_path = os.path.join(OUTPUT_DIR, job_id)
    if not os.path.isdir(job_path):
        return None
    if glob.glob(os.path.join(job_path, "*_metadata.json")):
        _recover_jobs_from_disk()
        return jobs.get(job_id)
    m = _read_manifest(job_id)
    if m is None:
        return None
    alive = time.time() - float(m.get("heartbeat") or 0) < HEARTBEAT_STALE_AFTER
    owner = m.get("user_id")
    return {
        'status': 'processing' if alive else 'queued',
        'logs': ["♻️ The server was updated; your video continues on the new instance."],
        'user_id': (int(owner) if isinstance(owner, str) and owner.isdigit() else owner),
        'result': None,
    }


def _presented_status(job_id, job):
    """A job we hold as 'queued' while draining is really the next instance's:
    if it has started it, say so instead of showing a queue that never moves."""
    if job.get('status') == 'queued' and _draining:
        m = _read_manifest(job_id)
        if m and _manifest_busy_elsewhere(m):
            return 'processing'
    return job['status']


@app.get("/api/status/{job_id}")
async def get_status(job_id: str, request: Request):
    job = jobs.get(job_id)
    if job is None:
        job = _job_view_from_disk(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")

    await _assert_job_owner(request, job)
    return {
        "status": _presented_status(job_id, job),
        "logs": _visible_logs(job['logs']),
        "result": job.get('result')
    }


def _locate_source(job_id: str):
    """Find a job's source video on disk, or None.

    Upload jobs keep it in uploads/{job_id}_*; URL jobs keep the download in
    the job dir (--keep-original) under the name recorded as ``source_video``
    in metadata.json. Either way it ages out with the normal retention caps.
    """
    matches = [
        f for f in glob.glob(os.path.join(UPLOAD_DIR, f"{glob.escape(job_id)}_*"))
        if not os.path.basename(f).startswith("thumb_")
    ]
    if matches:
        return matches[0]
    try:
        meta_files = glob.glob(os.path.join(OUTPUT_DIR, job_id, "*_metadata.json"))
        if meta_files:
            with open(meta_files[0], 'r') as f:
                name = json.load(f).get('source_video')
            if name:
                candidate = os.path.join(OUTPUT_DIR, job_id, os.path.basename(name))
                if os.path.exists(candidate):
                    return candidate
    except Exception:
        pass
    return None


# How long a signed source URL stays valid. Long enough to survive an editing
# session and a page reload, short enough that a link leaked through a log, a
# referer or a shared screenshot is dead by the time anyone tries it.
SOURCE_URL_TTL_SECONDS = int(os.environ.get("SOURCE_URL_TTL_SECONDS", "21600"))


def _source_signature(job_id: str, exp: int) -> str:
    """HMAC tying a job id to an expiry, keyed on the app's JWT secret."""
    secret = (_cloud_config.settings.jwt_secret if BILLING_ENABLED else "") or ""
    msg = f"{job_id}:{exp}".encode()
    return hmac.new(secret.encode(), msg, hashlib.sha256).hexdigest()[:32]


def _job_record(job_id: str):
    """The in-memory job record, or what the shared disk says about it."""
    return jobs.get(job_id) or _job_view_from_disk(job_id)


def _signed_source_url(job_id: str) -> str:
    """The URL to hand a `<video>` tag for this job's source.

    Every place that returns a source URL to the browser must go through here:
    a caller that builds the bare path instead ships a player that cannot
    authenticate, and the clip editor's source monitor 404s in cloud while
    working perfectly in self-host and in every test.
    """
    if not BILLING_ENABLED:
        return f"/api/source/{job_id}"
    exp = int(time.time()) + SOURCE_URL_TTL_SECONDS
    return f"/api/source/{job_id}?exp={exp}&sig={_source_signature(job_id, exp)}"


@app.get("/api/source-url/{job_id}")
async def get_source_url(job_id: str, request: Request):
    """Mint a short-lived signed URL for this job's source video.

    A `<video src>` cannot carry an Authorization header, which is why
    /api/source was left open in the first place. So the owner asks for a
    capability URL here, with the bearer token, and hands the player that.
    """
    record = _job_record(job_id)
    if record is None:
        # No record means no owner to check, and minting anyway would hand out
        # a working key to whoever asked: the one door in this design that
        # gives a capability without asking who you are. A real job resolves
        # here (metadata recovers it, an in-flight one has a manifest naming
        # its owner), so the only callers this turns away are asking about a
        # job that does not exist. Off billing there is nothing to protect and
        # the self-host preview must keep working.
        if BILLING_ENABLED:
            raise HTTPException(status_code=404, detail="Source not found")
    else:
        await _assert_job_owner(request, record)
    return {"url": _signed_source_url(job_id)}


@app.get("/api/source/{job_id}")
async def get_source_video(job_id: str, request: Request,
                           exp: int = 0, sig: str = ""):
    """Stream a job's original source video for the live-analysis preview and
    the clip editor's source monitor.

    Uploaded sources are blob URLs in the browser and don't survive a reload,
    so the recovered session points the preview here instead.

    This one endpoint serves the untouched original, which for a URL job is the
    file we downloaded. Left open it is a public downloader wearing a UUID, so
    a cloud job with an owner needs either a signed URL from /api/source-url or
    a bearer token on the request. Self-host and ownerless BYOK jobs are
    unchanged: there is no owner to check and no secret to sign with.
    """
    signed = (
        BILLING_ENABLED and sig
        and exp > time.time()
        and hmac.compare_digest(sig, _source_signature(job_id, exp))
    )
    # Gated on BILLING_ENABLED, not just on `signed`: _job_record falls through
    # to _job_view_from_disk, which rescans the whole output directory. Off
    # billing there is no owner to find, so that scan would run on every single
    # preview load and buy nothing.
    if not signed and BILLING_ENABLED:
        record = _job_record(job_id)
        if record is not None:
            await _assert_job_owner(request, record)
    source_path = _locate_source(job_id)
    if not source_path:
        raise HTTPException(status_code=404, detail="Source not found")
    return FileResponse(source_path, media_type="video/mp4")


@app.get("/api/jobs/{job_id}/download-all")
async def download_all_clips(job_id: str, request: Request):
    """Bundle the current version of every clip of a job into one ZIP."""
    await _ensure_job_files(job_id, request)
    if job_id in jobs:
        await _assert_job_owner(request, jobs[job_id])

    output_dir = os.path.join(OUTPUT_DIR, job_id)
    json_files = glob.glob(os.path.join(output_dir, "*_metadata.json"))
    if not json_files:
        raise HTTPException(status_code=404, detail="Job not found")

    with open(json_files[0], 'r', encoding='utf-8') as f:
        data = json.load(f)

    # The metadata file on disk never carries video_url — the pipeline doesn't
    # write it, it's injected into the in-memory job record. So prefer the live
    # record (it also tracks edits like subtitled_/hook_ renames) and fall back
    # to the canonical name a job/restore rebuilds, instead of finding nothing.
    base_name = os.path.basename(json_files[0]).replace('_metadata.json', '')
    mem_clips = ((jobs.get(job_id) or {}).get('result') or {}).get('clips') or []

    files = []
    for i, clip in enumerate(data.get('shorts', [])):
        url = None
        if i < len(mem_clips):
            url = (mem_clips[i] or {}).get('video_url')
        url = url or clip.get('video_url')
        filename = (os.path.basename(url.split('/')[-1]) if url
                    else _canonical_clip_file(output_dir, base_name, i))
        path = os.path.join(output_dir, filename)
        if filename and os.path.exists(path):
            files.append((i, path))

    if not files:
        raise HTTPException(status_code=404, detail="No clip files found for this job")

    zip_path = os.path.join(output_dir, f"clips_{int(time.time())}.zip")

    def build_zip():
        # Videos are already compressed; store instead of deflate for speed.
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_STORED) as zf:
            for i, path in files:
                zf.write(path, arcname=f"clip_{i + 1:02d}_{os.path.basename(path)}")

    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, build_zip)

    return FileResponse(
        zip_path,
        media_type="application/zip",
        filename=f"openshorts_clips_{job_id[:8]}.zip",
        background=BackgroundTask(os.remove, zip_path),
    )


# --- Project restore (paid mode) --------------------------------------------
# Re-hydrates an archived project from R2 back into output/{job_id}/ so every
# edit endpoint works on it again. Restored files land with a fresh mtime, so
# the retention clock restarts; re-restoring after a purge is cheap.
_restore_locks: Dict[str, asyncio.Lock] = {}
# Job ids are uuid4 strings; anything else under /videos is not a job dir
# (thumbnails, stray probes) and must not reach the database.
_JOB_ID_RE = re.compile(r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$')


@app.post("/api/projects/{job_id}/restore")
async def restore_project(job_id: str, request: Request):
    if not BILLING_ENABLED:
        raise HTTPException(status_code=404, detail="Not found")
    from sqlalchemy import select
    from cloud.auth import get_current_user_required
    from cloud.models import Project
    from cloud import database as cloud_db, storage as cloud_storage

    user = await get_current_user_required(request)
    async with cloud_db.session() as s:
        proj = (await s.execute(
            select(Project).where(Project.job_id == job_id)
        )).scalar_one_or_none()
    if proj is None or str(proj.user_id) != str(user.id):
        raise HTTPException(status_code=404, detail="Project not found")

    await _restore_job_files(job_id, proj, str(user.id))
    return {
        "job_id": job_id,
        "status": "completed",
        "result": jobs[job_id]['result'],
        "project_state": proj.state,
        "title": proj.title,
    }


async def _restore_job_files(job_id: str, proj, user_id: str) -> bool:
    """Bring a project's working files back from R2 and register the job.

    The ownership check is the caller's job: ``restore_project`` (the
    endpoint) verifies the session, ``_restore_for_public_path`` serves files
    that are public by job id anyway. Returns True when files were actually
    pulled, False on the idempotent fast path (everything already on disk).
    """
    from cloud import storage as cloud_storage

    pulled = False
    # Per-job lock: a double click must not download the project twice.
    lock = _restore_locks.setdefault(job_id, asyncio.Lock())
    async with lock:
        job_dir = os.path.join(OUTPUT_DIR, job_id)

        # Idempotent fast path: everything the project needs is already on disk.
        needed = {os.path.basename(proj.metadata_r2_key)}
        for c in (proj.state or {}).get("clips", []):
            for k in ("original_file", "server_file"):
                if c.get(k):
                    needed.add(c[k])
        if os.path.isdir(job_dir) and all(
            os.path.exists(os.path.join(job_dir, f)) for f in needed
        ):
            os.utime(job_dir, None)  # restart the retention clock
        else:
            prefix = cloud_storage.job_key(user_id, job_id, "")
            keys = await asyncio.to_thread(cloud_storage.list_keys, prefix)
            if not keys:
                raise HTTPException(status_code=502,
                                    detail="Project files are no longer available")
            # Download into a temp dir first so a partial failure never leaves a
            # half-restored job dir that the fast path would mistake for complete.
            tmp_dir = job_dir + ".restoring"
            shutil.rmtree(tmp_dir, ignore_errors=True)
            os.makedirs(tmp_dir, exist_ok=True)
            sem = asyncio.Semaphore(3)

            async def _download(key):
                fname = os.path.basename(key)
                if not fname:
                    return
                async with sem:
                    await asyncio.to_thread(
                        cloud_storage.download_file, key, os.path.join(tmp_dir, fname))

            try:
                await asyncio.gather(*(_download(k) for k in keys))
            except Exception as e:
                shutil.rmtree(tmp_dir, ignore_errors=True)
                raise HTTPException(status_code=502, detail=f"Restore download failed: {e}")
            # Owner sidecar keeps the multi-tenant guard after a server restart.
            with open(os.path.join(tmp_dir, ".owner"), "w") as f:
                f.write(user_id)
            pulled = True
            if os.path.isdir(job_dir):
                for fname in os.listdir(tmp_dir):
                    shutil.move(os.path.join(tmp_dir, fname), os.path.join(job_dir, fname))
                shutil.rmtree(tmp_dir, ignore_errors=True)
                os.utime(job_dir, None)
            else:
                os.rename(tmp_dir, job_dir)

        # Register (or refresh) the in-memory job — same shape as
        # _recover_jobs_from_disk, so every edit endpoint works unchanged.
        json_files = glob.glob(os.path.join(job_dir, "*_metadata.json"))
        if not json_files:
            raise HTTPException(status_code=502, detail="Project metadata missing")
        with open(json_files[0], 'r') as f:
            data = json.load(f)
        base_name = os.path.basename(json_files[0]).replace('_metadata.json', '')
        clips = data.get('shorts', [])
        for i, clip in enumerate(clips):
            if not clip.get('video_url'):
                clip['video_url'] = (
                    f"/videos/{job_id}/"
                    f"{_canonical_clip_file(job_dir, base_name, i)}")
        jobs[job_id] = {
            'status': 'completed',
            'logs': ["♻️ Project restored from your library."],
            'output_dir': job_dir,
            'user_id': user_id,
            'result': {'clips': clips, 'cost_analysis': data.get('cost_analysis')},
        }
    if pulled:
        print(f"♻️  Restored {job_id} from the library (working files were gone).")
    return pulled


async def _restore_for_public_path(job_id: str) -> bool:
    """Restorer for the /videos mount: a miss under /videos/<job_id>/ pulls
    the project back from R2 for its owner, without a session. Those files
    are public by job id already (the mount has no auth), so this grants
    nothing new; it only stops a reopened project's players from 404ing
    while the API side is still restoring it. True means "look again".
    """
    if not BILLING_ENABLED or not _JOB_ID_RE.match(job_id or ""):
        return False
    from sqlalchemy import select
    from cloud.models import Project
    from cloud import database as cloud_db
    async with cloud_db.session() as s:
        proj = (await s.execute(
            select(Project).where(Project.job_id == job_id)
        )).scalar_one_or_none()
    if proj is None:
        return False
    try:
        await _restore_job_files(job_id, proj, str(proj.user_id))
    except HTTPException as e:
        print(f"⚠️  /videos restore of {job_id} failed: {e.detail}")
        return False
    except Exception as e:
        print(f"⚠️  /videos restore of {job_id} failed: {e}")
        return False
    return True


async def _ensure_job_files(job_id: str, request: Request) -> bool:
    """Make a completed job usable again after its working files vanished.

    OUTPUT_DIR is not durable — a container restart or redeploy wipes it — so
    endpoints that read a job's files would 404 on a project the user can still
    see in their library. Pull it back from R2 on demand (same path as the
    explicit /restore), so editing keeps working instead of dead-ending.

    Returns True when the job is available afterwards. Never raises: callers
    keep their own 404s for jobs that genuinely don't exist.
    """
    job_dir = os.path.join(OUTPUT_DIR, job_id)
    if job_id in jobs and glob.glob(os.path.join(job_dir, "*_metadata.json")):
        return True
    if not BILLING_ENABLED:
        return False
    try:
        await restore_project(job_id, request)
        return True
    except HTTPException:
        return False
    except Exception as e:
        print(f"⚠️  Auto-restore failed for {job_id}: {e}")
        return False


from editor import VideoEditor
from subtitles import (generate_srt, generate_ass, burn_subtitles, generate_srt_from_video,
                       CAPTION_PRESETS, CAPTION_PRESET_LOOK_FIELDS, caption_preset,
                       caption_presets_for_api)
from hooks import add_hook_to_video
from translate import translate_video, get_supported_languages
from thumbnail import (analyze_video_for_titles, refine_titles, generate_thumbnail,
                       generate_youtube_description, extract_face_frames)

class EditRequest(BaseModel):
    job_id: str
    clip_index: int
    api_key: Optional[str] = None
    input_filename: Optional[str] = None

@app.post("/api/edit")
async def edit_clip(
    req: EditRequest,
    request: Request,
):
    # Cloud (paid) mode disables BYOK: ignore any body api_key so it can't skip
    # the entitlement gate or metering (mirrors resolve_gemini ignoring the
    # header). Self-host keeps BYOK — the body key wins there.
    body_key = None if BILLING_ENABLED else req.api_key
    final_api_key = body_key or await resolve_gemini(request)

    if not final_api_key:
        raise gemini_missing_error()

    await _ensure_job_files(req.job_id, request)
    if req.job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    job = jobs[req.job_id]
    await _assert_job_owner(request, job)
    if 'result' not in job or 'clips' not in job['result']:
        raise HTTPException(status_code=400, detail="Job result not available")

    # Meter the managed Gemini call so it can't be looped for free. Skip only for
    # genuine BYOK (self-host body key) — in cloud, body_key is always None.
    edit_minutes = _cloud_config.MANAGED_ANALYSIS_MINUTES if BILLING_ENABLED else 0
    reservation_id = None if body_key else await reserve_managed_action(
        request, edit_minutes, req.job_id, "edit")

    try:
        # Resolve Input Path: Prefer explict input_filename from frontend (chaining edits)
        if req.input_filename:
            # Security: Ensure just a filename, no paths
            safe_name = os.path.basename(req.input_filename)
            input_path = os.path.join(OUTPUT_DIR, req.job_id, safe_name)
            filename = safe_name
        else:
            # Fallback to original clip
            clip = job['result']['clips'][req.clip_index]
            filename = clip['video_url'].split('/')[-1]
            input_path = os.path.join(OUTPUT_DIR, req.job_id, filename)
        
        if not os.path.exists(input_path):
             raise HTTPException(status_code=404, detail=f"Video file not found: {input_path}")

        # Edit the clip WITHOUT its burned captions, then put them back on top —
        # otherwise the captions are baked into the edit and the next subtitle
        # pass stacks a second layer over them (see _reapply_captions).
        clean_name = _strip_burned_captions(os.path.join(OUTPUT_DIR, req.job_id), filename)
        had_captions = clean_name != filename
        if had_captions:
            filename = clean_name
            input_path = os.path.join(OUTPUT_DIR, req.job_id, clean_name)

        # Define output path for edited video
        edited_filename = f"edited_{filename}"
        output_path = os.path.join(OUTPUT_DIR, req.job_id, edited_filename)
        
        # Run editing in a thread to avoid blocking main loop
        # Since VideoEditor uses blocking calls (subprocess, API wait)
        def run_edit():
            editor = VideoEditor(api_key=final_api_key)
            
            # SAFE FILE RENAMING STRATEGY (Avoid UnicodeEncodeError in Docker)
            # Create a safe ASCII filename in the same directory
            safe_filename = f"temp_input_{req.job_id}.mp4"
            safe_input_path = os.path.join(OUTPUT_DIR, req.job_id, safe_filename)
            
            # Copy original file to safe path
            # (Copy is safer than rename if something crashes, we keep original)
            shutil.copy(input_path, safe_input_path)
            
            try:
                # 1. Upload (using safe path)
                vid_file = editor.upload_video(safe_input_path)
                
                # 2. Get duration
                import cv2
                cap = cv2.VideoCapture(safe_input_path)
                fps = cap.get(cv2.CAP_PROP_FPS)
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                duration = frame_count / fps if fps else 0
                cap.release()
                
                # Load transcript from metadata
                transcript = None
                try:
                    meta_files = glob.glob(os.path.join(OUTPUT_DIR, req.job_id, "*_metadata.json"))
                    if meta_files:
                        with open(meta_files[0], 'r') as f:
                            data = json.load(f)
                            transcript = data.get('transcript')
                except Exception as e:
                    print(f"⚠️ Could not load transcript for editing context: {e}")

                # 3. Get Plan (Filter String)
                # Zooms would crop burned-in captions/hooks off screen, so tell
                # the editor when the source already carries them. `filename` is
                # the original clip name (safe_input_path is an ASCII temp copy).
                has_captions = ("subtitled_" in filename) or ("hook_" in filename) or ("hooked_" in filename)
                filter_data = editor.get_ffmpeg_filter(vid_file, duration, fps=fps, width=width, height=height, transcript=transcript, has_captions=has_captions)
                
                # 4. Apply
                # Use safe output name first
                safe_output_path = os.path.join(OUTPUT_DIR, req.job_id, f"temp_output_{req.job_id}.mp4")
                editor.apply_edits(safe_input_path, safe_output_path, filter_data)
                
                # Move result to final destination (rename works even if dest name has unicode if filesystem supports it, 
                # but python might still struggle if locale is broken? No, os.rename usually handles it better than subprocess args)
                # Actually, output_path is defined above: f"edited_{filename}"
                # If filename has unicode, output_path has unicode.
                # Let's hope shutil.move / os.rename works.
                if os.path.exists(safe_output_path):
                    shutil.move(safe_output_path, output_path)
                
                return filter_data
            finally:
                # Cleanup temp safe input
                if os.path.exists(safe_input_path):
                    os.remove(safe_input_path)

        # Run in thread pool
        loop = asyncio.get_event_loop()
        plan = await loop.run_in_executor(None, run_edit)

        # Captions back on top, so the clip the user sees keeps them and the
        # clean edited file stays available for a later restyle.
        if had_captions:
            recap = await loop.run_in_executor(
                None, _reapply_captions, req.job_id, req.clip_index, output_path)
            if recap:
                edited_filename = os.path.basename(recap)

        new_video_url = f"/videos/{req.job_id}/{edited_filename}"

        # Persist the new current file like /api/subtitle does: in-memory job
        # result + metadata.json, so reload/recovery/re-archive see this version.
        if req.clip_index < len(job['result']['clips']):
            job['result']['clips'][req.clip_index]['video_url'] = new_video_url
        try:
            meta_files = glob.glob(os.path.join(OUTPUT_DIR, req.job_id, "*_metadata.json"))
            if meta_files:
                with open(meta_files[0], 'r') as f:
                    meta = json.load(f)
                shorts = meta.get('shorts', [])
                if req.clip_index < len(shorts):
                    shorts[req.clip_index]['video_url'] = new_video_url
                    meta['shorts'] = shorts
                    with open(meta_files[0], 'w') as f:
                        json.dump(meta, f, indent=4)
        except Exception as e:
            print(f"⚠️ Failed to update metadata.json: {e}")

        _archive_clip_edit_bg(req.job_id, req.clip_index, edited_filename)

        if reservation_id:
            await _metering.commit_reservation(reservation_id)
        return {
            "success": True,
            "new_video_url": new_video_url,
            "edit_plan": plan
        }

    except Exception as e:
        if reservation_id:
            await _metering.release_reservation(reservation_id)
        print(f"❌ Edit Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

class CaptionWordIn(BaseModel):
    """One user-edited caption word, clip-relative ms — the same shape the
    /transcript endpoint hands the subtitle modal."""
    text: str
    startMs: int
    endMs: int


class SubtitleRequest(BaseModel):
    job_id: str
    clip_index: int
    position: str = "bottom" # top, middle, bottom
    font_size: int = 16
    font_name: str = "Verdana"
    font_color: str = "#FFFFFF"
    border_color: str = "#000000"
    border_width: int = 2
    bg_color: str = "#000000"
    bg_opacity: float = 0.0
    style: str = "classic"  # classic (uniform color) or karaoke (word highlight)
    highlight_color: str = "#FFD700"
    effect: str = "none"  # none | glow | pop | box | wipe | bounce (karaoke only)
    base_opacity: float = 1.0  # opacity of non-active words (dimmed modern look)
    uppercase: bool = False
    # A named look from subtitles.CAPTION_PRESETS. Fills every style field the
    # caller did NOT set explicitly, so `{"preset": "hormozi"}` is enough and
    # `{"preset": "hormozi", "highlight_color": "#FF0000"}` keeps the override.
    preset: Optional[str] = None
    input_filename: Optional[str] = None
    # User-edited caption words. When present, the burn uses them VERBATIM
    # instead of regenerating from the stored transcript — without this, text
    # edits in the modal were silently discarded on the server render path.
    words: Optional[List[CaptionWordIn]] = None


@app.get("/api/clip/{job_id}/{clip_index}/transcript")
async def get_clip_transcript(job_id: str, clip_index: int, request: Request):
    """Return word-level captions for a specific clip, formatted for Remotion."""
    await _ensure_job_files(job_id, request)
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    await _assert_job_owner(request, jobs[job_id])
    output_dir = os.path.join(OUTPUT_DIR, job_id)
    json_files = glob.glob(os.path.join(output_dir, "*_metadata.json"))

    if not json_files:
        raise HTTPException(status_code=404, detail="Metadata not found")

    with open(json_files[0], 'r') as f:
        data = json.load(f)

    transcript = data.get('transcript')
    if not transcript:
        raise HTTPException(status_code=400, detail="Transcript not found in metadata")

    clips = data.get('shorts', [])
    if clip_index >= len(clips):
        raise HTTPException(status_code=404, detail="Clip not found")

    clip_data = clips[clip_index]

    # Recut clips are concatenations of source segments; remap the transcript
    # onto the clip timeline so every consumer keeps its flat start..end logic.
    recipe_segments = (clip_data.get('recipe') or {}).get('segments')
    if recipe_segments:
        transcript = recut.virtual_transcript(transcript, recipe_segments)
        clip_start, clip_end = 0.0, recut.total_duration(recipe_segments)
    else:
        clip_start = clip_data.get('start', 0)
        clip_end = clip_data.get('end', 0)

    # Extract words within clip range and convert to CaptionWord format
    captions = []
    for segment in transcript.get('segments', []):
        for word_info in segment.get('words', []):
            if word_info['end'] > clip_start and word_info['start'] < clip_end:
                captions.append({
                    "text": word_info.get('word', '').strip(),
                    "startMs": int((max(0, word_info['start'] - clip_start)) * 1000),
                    "endMs": int((max(0, word_info['end'] - clip_start)) * 1000),
                })

    duration_sec = clip_end - clip_start

    return {
        "captions": captions,
        "durationSec": duration_sec,
        "language": transcript.get('language', 'en'),
    }


# --- Clip editor: EDL + re-render ---

# The editor ships the WHOLE source transcript, not a window around the clip:
# the point of the source track is extending a cut into material the clip never
# covered, and you cannot pick a new in-point from words you were not sent.
# Cost is about 7 KB of JSON per minute of speech, fetched once per editor open.


def _source_duration_seconds(path):
    """Probe a video's duration; None when it can't be read."""
    try:
        import cv2
        cap = cv2.VideoCapture(path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        cap.release()
        # OpenCV reports -1/-1 for files it can't read — a naive truthiness
        # check would turn that into a phantom 1.0s duration.
        if fps > 0 and frames > 0:
            return round(frames / fps, 3)
    except Exception:
        pass
    return None


def _clip_recipe_parts(clip):
    """(segments, canonical_range) for a clip — synthesized from the flat
    start/end for clips that were never recut."""
    recipe = clip.get('recipe') or {}
    fallback = {"start": float(clip.get('start', 0) or 0),
                "end": float(clip.get('end', 0) or 0)}
    segments = recipe.get('segments') or [dict(fallback)]
    canonical_range = recipe.get('canonical_range') or dict(fallback)
    return segments, canonical_range


@app.get("/api/clip/{job_id}/{clip_index}/edl")
async def get_clip_edl(job_id: str, clip_index: int, request: Request):
    """The clip's editable recipe: which source segments it was cut from, the
    word timeline around them, and whether the source is still available for
    cuts outside the original range."""
    await _ensure_job_files(job_id, request)
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    job = jobs[job_id]
    await _assert_job_owner(request, job)

    output_dir = os.path.join(OUTPUT_DIR, job_id)
    json_files = glob.glob(os.path.join(output_dir, "*_metadata.json"))
    if not json_files:
        raise HTTPException(status_code=404, detail="Metadata not found")
    with open(json_files[0], 'r') as f:
        data = json.load(f)

    clips = data.get('shorts', [])
    if clip_index < 0 or clip_index >= len(clips):
        raise HTTPException(status_code=404, detail="Clip not found")
    clip = clips[clip_index]

    segments, canonical_range = _clip_recipe_parts(clip)
    transcript = data.get('transcript') or {}
    words = recut.transcript_words(transcript)

    source_path = _locate_source(job_id)
    source_duration = _source_duration_seconds(source_path) if source_path else None
    duration_estimated = source_duration is None
    if duration_estimated:
        # Best remaining scale for the source track: the last spoken word or
        # the furthest point any recipe touches.
        candidates = [canonical_range['end']] + [s['end'] for s in segments]
        if words:
            candidates.append(words[-1]['e'])
        source_duration = round(max(candidates), 3)

    words_out = [
        {"w": w["w"], "s": round(w["s"], 3), "e": round(w["e"], 3)}
        for w in words
    ]

    current_file = (clip.get('video_url') or '').split('/')[-1]
    if not current_file:
        base_name = os.path.basename(json_files[0]).replace('_metadata.json', '')
        current_file = _canonical_clip_file(output_dir, base_name, clip_index)

    total = recut.total_duration(segments)
    return {
        "job_id": job_id,
        "clip_index": clip_index,
        "title": clip.get('video_title_for_youtube_short') or '',
        "segments": segments,
        "framing": (clip.get('recipe') or {}).get('framing') or 'auto',
        "canonical_range": canonical_range,
        "duration": total,
        "current_file": current_file,
        "has_captions": bool(re.match(r'^subtitled_\d+_', current_file)),
        "words": words_out,
        "source": {
            "available": bool(source_path),
            "url": _signed_source_url(job_id) if source_path else None,
            "duration": source_duration,
            "duration_estimated": duration_estimated,
        },
        "limits": {
            "max_segments": recut.MAX_SEGMENTS,
            "min_segment_seconds": recut.MIN_SEGMENT_SECONDS,
            "max_total_seconds": recut.MAX_TOTAL_SECONDS,
        },
        "rerender_minutes": (max(1, math.ceil(total / 60.0))
                             if BILLING_ENABLED else 0),
    }


class RerenderSegment(BaseModel):
    start: float
    end: float


class RerenderRequest(BaseModel):
    job_id: str
    clip_index: int
    segments: List[RerenderSegment]
    snap_to_words: bool = False
    reapply_captions: bool = True
    # None = inherit the recipe's framing (so plain trims keep the look);
    # 'auto' resets to the classifier; 'full'/'track' force a layout.
    framing: Optional[str] = None


# Manual framing -> reframe-engine strategy. 'full' shows the whole source
# frame (WIDE: no side-cropping, blurred filler bands); 'track' forces the
# subject-tracking crop. Anything non-auto needs the retained source video.
_FRAMING_STRATEGIES = {"auto": None, "full": "WIDE", "track": "TRACK"}


# One lock per job (same pattern as _restore_locks): rerenders on the same job
# share metadata.json and the canonical files, so they must not interleave.
_rerender_locks: Dict[str, asyncio.Lock] = {}

# Scene-listing builds write stable preview/thumbnail names per job; serialize
# them so overlapping editor opens don't tear each other's files.
_scenes_locks: Dict[str, asyncio.Lock] = {}


@app.post("/api/clip/rerender")
async def rerender_clip(req: RerenderRequest, request: Request):
    """Re-render a clip from an edited EDL (the clip editor's save button).

    Two paths, chosen automatically:
    - FAST: every segment stays inside the range the canonical clip was cut
      from → recut straight from the already-reframed canonical file. No ML,
      no source needed.
    - SOURCE: a segment reaches outside → recut from the retained source and
      re-reframe with the same engine the pipeline used. 409 when the source
      already aged out.
    """
    await require_managed_entitlement(request)
    await _ensure_job_files(req.job_id, request)
    if req.job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    job = jobs[req.job_id]
    await _assert_job_owner(request, job)

    # Serialize rerenders per job: concurrent saves (easy for an MCP agent to
    # produce) would otherwise race on the shared metadata.json
    # read-modify-write below, with the last writer silently reverting the
    # other clip's recipe/video_url.
    lock = _rerender_locks.setdefault(req.job_id, asyncio.Lock())
    async with lock:
        return await _rerender_locked(req, request, job)


async def _rerender_locked(req: RerenderRequest, request: Request, job):
    output_dir = os.path.join(OUTPUT_DIR, req.job_id)
    json_files = glob.glob(os.path.join(output_dir, "*_metadata.json"))
    if not json_files:
        raise HTTPException(status_code=404, detail="Metadata not found")
    with open(json_files[0], 'r') as f:
        data = json.load(f)

    clips = data.get('shorts', [])
    if req.clip_index < 0 or req.clip_index >= len(clips):
        raise HTTPException(status_code=404, detail="Clip not found")
    clip = clips[req.clip_index]
    transcript = data.get('transcript') or {}

    base_name = os.path.basename(json_files[0]).replace('_metadata.json', '')
    clean_name = f"{base_name}_clip_{req.clip_index + 1}.mp4"
    canonical_path = os.path.join(output_dir, clean_name)

    _, canonical_range = _clip_recipe_parts(clip)
    source_path = _locate_source(req.job_id)
    source_duration = _source_duration_seconds(source_path) if source_path else None

    framing = req.framing or (clip.get('recipe') or {}).get('framing') or 'auto'
    if framing not in _FRAMING_STRATEGIES:
        raise HTTPException(status_code=400,
                            detail="framing must be one of: auto, full, track")
    force_strategy = _FRAMING_STRATEGIES[framing]

    try:
        segments = recut.normalize_segments(
            [{"start": s.start, "end": s.end} for s in req.segments],
            source_duration)
        if req.snap_to_words:
            snap_bound = source_duration or max(s['end'] for s in segments)
            segments = recut.snap_segments(segments, transcript, snap_bound)
            if not source_path:
                # Snapping trails into silence and may nudge a boundary past
                # the canonical range; without a source the fast path is the
                # only path, so clamp back instead of failing with a 409.
                segments = [
                    {"start": round(max(s['start'], canonical_range['start']), 3),
                     "end": round(min(s['end'], canonical_range['end']), 3)}
                    for s in segments]
            # Re-validate after snapping/clamping: a segment fully outside the
            # canonical range clamps to an inverted (end < start) window, and
            # snapping can stretch the total past the cap. Without this it
            # reaches ffmpeg and dies as a 500 instead of a clean 400.
            segments = recut.normalize_segments(segments, source_duration)
    except recut.RecutError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # A framing override always re-reframes from the source: the canonical file
    # has the old layout baked into its pixels.
    fast = (force_strategy is None
            and os.path.exists(canonical_path)
            and recut.within_range(segments, canonical_range['start'],
                                   canonical_range['end']))
    if not fast and not source_path:
        raise HTTPException(
            status_code=409,
            detail=("The source video is no longer on the server; changing the "
                    "framing needs it." if force_strategy is not None else
                    "The source video is no longer on the server, so segments "
                    "must stay within the original clip range."))

    total = recut.total_duration(segments)
    rerender_minutes = (max(1, math.ceil(total / 60.0))
                        if BILLING_ENABLED else 0)
    reservation_id = await reserve_managed_action(
        request, rerender_minutes, req.job_id, "rerender")

    v_transcript = (recut.virtual_transcript(transcript, segments)
                    if req.reapply_captions else None)

    def run_recut():
        if fast:
            return recut.perform_recut(
                input_path=canonical_path,
                segments=recut.rebase_segments(
                    segments, canonical_range['start'], canonical_range['end']),
                output_dir=output_dir, clean_name=clean_name,
                reframe=False, captions_transcript=v_transcript)
        return recut.perform_recut(
            input_path=source_path, segments=segments,
            output_dir=output_dir, clean_name=clean_name,
            reframe=True, output_format=data.get('output_format', 'auto'),
            watermark=bool(job.get('watermark')),
            force_strategy=force_strategy,
            captions_transcript=v_transcript)

    try:
        loop = asyncio.get_event_loop()
        served_name, _clean_recut_name = await loop.run_in_executor(None, run_recut)

        new_video_url = f"/videos/{req.job_id}/{served_name}"
        new_recipe = {"v": 1, "segments": segments,
                      "canonical_range": canonical_range}
        if framing != 'auto':
            new_recipe["framing"] = framing
        # Covering range, deliberately not segments[0]/segments[-1]: segments
        # may legally be out of source order, and downstream consumers only
        # need a sane positive window (the recipe is the real timeline).
        new_start = min(s['start'] for s in segments)
        new_end = max(s['end'] for s in segments)

        updates = {'video_url': new_video_url, 'start': new_start,
                   'end': new_end, 'recipe': new_recipe,
                   # The stacked stretches of THIS render (empty on the fast
                   # path, which never reframes): captions follow them.
                   'layout_ranges': layout_ranges.read(
                       os.path.join(output_dir, _clean_recut_name))}
        # Per-scene manual framing is keyed by scene indices of a specific cut;
        # this render neither applied it nor can it survive a changed cut, so
        # clear it rather than let /scenes serve stale overrides against the
        # wrong shots (re-frame after trimming to re-apply by hand).
        if clip.get('crop_overrides'):
            updates['crop_overrides'] = None
        clip.update(updates)
        data['shorts'] = clips
        with open(json_files[0], 'w') as f:
            json.dump(data, f, indent=2)
        mem_clips = (job.get('result') or {}).get('clips') or []
        if req.clip_index < len(mem_clips):
            mem_clips[req.clip_index].update(updates)

        _archive_clip_edit_bg(req.job_id, req.clip_index, served_name)
        if reservation_id:
            await _metering.commit_reservation(reservation_id)
        return {
            "success": True,
            "new_video_url": new_video_url,
            "recipe": new_recipe,
            "framing": framing,
            "start": new_start,
            "end": new_end,
            "duration": total,
            "render_path": "fast" if fast else "source",
        }
    except Exception as e:
        if reservation_id:
            await _metering.release_reservation(reservation_id)
        print(f"❌ Rerender Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# --- Manual framing -----------------------------------------------------------
#
# The reframe engine picks the crop automatically, and on a podcast it is right
# most of the time and grossly wrong occasionally: a wide shot of the whole
# table averages over one face, so the scene falls to GENERAL (letterboxed) or
# tracks the wrong person. There was no way to say "no, frame it here".
#
# The unit is the SCENE, not the clip, because a podcast cuts between a fixed
# close camera and a fixed wide one, and the right crop differs per camera.
# Scene boundaries already are the camera changes: PySceneDetect finds them.
#
# Scenes the user never touches keep the automatic camera, so correcting one
# bad shot cannot spoil the ones the tracker got right.

class ReframeRequest(BaseModel):
    job_id: str
    clip_index: int
    # scene index (string key, JSON-style) -> either a crop centre as a
    # fraction of the source width, or {"top": f, "bottom": f} to stack two
    # regions. Fractions travel instead of pixels so the editor never needs to
    # know the source dimensions.
    crop_overrides: Dict[str, Any]
    reapply_captions: bool = True


def _clip_scene_workfile(source_path, segments, output_dir, token):
    """Cut the clip out of the source so scenes can be detected on it.

    The name is unique per call, unlike the thumbnails and the preview: two
    editor opens on the same clip would otherwise write the same temp file at
    once, and the first to finish deletes it out from under the second.
    """
    work_path = os.path.join(output_dir, f"scenes_{token}.mp4")
    recut.run_cut_concat(source_path, segments, work_path, output_dir)
    return work_path


@app.get("/api/clip/{job_id}/{clip_index}/scenes")
async def get_clip_scenes(job_id: str, clip_index: int, request: Request):
    """Scenes of a clip, each with a SOURCE frame to frame it against.

    The frames come from the uncropped cut, not the delivered clip: the point
    is to show what the automatic crop threw away, which the 9:16 file no
    longer contains.
    """
    # Entitlement too, not just ownership: listing scenes cuts the clip from
    # source, runs scene detection and encodes a preview — real compute.
    await require_managed_entitlement(request)
    await _ensure_job_files(job_id, request)
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    job = jobs[job_id]
    await _assert_job_owner(request, job)

    output_dir = os.path.join(OUTPUT_DIR, job_id)
    json_files = glob.glob(os.path.join(output_dir, "*_metadata.json"))
    if not json_files:
        raise HTTPException(status_code=404, detail="Metadata not found")
    with open(json_files[0], 'r') as f:
        data = json.load(f)

    clips = data.get('shorts', [])
    if clip_index < 0 or clip_index >= len(clips):
        raise HTTPException(status_code=404, detail="Clip not found")

    if data.get('output_format') == 'horizontal':
        raise HTTPException(
            status_code=400,
            detail="Horizontal clips keep the full frame; there is no crop to reframe.")

    clip = clips[clip_index]
    segments, _canonical_range = _clip_recipe_parts(clip)
    # What was applied last time. Without this the editor reopens blank, and
    # since a re-render rebuilds from source using ONLY what it is sent, the
    # next save would silently drop every earlier adjustment.
    saved_overrides = clip.get('crop_overrides') or {}
    source_path = _locate_source(job_id)
    if not source_path:
        raise HTTPException(
            status_code=409,
            detail="The source video is no longer on the server, so the "
                   "framing of this clip can no longer be changed.")

    def build():
        import cv2
        import main as m

        # Stable for the files the browser fetches (no accumulation), unique
        # for the temp cut (no collision between overlapping requests).
        # temp_ prefix keeps these editor-only artifacts out of the self-host
        # S3 backup (it skips temp_*); stable names still avoid accumulation.
        token = str(clip_index)
        work_token = f"{clip_index}_{uuid.uuid4().hex[:8]}"
        preview_name = f"temp_preview_{clip_index}.mp4"
        preview_path = os.path.join(output_dir, preview_name)
        work_path = _clip_scene_workfile(source_path, segments, output_dir, work_token)
        try:
            scenes, fps = m.detect_scenes(work_path)
            fps = float(fps) or 30.0
            orig_w, orig_h = m.get_video_resolution(work_path)

            cap = cv2.VideoCapture(work_path)
            if not scenes:
                total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
                bounds = [(0, total)]
            else:
                bounds = [(s.get_frames(), e.get_frames()) for s, e in scenes]

            out = []
            for idx, (start_f, end_f) in enumerate(bounds):
                mid = (start_f + end_f) // 2
                cap.set(cv2.CAP_PROP_POS_FRAMES, mid)
                ok, frame = cap.read()
                thumb_name = f"temp_scene_{token}_{idx:03d}.jpg"
                suggested = 0.5
                suggested_y = 0.5
                if ok:
                    cv2.imwrite(os.path.join(output_dir, thumb_name), frame,
                                [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                    # Start the rectangle on the biggest face in the shot, so
                    # the common case is a nudge rather than a hunt.
                    try:
                        faces = m.detect_face_candidates(frame)
                        if faces:
                            box = max(faces,
                                      key=lambda f: f['box'][2] * f['box'][3])['box']
                            suggested = min(1.0, max(0.0,
                                                     (box[0] + box[2] / 2) / orig_w))
                            # SPLIT halves crop vertically too, so the face's
                            # height matters there (TRACK ignores it).
                            suggested_y = min(1.0, max(0.0,
                                                       (box[1] + box[3] / 2) / orig_h))
                    except Exception:
                        pass
                else:
                    thumb_name = None

                out.append({
                    "index": idx,
                    "start": round(start_f / fps, 3),
                    "end": round(end_f / fps, 3),
                    "thumbnail_url": (f"/videos/{job_id}/{thumb_name}"
                                      if thumb_name else None),
                    "suggested_center": round(suggested, 4),
                    "suggested_center_y": round(suggested_y, 4),
                })
            cap.release()
            return orig_w, orig_h, out, preview_name
        finally:
            # The uncropped cut becomes a light preview instead of being
            # discarded: judging the framing means knowing who is talking, and
            # the delivered 9:16 file no longer shows the rest of the room.
            try:
                if os.path.exists(work_path):
                    subprocess.run(
                        ["ffmpeg", "-y", "-loglevel", "error", "-i", work_path,
                         "-vf", "scale=640:-2", "-c:v", "libx264", "-preset",
                         "veryfast", "-crf", "30", "-c:a", "aac", "-b:a", "96k",
                         # faststart: the editor's <video> streams the preview;
                         # a tail moov would stall it until fully downloaded.
                         "-movflags", "+faststart",
                         preview_path], check=True, timeout=600)
            except Exception as exc:
                print(f"Scene preview failed: {exc}")
            finally:
                if os.path.exists(work_path):
                    os.remove(work_path)

    # Serialized per job: two overlapping opens would run two ffmpeg writers
    # on the same stable preview/thumbnail names and serve a torn file.
    lock = _scenes_locks.setdefault(job_id, asyncio.Lock())
    async with lock:
        try:
            loop = asyncio.get_event_loop()
            orig_w, orig_h, scenes_out, preview_name = await loop.run_in_executor(None, build)
        except Exception as e:
            print(f"Scene listing error: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    # Width of the crop window as a fraction of the source width: the editor
    # draws the rectangle with it and needs no pixel arithmetic of its own.
    # The aspect follows the job's output format (9:16, or 1:1 for square).
    aspect = 1.0 if data.get('output_format') == 'square' else 9.0 / 16.0
    crop_w = min(orig_w, orig_h * aspect)
    return {
        "job_id": job_id,
        "clip_index": clip_index,
        "source_width": orig_w,
        "source_height": orig_h,
        "crop_width_fraction": round(crop_w / orig_w, 4),
        "preview_url": f"/videos/{job_id}/{preview_name}",
        "saved_overrides": saved_overrides,
        "scenes": scenes_out,
    }


@app.post("/api/clip/reframe")
async def reframe_clip(req: ReframeRequest, request: Request):
    """Re-render a clip with hand-framed scenes, leaving its cut untouched.

    Deliberately separate from /api/clip/rerender: the overrides are keyed by
    scene index, and a scene index only means anything against a given cut. If
    framing rode along with a trim save, changing the trim would silently move
    every override onto the wrong shot.
    """
    await require_managed_entitlement(request)
    await _ensure_job_files(req.job_id, request)
    if req.job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    job = jobs[req.job_id]
    await _assert_job_owner(request, job)

    def _fraction(v):
        return min(1.0, max(0.0, float(v)))

    overrides = {}
    for key, value in (req.crop_overrides or {}).items():
        try:
            idx = int(key)
            if isinstance(value, dict):
                def _half(h):
                    if isinstance(h, dict):
                        return {"x": _fraction(h["x"]), "y": _fraction(h.get("y", 0.5))}
                    return {"x": _fraction(h), "y": 0.5}
                overrides[idx] = {"top": _half(value["top"]),
                                  "bottom": _half(value["bottom"])}
            else:
                overrides[idx] = _fraction(value)
        except (KeyError, TypeError, ValueError):
            continue
    if not overrides:
        raise HTTPException(status_code=400,
                            detail="No scene framing was provided.")

    # Same lock as /rerender: both read-modify-write the job's metadata.json,
    # and the metadata must be read INSIDE the lock or a rerender committing
    # in between gets clobbered by a write of stale data.
    lock = _rerender_locks.setdefault(req.job_id, asyncio.Lock())
    async with lock:
        return await _reframe_locked(req, request, job, overrides)


async def _reframe_locked(req: ReframeRequest, request: Request, job, overrides):
    output_dir = os.path.join(OUTPUT_DIR, req.job_id)
    json_files = glob.glob(os.path.join(output_dir, "*_metadata.json"))
    if not json_files:
        raise HTTPException(status_code=404, detail="Metadata not found")
    with open(json_files[0], 'r') as f:
        data = json.load(f)

    clips = data.get('shorts', [])
    if req.clip_index < 0 or req.clip_index >= len(clips):
        raise HTTPException(status_code=404, detail="Clip not found")
    clip = clips[req.clip_index]

    if data.get('output_format') == 'horizontal':
        raise HTTPException(
            status_code=400,
            detail="Horizontal clips keep the full frame; there is no crop to reframe.")

    segments, canonical_range = _clip_recipe_parts(clip)
    source_path = _locate_source(req.job_id)
    if not source_path:
        raise HTTPException(
            status_code=409,
            detail="The source video is no longer on the server, so the "
                   "framing of this clip can no longer be changed.")

    # Whole-clip framing (recipe.framing, the clip editor's selector) still
    # applies to the scenes the user did NOT hand-position: apply_crop_overrides
    # runs after force_strategy, so a per-scene choice beats the whole-clip one.
    framing = (clip.get('recipe') or {}).get('framing') or 'auto'
    force_strategy = _FRAMING_STRATEGIES.get(framing)

    # A reframe re-renders the full cut from source — same work as a source-path
    # rerender, so it meters the same.
    total = recut.total_duration(segments)
    rerender_minutes = (max(1, math.ceil(total / 60.0))
                        if BILLING_ENABLED else 0)
    reservation_id = await reserve_managed_action(
        request, rerender_minutes, req.job_id, "reframe")

    # Every default clip ships with burned captions; re-rendering without them
    # would silently hand back a caption-less file.
    v_transcript = (recut.virtual_transcript(data.get('transcript') or {}, segments)
                    if req.reapply_captions else None)

    base_name = os.path.basename(json_files[0]).replace('_metadata.json', '')
    # The CLEAN base name, exactly as /api/clip/rerender does — never the
    # current derived file. perform_recut prefixes whatever it is given, so
    # feeding it the newest derivative made every re-render add another
    # recut_<ts>_<hex>_ in front of the previous one. A few rounds of framing
    # and captioning and the path blew past the filesystem limit:
    # "Error opening output ...: File name too long".
    clean_name = f"{base_name}_clip_{req.clip_index + 1}.mp4"

    def run():
        return recut.perform_recut(
            input_path=source_path, segments=segments,
            output_dir=output_dir, clean_name=clean_name,
            reframe=True, output_format=data.get('output_format', 'auto'),
            watermark=bool(job.get('watermark')),
            force_strategy=force_strategy,
            crop_overrides=overrides,
            captions_transcript=v_transcript)

    try:
        loop = asyncio.get_event_loop()
        served_name, _clean = await loop.run_in_executor(None, run)

        new_video_url = f"/videos/{req.job_id}/{served_name}"
        new_recipe = {"v": 1, "segments": segments,
                      "canonical_range": canonical_range}
        if framing != 'auto':
            new_recipe["framing"] = framing
        updates = {
            'video_url': new_video_url,
            'recipe': new_recipe,
            'crop_overrides': {str(k): v for k, v in overrides.items()},
            'layout_ranges': layout_ranges.read(os.path.join(output_dir, _clean)),
        }
        clip.update(updates)
        data['shorts'] = clips
        with open(json_files[0], 'w') as f:
            json.dump(data, f, indent=2)
        mem_clips = (job.get('result') or {}).get('clips') or []
        if req.clip_index < len(mem_clips):
            mem_clips[req.clip_index].update(updates)

        _archive_clip_edit_bg(req.job_id, req.clip_index, served_name)
        if reservation_id:
            await _metering.commit_reservation(reservation_id)
        # The cut is untouched, but the response mirrors /rerender's shape
        # so the dashboard can reuse one handler without clearing the
        # clip's timing fields.
        return {
            "success": True,
            "new_video_url": new_video_url,
            "recipe": new_recipe,
            "start": min(s['start'] for s in segments),
            "end": max(s['end'] for s in segments),
            "framed_scenes": sorted(overrides),
        }
    except Exception as e:
        if reservation_id:
            await _metering.release_reservation(reservation_id)
        print(f"Reframe Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# --- Remotion Render Proxy ---
RENDER_SERVICE_URL = os.getenv("RENDER_SERVICE_URL", "http://renderer:3100")

@app.post("/api/render")
async def proxy_render(request: Request):
    """Proxy render requests to the Node.js Remotion render service."""
    await require_managed_entitlement(request)
    import httpx
    body = await request.json()
    render_minutes = _cloud_config.RENDER_MINUTES if BILLING_ENABLED else 0
    reservation_id = await reserve_managed_action(
        request, render_minutes, str(uuid.uuid4()), "render")
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(f"{RENDER_SERVICE_URL}/render", json=body)
        result = resp.json()
        if reservation_id:
            await _metering.commit_reservation(reservation_id)
        return result
    except Exception as e:
        if reservation_id:
            await _metering.release_reservation(reservation_id)
        raise HTTPException(status_code=502, detail=f"Render service unavailable: {e}")

@app.get("/api/render/{render_id}")
async def proxy_render_status(render_id: str):
    """Proxy render status polling to the Node.js Remotion render service."""
    import httpx
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(f"{RENDER_SERVICE_URL}/render/{render_id}")
            return resp.json()
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Render service unavailable: {e}")


class EffectsGenerateRequest(BaseModel):
    job_id: str
    clip_index: int
    input_filename: Optional[str] = None

@app.post("/api/effects/generate")
async def generate_effects_config(
    req: EffectsGenerateRequest,
    request: Request,
):
    """Generate structured EffectsConfig JSON for Remotion rendering via Gemini AI."""
    final_api_key = await resolve_gemini(request)

    if not final_api_key:
        raise gemini_missing_error()

    await _ensure_job_files(req.job_id, request)
    if req.job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    job = jobs[req.job_id]
    await _assert_job_owner(request, job)
    if 'result' not in job or 'clips' not in job['result']:
        raise HTTPException(status_code=400, detail="Job result not available")

    # Meter the managed Gemini call (no-op for self-host).
    fx_minutes = _cloud_config.MANAGED_ANALYSIS_MINUTES if BILLING_ENABLED else 0
    reservation_id = await reserve_managed_action(request, fx_minutes, req.job_id, "effects")

    try:
        # Resolve input path
        if req.input_filename:
            safe_name = os.path.basename(req.input_filename)
            input_path = os.path.join(OUTPUT_DIR, req.job_id, safe_name)
        else:
            clip = job['result']['clips'][req.clip_index]
            filename = clip['video_url'].split('/')[-1]
            input_path = os.path.join(OUTPUT_DIR, req.job_id, filename)

        if not os.path.exists(input_path):
            raise HTTPException(status_code=404, detail=f"Video file not found: {input_path}")

        def run_effects_generation():
            editor = VideoEditor(api_key=final_api_key)

            # Create safe ASCII filename to avoid encoding issues
            safe_filename = f"temp_effects_{req.job_id}.mp4"
            safe_input_path = os.path.join(OUTPUT_DIR, req.job_id, safe_filename)
            shutil.copy(input_path, safe_input_path)

            try:
                # Upload video to Gemini
                vid_file = editor.upload_video(safe_input_path)

                # Get video metadata via ffprobe
                probe_cmd = [
                    'ffprobe', '-v', 'error',
                    '-select_streams', 'v:0',
                    '-show_entries', 'stream=width,height,r_frame_rate,duration',
                    '-show_entries', 'format=duration',
                    '-of', 'json',
                    safe_input_path
                ]
                probe_result = subprocess.check_output(probe_cmd).decode().strip()
                probe_data = json.loads(probe_result)

                stream = probe_data.get('streams', [{}])[0]
                width = int(stream.get('width', 1080))
                height = int(stream.get('height', 1920))

                # Parse fps from r_frame_rate (e.g. "30/1")
                r_frame_rate = stream.get('r_frame_rate', '30/1')
                num, den = r_frame_rate.split('/')
                fps = round(int(num) / int(den), 2)

                # Get duration from stream or format
                duration = float(stream.get('duration', 0))
                if duration == 0:
                    duration = float(probe_data.get('format', {}).get('duration', 0))

                # Load transcript from metadata
                transcript = None
                try:
                    meta_files = glob.glob(os.path.join(OUTPUT_DIR, req.job_id, "*_metadata.json"))
                    if meta_files:
                        with open(meta_files[0], 'r') as f:
                            data = json.load(f)
                            transcript = data.get('transcript')
                except Exception as e:
                    print(f"⚠️ Could not load transcript for effects config: {e}")

                # Generate effects config
                effects_config = editor.get_effects_config(
                    vid_file, duration, fps=fps, width=width, height=height, transcript=transcript
                )

                return effects_config
            finally:
                if os.path.exists(safe_input_path):
                    os.remove(safe_input_path)

        loop = asyncio.get_event_loop()
        effects_config = await loop.run_in_executor(None, run_effects_generation)

        if effects_config is None:
            raise HTTPException(status_code=500, detail="Failed to generate effects config from Gemini")

        if reservation_id:
            await _metering.commit_reservation(reservation_id)
        return {"effects": effects_config}

    except HTTPException:
        if reservation_id:
            await _metering.release_reservation(reservation_id)
        raise
    except Exception as e:
        if reservation_id:
            await _metering.release_reservation(reservation_id)
        print(f"❌ Effects Generation Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def _apply_caption_preset(req: SubtitleRequest):
    """Fill the style fields the caller left out from ``req.preset``.

    Explicit fields win: only what the JSON did not carry comes from the
    preset (pydantic records the sent fields in ``model_fields_set``), so
    ``{"preset": "hormozi", "highlight_color": "#FF0000"}`` is the Hormozi
    look with a red active word. An unknown name 400s with the valid list.
    """
    if req.preset is None:
        return
    look = caption_preset(req.preset)
    if look is None:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown caption preset '{req.preset}'. "
                   f"One of: {', '.join(CAPTION_PRESETS)}.")
    for field in ("style", *CAPTION_PRESET_LOOK_FIELDS):
        if field not in req.model_fields_set:
            setattr(req, field, look[field])


@app.post("/api/subtitle")
async def add_subtitles(req: SubtitleRequest, request: Request):
    _apply_caption_preset(req)
    await require_managed_entitlement(request)
    await _ensure_job_files(req.job_id, request)
    if req.job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    # Reload job data from disk just in case metadata was updated
    job = jobs[req.job_id]
    await _assert_job_owner(request, job)

    # We need to access metadata.json to get the transcript
    output_dir = os.path.join(OUTPUT_DIR, req.job_id)
    json_files = glob.glob(os.path.join(output_dir, "*_metadata.json"))
    
    if not json_files:
        raise HTTPException(status_code=404, detail="Metadata not found")
        
    with open(json_files[0], 'r') as f:
        data = json.load(f)
        
    transcript = data.get('transcript')
    if not transcript:
        raise HTTPException(status_code=400, detail="Transcript not found in metadata. Please process a new video.")
        
    clips = data.get('shorts', [])
    if req.clip_index >= len(clips):
        raise HTTPException(status_code=404, detail="Clip not found")
        
    clip_data = clips[req.clip_index]

    # Recut clips concatenate several source segments, so their caption window
    # is not the flat start..end range — restyle against the clip-relative
    # remapped transcript instead.
    recipe_segments = (clip_data.get('recipe') or {}).get('segments')
    if recipe_segments:
        sub_transcript = recut.virtual_transcript(transcript, recipe_segments)
        sub_start, sub_end = 0.0, recut.total_duration(recipe_segments)
    else:
        sub_transcript = transcript
        sub_start = clip_data.get('start', 0)
        sub_end = clip_data.get('end', 0)

    # User-edited captions win over both: build a synthetic clip-relative
    # transcript from them so the SRT/ASS generators burn the edited words
    # verbatim (issue #69 — edits used to be dropped on this path).
    if req.words:
        if len(req.words) > 2000:
            raise HTTPException(status_code=400, detail="Too many caption words (max 2000).")
        # Leading space = Whisper's word-boundary convention; without it the
        # block collector treats each word as a continuation fragment and
        # glues the whole line together.
        edited = [
            {"word": " " + w.text.strip(), "start": max(0.0, w.startMs / 1000.0),
             "end": max(0.0, w.endMs / 1000.0)}
            for w in req.words if w.text.strip() and w.endMs > w.startMs >= 0
        ]
        if edited:
            edited.sort(key=lambda w: w["start"])
            sub_transcript = {
                "language": (transcript or {}).get("language", "en"),
                "segments": [{
                    "start": edited[0]["start"], "end": edited[-1]["end"],
                    "text": " ".join(w["word"] for w in edited),
                    "words": edited,
                }],
            }
            sub_start, sub_end = 0.0, max(w["end"] for w in edited)

    # Video Path
    if req.input_filename:
        # Use chained file
        filename = os.path.basename(req.input_filename)
    else:
        # Fallback to standard naming
        filename = clip_data.get('video_url', '').split('/')[-1]
        if not filename:
             base_name = os.path.basename(json_files[0]).replace('_metadata.json', '')
             filename = f"{base_name}_clip_{req.clip_index+1}.mp4"

    # Re-subtitling must replace previous subtitles instead of burning over them.
    filename = _strip_burned_captions(output_dir, filename)

    input_path = os.path.join(output_dir, filename)
    if not os.path.exists(input_path):
        # Try looking for edited version if url implied it?
        # Just fail if not found.
        raise HTTPException(status_code=404, detail=f"Video file not found: {input_path}")

    # Define outputs
    generation_id = int(time.time())
    is_karaoke = req.style == "karaoke"
    srt_filename = f"subs_{req.clip_index}_{generation_id}.{'ass' if is_karaoke else 'srt'}"
    srt_path = os.path.join(output_dir, srt_filename)

    # Style options shared by the karaoke ASS generator paths.
    # Stacked (SPLIT) stretches put their captions on the seam. The metadata
    # copy is authoritative; the sidecar covers clips rendered before it was
    # recorded there.
    seam_ranges = layout_ranges.split_ranges(
        clip_data.get('layout_ranges') or layout_ranges.read(input_path))
    karaoke_opts = dict(
        split_ranges=seam_ranges,
        alignment=req.position, fontsize=req.font_size, font_name=req.font_name,
        font_color=req.font_color, border_color=req.border_color,
        border_width=req.border_width, highlight_color=req.highlight_color,
        bg_color=req.bg_color, bg_opacity=req.bg_opacity,
        effect=req.effect, base_opacity=req.base_opacity, uppercase=req.uppercase,
    )

    # Output video
    # We create a new file "subtitled_..."
    output_filename = f"subtitled_{generation_id}_{filename}"
    output_path = os.path.join(output_dir, output_filename)

    # Burning captions is FREE. They're table stakes for short-form — a clip
    # without them barely works on any platform — and the cost is nil: the SRT
    # comes from the transcript already sitting in metadata.json, and the burn is
    # a single short FFmpeg pass (4s on CPU for a 12s clip, 1-2s on the GPU).
    # Charging 2 minutes for that meant 10% of the whole free monthly quota per
    # captioned clip, roughly what generating the clip cost in the first place —
    # so people skipped it: only 9% of delivered clips had captions (prod audit,
    # 25-jul-2026). The endpoint is already gated by require_managed_entitlement
    # above, so this is not an open door.
    #
    # The dubbed path is the exception and keeps the charge: it runs a fresh
    # Whisper transcription over the translated audio, which is real work.
    is_dubbed = filename.startswith("translated_")
    subtitle_minutes = (_cloud_config.subtitle_minutes_for(filename)
                        if BILLING_ENABLED else 0)
    reservation_id = await reserve_managed_action(
        request, subtitle_minutes, req.job_id, "subtitle")

    try:
        # 1. Generate SRT — from the existing transcript, or a fresh
        # transcription when the audio was dubbed (see the metering note above).
        if is_dubbed:
            print(f"🎙️ Dubbed video detected, transcribing audio for subtitles...")
            def run_transcribe_srt():
                if is_karaoke:
                    return generate_srt_from_video(input_path, srt_path, style="karaoke", **karaoke_opts)
                return generate_srt_from_video(input_path, srt_path)

            loop = asyncio.get_event_loop()
            success = await loop.run_in_executor(None, run_transcribe_srt)
        elif is_karaoke:
            success = generate_ass(sub_transcript, sub_start, sub_end, srt_path, **karaoke_opts)
        else:
            success = generate_srt(sub_transcript, sub_start, sub_end, srt_path)

        if not success:
             raise HTTPException(status_code=400, detail="No words found for this clip range.")

        # 2. Burn Subtitles
        # Run in thread pool
        def run_burn():
             burn_subtitles(input_path, srt_path, output_path,
                           alignment=req.position, fontsize=req.font_size,
                           font_name=req.font_name, font_color=req.font_color,
                           border_color=req.border_color, border_width=req.border_width,
                           bg_color=req.bg_color, bg_opacity=req.bg_opacity)
        
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, run_burn)
        
    except Exception as e:
        print(f"❌ Subtitle Error: {e}")
        if reservation_id:
            await _metering.release_reservation(reservation_id)
        raise HTTPException(status_code=500, detail=str(e))

    if reservation_id:
        await _metering.commit_reservation(reservation_id)

    # 3. Update Result and Metadata
    # Update InMemory Jobs
    if req.clip_index < len(job['result']['clips']):
         job['result']['clips'][req.clip_index]['video_url'] = f"/videos/{req.job_id}/{output_filename}"
    
    # Update Metadata on Disk (Persistence)
    try:
        if req.clip_index < len(clips):
            clips[req.clip_index]['video_url'] = f"/videos/{req.job_id}/{output_filename}"
            # Update the main data structure
            data['shorts'] = clips
            
            # Write back
            with open(json_files[0], 'w') as f:
                json.dump(data, f, indent=4)
                print(f"✅ Metadata updated with subtitled video for clip {req.clip_index}")
    except Exception as e:
        print(f"⚠️ Failed to update metadata.json: {e}")
        # Non-critical, but good for persistence

    _archive_clip_edit_bg(req.job_id, req.clip_index, output_filename)

    return {
        "success": True,
        "new_video_url": f"/videos/{req.job_id}/{output_filename}"
    }

class RemoveSubtitlesRequest(BaseModel):
    job_id: str
    clip_index: int
    input_filename: Optional[str] = None


@app.post("/api/subtitle/remove")
async def remove_subtitles(req: RemoveSubtitlesRequest, request: Request):
    """Point a clip back at its un-captioned original.

    Clips ship captioned by default now, so there has to be a way out — without
    this, a user who doesn't want captions is stuck with them. No re-encode and
    no quota: the pipeline always keeps the clean file next to the derived
    ``subtitled_<ts>_`` one, so removing is just choosing the other file.
    """
    await require_managed_entitlement(request)
    await _ensure_job_files(req.job_id, request)
    if req.job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    job = jobs[req.job_id]
    await _assert_job_owner(request, job)

    output_dir = os.path.join(OUTPUT_DIR, req.job_id)
    json_files = glob.glob(os.path.join(output_dir, "*_metadata.json"))
    if not json_files:
        raise HTTPException(status_code=404, detail="Metadata not found")
    with open(json_files[0], 'r') as f:
        data = json.load(f)
    clips = data.get('shorts', [])
    if req.clip_index >= len(clips):
        raise HTTPException(status_code=404, detail="Clip not found")

    filename = os.path.basename(
        req.input_filename
        or (clips[req.clip_index].get('video_url') or '').split('/')[-1]
        or f"{os.path.basename(json_files[0]).replace('_metadata.json', '')}"
           f"_clip_{req.clip_index + 1}.mp4")

    # Same walk-back the burn path uses, so this undoes any number of restyles.
    while True:
        m = re.match(r'^subtitled_\d+_(.+)$', filename)
        if not m or not os.path.exists(os.path.join(output_dir, m.group(1))):
            break
        filename = m.group(1)

    if not os.path.exists(os.path.join(output_dir, filename)):
        raise HTTPException(status_code=404,
                            detail="The original clip is no longer available.")

    new_url = f"/videos/{req.job_id}/{filename}"
    if req.clip_index < len(job.get('result', {}).get('clips', [])):
        job['result']['clips'][req.clip_index]['video_url'] = new_url
    try:
        clips[req.clip_index]['video_url'] = new_url
        data['shorts'] = clips
        with open(json_files[0], 'w') as f:
            json.dump(data, f, indent=4)
    except Exception as e:
        print(f"⚠️ Failed to update metadata.json: {e}")

    _archive_clip_edit_bg(req.job_id, req.clip_index, filename)
    return {"success": True, "new_video_url": new_url}


class HookRequest(BaseModel):
    job_id: str
    clip_index: int
    text: Optional[str] = ""
    input_filename: Optional[str] = None
    position: Optional[str] = "top" # top, center, bottom
    size: Optional[str] = "M" # S, M, L
    duration_seconds: Optional[float] = None  # None = hook visible for the whole clip
    style: Optional[str] = "classic"  # classic/dark/yellow/red/outline/outline_yellow
    remove: Optional[bool] = False  # strip the burned hook instead of adding one

@app.post("/api/hook")
async def add_hook(req: HookRequest, request: Request):
    await require_managed_entitlement(request)
    await _ensure_job_files(req.job_id, request)
    if req.job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    job = jobs[req.job_id]
    await _assert_job_owner(request, job)
    output_dir = os.path.join(OUTPUT_DIR, req.job_id)
    json_files = glob.glob(os.path.join(output_dir, "*_metadata.json"))
    
    if not json_files:
        raise HTTPException(status_code=404, detail="Metadata not found")
        
    with open(json_files[0], 'r') as f:
        data = json.load(f)
        
    clips = data.get('shorts', [])
    if req.clip_index >= len(clips):
        raise HTTPException(status_code=404, detail="Clip not found")
        
    clip_data = clips[req.clip_index]
    
    # Video Path
    if req.input_filename:
        filename = os.path.basename(req.input_filename)
    else:
        filename = clip_data.get('video_url', '').split('/')[-1]
        if not filename:
             base_name = os.path.basename(json_files[0]).replace('_metadata.json', '')
             filename = f"{base_name}_clip_{req.clip_index+1}.mp4"
         
    input_path = os.path.join(output_dir, filename)
    if not os.path.exists(input_path):
        raise HTTPException(status_code=404, detail=f"Video file not found: {input_path}")

    if not req.remove and not (req.text or "").strip():
        raise HTTPException(status_code=400, detail="Hook text is required")

    # Same invariant as /api/edit: derive from the clip WITHOUT its burned
    # captions, then put them back on top, so a later restyle never stacks a
    # second caption layer (see _reapply_captions). The hook layer is stripped
    # too: a new hook REPLACES the burned one (auto-hook or a previous manual
    # one) instead of stacking on top of it.
    clean_name = _strip_burned_captions(output_dir, filename)
    had_captions = clean_name != filename
    clean_name = _strip_burned_hook(output_dir, clean_name)
    filename = clean_name
    input_path = os.path.join(output_dir, clean_name)

    if req.remove:
        # Nothing to burn: the hook-less file is the target; captions (if the
        # clip had them) go back on below.
        output_filename = filename
        output_path = input_path
        reservation_id = None
    else:
        output_filename = f"hooked_{int(time.time())}_{filename}"
        output_path = os.path.join(output_dir, output_filename)

        # Map Size to Scale
        size_map = {"S": 0.8, "M": 1.0, "L": 1.3}
        font_scale = size_map.get(req.size, 1.0)

        # Meter the FFmpeg overlay re-encode (no-op for BYOK / self-host).
        hook_minutes = _cloud_config.HOOK_MINUTES if BILLING_ENABLED else 0
        reservation_id = await reserve_managed_action(
            request, hook_minutes, req.job_id, "hook")

        try:
            # Run in thread pool
            def run_hook():
                add_hook_to_video(input_path, req.text, output_path, position=req.position, font_scale=font_scale, duration=req.duration_seconds, style=req.style)

            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, run_hook)

        except Exception as e:
            print(f"❌ Hook Error: {e}")
            if reservation_id:
                await _metering.release_reservation(reservation_id)
            raise HTTPException(status_code=500, detail=str(e))

        if reservation_id:
            await _metering.commit_reservation(reservation_id)

    # Captions back on top (see /api/edit for the same invariant).
    if had_captions:
        recap = await asyncio.get_event_loop().run_in_executor(
            None, _reapply_captions, req.job_id, req.clip_index, output_path)
        if recap:
            output_filename = os.path.basename(recap)

    # Record the burned hook so the editor knows what the clip carries (the
    # auto-hook pipeline writes the same key).
    if req.remove:
        clip_data.pop('auto_hook', None)
    else:
        clip_data['auto_hook'] = {
            "text": req.text, "style": req.style, "position": req.position,
            "duration_seconds": req.duration_seconds,
        }

    # Update Persistence (Same logic as subtitles)
    # Update InMemory Jobs
    if req.clip_index < len(job['result']['clips']):
        mem_clip = job['result']['clips'][req.clip_index]
        mem_clip['video_url'] = f"/videos/{req.job_id}/{output_filename}"
        if req.remove:
            mem_clip.pop('auto_hook', None)
        else:
            mem_clip['auto_hook'] = clip_data['auto_hook']

    # Update Metadata on Disk
    try:
        if req.clip_index < len(clips):
            clips[req.clip_index]['video_url'] = f"/videos/{req.job_id}/{output_filename}"
            data['shorts'] = clips
            with open(json_files[0], 'w') as f:
                json.dump(data, f, indent=4)
                print(f"✅ Metadata updated with hook video for clip {req.clip_index}")
    except Exception as e:
        print(f"⚠️ Failed to update metadata.json: {e}")

    _archive_clip_edit_bg(req.job_id, req.clip_index, output_filename)

    return {
        "success": True,
        "new_video_url": f"/videos/{req.job_id}/{output_filename}",
        "burned_hook": None if req.remove else clip_data['auto_hook'],
    }

class TranslateRequest(BaseModel):
    job_id: str
    clip_index: int
    target_language: str
    source_language: Optional[str] = None
    input_filename: Optional[str] = None

@app.get("/api/translate/languages")
async def get_languages():
    """Return supported languages for translation."""
    return {"languages": get_supported_languages()}

@app.post("/api/translate")
async def translate_clip(
    req: TranslateRequest,
    request: Request,
    x_elevenlabs_key: Optional[str] = Header(None, alias="X-ElevenLabs-Key")
):
    """Translate a video clip to a different language using ElevenLabs dubbing."""
    if not x_elevenlabs_key:
        raise HTTPException(status_code=400, detail="Missing X-ElevenLabs-Key header")

    await _ensure_job_files(req.job_id, request)
    if req.job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    job = jobs[req.job_id]
    await _assert_job_owner(request, job)
    output_dir = os.path.join(OUTPUT_DIR, req.job_id)
    json_files = glob.glob(os.path.join(output_dir, "*_metadata.json"))

    if not json_files:
        raise HTTPException(status_code=404, detail="Metadata not found")

    with open(json_files[0], 'r') as f:
        data = json.load(f)

    clips = data.get('shorts', [])
    if req.clip_index >= len(clips):
        raise HTTPException(status_code=404, detail="Clip not found")

    clip_data = clips[req.clip_index]

    # Video Path
    if req.input_filename:
        filename = os.path.basename(req.input_filename)
    else:
        filename = clip_data.get('video_url', '').split('/')[-1]
        if not filename:
             base_name = os.path.basename(json_files[0]).replace('_metadata.json', '')
             filename = f"{base_name}_clip_{req.clip_index+1}.mp4"

    input_path = os.path.join(output_dir, filename)
    if not os.path.exists(input_path):
        raise HTTPException(status_code=404, detail=f"Video file not found: {input_path}")

    # Output video with language suffix
    base, ext = os.path.splitext(filename)
    output_filename = f"translated_{req.target_language}_{base}{ext}"
    output_path = os.path.join(output_dir, output_filename)

    try:
        # Run translation in thread pool (blocking API calls)
        def run_translate():
            return translate_video(
                video_path=input_path,
                output_path=output_path,
                target_language=req.target_language,
                api_key=x_elevenlabs_key,
                source_language=req.source_language,
            )

        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, run_translate)

    except Exception as e:
        print(f"❌ Translation Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    # AI Act art. 50(2): the voice in this file was synthesised, so the file
    # says so in a way a machine can read. Best-effort by design — see
    # ffmpeg_utils.mark_ai_generated.
    from ffmpeg_utils import mark_ai_generated
    await loop.run_in_executor(
        None, lambda: mark_ai_generated(output_path, "AI voice dubbing"))

    # Update InMemory Jobs
    if req.clip_index < len(job['result']['clips']):
         job['result']['clips'][req.clip_index]['video_url'] = f"/videos/{req.job_id}/{output_filename}"

    # Update Metadata on Disk
    try:
        if req.clip_index < len(clips):
            clips[req.clip_index]['video_url'] = f"/videos/{req.job_id}/{output_filename}"
            data['shorts'] = clips
            with open(json_files[0], 'w') as f:
                json.dump(data, f, indent=4)
                print(f"✅ Metadata updated with translated video for clip {req.clip_index}")
    except Exception as e:
        print(f"⚠️ Failed to update metadata.json: {e}")

    _archive_clip_edit_bg(req.job_id, req.clip_index, output_filename)

    return {
        "success": True,
        "new_video_url": f"/videos/{req.job_id}/{output_filename}"
    }

class SocialPostRequest(BaseModel):
    job_id: str
    clip_index: int
    api_key: Optional[str] = None  # BYOK; ignored for managed users
    user_id: Optional[str] = None  # BYOK profile; ignored for managed users
    platforms: List[str] # ["tiktok", "instagram", "youtube"]
    # Optional overrides if frontend wants to edit them
    title: Optional[str] = None
    description: Optional[str] = None
    scheduled_date: Optional[str] = None # ISO-8601 string
    timezone: Optional[str] = "UTC"

import httpx

@app.post("/api/social/post")
async def post_to_socials(req: SocialPostRequest, request: Request):
    await _ensure_job_files(req.job_id, request)
    if req.job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    # Resolve the Upload-Post key + profile. For managed users the server key is
    # used and their own profile is forced (body api_key / user_id are ignored).
    upload_key, forced_profile = await resolve_upload_post(request, req.api_key)
    if not upload_key:
        raise HTTPException(status_code=400, detail="Missing Upload-Post API key")
    post_user = resolve_post_profile(forced_profile, req.user_id)

    job = jobs[req.job_id]
    await _assert_job_owner(request, job)
    if 'result' not in job or 'clips' not in job['result']:
        raise HTTPException(status_code=400, detail="Job result not available")

    try:
        clip = job['result']['clips'][req.clip_index]
        # Video URL is relative /videos/..., we need absolute file path
        # clip['video_url'] is like "/videos/{job_id}/{filename}"
        # We constructed it as: f"/videos/{job_id}/{clip_filename}"
        # And file is at f"{OUTPUT_DIR}/{job_id}/{clip_filename}"
        
        filename = clip['video_url'].split('/')[-1]
        file_path = os.path.join(OUTPUT_DIR, req.job_id, filename)
        
        if not os.path.exists(file_path):
             raise HTTPException(status_code=404, detail=f"Video file not found: {file_path}")

        # Construct parameters for Upload-Post API
        # Fallbacks
        final_title = req.title or clip.get('title', 'Viral Short')
        final_description = req.description or clip.get('video_description_for_instagram') or clip.get('video_description_for_tiktok') or "Check this out!"
        
        # Prepare form data
        url = "https://api.upload-post.com/api/upload"
        headers = {
            "Authorization": f"Apikey {upload_key}"
        }

        # Prepare data as dict (httpx handles lists for multiple values)
        data_payload = {
            "user": post_user,
            "title": final_title,
            "platform[]": req.platforms, # Pass list directly
            "async_upload": "true"  # Enable async upload
        }

        # Add scheduling if present
        if req.scheduled_date:
            data_payload["scheduled_date"] = req.scheduled_date
            if req.timezone:
                data_payload["timezone"] = req.timezone
        
        # Add Platform specifics
        if "tiktok" in req.platforms:
             data_payload["tiktok_title"] = final_description
             data_payload["post_mode"] = TIKTOK_POST_MODE
             
        if "instagram" in req.platforms:
             data_payload["instagram_title"] = final_description
             data_payload["media_type"] = "REELS"

        if "youtube" in req.platforms:
             yt_title = req.title or clip.get('video_title_for_youtube_short', final_title)
             data_payload["youtube_title"] = yt_title
             data_payload["youtube_description"] = final_description
             data_payload["privacyStatus"] = "public"

        # Send File
        # httpx AsyncClient requires async file reading or bytes. 
        # Since we have MAX_FILE_SIZE_MB, reading into memory is safe-ish.
        with open(file_path, "rb") as f:
            file_content = f.read()
            
        files = {
            "video": (filename, file_content, "video/mp4")
        }

        # Switch to synchronous Client to avoid "sync request with AsyncClient" error with multipart/files
        with httpx.Client(timeout=120.0) as client:
            print(f"📡 Sending to Upload-Post for platforms: {req.platforms}")
            response = client.post(url, headers=headers, data=data_payload, files=files)
            
        if response.status_code not in [200, 201, 202]: # Added 201
             print(f"❌ Upload-Post Error: {response.text}")
             raise HTTPException(status_code=response.status_code, detail=f"Vendor API Error: {response.text}")

        return response.json()

    except Exception as e:
        print(f"❌ Social Post Exception: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/social/user")
async def get_social_user(request: Request):
    """Proxy to fetch user profiles from Upload-Post.

    BYOK: uses the caller's key and returns all profiles on that account.
    Managed: uses the server key but returns ONLY the caller's own profile.
    """
    api_key, forced_profile = await resolve_upload_post(request, None)
    if not api_key:
         raise HTTPException(status_code=400, detail="Missing X-Upload-Post-Key header")

    url = "https://api.upload-post.com/api/uploadposts/users"
    print(f"🔍 Fetching User ID from: {url}")
    headers = {"Authorization": f"Apikey {api_key}"}
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            resp = await client.get(url, headers=headers)
            if resp.status_code != 200:
                print(f"❌ Upload-Post User Fetch Error: {resp.text}")
                raise HTTPException(status_code=resp.status_code, detail=f"Failed to fetch user: {resp.text}")
            
            data = resp.json()
            print(f"🔍 Upload-Post User Response: {data}")
            
            user_id = None
            # The structure is {'success': True, 'profiles': [{'username': '...'}, ...]}
            profiles_list = []
            if isinstance(data, dict):
                 raw_profiles = data.get('profiles', [])
                 if isinstance(raw_profiles, list):
                     for p in raw_profiles:
                         username = p.get('username')
                         if username:
                             # Determine connected platforms
                             socials = p.get('social_accounts', {})
                             connected = []
                             # Check typical platforms
                             for platform in ['tiktok', 'instagram', 'youtube']:
                                 account_info = socials.get(platform)
                                 # If it's a dict and typically has data, or just not empty string
                                 if isinstance(account_info, dict):
                                     connected.append(platform)
                             
                             profiles_list.append({
                                 "username": username,
                                 "connected": connected
                             })
            
            # Managed users must only ever see their own profile.
            if forced_profile is not None:
                profiles_list = [p for p in profiles_list if p.get("username") == forced_profile]

            if not profiles_list:
                # Fallback if no profiles found
                return {"profiles": [], "error": "No profiles found"}

            return {"profiles": profiles_list}
            
            
        except Exception as e:
             raise HTTPException(status_code=500, detail=str(e))


# --- Social analytics (thin proxies over Upload-Post) ---
# Read-only mirrors of the posting flow above: managed users are locked to their
# own profile (the body/query profile is ignored), BYOK callers bring their own
# key and pick the profile with ?user=.

# Separate bucket from _probe_times: analytics polling must not eat into the
# metering-probe allowance, and vice versa. Protects the managed Upload-Post
# key's vendor rate limits from a runaway polling loop.
_analytics_times: dict = {}  # user_id -> [monotonic timestamps]
ANALYTICS_PER_HOUR = 60


def _check_analytics_rate(user_id):
    now = time.monotonic()
    times = _analytics_times.setdefault(str(user_id), [])
    times[:] = [t for t in times if now - t < 3600]
    if len(times) >= ANALYTICS_PER_HOUR:
        raise HTTPException(status_code=429,
                            detail="Too many analytics requests this hour. Please slow down.")
    times.append(now)


async def _social_analytics_auth(request: Request, byok_profile: Optional[str]):
    api_key, forced_profile = await resolve_upload_post(request, None)
    if not api_key:
        if BILLING_ENABLED:
            # Signed-in free user (or no auth at all): social posting is
            # paid-only in cloud, so there are no posts to measure either.
            raise HTTPException(status_code=402, detail={
                "error": "no_plan",
                "message": "Social analytics needs an active plan.",
            })
        raise HTTPException(status_code=400, detail="Missing X-Upload-Post-Key header")
    if forced_profile:
        user = await _user_from_request(request)
        if user:
            _check_analytics_rate(user.id)
    return api_key, resolve_post_profile(forced_profile, byok_profile)


async def _upload_post_get(api_key: str, url: str, params: dict):
    headers = {"Authorization": f"Apikey {api_key}"}
    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.get(url, headers=headers, params=params)
    if resp.status_code != 200:
        raise HTTPException(status_code=resp.status_code, detail=f"Vendor API Error: {resp.text}")
    return resp.json()


@app.get("/api/social/analytics")
async def social_profile_analytics(
    request: Request,
    platforms: str = "tiktok,instagram,youtube",
    user: Optional[str] = None,
):
    """Aggregated profile analytics: followers, views, engagement per platform."""
    api_key, profile = await _social_analytics_auth(request, user)
    return await _upload_post_get(
        api_key,
        f"https://api.upload-post.com/api/analytics/{profile}",
        {"platforms": platforms},
    )


@app.get("/api/social/analytics/posts")
async def social_post_analytics(
    request: Request,
    platform: Optional[str] = None,
    limit: Optional[int] = None,
    cursor: Optional[str] = None,
    since: Optional[str] = None,
    until: Optional[str] = None,
    user: Optional[str] = None,
):
    """Per-post metrics for the profile's published posts (Upload-Post cache)."""
    api_key, profile = await _social_analytics_auth(request, user)
    params = {"user": profile}
    for key, value in (("platform", platform), ("limit", limit),
                       ("cursor", cursor), ("since", since), ("until", until)):
        if value is not None:
            params[key] = value
    return await _upload_post_get(
        api_key,
        "https://api.upload-post.com/api/uploadposts/post-analytics/cached",
        params,
    )


_PERIOD_DAYS = {"last_day": 1, "last_week": 7, "last_month": 30,
                "last_3months": 90, "last_year": 365}


def _post_row_views(row: dict) -> float:
    metrics = row.get("post_metrics") or row.get("metrics") or row
    for key in ("views", "impressions", "plays"):
        value = metrics.get(key)
        if value is not None:
            try:
                return float(value)
            except (TypeError, ValueError):
                return 0.0
    return 0.0


@app.get("/api/social/analytics/impressions")
async def social_total_impressions(
    request: Request,
    period: Optional[str] = None,     # last_day | last_week | last_month | last_3months | last_year
    start_date: Optional[str] = None,  # YYYY-MM-DD
    end_date: Optional[str] = None,
    platform: Optional[str] = None,
    breakdown: Optional[bool] = None,
    user: Optional[str] = None,
):
    """Total impressions for the profile over a window.

    Computed by aggregating the profile-scoped post cache instead of proxying
    Upload-Post's /total-impressions: that endpoint echoes the requested
    profile but returns account-wide numbers (observed 2026-08-21 — a profile
    with zero posts got 85K Instagram impressions), which for managed users
    would leak other tenants' aggregates. The cache endpoint IS scoped by
    ?user=, so summing it is both correct and cheap.
    """
    api_key, profile = await _social_analytics_auth(request, user)

    days = _PERIOD_DAYS.get(period or "", 30)
    since = start_date or (datetime.now(timezone.utc) - timedelta(days=days)).strftime("%Y-%m-%d")
    params = {"user": profile, "since": since, "limit": 200}
    if end_date:
        params["until"] = end_date
    if platform:
        params["platform"] = platform

    total = 0.0
    per_platform: dict = {}
    for _page in range(5):  # 1000 posts is far beyond any real profile window
        data = await _upload_post_get(
            api_key,
            "https://api.upload-post.com/api/uploadposts/post-analytics/cached",
            params,
        )
        rows = data.get("posts") or data.get("data") or data.get("items") or []
        for row in rows:
            if not isinstance(row, dict):
                continue
            views = _post_row_views(row)
            total += views
            name = row.get("platform")
            if name:
                per_platform[name] = per_platform.get(name, 0) + views
        cursor = data.get("next_cursor")
        if not cursor or not data.get("has_more"):
            break
        params["cursor"] = cursor

    result = {
        "profile_username": profile,
        "total_impressions": round(total),
        "per_platform": {k: round(v) for k, v in per_platform.items()},
    }
    return result


async def _scheduled_posts_for(api_key: str, profile: str) -> list:
    """The caller's pending scheduled posts.

    Upload-Post's GET /uploadposts/schedule takes no profile filter and returns
    everything the *account* has pending — with the managed key that is every
    OpenShorts user's queue, so the filter below is what keeps one tenant from
    seeing (or cancelling) another's. Same class of bug as the impressions
    endpoint; do not "simplify" it away.
    """
    data = await _upload_post_get(
        api_key, "https://api.upload-post.com/api/uploadposts/schedule", {})
    rows = data.get("scheduled_posts") or data.get("data") or []
    return [r for r in rows
            if isinstance(r, dict) and r.get("profile_username") == profile]


@app.get("/api/social/scheduled")
async def social_scheduled(request: Request, user: Optional[str] = None):
    """Pending scheduled posts for the caller's profile, soonest first."""
    api_key, profile = await _social_analytics_auth(request, user)
    rows = await _scheduled_posts_for(api_key, profile)
    rows.sort(key=lambda r: r.get("scheduled_date") or "")
    return {"profile_username": profile, "scheduled_posts": rows}


@app.delete("/api/social/scheduled/{job_id}")
async def social_cancel_scheduled(job_id: str, request: Request, user: Optional[str] = None):
    """Cancel one pending scheduled post, if it belongs to the caller."""
    api_key, profile = await _social_analytics_auth(request, user)
    rows = await _scheduled_posts_for(api_key, profile)
    if not any(r.get("job_id") == job_id for r in rows):
        # 404 rather than 403: never confirm that someone else's job exists.
        raise HTTPException(status_code=404, detail="Scheduled post not found")
    headers = {"Authorization": f"Apikey {api_key}"}
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.delete(
            f"https://api.upload-post.com/api/uploadposts/schedule/{job_id}",
            headers=headers)
    if resp.status_code not in (200, 202, 204):
        raise HTTPException(status_code=resp.status_code,
                            detail=f"Vendor API Error: {resp.text}")
    return {"success": True, "job_id": job_id}


# --- Thumbnail Studio Endpoints ---

@app.post("/api/thumbnail/upload")
async def thumbnail_upload(
    request: Request,
    file: Optional[UploadFile] = File(None),
    url: Optional[str] = Form(None),
):
    """Upload video and start background Whisper transcription immediately."""
    await require_managed_entitlement(request)
    if not url and not file:
        raise HTTPException(status_code=400, detail="Must provide URL or File")

    session_id = str(uuid.uuid4())
    transcript_event = asyncio.Event()

    # Save file if uploaded directly. basename() stops a "../../x" filename from
    # escaping UPLOAD_DIR; the chunked read caps memory so a huge body can't OOM.
    video_path = None
    if file:
        safe_name = os.path.basename(file.filename or "upload") or "upload"
        video_path = os.path.join(UPLOAD_DIR, f"thumb_{session_id}_{safe_name}")
        size = 0
        limit_bytes = MAX_FILE_SIZE_MB * 1024 * 1024
        with open(video_path, "wb") as buffer:
            while chunk := await file.read(1024 * 1024):
                size += len(chunk)
                if size > limit_bytes:
                    os.remove(video_path)
                    raise HTTPException(status_code=413, detail=f"File too large. Max size {MAX_FILE_SIZE_MB}MB")
                buffer.write(chunk)

    # Meter a fixed guard cost for the background download + Whisper transcription
    # so an entitled user can't loop it for free. Settled when the job finishes.
    transcribe_minutes = _cloud_config.TRANSCRIBE_MINUTES if BILLING_ENABLED else 0
    reservation_id = await reserve_managed_action(
        request, transcribe_minutes, session_id, "thumbnail_transcribe")

    # Initialize session
    thumbnail_sessions[session_id] = {
        "user_id": await _owner_id(request),
        "video_path": video_path,
        "transcript_event": transcript_event,
        "transcript_ready": False,
        "transcript": None,
        "transcript_segments": [],
        "video_duration": 0,
        "language": "en",
        "context": "",
        "titles": [],
        "conversation": [],
        "_url": url,  # Store URL for deferred download
    }

    async def run_background_whisper():
        try:
            vpath = video_path
            # Download YouTube video if URL was provided
            if not vpath and url:
                from main import download_youtube_video
                loop = asyncio.get_event_loop()
                vpath, _ = await loop.run_in_executor(None, download_youtube_video, url, UPLOAD_DIR)
                thumbnail_sessions[session_id]["video_path"] = vpath

            from main import transcribe_video
            loop = asyncio.get_event_loop()
            transcript = await loop.run_in_executor(None, transcribe_video, vpath)
            segments = transcript.get("segments", [])
            duration = segments[-1]["end"] if segments else 0

            thumbnail_sessions[session_id].update({
                "transcript_ready": True,
                "transcript": transcript,
                "transcript_segments": segments,
                "video_duration": duration,
                "language": transcript.get("language", "en"),
            })
            print(f"✅ [Thumbnail] Background Whisper complete for session {session_id}")
            if reservation_id:
                await _metering.commit_reservation(reservation_id)
        except Exception as e:
            print(f"❌ [Thumbnail] Background Whisper failed: {e}")
            thumbnail_sessions[session_id]["transcript_error"] = str(e)
            if reservation_id:
                await _metering.release_reservation(reservation_id)
        finally:
            transcript_event.set()

    asyncio.create_task(run_background_whisper())

    return {"session_id": session_id}


@app.post("/api/thumbnail/analyze")
async def thumbnail_analyze(
    request: Request,
    file: Optional[UploadFile] = File(None),
    url: Optional[str] = Form(None),
    session_id: Optional[str] = Form(None),
    x_gemini_key: Optional[str] = Header(None, alias="X-Gemini-Key")
):
    """Analyze a video and suggest viral YouTube titles."""
    api_key = await resolve_gemini(request)
    if not api_key:
        raise gemini_missing_error()

    pre_transcript = None

    # Check for pre-existing session with background Whisper
    if session_id and session_id in thumbnail_sessions:
        session = thumbnail_sessions[session_id]
        await _assert_job_owner(request, session)

        # Wait for background Whisper to complete
        transcript_event = session.get("transcript_event")
        if transcript_event:
            print(f"⏳ [Thumbnail] Waiting for background Whisper to finish...")
            await transcript_event.wait()

        if session.get("transcript_error"):
            raise HTTPException(status_code=500, detail=f"Transcription failed: {session['transcript_error']}")

        video_path = session["video_path"]
        if not video_path or not os.path.exists(video_path):
            raise HTTPException(status_code=404, detail="Video file not found in session")

        if session.get("transcript_ready"):
            pre_transcript = session["transcript"]
    else:
        # No pre-existing session — need file or URL
        if not url and not file:
            raise HTTPException(status_code=400, detail="Must provide URL, File, or session_id")

        session_id = str(uuid.uuid4())

        if url:
            from main import download_youtube_video
            video_path, _ = download_youtube_video(url, UPLOAD_DIR)
        else:
            safe_name = os.path.basename(file.filename or "upload") or "upload"
            video_path = os.path.join(UPLOAD_DIR, f"thumb_{session_id}_{safe_name}")
            size = 0
            limit_bytes = MAX_FILE_SIZE_MB * 1024 * 1024
            with open(video_path, "wb") as buffer:
                while chunk := await file.read(1024 * 1024):
                    size += len(chunk)
                    if size > limit_bytes:
                        os.remove(video_path)
                        raise HTTPException(status_code=413, detail=f"File too large. Max size {MAX_FILE_SIZE_MB}MB")
                    buffer.write(chunk)

    # Meter the managed Gemini analysis (no-op for self-host).
    analyze_minutes = _cloud_config.MANAGED_ANALYSIS_MINUTES if BILLING_ENABLED else 0
    reservation_id = await reserve_managed_action(request, analyze_minutes, session_id, "thumbnail_analyze")

    try:
        # Run analysis in thread pool (skips Whisper if pre_transcript is available)
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, analyze_video_for_titles, api_key, video_path, pre_transcript)

        # Store/update session context
        if session_id not in thumbnail_sessions:
            thumbnail_sessions[session_id] = {"user_id": await _owner_id(request)}

        thumbnail_sessions[session_id].update({
            "context": result.get("transcript_summary", ""),
            "titles": result.get("titles", []),
            "thumbnail_texts": result.get("thumbnail_texts", []),
            "language": result.get("language", "en"),
            "conversation": thumbnail_sessions[session_id].get("conversation", []),
            "video_path": video_path,
            "transcript_segments": result.get("segments", []),
            "video_duration": result.get("video_duration", 0)
        })

        if reservation_id:
            await _metering.commit_reservation(reservation_id)
        return {
            "session_id": session_id,
            "titles": result.get("titles", []),
            "thumbnail_texts": result.get("thumbnail_texts", []),
            "context": result.get("transcript_summary", ""),
            "language": result.get("language", "en"),
            "recommended": result.get("recommended", [])
        }

    except Exception as e:
        if reservation_id:
            await _metering.release_reservation(reservation_id)
        print(f"❌ Thumbnail Analyze Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


class ThumbnailTitlesRequest(BaseModel):
    session_id: Optional[str] = None
    message: Optional[str] = None
    title: Optional[str] = None

@app.post("/api/thumbnail/titles")
async def thumbnail_titles(
    req: ThumbnailTitlesRequest,
    request: Request,
):
    """Refine title suggestions or accept a manual title."""
    api_key = await resolve_gemini(request)
    if not api_key:
        raise gemini_missing_error()

    # Manual title mode - just create a session with the user's title
    if req.title:
        session_id = req.session_id or str(uuid.uuid4())
        if session_id not in thumbnail_sessions:
            thumbnail_sessions[session_id] = {
                "user_id": await _owner_id(request),
                "context": "",
                "titles": [req.title],
                "language": "en",
                "conversation": []
            }
        else:
            await _assert_job_owner(request, thumbnail_sessions[session_id])
        return {"session_id": session_id, "titles": [req.title]}

    # Refinement mode
    if not req.session_id or req.session_id not in thumbnail_sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    if not req.message:
        raise HTTPException(status_code=400, detail="Must provide message or title")

    session = thumbnail_sessions[req.session_id]
    await _assert_job_owner(request, session)

    # Add user message to conversation history
    session["conversation"].append({"role": "user", "content": req.message})

    try:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            refine_titles,
            api_key,
            session["context"],
            req.message,
            session["conversation"]
        )

        new_titles = result.get("titles", [])
        session["titles"] = new_titles
        session["thumbnail_texts"] = result.get("thumbnail_texts", [])
        # The user may have asked for another language ("in English"): the
        # thumbnail text must follow the titles, not the transcript.
        if result.get("language"):
            session["language"] = result["language"]
        session["conversation"].append({"role": "assistant", "content": json.dumps(new_titles)})

        return {"titles": new_titles, "thumbnail_texts": session["thumbnail_texts"],
                "language": session["language"]}

    except Exception as e:
        print(f"❌ Thumbnail Titles Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/thumbnail/generate")
async def thumbnail_generate(
    request: Request,
    session_id: str = Form(...),
    title: str = Form(...),
    extra_prompt: str = Form(""),
    count: int = Form(3),
    burn_text: bool = Form(True),
    frame: str = Form(""),
    face: Optional[UploadFile] = File(None),
    background: Optional[UploadFile] = File(None),
):
    """Generate YouTube thumbnails with Gemini image generation.

    burn_text: the image model paints the scene with clean negative space and
    PIL sets the hook text (Anton, black stroke). Off = the model renders the
    text itself, which is prettier when it works and misspelled when it does
    not. frame: url of a frame from /api/thumbnail/frames to use as the
    person reference when no face photo is uploaded."""
    api_key = await resolve_gemini(request)
    if not api_key:
        raise gemini_missing_error()

    # Image generation is the one expensive managed Gemini call — paid plans only.
    if BILLING_ENABLED:
        user = await _user_from_request(request)
        if user is not None and user.plan == "free":
            raise HTTPException(status_code=403, detail={
                "error": "plan_required",
                "message": "AI thumbnail generation is available on paid plans.",
            })

    # Clamp count
    count = min(max(1, count), 6)

    # Gemini image generation is the expensive managed call — meter it against the
    # plan quota (a batch ≈ THUMBNAIL_MINUTES). No-op for BYOK / self-host.
    thumb_minutes = _cloud_config.THUMBNAIL_MINUTES if BILLING_ENABLED else 0
    reservation_id = await reserve_managed_action(request, thumb_minutes, session_id, "thumbnail")

    # Save optional uploaded images. basename() on the session id and filenames
    # keeps everything inside UPLOAD_DIR (no "../" escape from client input).
    face_path = None
    bg_path = None
    safe_session = os.path.basename(session_id) or "session"
    thumb_upload_dir = os.path.join(UPLOAD_DIR, f"thumb_{safe_session}")
    os.makedirs(thumb_upload_dir, exist_ok=True)

    try:
        if face and face.filename:
            face_name = os.path.basename(face.filename)
            face_path = os.path.join(thumb_upload_dir, f"face_{face_name}")
            with open(face_path, "wb") as f:
                f.write(await face.read())

        if background and background.filename:
            bg_name = os.path.basename(background.filename)
            bg_path = os.path.join(thumb_upload_dir, f"bg_{bg_name}")
            with open(bg_path, "wb") as f:
                f.write(await background.read())

        # Session context: transcript summary, the thumbnail text the critic
        # paired with this title, the language, and the chosen video frame.
        video_context, text_hint, language, frame_reference = "", "", "en", None
        session = thumbnail_sessions.get(session_id)
        if session:
            video_context = session.get("context", "")
            language = session.get("language", "en")
            titles = session.get("titles", [])
            texts = session.get("thumbnail_texts", [])
            if title in titles and titles.index(title) < len(texts):
                text_hint = texts[titles.index(title)]
            if frame:
                frame_reference = next((f for f in session.get("frames", []) if f["url"] == frame), None)

        loop = asyncio.get_event_loop()
        thumbnails = await loop.run_in_executor(
            None,
            functools.partial(
                generate_thumbnail, api_key, title, session_id, face_path, bg_path,
                extra_prompt, count, video_context, burn_text=burn_text,
                thumbnail_text_hint=text_hint, language=language,
                frame_reference=frame_reference),
        )

        if not thumbnails:
            raise HTTPException(status_code=500, detail="Thumbnail generation failed. Please check your Gemini API key has access to image generation (gemini-3.1-flash-image-preview model).")

        # Success — charge the reserved minutes.
        if reservation_id:
            await _metering.commit_reservation(reservation_id)
        return {"thumbnails": thumbnails}

    except HTTPException:
        if reservation_id:
            await _metering.release_reservation(reservation_id)
        raise
    except Exception as e:
        if reservation_id:
            await _metering.release_reservation(reservation_id)
        print(f"❌ Thumbnail Generate Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/thumbnail/frames/{session_id}")
async def thumbnail_frames(session_id: str, request: Request):
    """Frames of the session's video with a large, sharp face, as candidates
    for the thumbnail's person reference. Computed once and cached."""
    session = thumbnail_sessions.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    await _assert_job_owner(request, session)
    if "frames" in session:
        return {"frames": session["frames"]}

    # A URL session downloads in the background before transcribing; the
    # event fires after both, so waiting on it means the file is on disk.
    transcript_event = session.get("transcript_event")
    if transcript_event:
        await transcript_event.wait()
    video_path = session.get("video_path")
    if not video_path or not os.path.exists(video_path):
        return {"frames": []}

    lock = session.setdefault("_frames_lock", asyncio.Lock())
    async with lock:
        if "frames" not in session:
            loop = asyncio.get_event_loop()
            try:
                frames = await loop.run_in_executor(None, extract_face_frames, video_path, session_id)
            except Exception as e:
                print(f"❌ [Thumbnail] Frame extraction failed: {e}")
                frames = []
            session["frames"] = frames
    return {"frames": session["frames"]}


class ThumbnailDescribeRequest(BaseModel):
    session_id: str
    title: str

@app.post("/api/thumbnail/describe")
async def thumbnail_describe(
    req: ThumbnailDescribeRequest,
    request: Request,
):
    """Generate a YouTube description with chapters from the transcript."""
    api_key = await resolve_gemini(request)
    if not api_key:
        raise gemini_missing_error()

    if req.session_id not in thumbnail_sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    session = thumbnail_sessions[req.session_id]
    await _assert_job_owner(request, session)
    segments = session.get("transcript_segments", [])
    if not segments:
        raise HTTPException(status_code=400, detail="No transcript segments available. Please analyze a video first.")

    try:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            generate_youtube_description,
            api_key,
            req.title,
            segments,
            session.get("language", "en"),
            session.get("video_duration", 0)
        )
        return {"description": result.get("description", "")}

    except Exception as e:
        print(f"❌ Thumbnail Describe Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/thumbnail/publish")
async def thumbnail_publish(
    request: Request,
    background_tasks: BackgroundTasks,
    session_id: str = Form(...),
    title: str = Form(...),
    description: str = Form(...),
    thumbnail_url: str = Form(...),
    api_key: Optional[str] = Form(None),   # BYOK; ignored for managed users
    user_id: Optional[str] = Form(None),   # BYOK profile; ignored for managed users
):
    """Kick off a background upload to YouTube via Upload-Post and return immediately."""
    if session_id not in thumbnail_sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    # Managed users: server key + forced own profile; body fields ignored.
    upload_key, forced_profile = await resolve_upload_post(request, api_key)
    if not upload_key:
        raise HTTPException(status_code=400, detail="Missing Upload-Post API key")
    post_user = forced_profile or user_id
    if not post_user:
        raise HTTPException(status_code=400, detail="Missing Upload-Post user profile")

    session = thumbnail_sessions[session_id]
    await _assert_job_owner(request, session)
    video_path = session.get("video_path")
    if not video_path or not os.path.exists(video_path):
        raise HTTPException(status_code=404, detail="Original video file not found")

    # Resolve thumbnail path from URL — sanitize against path traversal so a
    # crafted thumbnail_url (e.g. "thumbnails/../../.env") can't read server
    # files and exfiltrate them via the Upload-Post multipart body.
    thumb_relative = thumbnail_url.lstrip("/")
    if thumb_relative.startswith("thumbnails/"):
        thumb_path = _safe_under(OUTPUT_DIR, thumb_relative)
    else:
        thumb_path = _safe_under(THUMBNAILS_DIR, thumb_relative)

    if not thumb_path:
        raise HTTPException(status_code=400, detail="Invalid thumbnail path")
    if not os.path.exists(thumb_path):
        raise HTTPException(status_code=404, detail="Thumbnail file not found")

    # Generate a unique ID for this publish job so the frontend can poll
    publish_id = str(uuid.uuid4())
    publish_jobs[publish_id] = {"status": "uploading", "result": None, "error": None}

    def do_upload():
        """Runs in a thread via BackgroundTasks — does the actual multipart upload."""
        try:
            upload_url = "https://api.upload-post.com/api/upload"
            headers = {"Authorization": f"Apikey {upload_key}"}
            data_payload = {
                "user": post_user,
                "platform[]": ["youtube"],
                "title": title,          # required base field (fallback)
                "async_upload": "true",
                "youtube_title": title,
                "youtube_description": description,
                "privacyStatus": "public",
            }
            video_filename = os.path.basename(video_path)
            thumb_filename = os.path.basename(thumb_path)

            print(f"📡 [Thumbnail] Publishing to YouTube via Upload-Post... (publish_id={publish_id})")
            with open(video_path, "rb") as vf, open(thumb_path, "rb") as tf:
                files = {
                    "video": (video_filename, vf.read(), "video/mp4"),
                    "thumbnail": (thumb_filename, tf.read(), "image/jpeg"),
                }

            # Use a long timeout — video uploads can take several minutes
            with httpx.Client(timeout=600.0) as client:
                response = client.post(upload_url, headers=headers, data=data_payload, files=files)

            if response.status_code not in [200, 201, 202]:
                err = f"Upload-Post API Error ({response.status_code}): {response.text}"
                print(f"❌ {err}")
                publish_jobs[publish_id]["status"] = "failed"
                publish_jobs[publish_id]["error"] = err
            else:
                print(f"✅ [Thumbnail] Published successfully (publish_id={publish_id})")
                publish_jobs[publish_id]["status"] = "done"
                publish_jobs[publish_id]["result"] = response.json()

        except Exception as e:
            err = str(e)
            print(f"❌ Thumbnail Publish Background Error: {err}")
            publish_jobs[publish_id]["status"] = "failed"
            publish_jobs[publish_id]["error"] = err

    background_tasks.add_task(do_upload)
    return {"publish_id": publish_id, "status": "uploading"}


@app.get("/api/thumbnail/publish/status/{publish_id}")
async def thumbnail_publish_status(publish_id: str):
    """Poll the status of a background publish job."""
    if publish_id not in publish_jobs:
        raise HTTPException(status_code=404, detail="Publish job not found")
    return publish_jobs[publish_id]


# @app.get("/api/gallery/clips")
# async def get_gallery_clips(limit: int = 20, offset: int = 0, refresh: bool = False):
#     """
#     Fetch clips from S3 for the gallery with pagination.
#
#     Args:
#         limit: Number of clips to return (default 20, max 100)
#         offset: Starting position for pagination
#         refresh: Force refresh cache
#     """
#     try:
#         # Clamp limit to reasonable values
#         limit = min(max(1, limit), 100)
#
#         # Get clips (uses cache internally)
#         all_clips = list_all_clips(limit=limit + offset, force_refresh=refresh)
#
#         # Apply offset for pagination
#         clips = all_clips[offset:offset + limit]
#
#         return {
#             "clips": clips,
#             "total": len(all_clips),
#             "limit": limit,
#             "offset": offset,
#             "has_more": len(all_clips) > offset + limit
#         }
#     except Exception as e:
#         print(f"❌ Gallery Error: {e}")
#         raise HTTPException(status_code=500, detail=str(e))


# ═══════════════════════════════════════════════════════════════════════
# SaaSShorts: AI UGC Video Generator for SaaS Products
# ═══════════════════════════════════════════════════════════════════════

from saasshorts import (
    scrape_website,
    research_saas_online,
    analyze_saas,
    generate_scripts,
    generate_full_video,
    generate_actor_images,
    get_elevenlabs_voices,
    DEFAULT_VOICES,
)

# State for SaaSShorts jobs (separate from video processing jobs)
saas_jobs: Dict[str, Dict] = {}


class SaaSAnalyzeRequest(BaseModel):
    url: Optional[str] = None
    description: Optional[str] = None  # Manual product/business description
    num_scripts: int = 3
    style: str = "ugc"
    language: str = "en"
    actor_gender: str = "female"


@app.post("/api/saasshorts/analyze")
async def saasshorts_analyze(
    req: SaaSAnalyzeRequest,
    request: Request,
):
    """Analyze a URL or manual description and generate video scripts."""
    gemini_key = await resolve_gemini(request)
    if not gemini_key:
        raise gemini_missing_error()

    if not req.url and not req.description:
        raise HTTPException(status_code=400, detail="Provide a URL or a product description")

    # Meter the managed Gemini research/analysis (no-op for self-host).
    saas_minutes = _cloud_config.MANAGED_ANALYSIS_MINUTES if BILLING_ENABLED else 0
    reservation_id = await reserve_managed_action(request, saas_minutes, "saasshorts", "saasshorts_analyze")

    try:
        loop = asyncio.get_event_loop()

        def run_analysis():
            web_research = None

            if req.url and req.url.strip():
                # URL provided: full scrape + research pipeline
                scraped = scrape_website(req.url)
                web_research = research_saas_online(req.url, gemini_key)
                analysis = analyze_saas(scraped, gemini_key, web_research=web_research)
            else:
                # Manual description: build analysis from description
                analysis = {
                    "product_name": req.description.split(",")[0].strip()[:60] if req.description else "Product",
                    "description": req.description,
                    "value_proposition": req.description,
                    "target_audience": "general audience",
                    "key_features": [req.description],
                    "pain_points": [],
                    "tone": "casual and authentic",
                }

            scripts = generate_scripts(analysis, gemini_key, req.num_scripts, req.style, req.language, req.actor_gender)
            return {
                "analysis": analysis,
                "scripts": scripts,
                "web_research": web_research,
            }

        result = await loop.run_in_executor(None, run_analysis)
        if reservation_id:
            await _metering.commit_reservation(reservation_id)
        return result

    except Exception as e:
        if reservation_id:
            await _metering.release_reservation(reservation_id)
        raise HTTPException(status_code=500, detail=str(e))


class SaaSActorRequest(BaseModel):
    actor_description: str
    num_options: int = 3
    product_description: Optional[str] = None


@app.post("/api/saasshorts/actor-upload")
async def saasshorts_actor_upload(request: Request, file: UploadFile = File(...)):
    """Upload a custom actor image (stored locally only, not S3)."""
    # SaaSShorts is part of the paid product — require entitlement in cloud mode
    # (no-op for self-host) so anonymous callers can't drive server work.
    await require_managed_entitlement(request)
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    try:
        # Bounded read: an actor image has no business being large. Cap it so an
        # anonymous caller can't stream a multi-GB body into RAM (OOM DoS).
        ACTOR_IMAGE_MAX_BYTES = 25 * 1024 * 1024  # 25 MB
        content = await file.read(ACTOR_IMAGE_MAX_BYTES + 1)
        if len(content) > ACTOR_IMAGE_MAX_BYTES:
            raise HTTPException(status_code=413, detail="Image too large (max 25 MB)")

        # Validate minimum size
        if len(content) < 1000:
            raise HTTPException(status_code=400, detail="File too small to be a valid image")

        upload_id = uuid.uuid4().hex[:8]
        upload_dir = os.path.join(OUTPUT_DIR, "actor_uploads")
        os.makedirs(upload_dir, exist_ok=True)
        filename = f"custom_{upload_id}.png"
        file_path = os.path.join(upload_dir, filename)

        with open(file_path, "wb") as f:
            f.write(content)

        return {"url": f"/videos/actor_uploads/{filename}"}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/saasshorts/actor-options")
async def saasshorts_actor_options(
    req: SaaSActorRequest,
    request: Request,
    x_fal_key: Optional[str] = Header(None, alias="X-Fal-Key"),
):
    """Generate multiple actor image options for the user to choose from."""
    await require_managed_entitlement(request)
    fal_key = x_fal_key
    if not fal_key:
        raise HTTPException(status_code=400, detail="Missing fal.ai API Key")

    try:
        job_id = str(uuid.uuid4())
        out_dir = os.path.join(OUTPUT_DIR, f"saas_actors_{job_id}")
        os.makedirs(out_dir, exist_ok=True)

        loop = asyncio.get_running_loop()
        import functools
        paths = await loop.run_in_executor(
            None,
            functools.partial(
                generate_actor_images,
                req.actor_description, fal_key, out_dir, "actor", req.num_options,
                product_description=req.product_description,
            ),
        )

        # Upload each actor image to public S3 with description
        desc = req.actor_description
        if req.product_description:
            desc += f" (holding {req.product_description})"
        urls = []
        for p in paths:
            s3_url = upload_actor_to_s3(p, description=desc)
            if s3_url:
                urls.append(s3_url)
            else:
                # Fallback to local URL if S3 fails
                urls.append(f"/videos/saas_actors_{job_id}/{os.path.basename(p)}")

        return {"images": urls}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/saasshorts/gallery")
async def saasshorts_video_gallery(limit: int = 50):
    """List all UGC videos from the public gallery."""
    try:
        loop = asyncio.get_running_loop()
        videos = await loop.run_in_executor(None, list_video_gallery, limit)
        return {"videos": videos, "total": len(videos)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class SaaSPostRequest(BaseModel):
    job_id: str
    api_key: Optional[str] = None  # BYOK; ignored for managed users
    user_id: Optional[str] = None  # BYOK profile; ignored for managed users
    platforms: List[str]
    title: Optional[str] = None
    description: Optional[str] = None
    scheduled_date: Optional[str] = None
    timezone: Optional[str] = "UTC"


@app.post("/api/saasshorts/post")
async def saasshorts_post_to_socials(req: SaaSPostRequest, request: Request):
    """Post an AI Shorts video to social media via Upload-Post."""
    if req.job_id not in saas_jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    upload_key, forced_profile = await resolve_upload_post(request, req.api_key)
    if not upload_key:
        raise HTTPException(status_code=400, detail="Missing Upload-Post API key")
    post_user = resolve_post_profile(forced_profile, req.user_id)

    job = saas_jobs[req.job_id]
    await _assert_job_owner(request, job)
    result = job.get("result")
    if not result or not result.get("video_url"):
        raise HTTPException(status_code=400, detail="No video available for this job")

    try:
        # Resolve video file path
        video_url = result["video_url"]  # e.g. /videos/saas_xxx/slug_final.mp4
        rel_path = video_url.replace("/videos/", "")
        file_path = os.path.join(OUTPUT_DIR, rel_path)

        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail=f"Video file not found")

        script = result.get("script", {})
        final_title = req.title or script.get("title", "AI Short")
        final_description = req.description or script.get("caption", "")
        if not final_description:
            final_description = script.get("full_narration", "Check this out!")

        url = "https://api.upload-post.com/api/upload"
        headers = {"Authorization": f"Apikey {upload_key}"}

        data_payload = {
            "user": post_user,
            "title": final_title,
            "platform[]": req.platforms,
            "async_upload": "true",
        }

        if req.scheduled_date:
            data_payload["scheduled_date"] = req.scheduled_date
            if req.timezone:
                data_payload["timezone"] = req.timezone

        if "tiktok" in req.platforms:
            data_payload["tiktok_title"] = final_description
            data_payload["post_mode"] = TIKTOK_POST_MODE
        if "instagram" in req.platforms:
            data_payload["instagram_title"] = final_description
            data_payload["media_type"] = "REELS"
        if "youtube" in req.platforms:
            data_payload["youtube_title"] = final_title
            data_payload["youtube_description"] = final_description
            data_payload["privacyStatus"] = "public"

        filename = os.path.basename(file_path)
        with open(file_path, "rb") as f:
            file_content = f.read()

        files = {"video": (filename, file_content, "video/mp4")}

        with httpx.Client(timeout=120.0) as client:
            print(f"📡 [AI Shorts] Sending to Upload-Post: {req.platforms}")
            response = client.post(url, headers=headers, data=data_payload, files=files)

        if response.status_code not in [200, 201, 202]:
            raise HTTPException(status_code=response.status_code, detail=f"Upload-Post Error: {response.text}")

        return response.json()

    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ [AI Shorts] Post Exception: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/gallery", response_class=HTMLResponse)
async def gallery_html_page():
    """SEO gallery page with all generated UGC videos."""
    import html as html_mod
    loop = asyncio.get_running_loop()
    videos = await loop.run_in_executor(None, list_video_gallery, 100)

    cards_html = ""
    ld_items = []
    for i, v in enumerate(videos):
        title = html_mod.escape(v.get("title", "Untitled"))
        video_url = v.get("video_url", "")
        actor_url = v.get("actor_url", "")
        video_id = v.get("video_id", "")
        duration = v.get("duration", 0)
        mode = v.get("video_mode", "")
        product = html_mod.escape(v.get("product_name", ""))
        caption = html_mod.escape(v.get("caption", "")[:120])

        mode_badge = '<span style="background:#22c55e;color:#000;padding:2px 8px;border-radius:9999px;font-size:10px;font-weight:700">LOW COST</span>' if mode == "lowcost" else '<span style="background:#8b5cf6;color:#fff;padding:2px 8px;border-radius:9999px;font-size:10px;font-weight:700">PREMIUM</span>'

        cards_html += f'''
        <a href="/video/{video_id}" style="text-decoration:none;color:inherit">
          <div style="background:#18181b;border-radius:16px;overflow:hidden;border:1px solid #27272a;transition:transform 0.2s" onmouseover="this.style.transform='scale(1.02)'" onmouseout="this.style.transform='scale(1)'">
            <div style="position:relative;aspect-ratio:9/16;background:#000">
              <video src="{video_url}" poster="{actor_url}" muted playsinline preload="metadata"
                     onmouseenter="this.play()" onmouseleave="this.pause();this.currentTime=0"
                     style="width:100%;height:100%;object-fit:cover"></video>
              <div style="position:absolute;top:8px;right:8px">{mode_badge}</div>
            </div>
            <div style="padding:12px">
              <h2 style="font-size:14px;font-weight:600;margin:0 0 4px 0;white-space:nowrap;overflow:hidden;text-overflow:ellipsis">{title}</h2>
              <p style="font-size:11px;color:#71717a;margin:0">{duration:.0f}s · {product}</p>
            </div>
          </div>
        </a>'''

        ld_items.append(f'{{"@type":"ListItem","position":{i+1},"url":"https://openshorts.app/video/{video_id}","name":"{title}"}}')

    ld_json = f'{{"@context":"https://schema.org","@type":"CollectionPage","name":"AI UGC Video Gallery","mainEntity":{{"@type":"ItemList","numberOfItems":{len(videos)},"itemListElement":[{",".join(ld_items)}]}}}}'

    return f'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>AI UGC Video Gallery | OpenShorts</title>
<meta name="description" content="Browse {len(videos)} AI-generated UGC marketing videos. Create viral TikTok and Instagram Reels for your SaaS product.">
<meta name="robots" content="index, follow">
<meta property="og:title" content="AI UGC Video Gallery | OpenShorts">
<meta property="og:type" content="website">
<meta property="og:description" content="Browse AI-generated UGC marketing videos for SaaS products.">
<script type="application/ld+json">{ld_json}</script>
<style>
*{{margin:0;padding:0;box-sizing:border-box}}
body{{background:#0a0a0c;color:#e4e4e7;font-family:-apple-system,BlinkMacSystemFont,sans-serif}}
.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(220px,1fr));gap:20px;padding:20px;max-width:1400px;margin:0 auto}}
nav{{padding:20px 40px;border-bottom:1px solid #27272a;display:flex;align-items:center;justify-content:space-between}}
h1{{font-size:28px;font-weight:700;padding:40px 20px 0;text-align:center}}
.subtitle{{text-align:center;color:#71717a;font-size:14px;padding:8px 20px 20px}}
.cta{{display:inline-block;background:#8b5cf6;color:#fff;padding:10px 24px;border-radius:12px;text-decoration:none;font-weight:600;font-size:14px}}
</style>
</head>
<body>
<nav><strong style="font-size:18px">OpenShorts</strong><a href="/" class="cta">Create Your Video</a></nav>
<h1>AI-Generated UGC Videos</h1>
<p class="subtitle">{len(videos)} videos generated · Low Cost & Premium modes</p>
<div class="grid">{cards_html}</div>
<div style="text-align:center;padding:40px"><a href="/" class="cta">Create Your Own UGC Video</a></div>
</body></html>'''


@app.get("/video/{video_id}", response_class=HTMLResponse)
async def video_html_page(video_id: str):
    """SEO individual video page with og:video meta tags."""
    import html as html_mod
    loop = asyncio.get_running_loop()
    videos = await loop.run_in_executor(None, list_video_gallery, 200)
    meta = next((v for v in videos if v.get("video_id") == video_id), None)
    if not meta:
        raise HTTPException(status_code=404, detail="Video not found")

    title = html_mod.escape(meta.get("title", "Untitled"))
    caption = html_mod.escape(meta.get("caption", ""))
    narration = html_mod.escape(meta.get("full_narration", ""))
    video_url = meta.get("video_url", "")
    actor_url = meta.get("actor_url", "")
    duration = meta.get("duration", 0)
    mode = meta.get("video_mode", "")
    product = html_mod.escape(meta.get("product_name", ""))
    product_url = html_mod.escape(meta.get("product_url", ""))
    language = meta.get("language", "en")
    hashtags = " ".join(meta.get("hashtags", []))
    cost = meta.get("cost_estimate", {}).get("total", 0)
    created = meta.get("created_at", "")
    actor_desc = html_mod.escape(meta.get("actor_description", ""))

    ld_json = f'{{"@context":"https://schema.org","@type":"VideoObject","name":"{title}","description":"{caption}","thumbnailUrl":"{actor_url}","contentUrl":"{video_url}","uploadDate":"{created}","duration":"PT{int(duration)}S","width":1080,"height":1920,"inLanguage":"{language}"}}'

    mode_label = "Low Cost" if mode == "lowcost" else "Premium"

    return f'''<!DOCTYPE html>
<html lang="{language}">
<head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{title} - AI UGC Video | OpenShorts</title>
<meta name="description" content="{caption} {hashtags}">
<meta property="og:type" content="video.other">
<meta property="og:title" content="{title}">
<meta property="og:description" content="{caption}">
<meta property="og:video" content="{video_url}">
<meta property="og:video:type" content="video/mp4">
<meta property="og:video:width" content="1080">
<meta property="og:video:height" content="1920">
<meta property="og:image" content="{actor_url}">
<meta name="twitter:card" content="player">
<meta name="twitter:title" content="{title}">
<meta name="twitter:image" content="{actor_url}">
<script type="application/ld+json">{ld_json}</script>
<style>
*{{margin:0;padding:0;box-sizing:border-box}}
body{{background:#0a0a0c;color:#e4e4e7;font-family:-apple-system,BlinkMacSystemFont,sans-serif}}
nav{{padding:20px 40px;border-bottom:1px solid #27272a;display:flex;align-items:center;gap:16px}}
nav a{{color:#a1a1aa;text-decoration:none;font-size:14px}}
.container{{max-width:1000px;margin:0 auto;padding:40px 20px;display:grid;grid-template-columns:1fr 1fr;gap:40px}}
@media(max-width:768px){{.container{{grid-template-columns:1fr}}}}
video{{width:100%;border-radius:16px;background:#000}}
h1{{font-size:22px;font-weight:700;margin-bottom:8px}}
.meta{{color:#71717a;font-size:13px;margin-bottom:20px}}
.section{{margin-bottom:20px}}
.section h2{{font-size:13px;color:#71717a;text-transform:uppercase;letter-spacing:1px;margin-bottom:6px}}
.section p{{font-size:14px;line-height:1.6}}
.badge{{display:inline-block;padding:3px 10px;border-radius:9999px;font-size:11px;font-weight:700}}
.cta{{display:inline-block;background:#8b5cf6;color:#fff;padding:10px 24px;border-radius:12px;text-decoration:none;font-weight:600;font-size:14px;margin-top:20px}}
</style>
</head>
<body>
<nav><strong>OpenShorts</strong><a href="/gallery">Gallery</a><span style="color:#3f3f46">›</span><span style="color:#e4e4e7;font-size:14px">{title}</span></nav>
<div class="container">
<div><video src="{video_url}" poster="{actor_url}" controls autoplay playsinline style="aspect-ratio:9/16;object-fit:cover"></video></div>
<div>
<h1>{title}</h1>
<p class="meta">{duration:.0f}s · {mode_label} · ${cost:.2f} · {product}</p>
<div class="section"><h2>Caption</h2><p>{caption}</p><p style="color:#8b5cf6;margin-top:4px">{hashtags}</p></div>
<div class="section"><h2>Script</h2><p>{narration}</p></div>
<div class="section"><h2>Actor</h2><p>{actor_desc}</p></div>
{f'<div class="section"><h2>Product</h2><p><a href="{product_url}" style="color:#8b5cf6" target="_blank">{product}</a></p></div>' if product_url else ''}
<a href="/gallery">← Back to Gallery</a>
<br><a href="/" class="cta">Create Your Own</a>
</div>
</div>
</body></html>'''


@app.get("/api/saasshorts/actor-gallery")
async def saasshorts_actor_gallery():
    """List all previously generated actor images from public S3."""
    try:
        loop = asyncio.get_running_loop()
        images = await loop.run_in_executor(None, list_actor_gallery)
        return {"images": images}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class SaaSGenerateRequest(BaseModel):
    script: dict
    voice_id: Optional[str] = None
    actor_description: Optional[str] = None
    selected_actor_url: Optional[str] = None  # Pre-selected actor image URL
    retry_job_id: Optional[str] = None
    video_mode: str = "lowcost"  # "lowcost" or "premium"
    # Publishing to the public /gallery is opt-in: generated videos carry the
    # user's product name, URL and full script.
    share_to_gallery: bool = False


@app.post("/api/saasshorts/generate")
async def saasshorts_generate(
    req: SaaSGenerateRequest,
    request: Request,
    x_fal_key: Optional[str] = Header(None, alias="X-Fal-Key"),
    x_elevenlabs_key: Optional[str] = Header(None, alias="X-ElevenLabs-Key"),
):
    """Generate a SaaS UGC video from a script. Returns a job_id for polling."""
    await require_managed_entitlement(request)
    fal_key = x_fal_key
    elevenlabs_key = x_elevenlabs_key

    if not fal_key:
        raise HTTPException(status_code=400, detail="Missing fal.ai API Key (X-Fal-Key header)")
    if not elevenlabs_key:
        raise HTTPException(status_code=400, detail="Missing ElevenLabs API Key (X-ElevenLabs-Key header)")

    # Support retry: reuse output_dir so cached assets (image, voice, head, broll) are kept
    reused = False
    if req.retry_job_id:
        # Check memory first, then disk. _safe_under() blocks a crafted
        # retry_job_id like "../../tmp/x" from escaping OUTPUT_DIR (the listdir
        # below deletes files and the pipeline writes here). A known in-memory
        # job keeps its trusted stored path.
        if req.retry_job_id in saas_jobs:
            await _assert_job_owner(request, saas_jobs[req.retry_job_id])
            old_dir = saas_jobs[req.retry_job_id]["output_dir"]
        else:
            old_dir = _safe_under(OUTPUT_DIR, f"saas_{req.retry_job_id}")

        if old_dir and os.path.isdir(old_dir):
            job_id = req.retry_job_id
            job_output_dir = old_dir
            reused = True
            # Clear the 0-byte final video so pipeline re-generates it
            for f in os.listdir(old_dir):
                fp = os.path.join(old_dir, f)
                if f.endswith("_final.mp4") and os.path.getsize(fp) == 0:
                    os.remove(fp)
            saas_jobs[job_id] = {
                "user_id": await _owner_id(request),
                "status": "processing",
                "logs": [f"Retrying job {job_id[:8]}... reusing cached assets from disk."],
                "result": None,
                "output_dir": job_output_dir,
            }

    if not reused:
        job_id = str(uuid.uuid4())
        job_output_dir = os.path.join(OUTPUT_DIR, f"saas_{job_id}")
        os.makedirs(job_output_dir, exist_ok=True)
        saas_jobs[job_id] = {
            "user_id": await _owner_id(request),
            "status": "processing",
            "logs": ["SaaSShorts job started."],
            "result": None,
            "output_dir": job_output_dir,
        }

    # If user selected a pre-generated actor, resolve it to a local path
    selected_actor_path = None
    if req.selected_actor_url:
        if req.selected_actor_url.startswith("http"):
            # Download from S3 public URL to job output dir
            import httpx
            from security_utils import assert_public_url
            try:
                # SSRF guard: block private / metadata hosts before fetching.
                safe_actor_url = assert_public_url(req.selected_actor_url)
                actor_local = os.path.join(job_output_dir, "selected_actor.png")
                with httpx.Client(timeout=30.0) as client:
                    resp = client.get(safe_actor_url)
                    if resp.status_code == 200:
                        with open(actor_local, "wb") as f:
                            f.write(resp.content)
                        selected_actor_path = actor_local
            except Exception:
                pass
        else:
            # Sanitize against traversal — the client controls selected_actor_url.
            src = _safe_under(OUTPUT_DIR, req.selected_actor_url.replace("/videos/", "").lstrip("/"))
            if src and os.path.exists(src):
                selected_actor_path = src

    config = {
        "fal_key": fal_key,
        "elevenlabs_key": elevenlabs_key,
        "voice_id": req.voice_id or "21m00Tcm4TlvDq8ikWAM",
        "actor_description": req.actor_description,
        "selected_actor_path": selected_actor_path,
        "video_mode": req.video_mode,
    }

    async def run_generation():
        await concurrency_semaphore.acquire()
        try:
            loop = asyncio.get_running_loop()

            def log_msg(msg):
                print(f"[SaaSShorts Job {job_id[:8]}] {msg}")
                if job_id in saas_jobs:
                    saas_jobs[job_id]["logs"].append(msg)

            def run():
                return generate_full_video(req.script, config, job_output_dir, log_msg)

            result = await loop.run_in_executor(None, run)

            if job_id in saas_jobs:
                video_filename = result["video_filename"]
                saas_jobs[job_id]["status"] = "completed"
                saas_jobs[job_id]["result"] = {
                    "video_url": f"/videos/saas_{job_id}/{video_filename}",
                    "video_filename": video_filename,
                    "duration": result.get("duration", 0),
                    "cost_estimate": result.get("cost_estimate", {}),
                    "script": req.script,
                }
                saas_jobs[job_id]["logs"].append("Video generation completed!")

                # Upload to public gallery — opt-in only: the metadata carries
                # the user's product name, URL and full script.
                if req.share_to_gallery:
                    try:
                        gallery_meta = {
                            "title": req.script.get("title", "Untitled"),
                            "hook_text": req.script.get("hook_text", ""),
                            "caption": req.script.get("caption", ""),
                            "hashtags": req.script.get("hashtags", []),
                            "full_narration": req.script.get("full_narration", ""),
                            "actor_description": req.script.get("actor_description", ""),
                            "style": req.script.get("style", "ugc"),
                            "language": req.script.get("language", "en"),
                            "duration": result.get("duration", 0),
                            "video_mode": req.video_mode,
                            "product_name": req.script.get("_product_name", ""),
                            "product_url": req.script.get("_product_url", ""),
                            "segments": req.script.get("segments", []),
                            "cost_estimate": result.get("cost_estimate", {}),
                        }
                        gallery_result = upload_video_to_gallery(
                            video_path=result["video_path"],
                            actor_image_path=result.get("actor_image", ""),
                            metadata=gallery_meta,
                            video_id=job_id[:8],
                        )
                        if gallery_result:
                            saas_jobs[job_id]["result"]["gallery_video_id"] = gallery_result["video_id"]
                            log_msg("📤 Uploaded to public gallery.")
                    except Exception as gallery_err:
                        log_msg(f"⚠️ Gallery upload skipped: {gallery_err}")

        except Exception as e:
            print(f"[SaaSShorts] ❌ Job {job_id} failed: {e}")
            if job_id in saas_jobs:
                saas_jobs[job_id]["status"] = "failed"
                saas_jobs[job_id]["logs"].append(f"Error: {str(e)}")
        finally:
            concurrency_semaphore.release()

    asyncio.create_task(run_generation())

    return {"job_id": job_id, "status": "processing"}


@app.get("/api/saasshorts/status/{job_id}")
async def saasshorts_status(job_id: str, request: Request):
    """Poll SaaSShorts job status."""
    if job_id not in saas_jobs:
        raise HTTPException(status_code=404, detail="SaaSShorts job not found")

    job = saas_jobs[job_id]
    await _assert_job_owner(request, job)
    return {
        "status": job["status"],
        "logs": job["logs"],
        "result": job.get("result"),
    }


@app.get("/api/saasshorts/voices")
async def saasshorts_voices(
    x_elevenlabs_key: Optional[str] = Header(None, alias="X-ElevenLabs-Key"),
):
    """List available ElevenLabs voices."""
    if x_elevenlabs_key:
        try:
            loop = asyncio.get_event_loop()
            voices = await loop.run_in_executor(
                None, get_elevenlabs_voices, x_elevenlabs_key
            )
            if voices:
                return {"voices": voices, "source": "elevenlabs"}
        except Exception:
            pass

    # Fallback to default voices
    return {
        "voices": [
            {"voice_id": vid, "name": name, "category": "default"}
            for name, vid in DEFAULT_VOICES.items()
        ],
        "source": "defaults",
    }
