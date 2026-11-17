"""A paid job's source video in R2 for SOURCE_ARCHIVE_DAYS after its upload.

On the processing server the source lives an hour at most, and the disk caps
can take it sooner. Once it is gone the source preview stops playing and the
clip editor can no longer extend a cut past the original range or reframe
(their 409 "no longer on the server"). Clips are already durable in R2; this
module makes the source durable too, for paid plans only, for a bounded time.

* **Where**: ``users/<user_id>/<job_id>/source/<random hex>.<ext>``. Under the
  user's prefix, so the account erasure (``cloud/account.py``) and the lapsed
  and free purges (``cloud/videos.py``), which delete by prefix, take it with
  no code of their own. Inside the job prefix but one level down, so it is
  told apart by shape (``is_source_key``) and the project restore, which pulls
  every key under the job prefix, skips it: a reopened project must not pull
  hundreds of MB it may never use. The random name is the capability when the
  bucket is served on a public custom domain (``settings.r2_public_base``):
  knowing a clip's URL (user id + job id) must not be enough to fetch the
  whole original.
* **When**: after the clips are archived, in the background, at most
  ``UPLOAD_CONCURRENCY`` uploads at a time in the process, only if the owner is
  on a live paid plan at that moment. A missing source or a failed upload is
  logged and nothing else happens: the job is already delivered.
* **How long**: a daily sweep lists the bucket and deletes every source object
  older than SOURCE_ARCHIVE_DAYS, read from R2's own LastModified. No database
  row, so there is no table to add to ``USER_OWNED_TABLES``. With the knob at 0
  nothing is uploaded and the sweep deletes whatever is left.
* **Back**: the preview redirects to a short-lived URL of the object (nothing
  lands on the server); the editor endpoints that need the file download it
  into the job directory under a fixed dot name (``LOCAL_NAME``), which a
  remote worker can never write (it only writes plain non-dot files).
"""
import asyncio
import os
import re
import secrets
import time
from datetime import datetime, timedelta, timezone

from . import config, storage
from .config import settings

SOURCE_SEGMENT = "source"
# Containers the pipeline accepts; anything else is stored and restored as
# .mp4, which is also the type /api/source has always served it as.
EXTENSIONS = (".mp4", ".mov", ".mkv", ".webm", ".m4v", ".avi")
_NAME_RE = re.compile(r"^[0-9a-f]{32}(\.(?:mp4|mov|mkv|webm|m4v|avi))$")
_UUID_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")
# The restored copy in the job directory: LOCAL_NAME + one of EXTENSIONS.
LOCAL_NAME = ".source_archive"

UPLOAD_CONCURRENCY = 2
# How long a redirect to the archived source stays valid. An editing session
# re-asks /api/source on every reload, so an hour is plenty; a leaked link dies
# with it (except on a public custom domain, where the random name is the lock).
LINK_SECONDS = 3600
# The bucket-wide listing is the cost of having no table, so it runs once a
# day, not on every six-hour tick of the retention sweeper.
SWEEP_EVERY_SECONDS = 24 * 3600
# Plausible source durations; anything else read back from R2 is ignored.
_MAX_DURATION = 48 * 3600

_upload_sem = None
_found: dict = {}          # (user_id, job_id) -> (checked_at, info or None)
_FOUND_HIT_TTL = 300.0
_FOUND_MISS_TTL = 60.0
_FOUND_MAX = 5000
_last_sweep = 0.0


def enabled() -> bool:
    return bool(settings.r2_configured) and config.SOURCE_ARCHIVE_DAYS > 0


def _valid_id(value) -> bool:
    return value is not None and bool(_UUID_RE.match(str(value)))


def source_prefix(user_id, job_id) -> str:
    return storage.job_key(user_id, job_id, f"{SOURCE_SEGMENT}/")


def is_source_key(key) -> bool:
    """``users/<uid>/<job>/source/<name>``: the shape only this module writes."""
    parts = str(key or "").split("/")
    return len(parts) == 5 and parts[0] == "users" and parts[3] == SOURCE_SEGMENT


def safe_extension(name) -> str:
    """The container extension of a local source path, from the allowlist."""
    ext = os.path.splitext(str(name or ""))[1].lower()
    return ext if ext in EXTENSIONS else ".mp4"


def local_candidates(job_dir) -> list:
    """Where a restored source can sit in a job directory."""
    return [os.path.join(job_dir, LOCAL_NAME + ext) for ext in EXTENSIONS]


def _clean_duration(value):
    try:
        d = float(value)
    except (TypeError, ValueError):
        return None
    return round(d, 3) if 0 < d <= _MAX_DURATION else None


def _semaphore():
    global _upload_sem
    if _upload_sem is None:
        _upload_sem = asyncio.Semaphore(UPLOAD_CONCURRENCY)
    return _upload_sem


def _forget(user_id, job_id):
    _found.pop((str(user_id), str(job_id)), None)


async def _owner_is_paying(user_id) -> bool:
    from . import metering
    plan = (await metering.live_plans([user_id])).get(str(user_id))
    return bool(plan) and plan != "free"


async def _user_exists(user_id) -> bool:
    import uuid as _uuid
    from . import database
    from .models import User
    async with database.session() as s:
        return (await s.get(User, _uuid.UUID(str(user_id)))) is not None


async def archive_source(user_id, job_id, local_path, duration_probe=None):
    """Upload a finished job's source if its owner pays right now.

    Returns the R2 key, or None when nothing was stored. Never raises: it runs
    after the delivery, and nothing about it may fail a job. ``duration_probe``
    is a blocking ``path -> seconds`` run off the loop, only once the plan check
    has passed; the result rides as object metadata so the editor's source
    track has a real scale without the file on disk.
    """
    if not enabled() or not _valid_id(user_id) or not _valid_id(job_id) or not local_path:
        return None
    try:
        if not await _owner_is_paying(user_id):
            return None
    except Exception as e:
        print(f"⚠️  Source archive skipped for {job_id}, plan lookup failed: {e}")
        return None
    # Open before queueing for a slot: the sweeps may delete the name while
    # this waits behind another upload, and the open handle keeps the bytes.
    try:
        fileobj = await asyncio.to_thread(open, local_path, "rb")
    except OSError:
        print(f"ℹ️  Source of {job_id} already gone from disk, not archived.")
        return None
    key = source_prefix(user_id, job_id) + secrets.token_hex(16) + safe_extension(local_path)
    try:
        meta = {}
        if duration_probe is not None:
            try:
                d = _clean_duration(await asyncio.to_thread(duration_probe, local_path))
            except Exception:
                d = None
            if d is not None:
                meta["duration"] = f"{d:.3f}"
        async with _semaphore():
            await asyncio.to_thread(storage.upload_fileobj, fileobj, key, "video/mp4", meta)
    except Exception as e:
        print(f"⚠️  Source archive upload failed for {job_id}: {e}")
        return None
    finally:
        try:
            fileobj.close()
        except Exception:
            pass

    # An erasure that ran while the upload was in flight purged the prefix
    # before this object existed: delete it now rather than wait for the sweep.
    try:
        if not await _user_exists(user_id):
            await asyncio.to_thread(storage.delete_key, key)
            print(f"🗑️  Source of {job_id} dropped: its account was deleted meanwhile.")
            return None
    except Exception as e:
        print(f"⚠️  Source archive owner check failed for {job_id}: {e}")
    # One copy per job: a delivery resumed on another instance uploads again.
    try:
        stale = [k for k, _m, _s in await asyncio.to_thread(
            lambda: list(storage.iter_objects(source_prefix(user_id, job_id)))) if k != key]
        if stale:
            await asyncio.to_thread(storage.delete_keys, stale)
    except Exception as e:
        print(f"⚠️  Could not drop older source copies of {job_id}: {e}")
    _forget(user_id, job_id)
    print(f"☁️  Source of {job_id} archived for {config.SOURCE_ARCHIVE_DAYS} days.")
    return key


def _lookup(user_id, job_id):
    """Blocking: the newest archived source of a job, or None."""
    best = None
    for key, modified, _size in storage.iter_objects(source_prefix(user_id, job_id)):
        if not is_source_key(key) or not _NAME_RE.match(key.rsplit("/", 1)[-1]):
            continue
        if best is None or (modified is not None
                            and (best[1] is None or modified > best[1])):
            best = (key, modified)
    if best is None:
        return None
    key, modified = best
    if modified is not None and config.SOURCE_ARCHIVE_DAYS > 0:
        if modified < datetime.now(timezone.utc) - timedelta(days=config.SOURCE_ARCHIVE_DAYS):
            return None   # past its date, the daily sweep just has not run yet
    meta = storage.object_metadata(key) or {}
    return {"key": key, "duration": _clean_duration(meta.get("duration"))}


async def find(user_id, job_id):
    """``{"key", "duration"}`` of the job's archived source, or None. Cached
    briefly: a player re-asks /api/source on every range request."""
    if not enabled() or not _valid_id(user_id) or not _valid_id(job_id):
        return None
    cache_key = (str(user_id), str(job_id))
    now = time.time()
    hit = _found.get(cache_key)
    if hit is not None:
        ttl = _FOUND_HIT_TTL if hit[1] is not None else _FOUND_MISS_TTL
        if now - hit[0] < ttl:
            return hit[1]
    try:
        info = await asyncio.to_thread(_lookup, str(user_id), str(job_id))
    except Exception as e:
        print(f"⚠️  Source archive lookup failed for {job_id}: {e}")
        return None
    if len(_found) >= _FOUND_MAX:
        _found.clear()
    _found[cache_key] = (now, info)
    return info


def signed_url(info) -> str:
    return storage.presigned_get(info["key"], expires=LINK_SECONDS)


async def fetch(user_id, job_id, job_dir):
    """Download the archived source into ``job_dir``; the local path or None.

    The caller serializes per job and checks the disk first; the name is built
    here from the allowlisted extension only, never from what R2 returns.
    """
    info = await find(user_id, job_id)
    if info is None or not os.path.isdir(job_dir):
        return None
    m = _NAME_RE.match(info["key"].rsplit("/", 1)[-1])
    if not m:
        return None
    dest = os.path.join(job_dir, LOCAL_NAME + m.group(1))
    part = os.path.join(job_dir, LOCAL_NAME + ".part")
    try:
        await asyncio.to_thread(storage.download_file, info["key"], part)
        os.replace(part, dest)
    except Exception as e:
        try:
            os.remove(part)
        except OSError:
            pass
        _forget(user_id, job_id)
        print(f"⚠️  Could not bring back the source of {job_id}: {e}")
        return None
    print(f"♻️  Source of {job_id} brought back from the archive.")
    return dest


def _expired_keys(cutoff):
    """Blocking: every source key last written before ``cutoff`` (all of them
    when ``cutoff`` is None). One pass over the bucket, a page at a time."""
    out = []
    for key, modified, _size in storage.iter_objects("users/"):
        if not is_source_key(key):
            continue
        if cutoff is None or (modified is not None and modified < cutoff):
            out.append(key)
    return out


async def purge_expired_sources(now=None, force=False) -> int:
    """Delete archived sources past SOURCE_ARCHIVE_DAYS. Self-throttled to once
    a day (``force`` for tests and manual runs). Returns the count deleted."""
    global _last_sweep
    if not settings.r2_configured:
        return 0
    started = time.time()
    if not force and started - _last_sweep < SWEEP_EVERY_SECONDS:
        return 0
    now = now or datetime.now(timezone.utc)
    days = config.SOURCE_ARCHIVE_DAYS
    cutoff = now - timedelta(days=days) if days > 0 else None
    try:
        doomed = await asyncio.to_thread(_expired_keys, cutoff)
        if doomed:
            await asyncio.to_thread(storage.delete_keys, doomed)
            _found.clear()
            print(f"🗑️  Source archive: deleted {len(doomed)} expired source(s).")
        # Only a pass that finished counts: a failed one retries on the next
        # six-hour tick instead of waiting a whole day.
        _last_sweep = started
        return len(doomed)
    except Exception as e:
        print(f"⚠️  Source archive sweep failed: {e}")
        return 0
