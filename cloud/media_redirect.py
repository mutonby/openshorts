"""Where a clip can be fetched from storage instead of from this host.

A finished clip is archived to R2 anyway, and behind a public custom domain
(``R2_PUBLIC_BASE``) the object comes off an edge cache. ``/videos`` is one
Python process that is also running the queue, so a player or a download is
better pointed there. This module answers one question: is the file the
caller asked for the one storage holds, byte for byte, right now?

It is, when all of these hold:

* a ``UserVideo`` row of that owner and job names exactly that file. The rows
  are the *served* file of each clip, so on the free plan that is the marked
  copy and never its clean twin, whatever sits next to it in the bucket;
* no re-archive of the job is in flight (``hold``): an edit writes the local
  file first and uploads after, and a few edits reuse a file name;
* the local copy, when there is one, has the size the row recorded.

The link carries the size (and the local file's time) as a query string: an
object written again under the same name would otherwise keep coming out of
the edge cache as it was.

``MEDIA_REDIRECT=0`` turns it off. Without ``R2_PUBLIC_BASE`` it is off too:
the signed S3 endpoint is not one a browser can be sent to (see
``settings.r2_public_base``).
"""
import os
import time
from typing import Optional
from urllib.parse import quote

from sqlalchemy import select

from . import database
from .config import settings
from .models import UserVideo

# How long the served files of a job are remembered. Short: an archive that
# finishes on its own (``forget``) is the normal way a job's answer changes.
_TTL_SECONDS = 60.0
_MAX_JOBS = 4000
_HOLD_SECONDS = 900.0

_served = {}    # (owner, job_id) -> (read_at, {filename: (r2_key, size_bytes)})
_held = {}      # job_id -> held_until


def enabled() -> bool:
    if os.environ.get("MEDIA_REDIRECT", "1").strip().lower() in ("0", "off", "false", "no"):
        return False
    return bool(settings.r2_configured and settings.r2_public_base)


def hold(job_id) -> None:
    """A file of this job is about to be archived again: answer nothing for it
    until ``forget`` (or the ceiling, if the archive never reports back)."""
    if len(_held) >= _MAX_JOBS:
        _held.clear()
    _held[str(job_id)] = time.time() + _HOLD_SECONDS


def forget(job_id) -> None:
    """The rows of this job changed (archived, re-pointed, deleted)."""
    job_id = str(job_id)
    _held.pop(job_id, None)
    for key in [k for k in _served if k[1] == job_id]:
        _served.pop(key, None)


def _is_held(job_id) -> bool:
    until = _held.get(job_id)
    if until is None:
        return False
    if time.time() >= until:
        _held.pop(job_id, None)
        return False
    return True


async def _served_files(owner, job_id):
    now = time.time()
    hit = _served.get((owner, job_id))
    if hit is not None and now - hit[0] < _TTL_SECONDS:
        return hit[1]
    async with database.session() as s:
        rows = (await s.execute(
            select(UserVideo.r2_key, UserVideo.size_bytes)
            .where(UserVideo.user_id == owner, UserVideo.job_id == job_id)
        )).all()
    prefix = f"users/{owner}/{job_id}/"
    files = {}
    for key, size in rows:
        # Only an object of this owner's own job: the row is trusted for the
        # name, not for where it points.
        if isinstance(key, str) and key.startswith(prefix) and "/" not in key[len(prefix):]:
            files[key[len(prefix):]] = (key, size)
    if len(_served) >= _MAX_JOBS:
        _served.clear()
    _served[(owner, job_id)] = (now, files)
    return files


async def storage_url(owner, job_id, filename, local_path=None) -> Optional[str]:
    """The storage link of ``filename`` of that owner's job, or None when this
    host has to serve it. ``local_path`` is where the working copy would be."""
    if not enabled() or not owner or not job_id or not filename:
        return None
    owner, job_id = str(owner), str(job_id)
    if _is_held(job_id):
        return None
    try:
        files = await _served_files(owner, job_id)
    except Exception as e:
        print(f"⚠️  Storage link lookup failed for {job_id}: {e}")
        return None
    entry = files.get(filename)
    if entry is None:
        return None
    key, size = entry
    version = str(int(size)) if size else "0"
    if local_path:
        try:
            st = os.stat(local_path)
        except OSError:
            st = None
        if st is not None:
            if not size or st.st_size != size:
                return None
            version = f"{version}-{int(st.st_mtime)}"
    return f"{settings.r2_public_base}/{quote(key, safe='/')}?v={version}"
