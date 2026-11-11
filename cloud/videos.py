"""Users' durable video library on R2: archive on completion, list for history,
purge after the subscription grace period.
"""
import asyncio
import glob
import os
import re
from datetime import timedelta

from fastapi import APIRouter, HTTPException, Request
from sqlalchemy import select, delete

from .config import settings, VIDEO_RETENTION_GRACE_DAYS, FREE_CLIP_RETENTION_DAYS
from . import database, storage, metering
from .models import UserVideo, Subscription, Project, User, ClipExpiryWarning
from .auth import get_current_user_required

import watermarked

router = APIRouter()


def _clip_title(clip) -> str:
    return clip.get("title") or clip.get("video_title_for_youtube_short") or "Short"


_HOOK_CHAIN_RE = re.compile(r'^subtitled_\d+_((?:hooked_\d+_|hook_).+)$')


def _canonical_name(base_name, clip_index):
    return f"{base_name}_clip_{clip_index + 1}.mp4" if base_name else None


def _twins_to_archive(served, base_name, clip_index):
    """The files that must ride to R2 next to a served clip file, in order.

    * its clean twin when it is a free-plan ``wm_`` copy (an upgrade re-points
      at it: ``unmark_user_library``), unless that twin is the canonical,
      which the caller archives on its own;
    * the ``hooked_`` intermediate of a captioned hook chain, resolved on the
      clean name, or a restored project's caption restyle cannot walk back to
      the hook-only file and would silently drop or stack layers.
    """
    out = []
    clean = watermarked.clean_name(served)
    canonical = _canonical_name(base_name, clip_index)
    if clean != served and clean != canonical:
        out.append(clean)
    m = _HOOK_CHAIN_RE.match(clean)
    if m and m.group(1) != canonical:
        out.append(m.group(1))
    return out


def _superseded_names(prev, original_file, new_filename, base_name, clip_index):
    """R2 objects an edit leaves behind: the previous served file and, when
    that was a ``wm_`` copy, its clean twin. Never the pristine original (or
    its twin), never the canonical, never the file just written."""
    if not prev:
        return []
    keep = {new_filename, watermarked.clean_name(new_filename),
            original_file or "", watermarked.clean_name(original_file or ""),
            _canonical_name(base_name, clip_index) or ""}
    out = []
    for name in (prev, watermarked.clean_name(prev)):
        if name and name not in keep and name not in out:
            out.append(name)
    return out


def unmarked_clip_state(state):
    """Pure half of ``unmark_user_library``: the project state with every
    ``wm_`` name replaced by its clean twin, plus ``[(index, wm, clean)]`` for
    the clips whose served file changed."""
    new_state = dict(state or {"v": 1, "clips": []})
    changes, clips = [], []
    for c in new_state.get("clips", []):
        c = dict(c)
        served = c.get("server_file") or ""
        if watermarked.is_marked(served):
            clean = watermarked.clean_name(served)
            changes.append((c.get("index"), served, clean))
            c["server_file"] = clean
        original = c.get("original_file") or ""
        if watermarked.is_marked(original):
            c["original_file"] = watermarked.clean_name(original)
        clips.append(c)
    new_state["clips"] = clips
    return new_state, changes


# app.py owns the working files and the in-memory job table; like the erasure
# purge (cloud/account.py) it registers a callback at startup rather than being
# imported from here.
_local_unmark = None


def register_local_unmark(fn):
    """Let app.py hand us a ``(job_id, clip_index, clean_filename)`` callback."""
    global _local_unmark
    _local_unmark = fn


async def unmark_user_library(user_id, job_id=None) -> int:
    """After an upgrade: serve every clip this user has through its clean twin.

    The free plan burns the mark into a ``wm_`` copy and archives the clean
    final next to it (``_twins_to_archive``), so losing the mark is a
    re-pointing, not a render: the history row and the project state move to
    the twin, the marked object is deleted, and app.py is told so a dashboard
    that is open on the job follows. A clip whose twin is not in R2 keeps its
    marked file: a dead link would be worse than the mark. Returns the number
    of clips re-pointed. ``job_id`` limits it to that job (a free job that
    finished after its owner paid, ``app._unmark_if_owner_paid``).
    """
    if not settings.r2_configured:
        return 0
    async with database.session() as s:
        q = select(Project).where(Project.user_id == user_id)
        if job_id is not None:
            q = q.where(Project.job_id == str(job_id))
        projects = list((await s.execute(q)).scalars())
    done = 0
    for proj in projects:
        new_state, changes = unmarked_clip_state(proj.state)
        if not changes:
            continue
        applied = []
        for index, wm_name, clean in changes:
            clean_key = storage.job_key(user_id, proj.job_id, clean)
            try:
                size = await asyncio.to_thread(storage.object_size, clean_key)
            except Exception as e:
                print(f"⚠️  R2 head failed for {clean_key}: {e}")
                size = None
            if size is None:
                continue
            applied.append((index, wm_name, clean, clean_key, size))
        if not applied:
            continue
        applied_idx = {a[0] for a in applied}
        # Clips whose twin is missing keep their marked entry.
        prior = {c.get("index"): c for c in (proj.state or {}).get("clips", [])}
        new_state["clips"] = [
            c if c.get("index") in applied_idx else dict(prior.get(c.get("index"), c))
            for c in new_state["clips"]]
        async with database.session() as s:
            async with s.begin():
                row = (await s.execute(
                    select(Project).where(Project.id == proj.id)
                )).scalar_one_or_none()
                if row is not None:
                    row.state = new_state
                for index, _wm, _clean, clean_key, size in applied:
                    vid = (await s.execute(
                        select(UserVideo).where(UserVideo.user_id == user_id,
                                                UserVideo.job_id == proj.job_id,
                                                UserVideo.clip_index == index)
                    )).scalars().first()
                    if vid is not None:
                        vid.r2_key, vid.size_bytes = clean_key, size
        for index, wm_name, clean, _key, _size in applied:
            try:
                await asyncio.to_thread(
                    storage.delete_key, storage.job_key(user_id, proj.job_id, wm_name))
            except Exception as e:
                print(f"⚠️  Could not delete marked R2 object {wm_name}: {e}")
            if _local_unmark is not None:
                try:
                    _local_unmark(proj.job_id, index, clean)
                except Exception as e:
                    print(f"⚠️  Local unmark failed for {proj.job_id}: {e}")
            done += 1
    if done:
        print(f"🏷️  Watermark removed from {done} clip(s) of user {user_id} after upgrade.")
    return done


async def archive_job(user_id, job_id, clips, output_dir, attestation=None):
    """Upload a completed managed job's clips + metadata JSON to R2 and record
    them for history. The metadata (transcript included) plus the Project row
    make the whole job re-openable and editable later, not just viewable.

    ``attestation`` is the "I hold the rights to this video" declaration the
    submitter ticked (truncated IP, user agent, timestamp, which input path).
    The privacy policy promises to keep it; the Project row is where it lives,
    so it is deleted with the account like everything else that names a user.
    """
    if not settings.r2_configured or not clips:
        return
    metadata_r2_key = None
    meta_files = glob.glob(os.path.join(output_dir, "*_metadata.json"))
    if meta_files:
        key = storage.job_key(user_id, job_id, os.path.basename(meta_files[0]))
        try:
            await asyncio.to_thread(storage.upload_file, meta_files[0], key, "application/json")
            metadata_r2_key = key
        except Exception as e:
            print(f"⚠️  R2 upload failed for {key}: {e}")

    base_name = (os.path.basename(meta_files[0]).replace("_metadata.json", "")
                 if meta_files else None)
    uploaded = []  # (clip_index, filename, key, title, size_bytes)
    for i, clip in enumerate(clips):
        video_url = clip.get("video_url") or ""
        filename = video_url.split("/")[-1]
        if not filename:
            continue
        local_path = os.path.join(output_dir, filename)
        if not os.path.exists(local_path):
            continue
        key = storage.job_key(user_id, job_id, filename)
        try:
            await asyncio.to_thread(storage.upload_file, local_path, key)
        except Exception as e:
            print(f"⚠️  R2 upload failed for {key}: {e}")
            continue
        uploaded.append((i, filename, key, _clip_title(clip), os.path.getsize(local_path)))

        # Also archive the clean canonical when the served file is a derived
        # one (captioned/recut): it is the editable master — the clip editor's
        # fast path and /api/subtitle re-styles cut from it, so a project
        # restored after a redeploy is only editable if it comes back too.
        # Restore downloads every key under the job prefix, so no change is
        # needed on that side; the free-retention purge deletes by prefix.
        if base_name:
            clean = f"{base_name}_clip_{i + 1}.mp4"
            clean_path = os.path.join(output_dir, clean)
            if clean != filename and os.path.exists(clean_path):
                clean_key = storage.job_key(user_id, job_id, clean)
                try:
                    await asyncio.to_thread(storage.upload_file, clean_path, clean_key)
                except Exception as e:
                    print(f"⚠️  R2 upload failed for {clean_key}: {e}")

        # Free plan: the served file is the wm_ copy of the final one, and the
        # clean final rides along so an upgrade can re-point at it without a
        # render (unmark_user_library). The canonical is handled just above.
        for twin_name in _twins_to_archive(filename, base_name, i):
            twin_path = os.path.join(output_dir, twin_name)
            if os.path.exists(twin_path):
                twin_key = storage.job_key(user_id, job_id, twin_name)
                try:
                    await asyncio.to_thread(storage.upload_file, twin_path, twin_key)
                except Exception as e:
                    print(f"⚠️  R2 upload failed for {twin_key}: {e}")

    if not uploaded:
        return
    async with database.session() as s:
        async with s.begin():
            # Upsert per-clip history rows: re-archiving a job must repoint the
            # existing rows, never duplicate them.
            existing = {
                v.clip_index: v for v in (await s.execute(
                    select(UserVideo).where(UserVideo.user_id == user_id,
                                            UserVideo.job_id == job_id)
                )).scalars()
            }
            for i, _filename, key, title, size in uploaded:
                row = existing.get(i)
                if row is not None:
                    row.r2_key, row.title, row.size_bytes = key, title, size
                else:
                    s.add(UserVideo(user_id=user_id, job_id=job_id, clip_index=i,
                                    r2_key=key, title=title, size_bytes=size))
            # Upsert the Project row (only re-openable with the metadata in R2).
            if metadata_r2_key:
                proj = (await s.execute(
                    select(Project).where(Project.job_id == job_id)
                )).scalar_one_or_none()
                state = {"v": 1, "clips": [
                    {"index": i, "original_file": filename, "server_file": filename,
                     "active_layers": None}
                    for i, filename, _key, _title, _size in uploaded
                ]}
                if attestation:
                    state["rights_attestation"] = {
                        "acknowledged": bool(attestation.get("acknowledged")),
                        # Already truncated to a /24 (or /48) by app.py.
                        "ip_network": attestation.get("ip"),
                        "user_agent": (attestation.get("user_agent") or "")[:300],
                        "timestamp": attestation.get("timestamp"),
                        "source": attestation.get("source"),
                    }
                elif proj is not None and isinstance(proj.state, dict):
                    # Re-archiving must not erase the original declaration.
                    prior = proj.state.get("rights_attestation")
                    if prior:
                        state["rights_attestation"] = prior
                total = sum(u[4] for u in uploaded)
                if proj is None:
                    s.add(Project(user_id=user_id, job_id=job_id, title=uploaded[0][3],
                                  metadata_r2_key=metadata_r2_key, state=state,
                                  size_bytes=total))
                else:
                    proj.metadata_r2_key = metadata_r2_key
                    proj.state = state
                    proj.size_bytes = total
    print(f"☁️  Archived {len(uploaded)} clip(s) to R2 for user {user_id}.")


async def archive_clip_edit(user_id, job_id, clip_index, output_dir, new_filename):
    """Re-archive one clip after a server-side edit (subtitles/hook/effects/dub).

    Uploads the new current file plus the refreshed metadata JSON, repoints the
    history row and the project state, and deletes the superseded R2 object —
    per clip only the pristine original and the current version are kept.
    """
    if not settings.r2_configured or not user_id:
        return
    local_path = os.path.join(output_dir, new_filename)
    if not os.path.exists(local_path):
        return
    new_key = storage.job_key(user_id, job_id, new_filename)
    await asyncio.to_thread(storage.upload_file, local_path, new_key)

    metadata_r2_key = None
    meta_files = glob.glob(os.path.join(output_dir, "*_metadata.json"))
    base_name = (os.path.basename(meta_files[0]).replace("_metadata.json", "")
                 if meta_files else None)

    # Same chain rule as archive_job: a captioned hook needs its hooked_
    # intermediate archived too, or the restored project cannot re-style; and
    # a wm_ copy needs its clean twin, or an upgrade cannot drop the mark.
    for twin_name in _twins_to_archive(new_filename, base_name, clip_index):
        twin_path = os.path.join(output_dir, twin_name)
        if os.path.exists(twin_path):
            try:
                await asyncio.to_thread(
                    storage.upload_file, twin_path,
                    storage.job_key(user_id, job_id, twin_name))
            except Exception as e:
                print(f"⚠️  R2 upload failed for {twin_name} of {job_id}: {e}")
    if meta_files:
        metadata_r2_key = storage.job_key(user_id, job_id, os.path.basename(meta_files[0]))
        try:
            await asyncio.to_thread(storage.upload_file, meta_files[0], metadata_r2_key,
                                    "application/json")
        except Exception as e:
            print(f"⚠️  R2 metadata refresh failed for {job_id}: {e}")
            metadata_r2_key = None

    size = os.path.getsize(local_path)
    superseded = []
    async with database.session() as s:
        async with s.begin():
            proj = (await s.execute(
                select(Project).where(Project.job_id == job_id)
            )).scalar_one_or_none()
            if proj is not None:
                state = dict(proj.state or {"v": 1, "clips": []})
                clips_state = [dict(c) for c in state.get("clips", [])]
                entry = next((c for c in clips_state if c.get("index") == clip_index), None)
                if entry is None:
                    entry = {"index": clip_index, "original_file": new_filename,
                             "active_layers": None}
                    clips_state.append(entry)
                prev = entry.get("server_file")
                superseded = [
                    storage.job_key(user_id, job_id, name) for name in
                    _superseded_names(prev, entry.get("original_file"), new_filename,
                                      base_name, clip_index)]
                entry["server_file"] = new_filename
                state["clips"] = clips_state
                proj.state = state
                if metadata_r2_key:
                    proj.metadata_r2_key = metadata_r2_key
            vid = (await s.execute(
                select(UserVideo).where(UserVideo.user_id == user_id,
                                        UserVideo.job_id == job_id,
                                        UserVideo.clip_index == clip_index)
            )).scalars().first()
            if vid is not None:
                vid.r2_key, vid.size_bytes = new_key, size
            else:
                s.add(UserVideo(user_id=user_id, job_id=job_id, clip_index=clip_index,
                                r2_key=new_key, size_bytes=size))
    for superseded_key in superseded:
        if superseded_key == new_key:
            continue
        try:
            await asyncio.to_thread(storage.delete_key, superseded_key)
        except Exception as e:
            print(f"⚠️  Could not delete superseded R2 object {superseded_key}: {e}")


@router.get("/api/projects")
async def list_projects(request: Request):
    """List the signed-in user's re-openable projects."""
    user = await get_current_user_required(request)
    async with database.session() as s:
        projs = list((await s.execute(
            select(Project).where(Project.user_id == user.id)
            .order_by(Project.created_at.desc()).limit(200)
        )).scalars())
    return {"projects": [{
        "job_id": p.job_id,
        "title": p.title,
        "clip_count": len((p.state or {}).get("clips", [])),
        "size_bytes": p.size_bytes,
        "created_at": p.created_at.isoformat() if p.created_at else None,
        "updated_at": p.updated_at.isoformat() if p.updated_at else None,
    } for p in projs]}


@router.put("/api/projects/{job_id}/state")
async def save_project_state(job_id: str, request: Request):
    """Persist the browser-only edit state of a project's clips.

    Body: {"clips": [{"index", "active_layers", "server_file"?}]}. The Remotion
    layers exist nowhere but the browser, so the frontend syncs them here
    (debounced) to survive reload / reopen."""
    user = await get_current_user_required(request)
    if int(request.headers.get("content-length") or 0) > 262144:
        raise HTTPException(status_code=413, detail="State too large")
    body = await request.json()
    clips_in = body.get("clips") or []
    async with database.session() as s:
        async with s.begin():
            proj = (await s.execute(
                select(Project).where(Project.job_id == job_id)
            )).scalar_one_or_none()
            if proj is None or str(proj.user_id) != str(user.id):
                raise HTTPException(status_code=404, detail="Project not found")
            state = dict(proj.state or {"v": 1, "clips": []})
            clips_state = [dict(c) for c in state.get("clips", [])]
            by_index = {c.get("index"): c for c in clips_state}
            for c in clips_in:
                idx = c.get("index")
                if not isinstance(idx, int):
                    continue
                entry = by_index.get(idx)
                if entry is None:
                    entry = {"index": idx, "original_file": None,
                             "server_file": None, "active_layers": None}
                    clips_state.append(entry)
                    by_index[idx] = entry
                entry["active_layers"] = c.get("active_layers")
                if c.get("server_file"):
                    entry["server_file"] = os.path.basename(str(c["server_file"]))
            state["clips"] = clips_state
            proj.state = state
    return {"success": True}


@router.get("/api/history")
async def history(request: Request):
    """List the signed-in user's saved videos with private, time-limited links."""
    user = await get_current_user_required(request)
    async with database.session() as s:
        vids = list((await s.execute(
            select(UserVideo).where(UserVideo.user_id == user.id)
            .order_by(UserVideo.created_at.desc()).limit(500)
        )).scalars())
    items = []
    for v in vids:
        safe_name = (v.title or "short").strip().replace("/", "-")[:60] + ".mp4"
        items.append({
            "id": str(v.id),
            "job_id": v.job_id,
            "clip_index": v.clip_index,
            "title": v.title,
            "created_at": v.created_at.isoformat() if v.created_at else None,
            "size_bytes": v.size_bytes,
            # The basename of the archived object. The preview player compares it
            # against the clip's current server file: after an edit the local file
            # is already the new one while the R2 re-archive is still in flight
            # (_archive_clip_edit_bg is fire-and-forget), and playing the durable
            # copy then would show the clip from BEFORE the edit.
            "filename": (v.r2_key or "").rsplit("/", 1)[-1],
            "view_url": storage.presigned_get(v.r2_key, expires=3600),
            "download_url": storage.presigned_get(v.r2_key, expires=3600, download_name=safe_name),
        })
    return {"videos": items}


async def purge_expired():
    """Delete R2 videos for users whose subscription ended > grace-period days ago."""
    async with database.session() as s:
        # Users with a canceled subscription past the grace period, who still have videos.
        canceled = list((await s.execute(
            select(Subscription.user_id, Subscription.last_event_at)
            .where(Subscription.status == "canceled")
        )).all())
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc)
    for user_id, last_event_at in canceled:
        if not last_event_at:
            continue
        if last_event_at + timedelta(days=VIDEO_RETENTION_GRACE_DAYS) > now:
            continue
        # A canceled Google-authed user is now an entitled FREE user, not a
        # lapsed one — their library follows the free 7-day per-clip expiry
        # (purge_free_expired) instead of the full wipe.
        async with database.session() as s:
            if await metering.is_free_user(s, user_id):
                continue
        # Any videos or projects left?
        async with database.session() as s:
            has = (await s.execute(
                select(UserVideo.id).where(UserVideo.user_id == user_id).limit(1)
            )).first() or (await s.execute(
                select(Project.id).where(Project.user_id == user_id).limit(1)
            )).first()
        if not has:
            continue
        try:
            n = await asyncio.to_thread(storage.delete_prefix, storage.user_prefix(user_id))
            async with database.session() as s:
                async with s.begin():
                    await s.execute(delete(UserVideo).where(UserVideo.user_id == user_id))
                    await s.execute(delete(Project).where(Project.user_id == user_id))
            print(f"🗑️  Purged {n} R2 object(s) for lapsed user {user_id}.")
        except Exception as e:
            print(f"⚠️  Video purge failed for {user_id}: {e}")


async def purge_free_expired():
    """Expire free users' clips/projects after FREE_CLIP_RETENTION_DAYS.

    Paid libraries are durable; the free plan keeps R2 storage bounded (and the
    limited retention is an upgrade lever). Only rows whose owner is currently
    on the free plan are touched — a user who upgraded keeps everything.
    """
    from datetime import datetime, timezone
    cutoff = datetime.now(timezone.utc) - timedelta(days=FREE_CLIP_RETENTION_DAYS)
    async with database.session() as s:
        old_videos = list((await s.execute(
            select(UserVideo).where(UserVideo.created_at < cutoff)
        )).scalars())
        old_projects = list((await s.execute(
            select(Project).where(Project.updated_at < cutoff)
        )).scalars())

    by_user = {}
    for v in old_videos:
        by_user.setdefault(v.user_id, {"videos": [], "projects": []})["videos"].append(v)
    for p in old_projects:
        by_user.setdefault(p.user_id, {"videos": [], "projects": []})["projects"].append(p)

    for user_id, items in by_user.items():
        async with database.session() as s:
            if not await metering.is_free_user(s, user_id):
                continue
        try:
            for v in items["videos"]:
                await asyncio.to_thread(storage.delete_key, v.r2_key)
            for p in items["projects"]:
                # Delete everything under the job prefix, not just the metadata
                # key: the archive also stores clean canonical clips (which have
                # no DB row of their own) and superseded derivatives would
                # otherwise leak as orphans.
                try:
                    prefix = storage.job_key(user_id, p.job_id, "")
                    for key in await asyncio.to_thread(storage.list_keys, prefix):
                        await asyncio.to_thread(storage.delete_key, key)
                except Exception:
                    if p.metadata_r2_key:
                        await asyncio.to_thread(storage.delete_key, p.metadata_r2_key)
            async with database.session() as s:
                async with s.begin():
                    if items["videos"]:
                        await s.execute(delete(UserVideo).where(
                            UserVideo.id.in_([v.id for v in items["videos"]])))
                    if items["projects"]:
                        await s.execute(delete(Project).where(
                            Project.id.in_([p.id for p in items["projects"]])))
            n = len(items["videos"]) + len(items["projects"])
            print(f"🗑️  Free retention: purged {n} expired object(s) for user {user_id}.")
        except Exception as e:
            print(f"⚠️  Free retention purge failed for {user_id}: {e}")


async def warn_free_expiring():
    """Email free users whose clips enter their LAST day before deletion.

    The 7-day free retention is the plan's strongest honest upgrade lever, but
    silently deleting clips converts nobody — the user never learns they were
    at risk. One email on day 6 turns the real deadline into a visible one.

    "Already warned" is recorded in clip_expiry_warnings rather than in memory.
    The warning window is a full day and this sweep runs every six hours, so a
    process-local set meant any deploy inside that window re-mailed everyone who
    had already been told.
    """
    from datetime import datetime, timezone
    from .emails import send_clips_expiring_email
    now = datetime.now(timezone.utc)
    doomed_after = now - timedelta(days=FREE_CLIP_RETENTION_DAYS)       # already purgeable
    warn_cutoff = now - timedelta(days=FREE_CLIP_RETENTION_DAYS - 1)   # < 1 day left

    async with database.session() as s:
        expiring = list((await s.execute(
            select(UserVideo).where(UserVideo.created_at < warn_cutoff,
                                    UserVideo.created_at >= doomed_after)
        )).scalars())
        already = set((await s.execute(select(ClipExpiryWarning.video_id))).scalars())

    by_user = {}
    for v in expiring:
        if v.id in already:
            continue
        by_user.setdefault(v.user_id, []).append(v)

    for user_id, vids in by_user.items():
        async with database.session() as s:
            if not await metering.is_free_user(s, user_id):
                continue
            user = await s.get(User, user_id)
        if not user or not user.email:
            continue
        # Record the warning BEFORE sending. A duplicate email is the failure we
        # are fixing; a warning row for a mail that then failed to send only
        # costs that user one notice, which is the cheaper way to be wrong.
        try:
            async with database.session() as s:
                async with s.begin():
                    for v in vids:
                        s.add(ClipExpiryWarning(video_id=v.id, user_id=user_id))
        except Exception as e:
            print(f"⚠️  Could not record expiry warning for {user_id}: {e}")
            continue
        try:
            await send_clips_expiring_email(user.email, len(vids))
            print(f"⏳ Expiry warning sent: {len(vids)} clip(s), user {user_id}")
        except Exception as e:
            print(f"⚠️  Expiry warning failed for {user_id}: {e}")


_SWEEP_INTERVAL = 6 * 3600  # every 6 hours


async def _sweeper_loop():
    while True:
        try:
            await asyncio.sleep(_SWEEP_INTERVAL)
            await warn_free_expiring()
            await purge_expired()
            await purge_free_expired()
            from .auth import purge_stale_magic_tokens
            await purge_stale_magic_tokens()
            from .account import purge_stale_deletion_records
            await purge_stale_deletion_records()
            from .evidence import purge_stale_events
            await purge_stale_events()
        except asyncio.CancelledError:
            break
        except Exception as e:
            print(f"⚠️  Video retention sweeper error: {e}")


def start_sweeper():
    asyncio.create_task(_sweeper_loop())
