"""Users' durable video library on R2: archive on completion, list for history,
purge after the subscription grace period.
"""
import asyncio
import glob
import os
from datetime import timedelta

from fastapi import APIRouter, HTTPException, Request
from sqlalchemy import select, delete

from .config import settings, VIDEO_RETENTION_GRACE_DAYS
from . import database, storage
from .models import UserVideo, Subscription, Project
from .auth import get_current_user_required

router = APIRouter()


def _clip_title(clip) -> str:
    return clip.get("title") or clip.get("video_title_for_youtube_short") or "Short"


async def archive_job(user_id, job_id, clips, output_dir):
    """Upload a completed managed job's clips + metadata JSON to R2 and record
    them for history. The metadata (transcript included) plus the Project row
    make the whole job re-openable and editable later, not just viewable."""
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
    if meta_files:
        metadata_r2_key = storage.job_key(user_id, job_id, os.path.basename(meta_files[0]))
        try:
            await asyncio.to_thread(storage.upload_file, meta_files[0], metadata_r2_key,
                                    "application/json")
        except Exception as e:
            print(f"⚠️  R2 metadata refresh failed for {job_id}: {e}")
            metadata_r2_key = None

    size = os.path.getsize(local_path)
    superseded_key = None
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
                if prev and prev not in (entry.get("original_file"), new_filename):
                    superseded_key = storage.job_key(user_id, job_id, prev)
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
    if superseded_key and superseded_key != new_key:
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


_SWEEP_INTERVAL = 6 * 3600  # every 6 hours


async def _sweeper_loop():
    while True:
        try:
            await asyncio.sleep(_SWEEP_INTERVAL)
            await purge_expired()
        except asyncio.CancelledError:
            break
        except Exception as e:
            print(f"⚠️  Video retention sweeper error: {e}")


def start_sweeper():
    asyncio.create_task(_sweeper_loop())
