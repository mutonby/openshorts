"""Users' durable video library on R2: archive on completion, list for history,
purge after the subscription grace period.
"""
import asyncio
import glob
import io
import json
import mimetypes
import os
import re
import uuid
from datetime import timedelta
from urllib.parse import urlsplit

from fastapi import APIRouter, HTTPException, Request
from sqlalchemy import select, delete
from sqlalchemy.exc import IntegrityError

from .config import settings, VIDEO_RETENTION_GRACE_DAYS, FREE_CLIP_RETENTION_DAYS
from . import database, storage, metering, media_redirect
from .models import UserVideo, Subscription, Project, User, ClipExpiryWarning
from .auth import get_current_user_required

import clean_twin
import watermarked

router = APIRouter()


def _clip_title(clip) -> str:
    return clip.get("title") or clip.get("video_title_for_youtube_short") or "Short"


_LAYER_PREFIX = re.compile(r'^(?:endcard_[a-f0-9]{8}_|subtitled_\d+_|hooked_\d+_|hook_)(.+)$')


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
    match = _LAYER_PREFIX.match(clean)
    while match:
        clean = match.group(1)
        if clean != canonical and clean not in out:
            out.append(clean)
        match = _LAYER_PREFIX.match(clean)
    return out


def _archived_twins(output_dir, served, base_name, clip_index):
    """The files of ``_twins_to_archive`` that are on disk, and for a clean
    twin kept as a recipe (clean_twin.py) the recipe's files instead: a
    restored project, or an upgrade after the working files are gone, can
    still encode it."""
    out = []
    for name in _twins_to_archive(served, base_name, clip_index):
        if os.path.exists(os.path.join(output_dir, name)):
            out.append(name)
        else:
            out += [n for n in clean_twin.archive_names(output_dir, name) if n not in out]
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
    keep.update(_twins_to_archive(new_filename, base_name, clip_index))
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


# app.py encodes a clean twin kept as a recipe (clean_twin.py) and uploads it:
# ``async (user_id, project, clean_filename) -> bool``.
_clean_render = None


def register_clean_render(fn):
    global _clean_render
    _clean_render = fn


async def _clean_size(user_id, proj, clean):
    """Size of ``clean`` in R2, encoding it from its recipe first if it is
    not there (a free clip from CLEAN_LAZY); None when it cannot be had."""
    clean_key = storage.job_key(user_id, proj.job_id, clean)
    try:
        size = await asyncio.to_thread(storage.object_size, clean_key)
    except Exception as e:
        print(f"⚠️  R2 head failed for {clean_key}: {e}")
        return None
    if size is None and _clean_render is not None and await _clean_render(user_id, proj, clean):
        try:
            size = await asyncio.to_thread(storage.object_size, clean_key)
        except Exception as e:
            print(f"⚠️  R2 head failed for {clean_key}: {e}")
            return None
    return size


async def unmark_user_library(user_id, job_id=None) -> int:
    """After an upgrade: serve every clip this user has through its clean twin.

    The free plan burns the mark into a ``wm_`` copy and archives the clean
    final next to it, or a recipe for it that can be encoded on demand.
    Losing the mark re-points history and project state to the clean twin,
    and app.py is told so a dashboard
    that is open on the job follows. A clip whose twin is not in R2 keeps its
    marked file: a dead link would be worse than the mark. Returns the number
    of clips re-pointed. ``job_id`` limits it to that job (a free job that
    finished after its owner paid, ``app._unmark_if_owner_paid``).
    """
    if not settings.r2_configured:
        return 0
    async with database.session() as s:
        # Newest first: the job a dashboard is open on gets its clean files
        # first when twins have to be encoded (clean_twin.py).
        q = select(Project).where(Project.user_id == user_id).order_by(
            Project.created_at.desc())
        if job_id is not None:
            q = q.where(Project.job_id == str(job_id))
        projects = list((await s.execute(q)).scalars())
    done = 0
    for proj in projects:
        _, changes = unmarked_clip_state(proj.state)
        if not changes:
            continue
        applied = []
        for index, wm_name, clean in changes:
            size = await _clean_size(user_id, proj, clean)
            if size is None:
                continue
            applied.append((index, wm_name, clean,
                            storage.job_key(user_id, proj.job_id, clean), size))
        if not applied:
            continue
        # The pristine original of an edited clip is a free clip's served file
        # too: its twin is what "back to the original" opens once unmarked.
        served = {clean for _i, _wm, clean, _k, _s in applied}
        for c in (proj.state or {}).get("clips", []):
            original = c.get("original_file") or ""
            if (watermarked.is_marked(original)
                    and watermarked.clean_name(original) not in served):
                await _clean_size(user_id, proj, watermarked.clean_name(original))
        async with database.session() as s:
            async with s.begin():
                row = (await s.execute(
                    select(Project).where(Project.id == proj.id).with_for_update()
                )).scalar_one_or_none()
                if row is None:
                    continue
                prior = {c.get("index"): c for c in (row.state or {}).get("clips", [])}
                # The project may have changed while storage was checked.
                applied = [a for a in applied if (prior.get(a[0]) or {}).get("server_file") == a[1]]
                if not applied:
                    continue
                applied_idx = {a[0] for a in applied}
                new_state, _ = unmarked_clip_state(row.state)
                new_state["clips"] = [
                    c if c.get("index") in applied_idx else dict(prior.get(c.get("index"), c))
                    for c in new_state["clips"]]
                row.state = new_state
                retain_snapshot = _archive_revision(new_state) > 0
                for index, _wm, _clean, clean_key, size in applied:
                    vid = (await s.execute(
                        select(UserVideo).where(UserVideo.user_id == user_id,
                                                UserVideo.job_id == proj.job_id,
                                                UserVideo.clip_index == index)
                    )).scalars().first()
                    if vid is not None:
                        vid.r2_key, vid.size_bytes = clean_key, size
        for index, wm_name, clean, _key, _size in applied:
            if not retain_snapshot:
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
        media_redirect.forget(proj.job_id)
    if done:
        print(f"🏷️  Watermark removed from {done} clip(s) of user {user_id} after upgrade.")
    return done


def _archive_filename(name, suffixes):
    if (not isinstance(name, str) or name.startswith(".")
            or re.search(r"[/\\\x00-\x1f]", name)
            or not name.endswith(suffixes)):
        raise ValueError("The project snapshot contains an invalid filename.")
    return name


def _clip_filename(job_id, url):
    prefix, _, name = urlsplit(url).path.rpartition("/")
    if prefix not in ("", f"/videos/{job_id}"):
        raise ValueError("The project snapshot contains a clip URL outside its job.")
    return _archive_filename(name, (".mp4",))


def _snapshot_dependencies(job_id, metadata_name, snapshot):
    base_name = _archive_filename(metadata_name, ("_metadata.json",)).removesuffix("_metadata.json")
    rows, dependency_names = [], {}
    suffixes = (".mp4", ".png", ".m4a", ".json")
    for index, clip in enumerate(snapshot["shorts"]):
        if not clip.get("video_url") or (clip.get("quality") or {}).get("valid") is False:
            continue
        filename = _clip_filename(job_id, clip["video_url"])
        rows.append((index, filename, _clip_title(clip)))
        dependency_names[filename] = None
        canonical = _canonical_name(base_name, index)
        if canonical:
            dependency_names[_archive_filename(canonical, suffixes)] = None
        for twin in _twins_to_archive(filename, base_name, index):
            dependency_names[_archive_filename(twin, suffixes)] = None
        before = clip.get("pro_before") or {}
        for reference in (before.get("base"), before.get("served")):
            if reference:
                prior = _archive_filename(reference, (".mp4",))
                dependency_names[prior] = None
                for twin in _twins_to_archive(prior, base_name, index):
                    dependency_names[_archive_filename(twin, suffixes)] = None

        # Editor assets can outlive the local output directory. A saved cut
        # still needs its original videos, music and card image after restore.
        broll_ids = [placement.get("asset_id") for placement in (clip.get("broll") or [])]
        if clip.get("bottom_video_asset_id"):
            broll_ids.append(clip["bottom_video_asset_id"])
        references = [("broll", ident, "mp4") for ident in broll_ids]
        for kind, extension in (("music", "m4a"), ("endcard", "png")):
            ident = (clip.get(kind) or {}).get("asset_id")
            if ident is not None:
                references.append((kind, ident, extension))
        for kind, ident, extension in references:
            if not isinstance(ident, str) or not re.fullmatch(r"[a-f0-9]{32}", ident):
                raise ValueError("The project snapshot contains an invalid editor asset ID.")
            dependency_names[f"{kind}_asset_{ident}.{extension}"] = None
            dependency_names[f"{kind}_asset_{ident}.json"] = None
    return rows, dependency_names


def _ensure_archive_object(directory, name, key):
    """Reuse an immutable media object or upload from one stable local inode."""
    path = os.path.join(directory, name)
    if os.path.islink(path):
        raise ValueError(f"Cannot archive a symbolic link: {name}")
    remote_size = storage.object_size(key)
    if remote_size is not None:
        if remote_size <= 0 or (os.path.isfile(path) and os.path.getsize(path) != remote_size):
            raise ValueError(f"Archived file does not match the snapshot: {name}")
        return remote_size
    if not os.path.isfile(path) and clean_twin.available(directory, name):
        # A free clip may retain its clean picture as a recipe. Keep that
        # recipe and all its inputs durable without eagerly encoding it.
        names = clean_twin.archive_names(directory, name)
        if not names or any(not os.path.isfile(part) for part in clean_twin.recipe_paths(directory, name)):
            raise FileNotFoundError(f"The clean clip recipe is missing: {name}")
        prefix = key.rsplit("/", 1)[0]
        for asset in names:
            _ensure_archive_object(directory, asset, f"{prefix}/{asset}")
        return 0
    if not os.path.isfile(path) and name.startswith(("hooked_", "hook_")):
        from hooks import has_recipe, materialize

        if has_recipe(directory, name):
            materialize(directory, name)
    content_type = mimetypes.guess_type(name)[0] or "application/octet-stream"
    with open(path, "rb") as handle:
        size = os.fstat(handle.fileno()).st_size
        if size <= 0:
            raise ValueError(f"Cannot archive an empty file: {name}")
        storage.upload_fileobj(handle, key, content_type)
    return size


def _archive_revision(state):
    try:
        return int((state or {}).get("archive_revision") or 0)
    except (TypeError, ValueError):
        return 0


async def _publish_archive_revision(user_id, job_id, metadata_key, revision, uploaded, attestation):
    """Move history and project pointers together, conditional on the revision."""
    for attempt in range(2):
        creating = False
        try:
            async with database.session() as session:
                async with session.begin():
                    project = (await session.execute(
                        select(Project).where(Project.job_id == job_id).with_for_update()
                    )).scalar_one_or_none()
                    if project is not None and str(project.user_id) != str(user_id):
                        raise ValueError("The project snapshot does not belong to this owner.")
                    state = dict(project.state or {}) if project is not None else {}
                    if attestation and "rights_attestation" not in state:
                        state["rights_attestation"] = {
                            "acknowledged": bool(attestation.get("acknowledged")),
                            "ip_network": attestation.get("ip"),
                            "user_agent": (attestation.get("user_agent") or "")[:300],
                            "timestamp": attestation.get("timestamp"),
                            "source": attestation.get("source"),
                        }
                    if project is not None and revision <= _archive_revision(state):
                        if state != project.state:
                            project.state = state
                        return False

                    clips = {clip["index"]: dict(clip) for clip in state.get("clips", [])}
                    for index, filename, _key, _title, _size in uploaded:
                        clip = clips.setdefault(index, {"index": index, "active_layers": None})
                        clip["original_file"] = clip.get("original_file") or filename
                        clip["server_file"] = filename
                    state.update(v=state.get("v", 1), clips=list(clips.values()),
                                 archive_revision=str(revision))
                    total = sum(size for _index, _filename, _key, _title, size in uploaded)
                    if project is None:
                        creating = True
                        project = Project(user_id=user_id, job_id=job_id, title=uploaded[0][3],
                                          metadata_r2_key=metadata_key)
                        session.add(project)
                        # Establish the unique job row before creating history
                        # rows. A concurrent creator retries under the row lock.
                        await session.flush()
                    project.metadata_r2_key = metadata_key
                    project.state = state
                    project.size_bytes = total

                    existing = {
                        video.clip_index: video for video in (await session.execute(
                            select(UserVideo).where(UserVideo.user_id == user_id, UserVideo.job_id == job_id)
                        )).scalars()
                    }
                    for index, _filename, key, title, size in uploaded:
                        video = existing.get(index)
                        if video is None:
                            session.add(UserVideo(user_id=user_id, job_id=job_id, clip_index=index,
                                                  r2_key=key, title=title, size_bytes=size))
                        else:
                            video.r2_key, video.title, video.size_bytes = key, title, size
            return True
        except IntegrityError:
            if not creating or attempt:
                raise


async def _archive_snapshot(user_id, job_id, output_dir, metadata_name, metadata_snapshot,
                            revision, rows, dependencies, attestation=None):
    if not rows:
        return
    uploaded_sizes = {}
    for name in dependencies:
        key = storage.job_key(user_id, job_id, name)
        uploaded_sizes[name] = await asyncio.to_thread(
            _ensure_archive_object, output_dir, name, key)
    metadata_key = storage.job_key(
        user_id, job_id, f"metadata/{revision}-{uuid.uuid4().hex}/{metadata_name}")
    with io.BytesIO(metadata_snapshot) as metadata_file:
        await asyncio.to_thread(storage.upload_fileobj, metadata_file, metadata_key, "application/json")
    uploaded = [(index, name, storage.job_key(user_id, job_id, name), title, uploaded_sizes[name])
                for index, name, title in rows]
    committed = await _publish_archive_revision(
        user_id, job_id, metadata_key, revision, uploaded, attestation)
    if committed:
        media_redirect.forget(job_id)
        print(f"☁️  Archived {len(uploaded)} clip(s) to R2 for user {user_id}.")


async def archive_job(user_id, job_id, clips, output_dir, attestation=None):
    """Archive the completed job as an immutable metadata and media snapshot."""
    if not settings.r2_configured or not user_id or not clips:
        return
    metadata_path = next(iter(glob.glob(os.path.join(output_dir, "*_metadata.json"))), None)
    if metadata_path is None or os.path.islink(metadata_path):
        raise FileNotFoundError("The project metadata could not be archived.")
    with open(metadata_path, "rb") as handle:
        metadata_snapshot = handle.read()
        revision = os.fstat(handle.fileno()).st_mtime_ns
    metadata_name = os.path.basename(metadata_path)
    snapshot = json.loads(metadata_snapshot)
    for position, clip in enumerate(clips):
        index = clip.get("clip_index", position)
        if type(index) is not int or not 0 <= index < len(snapshot["shorts"]):
            raise ValueError("The completed clip does not match the project snapshot.")
        # Initial pipeline metadata predates serving URLs. Preserve URLs
        # already recorded by an editor that saved before this task ran.
        if clip.get("video_url") and not snapshot["shorts"][index].get("video_url"):
            snapshot["shorts"][index]["video_url"] = clip["video_url"]
    rows, dependencies = _snapshot_dependencies(job_id, metadata_name, snapshot)
    await _archive_snapshot(user_id, job_id, output_dir, metadata_name,
                            json.dumps(snapshot, ensure_ascii=False).encode(), revision,
                            rows, dependencies, attestation)


async def archive_clip_edit(user_id, job_id, clip_index, output_dir, new_filename, *,
                            metadata_name, metadata_snapshot, revision):
    """Archive a captured edit only after every referenced file is durable."""
    if not settings.r2_configured or not user_id:
        return
    snapshot = json.loads(metadata_snapshot)
    rows, dependencies = _snapshot_dependencies(job_id, metadata_name, snapshot)
    if not any(index == clip_index and name == new_filename for index, name, _title in rows):
        raise ValueError("The edited clip does not match the project snapshot.")
    await _archive_snapshot(user_id, job_id, output_dir, metadata_name, metadata_snapshot,
                            revision, rows, dependencies)

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
                select(Project).where(Project.job_id == job_id).with_for_update()
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
                # Versioned metadata and its server filenames are advanced
                # together by the archive transaction, even when a browser
                # saves its local layers from an older tab.
                if c.get("server_file") and not _archive_revision(state):
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

    A plan that ended keeps its library VIDEO_RETENTION_GRACE_DAYS more
    (Terms, section 13: "deleted 7 days after it ends"); only after that does
    the account's library follow the free per-clip expiry.
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
            if await metering.in_paid_grace(s, user_id, VIDEO_RETENTION_GRACE_DAYS):
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
            # Not about to be deleted: the plan's grace still covers them, and
            # a warning recorded now would never be sent again.
            if await metering.in_paid_grace(s, user_id, VIDEO_RETENTION_GRACE_DAYS):
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
            # Paid jobs' source videos past SOURCE_ARCHIVE_DAYS (once a day).
            from .source_archive import purge_expired_sources
            await purge_expired_sources()
        except asyncio.CancelledError:
            break
        except Exception as e:
            print(f"⚠️  Video retention sweeper error: {e}")


def start_sweeper():
    asyncio.create_task(_sweeper_loop())
