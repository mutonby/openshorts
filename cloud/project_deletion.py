"""Durable owner-scoped deletion for managed projects."""
import asyncio

from fastapi import HTTPException
from sqlalchemy import delete, select, update
from sqlalchemy.dialects.postgresql import insert

from . import database, media_redirect, source_archive, storage
from .config import settings
from .models import AutopilotRun, ClipExpiryWarning, Project, ProjectDeletion, UserVideo


async def is_deleted(job_id):
    async with database.session() as session:
        return await session.get(ProjectDeletion, job_id) is not None


async def _owned_state(session, user_id, job_id, local_owner):
    archived = set((await session.execute(
        select(Project.user_id).where(Project.job_id == job_id).union(
            select(UserVideo.user_id).where(UserVideo.job_id == job_id))
    )).scalars())
    state = await session.get(ProjectDeletion, job_id)
    owners = {str(owner) for owner in archived}
    if local_owner is not None:
        owners.add(str(local_owner))
    if state is not None:
        owners.add(str(state.user_id))
    if owners != {str(user_id)}:
        raise HTTPException(status_code=404, detail="Project not found")
    return state, bool(archived)


async def require_owner(user_id, job_id, local_owner):
    async with database.session() as session:
        await _owned_state(session, user_id, job_id, local_owner)


async def begin(user_id, job_id, local_owner):
    """Verify every available owner and commit a deletion marker before purging."""
    async with database.session() as session:
        async with session.begin():
            state, archived = await _owned_state(session, user_id, job_id, local_owner)
            if state is None:
                await session.execute(insert(ProjectDeletion).values(
                    job_id=job_id, user_id=user_id, completed=False).on_conflict_do_nothing())
                state = await session.get(ProjectDeletion, job_id)
            if state is None or str(state.user_id) != str(user_id):
                raise HTTPException(status_code=404, detail="Project not found")
            return state.completed, archived


async def purge_archive(user_id, job_id, archived):
    if settings.r2_configured:
        await asyncio.to_thread(storage.delete_prefix, storage.job_key(user_id, job_id, ""), strict=True)
    elif archived:
        raise RuntimeError("Project storage is unavailable")


async def finish(user_id, job_id):
    """Remove the library rows together and preserve Autopilot upload dedupe."""
    async with database.session() as session:
        async with session.begin():
            videos = select(UserVideo.id).where(UserVideo.user_id == user_id, UserVideo.job_id == job_id)
            await session.execute(delete(ClipExpiryWarning).where(
                ClipExpiryWarning.user_id == user_id, ClipExpiryWarning.video_id.in_(videos)))
            await session.execute(delete(UserVideo).where(UserVideo.user_id == user_id, UserVideo.job_id == job_id))
            await session.execute(delete(Project).where(Project.user_id == user_id, Project.job_id == job_id))
            await session.execute(update(AutopilotRun).where(
                AutopilotRun.user_id == user_id, AutopilotRun.job_id == job_id).values(job_id=None))
            await session.execute(update(ProjectDeletion).where(
                ProjectDeletion.user_id == user_id, ProjectDeletion.job_id == job_id).values(completed=True))
    media_redirect.forget(job_id)
    source_archive._forget(user_id, job_id)
