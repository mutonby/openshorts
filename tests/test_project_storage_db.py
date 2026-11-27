"""Exercise project row locks and deletion against the disposable CI Postgres."""
import asyncio
from uuid import uuid4

import pytest
from fastapi import HTTPException
from sqlalchemy import select

from cloud import project_deletion, videos
from cloud.models import AutopilotRun, ClipExpiryWarning, Project, ProjectDeletion, User, UserVideo
from test_support_ops import clean, db, needs_db  # noqa: F401

pytestmark = needs_db


async def seed(database):
    owner, other, job = uuid4(), uuid4(), str(uuid4())
    async with database.session() as session:
        session.add_all([User(id=owner, email="owner@example.test"), User(id=other, email="other@example.test")])
        await session.commit()
    return owner, other, job


def test_concurrent_first_archives_keep_one_complete_newest_revision(clean):
    async def run():
        owner, _other, job = await seed(clean)
        async def publish(revision):
            return await videos._publish_archive_revision(owner, job, f"metadata/{revision}", revision,
                [(0, f"clip-{revision}.mp4", f"key-{revision}", str(revision), revision)], None)
        await asyncio.gather(*(publish(revision) for revision in (12, 14, 13, 11)))
        async with clean.session() as session:
            projects = list((await session.execute(select(Project).where(Project.job_id == job))).scalars())
            history = list((await session.execute(select(UserVideo).where(UserVideo.job_id == job))).scalars())
            assert len(projects) == len(history) == 1
            assert projects[0].metadata_r2_key == "metadata/14"
            assert projects[0].state["archive_revision"] == "14"
            assert projects[0].state["clips"][0]["server_file"] == "clip-14.mp4"
            assert history[0].r2_key == "key-14"
    asyncio.run(run())


def test_archive_preserves_browser_layers_and_checks_owner(clean):
    async def run():
        owner, other, job = await seed(clean)
        async with clean.session() as session:
            session.add(Project(user_id=owner, job_id=job, metadata_r2_key="old", state={"clips": [
                {"index": 0, "original_file": "original.mp4", "active_layers": {"hook": "Keep me"}}]}))
            await session.commit()
        await videos._publish_archive_revision(owner, job, "metadata/2", 2,
            [(0, "new.mp4", "new-key", "new", 2)], None)
        with pytest.raises(ValueError, match="owner"):
            await videos._publish_archive_revision(other, job, "foreign", 3,
                [(0, "bad.mp4", "bad-key", "bad", 3)], None)
        async with clean.session() as session:
            project = (await session.execute(select(Project).where(Project.job_id == job))).scalar_one()
            assert project.state["clips"][0] == {"index": 0, "original_file": "original.mp4",
                "server_file": "new.mp4", "active_layers": {"hook": "Keep me"}}
            assert project.metadata_r2_key == "metadata/2"
    asyncio.run(run())


def test_deletion_is_owner_scoped_retryable_and_keeps_autopilot_dedupe(clean):
    async def run():
        owner, other, job = await seed(clean)
        sibling = str(uuid4())
        async with clean.session() as session:
            session.add_all([Project(user_id=owner, job_id=job, metadata_r2_key="meta"),
                Project(user_id=other, job_id=sibling, metadata_r2_key="sibling")])
            video = UserVideo(user_id=owner, job_id=job, clip_index=0, r2_key="clip")
            session.add(video)
            await session.flush()
            session.add(ClipExpiryWarning(user_id=owner, video_id=video.id))
            session.add(AutopilotRun(user_id=owner, video_id="already-seen", job_id=job))
            await session.commit()
        with pytest.raises(HTTPException) as rejected:
            await project_deletion.begin(other, job, None)
        assert rejected.value.status_code == 404
        assert await project_deletion.begin(owner, job, None) == (False, True)
        assert await project_deletion.is_deleted(job)
        assert await project_deletion.begin(owner, job, None) == (False, True)
        await project_deletion.finish(owner, job)
        assert await project_deletion.begin(owner, job, None) == (True, False)
        async with clean.session() as session:
            assert not list((await session.execute(select(UserVideo).where(UserVideo.job_id == job))).scalars())
            assert not list((await session.execute(select(ClipExpiryWarning))).scalars())
            assert (await session.execute(select(Project).where(Project.job_id == sibling))).scalar_one()
            run = (await session.execute(select(AutopilotRun))).scalar_one()
            assert run.video_id == "already-seen" and run.job_id is None
            assert (await session.get(ProjectDeletion, job)).completed
    asyncio.run(run())
