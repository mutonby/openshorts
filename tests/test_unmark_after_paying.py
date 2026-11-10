"""A free job that lands after its owner paid is delivered without the mark.

The upgrade webhook unmarks the clips that exist at that moment. A video
still waiting in the queue (or rendering) was made for the free plan, so it
landed with the mark although its owner was already paying: the same broken
promise as the queue. At delivery, ``app._unmark_if_owner_paid`` runs the
upgrade's re-pointing (``videos.unmark_user_library``) for that job only.
Canonical files stay clean throughout; the mark lives only on ``wm_``.
"""
import asyncio
import json
from types import SimpleNamespace

import pytest

import watermarked

app = pytest.importorskip("app")

OWNER = "a11ce000-0000-0000-0000-000000000000"


@pytest.fixture
def marked_job(tmp_path, monkeypatch):
    monkeypatch.setattr(app, "OUTPUT_DIR", str(tmp_path))
    monkeypatch.setattr(app, "BILLING_ENABLED", True)
    monkeypatch.setattr(app, "jobs", {})
    d = tmp_path / "j1"
    d.mkdir()
    (d / "t_clip_1.mp4").write_bytes(b"clean")
    (d / "wm_t_clip_1.mp4").write_bytes(b"marked")
    watermarked.mark_job(str(d))
    (d / "t_metadata.json").write_text(json.dumps({
        "shorts": [{"video_url": "/videos/j1/wm_t_clip_1.mp4"}]}))
    job = {"status": "completed", "user_id": OWNER, "output_dir": str(d),
           "result": {"clips": [{"video_url": "/videos/j1/wm_t_clip_1.mp4"}]},
           "ready_files": {0: "wm_t_clip_1.mp4"}}
    app.jobs["j1"] = job
    state = {"live": {OWNER: "starter"}, "unmark_calls": [], "lookups": 0}

    async def lookup(owners):
        state["lookups"] += 1
        if state["live"] is None:
            raise RuntimeError("db down")
        return {u: p for u, p in state["live"].items() if u in owners}

    async def unmark(user_id, job_id=None):
        # What the R2 re-pointing does once the clean twin is found there.
        state["unmark_calls"].append((user_id, job_id))
        app._unmark_local_job(job_id, 0, "t_clip_1.mp4")
        return 1

    monkeypatch.setattr(app, "_owner_live_plans", lookup)
    monkeypatch.setattr(app, "cloud", SimpleNamespace(
        videos=SimpleNamespace(unmark_user_library=unmark)))
    return d, job, state


def test_owner_paid_meanwhile_gets_it_clean(marked_job):
    d, job, state = marked_job
    asyncio.run(app._unmark_if_owner_paid("j1", job))
    assert state["unmark_calls"] == [(OWNER, "j1")], "only this job, not the library"
    assert job["result"]["clips"][0]["video_url"] == "/videos/j1/t_clip_1.mp4"
    assert not watermarked.job_is_marked(str(d))
    assert (d / "t_clip_1.mp4").read_bytes() == b"clean"   # canonical untouched
    assert not (d / "wm_t_clip_1.mp4").exists()
    assert app._media_guard("j1/t_clip_1.mp4")             # guard opens with the marker


def test_free_owner_keeps_the_mark(marked_job):
    d, job, state = marked_job
    state["live"] = {}
    asyncio.run(app._unmark_if_owner_paid("j1", job))
    assert state["unmark_calls"] == []
    assert watermarked.job_is_marked(str(d))
    assert not app._media_guard("j1/t_clip_1.mp4")


def test_lookup_failure_leaves_it_marked_and_does_not_raise(marked_job):
    d, job, state = marked_job
    state["live"] = None
    asyncio.run(app._unmark_if_owner_paid("j1", job))
    assert watermarked.job_is_marked(str(d))


def test_unmarked_or_failed_jobs_are_not_looked_up(marked_job):
    d, job, state = marked_job
    watermarked.unmark_job(str(d))
    asyncio.run(app._unmark_if_owner_paid("j1", job))
    job["status"] = "failed"
    watermarked.mark_job(str(d))
    asyncio.run(app._unmark_if_owner_paid("j1", job))
    assert state["lookups"] == 0


def test_unmark_runs_before_the_owner_is_told():
    import inspect
    src = inspect.getsource(app.run_job_wrapper)
    i = src.index("_unmark_if_owner_paid(")
    assert src.index("_archive_managed_job(") < i
    assert i < src.index("_notify_job_webhook(") and i < src.index("_notify_clips_ready(")


def test_library_unmark_can_be_limited_to_one_job(monkeypatch):
    videos = pytest.importorskip("cloud.videos")
    from sqlalchemy.dialects import postgresql
    seen = []

    class _R:
        def scalars(self):
            return iter(())

    class _S:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        async def execute(self, stmt):
            seen.append(str(stmt.compile(dialect=postgresql.dialect())))
            return _R()

    monkeypatch.setattr(videos.database, "session", lambda: _S())
    monkeypatch.setattr(videos, "settings", SimpleNamespace(r2_configured=True))
    asyncio.run(videos.unmark_user_library(OWNER, job_id="j1"))
    asyncio.run(videos.unmark_user_library(OWNER))
    where = [sql.split("WHERE", 1)[1] for sql in seen]
    assert "projects.job_id" in where[0]
    assert "projects.job_id" not in where[1]
