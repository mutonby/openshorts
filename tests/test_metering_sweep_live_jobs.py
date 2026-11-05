"""The 3-hour reservation sweep must leave live jobs alone.

A job still queued or running settles its own reservation when it ends. If the
sweep refunds it first, the user sees their minutes back while the job still
holds their only job slot, and the clips it delivers are never charged.
"""
import asyncio
from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest

from cloud import metering


def _fake_db(rows):
    class _Session:
        async def execute(self, _stmt):
            return SimpleNamespace(all=lambda: list(rows))

    @asynccontextmanager
    async def session():
        yield _Session()

    return session


@pytest.fixture
def released(monkeypatch):
    out = []

    async def release(lid):
        out.append(lid)

    monkeypatch.setattr(metering, "release_reservation", release)
    return out


ROWS = [("res-live", "job-queued"), ("res-dead", "job-gone")]


def test_sweep_skips_reservations_of_live_jobs(monkeypatch, released):
    monkeypatch.setattr(metering.database, "session", _fake_db(ROWS))
    asyncio.run(metering._sweep_once(is_job_live=lambda jid: jid == "job-queued"))
    assert released == ["res-dead"]


def test_sweep_without_probe_releases_everything(monkeypatch, released):
    monkeypatch.setattr(metering.database, "session", _fake_db(ROWS))
    asyncio.run(metering._sweep_once())
    assert released == ["res-live", "res-dead"]


def test_app_live_probe_matches_job_limit():
    app = pytest.importorskip("app")
    app.jobs["t-queued"] = {"status": "queued"}
    app.jobs["t-processing"] = {"status": "processing"}
    app.jobs["t-failed"] = {"status": "failed"}
    app._begin_submission("t-submitting", "u1")
    try:
        assert app._job_is_live("t-queued")
        assert app._job_is_live("t-processing")
        assert app._job_is_live("t-submitting")
        assert not app._job_is_live("t-failed")
        assert not app._job_is_live("t-unknown")
    finally:
        for k in ("t-queued", "t-processing", "t-failed"):
            app.jobs.pop(k, None)
        app._end_submission("t-submitting")
