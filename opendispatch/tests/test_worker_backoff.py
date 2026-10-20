"""The worker's retry backoff must be atomic with the status flip.

run_once() used to do this in two steps:

    committed = q.mark_failed(rid, err, dead=dead)   # -> status "queued"
    if committed and not dead:
        q._update(rid, {"scheduled_for": new_sf_iso}) # -> backoff applied

Between those two calls the row is "queued" but still carries the ORIGINAL
past scheduled_for, so any other worker's list_due() sees it as immediately
due and claims it. The exponential backoff is silently defeated, and the
first worker's reschedule then lands on a row the other worker has already
moved to "publishing".

mark_failed() now takes retry_at and applies status + attempts + reschedule
in a single status-guarded write, so there is no window.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from opendispatch.api.queue import get_queue

PAST = "2020-01-01T00:00:00+00:00"


@pytest.fixture()
def q(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("OPEN_DISPATCH_DATA", str(tmp_path))
    monkeypatch.delenv("REDIS_URL", raising=False)
    monkeypatch.delenv("DATABASE_URL", raising=False)
    return get_queue()


def _enqueue(q, unit_id: str = "u1") -> str:
    unit = {"id": unit_id, "targets": ["bluesky"],
            "formats": {"bluesky_post": {"text": "hi"}}, "scheduled_for": None}
    return q.enqueue(unit, "bluesky", PAST)


def _now_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


def test_failed_row_is_not_due_before_its_backoff(q) -> None:
    rid = _enqueue(q)
    assert q.mark_publishing(rid)
    future = (datetime.now(tz=timezone.utc) + timedelta(seconds=45)).isoformat()

    assert q.mark_failed(rid, "boom", dead=False, retry_at=future) is True

    row = q.get(rid)
    assert row["status"] == "queued"
    assert row["scheduled_for"] == future
    # The core regression: a second worker must not see this as due yet.
    assert not any(r["id"] == rid for r in q.list_due(_now_iso()))


def test_second_worker_cannot_claim_a_backing_off_row(q) -> None:
    rid = _enqueue(q)
    q.mark_publishing(rid)
    future = (datetime.now(tz=timezone.utc) + timedelta(seconds=45)).isoformat()
    q.mark_failed(rid, "boom", dead=False, retry_at=future)

    assert q.mark_publishing(rid) is False, "backoff was bypassed"


def test_retry_at_omitted_keeps_legacy_behaviour(q) -> None:
    # Backwards compatible: no retry_at means the scheduled_for is untouched.
    rid = _enqueue(q)
    q.mark_publishing(rid)
    assert q.mark_failed(rid, "boom", dead=False) is True
    row = q.get(rid)
    assert row["status"] == "queued"
    assert row["scheduled_for"] == PAST


def test_dead_rows_are_not_rescheduled(q) -> None:
    rid = _enqueue(q)
    q.mark_publishing(rid)
    future = (datetime.now(tz=timezone.utc) + timedelta(seconds=45)).isoformat()
    assert q.mark_failed(rid, "fatal", dead=True, retry_at=future) is True
    row = q.get(rid)
    assert row["status"] == "dead"
    assert row["scheduled_for"] == PAST, "a dead row must not get a future time"


def test_attempts_still_increments_with_retry_at(q) -> None:
    rid = _enqueue(q)
    q.mark_publishing(rid)
    future = (datetime.now(tz=timezone.utc) + timedelta(seconds=45)).isoformat()
    q.mark_failed(rid, "boom", dead=False, retry_at=future)
    assert q.get(rid)["attempts"] == 1


def test_mark_failed_on_published_row_is_rejected(q) -> None:
    rid = _enqueue(q)
    q.mark_publishing(rid)
    q.mark_published(rid, "post-1")
    future = (datetime.now(tz=timezone.utc) + timedelta(seconds=45)).isoformat()
    assert q.mark_failed(rid, "late", dead=False, retry_at=future) is False
    assert q.get(rid)["status"] == "published"


def test_mark_failed_on_canceled_row_is_rejected(q) -> None:
    rid = _enqueue(q)
    q.mark_publishing(rid)
    q.mark_failed(rid, "boom", dead=False)  # back to queued
    q.cancel_campaign("u1")
    assert q.get(rid)["status"] == "canceled"
    future = (datetime.now(tz=timezone.utc) + timedelta(seconds=45)).isoformat()
    assert q.mark_failed(rid, "again", dead=False, retry_at=future) is False
    assert q.get(rid)["status"] == "canceled"


def test_worker_run_once_does_not_double_reschedule(q, monkeypatch: pytest.MonkeyPatch) -> None:
    """The worker must not patch scheduled_for separately after mark_failed."""
    import opendispatch.scheduler.worker as worker

    rid = _enqueue(q)
    monkeypatch.setattr(worker, "_publish", lambda row: (False, "", "adapter down"))
    monkeypatch.setattr(worker, "get_queue", lambda: q)
    monkeypatch.setattr(worker, "MAX_ATTEMPTS", 3)
    monkeypatch.setattr(worker, "_fire_webhook", lambda url, payload: None)

    calls: list[dict] = []
    monkeypatch.setattr(q, "_update", lambda row_id, patch: calls.append(patch))

    worker.run_once()

    row = q.get(rid)
    assert row["status"] == "queued"
    assert row["attempts"] == 1
    # A future scheduled_for must already be set by mark_failed itself.
    assert row["scheduled_for"] > _now_iso(), "backoff not applied in the same write"
    assert not any(p == {"scheduled_for": PAST} for p in calls), calls
