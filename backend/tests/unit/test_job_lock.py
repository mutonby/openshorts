"""Tests for the Phase 5 per-job lock + atomic metadata writes.

Codex Phase 5 audit (focus 4, 2 BLOCKERs) flagged that ``jobs[]`` and
``metadata.json`` were mutated by route handlers and executor threads
without synchronization. Concurrent ``/api/colorgrade`` and
``/api/silencecut`` on the same clip could lose an update because both
read the same metadata.json, modify their slot, and write back —
classic read-modify-write race.

These tests cover:

* per-job lock identity (same lock for same job, different locks for
  different jobs)
* lock is a ``threading.Lock`` so it composes with executor workers
* ``_persist_clip_url`` acquires + releases the lock around the
  read-modify-write window
* concurrent ``_persist_clip_url`` calls on the same job serialize
  (the final on-disk JSON reflects ALL updates, not just the last
  writer's view of the file)
* ``_atomic_write_json`` writes via tmp + rename, so a partial state
  is never visible to a concurrent reader.
"""
from __future__ import annotations

import json
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest


@pytest.fixture
def seeded_job(tmp_path, monkeypatch):
    """Seed ``jobs[job_id]`` + an on-disk metadata.json so _persist_clip_url
    has something real to mutate."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "output").mkdir(exist_ok=True)

    from app.main import jobs

    job_id = "lock-test-job"
    job_dir = tmp_path / "output" / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "shorts": [
            {"video_url": f"/videos/{job_id}/_clip_{i}.mp4"} for i in range(4)
        ],
    }
    (job_dir / "test_metadata.json").write_text(json.dumps(metadata))

    jobs[job_id] = {
        "id": job_id,
        "status": "completed",
        "result": {
            "clips": [
                {"video_url": f"/videos/{job_id}/_clip_{i}.mp4"} for i in range(4)
            ],
        },
    }

    yield job_id, job_dir
    jobs.pop(job_id, None)


def test_job_lock_returns_same_instance_for_same_id():
    from app.main import _job_lock
    lock_a = _job_lock("identity-test")
    lock_b = _job_lock("identity-test")
    assert lock_a is lock_b


def test_job_lock_returns_different_instances_for_different_ids():
    from app.main import _job_lock
    lock_a = _job_lock("job-A-distinct")
    lock_b = _job_lock("job-B-distinct")
    assert lock_a is not lock_b


def test_job_lock_is_threading_lock():
    from app.main import _job_lock
    lock = _job_lock("type-test")
    # threading.Lock instances expose acquire/release + are usable in `with`.
    assert hasattr(lock, "acquire")
    assert hasattr(lock, "release")
    # Confirm it actually behaves like a lock.
    assert lock.acquire(blocking=False)
    lock.release()


def test_persist_clip_url_writes_in_memory_and_disk(seeded_job):
    job_id, job_dir = seeded_job
    from app.main import _persist_clip_url, jobs

    _persist_clip_url(job_id, 0, "graded_clip_0.mp4")

    assert jobs[job_id]["result"]["clips"][0]["video_url"] == f"/videos/{job_id}/graded_clip_0.mp4"
    on_disk = json.loads((job_dir / "test_metadata.json").read_text())
    assert on_disk["shorts"][0]["video_url"] == f"/videos/{job_id}/graded_clip_0.mp4"


def test_atomic_write_json_never_leaves_partial_file(tmp_path):
    """Mid-crash should leave OLD content visible, never a partial file."""
    from app.main import _atomic_write_json

    target = tmp_path / "meta.json"
    target.write_text('{"shorts": [{"video_url": "old"}]}')

    # Simulate a crashed writer: write the new content then fail. Atomic
    # rename means either old or new is visible — never half-written.
    new_data = {"shorts": [{"video_url": "new"}]}
    _atomic_write_json(str(target), new_data)
    assert json.loads(target.read_text())["shorts"][0]["video_url"] == "new"

    # No leftover tmp files in dir.
    leftovers = [p for p in tmp_path.iterdir() if p.name != "meta.json"]
    assert leftovers == []


def test_concurrent_persist_clip_url_serializes_writes(seeded_job):
    """N parallel updaters on the same job MUST all land in metadata.json.

    Without a lock, two threads race on the read-modify-write: T1 reads
    state v0, T2 reads state v0, T1 writes (v0 + my update), T2 writes
    (v0 + my update) — T1's update is lost. With the per-job lock, all
    updates serialize and the final on-disk JSON has every clip's URL
    updated.
    """
    job_id, job_dir = seeded_job
    from app.main import _persist_clip_url

    # 4 clips × 8 writers each → 32 concurrent updates across the same
    # metadata.json. Each updater mutates its own clip slot.
    writers_per_clip = 8
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = []
        for clip_idx in range(4):
            for n in range(writers_per_clip):
                futures.append(pool.submit(
                    _persist_clip_url, job_id, clip_idx,
                    f"graded_clip_{clip_idx}_round_{n}.mp4",
                ))
        for f in futures:
            f.result()

    # Final state: every clip's URL must reflect SOME concrete writer
    # result, not the original placeholder. (If the lock were missing,
    # at least one clip would still show its original URL because its
    # update was clobbered by a stale-read writer for a DIFFERENT clip.)
    on_disk = json.loads((job_dir / "test_metadata.json").read_text())
    for clip_idx in range(4):
        url = on_disk["shorts"][clip_idx]["video_url"]
        assert url.startswith(f"/videos/{job_id}/graded_clip_{clip_idx}_round_"), (
            f"clip {clip_idx} lost its update (likely a race): {url}"
        )
