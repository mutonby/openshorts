"""RedisQueue tests — backed by a minimal in-memory fake redis client.

We don't pull a real Redis or fakeredis dep just for tests; the surface
RedisQueue uses (get/set/mget/sadd/smembers/zadd/zrem/zrangebyscore/pipeline)
is small enough to stub directly.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

from opendispatch.api.queue import RedisQueue


class FakeRedis:
    """Dict-based stub for the slice of Redis we actually use."""

    def __init__(self) -> None:
        self.kv: dict[str, str] = {}
        self.sets: dict[str, set[str]] = {}
        self.zsets: dict[str, dict[str, float]] = {}

    # ── kv ──
    def get(self, key: str) -> str | None:
        return self.kv.get(key)

    def set(self, key: str, value: str) -> None:
        self.kv[key] = value

    def mget(self, keys: list[str]) -> list[str | None]:
        return [self.kv.get(k) for k in keys]

    # ── sets ──
    def sadd(self, key: str, value: str) -> None:
        self.sets.setdefault(key, set()).add(value)

    def smembers(self, key: str) -> set[str]:
        return set(self.sets.get(key, set()))

    # ── zsets ──
    def zadd(self, key: str, mapping: dict[str, float]) -> None:
        z = self.zsets.setdefault(key, {})
        z.update(mapping)

    def zrem(self, key: str, member: str) -> None:
        self.zsets.get(key, {}).pop(member, None)

    def zrangebyscore(self, key: str, lo: float, hi: float) -> list[str]:
        return [m for m, score in self.zsets.get(key, {}).items() if lo <= score <= hi]

    # ── pipeline ──
    def pipeline(self):
        return FakePipeline(self)


class FakePipeline:
    def __init__(self, parent: FakeRedis) -> None:
        self._parent = parent
        self._ops: list[tuple] = []

    def set(self, key: str, value: str) -> "FakePipeline":
        self._ops.append(("set", key, value))
        return self

    def sadd(self, key: str, value: str) -> "FakePipeline":
        self._ops.append(("sadd", key, value))
        return self

    def zadd(self, key: str, mapping: dict[str, float]) -> "FakePipeline":
        self._ops.append(("zadd", key, mapping))
        return self

    def execute(self) -> None:
        for op in self._ops:
            if op[0] == "set":
                self._parent.kv[op[1]] = op[2]
            elif op[0] == "sadd":
                self._parent.sets.setdefault(op[1], set()).add(op[2])
            elif op[0] == "zadd":
                self._parent.zsets.setdefault(op[1], {}).update(op[2])


# ─── Tests ──────────────────────────────────────────────────────────────

def _now_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


def test_enqueue_stores_row_and_indexes():
    r = FakeRedis()
    q = RedisQueue(r)
    rid = q.enqueue({"hello": "world"}, "twitter:default", _now_iso())
    # Row stored
    raw = r.kv[f"dispatch:row:{rid}"]
    row = json.loads(raw)
    assert row["id"] == rid
    assert row["status"] == "queued"
    # Indexed in all_ids set + due zset
    assert rid in r.sets["dispatch:all_ids"]
    assert rid in r.zsets["dispatch:due"]


def test_get_returns_row():
    r = FakeRedis()
    q = RedisQueue(r)
    rid = q.enqueue({}, "tg:default", _now_iso())
    got = q.get(rid)
    assert got is not None
    assert got["id"] == rid


def test_get_missing_returns_none():
    q = RedisQueue(FakeRedis())
    assert q.get("nope") is None


def test_list_all_returns_all_rows():
    r = FakeRedis()
    q = RedisQueue(r)
    ids = [q.enqueue({}, f"tg:{i}", _now_iso()) for i in range(3)]
    rows = q.list_all()
    assert len(rows) == 3
    assert {row["id"] for row in rows} == set(ids)


def test_list_all_filters_by_status():
    r = FakeRedis()
    q = RedisQueue(r)
    rid_pub = q.enqueue({}, "tg:p", _now_iso())
    rid_q = q.enqueue({}, "tg:q", _now_iso())
    q.mark_publishing(rid_pub)
    q.mark_published(rid_pub, "post_1")

    published = q.list_all(status="published")
    queued = q.list_all(status="queued")
    assert len(published) == 1 and published[0]["id"] == rid_pub
    assert len(queued) == 1 and queued[0]["id"] == rid_q


def test_list_due_only_returns_past_due_queued():
    r = FakeRedis()
    q = RedisQueue(r)
    past = (datetime.now(tz=timezone.utc) - timedelta(minutes=5)).isoformat()
    future = (datetime.now(tz=timezone.utc) + timedelta(hours=1)).isoformat()
    past_id = q.enqueue({}, "tg:a", past)
    future_id = q.enqueue({}, "tg:b", future)
    due = q.list_due(_now_iso())
    assert past_id in {r["id"] for r in due}
    assert future_id not in {r["id"] for r in due}


def test_mark_publishing_then_published_clears_from_due_zset():
    r = FakeRedis()
    q = RedisQueue(r)
    rid = q.enqueue({}, "tg:default", _now_iso())
    assert rid in r.zsets["dispatch:due"]
    q.mark_publishing(rid)
    assert rid not in r.zsets["dispatch:due"]
    q.mark_published(rid, "p_1")
    row = q.get(rid)
    assert row["status"] == "published"
    assert row["post_id"] == "p_1"


def test_mark_publishing_skips_if_not_queued():
    """Multi-worker safety: another worker grabbed it first."""
    r = FakeRedis()
    q = RedisQueue(r)
    rid = q.enqueue({}, "tg:default", _now_iso())
    q.mark_publishing(rid)  # first worker
    # Second worker calls again — should be a no-op since status != queued
    q.mark_publishing(rid)
    row = q.get(rid)
    # Still 'publishing' — no double-flip
    assert row["status"] == "publishing"


def test_mark_failed_requeues_and_bumps_attempts():
    r = FakeRedis()
    q = RedisQueue(r)
    rid = q.enqueue({}, "tg:default", _now_iso())
    q.mark_publishing(rid)
    q.mark_failed(rid, "transient error")
    row = q.get(rid)
    assert row["status"] == "queued"
    assert row["attempts"] == 1
    assert row["last_error"] == "transient error"
    # Back in the due zset for the worker to pick up
    assert rid in r.zsets["dispatch:due"]


def test_mark_failed_dead_doesnt_requeue():
    r = FakeRedis()
    q = RedisQueue(r)
    rid = q.enqueue({}, "tg:default", _now_iso())
    q.mark_publishing(rid)
    q.mark_failed(rid, "permanent err", dead=True)
    row = q.get(rid)
    assert row["status"] == "dead"
    assert rid not in r.zsets["dispatch:due"]


def test_cancel_campaign_flips_only_queued_and_leaves_zset():
    r = FakeRedis()
    q = RedisQueue(r)
    unit = {"id": "camp-9"}
    rid_q1 = q.enqueue(unit, "tg:a", _now_iso())
    rid_q2 = q.enqueue(unit, "tw:b", _now_iso())
    rid_pub = q.enqueue(unit, "bs:c", _now_iso())
    q.mark_publishing(rid_pub)
    q.mark_published(rid_pub, "p-1")
    # a foreign campaign must survive
    rid_other = q.enqueue({"id": "camp-10"}, "tg:x", _now_iso())

    canceled = q.cancel_campaign("camp-9")
    assert {row["id"] for row in canceled} == {rid_q1, rid_q2}
    assert all(row["status"] == "canceled" for row in canceled)
    assert q.get(rid_pub)["status"] == "published"
    assert q.get(rid_other)["status"] == "queued"
    due = r.zsets["dispatch:due"]
    assert rid_q1 not in due and rid_q2 not in due and rid_other in due


def test_factory_falls_back_to_jsonl_when_no_redis_url(monkeypatch):
    monkeypatch.delenv("REDIS_URL", raising=False)
    from opendispatch.api.queue import JsonlQueue, _reset_singleton_for_tests, get_queue
    _reset_singleton_for_tests()
    q = get_queue()
    assert isinstance(q, JsonlQueue)


def test_factory_falls_back_when_redis_unavailable(monkeypatch):
    """REDIS_URL set but the redis package can't connect → JSONL fallback."""
    monkeypatch.setenv("REDIS_URL", "redis://nonexistent.invalid:6379/0")
    from opendispatch.api.queue import JsonlQueue, _reset_singleton_for_tests, get_queue
    _reset_singleton_for_tests()
    q = get_queue()
    # Either redis isn't installed (returns JsonlQueue) or it is installed but
    # can't reach the host (also returns JsonlQueue). Both acceptable.
    assert isinstance(q, JsonlQueue)
