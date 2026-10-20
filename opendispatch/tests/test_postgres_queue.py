"""PostgresQueue tests — uses a stub connection (no real Postgres needed).

We don't pull psycopg2 just for tests. The stub records every SQL statement
+ its parameters, so the assertions verify the queue *generates correct SQL*
rather than executing it against a real engine. Integration tests with a
real Postgres are a separate suite (CI-only).
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

from opendispatch.api.queue import PostgresQueue


class FakeCursor:
    """Records execute() calls, supports fetchone/fetchall via injected results."""

    def __init__(self, conn: "FakeConn") -> None:
        self._conn = conn
        self._next_result: list[tuple] = []

    def execute(self, sql: str, params: tuple = ()) -> None:
        self._conn.executions.append((sql.strip(), params))

    def fetchone(self) -> tuple | None:
        return self._next_result[0] if self._next_result else None

    def fetchall(self) -> list[tuple]:
        return list(self._next_result)

    def __enter__(self) -> "FakeCursor":
        return self

    def __exit__(self, *exc) -> None:
        pass


class FakeConn:
    """Minimal psycopg2-style connection stub."""

    def __init__(self) -> None:
        self.executions: list[tuple[str, tuple]] = []
        self.cursor_result: list[tuple] = []

    def cursor(self) -> FakeCursor:
        c = FakeCursor(self)
        c._next_result = self.cursor_result
        return c

    def __enter__(self) -> "FakeConn":
        return self

    def __exit__(self, *exc) -> None:
        pass


def _make_queue() -> tuple[PostgresQueue, FakeConn]:
    """Build a PostgresQueue + return the latest FakeConn so we can assert.

    The factory returns a FRESH conn each call. We capture the *last* one so
    tests can inspect what SQL the most-recent operation ran.
    """
    state = {"conn": None}

    def factory():
        c = FakeConn()
        state["conn"] = c
        return c

    q = PostgresQueue(factory)
    # The constructor already ran one execute (CREATE TABLE), so reset
    state["conn"].executions.clear()  # type: ignore[union-attr]
    return q, state["conn"]  # type: ignore[return-value]


# ─── Tests ──────────────────────────────────────────────────────────────

def test_ensure_schema_runs_ddl_on_construction():
    captured = {"conn": None}

    def factory():
        c = FakeConn()
        captured["conn"] = c
        return c

    PostgresQueue(factory)
    assert captured["conn"] is not None
    ddl_sql, _ = captured["conn"].executions[0]  # type: ignore[union-attr]
    assert "CREATE TABLE IF NOT EXISTS open_dispatch_queue" in ddl_sql
    assert "CREATE INDEX IF NOT EXISTS" in ddl_sql


def test_enqueue_generates_insert():
    q, _ = _make_queue()
    # The factory builds a new conn for each call, so we need to inspect
    # the conn captured *during* enqueue.
    state = {"executions": []}

    def factory():
        c = FakeConn()
        # Capture executions after enqueue finishes
        state["last"] = c  # type: ignore[index]
        return c

    q2 = PostgresQueue(factory)
    state["executions"] = []  # reset post-constructor
    state["last"].executions.clear()  # type: ignore[union-attr,index]

    rid = q2.enqueue({"hello": "world"}, "twitter:default",
                     datetime.now(tz=timezone.utc).isoformat())
    assert len(rid) == 36  # UUID length

    last = state["last"]  # type: ignore[index]
    sql, params = last.executions[-1]
    assert "INSERT INTO open_dispatch_queue" in sql
    # Params: (id, json_unit, platform, scheduled_for)
    assert params[0] == rid
    assert json.loads(params[1]) == {"hello": "world"}
    assert params[2] == "twitter:default"


def test_list_all_generates_select():
    state = {}

    def factory():
        c = FakeConn()
        c.cursor_result = []
        state["last"] = c
        return c

    q = PostgresQueue(factory)
    state["last"].executions.clear()  # type: ignore[union-attr]
    q.list_all()
    sql, params = state["last"].executions[0]  # type: ignore[union-attr]
    assert "SELECT * FROM open_dispatch_queue" in sql
    assert "ORDER BY created_at DESC" in sql
    assert params == ()


def test_list_all_with_status_filters():
    state = {}

    def factory():
        c = FakeConn()
        state["last"] = c
        return c

    q = PostgresQueue(factory)
    state["last"].executions.clear()  # type: ignore[union-attr]
    q.list_all(status="failed")
    sql, params = state["last"].executions[0]  # type: ignore[union-attr]
    assert "WHERE status = %s" in sql
    assert params == ("failed",)


def test_list_due_filters_by_status_and_time():
    state = {}

    def factory():
        c = FakeConn()
        c.cursor_result = []
        state["last"] = c
        return c

    q = PostgresQueue(factory)
    state["last"].executions.clear()  # type: ignore[union-attr]
    q.list_due("2026-05-21T12:00:00+00:00")
    sql, _ = state["last"].executions[0]  # type: ignore[union-attr]
    assert "status = 'queued'" in sql
    assert "scheduled_for <= %s::timestamptz" in sql


def test_mark_publishing_uses_conditional_update():
    state = {}

    def factory():
        c = FakeConn()
        state["last"] = c
        return c

    q = PostgresQueue(factory)
    state["last"].executions.clear()  # type: ignore[union-attr]
    q.mark_publishing("abc-123")
    sql, params = state["last"].executions[0]  # type: ignore[union-attr]
    # Must condition on status = 'queued' so two workers don't double-claim
    assert "WHERE id = %s AND status = 'queued'" in sql
    assert params == ("abc-123",)


def test_mark_published_updates_post_id_and_clears_error():
    state = {}

    def factory():
        c = FakeConn()
        state["last"] = c
        return c

    q = PostgresQueue(factory)
    state["last"].executions.clear()  # type: ignore[union-attr]
    q.mark_published("abc-123", "post_999")
    sql, params = state["last"].executions[0]  # type: ignore[union-attr]
    assert "UPDATE open_dispatch_queue" in sql
    assert "post_id" in sql
    assert "last_error" in sql
    assert "published" in sql
    assert "post_999" in params


def test_mark_failed_increments_attempts():
    state = {}

    def factory():
        c = FakeConn()
        state["last"] = c
        return c

    q = PostgresQueue(factory)
    state["last"].executions.clear()  # type: ignore[union-attr]
    q.mark_failed("abc-123", "rate limit")
    sql, params = state["last"].executions[0]  # type: ignore[union-attr]
    # Uses attempts = attempts + 1 (no read-modify-write)
    assert "attempts = attempts + 1" in sql
    # scheduled_for is rescheduled in the same statement via COALESCE, so the
    # params are (status, error, retry_at, row_id). With no retry_at, COALESCE
    # leaves scheduled_for untouched.
    assert "COALESCE(%s, scheduled_for)" in sql
    assert params == ("queued", "rate limit", None, "abc-123")


def test_mark_failed_applies_retry_at_in_the_same_statement():
    state = {}

    def factory():
        c = FakeConn()
        state["last"] = c
        return c

    q = PostgresQueue(factory)
    state["last"].executions.clear()  # type: ignore[union-attr]
    q.mark_failed("abc-123", "boom", retry_at="2099-01-01T00:00:00+00:00")
    _sql, params = state["last"].executions[0]  # type: ignore[union-attr]
    assert params == ("queued", "boom", "2099-01-01T00:00:00+00:00", "abc-123")


def test_mark_failed_dead_does_not_reschedule():
    state = {}

    def factory():
        c = FakeConn()
        state["last"] = c
        return c

    q = PostgresQueue(factory)
    state["last"].executions.clear()  # type: ignore[union-attr]
    q.mark_failed("abc-123", "fatal", dead=True, retry_at="2099-01-01T00:00:00+00:00")
    _sql, params = state["last"].executions[0]  # type: ignore[union-attr]
    # A dead row must keep its original scheduled_for, so the retry_at is
    # forced to None even though the caller passed one.
    assert params == ("dead", "fatal", None, "abc-123")


def test_mark_publishing_requires_the_row_to_be_due():
    state = {}

    def factory():
        c = FakeConn()
        state["last"] = c
        return c

    q = PostgresQueue(factory)
    state["last"].executions.clear()  # type: ignore[union-attr]
    q.mark_publishing("abc-123")
    sql, _params = state["last"].executions[0]  # type: ignore[union-attr]
    # The due guard prevents a worker with a stale id from claiming a row that
    # another worker has backed off to a future scheduled_for.
    assert "scheduled_for <= now()" in sql


def test_mark_failed_dead_sets_dead_status():
    state = {}

    def factory():
        c = FakeConn()
        state["last"] = c
        return c

    q = PostgresQueue(factory)
    state["last"].executions.clear()  # type: ignore[union-attr]
    q.mark_failed("abc-123", "permanent", dead=True)
    sql, params = state["last"].executions[0]  # type: ignore[union-attr]
    assert params[0] == "dead"


def test_cancel_campaign_generates_guarded_update_returning():
    state = {}

    def factory():
        c = FakeConn()
        c.cursor_result = [
            # (id, unit, platform, scheduled_for, status, attempts, post_id, last_error, created_at, updated_at)
            ("11111111-1111-1111-1111-111111111111", {"id": "camp-9"}, "twitter:default",
             None, "canceled", 0, None, None, None, None),
        ]
        state["last"] = c
        return c

    q = PostgresQueue(factory)
    state["last"].executions.clear()  # type: ignore[union-attr]
    rows = q.cancel_campaign("camp-9")
    sql, params = state["last"].executions[0]  # type: ignore[union-attr]
    assert "SET status = 'canceled'" in sql
    assert "(unit->>'id') = %s" in sql
    assert "status = 'queued'" in sql  # the guard: only still-queued rows
    assert "RETURNING *" in sql
    assert params == ("camp-9",)
    assert rows[0]["status"] == "canceled"
    assert rows[0]["platform"] == "twitter:default"


def test_factory_falls_back_when_database_url_unreachable(monkeypatch):
    """DATABASE_URL set but psycopg2 import or connect fails → JSONL."""
    monkeypatch.setenv("DATABASE_URL", "postgres://nonexistent.invalid:5432/x")
    monkeypatch.delenv("REDIS_URL", raising=False)
    from opendispatch.api.queue import JsonlQueue, _reset_singleton_for_tests, get_queue
    _reset_singleton_for_tests()
    q = get_queue()
    # Either psycopg2 isn't installed (returns JsonlQueue) or it is but can't
    # connect (also JsonlQueue). Both acceptable.
    assert isinstance(q, JsonlQueue)
