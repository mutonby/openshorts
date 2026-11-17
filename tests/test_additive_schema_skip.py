"""Boot must not take ACCESS EXCLUSIVE locks for DDL that is already applied.

``ALTER TABLE ... ADD COLUMN IF NOT EXISTS`` locks the table before checking,
so every boot of a new container deadlocked with the requests the previous
one was still serving (DeadlockDetectedError on POST /api/process). Statements
whose column/index already exists are now skipped after a catalog read.
"""
import asyncio

from cloud import database


class _Result:
    def __init__(self, found):
        self._found = found

    def first(self):
        return (1,) if self._found else None


class _Conn:
    def __init__(self, existing):
        self.existing = existing
        self.executed = []

    async def execute(self, stmt, params=None):
        sql = str(stmt)
        if "information_schema.columns" in sql:
            return _Result((params["t"], params["c"]) in self.existing)
        if "pg_indexes" in sql:
            return _Result(params["i"] in self.existing)
        self.executed.append(sql)
        return _Result(False)


def _applied(conn, statement):
    return asyncio.run(database._already_applied(conn, statement))


def test_existing_column_is_skipped():
    conn = _Conn({("users", "ban_reason")})
    assert _applied(conn, "ALTER TABLE users ADD COLUMN IF NOT EXISTS ban_reason TEXT")
    assert conn.executed == []


def test_missing_column_still_runs():
    conn = _Conn(set())
    assert not _applied(conn, "ALTER TABLE users ADD COLUMN IF NOT EXISTS ban_reason TEXT")


def test_index_checked_by_name():
    stmt = "CREATE INDEX IF NOT EXISTS ix_usage_job_id ON usage_ledger (job_id)"
    assert _applied(_Conn({"ix_usage_job_id"}), stmt)
    assert not _applied(_Conn(set()), stmt)


def test_unknown_shape_still_runs():
    assert not _applied(_Conn(set()), "DROP TABLE x")


def test_every_additive_statement_is_recognised():
    # A statement the regexes miss would silently go back to locking on each boot.
    for stmt in database._ADDITIVE_COLUMNS:
        assert (database._ADD_COLUMN_RE.match(stmt)
                or database._CREATE_INDEX_RE.match(stmt)), stmt
