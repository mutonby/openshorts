"""What keeps the API answering when the host is short of CPU (16-nov):
the owner check of a status poll skips the full user load, the DB pool has
room and a short wait, and a source sent to the remote worker keeps its lease
alive for the whole transfer."""
import asyncio
import os
import time

import app as app_module
import remote_worker as rw
from cloud import database


class _User:
    def __init__(self, uid):
        self.id = uid


class _Req:
    def __init__(self, path, method="GET", auth="Bearer t1"):
        self.method = method
        self.headers = {"Authorization": auth} if auth else {}

        class _U:
            pass
        self.url = _U()
        self.url.path = path


def _counting_loader(monkeypatch, user):
    calls = {"n": 0}

    async def _from_request(request):
        calls["n"] += 1
        return user["u"]
    monkeypatch.setattr(app_module, "_user_from_request", _from_request)
    monkeypatch.setattr(app_module, "_owner_ids", {})
    return calls


def test_repeated_polls_load_the_user_once(monkeypatch):
    user = {"u": _User("owner")}
    calls = _counting_loader(monkeypatch, user)
    for _ in range(5):
        assert asyncio.run(app_module._owner_id_from_request(_Req("/api/status/j1"))) == "owner"
    assert calls["n"] == 1
    # Another credential or another URL is looked up on its own.
    asyncio.run(app_module._owner_id_from_request(_Req("/api/status/j1", auth="Bearer t2")))
    asyncio.run(app_module._owner_id_from_request(_Req("/api/status/j2")))
    assert calls["n"] == 3


def test_writes_anonymous_and_unknown_callers_are_never_remembered(monkeypatch):
    user = {"u": None}
    calls = _counting_loader(monkeypatch, user)
    for _ in range(2):
        assert asyncio.run(app_module._owner_id_from_request(_Req("/api/status/j1"))) is None
    user["u"] = _User("owner")
    for _ in range(2):
        asyncio.run(app_module._owner_id_from_request(_Req("/api/x", method="POST")))
        asyncio.run(app_module._owner_id_from_request(_Req("/api/status/j1", auth=None)))
    assert calls["n"] == 6


def test_a_remembered_caller_expires(monkeypatch):
    user = {"u": _User("owner")}
    calls = _counting_loader(monkeypatch, user)

    def _expire():
        for k, v in list(app_module._owner_ids.items()):
            app_module._owner_ids[k] = (0.0, v[1])
    asyncio.run(app_module._owner_id_from_request(_Req("/api/status/j1")))
    user["u"] = None                     # banned or revoked meanwhile
    _expire()
    assert asyncio.run(app_module._owner_id_from_request(_Req("/api/status/j1"))) is None
    assert calls["n"] == 2


def test_postgres_pool_has_room_and_a_short_wait(monkeypatch):
    for name in ("DB_POOL_SIZE", "DB_MAX_OVERFLOW", "DB_POOL_TIMEOUT"):
        monkeypatch.delenv(name, raising=False)
    opts = database._pool_options("postgresql+asyncpg://u:p@db/x")
    assert opts == {"pool_size": 20, "max_overflow": 20, "pool_timeout": 10}
    monkeypatch.setenv("DB_POOL_SIZE", "8")
    assert database._pool_options("postgresql+asyncpg://u:p@db/x")["pool_size"] == 8
    assert database._pool_options("sqlite+aiosqlite:///:memory:") == {}


def test_sending_the_source_keeps_the_lease_alive(tmp_path, monkeypatch):
    src = tmp_path / "V.mp4"
    src.write_bytes(b"x" * (3 * 1024 + 5))
    monkeypatch.setattr(app_module, "SOURCE_CHUNK", 1024)
    monkeypatch.setattr(app_module, "SOURCE_SEEN_EVERY", 0)
    old = time.time() - 600
    rw.touch_seen(str(tmp_path))
    os.utime(os.path.join(tmp_path, rw.SEEN_FILE), (old, old))
    touches = []
    real_touch = rw.touch_seen
    monkeypatch.setattr(app_module.remote_worker, "touch_seen",
                        lambda d: (touches.append(d), real_touch(d)))
    body = b"".join(app_module._source_chunks(str(src), str(tmp_path)))
    assert body == src.read_bytes()
    assert len(touches) >= 3
    assert time.time() - rw.last_seen(str(tmp_path)) < 60


def test_quiet_minutes_wrap_around_the_hour():
    assert app_module._quiet_minutes("55-5") == set(range(55, 60)) | set(range(0, 6))
    assert app_module._quiet_minutes("28-32, 7") == {28, 29, 30, 31, 32, 7}
    assert app_module._quiet_minutes("") == set()
    assert app_module._quiet_minutes("x-y") == set()


def test_no_claim_in_a_quiet_minute(monkeypatch):
    monkeypatch.setattr(app_module, "REMOTE_WORKER_TOKEN", "t" * 40)
    monkeypatch.setattr(app_module, "REMOTE_WORKER_SLOTS", 4)
    monkeypatch.setattr(app_module, "_admitting", {"waiting"})
    monkeypatch.setattr(app_module, "REMOTE_QUIET_MINUTES", set(range(60)))
    assert app_module._claim_remote_job("w1") is None


def test_the_source_transfer_can_be_capped(tmp_path, monkeypatch):
    src = tmp_path / "V.mp4"
    src.write_bytes(b"x" * 250_000)
    monkeypatch.setattr(app_module, "SOURCE_CHUNK", 50_000)
    monkeypatch.setattr(app_module, "REMOTE_SOURCE_MBPS", 10)   # 250 kB = 0.2 s
    t = time.time()
    assert len(b"".join(app_module._source_chunks(str(src), str(tmp_path)))) == 250_000
    assert time.time() - t >= 0.15
