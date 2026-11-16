"""What keeps the API answering when the host is short of CPU (16-nov):
the owner check of a status poll skips the full user load, the DB pool has
room and a short wait, and a source sent to the remote worker keeps its lease
alive for the whole transfer."""
import asyncio
import os
import time

import app as app_module
import remote_worker as rw
from cloud import database, owner_cache


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
    owner_cache.clear()
    return calls


def _owner(req):
    return asyncio.run(app_module._owner_id_from_request(req))


def test_repeated_polls_load_the_user_once(monkeypatch):
    user = {"u": _User("owner")}
    calls = _counting_loader(monkeypatch, user)
    for path in ("/api/status/j1", "/api/status/j2", "/api/status/j1"):
        assert _owner(_Req(path)) == "owner"
    assert calls["n"] == 1
    # Another credential is looked up on its own.
    _owner(_Req("/api/status/j1", auth="Bearer t2"))
    assert calls["n"] == 2


def test_writes_anonymous_and_unknown_callers_are_never_remembered(monkeypatch):
    user = {"u": None}
    calls = _counting_loader(monkeypatch, user)
    for _ in range(2):
        assert _owner(_Req("/api/status/j1")) is None
    user["u"] = _User("owner")
    for _ in range(2):
        _owner(_Req("/api/x", method="POST"))
        _owner(_Req("/api/status/j1", auth=None))
    assert calls["n"] == 6


def test_a_remembered_caller_expires(monkeypatch):
    user = {"u": _User("owner")}
    calls = _counting_loader(monkeypatch, user)
    _owner(_Req("/api/status/j1"))
    user["u"] = None
    for k, v in list(owner_cache._entries.items()):
        owner_cache._entries[k] = (0.0, v[1])
    assert _owner(_Req("/api/status/j1")) is None
    assert calls["n"] == 2


def test_a_ban_or_revoke_is_felt_at_once(monkeypatch):
    user = {"u": _User("owner")}
    calls = _counting_loader(monkeypatch, user)
    _owner(_Req("/api/status/j1"))
    user["u"] = None                      # banned or key revoked
    owner_cache.forget_user("owner")
    assert _owner(_Req("/api/status/j1")) is None
    assert calls["n"] == 2


def test_a_full_gate_answers_from_memory_or_asks_to_retry(monkeypatch):
    import pytest
    from fastapi import HTTPException
    user = {"u": _User("owner")}
    _counting_loader(monkeypatch, user)
    monkeypatch.setattr(owner_cache, "OWNER_LOOKUP_WAIT", 0.05)
    _owner(_Req("/api/status/j1"))                     # remembered...
    for k, v in list(owner_cache._entries.items()):
        owner_cache._entries[k] = (time.monotonic() - 1, v[1])   # ...past its TTL

    async def _with_gate_full(req):
        gate = owner_cache._gate_now()
        for _ in range(owner_cache.OWNER_LOOKUP_CONCURRENCY):
            await gate.acquire()
        return await app_module._owner_id_from_request(req)
    # A caller seen recently still gets its answer, with no DB lookup.
    assert asyncio.run(_with_gate_full(_Req("/api/status/j1"))) == "owner"
    owner_cache._gate = None
    # An unknown one is told to retry, instead of queueing for the pool.
    with pytest.raises(HTTPException) as e:
        asyncio.run(_with_gate_full(_Req("/api/status/j1", auth="Bearer new")))
    assert e.value.status_code == 503


def test_postgres_pool_has_room_and_a_short_wait(monkeypatch):
    for name in ("DB_POOL_SIZE", "DB_MAX_OVERFLOW", "DB_POOL_TIMEOUT"):
        monkeypatch.delenv(name, raising=False)
    opts = database._pool_options("postgresql+asyncpg://u:p@db/x")
    assert opts == {"pool_size": 15, "max_overflow": 15, "pool_timeout": 10}
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


def test_a_child_probe_brings_back_what_it_learned(monkeypatch):
    from cloud import metering
    from yt_clients import NotASingleVideo

    def _probe(url, allow_paid=True):
        metering._probe_facts[url] = {"at": time.time(), "max_height": 1080, "duration": 60}
        metering._static_bot_verdicts.add(url)
        metering._paid_probe_events.append({"url": url})
        return 1.5
    monkeypatch.setattr(metering, "probe_url_minutes", _probe)
    out = metering._probe_in_child("https://v/1", True)
    # The child's module state is drained into the result...
    assert "https://v/1" not in metering._static_bot_verdicts
    assert metering.pop_paid_probe_events() == []
    metering._probe_facts.pop("https://v/1", None)
    # ...and lands in the parent's.
    assert metering._merge_child_probe("https://v/1", out) == 1.5
    assert metering.probe_facts("https://v/1")["max_height"] == 1080
    assert metering.pop_statics_bot_checked("https://v/1") is True
    assert metering.pop_paid_probe_events() == [{"url": "https://v/1"}]

    def _not_single(url, allow_paid=True):
        raise NotASingleVideo("This link is a playlist.")
    monkeypatch.setattr(metering, "probe_url_minutes", _not_single)
    import pytest
    with pytest.raises(NotASingleVideo):
        metering._merge_child_probe("u", metering._probe_in_child("u", True))


def test_the_probe_runs_in_a_pool_process(monkeypatch):
    from cloud import metering
    import pytest
    monkeypatch.setattr(metering, "PROBE_PROCESSES", 1)
    monkeypatch.setattr(metering, "_probe_pool", None)
    try:
        # Refused before any network: the scheme is not http(s).
        with pytest.raises(ValueError):
            asyncio.run(metering.run_probe("file:///etc/passwd"))
        assert metering._probe_pool is not None      # it ran there, not in a fallback
    finally:
        if metering._probe_pool is not None:
            metering._probe_pool.shutdown(wait=True)
        metering._probe_pool = None


def test_the_source_cap_is_shared_by_every_transfer(monkeypatch):
    monkeypatch.setattr(app_module, "REMOTE_SOURCE_MBPS", 8)      # 1 MB/s
    monkeypatch.setattr(app_module, "_source_bucket", {"free_at": 0.0})
    first = app_module._source_turn(500_000)
    second = app_module._source_turn(500_000)                    # another transfer
    assert 0.4 < first <= 0.5 and 0.9 < second <= 1.0
