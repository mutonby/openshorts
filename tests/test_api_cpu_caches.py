"""What keeps the API's one core free under heavy traffic: the status poll
folds only the log lines it has not seen, read-only requests reuse the
signed-in user for a few seconds (and a Stripe event or a ban forgets it), and
the static config tells the browser it may reuse it."""
import asyncio
import contextlib
import uuid

from fastapi.testclient import TestClient

import app as app_module
from cloud import auth, owner_cache
from log_view import FriendlyLogFold, friendly_logs_timed

RAW = [
    "Job started by worker.",
    "[download]  10.0% of 100MiB",
    "[download]  55.5% of 100MiB",
    "🎙️ Transcribing… 25% (3s)",
    "Video downloaded in 12s",
    "🎙️ Transcribing… 100% (7s)",
    "noise line that is hidden",
    "🔥 Found 2 viral clips!",
    "🎬 Processing Clip 1: 0.004s - 35.336s",
    "✅ Clip 1 ready: output/x/x_clip_1.mp4",
    "🎬 Processing Clip 2: 40s - 70s",
    "✅ Clip 2 ready: output/x/x_clip_2.mp4",
    "Process finished successfully.",
]


def test_fold_fed_in_pieces_matches_the_full_fold():
    times = [float(i) for i in range(len(RAW))]
    whole = friendly_logs_timed(RAW, times)
    for step in (1, 2, 3, 5):
        fold = FriendlyLogFold()
        for end in range(step, len(RAW) + step, step):
            fold.feed(RAW, times, min(end, len(RAW)))
        assert fold.view() == whole


def test_fold_starts_over_when_the_log_was_replaced():
    fold = FriendlyLogFold().feed(RAW)
    fold.feed(["Job started by worker."])
    assert fold.view() == friendly_logs_timed(["Job started by worker."])


def test_status_view_of_a_growing_log_is_incremental(monkeypatch):
    monkeypatch.setattr(app_module, "BILLING_ENABLED", True)
    monkeypatch.setattr(app_module, "DEBUG_LOGS", False)
    log = app_module._TimedLog()
    for line in RAW:
        log.append(line)
        lines, times = app_module._visible_logs_timed(log)
        expected = friendly_logs_timed(list(log), list(log.times))
        assert lines == [p[0] for p in expected]
        assert times == [p[1] for p in expected]
    assert log._friendly.consumed == len(RAW)


def test_a_line_without_its_time_yet_waits_for_the_next_poll(monkeypatch):
    monkeypatch.setattr(app_module, "BILLING_ENABLED", True)
    monkeypatch.setattr(app_module, "DEBUG_LOGS", False)
    log = app_module._TimedLog(["Job started by worker."])
    list.append(log, "🔥 Found 3 viral clips!")      # time not stamped yet
    lines, times = app_module._visible_logs_timed(log)
    assert lines == ["Job started by worker."] and len(times) == 1
    log.times.append(1.0)
    lines, _ = app_module._visible_logs_timed(log)
    assert lines == ["Job started by worker.", "🔥 Found 3 clips!"]


class _Req:
    def __init__(self, path, method="GET"):
        self.method = method
        self.headers = {"Authorization": "Bearer jwt"}

        class _U:
            pass
        self.url = _U()
        self.url.path = path


def _counting_user_load(monkeypatch):
    uid = uuid.uuid4()
    calls = {"n": 0}

    async def _load(session, user_id):
        calls["n"] += 1
        return auth.CurrentUser(id=user_id, email="a@example.com", entitled=True, plan="free")

    @contextlib.asynccontextmanager
    async def _session():
        yield None

    monkeypatch.setattr(auth, "_decode_jwt", lambda token: {"sub": str(uid)})
    monkeypatch.setattr(auth, "_load_current_user", _load)
    monkeypatch.setattr(auth.database, "session", _session)
    auth.forget_current_user()
    return uid, calls


def _get(path, method="GET"):
    return asyncio.run(auth.get_current_user_optional(_Req(path, method)))


def test_read_only_requests_reuse_the_signed_in_user(monkeypatch):
    uid, calls = _counting_user_load(monkeypatch)
    assert _get("/api/me").id == uid
    _get("/api/social/user")
    _get("/api/media-token", "POST")
    assert calls["n"] == 1


def test_a_submission_always_loads_the_user_fresh(monkeypatch):
    _uid, calls = _counting_user_load(monkeypatch)
    _get("/api/me")
    _get("/api/process", "POST")
    _get("/api/billing/checkout", "POST")
    assert calls["n"] == 3


def test_a_ban_or_a_stripe_event_forgets_the_user(monkeypatch):
    uid, calls = _counting_user_load(monkeypatch)
    _get("/api/me")
    owner_cache.forget_user(uid)          # ban / revoke / erase
    _get("/api/me")
    auth.forget_current_user()            # any Stripe event
    _get("/api/me")
    assert calls["n"] == 3


def test_ttl_zero_turns_the_memory_off(monkeypatch):
    _uid, calls = _counting_user_load(monkeypatch)
    monkeypatch.setattr(auth, "CURRENT_USER_TTL", 0)
    _get("/api/me")
    _get("/api/me")
    assert calls["n"] == 2


def test_public_config_may_be_reused_by_the_browser():
    res = TestClient(app_module.app).get("/api/config")
    assert res.status_code == 200
    assert "max-age=" in res.headers.get("cache-control", "")


def test_partial_result_is_parsed_again_only_when_something_changed(tmp_path, monkeypatch):
    import json as _json
    job_id = "j-partial"
    (tmp_path / "x_metadata.json").write_text(_json.dumps({"shorts": [{"n": 0}, {"n": 1}]}))
    (tmp_path / "clip0.mp4").write_bytes(b"x")
    monkeypatch.setitem(app_module.jobs, job_id, {"ready_files": {0: "clip0.mp4"}})
    app_module._partial_seen.pop(job_id, None)
    loads = {"n": 0}
    real_load = app_module.json.load

    def _counting_load(f):
        loads["n"] += 1
        return real_load(f)
    monkeypatch.setattr(app_module.json, "load", _counting_load)

    for _ in range(3):
        app_module._refresh_partial_result(job_id, str(tmp_path))
    assert loads["n"] == 1
    assert len(app_module.jobs[job_id]["result"]["clips"]) == 1

    (tmp_path / "clip1.mp4").write_bytes(b"x")
    app_module.jobs[job_id]["ready_files"][1] = "clip1.mp4"   # a new clip announced
    app_module._refresh_partial_result(job_id, str(tmp_path))
    assert loads["n"] == 2
    assert len(app_module.jobs[job_id]["result"]["clips"]) == 2


def test_quiet_access_log_keeps_errors_and_other_paths():
    import logging
    f = app_module._QuietAccessLog()

    def rec(path, status):
        return logging.LogRecord("uvicorn.access", logging.INFO, "", 0,
                                 '%s - "%s %s HTTP/%s" %d', ("1.2.3.4:1", "GET", path, "1.1", status), None)
    assert not f.filter(rec("/api/status/abc", 200))
    assert not f.filter(rec("/api/config", 200))
    assert f.filter(rec("/api/status/abc", 500))
    assert f.filter(rec("/api/me", 200))


def test_a_job_starts_through_nice_without_a_preexec_fn(monkeypatch):
    seen = {}

    class _P:
        stdout = None
        returncode = 0

        def poll(self):
            return 0

    def _popen(cmd, **kw):
        seen["cmd"], seen["preexec"] = cmd, kw.get("preexec_fn")
        return _P()
    monkeypatch.setattr(app_module.subprocess, "Popen", _popen)
    import types
    fake_thread = type("T", (), {"daemon": True, "start": lambda s: None, "join": lambda s, t=None: None})
    monkeypatch.setattr(app_module, "threading", types.SimpleNamespace(
        Thread=lambda **kw: fake_thread(), Lock=app_module.threading.Lock))
    monkeypatch.setenv("JOB_NICE", "10")
    monkeypatch.setattr(app_module.shutil, "which", lambda name: "/usr/bin/nice")
    asyncio.run(app_module._run_main("j", ["python", "main.py"], {}))
    assert seen["cmd"][:3] == ["nice", "-n", "10"] and seen["preexec"] is None
