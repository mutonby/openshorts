"""The split API (serve.py, front.py, docs/architecture/multiprocess-api.md).

Python runs one thread at a time per process, so the API's one process capped
the whole service at one core: on 18-nov-2026 it sat at 100 % (login and job
submissions answering 500, the GPU idle behind it) while the host had most of
its cores free. Now N front processes serve what needs none of the engine's
memory and pass the rest to the one engine. What must hold:

* a route the fronts serve touches none of the engine's in-memory state;
* a status poll answered from the engine's snapshot says what the engine
  itself would say, and a stale or foreign snapshot is never used;
* a forget (ban, revoke, Stripe event) in one process reaches the others;
* the proxy passes requests and responses through unchanged, streamed.
"""
import ast
import asyncio
import json
import os
import time
import uuid

import httpx
import pytest

import front_proxy
from cloud import owner_cache

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAMPLE_ID = "0d3c2b1a-1111-4222-8333-944455556666"


# --- 1. The fronts serve only routes that need no engine memory --------------

def _app_routes_and_state():
    """(method, path template, state names reached) for every app.py route.
    State = module-level mutable containers of app.py, followed through the
    module functions each handler calls."""
    tree = ast.parse(open(os.path.join(ROOT, "app.py")).read())
    mutable = set()
    for node in tree.body:
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            target = node.targets[0] if isinstance(node, ast.Assign) else node.target
            value = node.value
            if isinstance(target, ast.Name) and value is not None:
                src = ast.unparse(value)
                if src.startswith(("{}", "set()", "[]", "dict(", "collections.", "deque(",
                                   "asyncio.Queue", "asyncio.PriorityQueue", "asyncio.Lock",
                                   "asyncio.Semaphore", "threading.Lock", "OrderedDict(")):
                    mutable.add(target.id)
    funcs = {n.name: n for n in ast.walk(tree)
             if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
    memo = {}

    def reached(name, seen):
        if name in memo:
            return memo[name]
        if name in seen or name not in funcs:
            return set()
        seen = seen | {name}
        names = {x.id for x in ast.walk(funcs[name]) if isinstance(x, ast.Name)}
        out = names & mutable
        for callee in names:
            if callee in funcs and callee != name:
                out |= reached(callee, seen)
        memo[name] = out
        return out

    routes = []
    for node in tree.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for deco in node.decorator_list:
            if (isinstance(deco, ast.Call) and isinstance(deco.func, ast.Attribute)
                    and isinstance(deco.func.value, ast.Name) and deco.func.value.id == "app"
                    and deco.args and isinstance(deco.args[0], ast.Constant)):
                routes.append((deco.func.attr.upper(), deco.args[0].value,
                               reached(node.name, set())))
    return routes


# Per-process memories a front may keep on its own: rate-limit and cache
# copies would differ per process, so none is allowed today.
ALLOWED_IN_FRONT = set()


def test_app_routes_the_fronts_serve_touch_no_engine_state():
    served = []
    for method, template, state in _app_routes_and_state():
        path = template.replace("{job_id}", SAMPLE_ID)
        if "{" in path or not front_proxy.serves_locally(method, path):
            continue
        served.append(path)
        assert state <= ALLOWED_IN_FRONT, (
            f"{method} {template} is served by the front processes but reaches "
            f"engine memory {sorted(state - ALLOWED_IN_FRONT)}: leave it to the engine")
    # The list is not empty by accident (a renamed route would drop out).
    assert "/api/config" in served and "/api/media-token" in served
    assert f"/internal/worker/jobs/{SAMPLE_ID}/log" in served


@pytest.mark.parametrize("method,path,local", [
    ("GET", "/api/config", True),
    ("GET", "/api/me", True),
    ("POST", "/api/auth/magic-link", True),
    ("GET", "/api/auth/google/callback", True),
    ("POST", f"/internal/worker/jobs/{SAMPLE_ID}/log", True),
    ("PUT", f"/internal/worker/jobs/{SAMPLE_ID}/file", True),
    # The engine's: the queue, the claims, a plan change moving jobs, the
    # bandwidth budget of the sources, everything not listed.
    ("POST", "/api/process", False),
    ("GET", f"/api/status/{SAMPLE_ID}", False),     # only through a snapshot
    ("POST", "/internal/worker/claim", False),
    ("GET", f"/internal/worker/jobs/{SAMPLE_ID}/source", False),
    ("POST", "/api/stripe/webhook", False),
    ("POST", "/api/billing/end-trial", False),
    ("GET", "/api/config/extra", False),
    ("POST", "/api/config", False),
    ("GET", f"/internal/worker/jobs/{SAMPLE_ID}/log/../claim", False),
    ("GET", "/videos/x/y.mp4", False),
])
def test_what_a_front_serves_itself(method, path, local):
    assert front_proxy.serves_locally(method, path) is local


# --- 2. Status snapshots -----------------------------------------------------

@pytest.fixture
def engine(tmp_path, monkeypatch):
    app_module = pytest.importorskip("app")
    out = tmp_path / "output"
    out.mkdir()
    monkeypatch.setattr(app_module, "OUTPUT_DIR", str(out))
    monkeypatch.setattr(app_module, "jobs", {})
    monkeypatch.setattr(app_module, "_published", {})
    monkeypatch.setattr(app_module, "_draining", False)
    monkeypatch.setattr(app_module, "INSTANCE_ID", "inst-a")
    monkeypatch.setattr(app_module, "_queue_view_cache", {"stamp": None, "data": None})
    monkeypatch.setattr(app_module, "_queue_view_written", {})
    monkeypatch.setattr(app_module, "queue_snapshot",
                        lambda jid: {"position": 3, "ahead": 2, "eta_seconds": 600,
                                     "eta_is_minimum": False}
                        if app_module.jobs.get(jid, {}).get("status") == "queued" else None)
    return app_module, out


def _job(app_module, out, status, logs=(), clips=None, owner="u-1"):
    jid = str(uuid.uuid4())
    (out / jid).mkdir()
    app_module.jobs[jid] = {
        "status": status, "logs": app_module._TimedLog(list(logs)), "user_id": owner,
        "result": {"clips": clips} if clips is not None else None, "partial": None}
    return jid


class _Req:
    method = "GET"
    headers = {}
    scope = {}


def _engine_answer(app_module, jid):
    return asyncio.run(app_module.get_status(jid, _Req()))


def _front_answer(app_module, jid):
    view = app_module.front_status_view(jid)
    assert view is not None
    return asyncio.run(app_module._status_from_snapshot(_Req(), *view))


def test_a_poll_from_the_snapshot_says_what_the_engine_says(engine):
    app_module, out = engine
    queued = _job(app_module, out, "queued")
    running = _job(app_module, out, "processing", logs=["Job started by worker.", "🔥 Found 2 viral clips!"],
                   clips=[{"title": "a", "video_url": "/videos/x/a.mp4"}])
    asyncio.run(app_module._publish_statuses())
    for jid in (queued, running):
        assert _front_answer(app_module, jid) == _engine_answer(app_module, jid)
    assert _front_answer(app_module, queued)["queue"]["position"] == 3


def test_the_snapshot_follows_the_job(engine):
    app_module, out = engine
    jid = _job(app_module, out, "processing", logs=["Job started by worker."])
    asyncio.run(app_module._publish_statuses())
    app_module.jobs[jid]["logs"].append("🔥 Found 2 viral clips!")
    asyncio.run(app_module._publish_statuses())
    assert _front_answer(app_module, jid) == _engine_answer(app_module, jid)
    # Finished: the snapshot goes and the poll is the engine's again.
    app_module.jobs[jid]["status"] = "completed"
    asyncio.run(app_module._publish_statuses())
    assert not (out / jid / app_module.STATUS_SNAPSHOT_FILE).exists()
    assert app_module.front_status_view(jid) is None


def test_a_stale_or_foreign_snapshot_is_never_used(engine):
    app_module, out = engine
    jid = _job(app_module, out, "queued")
    asyncio.run(app_module._publish_statuses())
    assert app_module.front_status_view(jid) is not None
    view_path = out / app_module.QUEUE_VIEW_FILE
    beat = out / app_module.QUEUE_BEAT_FILE
    # The publisher stopped (busy, killed, draining): the poll goes to an engine.
    old = time.time() - app_module.STATUS_SNAPSHOT_FRESH_SECONDS - 1
    os.utime(beat, (old, old))
    assert app_module.front_status_view(jid) is None
    # Another instance publishes now: our old snapshot is not its word.
    os.utime(beat, None)
    data = json.loads(view_path.read_text())
    data.update(instance="inst-b")
    view_path.write_text(json.dumps(data) + " ")
    assert app_module.front_status_view(jid) is None
    # A path in the id never reaches the disk.
    assert app_module.front_status_view("../" + jid) is None


def test_a_quiet_queue_is_not_rewritten_every_second(engine):
    app_module, out = engine
    monkeypatch_written = app_module._queue_view_written
    monkeypatch_written.clear()
    jid = _job(app_module, out, "queued")
    asyncio.run(app_module._publish_statuses())
    view_path = out / app_module.QUEUE_VIEW_FILE
    first = os.stat(view_path).st_ino
    asyncio.run(app_module._publish_statuses())
    assert os.stat(view_path).st_ino == first          # same file, only the beat moved
    assert app_module.front_status_view(jid) is not None
    monkeypatch_written.clear()


def test_a_draining_engine_publishes_nothing_and_keeps_the_next_ones_snapshots(engine):
    app_module, out = engine
    jid = _job(app_module, out, "queued")
    asyncio.run(app_module._publish_statuses())
    snap = out / jid / app_module.STATUS_SNAPSHOT_FILE
    # The next instance resumed the job and wrote its own snapshot.
    snap.write_text(json.dumps({"instance": "inst-b", "status": "queued"}))
    app_module._draining = True
    asyncio.run(app_module._publish_statuses())
    assert json.loads(snap.read_text())["instance"] == "inst-b"


def test_the_owner_check_holds_on_the_snapshot(engine, monkeypatch):
    app_module, out = engine
    jid = _job(app_module, out, "queued", owner="owner-1")
    asyncio.run(app_module._publish_statuses())
    monkeypatch.setattr(app_module, "BILLING_ENABLED", True)

    async def stranger(_request):
        return "someone-else"
    monkeypatch.setattr(app_module, "_owner_id_from_request", stranger)
    with pytest.raises(app_module.HTTPException) as e:
        _front_answer(app_module, jid)
    assert e.value.status_code == 404


# --- 3. A forget in one process reaches the others ---------------------------

def test_a_forget_elsewhere_clears_this_process(tmp_path, monkeypatch):
    epoch = tmp_path / ".cache_epoch"
    monkeypatch.setenv("CACHE_EPOCH_FILE", str(epoch))
    monkeypatch.setattr(owner_cache, "EPOCH_CHECK_SECONDS", 0.0)
    owner_cache.clear()
    cleared = []
    owner_cache.on_forget(cleared.append)
    try:
        owner_cache.sync()                       # first look: nothing to clear
        owner_cache._remember("k", "user-1", time.monotonic())
        owner_cache.bump()                       # our own forget: no second clear
        owner_cache.sync()
        assert "k" in owner_cache._entries and cleared == []
        # Another process forgets someone.
        st = os.stat(epoch)
        os.utime(epoch, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000))
        owner_cache.sync()
        assert owner_cache._entries == {} and cleared == [None]
    finally:
        owner_cache._forget_hooks.remove(cleared.append)
        owner_cache.clear()


def test_a_forget_writes_the_epoch(tmp_path, monkeypatch):
    epoch = tmp_path / ".cache_epoch"
    monkeypatch.setenv("CACHE_EPOCH_FILE", str(epoch))
    owner_cache.clear()
    owner_cache.forget_user("user-1")
    assert epoch.exists()
    owner_cache.clear()


def test_one_process_has_no_epoch(monkeypatch):
    monkeypatch.delenv("CACHE_EPOCH_FILE", raising=False)
    owner_cache.clear()
    owner_cache._remember("k", "user-1", time.monotonic())
    owner_cache.sync()
    owner_cache.bump()
    assert "k" in owner_cache._entries
    owner_cache.clear()


# --- 4. The proxy ------------------------------------------------------------

def _recording_app(name, seen):
    async def asgi(scope, receive, send):
        if scope["type"] != "http":
            return
        body = b""
        while True:
            message = await receive()
            body += message.get("body", b"")
            if not message.get("more_body"):
                break
        seen.append({"app": name, "method": scope["method"], "path": scope["path"],
                     "query": scope["query_string"], "headers": dict(scope["headers"]),
                     "body": body, "view": scope.get("view-key")})
        payload = json.dumps({"app": name, "len": len(body)}).encode()
        await send({"type": "http.response.start", "status": 207,
                    "headers": [(b"content-type", b"application/json"),
                                (b"x-engine", b"1"), (b"set-cookie", b"a=1"),
                                (b"set-cookie", b"b=2"), (b"date", b"engine-date")]})
        await send({"type": "http.response.body", "body": payload})
    return asgi


def _front(seen, status_view=None, transport=None):
    engine = _recording_app("engine", seen)
    return front_proxy.Front(
        _recording_app("local", seen), status_view=status_view, status_key="view-key",
        transport=transport or httpx.ASGITransport(app=engine))


async def _call(front, method, path, **kw):
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=front),
                                 base_url="http://testserver") as client:
        return await client.request(method, path, **kw)


def test_the_engine_gets_the_request_as_it_came():
    seen = []
    body = os.urandom(300_000)
    r = asyncio.run(_call(_front(seen), "POST", "/api/process?x=1&y=%2F",
                          content=body, headers={"Authorization": "Bearer t",
                                                 "X-Forwarded-For": "203.0.113.9"}))
    assert r.status_code == 207 and r.json() == {"app": "engine", "len": len(body)}
    got = seen[-1]
    assert got["path"] == "/api/process" and got["query"] == b"x=1&y=%2F"
    assert got["body"] == body
    assert got["headers"][b"authorization"] == b"Bearer t"
    assert got["headers"][b"x-forwarded-for"] == b"203.0.113.9"
    # Repeated headers survive, the engine's own Date does not go out twice.
    assert r.headers.get_list("set-cookie") == ["a=1", "b=2"]
    assert "engine-date" not in r.headers.get_list("date")


def test_a_direct_hit_still_names_the_client():
    seen = []
    asyncio.run(_call(_front(seen), "GET", "/api/history"))
    assert seen[-1]["app"] == "engine"
    assert b"x-forwarded-for" in seen[-1]["headers"]
    assert seen[-1]["headers"][b"x-forwarded-proto"] == b"http"


def test_local_routes_and_preflights_stay_here():
    seen = []
    front = _front(seen)
    asyncio.run(_call(front, "GET", "/api/config"))
    asyncio.run(_call(front, "OPTIONS", "/api/process",
                      headers={"Origin": "https://x", "Access-Control-Request-Method": "POST"}))
    assert [s["app"] for s in seen] == ["local", "local"]
    # A plain OPTIONS (no preflight) is the engine's like any other request.
    asyncio.run(_call(front, "OPTIONS", "/api/process"))
    assert seen[-1]["app"] == "engine"


def test_a_status_poll_uses_the_snapshot_only_when_there_is_one():
    seen = []
    views = {SAMPLE_ID: ({"status": "queued"}, None)}
    front = _front(seen, status_view=views.get)
    asyncio.run(_call(front, "GET", f"/api/status/{SAMPLE_ID}"))
    assert seen[-1]["app"] == "local" and seen[-1]["view"] == views[SAMPLE_ID]
    other = "0d3c2b1a-1111-4222-8333-000000000000"
    asyncio.run(_call(front, "GET", f"/api/status/{other}"))
    assert seen[-1]["app"] == "engine"


def test_an_engine_that_is_not_up_is_a_503_to_retry():
    def refuse(request):
        raise httpx.ConnectError("no socket", request=request)
    seen = []
    r = asyncio.run(_call(_front(seen, transport=httpx.MockTransport(refuse)), "GET", "/api/history"))
    assert r.status_code == 503 and r.headers["retry-after"] == "3"


# --- 5. Pool sizes -------------------------------------------------------------

def test_a_front_keeps_a_small_pool(monkeypatch):
    database = pytest.importorskip("cloud.database")
    url = "postgresql+asyncpg://u:p@h/db"
    monkeypatch.setenv("API_ROLE", "front")
    assert database._pool_options(url)["pool_size"] == 3
    assert database._pool_options(url)["max_overflow"] == 7
    monkeypatch.setenv("API_ROLE", "engine")
    assert database._pool_options(url)["pool_size"] == 15
