"""End to end: a free job that lands after its owner paid is delivered clean.

The real ``run_job_wrapper`` tail runs: the plan lookup (``metering.live_plans``),
``videos.unmark_user_library`` re-pointing the project and the library row in
the database, ``app._unmark_local_job`` on the working copy, then the caller's
webhook and the clips-ready email. Only the database, R2 and the network are
doubles. The webhook must carry the clean clip (local URL and R2 link), no
``.marked`` may be left, and the email goes out after the unmark.
"""
import asyncio
import json
import uuid
from types import SimpleNamespace

import pytest

import watermarked

app = pytest.importorskip("app")
cloud = pytest.importorskip("cloud")
from cloud import config as cloud_config, database, emails, metering, storage, videos  # noqa: E402
from sqlalchemy.dialects import postgresql  # noqa: E402

OWNER = "a11ce000-0000-0000-0000-000000000000"


class _Result:
    def __init__(self, rows):
        self.rows = rows

    def all(self):
        return self.rows

    def scalars(self):
        return _Scalars(self.rows)

    def scalar_one_or_none(self):
        return self.rows[0] if self.rows else None


class _Scalars(list):
    def first(self):
        return self[0] if self else None


class _Db:
    """Answers the handful of statements this path runs, by table."""

    def __init__(self, plan):
        self.plan = plan
        self.project = SimpleNamespace(id=1, job_id="j1", user_id=uuid.UUID(OWNER), state={
            "v": 1, "clips": [{"index": 0, "server_file": "wm_t_clip_1.mp4",
                               "original_file": "t_clip_1.mp4"}]})
        self.video = SimpleNamespace(job_id="j1", clip_index=0, size_bytes=6,
                                     r2_key=storage_key("wm_t_clip_1.mp4"))
        self.statements = []

    def session(self):
        return _Session(self)


def storage_key(name):
    return f"users/{OWNER}/jobs/j1/{name}"


class _Session:
    def __init__(self, db):
        self.db = db

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    def begin(self):
        return self

    async def get(self, model, key):
        return SimpleNamespace(id=key, email="owner@example.com")

    async def execute(self, stmt):
        sql = str(stmt.compile(dialect=postgresql.dialect()))
        self.db.statements.append(sql)
        if "FROM subscriptions" in sql:
            return _Result([(uuid.UUID(OWNER), self.db.plan)] if self.db.plan else [])
        if "FROM projects" in sql:
            return _Result([self.db.project])
        if "FROM user_videos" in sql:
            return _Result([self.db.video])
        raise AssertionError(f"unexpected statement: {sql}")


@pytest.fixture
def world(tmp_path, monkeypatch):
    d = tmp_path / "j1"
    d.mkdir()
    (d / "t_clip_1.mp4").write_bytes(b"clean!")
    (d / "wm_t_clip_1.mp4").write_bytes(b"marked")
    watermarked.mark_job(str(d))
    (d / "t_metadata.json").write_text(json.dumps({
        "shorts": [{"video_url": "/videos/j1/wm_t_clip_1.mp4"}]}))
    job = {"status": "queued", "user_id": OWNER, "output_dir": str(d), "logs": [],
           "webhook_url": "https://hooks.example/cb", "webhook_secret": None,
           "base_url": "https://api.example",
           "result": {"clips": [{"title": "t", "video_url": "/videos/j1/wm_t_clip_1.mp4"}]},
           "ready_files": {0: "wm_t_clip_1.mp4"}}

    monkeypatch.setattr(app, "OUTPUT_DIR", str(tmp_path))
    monkeypatch.setattr(app, "BILLING_ENABLED", True)
    monkeypatch.setattr(app, "cloud", cloud)
    monkeypatch.setattr(app, "_metering", metering)
    monkeypatch.setattr(app, "_cloud_config", cloud_config)
    monkeypatch.setattr(app, "jobs", {"j1": job})
    monkeypatch.setattr(app, "_running_jobs", {"j1"})
    monkeypatch.setattr(app, "concurrency_semaphore", asyncio.Semaphore(0))
    monkeypatch.setattr(app, "job_queue", asyncio.PriorityQueue())
    app.job_queue.put_nowait((2, 0, "j1"))
    app.job_queue.get_nowait()                     # as the worker did

    db = _Db(plan="starter")
    monkeypatch.setattr(database, "session", db.session)
    monkeypatch.setattr(videos, "settings", SimpleNamespace(r2_configured=True))
    monkeypatch.setattr(videos, "_local_unmark", app._unmark_local_job)
    deleted = []
    monkeypatch.setattr(storage, "object_size", lambda key: 6)
    monkeypatch.setattr(storage, "delete_key", lambda key: deleted.append(key))
    monkeypatch.setattr(storage, "job_key", lambda uid, jid, name: storage_key(name))
    monkeypatch.setattr(storage, "presigned_get",
                        lambda key, expires=3600, download_name=None: f"https://r2.example/{key}")

    events = []

    async def run_job(job_id, data):                # the render itself, done
        data["status"] = "completed"

    async def deliver(url, body, secret):
        events.append(("webhook", json.loads(body)))

    async def email(to, title, n, link):
        events.append(("email", {"to": to, "link": link, "marked": watermarked.job_is_marked(str(d))}))

    async def noop(*a, **k):
        return None

    monkeypatch.setattr(app, "run_job", run_job)
    monkeypatch.setattr(app, "_deliver_webhook", deliver)
    monkeypatch.setattr(emails, "send_clips_ready_email", email)
    for name in ("_settle_reservation", "_archive_managed_job", "_autopilot_job_finished",
                 "_record_job_alert", "_track_proxy_usage", "_notify_clip_activity"):
        monkeypatch.setattr(app, name, noop)
    return SimpleNamespace(dir=d, job=job, db=db, events=events, deleted=deleted)


def _finish(world):
    async def run():
        await app.run_job_wrapper("j1")
        for _ in range(5):                         # let the detached webhook task run
            await asyncio.sleep(0)
    asyncio.run(run())


def test_paid_owner_gets_clean_clips_in_webhook_and_no_marker(world):
    _finish(world)
    d = world.dir
    assert not watermarked.job_is_marked(str(d))
    assert (d / "t_clip_1.mp4").read_bytes() == b"clean!"          # canonical untouched
    assert not (d / "wm_t_clip_1.mp4").exists()
    assert world.deleted == [storage_key("wm_t_clip_1.mp4")]        # marked R2 object gone
    assert world.db.video.r2_key == storage_key("t_clip_1.mp4")
    assert world.db.project.state["clips"][0]["server_file"] == "t_clip_1.mp4"

    kinds = [k for k, _ in world.events]
    assert sorted(kinds) == ["email", "webhook"]
    hook = next(p for k, p in world.events if k == "webhook")
    clip = hook["clips"][0]
    assert hook["event"] == "job.completed"
    assert clip["video_url"].endswith("/videos/j1/t_clip_1.mp4")
    assert clip["download_url"] == f"https://r2.example/{storage_key('t_clip_1.mp4')}"
    assert "wm_" not in json.dumps(hook)

    mail = next(p for k, p in world.events if k == "email")
    assert mail["marked"] is False, "the email went out after the unmark"
    assert "wm_" not in mail["link"]
    # One plan lookup for the owner, then the project of THIS job only.
    project_selects = [s for s in world.db.statements if "FROM projects" in s]
    assert "projects.job_id" in project_selects[0].split("WHERE", 1)[1]


def test_free_owner_keeps_the_marked_delivery(world):
    world.db.plan = None
    _finish(world)
    assert watermarked.job_is_marked(str(world.dir))
    hook = next(p for k, p in world.events if k == "webhook")
    assert hook["clips"][0]["video_url"].endswith("/videos/j1/wm_t_clip_1.mp4")
    assert world.deleted == []
