"""A paid job's source video in R2 (cloud/source_archive.py).

The source used to live on the processing server an hour at most; once it was
gone the preview stopped playing and the editor answered 409 to any cut outside
the original range or any reframe. These tests pin the rules of the archive
against an in-memory R2:

* only an owner on a live paid plan gets one, and nothing about the upload can
  fail a job;
* the object sits under the user's prefix (erasure and purges take it) but the
  project restore never pulls it;
* the editor brings it back on demand, once per job, and the preview is a
  redirect to R2 that only the owner (or a signed link) gets;
* the daily sweep deletes what is older than SOURCE_ARCHIVE_DAYS and leaves the
  rest, clips included, alone.
"""
import asyncio
import io
import json
import os
import time
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import httpx
import pytest

sa = pytest.importorskip("cloud.source_archive")
from cloud import storage as real_storage  # noqa: E402

OWNER = "a11ce000-0000-0000-0000-000000000001"
OTHER = "b0b00000-0000-0000-0000-000000000002"
JOB = "c0ffee00-0000-4000-8000-000000000003"


class FakeR2:
    """Just enough of cloud/storage.py, in memory."""

    def __init__(self):
        self.objects = {}        # key -> {"data", "modified", "meta"}
        self.downloads = []
        self.fail_upload = False
        self.job_key = real_storage.job_key
        self.user_prefix = real_storage.user_prefix

    def put(self, key, data=b"src", modified=None, meta=None):
        self.objects[key] = {"data": data, "meta": dict(meta or {}),
                             "modified": modified or datetime.now(timezone.utc)}

    def upload_fileobj(self, fileobj, key, content_type="video/mp4", metadata=None):
        if self.fail_upload:
            raise RuntimeError("R2 said no")
        self.put(key, fileobj.read(), meta=metadata)

    def iter_objects(self, prefix):
        for key in sorted(self.objects):
            if key.startswith(prefix):
                o = self.objects[key]
                yield key, o["modified"], len(o["data"])

    def list_keys(self, prefix):
        return [k for k, _m, _s in self.iter_objects(prefix)]

    def object_metadata(self, key):
        o = self.objects.get(key)
        return None if o is None else dict(o["meta"])

    def download_file(self, key, local_path):
        self.downloads.append(key)
        with open(local_path, "wb") as f:
            f.write(self.objects[key]["data"])

    def delete_key(self, key):
        self.objects.pop(key, None)

    def delete_keys(self, keys):
        keys = list(keys)
        for k in keys:
            self.objects.pop(k, None)
        return len(keys)

    def delete_prefix(self, prefix):
        doomed = [k for k in self.objects if k.startswith(prefix)]
        return self.delete_keys(doomed)

    def presigned_get(self, key, expires=3600, download_name=None):
        return f"https://r2.test/{key}?expires={expires}"

    def source_keys(self):
        return [k for k in self.objects if sa.is_source_key(k)]


@pytest.fixture
def r2(monkeypatch):
    fake = FakeR2()
    monkeypatch.setattr(sa, "storage", fake)
    monkeypatch.setattr(sa, "settings", SimpleNamespace(r2_configured=True))
    monkeypatch.setattr(sa.config, "SOURCE_ARCHIVE_DAYS", 30)
    monkeypatch.setattr(sa, "_upload_sem", None)
    monkeypatch.setattr(sa, "_found", {})
    monkeypatch.setattr(sa, "_last_sweep", 0.0)
    state = {"plans": {OWNER: "starter"}, "users": {OWNER, OTHER}, "lookups": 0}

    async def paying(user_id):
        state["lookups"] += 1
        plan = state["plans"].get(str(user_id))
        return bool(plan) and plan != "free"

    async def exists(user_id):
        return str(user_id) in state["users"]

    monkeypatch.setattr(sa, "_owner_is_paying", paying)
    monkeypatch.setattr(sa, "_user_exists", exists)
    fake.state = state
    return fake


@pytest.fixture
def source_file(tmp_path):
    p = tmp_path / "uploads" / f"{JOB}_talk.MOV"
    p.parent.mkdir()
    p.write_bytes(b"the whole original")
    return p


def _archive(path, probe=None):
    return asyncio.run(sa.archive_source(OWNER, JOB, str(path), duration_probe=probe))


# --- Upload ---------------------------------------------------------------

class TestUpload:
    def test_paid_owner_gets_the_source_under_their_job_prefix(self, r2, source_file):
        key = _archive(source_file, probe=lambda p: 754.25)
        assert key is not None and r2.source_keys() == [key]
        # Under the user's prefix: the erasure (delete_prefix(user_prefix)) and
        # the free purge (every key under the job prefix) take it as is.
        assert key.startswith(real_storage.user_prefix(OWNER))
        assert key.startswith(real_storage.job_key(OWNER, JOB, ""))
        name = key.rsplit("/", 1)[-1]
        assert len(name) == len("0" * 32 + ".mov") and name.endswith(".mov")
        assert r2.objects[key]["data"] == b"the whole original"
        assert r2.objects[key]["meta"] == {"duration": "754.250"}

    def test_the_name_is_random_not_derived_from_the_job(self, r2, source_file):
        # On a public custom domain the key is the lock: a clip URL already
        # shows the user and job ids, so a fixed name would open the original.
        a = _archive(source_file)
        b = _archive(source_file)
        assert a != b
        assert r2.source_keys() == [b], "a second upload replaces the first copy"

    def test_free_owner_gets_nothing_and_is_never_probed(self, r2, source_file):
        r2.state["plans"] = {OWNER: "free"}
        probed = []
        assert _archive(source_file, probe=lambda p: probed.append(p) or 10.0) is None
        assert r2.objects == {} and probed == []

    def test_no_plan_at_all_gets_nothing(self, r2, source_file):
        r2.state["plans"] = {}
        assert _archive(source_file) is None
        assert r2.objects == {}

    def test_a_failed_upload_is_logged_and_nothing_else(self, r2, source_file, capsys):
        r2.fail_upload = True
        assert _archive(source_file) is None
        assert r2.objects == {}
        assert "Source archive upload failed" in capsys.readouterr().out

    def test_a_vanished_source_is_not_an_error(self, r2, tmp_path):
        assert _archive(tmp_path / "gone.mp4") is None
        assert r2.objects == {}

    def test_a_plan_lookup_failure_skips_quietly(self, r2, source_file, monkeypatch):
        async def boom(_u):
            raise RuntimeError("db down")
        monkeypatch.setattr(sa, "_owner_is_paying", boom)
        assert _archive(source_file) is None
        assert r2.objects == {}

    @pytest.mark.parametrize("off", ["days", "r2"])
    def test_disabled_does_nothing(self, r2, source_file, monkeypatch, off):
        if off == "days":
            monkeypatch.setattr(sa.config, "SOURCE_ARCHIVE_DAYS", 0)
        else:
            monkeypatch.setattr(sa, "settings", SimpleNamespace(r2_configured=False))
        assert _archive(source_file) is None
        assert r2.objects == {} and r2.state["lookups"] == 0

    def test_erased_while_uploading_leaves_no_orphan(self, r2, source_file):
        r2.state["users"] = set()
        assert _archive(source_file) is None
        assert r2.objects == {}

    def test_ids_that_are_not_uuids_never_reach_a_key(self, r2, source_file):
        assert asyncio.run(sa.archive_source(OWNER, "../x", str(source_file))) is None
        assert asyncio.run(sa.archive_source("nobody/..", JOB, str(source_file))) is None
        assert r2.objects == {}

    def test_paid_plan_rule(self, monkeypatch):
        metering = pytest.importorskip("cloud.metering")
        plans = {}

        async def live_plans(ids):
            return {u: p for u, p in plans.items() if u in {str(i) for i in ids}}

        monkeypatch.setattr(metering, "live_plans", live_plans)
        assert asyncio.run(sa._owner_is_paying(OWNER)) is False
        plans[OWNER] = "free"
        assert asyncio.run(sa._owner_is_paying(OWNER)) is False
        plans[OWNER] = "creator"
        assert asyncio.run(sa._owner_is_paying(OWNER)) is True


class TestUploadConcurrency:
    def test_at_most_two_uploads_at_once(self, r2, tmp_path, monkeypatch):
        live, peak = [0], [0]
        import threading
        lock = threading.Lock()

        def slow_upload(fileobj, key, content_type="video/mp4", metadata=None):
            with lock:
                live[0] += 1
                peak[0] = max(peak[0], live[0])
            time.sleep(0.05)
            with lock:
                live[0] -= 1
            r2.put(key, fileobj.read())

        monkeypatch.setattr(r2, "upload_fileobj", slow_upload)
        jobs = [f"c0ffee00-0000-4000-8000-00000000001{i}" for i in range(6)]
        paths = []
        for j in jobs:
            p = tmp_path / f"{j}.mp4"
            p.write_bytes(b"x")
            paths.append(p)

        async def run_all():
            await asyncio.gather(*(sa.archive_source(OWNER, j, str(p))
                                   for j, p in zip(jobs, paths)))
        asyncio.run(run_all())
        assert len(r2.source_keys()) == 6
        assert peak[0] <= sa.UPLOAD_CONCURRENCY == 2

    def test_the_bytes_survive_a_sweep_while_queued(self, r2, source_file, monkeypatch):
        # The disk sweeps may delete the name while the upload waits for its
        # slot; the handle opened before the wait still reads the file.
        real_upload = r2.upload_fileobj

        def upload_after_delete(fileobj, key, content_type="video/mp4", metadata=None):
            os.remove(source_file)
            real_upload(fileobj, key, content_type, metadata)

        monkeypatch.setattr(r2, "upload_fileobj", upload_after_delete)
        key = _archive(source_file)
        assert r2.objects[key]["data"] == b"the whole original"


# --- Lookup, fetch and sweep -------------------------------------------------

def _src_key(user=OWNER, job=JOB, name="0" * 32 + ".mp4"):
    return real_storage.job_key(user, job, f"source/{name}")


class TestFindAndFetch:
    def test_find_reads_the_duration_back_as_a_number(self, r2):
        r2.put(_src_key(), meta={"duration": "12.5"})
        info = asyncio.run(sa.find(OWNER, JOB))
        assert info == {"key": _src_key(), "duration": 12.5}

    def test_garbage_metadata_is_not_a_duration(self, r2):
        r2.put(_src_key(), meta={"duration": "1e400; rm -rf"})
        assert asyncio.run(sa.find(OWNER, JOB))["duration"] is None

    def test_another_users_prefix_is_never_searched(self, r2):
        r2.put(_src_key(user=OTHER))
        assert asyncio.run(sa.find(OWNER, JOB)) is None

    def test_an_object_past_its_date_is_not_offered(self, r2):
        r2.put(_src_key(), modified=datetime.now(timezone.utc) - timedelta(days=31))
        assert asyncio.run(sa.find(OWNER, JOB)) is None

    def test_names_this_module_never_writes_are_ignored(self, r2):
        r2.put(_src_key(name="../../evil.mp4"))
        r2.put(_src_key(name="x.sh"))
        assert asyncio.run(sa.find(OWNER, JOB)) is None

    def test_fetch_lands_under_the_fixed_dot_name(self, r2, tmp_path):
        r2.put(_src_key(name="a" * 32 + ".webm"), data=b"webm bytes")
        job_dir = tmp_path / JOB
        job_dir.mkdir()
        path = asyncio.run(sa.fetch(OWNER, JOB, str(job_dir)))
        assert path == str(job_dir / ".source_archive.webm")
        assert open(path, "rb").read() == b"webm bytes"
        assert not (job_dir / ".source_archive.part").exists()


class TestSweep:
    def test_deletes_the_expired_and_keeps_the_rest(self, r2):
        now = datetime.now(timezone.utc)
        old = _src_key(name="1" * 32 + ".mp4")
        young = _src_key(job="c0ffee00-0000-4000-8000-000000000099")
        clip = real_storage.job_key(OWNER, JOB, "x_clip_1.mp4")
        r2.put(old, modified=now - timedelta(days=31))
        r2.put(young, modified=now - timedelta(days=29))
        r2.put(clip, modified=now - timedelta(days=400))
        assert asyncio.run(sa.purge_expired_sources(force=True)) == 1
        assert set(r2.objects) == {young, clip}

    def test_runs_once_a_day(self, r2):
        r2.put(_src_key(), modified=datetime.now(timezone.utc) - timedelta(days=40))
        assert asyncio.run(sa.purge_expired_sources()) == 1
        r2.put(_src_key(name="2" * 32 + ".mp4"),
               modified=datetime.now(timezone.utc) - timedelta(days=40))
        assert asyncio.run(sa.purge_expired_sources()) == 0, "throttled"
        assert asyncio.run(sa.purge_expired_sources(force=True)) == 1

    def test_turned_off_deletes_everything_left(self, r2, monkeypatch):
        monkeypatch.setattr(sa.config, "SOURCE_ARCHIVE_DAYS", 0)
        r2.put(_src_key(), modified=datetime.now(timezone.utc))
        assert asyncio.run(sa.purge_expired_sources(force=True)) == 1
        assert r2.objects == {}

    def test_the_erasure_prefix_takes_it(self, r2):
        r2.put(_src_key())
        r2.delete_prefix(real_storage.user_prefix(OWNER))
        assert r2.objects == {}

    def test_is_wired_into_the_retention_sweeper(self):
        import inspect
        videos = pytest.importorskip("cloud.videos")
        assert "purge_expired_sources" in inspect.getsource(videos._sweeper_loop)


# --- app.py: restore, on-demand fetch, preview -------------------------------

app = pytest.importorskip("app")


def _client_get(path, headers=None):
    async def _do():
        transport = httpx.ASGITransport(app=app.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as c:
            return await c.get(path, headers=headers or {}, follow_redirects=False)
    return asyncio.run(_do())


@pytest.fixture
def cloud_app(r2, tmp_path, monkeypatch):
    """app.py in billing mode with the in-memory R2 and one finished job whose
    source is gone from disk."""
    out = tmp_path / "output"
    job_dir = out / JOB
    job_dir.mkdir(parents=True)
    (tmp_path / "up").mkdir()
    monkeypatch.setattr(app, "OUTPUT_DIR", str(out))
    monkeypatch.setattr(app, "UPLOAD_DIR", str(tmp_path / "up"))
    monkeypatch.setattr(app, "BILLING_ENABLED", True)
    monkeypatch.setattr(app, "_source_archive", sa)
    monkeypatch.setattr(app, "_cloud_config", SimpleNamespace(
        settings=SimpleNamespace(jwt_secret="test-secret"), EDITOR_ACTION_MINUTES=1))
    monkeypatch.setattr(app, "_source_fetch_locks", {})
    (job_dir / "x_metadata.json").write_text(json.dumps({
        "source_video": "source.mp4",
        "transcript": {"segments": [{"start": 0.0, "end": 30.0, "text": "hi",
                                     "words": [{"word": "hi", "start": 10.0, "end": 10.4}]}]},
        "shorts": [{"start": 10.0, "end": 20.0, "video_url": f"/videos/{JOB}/x_clip_1.mp4"}],
    }))
    (job_dir / ".owner").write_text(OWNER)
    app.jobs[JOB] = {"status": "completed", "logs": [], "user_id": OWNER,
                     "output_dir": str(job_dir),
                     "result": {"clips": [{"start": 10.0, "end": 20.0,
                                           "video_url": f"/videos/{JOB}/x_clip_1.mp4"}]}}

    async def caller(request):
        return request.headers.get("X-Test-User")

    monkeypatch.setattr(app, "_owner_id_from_request", caller)
    r2.put(_src_key(), data=b"archived original", meta={"duration": "95.0"})
    try:
        yield job_dir
    finally:
        app.jobs.pop(JOB, None)


class TestRestoreSkipsTheSource:
    def test_reopening_a_project_never_pulls_the_source(self, r2, tmp_path, monkeypatch):
        monkeypatch.setattr(app, "OUTPUT_DIR", str(tmp_path))
        monkeypatch.setattr(app, "_source_archive", sa)
        monkeypatch.setattr(app, "_restore_locks", {})
        meta_key = real_storage.job_key(OWNER, JOB, "x_metadata.json")
        clip_key = real_storage.job_key(OWNER, JOB, "x_clip_1.mp4")
        r2.put(meta_key, data=json.dumps({"shorts": [{"start": 0, "end": 5}]}).encode())
        r2.put(clip_key, data=b"clip")
        r2.put(_src_key(), data=b"hundreds of MB")
        monkeypatch.setattr(real_storage, "list_keys", r2.list_keys)
        monkeypatch.setattr(real_storage, "download_file", r2.download_file)
        proj = SimpleNamespace(metadata_r2_key=meta_key, state={"clips": [
            {"index": 0, "original_file": "x_clip_1.mp4", "server_file": "x_clip_1.mp4"}]})
        try:
            assert asyncio.run(app._restore_job_files(JOB, proj, OWNER)) is True
        finally:
            app.jobs.pop(JOB, None)
        assert _src_key() not in r2.downloads
        assert sorted(os.listdir(tmp_path / JOB)) == [".owner", "x_clip_1.mp4", "x_metadata.json"]


class TestOnDemand:
    def test_fetched_once_for_concurrent_callers(self, cloud_app, r2):
        async def both():
            return await asyncio.gather(app._locate_or_fetch_source(JOB),
                                        app._locate_or_fetch_source(JOB))
        a, b = asyncio.run(both())
        assert a == b == str(cloud_app / ".source_archive.mp4")
        assert r2.downloads == [_src_key()]
        # From now on it is an ordinary source on disk.
        assert app._locate_source(JOB) == a

    def test_nothing_archived_is_still_none(self, cloud_app, r2):
        r2.objects.clear()
        assert asyncio.run(app._locate_or_fetch_source(JOB)) is None

    def test_self_host_never_asks_r2(self, cloud_app, r2, monkeypatch):
        monkeypatch.setattr(app, "BILLING_ENABLED", False)
        assert asyncio.run(app._locate_or_fetch_source(JOB)) is None
        assert r2.downloads == []

    def test_the_edl_says_available_with_the_archived_duration(self, cloud_app, r2, monkeypatch):
        monkeypatch.setattr(app, "_signed_source_url", lambda j: f"/signed/{j}")
        r = _client_get(f"/api/clip/{JOB}/0/edl", headers={"X-Test-User": OWNER})
        assert r.status_code == 200, r.text
        src = r.json()["source"]
        assert src["available"] is True and src["url"] == f"/signed/{JOB}"
        assert src["duration"] == 95.0 and src["duration_estimated"] is False
        assert r2.downloads == [], "the EDL only looks, it never downloads"


class TestPreviewRedirect:
    def test_the_owner_is_sent_to_r2(self, cloud_app, r2):
        r = _client_get(f"/api/source/{JOB}", headers={"X-Test-User": OWNER})
        assert r.status_code == 307
        assert r.headers["location"].startswith(f"https://r2.test/{_src_key()}")
        assert r.headers["cache-control"] == "no-store"
        assert r2.downloads == [], "nothing is pulled onto the server for a preview"

    def test_someone_else_gets_a_404_and_no_link(self, cloud_app, r2):
        r = _client_get(f"/api/source/{JOB}", headers={"X-Test-User": OTHER})
        assert r.status_code == 404
        assert "location" not in r.headers

    def test_nobody_gets_a_404(self, cloud_app, r2):
        assert _client_get(f"/api/source/{JOB}").status_code == 404

    def test_a_signed_link_from_source_url_works_without_credentials(self, cloud_app, r2):
        exp = int(time.time()) + 60
        sig = app._source_signature(JOB, exp)
        r = _client_get(f"/api/source/{JOB}?exp={exp}&sig={sig}")
        assert r.status_code == 307

    def test_a_forged_signature_does_not(self, cloud_app, r2):
        exp = int(time.time()) + 60
        r = _client_get(f"/api/source/{JOB}?exp={exp}&sig={'0' * 32}")
        assert r.status_code == 404

    def test_a_local_source_is_still_streamed(self, cloud_app, r2):
        (cloud_app / "source.mp4").write_bytes(b"local")
        r = _client_get(f"/api/source/{JOB}", headers={"X-Test-User": OWNER})
        assert r.status_code == 200 and r.content == b"local"


class TestDeliveryNeverWaits:
    def test_the_archive_step_returns_before_the_upload(self, cloud_app, monkeypatch):
        (cloud_app / "source.mp4").write_bytes(b"fresh")
        started, release = [], None

        async def slow(owner, job_id, path, duration_probe=None):
            started.append((owner, job_id, os.path.basename(path)))
            await release.wait()

        async def archive_job(*a, **k):
            return None

        monkeypatch.setattr(sa, "archive_source", slow)
        monkeypatch.setattr(app, "cloud", SimpleNamespace(
            videos=SimpleNamespace(archive_job=archive_job)))

        async def run():
            nonlocal release
            release = asyncio.Event()
            await asyncio.wait_for(app._archive_managed_job(JOB), timeout=1)
            await asyncio.sleep(0)
            assert started == [(OWNER, JOB, "source.mp4")]
            assert app._source_archive_tasks, "still uploading after the step returned"
            release.set()
            await asyncio.gather(*list(app._source_archive_tasks))
        asyncio.run(run())

    def test_an_upload_that_raises_is_contained(self, cloud_app, monkeypatch, capsys):
        (cloud_app / "source.mp4").write_bytes(b"fresh")

        async def boom(*a, **k):
            raise RuntimeError("kaboom")

        async def archive_job(*a, **k):
            return None

        monkeypatch.setattr(sa, "archive_source", boom)
        monkeypatch.setattr(app, "cloud", SimpleNamespace(
            videos=SimpleNamespace(archive_job=archive_job)))

        async def run():
            await app._archive_managed_job(JOB)
            await asyncio.gather(*list(app._source_archive_tasks))
        asyncio.run(run())
        assert "Source archive error" in capsys.readouterr().out


class TestRerenderFetchesOnlyWhenNeeded:
    """/api/clip/rerender: a cut past the original range pulls the archived
    source back; a trim inside it stays on the fast path and never does."""

    @pytest.fixture
    def recut_calls(self, cloud_app, monkeypatch):
        (cloud_app / "x_clip_1.mp4").write_bytes(b"canonical")
        calls = []

        def fake_recut(input_path=None, **kw):
            calls.append(os.path.basename(input_path))
            raise RuntimeError("stop before ffmpeg")

        async def noop(*a, **k):
            return None

        monkeypatch.setattr(app, "require_managed_entitlement", noop)
        monkeypatch.setattr(app, "reserve_managed_action", noop)
        monkeypatch.setattr(app.recut, "perform_recut", fake_recut)
        return calls

    def _post(self, segments):
        async def _do():
            transport = httpx.ASGITransport(app=app.app, raise_app_exceptions=False)
            async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as c:
                return await c.post("/api/clip/rerender", headers={"X-Test-User": OWNER},
                                    json={"job_id": JOB, "clip_index": 0,
                                          "segments": segments, "reapply_captions": False})
        return asyncio.run(_do())

    def test_extending_the_cut_brings_the_source_back(self, recut_calls, r2):
        self._post([{"start": 5.0, "end": 25.0}])
        assert recut_calls == [".source_archive.mp4"]
        assert r2.downloads == [_src_key()]

    def test_a_trim_inside_the_range_never_downloads(self, recut_calls, r2):
        self._post([{"start": 11.0, "end": 19.0}])
        assert recut_calls == ["x_clip_1.mp4"]
        assert r2.downloads == []

    def test_without_an_archive_it_is_still_the_409(self, recut_calls, r2):
        r2.objects.clear()
        r = self._post([{"start": 5.0, "end": 25.0}])
        assert r.status_code == 409
        assert recut_calls == []


class TestSourceUrlForAJobWithoutFiles:
    """The job directory and the in-memory record are gone, the source is in R2:
    /api/source-url still mints for the owner the server knows, and only them."""

    @pytest.fixture
    def emptied(self, cloud_app, monkeypatch):
        import shutil
        app.jobs.pop(JOB, None)
        shutil.rmtree(cloud_app)
        monkeypatch.setattr(app, "_signed_source_url", lambda j: f"/signed/{j}")

        async def owner_of(kind, segment):
            return OWNER if segment == JOB else None

        async def user(request):
            uid = request.headers.get("X-Test-User")
            return SimpleNamespace(id=uid) if uid else None

        monkeypatch.setattr(app, "_media_owner", owner_of)
        monkeypatch.setattr(app, "_user_from_request", user)

    def test_the_owner_gets_a_link(self, emptied):
        r = _client_get(f"/api/source-url/{JOB}", headers={"X-Test-User": OWNER})
        assert r.status_code == 200 and r.json()["url"] == f"/signed/{JOB}"

    def test_someone_else_does_not(self, emptied):
        assert _client_get(f"/api/source-url/{JOB}",
                           headers={"X-Test-User": OTHER}).status_code == 404
        assert _client_get(f"/api/source-url/{JOB}").status_code == 404

    def test_nothing_archived_is_still_a_404(self, emptied, r2):
        r2.objects.clear()
        assert _client_get(f"/api/source-url/{JOB}",
                           headers={"X-Test-User": OWNER}).status_code == 404


def test_upload_fileobj_builds_a_transfer_config_boto3_accepts(tmp_path, monkeypatch):
    """The storage fake used above never builds the real TransferConfig: an
    option boto3 does not know made every real upload raise."""
    pytest.importorskip("boto3")
    from cloud import storage as real_storage
    sent = {}

    class Client:
        def upload_fileobj(self, fileobj, bucket, key, ExtraArgs=None, Config=None):
            sent.update(key=key, extra=ExtraArgs, config=Config, data=fileobj.read())

    monkeypatch.setattr(real_storage, "client", lambda: Client())
    path = tmp_path / "source.mp4"
    path.write_bytes(b"video")
    with open(path, "rb") as f:
        real_storage.upload_fileobj(f, "users/u/j/source/x.mp4", "video/mp4", {"duration": 12.5})
    assert sent["data"] == b"video" and sent["key"] == "users/u/j/source/x.mp4"
    assert sent["extra"] == {"ContentType": "video/mp4", "Metadata": {"duration": "12.5"}}
    assert sent["config"].max_request_concurrency == 2
