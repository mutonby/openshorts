"""Remote CPU worker.

A second host renders queued free jobs: this API fetches the source, the
worker runs main.py and sends the files back. No credential may cross the
wire, the worker may only write inside its job's directory, a job that the
worker drops must come back to this host, and a remote job must never hold
one of the GPU slots.
"""
import asyncio
import json
import os
import time
import uuid

import pytest

import remote_worker as rw


# --- what crosses the wire --------------------------------------------------

def test_job_options_keep_settings_and_drop_credentials():
    base = {"AUTO_LAYOUT": "1", "FFMPEG_THREADS": "2"}
    job = {"AUTO_LAYOUT": "1", "FFMPEG_THREADS": "2",           # same as the host
           "WATERMARK": "1", "CLIP_TARGET_MAX": "3", "MAX_SOURCE_MINUTES": "20",
           "GEMINI_API_KEY": "k", "PROXY_URL": "http://u:p@h", "STATIC_PROXY_URLS": "x",
           "YOUTUBE_COOKIES": "c", "R2_SECRET_ACCESS_KEY": "s", "DATABASE_URL": "d",
           "SOURCE_START_SECONDS": "90", "NVENC_SLOTS": "4", "PATH": "/x",
           "LD_PRELOAD": "/evil.so", "lower": "x", "BAD NAME": "x"}
    job.update(AUTO_HOOK_STYLE="pill", HF_ENDPOINT="http://evil", LLM_BASE_URL="http://evil",
               GOOGLE_GEMINI_BASE_URL="http://evil", SSL_CERT_FILE="/x", TMPDIR="/x",
               CLIP_EVIL="$(rm -rf /)", AUTO_NEWLINE="a\nb")
    out = rw.job_options(job, base)
    assert out == {"WATERMARK": "1", "CLIP_TARGET_MAX": "3", "MAX_SOURCE_MINUTES": "20",
                   "AUTO_HOOK_STYLE": "pill"}


def test_worker_args_for_a_url_job_and_an_upload():
    cmd = ["/venv/python", "-u", "main.py", "-u", "https://youtu.be/x", "--keep-original",
           "-o", "output/j", "--format", "square"]
    assert rw.worker_args(cmd) == ("https://youtu.be/x", ["--format", "square"])
    assert rw.source_is_url(cmd)
    up = ["python", "-u", "main.py", "-i", "uploads/j_v.mp4", "-o", "output/j"]
    assert rw.worker_args(up) == ("uploads/j_v.mp4", [])
    assert not rw.source_is_url(up)


@pytest.mark.parametrize("extra", [["--transcript", "output/j/t.json"], ["--skip-analysis"],
                                   ["--format", "../x"], ["--some-new-flag"], ["--format"]])
def test_commands_the_worker_cannot_repeat(extra):
    cmd = ["python", "-u", "main.py", "-i", "uploads/v.mp4", "-o", "output/j"] + extra
    assert rw.worker_args(cmd) is None


def test_the_worker_runs_only_the_closed_argument_list():
    assert rw.safe_args([]) == [] and rw.safe_args(["--format", "square"]) == ["--format", "square"]
    for bad in (["-o", "/tmp/x"], ["--format", "square", "-u", "http://x"],
                ["--transcript", "/etc/passwd"], ["--format", "evil"]):
        assert rw.safe_args(bad) is None


@pytest.mark.parametrize("name", [
    "V_clip_1.mp4", "V_metadata.json", "V_clip_1.mp4.layout.json", "autosubs_1_ab.ass",
    "thumb.png", "🎮 title_clip_2.mp4", "hooked_1_V_clip_1.MP4"])
def test_names_the_worker_may_write(name):
    assert rw.safe_relpath(name) == name


@pytest.mark.parametrize("name", [
    "", "/etc/passwd", "../x", "a/../../x", "a//b", "./x", "a\\b", "x\x00y",
    ".owner", ".resume.json", ".remote.json", ".remote.done", ".remote.up.abc",
    ".owner/x", "a/b/c", ".V_metadata.json.abc.tmp", "x" * 300,
    # No subdirectory: one named like a clip or the metadata would outlive a
    # purge and break the run that follows.
    "sub/file.png", "V_metadata.json/x.json", "V_clip_1.mp4/x.json", " lead.mp4", "trail.mp4 ",
    # Dotfiles are this host's: the free-plan mark, encode recipes, checkpoints.
    ".marked", ".hookrecipe_V_clip_1.mp4.json", ".hookrecipe_V_clip_1.mp4.png",
    ".transcript_checkpoint.json", "sub/.hidden.mp4", ".queue_expired",
    # Only what a render produces.
    "run.sh", "evil.py", "V.mp4.range", "noextension", "a.html", "a\nb.mp4"])
def test_names_the_worker_may_not_write(name):
    assert rw.safe_relpath(name) is None


# --- the file protocol ------------------------------------------------------

def test_claim_lifecycle(tmp_path):
    d = str(tmp_path)
    claim = rw.begin(d, "w1")
    assert rw.is_remote(d) and rw.valid(d, claim)["state"] == rw.PREPARING
    assert rw.valid(d, "wrong") is None and rw.valid(d, "") is None
    src = tmp_path / "V.mp4"
    src.write_bytes(b"12345")
    assert rw.set_ready(d, str(src), ["--format", "square"], {"WATERMARK": "1"})
    data = rw.valid(d, claim)
    assert (data["state"], data["source_size"], data["args"]) == (rw.READY, 5, ["--format", "square"])
    rw.append_log(d, ["a"])
    rw.mark_done(d, 0)
    rw.clear(d)
    assert not rw.is_remote(d) and rw.valid(d, claim) is None
    assert sorted(os.listdir(d)) == ["V.mp4"]          # only the job's own file is left


def test_a_new_claim_voids_the_old_one(tmp_path):
    d = str(tmp_path)
    old = rw.begin(d, "w1")
    new = rw.begin(d, "w2")
    assert rw.valid(d, old) is None and rw.valid(d, new)


def test_log_is_read_by_complete_lines(tmp_path):
    d = str(tmp_path)
    rw.append_log(d, ["one", " two\nthree ", "", None])
    lines, off = rw.read_log(d, 0)
    assert lines == ["one", "two three", "None"]
    with open(os.path.join(d, rw.LOG_FILE), "a") as f:
        f.write("half")
    assert rw.read_log(d, off) == ([], off)
    with open(os.path.join(d, rw.LOG_FILE), "a") as f:
        f.write(" line\n")
    assert rw.read_log(d, off)[0] == ["half line"]


def _put(d, name, offset, data, size=None, limit=1 << 20, job_limit=None, claim=None):
    if claim is None:
        claim = (rw.read(d) or {}).get("claim") or rw.begin(d, "w")
    rel, tmp, room = rw.open_upload(d, name, offset, limit, job_limit)
    assert len(data) <= room
    with open(tmp, "ab") as f:
        f.write(data)
    if size is not None:
        rw.finish_upload(d, rel, size, claim)
    return rel


def test_chunked_upload_lands_whole(tmp_path):
    d = str(tmp_path)
    _put(d, "V_clip_1.mp4", 0, b"abc")
    assert not os.path.exists(os.path.join(d, "V_clip_1.mp4"))      # never half served
    _put(d, "V_clip_1.mp4", 3, b"def", size=6)
    assert open(os.path.join(d, "V_clip_1.mp4"), "rb").read() == b"abcdef"
    assert [n for n in os.listdir(d) if n.startswith(rw.UPLOAD_PREFIX)] == []


def test_upload_refuses_a_wrong_offset_and_restarts_from_zero(tmp_path):
    d = str(tmp_path)
    _put(d, "a.mp4", 0, b"abc")
    with pytest.raises(rw.UploadError) as e:
        _put(d, "a.mp4", 5, b"x")
    assert e.value.status == 409
    _put(d, "a.mp4", 0, b"xyz", size=3)                 # offset 0 starts over
    assert open(os.path.join(d, "a.mp4"), "rb").read() == b"xyz"
    with pytest.raises(rw.UploadError):
        rw.finish_upload(d, "b.mp4", 3, rw.read(d)["claim"])   # nothing was sent


def test_upload_refuses_protected_names_and_escapes(tmp_path):
    d = str(tmp_path / "job")
    os.makedirs(d)
    for name in (".owner", ".resume.json", "../other/x.mp4", ".remote.json", ".marked",
                 ".hookrecipe_a.mp4.json"):
        with pytest.raises(rw.UploadError) as e:
            rw.open_upload(d, name, 0, 100)
        assert e.value.status == 400
    os.symlink(str(tmp_path / "elsewhere.mp4"), os.path.join(d, "link.mp4"))
    with pytest.raises(rw.UploadError):
        rw.open_upload(d, "link.mp4", 0, 100)            # a link out of the job dir
    _put(d, "big.mp4", 0, b"x" * 100, limit=100)
    with pytest.raises(rw.UploadError) as e:
        rw.open_upload(d, "big.mp4", 100, 100)             # already at the size limit
    assert e.value.status == 413


def test_the_worker_never_replaces_a_file_that_was_there_before_it(tmp_path):
    d = str(tmp_path)
    rw.begin(d, "w")
    (tmp_path / "V.mp4").write_bytes(b"the source this host downloaded")
    with pytest.raises(rw.UploadError) as e:
        rw.open_upload(d, "V.mp4", 0, 1000)
    assert e.value.status == 400
    _put(d, "V_clip_1.mp4", 0, b"a", size=1)
    _put(d, "V_clip_1.mp4", 0, b"bb", size=2)              # its own file: yes
    assert (tmp_path / "V_clip_1.mp4").read_bytes() == b"bb"


def test_a_job_has_an_upload_budget(tmp_path, monkeypatch):
    d = str(tmp_path)
    rw.begin(d, "w")
    _put(d, "a.mp4", 0, b"x" * 60, size=60, job_limit=100)
    _rel, _tmp, room = rw.open_upload(d, "b.mp4", 0, 1000, 100)
    assert room == 40                                      # what is left of the job's 100
    _put(d, "b.mp4", 0, b"x" * 40, size=40, job_limit=100)
    with pytest.raises(rw.UploadError) as e:
        rw.open_upload(d, "c.mp4", 0, 1000, 100)
    assert e.value.status == 413
    assert rw.open_upload(d, "a.mp4", 0, 1000, 100)[2] == 60   # replacing frees its bytes
    monkeypatch.setattr(rw, "MAX_FILES", 2)
    with pytest.raises(rw.UploadError):
        rw.open_upload(d, "d.mp4", 0, 1000, 10_000)
    monkeypatch.setattr(rw, "MAX_FILES", 400)
    monkeypatch.setattr(rw, "MAX_OPEN_UPLOADS", 1)
    _put(d, "a.mp4", 0, b"x")                              # a restart left unfinished
    with pytest.raises(rw.UploadError) as e:
        rw.open_upload(d, "e.mp4", 0, 1000, 10_000)
    assert e.value.status == 429


def test_only_one_upload_request_per_job_at_a_time(tmp_path):
    d = str(tmp_path)
    rw.begin(d, "w")
    with rw.upload_lock(d):
        with pytest.raises(rw.UploadError) as e:
            with rw.upload_lock(d):
                pass
        assert e.value.status == 429
    with rw.upload_lock(d):                               # free again
        pass


def test_the_budget_is_checked_again_on_what_actually_landed(tmp_path):
    """Whatever got appended to the temp file (requests that raced before the
    lock existed, a restart in between), the file only lands within budget."""
    d = str(tmp_path)
    claim = rw.begin(d, "w")
    rel, tmp, room = rw.open_upload(d, "big.mp4", 0, 10, 25)
    open(tmp, "ab").write(b"x" * 80)                      # eight writers' worth
    with pytest.raises(rw.UploadError) as e:
        rw.finish_upload(d, rel, 80, claim, 10, 25)
    assert e.value.status == 413
    assert not os.path.exists(os.path.join(d, "big.mp4")) and not os.path.exists(tmp)
    _put(d, "a.mp4", 0, b"x" * 20, size=20, job_limit=25)
    rel, tmp, room = rw.open_upload(d, "b.mp4", 0, 10, 25)
    open(tmp, "ab").write(b"x" * 9)
    with pytest.raises(rw.UploadError):
        rw.finish_upload(d, rel, 9, claim, 10, 25)        # 20 + 9 > 25


def test_files_the_api_parses_have_a_small_cap(tmp_path):
    d = str(tmp_path)
    rw.begin(d, "w")
    huge = 1024 ** 3
    assert rw.open_upload(d, "V_clip_1.mp4", 0, huge)[2] == huge
    for name in ("V_metadata.json", "autosubs_1_a.ass", "a.srt"):
        assert rw.open_upload(d, name, 0, huge)[2] == 8 * 1024 * 1024
    assert rw.open_upload(d, "thumb.png", 0, huge)[2] == 32 * 1024 * 1024


def test_metadata_from_the_worker_is_rewritten_as_data(tmp_path):
    d = str(tmp_path)
    rw.begin(d, "w")
    meta = {"shorts": [
        {"title": 7, "video_url": "/videos/another-job/secret.mp4", "start": 1, "end": 9,
         "recipe": {"segments": [[0, 1]] * 5000}, "why": "x" * 9000},
        {"title": "ok", "video_url": "https://evil.example/x.mp4",
         "recipe": {"segments": [[0, 1], [2, 3]]}}],
        "source_video": ".owner", "cost_analysis": {"total": 1}}
    raw = json.dumps(meta).encode()
    _put(d, "V_metadata.json", 0, raw, size=len(raw))
    out = json.load(open(os.path.join(d, "V_metadata.json")))
    assert all("video_url" not in c for c in out["shorts"])
    assert out["shorts"][0]["title"] == "7" and len(out["shorts"][0]["why"]) == 5000
    assert "segments" not in out["shorts"][0]["recipe"]
    assert out["shorts"][1]["recipe"]["segments"] == [[0, 1], [2, 3]]
    assert "source_video" not in out and out["cost_analysis"] == {"total": 1}
    for bad in (b"[1,2]", b"{not json", json.dumps({"shorts": "x"}).encode(),
                json.dumps({"shorts": [1]}).encode(),
                json.dumps({"shorts": [{}] * 100}).encode()):
        with pytest.raises(rw.UploadError) as e:
            _put(d, "W_metadata.json", 0, bad, size=len(bad))
        assert e.value.status == 400
    assert not os.path.exists(os.path.join(d, "W_metadata.json"))


def test_an_upload_held_open_cannot_land_after_the_claim_is_gone(tmp_path):
    d = str(tmp_path)
    old = rw.begin(d, "w")
    rel, tmp, _ = rw.open_upload(d, "V_metadata.json", 0, 1000)
    open(tmp, "ab").write(b"{}")
    rw.clear(d)                                            # the job went back to this host
    rw.begin(d, "w2")
    open(rw._upload_tmp(d, rel), "ab").write(b"{}")
    with pytest.raises(rw.UploadError) as e:
        rw.finish_upload(d, rel, 2, old)
    assert e.value.status == 410
    assert not os.path.exists(os.path.join(d, "V_metadata.json"))


def test_a_failed_remote_run_leaves_none_of_the_workers_files(tmp_path):
    d = str(tmp_path)
    rw.begin(d, "w")
    (tmp_path / "V.mp4").write_bytes(b"source")
    (tmp_path / ".owner").write_text("u")
    _put(d, "V_metadata.json", 0, b"{}", size=2)
    _put(d, "V_clip_1.mp4", 0, b"c", size=1)
    _put(d, "half.mp4", 0, b"c")
    rw.clear(d, purge=True)
    assert sorted(os.listdir(d)) == [".owner", "V.mp4"]


def test_the_log_has_a_budget(tmp_path, monkeypatch):
    d = str(tmp_path)
    monkeypatch.setattr(rw, "MAX_LOG_BYTES", 50)
    assert rw.append_log(d, ["x" * 20, "y" * 20, "z" * 20]) == 2
    assert rw.append_log(d, ["more"]) == 1
    assert rw.append_log(d, ["x" * 20]) == 0               # dropped, still a heartbeat
    assert os.path.getsize(os.path.join(d, rw.LOG_FILE)) <= 50
    assert rw.append_log(d, ["a\x00b\x1b[31m"]) == 0


# --- the worker's own guards --------------------------------------------------

def test_worker_refuses_a_job_id_that_is_not_a_uuid():
    worker = pytest.importorskip("worker")
    assert worker.valid_job_id("11111111-1111-1111-1111-111111111111")
    for bad in ("..", ".", "/etc", "../../x", "", None, 7,
                "11111111-1111-1111-1111-111111111111/..", "11111111111111111111111111111111"):
        assert not worker.valid_job_id(bad)


def test_worker_accepts_only_the_jobs_own_source_paths():
    worker = pytest.importorskip("worker")
    j = "11111111-1111-1111-1111-111111111111"
    assert worker.safe_job_path(j, f"output/{j}/V.mp4") == os.path.join("output", j, "V.mp4")
    assert worker.safe_job_path(j, "uploads/x_v.mp4") == os.path.join("uploads", "x_v.mp4")
    for bad in ("/etc/passwd", f"output/{j}/../x", "output/other/V.mp4", "main.py",
                f"output/{j}/a/b.mp4", "uploads/../main.py", None, ""):
        assert worker.safe_job_path(j, bad) is None


def test_worker_env_never_takes_code_paths_or_credentials(monkeypatch):
    worker = pytest.importorskip("worker")
    monkeypatch.setenv("GEMINI_API_KEY", "own")
    monkeypatch.setenv("WORKER_TOKEN", "t")
    env = worker.job_env({"WATERMARK": "1", "PATH": "/evil", "LD_PRELOAD": "x",
                          "PYTHONPATH": "/evil", "GEMINI_API_KEY": "theirs",
                          "PROXY_URL": "p", "WORKER_API_URL": "http://evil",
                          "GOOGLE_GEMINI_BASE_URL": "http://evil", "HF_ENDPOINT": "http://evil",
                          "SSL_CERT_FILE": "/evil", "TMPDIR": "/evil"})
    for name in ("GOOGLE_GEMINI_BASE_URL", "HF_ENDPOINT", "SSL_CERT_FILE"):
        assert env.get(name) == os.environ.get(name)
    assert env["WATERMARK"] == "1" and env["GEMINI_API_KEY"] == "own"
    assert env.get("PATH") == os.environ.get("PATH")
    assert "LD_PRELOAD" not in env and "PROXY_URL" not in env and "WORKER_TOKEN" not in env


def test_worker_env_does_not_inherit_a_host_wide_privacy_blur(monkeypatch):
    worker = pytest.importorskip("worker")
    monkeypatch.setenv("BLUR_SENSITIVE", "1")
    env = worker.job_env({"WATERMARK": "1"})
    assert "BLUR_SENSITIVE" not in env
    env = worker.job_env({"BLUR_SENSITIVE": "1", "WATERMARK": "1"})
    assert env["BLUR_SENSITIVE"] == "1" and env["WATERMARK"] == "1"


def test_blur_sensitive_is_an_allowed_worker_option():
    assert rw.allowed_option("BLUR_SENSITIVE", "1") is True
    assert rw.allowed_option("BLUR_SENSITIVE", "yes") is True
    assert rw.allowed_option("BLUR_SENSITIVE", "1\n2") is False
    assert rw.job_options({"BLUR_SENSITIVE": "1"}, {}) == {"BLUR_SENSITIVE": "1"}
    # Identical to this host: not repeated on the wire.
    assert rw.job_options({"BLUR_SENSITIVE": "1"}, {"BLUR_SENSITIVE": "1"}) == {}


# --- inside the API -----------------------------------------------------------

app_module = pytest.importorskip("app")
TOKEN = "t" * 40


@pytest.fixture
def api(tmp_path, monkeypatch):
    root = tmp_path / "output"
    root.mkdir()
    (tmp_path / "uploads").mkdir()
    monkeypatch.chdir(tmp_path)
    m = app_module
    monkeypatch.setattr(m, "OUTPUT_DIR", "output")
    monkeypatch.setattr(m, "UPLOAD_DIR", "uploads")
    monkeypatch.setattr(m, "INSTANCE_ID", "me")
    for name, value in (("_draining", False), ("_stopping", False),
                        ("_running_jobs", set()), ("_remote_jobs", set()),
                        ("_admitting", set()), ("_delivering_jobs", set()),
                        ("jobs", {}), ("job_queue", asyncio.PriorityQueue()),
                        ("concurrency_semaphore", asyncio.Semaphore(2)),
                        ("REMOTE_WORKER_TOKEN", TOKEN), ("REMOTE_WORKER_SLOTS", 2),
                        ("_remote_breaker", {"recent": [], "until": 0.0}),
                        ("REMOTE_LEASE_SECONDS", 180), ("AUTO_RETRY_DELAY_SECONDS", 0)):
        monkeypatch.setattr(m, name, value)
    monkeypatch.setenv("GEMINI_API_KEY", "server-key")

    async def nothing(*a, **k):
        return None
    for attr in ("_settle_reservation", "_archive_managed_job", "_unmark_if_owner_paid",
                 "_autopilot_job_finished", "_notify_job_webhook", "_record_job_alert",
                 "_track_proxy_usage", "_notify_clips_ready", "_notify_clip_activity"):
        monkeypatch.setattr(m, attr, nothing)
    return m


def _queued(m, priority=2, key="server-key", cmd=None, enqueue=True):
    job_id = str(uuid.uuid4())
    d = os.path.join("output", job_id)
    os.makedirs(d)
    cmd = cmd or ["python", "-u", "main.py", "-u", "https://youtu.be/x", "--keep-original",
                  "-o", d]
    with open(os.path.join(d, m._RESUME_FILE), "w") as f:
        json.dump({"cmd": cmd, "priority": priority, "attempts": 0}, f)
    m.jobs[job_id] = {"status": "queued", "logs": [], "cmd": cmd, "output_dir": d,
                      "env": {"GEMINI_API_KEY": key, "WATERMARK": "1", "PROXY_URL": "p"}}
    if enqueue:
        m.job_queue.put_nowait((priority, time.time(), job_id))
    return job_id, d


def _claim(m, monkeypatch):
    """_claim_remote_job without starting the job task."""
    started = []
    monkeypatch.setattr(m, "run_job_wrapper", lambda jid: started.append(jid))
    monkeypatch.setattr(m.asyncio, "create_task", lambda coro: None)
    return m._claim_remote_job("w1"), started


def test_nothing_is_claimed_while_the_gpu_has_room(api, monkeypatch):
    _queued(api)
    assert _claim(api, monkeypatch)[0] is None           # nobody waits for a slot
    assert api.job_queue.qsize() == 1


def test_claim_takes_the_first_free_job_and_no_gpu_slot(api, monkeypatch):
    api._admitting.add("someone-waiting")
    paid, _ = _queued(api, priority=1)
    free, d = _queued(api, priority=2)
    (job_id, claim), _ = _claim(api, monkeypatch)
    assert job_id == free
    assert api._remote_jobs == {free} and free in api._running_jobs
    assert api._local_running() == 0                      # not counted against the card
    assert [e[2] for e in api.job_queue._queue] == [paid]
    assert rw.valid(d, claim)
    api._release_slot(free)
    assert api.concurrency_semaphore._value == 2          # never took one, never returns one
    assert not api._remote_jobs and not api._running_jobs


@pytest.mark.parametrize("why", ["paid", "own_key", "retried", "transcript", "twitch",
                                 "draining", "off", "full", "short_token", "breaker"])
def test_jobs_that_stay_on_this_host(api, monkeypatch, why):
    api._admitting.add("someone-waiting")
    kw = {}
    if why == "paid":
        kw["priority"] = 1
    if why == "own_key":
        kw["key"] = "callers-own-key"
    if why == "transcript":
        kw["cmd"] = ["python", "-u", "main.py", "-i", "uploads/v.mp4", "--transcript", "t.json",
                     "-o", "output/x"]
    if why == "twitch":
        kw["cmd"] = ["python", "-u", "main.py", "-u", "https://www.twitch.tv/videos/1",
                     "-o", "output/x"]
    job_id, _ = _queued(api, **kw)
    if why == "retried":
        api.jobs[job_id]["auto_retries"] = 1
    if why == "draining":
        monkeypatch.setattr(api, "_draining", True)
    if why == "off":
        monkeypatch.setattr(api, "REMOTE_WORKER_TOKEN", "")
    if why == "full":
        api._remote_jobs.update({"a", "b"})
    if why == "short_token":
        monkeypatch.setattr(api, "REMOTE_WORKER_TOKEN", "t" * 31)
    if why == "breaker":
        monkeypatch.setattr(api, "REMOTE_BREAKER_FAILURES", 2)
        api._remote_outcome(False)
        api._remote_outcome(False)
    assert _claim(api, monkeypatch)[0] is None
    assert api.job_queue.qsize() == 1


def _fake_prepare(m, monkeypatch, d, returncode=0, source_name="V.mp4"):
    async def run_main(job_id, cmd, env, tick=None):
        assert cmd[-1] == "--prepare-only"
        if returncode == 0:
            with open(os.path.join(d, source_name), "wb") as f:
                f.write(b"source")
            m._handle_job_line(job_id, f"SOURCE_READY {os.path.join(d, source_name)}")
        return returncode
    monkeypatch.setattr(m, "_run_main", run_main)


def test_a_remote_job_ends_like_a_local_one(api, monkeypatch):
    m = api
    job_id, d = _queued(m, enqueue=False)
    m._running_jobs.add(job_id)
    m._remote_jobs.add(job_id)
    claim = rw.begin(d, "w1")
    _fake_prepare(m, monkeypatch, d)

    async def fast_sleep(_):
        await real_sleep(0)
    real_sleep = asyncio.sleep
    monkeypatch.setattr(m.asyncio, "sleep", fast_sleep)

    async def scenario():
        task = asyncio.ensure_future(m._run_job_remote(job_id, m.jobs[job_id]))
        for _ in range(200):
            await real_sleep(0)
            if (rw.read(d) or {}).get("state") == rw.READY:
                break
        data = rw.valid(d, claim)
        # What the worker is told: the options, never a credential.
        assert data["env"] == {"WATERMARK": "1", "HOOK_LAZY": "0"} and data["args"] == []
        # The worker renders and sends its files, lines and exit code.
        with open(os.path.join(d, "V_metadata.json"), "w") as f:
            json.dump({"shorts": [{"title": "a"}]}, f)
        _put(d, "V_clip_1.mp4", 0, b"clip", size=4)
        _put(d, "wm_V_clip_1.mp4", 0, b"clip", size=4)
        m.jobs[job_id]["watermark"] = True
        rw.append_log(d, ["🎙️ Transcribing… 100% (72s)", "CLIP_READY 0 wm_V_clip_1.mp4",
                          'JOB_CPU {"asr_device": "cpu"}'])
        rw.mark_done(d, 0)
        await task

    asyncio.run(scenario())
    job = m.jobs[job_id]
    assert job["status"] == "completed"
    assert job["ready_files"] == {0: "wm_V_clip_1.mp4"}
    assert job["result"]["clips"][0]["video_url"].endswith("V_clip_1.mp4")
    assert "🎙️ Transcribing… 100% (72s)" in job["logs"]
    assert not any("CLIP_READY" in l or "JOB_CPU" in l for l in job["logs"])
    assert not rw.is_remote(d)                            # the claim ended with the job
    assert os.path.exists(os.path.join(d, "V.mp4"))       # --keep-original: the source stays
    assert os.path.exists(os.path.join(d, "V_clip_1.mp4"))  # a good run keeps its clips
    # Free plan: this host set the mark; the worker could not have.
    assert app_module.watermarked.job_is_marked(d)


def test_a_quiet_worker_gives_the_job_back_as_a_retry(api, monkeypatch):
    m = api
    job_id, d = _queued(m, enqueue=False)
    m._running_jobs.add(job_id)
    m._remote_jobs.add(job_id)
    rw.begin(d, "w1")
    _fake_prepare(m, monkeypatch, d)
    monkeypatch.setattr(m, "REMOTE_LEASE_SECONDS", 0)

    async def fast_sleep(_):
        await real_sleep(0.01)
    real_sleep = asyncio.sleep
    monkeypatch.setattr(m.asyncio, "sleep", fast_sleep)
    asyncio.run(m._run_job_remote(job_id, m.jobs[job_id]))
    job = m.jobs[job_id]
    assert job["status"] == "failed"
    assert m._should_auto_retry(job)                      # so it is retried, here
    assert not rw.is_remote(d)                            # a late worker call gets 410
    assert m._remote_breaker["recent"] == [False]
    job["auto_retries"] = 1
    assert not m._remote_eligible(job_id, 2)              # and never goes remote again


def _run_remote_with(m, monkeypatch, d, job_id, worker_does):
    """_run_job_remote with a fake download; ``worker_does`` acts as the
    worker once the job is ready."""
    m._running_jobs.add(job_id)
    m._remote_jobs.add(job_id)
    claim = rw.begin(d, "w1")
    _fake_prepare(m, monkeypatch, d)
    real_sleep = asyncio.sleep

    async def fast_sleep(_):
        await real_sleep(0)
    monkeypatch.setattr(m.asyncio, "sleep", fast_sleep)

    async def scenario():
        task = asyncio.ensure_future(m._run_job_remote(job_id, m.jobs[job_id]))
        for _ in range(200):
            await real_sleep(0)
            if (rw.read(d) or {}).get("state") == rw.READY:
                break
        worker_does(claim)
        await task
    asyncio.run(scenario())
    return claim


def test_what_a_failed_remote_run_printed_decides_nothing(api, monkeypatch):
    """A worker cannot fail a user's video for good by printing a permanent
    error, nor plant text for the alerts: the job is retried here and its log
    starts again from this host's own lines."""
    m = api
    job_id, d = _queued(m, enqueue=False)

    def worker(claim):
        _put(d, "V_clip_1.mp4", 0, b"junk", size=4, claim=claim)
        rw.append_log(d, ["ERROR: Video unavailable", "❌ 407 proxy <a href=x>",
                          "no audio stream found"])
        rw.mark_done(d, 1)
    _run_remote_with(m, monkeypatch, d, job_id, worker)
    job = m.jobs[job_id]
    assert job["status"] == "failed" and job["remote_failed"]
    assert not any("unavailable" in l.lower() or "407" in l or "no audio" in l for l in job["logs"])
    assert m._should_auto_retry(job)
    assert not os.path.exists(os.path.join(d, "V_clip_1.mp4"))     # purged

    async def retry():
        m._schedule_auto_retry(job_id, job)
    asyncio.run(retry())
    assert "remote_failed" not in job and not m._remote_eligible(job_id, 2)


def test_a_job_that_completes_while_the_server_is_stopping_keeps_no_claim(api, monkeypatch):
    m = api
    job_id, d = _queued(m, enqueue=False)

    def worker(claim):
        raw = json.dumps({"shorts": [{"title": "a"}]}).encode()
        _put(d, "V_metadata.json", 0, raw, size=len(raw), claim=claim)
        _put(d, "wm_V_clip_1.mp4", 0, b"clip", size=4, claim=claim)
        rw.append_log(d, ["CLIP_READY 0 wm_V_clip_1.mp4"])
        monkeypatch.setattr(m, "_stopping", True)                  # SIGTERM during the run
        rw.mark_done(d, 0)
    claim = _run_remote_with(m, monkeypatch, d, job_id, worker)
    assert m.jobs[job_id]["status"] == "completed"
    assert rw.valid(d, claim) is None and not rw.is_remote(d)
    assert os.path.exists(os.path.join(d, "wm_V_clip_1.mp4"))      # delivered clips stay


def test_a_free_job_is_marked_before_the_worker_can_send_anything(api, monkeypatch):
    m = api
    job_id, d = _queued(m, enqueue=False)
    m.jobs[job_id]["watermark"] = True
    seen = {}

    def worker(claim):
        seen["marked_at_ready"] = app_module.watermarked.job_is_marked(d)
        # A clean name announced for a free job is not served.
        rw.append_log(d, ["CLIP_READY 0 V_clip_1.mp4", "CLIP_READY 1 wm_V_clip_2.mp4"])
        rw.mark_done(d, 1)
    _run_remote_with(m, monkeypatch, d, job_id, worker)
    assert seen["marked_at_ready"] is True
    assert m.jobs[job_id].get("ready_files") == {1: "wm_V_clip_2.mp4"}


def test_the_breaker_counts_a_window_not_a_streak(api, monkeypatch):
    m = api
    monkeypatch.setattr(m, "REMOTE_BREAKER_FAILURES", 4)
    for ok in (False, False, False, True, False):           # one good job in between
        m._remote_outcome(ok)
    assert m._remote_breaker["until"] > time.time()


def test_a_failed_download_fails_the_job_without_any_worker(api, monkeypatch):
    m = api
    job_id, d = _queued(m, enqueue=False)
    m._running_jobs.add(job_id)
    m._remote_jobs.add(job_id)
    rw.begin(d, "w1")
    _fake_prepare(m, monkeypatch, d, returncode=1)
    asyncio.run(m._run_job_remote(job_id, m.jobs[job_id]))
    assert m.jobs[job_id]["status"] == "failed"
    assert (rw.read(d) or {}).get("state") != rw.READY


def test_resuming_a_job_voids_its_remote_claim(api):
    m = api
    job_id, d = _queued(m, enqueue=False)
    del m.jobs[job_id]
    claim = rw.begin(d, "w1")
    m._register_resumed([(job_id, json.load(open(os.path.join(d, m._RESUME_FILE))))])
    assert m.jobs[job_id]["status"] == "queued"
    assert rw.valid(d, claim) is None


def test_jobs_rendered_remotely_do_not_fill_the_card_during_a_deploy(api):
    m = api
    job_id, d = _queued(m, enqueue=False)
    manifest = os.path.join(d, m._RESUME_FILE)
    data = json.load(open(manifest))
    data.update(instance="the-old-container", heartbeat=time.time())
    json.dump(data, open(manifest, "w"))
    assert m._jobs_busy_elsewhere() == 1
    rw.begin(d, "w1")
    assert m._jobs_busy_elsewhere() == 0


# --- the endpoints ------------------------------------------------------------

@pytest.fixture
def client(api):
    from fastapi.testclient import TestClient
    return TestClient(api.app)


AUTH = {"Authorization": f"Bearer {TOKEN}"}


def test_endpoints_do_not_exist_without_the_token(api, client, monkeypatch):
    job_id, d = _queued(api, enqueue=False)
    claim = rw.begin(d, "w1")
    paths = [("post", "/internal/worker/claim"), ("get", f"/internal/worker/jobs/{job_id}"),
             ("get", f"/internal/worker/jobs/{job_id}/source"),
             ("post", f"/internal/worker/jobs/{job_id}/done")]
    for method, path in paths:
        assert getattr(client, method)(path).status_code == 404
        bad = {"Authorization": "Bearer nope", "X-Worker-Claim": claim}
        assert getattr(client, method)(path, headers=bad).status_code == 404
    monkeypatch.setattr(api, "REMOTE_WORKER_SLOTS", 0)       # switched off
    for method, path in paths:
        good = dict(AUTH, **{"X-Worker-Claim": claim})
        assert getattr(client, method)(path, headers=good).status_code == 404


def test_a_claim_that_is_not_the_jobs_is_gone(api, client):
    job_id, d = _queued(api, enqueue=False)
    rw.begin(d, "w1")
    h = dict(AUTH, **{"X-Worker-Claim": "0" * 32})
    assert client.get(f"/internal/worker/jobs/{job_id}", headers=h).status_code == 410
    assert client.post(f"/internal/worker/jobs/{job_id}/log", headers=h,
                       json={"lines": ["x"]}).status_code == 410
    assert client.get("/internal/worker/jobs/not-a-uuid", headers=h).status_code == 404
    assert client.get("/internal/worker/jobs/..%2F..%2Fetc", headers=h).status_code == 404


def test_worker_round_trip_over_http(api, client):
    job_id, d = _queued(api, enqueue=False)
    claim = rw.begin(d, "w1")
    h = dict(AUTH, **{"X-Worker-Claim": claim})
    assert client.get(f"/internal/worker/jobs/{job_id}", headers=h).json() == {"state": "preparing"}
    assert client.get(f"/internal/worker/jobs/{job_id}/source", headers=h).status_code == 404
    src = os.path.join(d, "V.mp4")
    open(src, "wb").write(b"source-bytes")
    rw.set_ready(d, src, ["--format", "square"], {"WATERMARK": "1"})
    info = client.get(f"/internal/worker/jobs/{job_id}", headers=h).json()
    assert info == {"state": "ready", "source": src, "source_size": 12,
                    "args": ["--format", "square"], "env": {"WATERMARK": "1"}}
    assert client.get(f"/internal/worker/jobs/{job_id}/source", headers=h).content == b"source-bytes"

    url = f"/internal/worker/jobs/{job_id}/file"
    assert client.put(url, headers=h, params={"name": "V_clip_1.mp4", "offset": 0},
                      content=b"abc").status_code == 200
    assert client.put(url, headers=h, params={"name": "V_clip_1.mp4", "offset": 9},
                      content=b"x").status_code == 409
    assert client.put(url, headers=h, params={"name": "V_clip_1.mp4", "offset": 3, "size": 6},
                      content=b"def").status_code == 200
    assert open(os.path.join(d, "V_clip_1.mp4"), "rb").read() == b"abcdef"
    for name in (".owner", "../x.mp4", ".resume.json", ".marked", ".hookrecipe_a.mp4.json",
                 "V.mp4", ""):
        assert client.put(url, headers=h, params={"name": name, "offset": 0, "size": 1},
                          content=b"x").status_code == 400, name
    assert not os.path.exists(os.path.join(d, ".owner"))
    assert open(src, "rb").read() == b"source-bytes"        # the source was not replaced
    assert client.put(url, headers=h, params={"offset": "x"}).status_code == 400
    assert client.put(url, params={"offset": "x"}).status_code == 404     # no 422 without auth
    big = {"lines": ["x" * 1000] * 1200}
    assert client.post(f"/internal/worker/jobs/{job_id}/log", headers=h,
                       json=big).status_code == 413

    assert client.post(f"/internal/worker/jobs/{job_id}/log", headers=h,
                       json={"lines": ["hello"]}).status_code == 200
    assert rw.read_log(d, 0)[0] == ["hello"]
    assert client.post(f"/internal/worker/jobs/{job_id}/done", headers=h,
                       json={}).status_code == 400
    assert client.post(f"/internal/worker/jobs/{job_id}/done", headers=h,
                       json={"returncode": 0}).status_code == 200
    assert rw.read_done(d) == 0


def test_the_source_endpoint_serves_only_the_jobs_own_file(api, client, tmp_path):
    job_id, d = _queued(api, enqueue=False)
    claim = rw.begin(d, "w1")
    h = dict(AUTH, **{"X-Worker-Claim": claim})
    secret = tmp_path / "secret.txt"
    secret.write_text("no")
    rw.set_ready(d, str(secret), [], {})
    assert client.get(f"/internal/worker/jobs/{job_id}/source", headers=h).status_code == 404


# --- end to end: the real worker.py against the real endpoints -----------------

STUB_MAIN = r'''
import json, os, sys, time
a = sys.argv
src, out = a[a.index("-i") + 1], a[a.index("-o") + 1]
assert open(src, "rb").read() == b"source-bytes", "source not fetched whole"
assert "--format" in a and "--keep-original" not in a and "-u" not in a[2:]
os.makedirs(out, exist_ok=True)
json.dump({"shorts": [{"title": "a"}, {"title": "b"}]},
          open(os.path.join(out, "V_metadata.json"), "w"))
open(os.path.join(out, "V_clip_1.mp4"), "wb").write(b"1" * 70000)
print("CLIP_READY 0 V_clip_1.mp4", flush=True)
time.sleep(1)
open(os.path.join(out, "V_clip_2.mp4"), "wb").write(b"2" * 10)
open(os.path.join(out, ".marked"), "w").write("1")
print("CLIP_READY 1 V_clip_2.mp4", flush=True)
print("SEEN watermark=%s proxy=%s key=%s" % (
    os.environ.get("WATERMARK"), os.environ.get("PROXY_URL"),
    os.environ.get("GEMINI_API_KEY")), flush=True)
'''


def test_end_to_end_with_the_real_worker(api, monkeypatch, tmp_path):
    uvicorn = pytest.importorskip("uvicorn")
    pytest.importorskip("requests")
    import socket
    import subprocess
    import sys
    import threading
    m = api
    m._admitting.add("someone-waiting-for-the-gpu")
    job_id, d = _queued(m)
    m.jobs[job_id]["cmd"] += ["--format", "square"]

    async def prepare(jid, cmd, env, tick=None):
        with open(os.path.join(d, "V.mp4"), "wb") as f:
            f.write(b"source-bytes")
        m._handle_job_line(jid, f"SOURCE_READY {os.path.join(d, 'V.mp4')}")
        return 0
    monkeypatch.setattr(m, "_run_main", prepare)

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(m.app, host="127.0.0.1", port=port,
                                           lifespan="off", log_level="error"))
    threading.Thread(target=server.run, daemon=True).start()
    host = tmp_path / "workerhost"
    host.mkdir()
    (host / "main.py").write_text(STUB_MAIN)
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env = dict(os.environ, WORKER_API_URL=f"http://127.0.0.1:{port}", WORKER_TOKEN=TOKEN,
               WORKER_ALLOW_HTTP="1", WORKER_SLOTS="1", WORKER_POLL_SECONDS="0.2",
               GEMINI_API_KEY="the-workers-own-key", JOB_NICE="0")
    worker = subprocess.Popen([sys.executable, os.path.join(repo, "worker.py")],
                              cwd=str(host), env=env, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT)
    try:
        deadline = time.time() + 40
        job = m.jobs[job_id]
        while time.time() < deadline and (job["status"] != "completed" or job_id in m._running_jobs):
            time.sleep(0.2)
    finally:
        worker.terminate()
        out = worker.communicate(timeout=30)[0].decode("utf-8", "replace")
        server.should_exit = True
    assert job["status"] == "completed", out + "\n" + "\n".join(map(str, job["logs"]))
    assert [c["video_url"].rsplit("/", 1)[1] for c in job["result"]["clips"]] == [
        "V_clip_1.mp4", "V_clip_2.mp4"]
    assert open(os.path.join(d, "V_clip_1.mp4"), "rb").read() == b"1" * 70000
    assert open(os.path.join(d, ".marked")).read() == "1"   # written by the API, not sent
    # The job's option reached the worker; the API's proxy and model key did not.
    assert "SEEN watermark=1 proxy=None key=the-workers-own-key" in job["logs"]
    assert not m._remote_jobs and job_id not in m._running_jobs
    assert m.concurrency_semaphore._value == 2
    assert not rw.is_remote(d)
    assert os.listdir(str(host / "output")) == []          # the worker kept nothing


def test_worker_lines_cannot_set_this_hosts_markers(api):
    m = api
    job_id, d = _queued(m, enqueue=False)
    for line in ("SOURCE_READY /etc/passwd", "PROXY_BYTES=999999999", "PROXY_ROUTE=paid",
                 "CLIP_READY 0 ../other/x.mp4", "CLIP_READY 1 .owner", "CLIP_READY 2 V_clip_3.mp4"):
        m._handle_job_line(job_id, line, remote=True)
    job = m.jobs[job_id]
    assert "remote_source" not in job and "proxy_bytes" not in job and "proxy_route" not in job
    assert job["ready_files"] == {2: "V_clip_3.mp4"}
    assert job["logs"] == []
