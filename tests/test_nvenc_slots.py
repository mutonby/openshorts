"""NVENC encode slots: bin/ffmpeg caps how many job encodes share the card."""
import fcntl
import os
import shutil
import subprocess
import time

import pytest

WRAPPER = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "bin", "ffmpeg")

needs_flock = pytest.mark.skipif(shutil.which("flock") is None, reason="no flock binary")

NVENC_ARGS = ["-i", "a.mp4", "-c:v", "h264_nvenc", "out.mp4"]


@pytest.fixture
def host(tmp_path):
    """A lock dir and a fake ffmpeg that reports which slots are taken."""
    locks = tmp_path / "locks"
    locks.mkdir()
    fake = tmp_path / "fake-ffmpeg"
    # A slot is taken when a fresh `flock -n` on its file fails.
    fake.write_text(
        '#!/bin/sh\n'
        'for i in 0 1; do\n'
        '  f="$NVENC_LOCK_DIR/.nvenc-$i.lock"\n'
        '  if [ -e "$f" ] && ! flock -n "$f" true; then echo "held $i"; fi\n'
        'done\n'
        'echo ran\n')
    fake.chmod(0o755)

    def run(*args, slots="2", wait="30"):
        env = {"PATH": os.environ.get("PATH", ""), "FFMPEG_REAL_BIN": str(fake),
               "NVENC_SLOTS": slots, "NVENC_LOCK_DIR": str(locks),
               "NVENC_SLOT_WAIT": wait}
        start = time.monotonic()
        out = subprocess.run(["sh", WRAPPER, *args], env=env, capture_output=True,
                             text=True, check=True, timeout=60).stdout
        return out.splitlines(), time.monotonic() - start

    def hold(i):
        fh = open(locks / f".nvenc-{i}.lock", "a+")
        fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return fh

    return run, hold


@needs_flock
def test_an_encode_holds_a_slot_while_it_runs(host):
    run, _ = host
    lines, _ = run(*NVENC_ARGS)
    assert lines == ["held 0", "ran"]


@needs_flock
def test_takes_the_next_free_slot(host):
    run, hold = host
    busy = hold(0)
    try:
        lines, took = run(*NVENC_ARGS)
    finally:
        busy.close()
    assert lines == ["held 0", "held 1", "ran"]
    assert took < 5


@needs_flock
def test_runs_anyway_after_the_wait_limit(host):
    run, hold = host
    busy = [hold(0), hold(1)]
    try:
        lines, took = run(*NVENC_ARGS, wait="1")
    finally:
        for fh in busy:
            fh.close()
    assert lines[-1] == "ran"
    assert 1 <= took < 10


@needs_flock
def test_starts_as_soon_as_a_slot_frees(host, tmp_path):
    run, _ = host
    lock = tmp_path / "locks" / ".nvenc-0.lock"
    # Another encode holds the only slot for 2 s; this one must start then,
    # not at the 30 s limit.
    holder = subprocess.Popen(["flock", str(lock), "sleep", "2"])
    try:
        deadline = time.monotonic() + 5
        while (subprocess.run(["flock", "-n", str(lock), "true"]).returncode == 0
               and time.monotonic() < deadline):
            time.sleep(0.05)
        lines, took = run(*NVENC_ARGS, slots="1")
    finally:
        holder.kill()
        holder.wait()
    assert lines == ["held 0", "ran"]
    assert 1 <= took < 8


@needs_flock
@pytest.mark.parametrize("args", [
    ["-f", "rawvideo", "-i", "pipe:0", "-c:v", "h264_nvenc", "out.mp4"],
    ["-f", "rawvideo", "-i", "-", "-c:v", "h264_nvenc", "out.mp4"],
    ["-i", "a.mp4", "-c:v", "libx264", "out.mp4"],
    ["-hwaccel", "cuda", "-i", "a.mp4", "-f", "rawvideo", "-"],
])
def test_piped_and_non_nvenc_runs_are_never_gated(host, args):
    run, hold = host
    busy = [hold(0), hold(1)]
    try:
        lines, took = run(*args)
    finally:
        for fh in busy:
            fh.close()
    assert lines[-1] == "ran"
    assert took < 5


@needs_flock
@pytest.mark.parametrize("slots", ["0", "", "x"])
def test_no_cap_when_slots_unset(host, slots):
    run, _ = host
    lines, _ = run(*NVENC_ARGS, slots=slots)
    assert lines == ["ran"]


def test_no_cap_without_a_lock_dir(tmp_path):
    fake = tmp_path / "fake-ffmpeg"
    fake.write_text('#!/bin/sh\necho ran\n')
    fake.chmod(0o755)
    env = {"PATH": os.environ.get("PATH", ""), "FFMPEG_REAL_BIN": str(fake),
           "NVENC_SLOTS": "2", "NVENC_LOCK_DIR": str(tmp_path / "missing")}
    out = subprocess.run(["sh", WRAPPER, *NVENC_ARGS], env=env, capture_output=True,
                         text=True, check=True, timeout=30).stdout
    assert out.splitlines() == ["ran"]


def test_job_env_carries_the_cap(monkeypatch):
    app_module = pytest.importorskip("app")
    base = {"FOO": "1"}
    monkeypatch.delenv("JOB_NVENC_SLOTS", raising=False)
    env = app_module._job_encode_env(base)
    assert env["NVENC_SLOTS"] == "4"
    assert env["NVENC_LOCK_DIR"] == os.path.abspath(app_module.OUTPUT_DIR)
    assert env["FOO"] == "1" and "NVENC_SLOTS" not in base
    monkeypatch.setenv("JOB_NVENC_SLOTS", "0")
    assert app_module._job_encode_env(base)["NVENC_SLOTS"] == "0"
    monkeypatch.setenv("JOB_NVENC_SLOTS", "lots")
    assert app_module._job_encode_env(base)["NVENC_SLOTS"] == "4"


def test_nvenc_probe_never_waits_for_a_slot(monkeypatch):
    import ffmpeg_utils
    seen = {}

    def fake_run(cmd, **kwargs):
        seen.update(kwargs.get("env") or {})
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setenv("NVENC_SLOTS", "4")
    monkeypatch.setattr(ffmpeg_utils.subprocess, "run", fake_run)
    assert ffmpeg_utils._probe_nvenc() is True
    assert seen["NVENC_SLOTS"] == "0"


@needs_flock
def test_a_waiting_encode_writes_its_wait_to_the_job_log(host, tmp_path):
    """main.py sums NVENC_WAIT_LOG into JOB_CPU (nvenc_wait_s, nvenc_waits)."""
    _, hold = host
    log = tmp_path / "nvenc-wait"
    env = {"PATH": os.environ.get("PATH", ""), "FFMPEG_REAL_BIN": "/bin/true",
           "NVENC_SLOTS": "1", "NVENC_LOCK_DIR": str(tmp_path / "locks"),
           "NVENC_SLOT_WAIT": "30", "NVENC_WAIT_LOG": str(log)}
    subprocess.run(["sh", WRAPPER, *NVENC_ARGS], env=env, check=True, timeout=60)
    assert not log.exists()                      # no wait, no line
    busy = hold(0)
    p = subprocess.Popen(["sh", WRAPPER, *NVENC_ARGS], env=env)
    time.sleep(2.5)
    busy.close()
    assert p.wait(timeout=30) == 0
    waits = [int(x) for x in log.read_text().split()]
    assert len(waits) == 1 and waits[0] >= 2
