"""Remote CPU worker: renders queued jobs of an OpenShorts API on another host.

    WORKER_API_URL=https://api.example.com WORKER_TOKEN=... python worker.py

It only makes outbound calls (see docs/architecture/remote-worker.md): claim a
job, fetch the source the API host already downloaded, run the same main.py
here, send every output file back, report the exit code. It holds no database,
storage, proxy or cookie credential, only its own model key (GEMINI_API_KEY in
its environment) and the worker token.

Run it from the image's /app, so the relative paths of a job (``output/<id>``,
``uploads/<file>``) are the ones the API host uses.
"""
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
import uuid

import requests

import remote_worker

API = os.environ.get("WORKER_API_URL", "").rstrip("/")
TOKEN = os.environ.get("WORKER_TOKEN", "").strip()
SLOTS = max(1, int(os.environ.get("WORKER_SLOTS", "2") or 2))
NAME = os.environ.get("WORKER_NAME") or socket.gethostname()
POLL_SECONDS = float(os.environ.get("WORKER_POLL_SECONDS", "5") or 5)
READY_TIMEOUT = float(os.environ.get("WORKER_READY_TIMEOUT", "1800") or 1800)
CHUNK = 32 * 1024 * 1024
NICE = max(0, min(19, int(os.environ.get("JOB_NICE", "10") or 10)))
MAX_SOURCE_BYTES = int(float(os.environ.get("WORKER_MAX_SOURCE_GB", "20") or 20) * 1024 ** 3)

_stop = threading.Event()


class Gone(Exception):
    """The API says the job is no longer ours: drop it."""


def log(msg):
    print(f"[worker] {msg}", flush=True)


def _headers(claim=None):
    h = {"Authorization": f"Bearer {TOKEN}"}
    if claim:
        h["X-Worker-Claim"] = claim
    return h


def _call(method, path, claim=None, tries=5, **kw):
    """One API call, retried on network errors and 5xx. 410 raises Gone."""
    last = None
    timeout = kw.pop("timeout", 60)
    for attempt in range(tries):
        try:
            # No redirects: the token and the claim go to the API and nowhere else.
            r = requests.request(method, f"{API}{path}", headers=_headers(claim),
                                 timeout=timeout, allow_redirects=False, **kw)
            if r.status_code == 410:
                raise Gone()
            if r.status_code < 500:
                return r
            last = f"HTTP {r.status_code}"
        except Gone:
            raise
        except requests.RequestException as e:
            last = type(e).__name__
        time.sleep(min(30, 2 ** attempt))
    raise RuntimeError(f"{method} {path} failed: {last}")


def valid_job_id(job_id):
    """A job id is a canonical UUID: it becomes a directory name here."""
    try:
        return isinstance(job_id, str) and str(uuid.UUID(job_id)) == job_id
    except (ValueError, AttributeError, TypeError):
        return False


def safe_job_path(job_id, path):
    """``path`` if it is the job's source location as the API host names it:
    inside ``output/<job_id>/`` or ``uploads/``, no traversal. Else None."""
    if not isinstance(path, str) or not path or path.startswith("/") or "\x00" in path:
        return None
    parts = path.replace("\\", "/").split("/")
    if any(p in ("", ".", "..") for p in parts):
        return None
    if parts[:2] == ["output", job_id] and len(parts) == 3:
        return os.path.join(*parts)
    if parts[0] == "uploads" and len(parts) == 2:
        return os.path.join(*parts)
    return None


def job_env(options):
    """This worker's environment plus the job's settings, from the same
    closed list of names the API sends (``remote_worker.allowed_option``):
    never a path, an endpoint or a credential."""
    env = dict(os.environ)
    for name, value in (options or {}).items():
        if remote_worker.allowed_option(name, value):
            env[name] = str(value)
    for name in ("WORKER_TOKEN", "WORKER_API_URL"):
        env.pop(name, None)       # main.py has no use for them
    return env


def fetch_source(job_id, claim, target, size):
    os.makedirs(os.path.dirname(target), exist_ok=True)
    r = _call("GET", f"/internal/worker/jobs/{job_id}/source", claim, stream=True, timeout=120)
    if r.status_code != 200:
        raise RuntimeError(f"source: HTTP {r.status_code}")
    limit = min(MAX_SOURCE_BYTES, int(size)) if size is not None else MAX_SOURCE_BYTES
    got = 0
    with open(target, "wb") as f:
        for chunk in r.iter_content(1024 * 1024):
            got += len(chunk)
            if got > limit:
                raise RuntimeError("source: larger than announced")
            f.write(chunk)
    if size is not None and got != int(size):
        raise RuntimeError(f"source: {got} bytes, expected {size}")


def upload_file(job_id, claim, out_dir, rel):
    """Send one output file in chunks; a refused offset starts it over once."""
    path = os.path.join(out_dir, rel)
    size = os.path.getsize(path)
    for attempt in range(2):
        offset, ok = 0, True
        with open(path, "rb") as f:
            while True:
                data = f.read(CHUNK)
                last = offset + len(data) >= size
                params = {"name": rel.replace(os.sep, "/"), "offset": offset}
                if last:
                    params["size"] = size
                r = _call("PUT", f"/internal/worker/jobs/{job_id}/file", claim,
                          params=params, data=data, timeout=600)
                if r.status_code != 200:
                    ok = False
                    log(f"{job_id}: upload of {rel!r} refused (HTTP {r.status_code})")
                    break
                offset += len(data)
                if last:
                    break
        if ok:
            return True
        if r.status_code != 409:
            return False
    return False


class Uploader:
    """Sends each output file once per version (size + mtime)."""

    def __init__(self, job_id, claim, out_dir, skip):
        self.job_id, self.claim, self.out_dir = job_id, claim, out_dir
        self.skip = {os.path.normpath(p) for p in skip}
        self.sent = {}
        self.lock = threading.Lock()

    def _stamp(self, path):
        st = os.stat(path)
        return st.st_size, st.st_mtime_ns

    def send(self, rel):
        path = os.path.join(self.out_dir, rel)
        if os.path.normpath(path) in self.skip or not os.path.isfile(path) or os.path.islink(path):
            return
        if remote_worker.safe_relpath(rel.replace(os.sep, "/")) is None:
            return        # dotfiles and scratch files stay here: the API refuses them
        with self.lock:
            try:
                stamp = self._stamp(path)
            except OSError:
                return
            if self.sent.get(rel) == stamp:
                return
            if upload_file(self.job_id, self.claim, self.out_dir, rel):
                self.sent[rel] = stamp

    def send_all(self):
        for name in sorted(os.listdir(self.out_dir)):
            self.send(name)


def run_job(job_id, claim):
    if not valid_job_id(job_id):
        return
    out_dir = os.path.join("output", job_id)
    source = None
    proc = None
    try:
        deadline = time.time() + READY_TIMEOUT
        while True:
            r = _call("GET", f"/internal/worker/jobs/{job_id}", claim)
            info = r.json() if r.status_code == 200 else {}
            if info.get("state") == "ready":
                break
            if time.time() > deadline or _stop.is_set():
                raise RuntimeError("the source never became ready")
            time.sleep(3)

        source = safe_job_path(job_id, info.get("source"))
        if source is None:
            raise RuntimeError("source path refused")
        os.makedirs(out_dir, exist_ok=True)
        if int(info.get("source_size") or 0) > MAX_SOURCE_BYTES:
            raise RuntimeError("source too large for this worker")
        args = remote_worker.safe_args(info.get("args"))
        if args is None:
            raise RuntimeError("arguments refused")
        fetch_source(job_id, claim, source, info.get("source_size"))
        cmd = [sys.executable, "-u", "main.py", "-i", source, "-o", out_dir] + args
        log(f"{job_id}: rendering ({os.path.getsize(source) >> 20} MB source)")
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                env=job_env(info.get("env")), cwd=os.getcwd(),
                                preexec_fn=(lambda: os.nice(NICE)) if NICE else None)
        uploader = Uploader(job_id, claim, out_dir, skip=[source])
        pending, lock = [], threading.Lock()

        def reader():
            for raw in iter(proc.stdout.readline, b""):
                line = raw.decode("utf-8", "replace").rstrip()
                if not line:
                    continue
                if line.startswith("CLIP_READY "):
                    # The API serves a clip as soon as it hears this line: the
                    # file and the metadata that lists it go first.
                    try:
                        uploader.send(line.split(" ", 2)[2])
                        for name in os.listdir(out_dir):
                            if name.endswith("_metadata.json"):
                                uploader.send(name)
                    except Gone:
                        pass
                    except Exception as e:
                        log(f"{job_id}: early upload failed ({e})")
                with lock:
                    pending.append(line)
            proc.stdout.close()

        t = threading.Thread(target=reader, daemon=True)
        t.start()

        def flush():
            with lock:
                lines, pending[:] = list(pending), []
            try:
                r = _call("POST", f"/internal/worker/jobs/{job_id}/log", claim,
                          json={"lines": lines})
            except Gone:
                raise
            except Exception:
                with lock:
                    pending[:0] = lines      # the next beat sends them
                return
            if r.status_code != 200:
                raise Gone()

        while proc.poll() is None:
            if _stop.is_set():
                raise RuntimeError("worker stopping")
            flush()                          # also the heartbeat
            time.sleep(3)
        t.join(30)
        uploader.send_all()
        flush()
        _call("POST", f"/internal/worker/jobs/{job_id}/done", claim,
              json={"returncode": int(proc.returncode)})
        log(f"{job_id}: done (exit {proc.returncode})")
    except Gone:
        log(f"{job_id}: no longer ours, dropped")
    except Exception as e:
        log(f"{job_id}: failed here ({type(e).__name__}: {e})")
        # Say nothing more: the API takes the job back once we go quiet, or
        # at once if we can still tell it.
        try:
            _call("POST", f"/internal/worker/jobs/{job_id}/log", claim, tries=1,
                  json={"lines": [f"Execution error: remote worker failed ({type(e).__name__}) (killed)."]})
            _call("POST", f"/internal/worker/jobs/{job_id}/done", claim, tries=1,
                  json={"returncode": -15})
        except Exception:
            pass
    finally:
        if proc is not None and proc.poll() is None:
            proc.kill()
        shutil.rmtree(out_dir, ignore_errors=True)
        if source and os.path.exists(source):
            try:
                os.remove(source)
            except OSError:
                pass


def main():
    if not API or not TOKEN:
        sys.exit("WORKER_API_URL and WORKER_TOKEN are required")
    if not API.startswith("https://") and os.environ.get("WORKER_ALLOW_HTTP") != "1":
        sys.exit("WORKER_API_URL must be https")
    signal.signal(signal.SIGTERM, lambda *_: _stop.set())
    signal.signal(signal.SIGINT, lambda *_: _stop.set())
    log(f"{NAME}: {SLOTS} slot(s), API {API}")
    running = []
    while not _stop.is_set():
        running = [t for t in running if t.is_alive()]
        claimed = False
        if len(running) < SLOTS:
            try:
                r = _call("POST", "/internal/worker/claim", tries=1, json={"worker": NAME})
                body = r.json() if r.status_code == 200 else {}
                if body.get("job_id") and not valid_job_id(body["job_id"]):
                    log("claim: job id refused")
                elif body.get("job_id") and body.get("claim"):
                    t = threading.Thread(target=run_job, args=(body["job_id"], body["claim"]),
                                         daemon=True)
                    t.start()
                    running.append(t)
                    claimed = True
                elif r.status_code != 200:
                    log(f"claim: HTTP {r.status_code}")
            except Exception as e:
                log(f"claim failed ({type(e).__name__})")
        if not claimed:
            _stop.wait(POLL_SECONDS)
    log("stopping: waiting for running jobs to notice")
    for t in running:
        t.join(20)


if __name__ == "__main__":
    main()
