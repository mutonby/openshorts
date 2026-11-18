"""Ship the API's own log to object storage, in short gzip chunks.

The container's stdout is all there is, and the runtime's rotating log keeps
only a short window: an incident found a few hours later had no lines left to
read. With ``LOG_ARCHIVE_BUCKET`` set, everything the API process writes to
stdout/stderr (prints, uvicorn's error and access loggers) is also appended,
with a UTC timestamp per line, to a local chunk file; each closed chunk is
gzipped and uploaded as

    <LOG_ARCHIVE_PREFIX>/<YYYY-MM-DD>/<HHMM>-<host>-<boot>.log.gz

``host`` + ``boot`` keep two containers (a deploy handover) or two boots of the
same container from writing the same key. On shutdown the open chunk is
uploaded too. Read one back with ``aws s3 cp s3://<bucket>/<key> - | gunzip``.

Rules:
* Off unless ``LOG_ARCHIVE_BUCKET`` is set, so self-host never changes.
* The bucket must be private. Logging into the media bucket is refused when
  that bucket is served publicly (``R2_PUBLIC_BASE``). Retention is the
  bucket's lifecycle rule, not code (see docs/compliance/retencion.md).
* Logging must never break or block the API: every write and upload swallows
  its own errors, the stdout write always happens first, and when the local
  spool passes ``LOG_ARCHIVE_MAX_LOCAL_MB`` the oldest chunks are dropped.

Env: ``LOG_ARCHIVE_BUCKET``, ``LOG_ARCHIVE_PREFIX`` (``api``),
``LOG_ARCHIVE_CHUNK_MINUTES`` (10), ``LOG_ARCHIVE_DIR`` (temp dir),
``LOG_ARCHIVE_MAX_LOCAL_MB`` (512), and optionally ``LOG_ARCHIVE_ENDPOINT``,
``LOG_ARCHIVE_ACCESS_KEY_ID``, ``LOG_ARCHIVE_SECRET_ACCESS_KEY`` (default: the
R2 credentials the app already has).
"""
import atexit
import gzip
import logging
import os
import re
import socket
import sys
import tempfile
import threading
import time
from datetime import datetime, timezone

_UPLOAD_EVERY = 30.0
_FILE_RE = re.compile(r"^(\d{10})-([A-Za-z0-9]+)\.log$")

_state = {"sink": None, "thread": None, "orig_err": None}


def _env_int(name, default):
    try:
        return max(1, int(os.environ.get(name, "") or default))
    except ValueError:
        return default


def _safe(text, limit=40):
    return (re.sub(r"[^A-Za-z0-9-]", "", text or "") or "unknown")[:limit]


def _complain(msg):
    """To the real stderr only: never through the tee (no loop)."""
    stream = _state.get("orig_err") or sys.__stderr__
    try:
        stream.write(f"[log_archive] {msg}\n")
        stream.flush()
    except Exception:
        pass


class _Sink:
    """Line-buffered, chunked spool on local disk. Thread-safe; never raises."""

    def __init__(self, directory, chunk_seconds, boot, max_bytes, clock=time.time):
        self.dir = directory
        self.chunk = chunk_seconds
        self.boot = boot
        self.max_bytes = max_bytes
        self.clock = clock
        self.lock = threading.Lock()
        self.pending = ""
        self.fh = None
        self.fh_start = None
        self.dropping = False
        os.makedirs(self.dir, exist_ok=True)

    def chunk_start(self, now=None):
        now = self.clock() if now is None else now
        return int(now // self.chunk * self.chunk)

    def _path(self, start):
        return os.path.join(self.dir, f"{start:010d}-{self.boot}.log")

    def _open(self, start):
        if self.fh is not None and self.fh_start == start:
            return self.fh
        self._close()
        self.fh = open(self._path(start), "a", encoding="utf-8", errors="replace")
        self.fh_start = start
        return self.fh

    def _close(self):
        if self.fh is not None:
            try:
                self.fh.close()
            except Exception:
                pass
        self.fh = None
        self.fh_start = None

    def write(self, text):
        if not text:
            return
        try:
            with self.lock:
                self.pending += text
                if "\n" not in self.pending:
                    if len(self.pending) < 65536:
                        return
                    self.pending += "\n"
                lines, _, self.pending = self.pending.rpartition("\n")
                if self.dropping:
                    return
                now = self.clock()
                stamp = datetime.fromtimestamp(now, timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
                fh = self._open(self.chunk_start(now))
                fh.write("".join(f"{stamp} {line}\n" for line in lines.split("\n")))
                fh.flush()
        except Exception:
            pass

    def rotate(self):
        """Close the open chunk so the next upload pass takes it."""
        with self.lock:
            self._close()

    def ready_files(self, include_open=False):
        """Chunk files to upload, oldest first: every closed one, plus the open
        one when ``include_open`` (shutdown)."""
        try:
            names = sorted(n for n in os.listdir(self.dir) if _FILE_RE.match(n))
        except Exception:
            return []
        current = self._path(self.chunk_start()) if not include_open else None
        with self.lock:
            open_path = self._path(self.fh_start) if self.fh is not None else None
        out = []
        for name in names:
            path = os.path.join(self.dir, name)
            if not include_open and path in (current, open_path):
                continue
            out.append(path)
        return out

    def enforce_cap(self):
        """Drop the oldest chunks past the local cap; stop writing while even
        the open chunk alone is over it."""
        try:
            names = sorted(n for n in os.listdir(self.dir) if _FILE_RE.match(n))
            sizes = [(n, os.path.getsize(os.path.join(self.dir, n))) for n in names]
        except Exception:
            return
        total = sum(s for _, s in sizes)
        dropped = 0
        for name, size in sizes[:-1]:
            if total <= self.max_bytes:
                break
            try:
                os.remove(os.path.join(self.dir, name))
                total -= size
                dropped += 1
            except Exception:
                pass
        self.dropping = total > self.max_bytes
        if dropped:
            _complain(f"local spool over cap: dropped {dropped} chunk(s)")


class _Tee:
    """A text stream that writes to the real one first, then to the sink."""

    def __init__(self, stream, sink):
        self._stream = stream
        self._sink = sink

    def write(self, text):
        n = self._stream.write(text)
        self._sink.write(text)
        return n

    def writelines(self, lines):
        for line in lines:
            self.write(line)

    def flush(self):
        return self._stream.flush()

    def __getattr__(self, name):
        return getattr(self._stream, name)


def object_key(prefix, path, host):
    """Storage key for a local chunk file, or None for a foreign file name."""
    m = _FILE_RE.match(os.path.basename(path))
    if not m:
        return None
    start, boot = int(m.group(1)), m.group(2)
    t = datetime.fromtimestamp(start, timezone.utc)
    return f"{prefix}/{t:%Y-%m-%d}/{t:%H%M}-{host}-{boot}.log.gz"


def _make_client():
    import boto3
    from botocore.config import Config
    endpoint = os.environ.get("LOG_ARCHIVE_ENDPOINT", "").strip()
    key_id = os.environ.get("LOG_ARCHIVE_ACCESS_KEY_ID", "").strip()
    secret = os.environ.get("LOG_ARCHIVE_SECRET_ACCESS_KEY", "").strip()
    if not (endpoint and key_id and secret):
        from .config import settings
        endpoint = endpoint or settings.r2_endpoint
        key_id = key_id or settings.r2_access_key_id
        secret = secret or settings.r2_secret_access_key
    return boto3.client(
        "s3", endpoint_url=endpoint, aws_access_key_id=key_id,
        aws_secret_access_key=secret, region_name="auto",
        config=Config(signature_version="s3v4", retries={"max_attempts": 3},
                      connect_timeout=10, read_timeout=30))


class _Uploader:
    def __init__(self, sink, bucket, prefix, host, client_factory=_make_client):
        self.sink = sink
        self.bucket = bucket
        self.prefix = prefix
        self.host = host
        self.client_factory = client_factory
        self.client = None
        self.lock = threading.Lock()
        self.failing = False

    def upload_ready(self, include_open=False):
        """Upload every ready chunk; returns how many went up. Never raises."""
        sent = 0
        with self.lock:
            # A chunk whose time is over stays open until the next line; on a
            # quiet process that could be hours, so close it here.
            if include_open or (self.sink.fh_start is not None
                                and self.sink.fh_start < self.sink.chunk_start()):
                self.sink.rotate()
            for path in self.sink.ready_files(include_open=include_open):
                key = object_key(self.prefix, path, self.host)
                if key is None:
                    continue
                try:
                    with open(path, "rb") as f:
                        body = gzip.compress(f.read(), compresslevel=6)
                    if self.client is None:
                        self.client = self.client_factory()
                    self.client.put_object(Bucket=self.bucket, Key=key, Body=body,
                                           ContentType="text/plain",
                                           ContentEncoding="gzip")
                    os.remove(path)
                    sent += 1
                    if self.failing:
                        self.failing = False
                        _complain("uploads recovered")
                except Exception as e:
                    if not self.failing:
                        _complain(f"upload failed, will retry: {type(e).__name__}: {e}")
                    self.failing = True
                    break
            self.sink.enforce_cap()
        return sent

    def run(self, stop):
        while not stop.wait(_UPLOAD_EVERY):
            self.upload_ready()


def enabled():
    return bool(os.environ.get("LOG_ARCHIVE_BUCKET", "").strip())


def _bucket_is_public(bucket):
    try:
        from .config import settings
        return bucket == settings.r2_bucket and bool(settings.r2_public_base)
    except Exception:
        return False


def install():
    """Start teeing this process's output and the upload thread. Idempotent;
    a no-op unless ``LOG_ARCHIVE_BUCKET`` is set."""
    if _state["sink"] is not None or not enabled():
        return False
    bucket = os.environ["LOG_ARCHIVE_BUCKET"].strip()
    if _bucket_is_public(bucket):
        _complain("LOG_ARCHIVE_BUCKET is the public media bucket: refusing to ship logs there")
        return False
    prefix = _safe(os.environ.get("LOG_ARCHIVE_PREFIX", "api"), 64)
    chunk = _env_int("LOG_ARCHIVE_CHUNK_MINUTES", 10) * 60
    directory = (os.environ.get("LOG_ARCHIVE_DIR", "").strip()
                 or os.path.join(tempfile.gettempdir(), "openshorts-log-archive"))
    max_bytes = _env_int("LOG_ARCHIVE_MAX_LOCAL_MB", 512) * 1024 * 1024
    host = _safe(os.environ.get("HOSTNAME") or socket.gethostname(), 24)
    boot = f"{int(time.time()):x}{os.getpid():x}"
    try:
        sink = _Sink(directory, chunk, boot, max_bytes)
    except Exception as e:
        _complain(f"cannot spool logs in {directory}: {e}")
        return False

    orig_out, orig_err = sys.stdout, sys.stderr
    _state["orig_err"] = orig_err
    tee_out, tee_err = _Tee(orig_out, sink), _Tee(orig_err, sink)
    sys.stdout, sys.stderr = tee_out, tee_err
    # uvicorn configured its handlers before the app was imported, so they hold
    # the original streams: point them at the tees.
    for name in ("", "uvicorn", "uvicorn.error", "uvicorn.access"):
        for handler in logging.getLogger(name).handlers:
            if isinstance(handler, logging.StreamHandler):
                if handler.stream is orig_out:
                    handler.setStream(tee_out)
                elif handler.stream is orig_err:
                    handler.setStream(tee_err)

    uploader = _Uploader(sink, bucket, prefix, host)
    stop = threading.Event()
    thread = threading.Thread(target=uploader.run, args=(stop,),
                              name="log-archive", daemon=True)
    thread.start()
    _state.update(sink=sink, thread=thread, uploader=uploader, stop=stop)
    atexit.register(flush)
    print(f"🗄️  Log archive on: {chunk // 60}-min chunks to bucket '{bucket}' under '{prefix}/'")
    return True


def flush():
    """Upload everything spooled, the open chunk included (shutdown)."""
    uploader = _state.get("uploader")
    if uploader is None:
        return 0
    try:
        sys.stdout.flush()
        sys.stderr.flush()
    except Exception:
        pass
    return uploader.upload_ready(include_open=True)
