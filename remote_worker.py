"""Server side of the remote CPU worker (see docs/architecture/remote-worker.md).

A second machine without a GPU can render queued free jobs. It runs
``worker.py``, which only ever makes outbound HTTPS calls to this API: it
claims a job, fetches the source this host already downloaded, runs the same
``main.py`` on its CPU and sends the output files back into the job's
directory. From there the job is delivered exactly like a local one.

Everything the worker says lands as files in the job directory, not in the
memory of the instance that answered the request:

    .remote.json   the claim (id, worker, state, command, options, source)
    .remote.log    the output lines of the worker's main.py
    .remote.seen   touched on every call of the worker (its heartbeat)
    .remote.files  the names the worker has uploaded so far
    .remote.done   {"returncode": n} once its main.py exited

so during a deploy handover the worker can talk to either container while the
one that dispatched the job follows those files, as it follows a subprocess.

This module holds the parts with no FastAPI or queue in them: what crosses
the wire, which file names a worker may write, and the file protocol.
"""
import contextlib
import fcntl
import hmac
import json
import os
import re
import secrets
import time

from jsonfile import write_json_atomic

CLAIM_FILE = ".remote.json"
LOG_FILE = ".remote.log"
SEEN_FILE = ".remote.seen"
DONE_FILE = ".remote.done"
FILES_FILE = ".remote.files"
LOCK_FILE = ".remote.lock"
# A token shorter than this is as good as none: the feature stays off.
MIN_TOKEN_LENGTH = 32
UPLOAD_PREFIX = ".remote.up."
RESERVED_PREFIX = ".remote."

PREPARING = "preparing"
READY = "ready"

# What the worker may send back: the files a render produces, by extension,
# and never a dotfile. The API trusts the dotfiles of a job directory (who
# owns it, its manifest, the free-plan mark, encode recipes), so those are
# written on this host only.
_MAX_NAME_BYTES = 255
_UPLOAD_EXTENSIONS = frozenset({
    ".mp4", ".json", ".ass", ".srt", ".vtt", ".png", ".jpg", ".jpeg", ".webp"})
# Everything but video is small, and some of it is parsed by the API (the
# metadata on every poll): a size no render reaches is refused outright.
_SMALL_FILE_BYTES = 8 * 1024 * 1024
_SMALL_EXTENSIONS = frozenset({".json", ".ass", ".srt", ".vtt"})
_IMAGE_FILE_BYTES = 32 * 1024 * 1024

# Job options travel to the worker by name, from a closed list; credentials
# never do. The worker has its own model key and needs no proxy, cookie or
# storage secret: this host downloads the source and delivers the result.
_OPTION_NAMES = frozenset({
    "WATERMARK", "MAX_SOURCE_MINUTES", "SOURCE_CAP_MINUTES", "HOOK_LAZY",
    "CUT_MAX_FPS"})
_OPTION_PREFIXES = ("AUTO_", "CLIP_")
_OPTION_VALUE = re.compile(r"^[A-Za-z0-9 _.,:+-]{0,120}$")
_FORMATS = frozenset({"auto", "vertical", "horizontal", "square"})


def allowed_option(name, value=""):
    """Whether ``name=value`` is a job setting that may cross between hosts."""
    if not isinstance(name, str) or not re.match(r"^[A-Z][A-Z0-9_]{0,63}$", name):
        return False
    if name not in _OPTION_NAMES and not name.startswith(_OPTION_PREFIXES):
        return False
    return bool(_OPTION_VALUE.match(str(value)))


def job_options(job_env, base_env):
    """The per-job settings of ``job_env`` the worker needs: the allowed
    names whose value differs from this process's own environment."""
    out = {}
    for name, value in (job_env or {}).items():
        if not allowed_option(name, value):
            continue
        if (base_env or {}).get(name) == value:
            continue
        out[name] = str(value)
    return out


def worker_args(cmd):
    """``(source, rest)`` from a job's ``python -u main.py ...`` command: the
    ``-u``/``-i`` value and the arguments the worker repeats after its own
    ``-i <source> -o <dir>``. None when the command cannot run remotely."""
    cmd = [str(a) for a in (cmd or [])]
    start = next((i + 1 for i, a in enumerate(cmd) if a.endswith("main.py")), None)
    if start is None:
        return None
    source, rest, i = None, [], start
    while i < len(cmd):
        a = cmd[i]
        if a in ("-u", "--url", "-i", "--input"):
            if i + 1 >= len(cmd):
                return None
            source = cmd[i + 1]
            i += 2
        elif a in ("-o", "--output"):
            i += 2
        elif a == "--keep-original":
            i += 1
        elif a == "--format" and i + 1 < len(cmd) and cmd[i + 1] in _FORMATS:
            rest += [a, cmd[i + 1]]
            i += 2
        else:
            # A precomputed transcript, a whole-video run, anything new: the
            # worker repeats only what is known to be safe to repeat.
            return None
    if not source:
        return None
    return source, rest


def safe_args(args):
    """The same closed list, checked on the worker's side."""
    args = [str(a) for a in (args or [])]
    if args == []:
        return []
    if len(args) == 2 and args[0] == "--format" and args[1] in _FORMATS:
        return args
    return None


def source_is_url(cmd):
    cmd = [str(a) for a in (cmd or [])]
    start = next((i + 1 for i, a in enumerate(cmd) if a.endswith("main.py")), 0)
    return any(a in ("-u", "--url") for a in cmd[start:])


def safe_relpath(name):
    """``name`` if it is a plain file name for a job directory, else None: one
    component (a render writes no subdirectory, and a directory left behind
    with a clip's name would break the next run), no dotfile, a known output
    extension."""
    if not isinstance(name, str) or not name or "/" in name or "\\" in name:
        return None
    if name in (".", "..") or name.startswith(".") or name != name.strip():
        return None
    if any(ord(c) < 32 or ord(c) == 127 for c in name):
        return None
    if len(name.encode("utf-8", "replace")) > _MAX_NAME_BYTES:
        return None
    if os.path.splitext(name)[1].lower() not in _UPLOAD_EXTENSIONS:
        return None
    return name


def file_limit(rel, max_bytes):
    """The size limit of ``rel`` by what it is."""
    ext = os.path.splitext(rel)[1].lower()
    if ext in _SMALL_EXTENSIONS:
        return min(max_bytes, _SMALL_FILE_BYTES)
    if ext != ".mp4":
        return min(max_bytes, _IMAGE_FILE_BYTES)
    return max_bytes


@contextlib.contextmanager
def upload_lock(job_dir):
    """One upload request per job at a time, across both containers of a
    deploy: the budget is counted from the disk, so two requests writing at
    once would each be granted the same room (and could append to the same
    file). The worker sends its files one after another anyway. Raises
    UploadError(429) when another request holds it."""
    fd = os.open(os.path.join(job_dir, LOCK_FILE), os.O_CREAT | os.O_RDWR, 0o600)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            raise UploadError(429, "another upload of this job is in progress")
        yield
    finally:
        os.close(fd)


def _inside(job_dir, rel):
    root = os.path.realpath(job_dir)
    path = os.path.realpath(os.path.join(root, rel))
    return path if path.startswith(root + os.sep) else None


# --- The claim --------------------------------------------------------------

def begin(job_dir, worker, source_only=False):
    """Claim the job for ``worker``: a fresh claim id, state ``preparing``.
    ``source_only``: the worker asked for the source alone (it has a card of
    its own), so this host prepares no transcript and no scenes for it."""
    clear(job_dir)
    claim = secrets.token_hex(16)
    write_json_atomic(os.path.join(job_dir, CLAIM_FILE), {
        "claim": claim, "worker": str(worker or "")[:64], "state": PREPARING,
        "source_only": source_only is True, "claimed_at": time.time()})
    touch_seen(job_dir)
    return claim


def source_only(job_dir):
    """Whether the worker that holds the job asked for the source alone."""
    return (read(job_dir) or {}).get("source_only") is True


def source_only_env(env):
    """The settings of this host's ``--prepare-only`` run for a worker that
    asked for the source alone: nothing is transcribed and no scene pass runs
    here, both would only keep this host's card and cores from its own jobs
    while the worker's sit idle. A transcript the download cache already holds
    for the source still travels: it costs this host nothing."""
    mode = str(env.get("PREPARE_TRANSCRIPT") or "").strip().lower()
    return {"PREPARE_TRANSCRIPT": "cache" if mode in ("1", "cache") else "",
            "PREPARE_SCENES": ""}


def read(job_dir):
    try:
        with open(os.path.join(job_dir, CLAIM_FILE)) as f:
            data = json.load(f)
        return data if isinstance(data, dict) else None
    except (OSError, ValueError):
        return None


def is_remote(job_dir):
    return os.path.exists(os.path.join(job_dir, CLAIM_FILE))


def valid(job_dir, claim):
    """The live claim of ``job_dir`` if ``claim`` is it, else None."""
    data = read(job_dir)
    if not data or not isinstance(claim, str) or not claim:
        return None
    if not hmac.compare_digest(str(data.get("claim") or ""), claim):
        return None
    return data


def set_ready(job_dir, source, rest, options):
    """The source is on disk: publish what the worker runs."""
    data = read(job_dir)
    if data is None:
        return False
    try:
        size = os.path.getsize(source)
    except OSError:
        return False
    transcript = transcript_path(job_dir)
    scenes = scenes_path(job_dir)
    data.update(state=READY, source=source, source_size=size,
                transcript_size=os.path.getsize(transcript) if transcript else 0,
                scenes_size=os.path.getsize(scenes) if scenes else 0,
                args=list(rest), env=dict(options), ready_at=time.time())
    write_json_atomic(os.path.join(job_dir, CLAIM_FILE), data)
    return True


def uploaded(job_dir):
    """The relative names the worker has put in the job directory."""
    try:
        with open(os.path.join(job_dir, FILES_FILE)) as f:
            data = json.load(f)
        return [n for n in data if isinstance(n, str)] if isinstance(data, list) else []
    except (OSError, ValueError):
        return []


def clear(job_dir, purge=False):
    """Drop the claim and every protocol file: the worker's next call is told
    the job is gone, and nothing half-uploaded stays behind. ``purge``: the
    job did not end well, so what the worker uploaded goes too and the run
    that follows on this host starts from this host's own files."""
    if purge:
        for rel in uploaded(job_dir):
            target = _inside(job_dir, rel) if safe_relpath(rel) else None
            if target and os.path.isfile(target) and not os.path.islink(target):
                try:
                    os.remove(target)
                except OSError:
                    pass
    try:
        names = os.listdir(job_dir)
    except OSError:
        return
    for name in names:
        if name.startswith(RESERVED_PREFIX):
            try:
                os.remove(os.path.join(job_dir, name))
            except OSError:
                pass


# --- What the worker sends --------------------------------------------------

def touch_seen(job_dir):
    path = os.path.join(job_dir, SEEN_FILE)
    try:
        with open(path, "a"):
            pass
        os.utime(path, None)
    except OSError:
        pass


def last_seen(job_dir):
    try:
        return os.path.getmtime(os.path.join(job_dir, SEEN_FILE))
    except OSError:
        return 0.0


MAX_LOG_LINE = 500               # also bounds the secret scrub of each line
MAX_LOG_LINES = 500              # per call
MAX_LOG_BYTES = 2 * 1024 * 1024  # per job: a render prints a few hundred lines
READ_LOG_CHUNK = 256 * 1024


def append_log(job_dir, lines):
    """Append the worker's lines, up to the job's log budget; past it they are
    dropped (the call still counts as a heartbeat)."""
    path = os.path.join(job_dir, LOG_FILE)
    try:
        room = MAX_LOG_BYTES - os.path.getsize(path)
    except OSError:
        room = MAX_LOG_BYTES
    clean = []
    for line in list(lines or [])[:MAX_LOG_LINES]:
        text = "".join(c if ord(c) >= 32 else " " for c in str(line)).strip()[:MAX_LOG_LINE]
        cost = len(text.encode("utf-8")) + 1
        if not text:
            continue
        if cost > room:
            break
        room -= cost
        clean.append(text)
    if clean:
        with open(path, "a", encoding="utf-8") as f:
            f.write("\n".join(clean) + "\n")
    touch_seen(job_dir)
    return len(clean)


def read_log(job_dir, offset):
    """``(complete lines after byte offset, new offset)``, a bounded chunk at
    a time."""
    try:
        with open(os.path.join(job_dir, LOG_FILE), "rb") as f:
            f.seek(offset)
            data = f.read(READ_LOG_CHUNK)
    except OSError:
        return [], offset
    end = data.rfind(b"\n")
    if end < 0:
        return [], offset
    lines = data[:end].decode("utf-8", "replace").split("\n")
    return [line for line in lines if line], offset + end + 1


def mark_done(job_dir, returncode, stopping=False):
    """``stopping``: the worker ended the job because it is being stopped or
    restarted, not because the job failed. Such a job runs again and does not
    count against the worker (``app._remote_outcome``)."""
    write_json_atomic(os.path.join(job_dir, DONE_FILE),
                      {"returncode": int(returncode), "stopping": bool(stopping)})
    touch_seen(job_dir)


def read_stopping(job_dir) -> bool:
    try:
        with open(os.path.join(job_dir, DONE_FILE)) as f:
            return json.load(f).get("stopping") is True
    except (OSError, ValueError, AttributeError, TypeError):
        return False


def read_done(job_dir):
    try:
        with open(os.path.join(job_dir, DONE_FILE)) as f:
            return int(json.load(f)["returncode"])
    except (OSError, ValueError, KeyError, TypeError):
        return None


def _upload_tmp(job_dir, rel):
    import hashlib
    return os.path.join(job_dir, UPLOAD_PREFIX + hashlib.sha256(rel.encode()).hexdigest()[:32])


class UploadError(Exception):
    def __init__(self, status, detail):
        super().__init__(detail)
        self.status, self.detail = status, detail


def upload_offset(job_dir, rel):
    """Bytes of ``rel`` already received in an unfinished upload."""
    try:
        return os.path.getsize(_upload_tmp(job_dir, rel))
    except OSError:
        return 0


MAX_OPEN_UPLOADS = 4
MAX_FILES = 400


def _usage(job_dir, names):
    """``(bytes the worker holds in the job dir, unfinished uploads)``."""
    total = pending = 0
    for rel in names:
        try:
            total += os.path.getsize(os.path.join(job_dir, rel))
        except OSError:
            pass
    try:
        entries = os.listdir(job_dir)
    except OSError:
        entries = []
    for name in entries:
        if name.startswith(UPLOAD_PREFIX):
            pending += 1
            try:
                total += os.path.getsize(os.path.join(job_dir, name))
            except OSError:
                pass
    return total, pending


def open_upload(job_dir, name, offset, max_bytes, max_job_bytes=None):
    """Validate one chunk of an upload; returns ``(rel, tmp path, room)``:
    where to append and how many more bytes this job may still send. The
    chunk must start where the last one ended (0 starts over). A file that
    was in the job directory before the worker came (the source) is never
    replaced."""
    rel = safe_relpath(name)
    if rel is None:
        raise UploadError(400, "file name not allowed")
    target = _inside(job_dir, rel)
    if target is None:
        raise UploadError(400, "file name not allowed")
    names = uploaded(job_dir)
    if rel not in names and os.path.lexists(target):
        raise UploadError(400, "file name not allowed")
    if os.path.isdir(target) or os.path.islink(target):
        raise UploadError(400, "file name not allowed")
    tmp = _upload_tmp(job_dir, rel)
    have = upload_offset(job_dir, rel)
    if offset == 0 and have:
        os.remove(tmp)
        have = 0
    if offset != have:
        raise UploadError(409, f"expected offset {have}")
    max_bytes = file_limit(rel, max_bytes)
    if offset >= max_bytes:
        raise UploadError(413, "file too large")
    used, pending = _usage(job_dir, names)
    if offset == 0 and rel not in names and len(names) >= MAX_FILES:
        raise UploadError(413, "too many files")
    if offset == 0 and pending >= MAX_OPEN_UPLOADS:
        raise UploadError(429, "too many unfinished uploads")
    room = max_bytes - offset
    if max_job_bytes is not None:
        # Replacing a file it already sent frees that file's bytes.
        try:
            old = os.path.getsize(target) if rel in names else 0
        except OSError:
            old = 0
        room = min(room, max_job_bytes - (used - old))
    if room <= 0:
        raise UploadError(413, "job upload budget spent")
    return rel, tmp, room


def discard_upload(job_dir, name):
    rel = safe_relpath(name)
    if rel is not None:
        try:
            os.remove(_upload_tmp(job_dir, rel))
        except OSError:
            pass


def finish_upload(job_dir, rel, size, claim, max_bytes=None, max_job_bytes=None):
    """Move a complete upload into place (atomic: a clip is never served half
    written). ``size`` is what the worker says the file measures. The claim
    and the budget are checked again here, on what is on disk: a request held
    open must not land a file after the job went back to this host, nor one
    larger than it was granted."""
    tmp = _upload_tmp(job_dir, rel)
    have = upload_offset(job_dir, rel) if os.path.exists(tmp) else -1
    if have != int(size):
        raise UploadError(409, f"expected offset {max(have, 0)}")
    target = _inside(job_dir, rel)
    if target is None:
        raise UploadError(400, "file name not allowed")
    if os.path.isdir(target) or os.path.islink(target):
        raise UploadError(400, "file name not allowed")
    names = uploaded(job_dir)
    over = max_bytes is not None and have > file_limit(rel, max_bytes)
    if not over and max_job_bytes is not None:
        try:
            old = os.path.getsize(target) if rel in names else 0
        except OSError:
            old = 0
        over = _usage(job_dir, names)[0] - old > max_job_bytes
    if over:
        try:
            os.remove(tmp)
        except OSError:
            pass
        raise UploadError(413, "file too large")
    if rel.endswith("_metadata.json"):
        clean_metadata(tmp)
    if valid(job_dir, claim) is None:
        try:
            os.remove(tmp)
        except OSError:
            pass
        raise UploadError(410, "This job is no longer yours.")
    if rel not in names:
        # Listed before the file exists: a crash in between leaves a name
        # with no file, never a file nobody will clean up.
        write_json_atomic(os.path.join(job_dir, FILES_FILE), names + [rel])
    os.replace(tmp, target)
    touch_seen(job_dir)
    return target


# --- What the worker is given besides the source ------------------------------

# The transcript this host left for the job (main.py's own checkpoint file,
# written by ``--prepare-only``): the worker puts it at the same place in its
# job directory and its main.py reuses it like a resumed run does, instead of
# transcribing on a CPU what this host's GPU does in seconds. A dotfile, so
# the worker can read it and never write it back (``safe_relpath``).
TRANSCRIPT_FILE = ".transcript_checkpoint.json"
MAX_TRANSCRIPT_BYTES = _SMALL_FILE_BYTES
MAX_TRANSCRIPT_SEGMENTS = 100_000


def valid_transcript(payload) -> bool:
    """The checkpoint's shape, checked on both sides of the wire: the source
    it belongs to (a plain file name and a length) and a transcript with
    segments. Only shape: main.py ties it to the source it actually has."""
    if not isinstance(payload, dict):
        return False
    source, transcript = payload.get("source"), payload.get("transcript")
    if not isinstance(source, dict) or not isinstance(transcript, dict):
        return False
    name, duration = source.get("name"), source.get("duration")
    if (not isinstance(name, str) or not name or os.path.basename(name) != name
            or isinstance(duration, bool) or not isinstance(duration, (int, float))):
        return False
    segments = transcript.get("segments")
    return (isinstance(segments, list) and 0 < len(segments) <= MAX_TRANSCRIPT_SEGMENTS
            and all(isinstance(seg, dict) for seg in segments))


def transcript_path(job_dir):
    """The job's prepared transcript if there is a usable one, else None."""
    path = os.path.join(job_dir, TRANSCRIPT_FILE)
    try:
        if os.path.islink(path) or not os.path.isfile(path):
            return None
        if not 0 < os.path.getsize(path) <= MAX_TRANSCRIPT_BYTES:
            return None
    except OSError:
        return None
    return path


def read_transcript(job_dir):
    """The prepared transcript re-serialised as data, or None."""
    path = transcript_path(job_dir)
    if path is None:
        return None
    try:
        with open(path) as f:
            payload = json.load(f)
    except (OSError, ValueError, RecursionError):
        return None
    if not valid_transcript(payload):
        return None
    return json.dumps({"source": payload["source"], "transcript": payload["transcript"]})


def drop_transcript(job_dir):
    for name in (TRANSCRIPT_FILE, SCENES_FILE):
        try:
            os.remove(os.path.join(job_dir, name))
        except OSError:
            pass


# The per-frame scene probabilities this host's card computed for the whole
# source (main.prepare_scenes), served like the transcript: the worker maps
# them onto its clips instead of running TransNetV2 on its CPU
# (scene_detection.use_prepared, which checks them again on that side).
SCENES_FILE = ".scene_probs.json"
MAX_SCENES_FRAMES = 432_000
MAX_SCENES_BYTES = _SMALL_FILE_BYTES


def valid_scenes(payload) -> bool:
    """Shape only: a plain file name and a length for the source, a sane
    frame rate and frame count, base64 text of exactly that many float32s."""
    if not isinstance(payload, dict):
        return False
    source = payload.get("source")
    if not isinstance(source, dict):
        return False
    name, duration = source.get("name"), source.get("duration")
    fps, frames, probs = payload.get("fps"), payload.get("frames"), payload.get("probs")
    if not (isinstance(name, str) and name and os.path.basename(name) == name):
        return False
    if isinstance(duration, bool) or not isinstance(duration, (int, float)):
        return False
    if isinstance(fps, bool) or not isinstance(fps, (int, float)) or not 1 <= fps <= 240:
        return False
    if isinstance(frames, bool) or not isinstance(frames, int) or not 0 < frames <= MAX_SCENES_FRAMES:
        return False
    return (isinstance(probs, str) and len(probs) == 4 * ((4 * frames + 2) // 3)
            and re.fullmatch(r"[A-Za-z0-9+/]*={0,2}", probs) is not None)


def scenes_path(job_dir):
    """The job's prepared scene probabilities if there is a usable file."""
    path = os.path.join(job_dir, SCENES_FILE)
    try:
        if os.path.islink(path) or not os.path.isfile(path):
            return None
        if not 0 < os.path.getsize(path) <= MAX_SCENES_BYTES:
            return None
    except OSError:
        return None
    return path


def read_scenes(job_dir):
    """The prepared scene probabilities re-serialised as data, or None."""
    path = scenes_path(job_dir)
    if path is None:
        return None
    try:
        with open(path) as f:
            payload = json.load(f)
    except (OSError, ValueError, RecursionError):
        return None
    if not valid_scenes(payload):
        return None
    return json.dumps({k: payload[k] for k in ("source", "fps", "frames", "probs")})


MAX_CLIPS = 60
MAX_SEGMENTS = 200
MAX_TEXT = 5000


def clean_metadata(path):
    """Rewrite a worker's ``*_metadata.json`` as the data it is. The API fills
    in where each clip is served from, so a ``video_url`` that came from
    another machine is dropped (it could point at another job or another
    site); a source name is a plain file name; lists have a sane length and
    text fields are text. Anything that is not the expected shape makes the
    upload fail, like a file that never arrived."""
    try:
        with open(path) as f:
            data = json.load(f)
    except (OSError, ValueError, RecursionError):
        raise UploadError(400, "metadata is not valid JSON")
    if not isinstance(data, dict):
        raise UploadError(400, "metadata is not an object")
    shorts = data.get("shorts")
    if shorts is not None:
        if not isinstance(shorts, list) or len(shorts) > MAX_CLIPS:
            raise UploadError(400, "metadata has no usable clip list")
        clean = []
        for clip in shorts:
            if not isinstance(clip, dict):
                raise UploadError(400, "metadata has no usable clip list")
            clip.pop("video_url", None)
            for key, value in list(clip.items()):
                if isinstance(value, str) and len(value) > MAX_TEXT:
                    clip[key] = value[:MAX_TEXT]
                elif key in ("title", "video_title_for_youtube_short", "description",
                             "video_description_for_tiktok", "video_description_for_instagram",
                             "hook", "why") and value is not None and not isinstance(value, str):
                    clip[key] = str(value)[:MAX_TEXT]
            recipe = clip.get("recipe")
            if isinstance(recipe, dict):
                segments = recipe.get("segments")
                if segments is not None and (not isinstance(segments, list)
                                             or len(segments) > MAX_SEGMENTS):
                    recipe.pop("segments", None)
            elif recipe is not None:
                clip.pop("recipe", None)
            clean.append(clip)
        data["shorts"] = clean
    source = data.get("source_video")
    if source is not None:
        base = os.path.basename(str(source))
        if not base or base.startswith("."):
            data.pop("source_video", None)
    write_json_atomic(path, data)
