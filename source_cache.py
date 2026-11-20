"""Downloaded YouTube sources kept for a while, shared by every job on the host.

Many users clip the same popular videos, often within hours of each other.
Every repeat used to spend a request of a YouTube account (``yt_account``),
and the accounts are what YouTube rate-limits, plus a GPU slot's worth of
download and transcription.

So a whole-source download of a YouTube video is kept here under its video id
and the next job that asks for that id takes the file instead of downloading:

* ``fetch(url, output_dir)``: the cached source hard-linked into the job's
  directory (one inode, no copy; a copy only across filesystems), or None.
* ``claim(url)``: an exclusive lock on the id while one job downloads it, so a
  burst of jobs on the same video (newest-first queues start them together)
  waits for the one download instead of all spending the account at once.
  Bounded by ``SOURCE_CACHE_WAIT_SECONDS``; past it the job downloads anyway.
* ``store(url, path, title)``: called after a download that fetched the whole
  source over a free route (never a ranged fetch, never the 720p per-GB
  proxy). Linked in, so the job's own cut (``cap_source_duration`` replaces
  the file, never writes into it) leaves the cached copy whole.
* ``fetch_transcript`` / ``store_transcript``: the transcript of that exact
  source (same duration, same ASR backend) rides along, so a repeat skips
  transcription too.
* ``probe_get`` / ``probe_put``: what the metering probe learned at submit
  (length, best height), so a repeated submission does not ask YouTube
  either. Kept even for a video never downloaded (a job still queued).

Entries live ``SOURCE_CACHE_HOURS`` (48) since they were stored, and the
whole cache stays under ``SOURCE_CACHE_MAX_GB`` (150), least recently used
out first. Off unless ``SOURCE_CACHE_DIR`` is set (Cloud sets it to a dot
directory of the output dir: never served by ``/videos``, never swept by the
job sweeps). Nothing here is personal data: public video files keyed by their
YouTube id, with no user or job attached.

Whatever is read back from the cache directory is data: the title becomes a
file name only through ``sanitize`` and a basename, the sizes are numbers.
Fail-open everywhere: any error means "not cached" and the job downloads as
it always did.
"""
import contextlib
import fcntl
import hashlib
import json
import os
import re
import shutil
import tempfile
import time
from urllib.parse import parse_qs, urlparse

_ID = re.compile(r"^[A-Za-z0-9_-]{11}$")
_HOSTS = {"youtube.com", "www.youtube.com", "m.youtube.com", "music.youtube.com",
          "youtube-nocookie.com", "www.youtube-nocookie.com"}
_PATH_PREFIXES = ("shorts", "live", "embed", "v", "e")


def _float_env(name, default):
    try:
        return max(0.0, float(os.environ.get(name, str(default)).strip()))
    except (TypeError, ValueError):
        return float(default)


def cache_dir():
    """The cache directory, or "" when the cache is off."""
    return os.environ.get("SOURCE_CACHE_DIR", "").strip()


def ttl_seconds():
    return _float_env("SOURCE_CACHE_HOURS", 48) * 3600


def max_bytes():
    return int(_float_env("SOURCE_CACHE_MAX_GB", 150) * 1e9)


def max_entry_bytes():
    """The largest source worth keeping (``SOURCE_CACHE_MAX_ENTRY_GB``, 4; 0
    for no limit). One many-hour stream is tens of GB: kept, it pushes dozens
    of ordinary sources out of the cache to save a download nobody repeats."""
    return int(_float_env("SOURCE_CACHE_MAX_ENTRY_GB", 4) * 1e9)


def wait_seconds():
    return _float_env("SOURCE_CACHE_WAIT_SECONDS", 900)


def enabled():
    return bool(cache_dir()) and ttl_seconds() > 0 and max_bytes() > 0


def video_id(url):
    """The 11-character YouTube id of a single-video URL, or None.

    Only the shapes that name one video; clip links (``/clip/``) are a
    stretch of a video with an id of their own and are not cached."""
    try:
        u = urlparse(str(url or "").strip())
    except ValueError:
        return None
    host = (u.hostname or "").lower()
    parts = [p for p in (u.path or "").split("/") if p]
    vid = None
    if host == "youtu.be":
        vid = parts[0] if parts else None
    elif host in _HOSTS:
        if parts[:1] == ["watch"]:
            vid = (parse_qs(u.query).get("v") or [None])[0]
        elif len(parts) >= 2 and parts[0] in _PATH_PREFIXES:
            vid = parts[1]
    return vid if vid and _ID.match(vid) else None


def sanitize(title):
    """A title read back from the cache, made safe to use as a file name."""
    name = os.path.basename(str(title or "")).replace("\x00", "")
    name = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "", name).strip().lstrip(".")
    return name[:150] or "youtube_video"


def _paths(vid):
    d = cache_dir()
    return (os.path.join(d, f"{vid}.mp4"), os.path.join(d, f"{vid}.json"),
            os.path.join(d, f"{vid}.lock"), os.path.join(d, f"{vid}.transcript.json"))


def _read_json(path):
    try:
        with open(path) as f:
            data = json.load(f)
        return data if isinstance(data, dict) else None
    except (OSError, ValueError):
        return None


def _write_json(path, data):
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path), prefix=".tmp-")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(data, f)
        os.replace(tmp, path)
    except Exception:
        with contextlib.suppress(OSError):
            os.remove(tmp)
        raise


def _drop(vid):
    for p in _paths(vid):
        if not p.endswith(".lock"):
            with contextlib.suppress(OSError):
                os.remove(p)


def _entry(vid, now=None):
    """(media path, meta) of a fresh, intact entry, or None."""
    media, meta_path, _, _ = _paths(vid)
    meta = _read_json(meta_path)
    if not meta:
        return None
    now = time.time() if now is None else now
    try:
        stored = float(meta.get("stored_at"))
        size = int(meta.get("size"))
    except (TypeError, ValueError):
        _drop(vid)
        return None
    if not (0 <= now - stored <= ttl_seconds()):
        _drop(vid)
        return None
    try:
        if os.path.getsize(media) != size or size <= 0:
            _drop(vid)
            return None
    except OSError:
        _drop(vid)
        return None
    return media, meta


def _link_or_copy(src, dst):
    with contextlib.suppress(FileNotFoundError):
        os.remove(dst)
    try:
        os.link(src, dst)
    except OSError:
        shutil.copyfile(src, dst)


def fetch(url, output_dir):
    """(path in ``output_dir``, title) of a cached copy of ``url``, or None."""
    if not enabled():
        return None
    vid = video_id(url)
    if not vid:
        return None
    try:
        hit = _entry(vid)
        if hit is None:
            return None
        media, meta = hit
        title = sanitize(meta.get("title"))
        dst = os.path.join(output_dir, f"{title}.mp4")
        _link_or_copy(media, dst)
        # Least recently used goes first: a hit counts as a use.
        with contextlib.suppress(OSError):
            os.utime(_paths(vid)[1])
        print(f"♻️ Source reused from the download cache ({vid}): no YouTube request.")
        return dst, title
    except Exception as e:
        print(f"⚠️ Download cache skipped ({type(e).__name__}: {e}).")
        return None


@contextlib.contextmanager
def claim(url):
    """Hold the id's lock while downloading it (no-op when off or not YouTube).

    Waits up to ``wait_seconds()`` for another job's download of the same
    video, then goes ahead without the lock: a stuck download elsewhere must
    not stall this job for longer than the wait."""
    vid = video_id(url) if enabled() else None
    fd = None
    if vid:
        try:
            os.makedirs(cache_dir(), exist_ok=True)
            fd = os.open(_paths(vid)[2], os.O_RDWR | os.O_CREAT, 0o600)
            deadline = time.monotonic() + wait_seconds()
            told = False
            while True:
                try:
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    if time.monotonic() >= deadline:
                        print("⏳ Another job is still downloading this video: "
                              "downloading it here too.")
                        os.close(fd)
                        fd = None
                        break
                    if not told:
                        print("⏳ Another job is downloading this same video: waiting for it.")
                        told = True
                    time.sleep(2)
        except Exception as e:
            print(f"⚠️ Download cache lock skipped ({type(e).__name__}: {e}).")
            if fd is not None:
                with contextlib.suppress(OSError):
                    os.close(fd)
            fd = None
    try:
        yield
    finally:
        if fd is not None:
            with contextlib.suppress(OSError):
                fcntl.flock(fd, fcntl.LOCK_UN)
            with contextlib.suppress(OSError):
                os.close(fd)


def store(url, path, title):
    """Keep ``path`` (a whole, freshly downloaded source) for the next job."""
    if not enabled():
        return False
    vid = video_id(url)
    if not vid:
        return False
    try:
        size = os.path.getsize(path)
        if size <= 0 or size > max_bytes():
            return False
        if 0 < max_entry_bytes() < size:
            return False
        d = cache_dir()
        os.makedirs(d, exist_ok=True)
        media, meta_path, _, transcript = _paths(vid)
        fd, tmp = tempfile.mkstemp(dir=d, prefix=".tmp-", suffix=".mp4")
        os.close(fd)
        _link_or_copy(path, tmp)
        os.replace(tmp, media)
        with contextlib.suppress(OSError):
            os.remove(transcript)      # it described the file just replaced
        _write_json(meta_path, {"title": sanitize(title), "size": size,
                                "stored_at": time.time()})
        print(f"📦 Source kept in the download cache ({vid}).")
        prune()
        return True
    except Exception as e:
        print(f"⚠️ Could not keep the source in the download cache ({type(e).__name__}: {e}).")
        return False


def _backend():
    return os.environ.get("TRANSCRIBE_BACKEND", "").strip().lower() or "default"


def fetch_transcript(url, duration):
    """The cached transcript of this exact source, or None.

    Only for the same length (a job that cut its source is not that file
    any more) and the same ASR backend."""
    if not enabled():
        return None
    vid = video_id(url)
    if not vid or _entry(vid) is None:
        return None
    data = _read_json(_paths(vid)[3])
    try:
        if (not data or data.get("backend") != _backend()
                or abs(float(data.get("duration")) - float(duration)) > 0.5):
            return None
        t = data.get("transcript")
        if not isinstance(t, dict) or not isinstance(t.get("segments"), list) or not t["segments"]:
            return None
        print(f"♻️ Transcript reused from the download cache ({vid}).")
        return t
    except (TypeError, ValueError):
        return None


def store_transcript(url, duration, transcript, source_path):
    """Keep the transcript next to the cached source, if ``source_path`` is it."""
    if not enabled() or not isinstance(transcript, dict) or not transcript.get("segments"):
        return False
    vid = video_id(url)
    if not vid:
        return False
    try:
        hit = _entry(vid)
        if hit is None:
            return False
        media, _ = hit
        # The very file this job transcribed, uncut: the same inode, or (a
        # copy across filesystems) the same size.
        a, b = os.stat(media), os.stat(source_path)
        same = (a.st_ino == b.st_ino and a.st_dev == b.st_dev) or a.st_size == b.st_size
        if not same:
            return False
        _write_json(_paths(vid)[3], {"backend": _backend(), "duration": float(duration),
                                     "transcript": transcript})
        return True
    except Exception as e:
        print(f"⚠️ Could not keep the transcript in the download cache ({type(e).__name__}: {e}).")
        return False


# --- The extraction ---------------------------------------------------------
# A probe and the download that follows it each asked YouTube for the same
# thing: the video's formats and their media links. With the account's cookies
# both spend its request budget, and that budget is what every download waits
# for. The probe's answer is kept here until its links expire (about six
# hours), so the download uses it instead of asking again.

# Not used this close to the links' expiry: the download has to finish on
# them. A source fetched in fragments does not fail when they die half-way, it
# comes out short, and a short source would then be kept for every later job.
INFO_MARGIN_SECONDS = 2 * 3600
INFO_MAX_BYTES = 8 * 1024 * 1024
INFO_FILE_SECONDS = 8 * 3600     # the file itself, whatever it says inside
_EXPIRE = re.compile(r"[?&/]expire[=/](\d{9,11})")


def info_enabled():
    return enabled() and os.environ.get("SOURCE_CACHE_INFO", "1").strip() != "0"


def _info_path(vid):
    return os.path.join(cache_dir(), f"{vid}.info.json")


def route_key(proxy):
    """Which exit a request left through, as a name that can be written to
    disk: never the proxy URL, it carries the proxy's credentials. Media links
    are bound to the address that asked for them, so an extraction is only
    used again through the same exit."""
    if not proxy:
        return "direct"
    return "p-" + hashlib.sha256(str(proxy).encode()).hexdigest()[:16]


def links_expire(info):
    """When the first of the extraction's media links stops working (epoch
    seconds), or None when none of them says."""
    soonest = None
    try:
        for f in info.get("formats") or []:
            for u in (f.get("url"), f.get("manifest_url")):
                m = _EXPIRE.search(u) if isinstance(u, str) else None
                if m:
                    t = int(m.group(1))
                    soonest = t if soonest is None else min(soonest, t)
    except (AttributeError, TypeError, ValueError):
        return None
    return soonest


def info_put(url, info, proxy, authed, extractor_args):
    """Keep what YouTube answered a probe for the download that follows it
    (best effort). ``info`` is yt-dlp's own sanitized result."""
    if not info_enabled():
        return False
    vid = video_id(url)
    expires = links_expire(info) if isinstance(info, dict) else None
    if not vid or not expires or expires - time.time() <= INFO_MARGIN_SECONDS:
        return False
    try:
        data = {"at": time.time(), "expires": expires, "route": route_key(proxy),
                "authed": bool(authed), "args": extractor_args, "info": info}
        if len(json.dumps(data)) > INFO_MAX_BYTES:
            return False
        os.makedirs(cache_dir(), exist_ok=True)
        _write_json(_info_path(vid), data)
        return True
    except Exception:
        return False


def info_get(url, now=None, need_seconds=0):
    """The stored extraction of this video while its links are good:
    ``{"info", "route", "authed", "args"}``, or None. ``need_seconds``: extra
    life the caller's download needs (a ranged fetch runs at about the speed
    of the video, so it needs as long as what it fetches)."""
    if not info_enabled():
        return None
    vid = video_id(url)
    if not vid:
        return None
    data = _read_json(_info_path(vid))
    now = time.time() if now is None else now
    try:
        if (not data or not isinstance(data.get("info"), dict)
                or not isinstance(data.get("route"), str)
                or float(data["expires"]) - now
                <= INFO_MARGIN_SECONDS + max(0.0, float(need_seconds or 0))):
            return None
        info = data["info"]
        if info.get("_type", "video") != "video" or not info.get("formats"):
            return None
        return {"info": info, "route": data["route"], "authed": bool(data.get("authed")),
                "args": data.get("args")}
    except (KeyError, TypeError, ValueError):
        return None


def info_drop(url):
    """Forget the stored extraction (its links did not work)."""
    vid = video_id(url)
    if vid and cache_dir():
        with contextlib.suppress(OSError):
            os.remove(_info_path(vid))


def _probe_path(vid):
    return os.path.join(cache_dir(), f"{vid}.probe.json")


def probe_get(url):
    """``{"seconds", "max_height"}`` from a probe of this video in the last
    ``SOURCE_CACHE_HOURS``, or None."""
    if not enabled():
        return None
    vid = video_id(url)
    if not vid:
        return None
    data = _read_json(_probe_path(vid))
    try:
        if not data or not (0 <= time.time() - float(data["at"]) <= ttl_seconds()):
            return None
        seconds = float(data["seconds"])
        if not 0 < seconds < 7 * 86400:
            return None
        height = data.get("max_height")
        height = int(height) if isinstance(height, (int, float)) and height > 0 else None
    except (KeyError, TypeError, ValueError):
        return None
    print(f"♻️ Video length known from the download cache ({vid}): no YouTube request.")
    return {"seconds": seconds, "max_height": height}


def probe_put(url, seconds, max_height=None):
    """Remember what a probe learned about this video (best effort)."""
    if not enabled():
        return False
    vid = video_id(url)
    try:
        seconds = float(seconds)
    except (TypeError, ValueError):
        return False
    if not vid or not 0 < seconds < 7 * 86400:
        return False
    try:
        os.makedirs(cache_dir(), exist_ok=True)
        _write_json(_probe_path(vid), {"seconds": seconds, "at": time.time(),
                                       "max_height": int(max_height) if max_height else None})
        return True
    except Exception:
        return False


def prune(now=None):
    """Drop expired entries and the ones over the entry size, then the least
    recently used past the size cap."""
    d = cache_dir()
    if not d:
        return
    now = time.time() if now is None else now
    try:
        names = os.listdir(d)
    except OSError:
        return
    entries = []
    for name in names:
        full = os.path.join(d, name)
        if name.startswith(".tmp-"):
            # A store that died half-way; a live one is seconds old.
            with contextlib.suppress(OSError):
                if now - os.path.getmtime(full) > 3600:
                    os.remove(full)
            continue
        if name.endswith(".probe.json"):
            with contextlib.suppress(OSError):
                if now - os.path.getmtime(full) > max(3600.0, ttl_seconds()):
                    os.remove(full)
            continue
        if name.endswith(".info.json"):
            with contextlib.suppress(OSError):
                if now - os.path.getmtime(full) > INFO_FILE_SECONDS:
                    os.remove(full)
            continue
        if not name.endswith(".json") or name.endswith(".transcript.json"):
            continue
        vid = name[:-5]
        if not _ID.match(vid):
            continue
        hit = _entry(vid, now)
        if hit is None:
            continue
        try:
            used = os.path.getmtime(full)
        except OSError:
            used = 0.0
        size = int(hit[1]["size"])
        if 0 < max_entry_bytes() < size:
            _drop(vid)           # kept before the limit, or under a larger one
            continue
        entries.append((used, vid, size))
    # Media files with no entry (a meta write that failed) are garbage too,
    # and so are the lock files of videos nobody asked for in a whole TTL.
    known = {vid for _, vid, _ in entries}
    for name in names:
        stem, ext = os.path.splitext(name)
        if ext not in (".mp4", ".lock") or name.startswith(".") or stem in known:
            continue
        if not _ID.match(stem) or _read_json(_paths(stem)[1]) is not None:
            continue
        full = os.path.join(d, name)
        age = 3600 if ext == ".mp4" else max(3600.0, ttl_seconds())
        with contextlib.suppress(OSError):
            if now - os.path.getmtime(full) > age:
                os.remove(full)
    total = sum(size for _, _, size in entries)
    cap = max_bytes()
    for used, vid, size in sorted(entries):
        if total <= cap:
            break
        _drop(vid)
        total -= size


def is_whole_free_download(route):
    """Whether a finished download (its ``PROXY_ROUTE`` summary) is the whole
    source at full quality: not a ranged fetch (only the paid minutes) and
    not won on the per-GB proxy (capped at 720p to save its bandwidth)."""
    if not isinstance(route, dict) or route.get("ranged") or not route.get("winner"):
        return False
    won = [a for a in (route.get("attempts") or []) if isinstance(a, dict) and a.get("ok")]
    return bool(won) and not won[-1].get("paid")
