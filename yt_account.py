"""One YouTube account, many processes: pace it and back off when it is limited.

Every probe and every download that sends ``YOUTUBE_COOKIES`` spends the
request budget of the same account. YouTube answers a busy account with
"Your account has been rate-limited by YouTube for up to an hour", and from
then on every authenticated request fails AND counts against it: the probes
and the download each walk every route with the cookies, so one submission
could hit the limited account eight times, and the limit never lifted.

Three things, shared by every process on the host through files in the
output directory (the job processes, the API, both containers of a deploy):

* ``request()``: at most ``YT_ACCOUNT_SLOTS`` authenticated extractions at a
  time, started at least ``YT_ACCOUNT_MIN_INTERVAL`` seconds apart.
* ``note_rate_limited()`` / ``note_ok()`` / ``cooling_down()``: YouTube limits
  an account by degrees. A stray "rate-limited" answer among requests that
  work is routine (the caller just stops using the cookies for that one
  video). ``YT_ACCOUNT_STRIKES`` of them in a row within
  ``YT_ACCOUNT_STRIKE_WINDOW`` seconds, with no authenticated request
  succeeding in between, is the account shut: with
  ``YT_ACCOUNT_COOLDOWN_SECONDS`` set (it is 0, off, by default) nobody sends
  the cookies again for that long, and callers run only their anonymous
  attempts meanwhile.
* ``is_rate_limit()``: the error text, so a job that failed only because of
  this is retried after the cooldown instead of reported as a broken video.

Fail-open everywhere: a state directory that cannot be written, a lock that
never frees, a clock that jumped: the request goes ahead as it always did.
"""
import contextlib
import fcntl
import json
import os
import tempfile
import time


def _float_env(name, default):
    try:
        return max(0.0, float(os.environ.get(name, str(default)).strip()))
    except (TypeError, ValueError):
        return float(default)


def _int_env(name, default):
    try:
        return max(0, int(os.environ.get(name, str(default)).strip()))
    except (TypeError, ValueError):
        return int(default)


# Off by default. A cooldown only helps where a download can succeed without
# the account; in production the anonymous attempts are bot-checked on every
# IP, so "nobody sends the cookies" meant "nothing downloads": on 12-nov-2026
# a 10-minute cooldown took a half-limited account (about half of the
# downloads still working) to one success in a hundred. Set it only with a
# route that works anonymously.
COOLDOWN_SECONDS = _float_env("YT_ACCOUNT_COOLDOWN_SECONDS", 0)
STRIKES = _int_env("YT_ACCOUNT_STRIKES", 6)
STRIKE_WINDOW = _float_env("YT_ACCOUNT_STRIKE_WINDOW", 180)
MIN_INTERVAL = _float_env("YT_ACCOUNT_MIN_INTERVAL", 2.0)
SLOTS = _int_env("YT_ACCOUNT_SLOTS", 2)
# How long a request waits for its turn before going ahead anyway.
MAX_WAIT = _float_env("YT_ACCOUNT_MAX_WAIT", 90)

_COOLDOWN_FILE = ".yt-account.cooldown"
_STRIKES_FILE = ".yt-account.strikes"
_PACE_FILE = ".yt-account.pace"
_SLOT_FILE = ".yt-account-{}.lock"
_MARKERS = ("rate-limited by youtube", "rate limited by youtube")


def is_rate_limit(text) -> bool:
    """Whether an error text is YouTube limiting the account (not the video)."""
    t = str(text or "").lower()
    return any(m in t for m in _MARKERS)


def state_dir():
    """Where the shared state lives: the output directory every process of
    the host sees, or the temp dir when there is none (a one-off script)."""
    d = os.environ.get("YT_ACCOUNT_STATE_DIR", "").strip() or "output"
    return d if os.path.isdir(d) else tempfile.gettempdir()


def cooling_down(now=None) -> float:
    """Seconds left before the account may be used again (0: use it)."""
    if COOLDOWN_SECONDS <= 0:
        return 0.0            # switched off: a stamp left on disk means nothing
    now = time.time() if now is None else now
    try:
        with open(os.path.join(state_dir(), _COOLDOWN_FILE)) as f:
            until = float(json.load(f)["until"])
    except (OSError, ValueError, KeyError, TypeError):
        return 0.0
    left = until - now
    # A stamp further ahead than a cooldown can be is a clock that jumped.
    return left if 0 < left <= COOLDOWN_SECONDS + 60 else 0.0


def _strikes(fd):
    try:
        os.lseek(fd, 0, os.SEEK_SET)
        data = json.loads(os.read(fd, 256).decode() or "{}")
        return int(data.get("n") or 0), float(data.get("first") or 0)
    except (OSError, ValueError, TypeError, AttributeError):
        return 0, 0.0


def _write_strikes(fd, n, first):
    try:
        os.lseek(fd, 0, os.SEEK_SET)
        os.ftruncate(fd, 0)
        os.write(fd, json.dumps({"n": n, "first": first}).encode())
    except OSError:
        pass


def note_rate_limited(now=None) -> float:
    """YouTube answered an authenticated request with "rate-limited". Counts a
    strike; enough of them close together start the cooldown. Returns the
    seconds the account is now off limits (0: still usable)."""
    if COOLDOWN_SECONDS <= 0:
        return 0.0
    now = time.time() if now is None else now
    d = state_dir()
    try:
        fd = os.open(os.path.join(d, _STRIKES_FILE), os.O_CREAT | os.O_RDWR, 0o600)
    except OSError:
        return 0.0
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        n, first = _strikes(fd)
        if not n or now - first > STRIKE_WINDOW or now < first:
            n, first = 0, now
        n += 1
        if n < max(1, STRIKES):
            _write_strikes(fd, n, first)
            return 0.0
        _write_strikes(fd, 0, 0.0)
        path = os.path.join(d, _COOLDOWN_FILE)
        tmp = f"{path}.{os.getpid()}.tmp"
        with open(tmp, "w") as f:
            json.dump({"until": now + COOLDOWN_SECONDS, "at": now}, f)
        os.replace(tmp, path)
        return COOLDOWN_SECONDS
    except OSError:
        return 0.0
    finally:
        os.close(fd)


def note_ok():
    """An authenticated request worked: the account is not shut, whatever a
    few requests before it were told."""
    try:
        path = os.path.join(state_dir(), _STRIKES_FILE)
        if os.path.getsize(path) > 0:
            fd = os.open(path, os.O_RDWR)
            try:
                fcntl.flock(fd, fcntl.LOCK_EX)
                if _strikes(fd)[0]:
                    _write_strikes(fd, 0, 0.0)
            finally:
                os.close(fd)
    except OSError:
        pass


def _take_slot(deadline):
    """An open, locked slot file descriptor, or None (no slots configured, or
    none freed in time: go ahead without one)."""
    if SLOTS <= 0:
        return None
    d = state_dir()
    while True:
        for i in range(SLOTS):
            try:
                fd = os.open(os.path.join(d, _SLOT_FILE.format(i)),
                             os.O_CREAT | os.O_RDWR, 0o600)
            except OSError:
                return None
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                return fd
            except OSError:
                os.close(fd)
        if time.time() >= deadline:
            return None
        time.sleep(0.25)


def _pace(deadline):
    """Wait until MIN_INTERVAL after the last authenticated request started,
    then stamp this one. One process at a time reads and writes the stamp."""
    if MIN_INTERVAL <= 0:
        return
    try:
        fd = os.open(os.path.join(state_dir(), _PACE_FILE), os.O_CREAT | os.O_RDWR, 0o600)
    except OSError:
        return
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        try:
            last = float(os.read(fd, 64).decode().strip() or 0)
        except ValueError:
            last = 0.0
        now = time.time()
        wait = last + MIN_INTERVAL - now
        if 0 < wait <= MIN_INTERVAL and now + wait <= deadline:
            time.sleep(wait)
        os.lseek(fd, 0, os.SEEK_SET)
        os.ftruncate(fd, 0)
        os.write(fd, repr(time.time()).encode())
    except OSError:
        pass
    finally:
        os.close(fd)


@contextlib.contextmanager
def request():
    """Hold one of the account's slots for one authenticated extraction, and
    start it no sooner than MIN_INTERVAL after the previous one."""
    deadline = time.time() + MAX_WAIT
    fd = _take_slot(deadline)
    try:
        _pace(deadline)
        yield
    finally:
        if fd is not None:
            os.close(fd)
