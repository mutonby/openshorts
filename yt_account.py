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

Several accounts (``YOUTUBE_COOKIES``, ``YOUTUBE_COOKIES_2``, ...) share the
load: each video takes the next account in turn (``Session``), every account
has its own slots and pacing, and an account that YouTube keeps answering
"rate-limited" rests for ``YT_ACCOUNT_REST_SECONDS`` while the others carry
on. A video whose account is limited moves to another one that is not resting
before it gives up. With every account resting the one closest to waking is
used: no cookies at all is worse than a limited account.

Fail-open everywhere: a state directory that cannot be written, a lock that
never frees, a clock that jumped: the request goes ahead as it always did.
"""
import atexit
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
# How many "rate-limited" answers one video takes before its probe or its
# download stops sending the cookies. Not one: the limit is often on a single
# exit IP, and the next static route got the video (seen on every refresh of
# the cookies, and on 12-nov-2026 stopping at the first answer lost the videos
# the second and third route used to rescue). Not unbounded: when the account
# itself is shut every route answers the same and each one counts against it.
ANSWERS_PER_VIDEO = _int_env("YT_ACCOUNT_ANSWERS_PER_VIDEO", 2)
# How long a request waits for its turn before going ahead anyway.
MAX_WAIT = _float_env("YT_ACCOUNT_MAX_WAIT", 90)

# An account that got REST_STRIKES "rate-limited" answers in a row (no request
# of its own succeeding in between, within STRIKE_WINDOW) rests this long.
# Only ever applied with more than one account configured.
REST_SECONDS = _float_env("YT_ACCOUNT_REST_SECONDS", 900)
REST_STRIKES = _int_env("YT_ACCOUNT_REST_STRIKES", 3)
MAX_ACCOUNTS = 9

_COOLDOWN_FILE = ".yt-account.cooldown"
_REST_FILE = ".yt-account-rest-{}"
_TURN_FILE = ".yt-account.turn"
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


def _tag(account):
    """File name part of one account: nothing for the first, so a host with a
    single account keeps the files it always had."""
    return f"a{int(account)}-" if account else ""


def _take_slot(deadline, account=0):
    """An open, locked slot file descriptor, or None (no slots configured, or
    none freed in time: go ahead without one)."""
    if SLOTS <= 0:
        return None
    d = state_dir()
    while True:
        for i in range(SLOTS):
            try:
                fd = os.open(os.path.join(d, _SLOT_FILE.format(f"{_tag(account)}{i}")),
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


def _pace(deadline, account=0):
    """Wait until MIN_INTERVAL after the last authenticated request of this
    account started, then stamp this one. One process at a time reads and
    writes the stamp."""
    if MIN_INTERVAL <= 0:
        return
    try:
        fd = os.open(os.path.join(state_dir(), _PACE_FILE + (f".{_tag(account)[:-1]}" if account else "")),
                     os.O_CREAT | os.O_RDWR, 0o600)
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
def request(account=0):
    """Hold one of the account's slots for one authenticated extraction, and
    start it no sooner than MIN_INTERVAL after the account's previous one."""
    deadline = time.time() + MAX_WAIT
    fd = _take_slot(deadline, account)
    try:
        _pace(deadline, account)
        yield
    finally:
        if fd is not None:
            os.close(fd)


# --- several accounts ------------------------------------------------------

def accounts(env=None):
    """The cookie files configured, in order: YOUTUBE_COOKIES, then
    YOUTUBE_COOKIES_2 .. _9. Empty ones are skipped."""
    env = os.environ if env is None else env
    names = ["YOUTUBE_COOKIES"] + [f"YOUTUBE_COOKIES_{i}" for i in range(2, MAX_ACCOUNTS + 1)]
    out = []
    for name in names:
        text = str(env.get(name) or "").strip()
        if text:
            out.append(text + "\n")
    return out


def _rest_state(fd):
    try:
        os.lseek(fd, 0, os.SEEK_SET)
        data = json.loads(os.read(fd, 256).decode() or "{}")
        return (int(data.get("n") or 0), float(data.get("first") or 0),
                float(data.get("until") or 0))
    except (OSError, ValueError, TypeError, AttributeError):
        return 0, 0.0, 0.0


def _write_rest(fd, n, first, until):
    try:
        os.lseek(fd, 0, os.SEEK_SET)
        os.ftruncate(fd, 0)
        os.write(fd, json.dumps({"n": n, "first": first, "until": until}).encode())
    except OSError:
        pass


def resting(account, now=None) -> float:
    """Seconds this account still rests (0: use it)."""
    if REST_SECONDS <= 0 or account is None:
        return 0.0
    now = time.time() if now is None else now
    try:
        with open(os.path.join(state_dir(), _REST_FILE.format(int(account)))) as f:
            until = float(json.load(f).get("until") or 0)
    except (OSError, ValueError, TypeError, AttributeError):
        return 0.0
    left = until - now
    return left if 0 < left <= REST_SECONDS + 60 else 0.0


def note_account_limited(account, now=None) -> float:
    """One "rate-limited" answer for this account. Enough of them in a row
    put it to rest; returns the seconds it now rests (0: still in use)."""
    if REST_SECONDS <= 0 or account is None:
        return 0.0
    now = time.time() if now is None else now
    try:
        fd = os.open(os.path.join(state_dir(), _REST_FILE.format(int(account))),
                     os.O_CREAT | os.O_RDWR, 0o600)
    except OSError:
        return 0.0
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        n, first, until = _rest_state(fd)
        if 0 < until - now <= REST_SECONDS + 60:
            return until - now            # already resting: do not extend it
        if not n or now - first > STRIKE_WINDOW or now < first:
            n, first = 0, now
        n += 1
        if n < max(1, REST_STRIKES):
            _write_rest(fd, n, first, 0.0)
            return 0.0
        _write_rest(fd, 0, 0.0, now + REST_SECONDS)
        return REST_SECONDS
    except OSError:
        return 0.0
    finally:
        os.close(fd)


def note_account_ok(account):
    """A request with this account worked: forget its strikes."""
    if account is None:
        return
    try:
        path = os.path.join(state_dir(), _REST_FILE.format(int(account)))
        if os.path.getsize(path) > 0:
            fd = os.open(path, os.O_RDWR)
            try:
                fcntl.flock(fd, fcntl.LOCK_EX)
                n, _first, until = _rest_state(fd)
                if n:
                    _write_rest(fd, 0, 0.0, until)
            finally:
                os.close(fd)
    except OSError:
        pass


def _turn():
    """A counter shared by every process of the host, so consecutive videos
    take consecutive accounts."""
    try:
        fd = os.open(os.path.join(state_dir(), _TURN_FILE), os.O_CREAT | os.O_RDWR, 0o600)
    except OSError:
        return os.getpid()
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        try:
            n = int(os.read(fd, 32).decode().strip() or 0)
        except ValueError:
            n = 0
        os.lseek(fd, 0, os.SEEK_SET)
        os.ftruncate(fd, 0)
        os.write(fd, str((n + 1) % 1_000_000).encode())
        return n
    except OSError:
        return os.getpid()
    finally:
        os.close(fd)


def pick(exclude=(), awake_only=False, count=None, now=None):
    """Which account the next video uses: the next in turn among those not
    resting. With all of them resting, the one closest to waking (or None
    with ``awake_only``). None when there is no account left to pick."""
    count = len(accounts()) if count is None else count
    candidates = [i for i in range(count) if i not in exclude]
    if not candidates:
        return None
    left = {i: resting(i, now) for i in candidates}
    awake = [i for i in candidates if left[i] <= 0]
    if not awake:
        return None if awake_only else min(candidates, key=lambda i: left[i])
    return awake[_turn() % len(awake)] if len(awake) > 1 else awake[0]


class Session:
    """The account one video uses, as a cookie file yt-dlp can read, and the
    move to another account when YouTube limits this one.

    ``path`` is None without any account configured (self-host without
    cookies, the remote worker): callers then run anonymous, as before."""

    _live = {}      # cookie file path -> its Session, in this process

    @classmethod
    def of(cls, path):
        """The session that owns this cookie file; for a file somebody else
        wrote, one that only counts the answers (no account to move to)."""
        found = cls._live.get(path)
        if found is None:
            found = cls.__new__(cls)
            found._texts, found._tried = [], set()
            found.account, found.path, found.answers = None, path, 0
        return found

    def __init__(self):
        self._texts = accounts()
        self._tried = set()
        self.account = None
        self.path = None
        self.answers = 0
        self._use(pick(count=len(self._texts)))
        if self.path:
            Session._live[self.path] = self
            atexit.register(self.close)

    def _use(self, account):
        if account is None:
            return False
        try:
            if self.path is None:
                fd, self.path = tempfile.mkstemp(prefix="yt_ck_", suffix=".txt")
                os.close(fd)
            # Rewritten in place: the path is already inside the options of
            # the caller's next attempt.
            with open(self.path, "w") as f:
                f.write(self._texts[account])
            os.chmod(self.path, 0o600)
        except OSError:
            if self.account is None:
                self.path = None
            return False
        self.account = account
        self._tried.add(account)
        self.answers = 0
        return True

    @property
    def spent(self):
        """No more requests with the cookies for this video."""
        return self.path is not None and (
            self.answers >= ANSWERS_PER_VIDEO or cooling_down() > 0)

    def request(self):
        return request(self.account or 0)

    def ok(self):
        note_ok()
        note_account_ok(self.account)

    def limited(self) -> bool:
        """YouTube answered "rate-limited". True when the video moved to
        another account (its next attempt uses it)."""
        note_rate_limited()
        note_account_limited(self.account)
        if self._use(pick(exclude=self._tried, awake_only=True, count=len(self._texts))):
            return True
        self.answers += 1
        return False

    def close(self):
        path, self.path = self.path, None
        if path:
            Session._live.pop(path, None)
            try:
                os.remove(path)
            except OSError:
                pass
