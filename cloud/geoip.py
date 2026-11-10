"""Country of a client IP, from a local MMDB file (Cloud, free queue lanes).

The free queue's slow lane (``cloud/queue_mode.py``) needs the country of the
client that submits. The API is not behind a CDN that would geolocate it, so
the country comes from the connection's own address (``request.client.host``,
which uvicorn takes from the reverse proxy we run) looked up in an MMDB file
loaded into memory: microseconds per lookup, no network per request.

- **Data.** ``GEOIP_COUNTRY_DB_URL`` (default: the ``user-country`` database of
  the ip-location-db project, PDDL 1.0, public domain: no attribution, no
  account, no licence key; rebuilt daily, ~8 MB, IPv4 + IPv6). Any MMDB with a
  ``country_code`` or GeoLite2-style ``country.iso_code`` field works.
- **Where.** ``GEOIP_COUNTRY_DB_PATH``, by default ``.geoip/country.mmdb`` in
  the shared output dir, so both containers of a deploy read one copy and a
  deploy does not download it again.
- **Refresh.** Only while something needs it (``wanted()``: a slow-lane country
  is configured). A background loop downloads it when it is missing or older
  than ``GEOIP_REFRESH_HOURS`` (72): to a temp file, checked against the
  published SHA-256 (``GEOIP_COUNTRY_DB_SHA256_URL``) and opened before it
  replaces the old one (``os.replace``), under a non-blocking flock so only one
  container downloads. Each instance reloads the file when its mtime changes.
- **Never an error.** No file, a corrupt or unreadable one, no ``maxminddb``
  installed, an address that is not public (private, loopback, link-local,
  reserved) or not found: the country is unknown, and unknown means the
  normal lane.
"""
import asyncio
import hashlib
import ipaddress
import os
import re
import tempfile
import time

from .config import _flag

DEFAULT_URL = "https://github.com/sapics/ip-location-db/releases/download/latest/user-country.mmdb"
DEFAULT_SHA256_URL = ("https://github.com/sapics/ip-location-db/releases/download/"
                      "checksum/user-country.mmdb.sha256")
CHECK_EVERY = 3600          # seconds between looks at the file's age / mtime
RETRY_SOON = 60             # ...while this instance has no database loaded
DOWNLOAD_TIMEOUT = 180      # whole download, not per read
MAX_BYTES = 64 * 1024 * 1024
_CODE = re.compile(r"^[A-Z]{2}$")

_reader = None              # maxminddb reader, in memory
_loaded_mtime = None
_state_dir = None
_task = None


def _env_float(name, default):
    try:
        return float(os.environ.get(name, "") or default)
    except ValueError:
        return default


def db_path() -> str:
    explicit = os.environ.get("GEOIP_COUNTRY_DB_PATH", "").strip()
    if explicit:
        return explicit
    base = _state_dir or os.environ.get("OUTPUT_DIR", "output")
    return os.path.join(base, ".geoip", "country.mmdb")


def db_url() -> str:
    raw = os.environ.get("GEOIP_COUNTRY_DB_URL")
    return DEFAULT_URL if raw is None else raw.strip()


def sha256_url() -> str:
    raw = os.environ.get("GEOIP_COUNTRY_DB_SHA256_URL")
    if raw is not None:
        return raw.strip()
    return DEFAULT_SHA256_URL if db_url() == DEFAULT_URL else ""


def refresh_seconds() -> float:
    return max(1.0, _env_float("GEOIP_REFRESH_HOURS", 72)) * 3600


# --- Lookup ---------------------------------------------------------------------

def _public(ip: str):
    try:
        addr = ipaddress.ip_address(str(ip).strip())
    except ValueError:
        return None
    mapped = getattr(addr, "ipv4_mapped", None)
    if mapped is not None:
        addr = mapped
    return str(addr) if addr.is_global else None


def country_of(ip) -> str:
    """ISO code of ``ip``'s country, or "" when it cannot be told."""
    if _reader is None or not ip:
        return ""
    addr = _public(ip)
    if addr is None:
        return ""
    try:
        rec = _reader.get(addr)
    except Exception:
        return ""
    if not isinstance(rec, dict):
        return ""
    code = rec.get("country_code")
    if not code and isinstance(rec.get("country"), dict):
        code = rec["country"].get("iso_code")
    code = str(code or "").strip().upper()
    return code if _CODE.match(code) else ""


def loaded() -> bool:
    return _reader is not None


# --- Loading ----------------------------------------------------------------------

def _open(path):
    """A validated in-memory reader for ``path``, or None."""
    try:
        import maxminddb
    except ImportError:
        return None
    try:
        reader = maxminddb.open_database(path, maxminddb.MODE_MEMORY)
        meta = reader.metadata()
        if "country" not in str(getattr(meta, "database_type", "")).lower():
            raise ValueError(f"not a country database: {meta.database_type}")
        reader.get("8.8.8.8")  # walks the tree once: a broken file fails here
        return reader
    except Exception as e:
        print(f"⚠️  GeoIP: cannot use {path}: {e}")
        return None


def reload_if_changed() -> bool:
    """Load the file when it appeared or changed since the last load. Keeps
    the current reader if the new file is unusable. Returns True on a load."""
    global _reader, _loaded_mtime
    path = db_path()
    try:
        mtime = os.path.getmtime(path)
    except OSError:
        return False
    if mtime == _loaded_mtime and _reader is not None:
        return False
    reader = _open(path)
    _loaded_mtime = mtime
    if reader is None:
        return False
    _reader = reader
    print(f"🌍 GeoIP country database loaded ({os.path.getsize(path) // 1024} KB).")
    return True


def _fetch(url, timeout=None) -> bytes:
    """GET ``url`` whole, bounded in size and in total time (httpx timeouts
    are per operation: a slow trickle would never hit them)."""
    import httpx
    total = DOWNLOAD_TIMEOUT if timeout is None else timeout
    deadline = time.monotonic() + total
    with httpx.Client(timeout=min(30.0, total), follow_redirects=True) as client:
        with client.stream("GET", url) as resp:
            resp.raise_for_status()
            chunks, size = [], 0
            for chunk in resp.iter_bytes():
                size += len(chunk)
                if size > MAX_BYTES:
                    raise ValueError("database larger than expected")
                if time.monotonic() > deadline:
                    raise TimeoutError(f"download took over {total:.0f} s")
                chunks.append(chunk)
    return b"".join(chunks)


def refresh(now=None, fetch=None) -> bool:
    """Download the database when it is missing or stale (worker thread).
    Returns True when a new file was put in place."""
    url = db_url()
    if not url:
        return False
    path = db_path()
    now = time.time() if now is None else now
    try:
        if now - os.path.getmtime(path) < refresh_seconds():
            return False
    except OSError:
        pass
    fetch = fetch or _fetch
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    lock = _try_lock(path + ".lock")
    if lock is None:
        return False  # the other container is downloading it
    tmp = None
    try:
        data = fetch(url)
        want = sha256_url()
        if want:
            expected = fetch(want).decode("ascii", "replace").split()[0].strip().lower()
            if hashlib.sha256(data).hexdigest() != expected:
                raise ValueError("checksum mismatch")
        fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path) or ".", suffix=".part")
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        if _open(tmp) is None:
            raise ValueError("downloaded file is not a usable country database")
        os.replace(tmp, path)
        tmp = None
        return True
    except Exception as e:
        print(f"⚠️  GeoIP: refresh failed, keeping the current database: {e}")
        return False
    finally:
        if tmp:
            try:
                os.remove(tmp)
            except OSError:
                pass
        if lock:
            lock.close()


def _try_lock(path):
    try:
        import fcntl
        fh = open(path, "a")
    except (ImportError, OSError):
        return False
    try:
        fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        fh.close()
        return None
    return fh


async def _loop(wanted):
    while True:
        try:
            if wanted():
                await asyncio.to_thread(refresh)
                await asyncio.to_thread(reload_if_changed)
        except asyncio.CancelledError:
            return
        except Exception as e:
            print(f"⚠️  GeoIP loop error: {e}")
        # Without a database (the other container holds the download lock, a
        # failed download): look again soon, not in an hour.
        await asyncio.sleep(CHECK_EVERY if loaded() or not wanted() else RETRY_SOON)


def start(state_dir, wanted):
    """From the app lifespan (Cloud): keep the database present and loaded
    while ``wanted()``. Never raises; ``GEOIP_DISABLED=1`` turns it off."""
    global _state_dir, _task
    _state_dir = state_dir
    if _flag("GEOIP_DISABLED"):
        return
    _task = asyncio.create_task(_loop(wanted))


def reset():
    """Tests: no reader, nothing loaded."""
    global _reader, _loaded_mtime, _state_dir
    _reader = _loaded_mtime = _state_dir = None
