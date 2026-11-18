"""The local web UI: ``python -m livewatch ui`` opens http://127.0.0.1:8765.

One process: this HTTP server (standard library only) and, when started from
the page, the watcher and replay runs in background threads. The page polls
JSON endpoints once a second.

Local-only by design:
- binds 127.0.0.1, and refuses requests whose Host is not localhost/127.0.0.1
  on its port (DNS-rebinding);
- every request that changes something must carry ``X-Livewatch: 1``. A web
  page on another site cannot add that header without a CORS preflight,
  which this server never approves, so it cannot drive the UI;
- clip and music file names from the URL are checked against a strict
  pattern and must resolve inside their folders.
"""
from __future__ import annotations

import json
import mimetypes
import os
import re
import shutil
import threading
import time
import webbrowser
from datetime import datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Dict, Optional
from urllib.parse import parse_qs, unquote, urlparse

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib  # type: ignore

from . import __version__, config, music, poster, render, tomlwrite, transcribe
from .settings_schema import SECRET_KEYS, schema
from .service import Logger, Service, post_now, write_meta

WEB_DIR = Path(__file__).resolve().parent / "web"
CLIP_ID = re.compile(r"^[A-Za-z0-9_.@-]{1,120}/[A-Za-z0-9_-]{1,40}$")
MAX_JSON = 1_000_000
MAX_UPLOAD = music.MAX_BYTES + 1024


class ApiError(Exception):
    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status
        self.message = message


def _mask(v: str) -> str:
    return ("•" * 6 + v[-4:]) if len(v) > 8 else "•" * len(v)


class Controller:
    """Everything the page can ask for; no HTTP in here (unit-testable)."""

    def __init__(self, config_path: Path):
        self.config_path = Path(config_path).resolve()
        self.base = self.config_path.parent
        self.env_path = self.base / ".env"
        self._lock = threading.Lock()
        self.svc: Optional[Service] = None
        self.replay = None
        self.replay_thread: Optional[threading.Thread] = None
        self.replay_error = ""
        self._loggers: Dict[str, Logger] = {}
        self._posting: Dict[str, threading.Thread] = {}
        config.load_dotenv(self.env_path)

    # ------------------------------------------------------------- config
    def raw_config(self) -> dict:
        if not self.config_path.is_file():
            return {}
        return tomllib.loads(self.config_path.read_text(encoding="utf-8"))

    def settings(self) -> config.Settings:
        return config.from_dict(self.raw_config(), base_dir=self.base)

    def config_version(self) -> str:
        """Fingerprint of the settings file as it is now (path + content)."""
        import hashlib
        body = self.config_path.read_bytes() if self.config_path.is_file() else b""
        return hashlib.sha1(str(self.config_path).encode() + b"\0" + body).hexdigest()[:16]

    BACKUPS_KEPT = 20

    @property
    def backup_dir(self) -> Path:
        return self.base / "livewatch_backups"

    def backup_config(self) -> Optional[Path]:
        """Copy the settings file aside before it is overwritten; keep the last 20."""
        if not self.config_path.is_file():
            return None
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        dest = self.backup_dir / f"{self.config_path.stem}-{stamp}.toml"
        shutil.copy2(self.config_path, dest)
        old = sorted(self.backup_dir.glob(f"{self.config_path.stem}-*.toml"))
        for p in old[:-self.BACKUPS_KEPT]:
            p.unlink(missing_ok=True)
        return dest

    def backups(self) -> list:
        if not self.backup_dir.is_dir():
            return []
        out = []
        for p in sorted(self.backup_dir.glob(f"{self.config_path.stem}-*.toml"), reverse=True):
            try:
                data = tomllib.loads(p.read_text(encoding="utf-8"))
                chans = [f"{c.get('platform')}:{c.get('name')}" for c in data.get("channel", [])]
            except (OSError, ValueError):
                chans = []
            out.append({"name": p.name, "saved": p.stat().st_mtime, "channels": chans})
        return out

    def restore_backup(self, name: str) -> dict:
        p = (self.backup_dir / str(name or "")).resolve()
        if p.parent != self.backup_dir.resolve() or not p.is_file() or p.suffix != ".toml":
            raise ApiError(404, "no such backup")
        raw = tomllib.loads(p.read_text(encoding="utf-8"))
        return self.save_config(raw)  # validated, and the current file is backed up first

    def save_config(self, raw: dict, version: Optional[str] = None) -> dict:
        if not isinstance(raw, dict):
            raise ApiError(400, "settings must be an object")
        if version is not None and version != self.config_version():
            # a page loaded from another (or an older) settings file must not overwrite this one
            raise ApiError(409, "the settings file changed since this page loaded it; "
                                "press Discard changes to reload, then edit again")
        try:
            config.from_dict(raw, base_dir=self.base)  # the watcher's own validation
            text = tomlwrite.dumps(raw)
            tomllib.loads(text)
        except (config.ConfigError, TypeError, ValueError) as e:
            raise ApiError(400, str(e))
        self.backup_config()
        tmp = self.config_path.with_suffix(".toml.tmp")
        tmp.write_text(text, encoding="utf-8")
        tmp.replace(self.config_path)
        running = bool(self.svc and not self.svc.finished.is_set())
        return {"ok": True, "restart_needed": running, "version": self.config_version()}

    def logger(self, s: Optional[config.Settings] = None) -> Logger:
        s = s or self._safe_settings()
        key = str(s.data_dir)
        if key not in self._loggers:
            s.data_dir.mkdir(parents=True, exist_ok=True)
            self._loggers[key] = Logger(s.data_dir / "livewatch.log")
        return self._loggers[key]

    def _safe_settings(self) -> config.Settings:
        try:
            return self.settings()
        except config.ConfigError:
            return config.from_dict({}, base_dir=self.base)

    # ------------------------------------------------------------ secrets
    def secrets(self) -> dict:
        return {k: {"set": bool(os.environ.get(k)), "hint": _mask(os.environ.get(k, ""))}
                for k in SECRET_KEYS}

    def save_secrets(self, values: dict) -> dict:
        if not isinstance(values, dict):
            raise ApiError(400, "expected an object")
        for k, v in values.items():
            if k not in SECRET_KEYS or not isinstance(v, str):
                raise ApiError(400, f"unknown key {k}")
            if len(v) > 500 or any(c in v for c in "\r\n\0"):
                raise ApiError(400, f"{k} does not look like an API key")
        lines = self.env_path.read_text(encoding="utf-8").splitlines() if self.env_path.exists() else []
        for k, v in values.items():
            v = v.strip()
            lines = [ln for ln in lines if not re.match(rf"^\s*(export\s+)?{k}\s*=", ln)]
            if v:
                lines.append(f"{k}={v}")
                os.environ[k] = v
            else:
                os.environ.pop(k, None)
        tmp = self.env_path.with_suffix(".tmp")
        tmp.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        tmp.replace(self.env_path)
        return self.secrets()

    # -------------------------------------------------------------- watcher
    def start(self) -> dict:
        with self._lock:
            if self.svc and not self.svc.finished.is_set():
                raise ApiError(409, "already running")
            try:
                s = self.settings()
            except config.ConfigError as e:
                raise ApiError(400, f"fix the settings first: {e}")
            if not s.channels:
                raise ApiError(400, "add at least one channel in Settings first")
            self.svc = Service(s, log=self.logger(s))
            threading.Thread(target=self.svc.run, daemon=True, name="watcher").start()
        return {"ok": True}

    def stop(self) -> dict:
        if self.svc:
            self.svc.stop()
        return {"ok": True}

    def check_now(self) -> dict:
        if self.svc:
            self.svc.check_now()
        return {"ok": True}

    def status(self) -> dict:
        cfg_error = ""
        try:
            s = self.settings()
        except config.ConfigError as e:
            cfg_error = str(e)
            s = config.from_dict({}, base_dir=self.base)
        if self.svc:
            st = self.svc.status()
        else:
            st = {"running": False, "stopping": False, "started_at": None, "queue": 0,
                  "current_step": "", "counts": {"peaks": 0, "clips": 0, "posted": 0, "failed": 0},
                  "threshold": s.threshold, "encoder": render.pick_encoder(s.encoder),
                  "channels": [{"key": c.key, "platform": c.platform, "name": c.name,
                                "display": c.display, "url": c.url, "state": "stopped",
                                "live": False, "recording": False, "category": "", "title": "",
                                "game_filter": c.game_filter} for c in s.channels]}
        st.update({
            "version": __version__, "config_error": cfg_error,
            "config_exists": self.config_path.is_file(), "config_path": str(self.config_path),
            "ai": "gemini" if os.environ.get("GEMINI_API_KEY") else "offline templates",
            "posting": bool(s.post_enabled and os.environ.get("UPLOADPOST_API_KEY")
                            and s.upload_post_user),
            "post_enabled": s.post_enabled, "whisper": transcribe.available() or "missing",
            "music_mode": s.music_mode, "music_tracks": len(music.Library(s.music_dir).tracks())
            if s.music_dir else 0,
            "disk_free_gb": round(shutil.disk_usage(self.base).free / 1e9, 1),
            "replay": self.replay_status(),
        })
        return st

    def log_since(self, seq: int) -> dict:
        lg = self.logger()
        lines = lg.since(seq)
        return {"seq": lg.seq, "boot": lg.boot,
                "lines": [{"seq": n, "text": t} for n, t in lines[-500:]]}

    # --------------------------------------------------------------- replay
    def replay_status(self) -> Optional[dict]:
        r = self.replay
        if not r:
            return None
        running = bool(self.replay_thread and self.replay_thread.is_alive())
        return {"running": running, "url": r.url, "step": r.step, "progress": round(r.progress, 3),
                "title": r.meta.get("title", ""), "channel": r.meta.get("channel", ""),
                "results": r.results, "error": self.replay_error}

    def start_replay(self, body: dict) -> dict:
        from .replay import Replay, ReplayError
        if self.replay_thread and self.replay_thread.is_alive():
            raise ApiError(409, "a replay is already running")
        try:
            s = self.settings()
            post = body.get("post")
            rep = Replay(s, str(body.get("url", "")), minutes=float(body.get("minutes", 30)),
                         start_minute=float(body.get("start", 0)),
                         max_clips=int(body.get("max_clips", 5)),
                         post=bool(post) if post is not None else None, log=self.logger(s))
        except (ReplayError, config.ConfigError, ValueError, TypeError) as e:
            raise ApiError(400, str(e))
        self.replay, self.replay_error = rep, ""

        def run():
            try:
                rep.run()
            except ReplayError as e:
                self.replay_error = str(e)
                rep.log(f"replay: {e}")
            except Exception as e:  # noqa: BLE001 - shown to the user
                self.replay_error = f"{type(e).__name__}: {e}"
                rep.log(f"replay failed: {self.replay_error}")
            finally:
                rep.step = rep.step if rep.step == "done" else "stopped"

        self.replay_thread = threading.Thread(target=run, daemon=True, name="replay")
        self.replay_thread.start()
        return {"ok": True}

    def cancel_replay(self) -> dict:
        if self.replay:
            self.replay.cancel()
        return {"ok": True}

    # ---------------------------------------------------------------- clips
    def clips_root(self) -> Path:
        return self._safe_settings().data_dir / "clips"

    def clip_dir(self, clip_id: str) -> Path:
        if not CLIP_ID.match(clip_id or ""):
            raise ApiError(404, "no such clip")
        root = self.clips_root().resolve()
        d = (root / clip_id).resolve()
        if root not in d.parents or not (d / "meta.json").is_file():
            raise ApiError(404, "no such clip")
        return d

    def clips(self) -> list:
        root = self.clips_root()
        if not root.is_dir():
            return []
        metas = sorted(root.glob("*/*/meta.json"), key=lambda p: p.stat().st_mtime, reverse=True)
        out = []
        for m in metas[:300]:
            try:
                meta = json.loads(m.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            cid = f"{m.parent.parent.name}/{m.parent.name}"
            if not CLIP_ID.match(cid):
                continue
            out.append({
                "id": cid, "channel": meta.get("channel"), "streamer": meta.get("streamer"),
                "moment": meta.get("moment"), "duration": meta.get("duration"),
                "score": meta.get("score"), "reason": meta.get("reason"),
                "category": meta.get("category"), "chat": meta.get("chat", [])[:8],
                "transcript": (meta.get("transcript") or "")[:1500],
                "text": meta.get("text", {}), "posts": meta.get("posts", {}),
                "posted": meta.get("posted"), "music": meta.get("music"),
                "has_video": (m.parent / "short.mp4").is_file(),
                "posting_now": cid in self._posting and self._posting[cid].is_alive(),
                "created": m.stat().st_mtime,
            })
        return out

    def update_posts(self, clip_id: str, posts: dict) -> dict:
        d = self.clip_dir(clip_id)
        meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
        if not isinstance(posts, dict):
            raise ApiError(400, "expected an object of networks")
        clean = {}
        for p, t in posts.items():
            if p not in config.POST_PLATFORMS or not isinstance(t, dict):
                raise ApiError(400, f"unknown network {p}")
            title, desc = t.get("title", ""), t.get("description", "")
            if not isinstance(title, str) or not isinstance(desc, str):
                raise ApiError(400, "title and description must be text")
            if len(title) > 300 or len(desc) > 5000:
                raise ApiError(400, f"{p}: text too long")
            clean[p] = {"title": title, "description": desc}
        meta["posts"] = clean
        write_meta(d, meta)
        return {"ok": True}

    def post_clip(self, clip_id: str, platforms: Optional[list] = None) -> dict:
        d = self.clip_dir(clip_id)
        s = self.settings()
        if not (os.environ.get("UPLOADPOST_API_KEY") and s.upload_post_user):
            raise ApiError(400, "add your Upload-Post API key (Settings > Keys) and profile name "
                                "(Settings > Posting) first")
        if clip_id in self._posting and self._posting[clip_id].is_alive():
            raise ApiError(409, "this clip is already being posted")
        meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
        posts = meta.get("posts") or {}
        if platforms:
            posts = {p: v for p, v in posts.items() if p in platforms}
        if not posts:
            raise ApiError(400, "no networks selected")
        s.upload_post_api_key = os.environ.get("UPLOADPOST_API_KEY", "")
        ledger = poster.PostLedger(s.data_dir / "post_ledger.json")
        when = ledger.slot(datetime.now(timezone.utc), timedelta(minutes=s.min_gap_minutes),
                           s.daily_cap)
        if when is None:
            raise ApiError(429, "daily post cap reached (Settings > Posting)")
        log = self.logger(s)

        def run():
            meta["posted"] = {"status": "posting"}
            write_meta(d, meta)
            meta["posted"] = post_now(d / "short.mp4", posts, s, ledger, when, log)
            write_meta(d, meta)

        th = threading.Thread(target=run, daemon=True, name=f"post-{clip_id}")
        self._posting[clip_id] = th
        th.start()
        return {"ok": True, "scheduled_for": when.isoformat()}

    def delete_clip(self, clip_id: str) -> dict:
        d = self.clip_dir(clip_id)
        shutil.rmtree(d)
        return {"ok": True}

    def open_folder(self, clip_id: Optional[str] = None) -> dict:
        d = self.clip_dir(clip_id) if clip_id else self.clips_root()
        d.mkdir(parents=True, exist_ok=True)
        if hasattr(os, "startfile"):
            os.startfile(str(d))  # type: ignore[attr-defined]
        return {"ok": True, "path": str(d)}

    # ---------------------------------------------------------- upload-post
    def _upload_key(self) -> str:
        key = os.environ.get("UPLOADPOST_API_KEY", "")
        if not key:
            raise ApiError(400, "add your Upload-Post API key first (Settings > API keys)")
        return key

    def upload_profiles(self) -> dict:
        try:
            profiles = poster.list_profiles(self._upload_key())
        except poster.UploadPostError as e:
            raise ApiError(502, str(e))
        except OSError as e:
            raise ApiError(502, f"could not reach Upload-Post: {e}")
        current = self._safe_settings().upload_post_user
        return {"profiles": profiles, "current": current,
                "current_ok": any(p["username"] == current for p in profiles)}

    def create_upload_profile(self, body: dict) -> dict:
        try:
            name = poster.create_profile(self._upload_key(), str((body or {}).get("username", "")))
        except poster.UploadPostError as e:
            raise ApiError(400, str(e))
        except OSError as e:
            raise ApiError(502, f"could not reach Upload-Post: {e}")
        return {"ok": True, "username": name}

    def upload_connect_url(self, body: dict, return_url: str) -> dict:
        try:
            url = poster.connect_url(self._upload_key(), str((body or {}).get("username", "")),
                                     return_url)
        except poster.UploadPostError as e:
            raise ApiError(400, str(e))
        except OSError as e:
            raise ApiError(502, f"could not reach Upload-Post: {e}")
        return {"url": url}

    # ---------------------------------------------------------------- music
    def library(self) -> music.Library:
        s = self._safe_settings()
        return music.Library(s.music_dir or s.data_dir / "music")

    def music_list(self) -> dict:
        lib = self.library()
        return {"folder": str(lib.folder), "tracks": lib.describe()}

    def music_file(self, name: str) -> Path:
        lib = self.library()
        p = (lib.folder / unquote(name)).resolve()
        if p.parent != lib.folder.resolve() or not p.is_file() or p.suffix.lower() not in music.EXTENSIONS:
            raise ApiError(404, "no such track")
        return p


# --------------------------------------------------------------------------- #
# HTTP
# --------------------------------------------------------------------------- #
class Handler(BaseHTTPRequestHandler):
    server_version = "livewatch"
    ctl: Controller
    port: int

    def log_message(self, fmt, *args):  # keep the console for the watcher log
        pass

    # ------------------------------------------------------------- helpers
    def _host_ok(self) -> bool:
        host = (self.headers.get("Host") or "").lower()
        return host in (f"127.0.0.1:{self.port}", f"localhost:{self.port}")

    def _json(self, data, status: int = 200):
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _error(self, status: int, message: str):
        self._json({"error": message}, status)

    def _body(self, limit: int = MAX_JSON) -> bytes:
        n = int(self.headers.get("Content-Length") or 0)
        if n < 0 or n > limit:
            raise ApiError(413, "request too large")
        return self.rfile.read(n) if n else b""

    def _json_body(self):
        try:
            return json.loads(self._body() or b"{}")
        except ValueError:
            raise ApiError(400, "invalid JSON")

    def _send_file(self, path: Path, ctype: str):
        size = path.stat().st_size
        rng = self.headers.get("Range")
        start, end = 0, size - 1
        status = 200
        if rng:
            m = re.match(r"^bytes=(\d*)-(\d*)$", rng.strip())
            if not m or (not m.group(1) and not m.group(2)):
                self.send_response(416)
                self.send_header("Content-Range", f"bytes */{size}")
                self.end_headers()
                return
            if m.group(1):
                start = int(m.group(1))
                end = int(m.group(2)) if m.group(2) else size - 1
            else:
                start = max(0, size - int(m.group(2)))
            end = min(end, size - 1)
            if start > end:
                self.send_response(416)
                self.send_header("Content-Range", f"bytes */{size}")
                self.end_headers()
                return
            status = 206
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(end - start + 1))
        self.send_header("Cache-Control", "no-store")
        if status == 206:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.end_headers()
        with open(path, "rb") as fh:
            fh.seek(start)
            left = end - start + 1
            while left > 0:
                chunk = fh.read(min(256 * 1024, left))
                if not chunk:
                    break
                try:
                    self.wfile.write(chunk)
                except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                    return
                left -= len(chunk)

    # ------------------------------------------------------------- routing
    def do_GET(self):
        self._dispatch("GET")

    def do_POST(self):
        self._dispatch("POST")

    def do_PUT(self):
        self._dispatch("PUT")

    def do_DELETE(self):
        self._dispatch("DELETE")

    def _dispatch(self, method: str):
        if not self._host_ok():
            self._error(403, "livewatch only answers on localhost")
            return
        if method != "GET":
            if self.headers.get("X-Livewatch") != "1":
                self._error(403, "missing X-Livewatch header")
                return
            origin = self.headers.get("Origin")
            if origin and urlparse(origin).netloc.lower() not in (f"127.0.0.1:{self.port}",
                                                                    f"localhost:{self.port}"):
                self._error(403, "cross-origin request refused")
                return
        url = urlparse(self.path)
        path = url.path
        q = parse_qs(url.query)
        try:
            self._route(method, path, q)
        except ApiError as e:
            self._error(e.status, e.message)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass
        except Exception as e:  # noqa: BLE001 - reported to the page
            self._error(500, f"{type(e).__name__}: {e}")

    def _route(self, method: str, path: str, q: dict):
        ctl = self.ctl
        if method == "GET":
            if path in ("/", "/index.html"):
                return self._send_file(WEB_DIR / "index.html", "text/html; charset=utf-8")
            if path == "/api/status":
                return self._json(ctl.status())
            if path == "/api/log":
                return self._json(ctl.log_since(int((q.get("since") or ["0"])[0] or 0)))
            if path == "/api/config":
                cfg_error = ""
                try:
                    ctl.settings()
                except config.ConfigError as e:
                    cfg_error = str(e)
                return self._json({"raw": ctl.raw_config(), "schema": schema(),
                                   "error": cfg_error, "path": str(ctl.config_path),
                                   "version": ctl.config_version()})
            if path == "/api/secrets":
                return self._json(ctl.secrets())
            if path == "/api/clips":
                return self._json({"clips": ctl.clips()})
            if path == "/api/music":
                return self._json(ctl.music_list())
            if path == "/api/uploadpost/profiles":
                return self._json(ctl.upload_profiles())
            if path == "/api/config/backups":
                return self._json({"backups": ctl.backups()})
            m = re.match(r"^/media/(.+)/short\.mp4$", path)
            if m:
                d = ctl.clip_dir(unquote(m.group(1)))
                f = d / "short.mp4"
                if not f.is_file():
                    raise ApiError(404, "not rendered")
                return self._send_file(f, "video/mp4")
            m = re.match(r"^/music/([^/]+)$", path)
            if m:
                f = ctl.music_file(m.group(1))
                return self._send_file(f, mimetypes.guess_type(f.name)[0] or "audio/mpeg")
            raise ApiError(404, "not found")

        if method == "POST":
            if path == "/api/start":
                return self._json(ctl.start())
            if path == "/api/stop":
                return self._json(ctl.stop())
            if path == "/api/check":
                return self._json(ctl.check_now())
            if path == "/api/replay":
                return self._json(ctl.start_replay(self._json_body()))
            if path == "/api/replay/cancel":
                return self._json(ctl.cancel_replay())
            if path == "/api/open":
                return self._json(ctl.open_folder())
            if path == "/api/uploadpost/profiles":
                return self._json(ctl.create_upload_profile(self._json_body()))
            if path == "/api/config/restore":
                body = self._json_body()
                return self._json(ctl.restore_backup(body.get("name") if isinstance(body, dict) else ""))
            if path == "/api/uploadpost/connect":
                return self._json(ctl.upload_connect_url(
                    self._json_body(), f"http://127.0.0.1:{self.port}/"))
            if path == "/api/music":
                name = unquote(self.headers.get("X-Filename") or "")
                data = self._body(MAX_UPLOAD)
                try:
                    saved = ctl.library().add(name, data)
                except ValueError as e:
                    raise ApiError(400, str(e))
                return self._json({"ok": True, "name": saved})
            m = re.match(r"^/api/music/([^/]+)/(enable|disable)$", path)
            if m:
                try:
                    ctl.library().set_enabled(ctl.music_file(m.group(1)).name, m.group(2) == "enable")
                except FileNotFoundError:
                    raise ApiError(404, "no such track")
                return self._json({"ok": True})
            m = re.match(r"^/api/clips/(.+)/(post|open)$", path)
            if m:
                cid = unquote(m.group(1))
                if m.group(2) == "open":
                    return self._json(ctl.open_folder(cid))
                body = self._json_body()
                plats = body.get("platforms") if isinstance(body, dict) else None
                return self._json(ctl.post_clip(cid, plats if isinstance(plats, list) else None))
            raise ApiError(404, "not found")

        if method == "PUT":
            if path == "/api/config":
                body = self._json_body()
                if not isinstance(body, dict) or "raw" not in body or not isinstance(body.get("version"), str):
                    raise ApiError(400, "reload the page: it is older than the server")
                return self._json(ctl.save_config(body["raw"], body["version"]))
            if path == "/api/secrets":
                return self._json(ctl.save_secrets(self._json_body()))
            m = re.match(r"^/api/clips/(.+)/posts$", path)
            if m:
                return self._json(ctl.update_posts(unquote(m.group(1)), self._json_body()))
            raise ApiError(404, "not found")

        if method == "DELETE":
            m = re.match(r"^/api/clips/(.+)$", path)
            if m:
                return self._json(ctl.delete_clip(unquote(m.group(1))))
            m = re.match(r"^/api/music/([^/]+)$", path)
            if m:
                ctl.library().remove(ctl.music_file(m.group(1)).name)
                return self._json({"ok": True})
            raise ApiError(404, "not found")


def make_server(ctl: Controller, port: int = 8765) -> ThreadingHTTPServer:
    handler = type("BoundHandler", (Handler,), {"ctl": ctl, "port": port})
    srv = ThreadingHTTPServer(("127.0.0.1", port), handler)
    srv.daemon_threads = True
    # an ephemeral port (0) is known only after binding
    handler.port = srv.server_address[1]
    return srv


def serve(config_path: Path, port: int = 8765, open_browser: bool = True) -> None:
    ctl = Controller(config_path)
    srv = make_server(ctl, port)
    url = f"http://127.0.0.1:{srv.server_address[1]}/"
    print(f"livewatch UI on {url}  (Ctrl+C to quit)", flush=True)
    if open_browser:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    try:
        srv.serve_forever(poll_interval=0.5)
    except KeyboardInterrupt:
        print("stopping...", flush=True)
    finally:
        if ctl.svc:
            ctl.svc.stop()
            deadline = time.time() + 60
            while not ctl.svc.finished.is_set() and time.time() < deadline:
                time.sleep(0.5)
        if ctl.replay:
            ctl.replay.cancel()
        srv.server_close()
