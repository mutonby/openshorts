import asyncio
import atexit
import csv
import datetime
import json
import os
import re
import shutil
import string
import subprocess
import time
import uuid
from urllib.parse import urlparse

import httpx
from fastapi import APIRouter, HTTPException, Request

router = APIRouter()

DEFAULT_CHUNK_MINUTES = float(os.environ.get("LIVE_CHUNK_MINUTES", "20"))
MIN_FREE_GB = float(os.environ.get("LIVE_MIN_FREE_GB", "5"))
MIN_CHUNK_SECONDS = 60
STALL_SECONDS = 90
MAX_RESTARTS = 5
POLL_SECONDS = 3
WAIT_RETRY_SECONDS = 30
MAX_SCHEDULE_DAYS = 7
REGISTRY = os.path.join("output", ".live_sessions.json")
FORMAT = ("bestvideo[height<=1080][protocol^=m3u8]+bestaudio[protocol^=m3u8]/"
          "best[height<=1080][protocol^=m3u8]/best[protocol^=m3u8]")
FRAGMENTED = "movflags=+frag_keyframe+empty_moov+default_base_moof"
ACTIVE = ("scheduled", "waiting", "recording", "paused", "ended")
LANGUAGES = ("en", "hi")
JOB_OPTION_KEYS = ("target_clips", "clip_min_seconds", "clip_max_seconds",
                   "auto_hook", "auto_hook_style", "layouts")
UPCOMING_HINTS = ("will begin", "begin in", "upcoming", "premiere", "scheduled")
_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)
_LOCAL_HOSTS = ("localhost", "127.0.0.1", "::1")

_sessions = {}
_recorders = set()


def _core():
    import app
    return app


def slug(text, limit=60):
    s = re.sub(r"[^A-Za-z0-9]+", "-", str(text or "")).strip("-")
    return s[:limit].strip("-") or "clip"


ROMANIZE_PROMPT = (
    "Transliterate this short video title into the English (Latin) alphabet, the way "
    "people write Hindi/Hinglish in English letters. Keep the original words and their "
    "meaning: do not translate them into English. Drop emoji and hashtags. Use only "
    "letters A-Z, digits and spaces.\n\nTitle: {title}"
)


def romanize(title):
    title = str(title or "").strip()
    if not title or title.isascii():
        return title
    try:
        from pydantic import BaseModel

        import llm_backend

        class RomanTitle(BaseModel):
            title: str

        if not llm_backend.active():
            return ""
        parsed, _ = llm_backend.generate_json(ROMANIZE_PROMPT.format(title=title), RomanTitle)
        return str(parsed.get("title") or "")
    except Exception as e:
        print(f"[live] could not romanize a title: {e}")
        return ""


def read_segments(list_path):
    try:
        with open(list_path, newline="") as f:
            rows = list(csv.reader(f))
    except OSError:
        return []
    out = []
    for row in rows:
        if len(row) < 3:
            continue
        try:
            out.append((os.path.basename(row[0]), float(row[1]), float(row[2])))
        except ValueError:
            continue
    return out


def recorder_cmd(formats, chunk_dir, state_dir, run, start_number, chunk_seconds):
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y"]
    for f in formats:
        ua = (f.get("http_headers") or {}).get("User-Agent")
        if ua:
            cmd += ["-user_agent", ua]
        cmd += ["-i", f["url"]]
    audio_input = len(formats) - 1
    cmd += ["-map", "0:v:0", "-map", f"{audio_input}:a:0", "-c", "copy",
            "-f", "segment", "-segment_time", str(int(chunk_seconds)),
            "-segment_start_number", str(start_number), "-reset_timestamps", "1",
            "-segment_format", "mp4", "-segment_format_options", FRAGMENTED,
            "-segment_list", os.path.join(state_dir, f"list_{run}.csv"),
            "-segment_list_type", "csv",
            os.path.join(chunk_dir, "part_%05d.mp4")]
    return cmd


def clean_job_options(raw):
    if not isinstance(raw, dict):
        return {}
    return {k: str(raw[k])[:40] for k in JOB_OPTION_KEYS
            if raw.get(k) not in (None, "")}


def part_index(name):
    m = re.match(r"^part_(\d+)\.mp4$", name)
    return int(m.group(1)) if m else None


def is_upcoming_error(message):
    text = str(message or "").lower()
    return any(h in text for h in UPCOMING_HINTS)


def _extract(url):
    import ssrf_guard
    import yt_dlp
    opts = {"quiet": True, "no_warnings": True, "noplaylist": True,
            "skip_download": True, "cachedir": False, "format": FORMAT}
    with ssrf_guard.guarded(), yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=False)
    ssrf_guard.assert_public_urls(info)
    return info


def _free_gb(path):
    try:
        return shutil.disk_usage(path).free / 1e9
    except OSError:
        return float("inf")


def _stop_process(proc, graceful=True):
    if proc is None or proc.poll() is not None:
        return
    if graceful:
        try:
            proc.stdin.write(b"q")
            proc.stdin.flush()
            proc.wait(timeout=20)
            return
        except Exception:
            pass
    try:
        proc.kill()
        proc.wait(timeout=10)
    except Exception:
        pass


def _write_json(path, data):
    tmp = f"{path}.{uuid.uuid4().hex[:6]}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=1, ensure_ascii=False)
    os.replace(tmp, path)


def _read_registry():
    try:
        with open(REGISTRY, encoding="utf-8") as f:
            data = json.load(f)
        return {str(k): str(v) for k, v in data.items()} if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def _registry_set(session_id, session_dir):
    reg = _read_registry()
    if session_dir:
        reg[session_id] = session_dir
    else:
        reg.pop(session_id, None)
    try:
        os.makedirs(os.path.dirname(REGISTRY), exist_ok=True)
        _write_json(REGISTRY, reg)
    except OSError:
        pass


def _kill_stray(pid):
    try:
        pid = int(pid)
    except (TypeError, ValueError):
        return False
    if pid <= 0:
        return False
    try:
        if os.name == "nt":
            out = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH"],
                                 capture_output=True, text=True, timeout=15).stdout
            if "ffmpeg" not in out.lower():
                return False
            subprocess.run(["taskkill", "/PID", str(pid), "/F"], capture_output=True, timeout=15)
        else:
            out = subprocess.run(["ps", "-p", str(pid), "-o", "comm="],
                                 capture_output=True, text=True, timeout=15).stdout
            if "ffmpeg" not in out.lower():
                return False
            os.kill(pid, 9)
    except Exception:
        return False
    time.sleep(1)
    return True


@atexit.register
def _kill_recorders():
    for proc in list(_recorders):
        _stop_process(proc, graceful=False)


class LiveSession:
    def __init__(self, url, save_dir, chunk_minutes, start_at=None, language=None, job_options=None):
        self.id = uuid.uuid4().hex[:12]
        self.job_options = clean_job_options(job_options)
        self.url = url
        self.chunk_seconds = chunk_minutes * 60
        self.start_at = start_at
        self.language = language
        stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.dir = os.path.join(save_dir, f"live_{stamp}_{self.id[:6]}")
        self.state = "scheduled" if start_at and start_at > time.time() else "waiting"
        self.error = None
        self.title = None
        self.started_at = time.time()
        self.chunks = []
        self.clips = []
        self.log = []
        self.recorder = None
        self.run = 0
        self.next_index = 0
        self.seen = set()
        self.segment_started = None
        self.restarts = 0
        self.next_try = 0.0
        self.waiting_noted = False
        self.want_recording = True
        self.task = None

    @property
    def chunk_dir(self):
        return os.path.join(self.dir, "chunks")

    @property
    def state_dir(self):
        return os.path.join(self.dir, ".recording")

    def note(self, msg):
        line = f"{datetime.datetime.now().strftime('%H:%M:%S')} {msg}"
        self.log = (self.log + [line])[-40:]
        print(f"[live {self.id}] {msg}")

    def recorded_seconds(self):
        done = sum(c["duration"] for c in self.chunks)
        if self.state == "recording" and self.segment_started:
            done += max(0.0, time.time() - self.segment_started)
        return round(done, 1)

    def public(self):
        counts = {}
        for c in self.chunks:
            counts[c["status"]] = counts.get(c["status"], 0) + 1
        return {
            "id": self.id, "url": self.url, "title": self.title, "state": self.state,
            "error": self.error, "dir": self.dir, "chunkMinutes": self.chunk_seconds / 60,
            "startAt": self.start_at, "language": self.language or "auto",
            "startedAt": self.started_at, "recordedSeconds": self.recorded_seconds(),
            "chunkCounts": counts,
            "chunks": [{**{k: c.get(k) for k in ("index", "status", "duration", "jobId",
                                                   "clips", "error", "recordedAt")},
                        "available": os.path.isfile(c.get("file") or "")}
                       for c in self.chunks],
            "clips": self.clips[-200:], "log": self.log[-15:],
        }

    def saved(self):
        data = self.public()
        data.update({"chunks": self.chunks, "clips": self.clips, "log": self.log,
                     "run": self.run, "nextIndex": self.next_index,
                     "chunkSeconds": self.chunk_seconds, "jobOptions": self.job_options,
                     "recorderPid": self.recorder.pid if self.recorder is not None else None})
        return data

    def save(self):
        try:
            _write_json(os.path.join(self.dir, "session.json"), self.saved())
        except OSError:
            pass

    @classmethod
    def restore(cls, data):
        s = cls.__new__(cls)
        cls.__init__(s, data["url"], os.path.dirname(data["dir"]),
                     float(data.get("chunkSeconds") or DEFAULT_CHUNK_MINUTES * 60) / 60,
                     data.get("startAt"), data.get("language") if data.get("language") in LANGUAGES else None)
        s.id = data["id"]
        s.dir = data["dir"]
        s.job_options = clean_job_options(data.get("jobOptions"))
        s.title = data.get("title")
        s.started_at = float(data.get("startedAt") or time.time())
        s.chunks = [c for c in data.get("chunks") or [] if isinstance(c, dict)]
        s.clips = [c for c in data.get("clips") or [] if isinstance(c, dict)]
        s.log = [str(x) for x in data.get("log") or []][-40:]
        s.run = int(data.get("run") or 0)
        s.next_index = int(data.get("nextIndex") or 0)
        s.error = data.get("error")
        state = data.get("state")
        if state == "paused":
            s.state = "paused"
            s.want_recording = False
        elif state == "ended":
            s.state = "ended"
        elif state == "scheduled" and (s.start_at or 0) > time.time():
            s.state = "scheduled"
        else:
            s.state = "waiting"
        s.seen = {os.path.basename(c.get("file") or "") for c in s.chunks}
        if data.get("recorderPid") and _kill_stray(data["recorderPid"]):
            s.note("stopped a recorder left over from before the restart")
        s.reconcile()
        s.note("server restarted: session resumed")
        return s

    def reconcile(self):
        core = _core()
        for c in self.chunks:
            if c.get("status") == "submitting":
                c["status"] = "recorded" if os.path.isfile(c.get("file") or "") else "failed"
            elif c.get("status") == "clipping":
                job = core.jobs.get(c.get("jobId") or "")
                if job is None:
                    c["status"] = "recorded" if os.path.isfile(c.get("file") or "") else "failed"
                    c["jobId"] = None
        self.adopt_orphans()

    def adopt_orphans(self):
        core = _core()
        try:
            names = sorted(os.listdir(self.chunk_dir))
        except OSError:
            return
        for name in names:
            index = part_index(name)
            if index is None or name in self.seen:
                continue
            path = os.path.join(self.chunk_dir, name)
            self.seen.add(name)
            self.next_index = max(self.next_index, index + 1)
            duration = core._media_duration_seconds(path)
            self._add_chunk(index, path, duration, os.path.getmtime(path) - duration)
            self.note(f"part {index} recovered from disk ({duration:.0f}s)")

    def _add_chunk(self, index, path, duration, started):
        self.chunks.append({
            "index": index, "file": path, "duration": round(max(0.0, duration), 1),
            "status": "recorded", "jobId": None, "clips": 0, "error": None,
            "recordedAt": datetime.datetime.fromtimestamp(started).isoformat(timespec="seconds"),
        })
        self.chunks.sort(key=lambda c: c["index"])

    async def start_recorder(self):
        try:
            info = await asyncio.to_thread(_extract, self.url)
        except Exception as e:
            if is_upcoming_error(e):
                return "upcoming"
            return f"could not read the stream ({str(e)[:200]})"
        if not self.want_recording:
            return "cancelled"
        status = info.get("live_status")
        if status == "is_upcoming":
            return "upcoming"
        if status != "is_live":
            return "not_live"
        self.title = self.title or info.get("title")
        formats = info.get("requested_formats") or [info]
        if not all(f.get("url") for f in formats):
            return "no recordable format"
        os.makedirs(self.chunk_dir, exist_ok=True)
        os.makedirs(self.state_dir, exist_ok=True)
        self.run += 1
        cmd = recorder_cmd(formats, self.chunk_dir, self.state_dir, self.run,
                           self.next_index, self.chunk_seconds)
        log = open(os.path.join(self.state_dir, "ffmpeg.log"), "ab")
        try:
            self.recorder = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                                             stdout=subprocess.DEVNULL, stderr=log,
                                             creationflags=_NO_WINDOW)
        finally:
            log.close()
        _recorders.add(self.recorder)
        self.segment_started = time.time()
        self.note(f"recording ({formats[0].get('height') or '?'}p)")
        return None

    async def stop_recorder(self, graceful=True):
        proc, self.recorder = self.recorder, None
        if proc is None:
            return
        await asyncio.to_thread(_stop_process, proc, graceful)
        _recorders.discard(proc)
        self.collect()
        if not graceful:
            await asyncio.to_thread(self.adopt_orphans)

    def collect(self):
        for run in range(1, self.run + 1):
            for name, start, end in read_segments(os.path.join(self.state_dir, f"list_{run}.csv")):
                if name in self.seen:
                    continue
                index = part_index(name)
                if index is None:
                    continue
                self.seen.add(name)
                self.next_index = max(self.next_index, index + 1)
                duration = max(0.0, end - start)
                self._add_chunk(index, os.path.join(self.chunk_dir, name), duration,
                                time.time() - duration)
                self.segment_started = time.time()

    def stalled(self):
        try:
            parts = [os.path.join(self.chunk_dir, n) for n in os.listdir(self.chunk_dir)
                     if part_index(n) is not None]
        except OSError:
            return False
        newest = max((os.path.getmtime(p) for p in parts), default=0)
        return time.time() - max(newest, self.segment_started or 0) > STALL_SECONDS

    async def try_start(self):
        if time.time() < self.next_try:
            return
        err = await self.start_recorder()
        if err is None and self.state != "waiting":
            await self.stop_recorder(graceful=False)
            return
        if err is None:
            self.state = "recording"
            self.restarts = 0
            self.error = None
            self.waiting_noted = False
            return
        if err == "cancelled":
            return
        if err == "upcoming":
            self.next_try = time.time() + WAIT_RETRY_SECONDS
            if not self.waiting_noted:
                self.note("stream has not started yet; waiting for it to go live")
                self.waiting_noted = True
            return
        if err == "not_live":
            if self.chunks:
                self.state = "ended"
                self.note("stream ended; clipping what was recorded")
            else:
                self.state = "error"
                self.error = "this stream is not live"
                self.note(self.error)
            return
        self.restarts += 1
        self.next_try = time.time() + 15
        self.note(f"could not start recording: {err}")
        if self.restarts >= MAX_RESTARTS:
            self.error = err
            self.state = "ended" if self.chunks else "error"

    async def supervise_recorder(self):
        proc = self.recorder
        if proc is not None and proc.poll() is None and not self.stalled():
            return
        await self.stop_recorder(graceful=False)
        if not self.want_recording or self.state != "recording":
            return
        self.state = "waiting"
        self.next_try = 0.0
        await self.try_start()

    def clipping(self):
        return next((c for c in self.chunks if c["status"] in ("submitting", "clipping")), None)

    async def submit(self, chunk):
        core = _core()
        chunk["status"] = "submitting"
        path = chunk["file"]
        duration = await asyncio.to_thread(core._media_duration_seconds, path) if os.path.isfile(path) else 0
        if self.state == "stopped":
            chunk["status"] = "kept"
            return
        if duration < max(MIN_CHUNK_SECONDS, core.MIN_SOURCE_SECONDS):
            self._drop(path)
            chunk["status"] = "skipped"
            chunk["error"] = "too short to clip" if duration else "could not read the recording"
            return
        upload_id = str(uuid.uuid4())
        core.pending_uploads[upload_id] = {
            "user_id": None, "filename": os.path.basename(path), "path": path,
            "created": time.time(), "bytes": os.path.getsize(path), "complete": True,
            "in_place": True,
        }
        payload = {**self.job_options, "upload_id": upload_id, "acknowledged": True}
        if self.language:
            payload["language"] = self.language
        try:
            transport = httpx.ASGITransport(app=core.app)
            async with httpx.AsyncClient(transport=transport, base_url="http://live-clipper",
                                         timeout=600) as client:
                r = await client.post("/api/process", json=payload)
            body = r.json()
        except Exception as e:
            r, body = None, {"detail": str(e)}
        if r is None or r.status_code != 200 or not body.get("job_id"):
            core.pending_uploads.pop(upload_id, None)
            chunk["status"] = "failed"
            chunk["error"] = str(body.get("detail") or body)[:300]
            self.note(f"part {chunk['index']} not submitted: {chunk['error']}")
            return
        chunk["jobId"] = body["job_id"]
        if self.state == "stopped":
            chunk["status"] = "kept"
            asyncio.create_task(_cancel_until_done(body["job_id"]))
            return
        chunk["status"] = "clipping"
        self.note(f"part {chunk['index']} clipping (job {body['job_id'][:8]})")

    @staticmethod
    def _drop(*paths):
        for p in paths:
            try:
                os.remove(p)
            except OSError:
                pass

    async def harvest(self, chunk):
        core = _core()
        job = core.jobs.get(chunk["jobId"])
        if job is None:
            chunk["status"] = "recorded" if os.path.isfile(chunk["file"]) else "failed"
            chunk["jobId"] = None
            return
        status = job.get("status")
        if status not in ("completed", "failed"):
            return
        if status == "failed":
            chunk["status"] = "failed"
            chunk["error"] = str((list(job.get("logs") or []) or ["failed"])[-1])[:300]
            self.note(f"part {chunk['index']} failed (recording kept): {chunk['error']}")
            return
        out_dir = os.path.join(core.OUTPUT_DIR, chunk["jobId"])
        for n, clip in enumerate((job.get("result") or {}).get("clips") or [], 1):
            filename = os.path.basename(str(clip.get("video_url") or ""))
            src = os.path.join(out_dir, filename)
            if not filename or not os.path.isfile(src):
                continue
            title = str(clip.get("video_title_for_youtube_short") or "")
            roman = await asyncio.to_thread(romanize, title)
            base = f"part{chunk['index']:03d}_clip{n}_{slug(roman)}"
            try:
                shutil.copy2(src, os.path.join(self.dir, base + ".mp4"))
                _write_json(os.path.join(self.dir, base + ".json"), {
                    "title": title,
                    "title_roman": roman,
                    "hook": clip.get("viral_hook_text"),
                    "tiktok": clip.get("video_description_for_tiktok"),
                    "instagram": clip.get("video_description_for_instagram"),
                    "why": clip.get("why"),
                    "score": clip.get("predicted_score"),
                    "part": chunk["index"],
                    "part_recorded_at": chunk.get("recordedAt"),
                    "start_in_part": clip.get("start"),
                    "end_in_part": clip.get("end"),
                    "source": self.url,
                })
            except OSError as e:
                self.note(f"could not save a clip: {e}")
                continue
            chunk["clips"] += 1
            self.clips.append({**clip, "file": base + ".mp4", "title": title, "part": chunk["index"],
                               "videoUrl": f"/videos/{chunk['jobId']}/{filename}",
                               "jobId": chunk["jobId"], "clipIndex": n - 1})
        self._drop(chunk["file"])
        chunk["status"] = "done"
        self.note(f"part {chunk['index']} done: {chunk['clips']} clip(s) saved")

    async def tick(self):
        if self.state == "scheduled" and time.time() >= (self.start_at or 0):
            self.state = "waiting"
            self.note("start time reached")
        if self.state == "waiting":
            await self.try_start()
        elif self.state == "recording":
            self.collect()
            if _free_gb(self.dir) < MIN_FREE_GB:
                self.want_recording = False
                await self.stop_recorder()
                self.state = "paused"
                self.error = f"paused: less than {MIN_FREE_GB:g} GB free on disk"
                self.note(self.error)
            else:
                await self.supervise_recorder()
        current = self.clipping()
        if current is not None and current["status"] == "clipping":
            await self.harvest(current)
            current = self.clipping()
        if current is None and self.state in ("waiting", "recording", "ended"):
            nxt = next((c for c in self.chunks if c["status"] == "recorded"), None)
            if nxt is not None:
                await self.submit(nxt)
            elif self.state == "ended" and self.recorder is None:
                self.finish("finished")

    def finish(self, state):
        self.state = state
        self.note(state)
        _registry_set(self.id, None)
        shutil.rmtree(self.state_dir, ignore_errors=True)
        try:
            os.rmdir(self.chunk_dir)
        except OSError:
            pass

    async def loop(self):
        try:
            while self.state in ACTIVE:
                try:
                    await self.tick()
                except Exception as e:
                    self.note(f"error: {e}")
                self.save()
                await asyncio.sleep(POLL_SECONDS)
        finally:
            if self.recorder is not None:
                await self.stop_recorder(graceful=False)
            if self.state not in ACTIVE:
                _registry_set(self.id, None)
            self.save()

    async def pause(self):
        if self.state != "recording":
            raise HTTPException(status_code=409, detail=f"Session is {self.state}")
        self.state = "paused"
        self.want_recording = False
        await self.stop_recorder()
        self.note("paused")

    async def resume(self):
        if self.state != "paused":
            raise HTTPException(status_code=409, detail=f"Session is {self.state}")
        self.error = None
        self.want_recording = True
        self.state = "waiting"
        self.next_try = 0.0
        self.note("resumed")

    async def stop(self):
        if self.state not in ACTIVE:
            raise HTTPException(status_code=409, detail=f"Session is {self.state}")
        self.state = "stopped"
        self.want_recording = False
        await self.stop_recorder(graceful=False)
        current = self.clipping()
        if current is not None:
            if current.get("jobId"):
                asyncio.create_task(_cancel_until_done(current["jobId"]))
            current["status"] = "kept"
        for c in self.chunks:
            if c["status"] == "recorded":
                c["status"] = "kept"
        self.finish("stopped")
        self.save()


async def _cancel_until_done(job_id):
    core = _core()
    for _ in range(300):
        job = core.jobs.get(job_id) or {}
        if job.get("status") not in ("queued", "processing"):
            return
        core.cancel_job(job_id, "Cancelled: live clipping stopped.")
        await asyncio.sleep(2)


def resume_all():
    for session_id, session_dir in _read_registry().items():
        if session_id in _sessions:
            continue
        try:
            with open(os.path.join(session_dir, "session.json"), encoding="utf-8") as f:
                data = json.load(f)
            session = LiveSession.restore(data)
        except Exception as e:
            print(f"[live] could not resume {session_id}: {e}")
            continue
        _sessions[session.id] = session
        session.task = asyncio.create_task(session.loop())


def _guard(request: Request):
    if _core().BILLING_ENABLED:
        raise HTTPException(status_code=404, detail="Not found")
    origin = request.headers.get("origin")
    if origin and (urlparse(origin).hostname or "") not in _LOCAL_HOSTS:
        raise HTTPException(status_code=403, detail="Only the local dashboard can use this")


def _session(session_id):
    s = _sessions.get(session_id)
    if s is None:
        raise HTTPException(status_code=404, detail="Unknown live session")
    return s


def _drives():
    if os.name != "nt":
        return ["/"]
    return [f"{d}:\\" for d in string.ascii_uppercase if os.path.isdir(f"{d}:\\")]


@router.get("/api/watermark")
async def get_watermark(request: Request):
    _guard(request)
    import text_watermark
    return text_watermark.load()


@router.put("/api/watermark")
async def put_watermark(request: Request):
    _guard(request)
    import text_watermark
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Send the settings as JSON")
    if not isinstance(body, dict):
        raise HTTPException(status_code=400, detail="Send the settings as JSON")
    if body.get("enabled") and not " ".join(str(body.get("text") or "").split()):
        raise HTTPException(status_code=400, detail="Enter the watermark text")
    return text_watermark.save(body)


def _probe(url):
    import ssrf_guard
    import yt_dlp
    opts = {"quiet": True, "no_warnings": True, "noplaylist": True,
            "skip_download": True, "cachedir": False}
    try:
        with ssrf_guard.guarded(), yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(url, download=False, process=False)
    except Exception as e:
        if is_upcoming_error(e):
            return {"live": True, "status": "is_upcoming", "title": None}
        return {"live": False, "status": None, "title": None}
    status = (info or {}).get("live_status")
    if status is None and (info or {}).get("is_live"):
        status = "is_live"
    return {"live": status in ("is_live", "is_upcoming"), "status": status,
            "title": (info or {}).get("title")}


@router.get("/api/live/probe")
async def live_probe(request: Request, url: str = ""):
    _guard(request)
    url = url.strip()
    if not url:
        return {"live": False, "status": None, "title": None}
    try:
        await _core()._validate_source_url(url)
    except HTTPException:
        return {"live": False, "status": None, "title": None}
    return await asyncio.to_thread(_probe, url)


@router.get("/api/live/{session_id}/recording/{index}")
async def live_recording(request: Request, session_id: str, index: int):
    from fastapi.responses import FileResponse
    _guard(request)
    s = _session(session_id)
    chunk = next((c for c in s.chunks if c.get("index") == index), None)
    path = (chunk or {}).get("file") or ""
    if not chunk or not os.path.isfile(path) or os.path.dirname(os.path.abspath(path)) != os.path.abspath(s.chunk_dir):
        raise HTTPException(status_code=404, detail="Recording not on disk")
    return FileResponse(path, media_type="video/mp4")


@router.get("/api/live/browse")
async def live_browse(request: Request, path: str = ""):
    _guard(request)
    path = os.path.expanduser(path.strip())
    if not path:
        return {"path": "", "parent": None, "dirs": _drives(), "home": os.path.expanduser("~")}
    if not os.path.isabs(path) or not os.path.isdir(path):
        raise HTTPException(status_code=400, detail="Not a folder")
    path = os.path.abspath(path)
    try:
        names = sorted((e.name for e in os.scandir(path)
                        if e.is_dir(follow_symlinks=False)
                        and not e.name.startswith((".", "$"))), key=str.lower)
    except OSError as e:
        raise HTTPException(status_code=400, detail=f"Cannot open this folder: {e.strerror or e}")
    parent = os.path.dirname(path)
    return {"path": path, "parent": "" if parent == path else parent,
            "dirs": names[:500], "home": os.path.expanduser("~")}


@router.get("/api/live")
async def live_list(request: Request):
    _guard(request)
    return {"defaultChunkMinutes": DEFAULT_CHUNK_MINUTES,
            "sessions": [s.public() for s in sorted(_sessions.values(),
                                                     key=lambda x: -x.started_at)]}


@router.post("/api/live/start")
async def live_start(request: Request):
    _guard(request)
    try:
        body = await request.json()
    except Exception:
        body = {}
    url = str(body.get("url") or "").strip()
    save_dir = os.path.expanduser(str(body.get("save_dir") or "").strip())
    language = str(body.get("language") or "auto").strip().lower()
    try:
        chunk_minutes = float(body.get("chunk_minutes") or DEFAULT_CHUNK_MINUTES)
        start_at = float(body["start_at"]) if body.get("start_at") not in (None, "") else None
    except (TypeError, ValueError):
        raise HTTPException(status_code=400, detail="chunk_minutes and start_at must be numbers")
    if not url:
        raise HTTPException(status_code=400, detail="Paste the live stream URL")
    if not save_dir or not os.path.isabs(save_dir):
        raise HTTPException(status_code=400, detail="Choose a save folder (a full path, e.g. D:\\Clips)")
    if not 5 <= chunk_minutes <= 60:
        raise HTTPException(status_code=400, detail="chunk_minutes must be between 5 and 60")
    if language not in LANGUAGES + ("auto",):
        raise HTTPException(status_code=400, detail="language must be en, hi or auto")
    if start_at is not None and start_at > time.time() + MAX_SCHEDULE_DAYS * 86400:
        raise HTTPException(status_code=400, detail=f"Start time must be within {MAX_SCHEDULE_DAYS} days")
    await _core()._validate_source_url(url)
    session = LiveSession(url, save_dir, chunk_minutes, start_at,
                          language if language in LANGUAGES else None,
                          body.get("job_options"))
    try:
        os.makedirs(session.chunk_dir, exist_ok=True)
        os.makedirs(session.state_dir, exist_ok=True)
    except OSError as e:
        raise HTTPException(status_code=400, detail=f"Cannot create the save folder: {e}")
    if session.state == "scheduled":
        when = datetime.datetime.fromtimestamp(start_at).strftime("%d %b %H:%M")
        session.note(f"scheduled to start at {when}")
    _sessions[session.id] = session
    session.save()
    _registry_set(session.id, session.dir)
    session.task = asyncio.create_task(session.loop())
    return session.public()


@router.post("/api/live/{session_id}/pause")
async def live_pause(request: Request, session_id: str):
    _guard(request)
    s = _session(session_id)
    await s.pause()
    return s.public()


@router.post("/api/live/{session_id}/resume")
async def live_resume(request: Request, session_id: str):
    _guard(request)
    s = _session(session_id)
    await s.resume()
    return s.public()


@router.post("/api/live/{session_id}/stop")
async def live_stop(request: Request, session_id: str):
    _guard(request)
    s = _session(session_id)
    await s.stop()
    return s.public()
