"""Replay mode: run the whole pipeline on an ended YouTube stream.

For testing outputs and publishing without waiting for anyone to go live:

1. Read the stream's metadata (title, channel, length).
2. Download its chat replay up to the end of the chosen window. YouTube serves
   a replay sequentially at roughly 10-15x real time, so 30 minutes of
   stream take about 2-3 minutes; the download stops once the window is covered.
3. Run the same peak detector over the chat, in simulated time.
4. Download only the few seconds around each of the best peaks
   (``yt-dlp --download-sections``), never the whole multi-hour video.
5. Make each into a Short with the normal pipeline: transcript, hook, text,
   render and, if asked, posting.

Replays have no loudness signal (the full audio is never downloaded), so
peaks come from chat alone; a video without a chat replay has no peaks.
"""
from __future__ import annotations

import json
import re
import subprocess
import threading
import time
from dataclasses import replace
from pathlib import Path
from typing import Callable, List, Optional
from urllib.parse import parse_qs, urlparse

from . import procs
from .chat import ChatMessage, parse_youtube_replay_line
from .config import Channel, Settings
from .detector import Peak, PeakDetector

BASE = 1_000_000_000.0          # simulated wall clock: BASE + seconds into the video
MARGIN = 6.0                    # extra seconds fetched each side of a clip (keyframe slack)
_YT_HOSTS = {"youtube.com", "www.youtube.com", "m.youtube.com", "youtu.be", "music.youtube.com"}
_ID = re.compile(r"^[A-Za-z0-9_-]{11}$")


class ReplayError(RuntimeError):
    pass


def video_id(url: str) -> Optional[str]:
    """The 11-character id of a YouTube watch/live/shorts/youtu.be link, else None."""
    try:
        u = urlparse(str(url or "").strip())
    except ValueError:
        return None
    host = (u.hostname or "").lower()
    if u.scheme not in ("http", "https") or host not in _YT_HOSTS:
        return None
    if host == "youtu.be":
        cand = u.path.lstrip("/").split("/")[0]
    elif u.path == "/watch":
        cand = (parse_qs(u.query).get("v") or [""])[0]
    else:
        m = re.match(r"^/(?:live|shorts|embed)/([^/?#]+)", u.path)
        cand = m.group(1) if m else ""
    return cand if _ID.match(cand or "") else None


def find_peaks(messages: List[tuple], settings: Settings, start: float, end: float,
               max_clips: int) -> List[Peak]:
    """Peaks in [start, end] seconds, best first, at most ``max_clips``.

    ``Peak.ts`` is returned as seconds into the video.
    """
    det = PeakDetector(threshold=settings.threshold, warmup=settings.warmup_seconds,
                       cooldown=settings.cooldown_seconds, min_chat_rate=settings.min_chat_rate,
                       audio_weight=0.0, max_per_hour=10_000)
    for off, msg in messages:
        if start - PeakDetector.BASELINE <= off <= end + 30:
            det.add_message(ChatMessage(BASE + off, msg.user, msg.text))
    peaks = []
    for now in range(int(BASE + start), int(BASE + end + 30)):
        p = det.evaluate(now)
        if p and start <= p.ts - BASE <= end:
            peaks.append(replace(p, ts=p.ts - BASE))
    peaks.sort(key=lambda p: p.score, reverse=True)
    return peaks[:max_clips]


class Replay:
    """One replay run. ``run()`` blocks; ``cancel()`` may be called from another thread."""

    def __init__(self, settings: Settings, url: str, *, minutes: float = 30, start_minute: float = 0,
                 max_clips: int = 5, post: Optional[bool] = None, log: Callable[[str], None] = print,
                 service=None):
        vid = video_id(url)
        if not vid:
            raise ReplayError("not a YouTube video link")
        if not (1 <= minutes <= 600) or not (0 <= start_minute <= 6000):
            raise ReplayError("the window must be 1-600 minutes, starting within the video")
        if not (1 <= max_clips <= 20):
            raise ReplayError("max clips must be 1-20")
        self.settings = settings
        self.url = f"https://www.youtube.com/watch?v={vid}"
        self.vid = vid
        self.minutes = minutes
        self.start = start_minute * 60
        self.max_clips = max_clips
        self.post = post
        self.log = log
        self.svc = service
        self.work = settings.data_dir / "replay" / vid
        self.stop_event = threading.Event()
        self._proc: Optional[subprocess.Popen] = None
        self.step = ""
        self.progress = 0.0
        self.results: List[dict] = []
        self.meta: dict = {}

    def cancel(self):
        self.stop_event.set()
        procs.terminate(self._proc)

    def _check(self):
        if self.stop_event.is_set():
            raise ReplayError("cancelled")

    # ------------------------------------------------------------------ steps
    def fetch_meta(self) -> dict:
        self.step = "reading the video's details"
        r = procs.run([procs.ytdlp(), "--no-warnings", "--skip-download", "--no-playlist",
                       "-J", self.url], timeout=120)
        if r.returncode != 0:
            raise ReplayError("could not read the video: "
                              + (r.stderr or b"").decode("utf-8", "replace")[-300:])
        data = json.loads((r.stdout or b"{}").decode("utf-8", "replace"))
        self.meta = {
            "title": str(data.get("title") or ""), "channel": str(data.get("channel") or ""),
            "handle": str(data.get("uploader_id") or data.get("channel_id") or ""),
            "duration": float(data.get("duration") or 0),
            "live_status": str(data.get("live_status") or ""),
            "has_chat": "live_chat" in (data.get("subtitles") or {}),
        }
        if self.meta["live_status"] in ("is_live", "is_upcoming"):
            raise ReplayError("this stream is still live or upcoming: watch it live instead")
        return self.meta

    def fetch_chat(self, end: float) -> List[tuple]:
        """Chat replay messages up to ``end`` seconds, as (offset, ChatMessage)."""
        self.step = "downloading the chat replay"
        chat_dir = self.work / "chat"
        chat_dir.mkdir(parents=True, exist_ok=True)
        for old in chat_dir.glob("c*"):
            try:
                old.unlink()
            except OSError:
                pass  # still held by an earlier run's process; this run uses its own name
        stem = f"c{int(time.time() * 1000)}"
        self._proc = procs.popen(
            [procs.ytdlp(), "--no-warnings", "--quiet", "--skip-download", "--no-playlist",
             "--write-subs", "--sub-langs", "live_chat", "-o", str(chat_dir / f"{stem}.%(ext)s"),
             self.url],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        msgs: List[tuple] = []
        pending = b""
        fh = None
        last_off = 0.0
        last_report = 0.0
        try:
            while True:
                self._check()
                if fh is None:
                    for p in (chat_dir / f"{stem}.live_chat.json.part",
                              chat_dir / f"{stem}.live_chat.json"):
                        if p.exists():
                            fh = open(p, "rb")
                            break
                chunk = fh.read() if fh else b""
                if chunk:
                    pending += chunk
                    *lines, pending = pending.split(b"\n")
                    for ln in lines:
                        for off, m in parse_youtube_replay_line(ln.decode("utf-8", "replace")):
                            last_off = max(last_off, off)
                            if off <= end + 30:
                                msgs.append((off, m))
                    self.progress = min(1.0, last_off / max(end, 1))
                    if time.time() - last_report > 15:
                        last_report = time.time()
                        self.log(f"replay: chat downloaded up to {_mmss(last_off)} of {_mmss(end)} "
                                 f"({len(msgs)} messages)")
                if last_off > end + 30:
                    break
                if self._proc.poll() is not None and not chunk:
                    break
                time.sleep(0.5)
        finally:
            if fh:
                fh.close()
            procs.terminate(self._proc)
        return msgs

    def fetch_section(self, a: float, b: float, out_dir: Path) -> Path:
        """Only [a, b] seconds of the video, 720p or less, as one file."""
        out_dir.mkdir(parents=True, exist_ok=True)
        for old in out_dir.glob("raw.*"):
            old.unlink(missing_ok=True)
        cmd = [procs.ytdlp(), "--no-warnings", "--quiet", "--no-playlist",
               "-f", "bv*[height<=720][vcodec^=avc1]+ba[ext=m4a]/bv*[height<=720]+ba/b[height<=720]/b",
               "--download-sections", f"*{a:.2f}-{b:.2f}", "--merge-output-format", "mp4",
               "-o", str(out_dir / "raw.%(ext)s"), self.url]
        self._proc = procs.popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        try:
            _, err = self._proc.communicate(timeout=900)
        except subprocess.TimeoutExpired:
            procs.terminate(self._proc)
            raise ReplayError("downloading the clip timed out")
        self._check()
        files = [p for p in out_dir.glob("raw.*") if p.suffix in (".mp4", ".mkv", ".webm")]
        if self._proc.returncode != 0 or not files:
            raise ReplayError("downloading the clip failed: "
                              + (err or b"").decode("utf-8", "replace")[-300:])
        return files[0]

    # -------------------------------------------------------------------- run
    def run(self) -> List[dict]:
        from .service import ClipJob, LiveInfo, Service
        s = self.settings
        if self.post is not None:
            s = replace(s, post_enabled=bool(self.post))
        svc = self.svc or Service(s, log=self.log if hasattr(self.log, "since") else None)
        svc.settings = s
        meta = self.fetch_meta()
        dur = meta["duration"] or (self.start + self.minutes * 60)
        if self.start >= dur:
            raise ReplayError(f"the video is only {_mmss(dur)} long")
        end = min(dur, self.start + self.minutes * 60)
        self.log(f"replay: {meta['channel']} - {meta['title'][:80]} "
                 f"({_mmss(dur)}), analysing {_mmss(self.start)}-{_mmss(end)}")
        if not meta["has_chat"]:
            raise ReplayError("this video has no chat replay, so there is nothing to find peaks in")
        msgs = self.fetch_chat(end)
        self._check()
        self.log(f"replay: {len(msgs)} chat messages in the window")
        self.step = "finding peaks"
        peaks = find_peaks(msgs, s, self.start, end, self.max_clips)
        if not peaks:
            self.log("replay: no peaks found. Try a longer window or a lower threshold.")
            return []
        self.log(f"replay: {len(peaks)} peak(s) at "
                 + ", ".join(f"{_mmss(p.ts)} ({p.score})" for p in sorted(peaks, key=lambda p: p.ts)))
        handle = re.sub(r"[^A-Za-z0-9_.@-]", "", meta["handle"])[:60] or self.vid
        ch = Channel(platform="youtube", name=handle, label=meta["channel"][:60] or handle)
        info = LiveInfo(True, title=meta["title"], category="YouTube replay")
        for i, peak in enumerate(sorted(peaks, key=lambda p: p.ts), 1):
            self._check()
            clip_start = max(0.0, peak.ts - s.pre_seconds)
            clip_end = min(dur, peak.ts + s.post_seconds)
            if clip_end - clip_start < s.min_seconds:
                clip_end = min(dur, clip_start + s.min_seconds)
            a, b = max(0.0, clip_start - MARGIN), min(dur, clip_end + MARGIN)
            self.step = f"clip {i}/{len(peaks)}: downloading {_mmss(clip_start)}-{_mmss(clip_end)}"
            self.log(f"replay: {self.step}")
            job_dir = s.data_dir / "clips" / f"replay_{self.vid}" / f"t{int(peak.ts):06d}"
            try:
                raw = self.fetch_section(a, b, job_dir)
            except ReplayError as e:
                self.log(f"replay: {e}")
                self.results.append({"at": peak.ts, "error": str(e)})
                continue
            job = ClipJob(ch, replace(peak, ts=time.time()), clip_start, clip_end, info,
                          s.style_for(ch, i), job_dir, desc_style=s.description_for(i))
            job.raw, job.raw_offset, job.duration = raw, clip_start - a, clip_end - clip_start
            self.step = f"clip {i}/{len(peaks)}: making the Short"
            try:
                out = svc.process(job)
                self.results.append({"at": peak.ts, "file": str(out), "dir": str(job_dir)})
            except Exception as e:
                self.log(f"replay: clip {i} failed: {e}")
                self.results.append({"at": peak.ts, "error": str(e)})
        self.step = "done"
        ok = sum(1 for r in self.results if "file" in r)
        self.log(f"replay: done, {ok} Short(s) made")
        return self.results


def _mmss(sec: float) -> str:
    sec = int(max(0, sec))
    h, rem = divmod(sec, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"
