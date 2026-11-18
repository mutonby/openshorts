"""The watcher: who is live, what to record, when a moment happened, and the clip queue.

Threads:
- main loop: live checks every ``check_interval`` (in a small pool), asks each
  detector once a second, cuts the raw moment out of the buffer as soon as
  the seconds after it are recorded.
- per live channel: a recorder, a chat reader, and a loudness meter fed by
  the recorder's chunks.
- one clip worker: transcribe -> write the text -> captions -> render ->
  post, one clip at a time and at low priority, so recording never starves.
"""
from __future__ import annotations

import csv
import json
import queue
import shutil
import subprocess
import threading
import time
import traceback
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional

from . import captions, card, music, poster, procs, publish, render, transcribe, viral
from .chat import TwitchChatReader, YouTubeChatReader
from .config import Channel, Settings, matches_game
from .detector import Peak, PeakDetector
from .recorder import (BEHIND, Recorder, Segment, SegmentBuffer, quality_steps,
                       segment_loudness)

MIN_FREE_GB = 2.0


def keep_awake(on: bool) -> None:
    """Windows: keep the PC from sleeping while this thread watches (sleep cuts
    every recording). The screen may still turn off. No-op elsewhere."""
    if procs.IS_WINDOWS:
        try:
            import ctypes
            ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
            flags = ES_CONTINUOUS | (ES_SYSTEM_REQUIRED if on else 0)
            ctypes.windll.kernel32.SetThreadExecutionState(flags)
        except (AttributeError, OSError):
            pass


def _log_line(msg: str) -> str:
    return f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"


class Logger:
    """Prints, appends to the log file and keeps the last lines for the web UI."""

    KEEP = 2000

    def __init__(self, path: Optional[Path]):
        self.path = path
        self._lock = threading.Lock()
        self.seq = 0
        self.boot = f"{time.time():.6f}"  # a page that saw another Logger starts over
        self.lines: "deque[tuple]" = deque(maxlen=self.KEEP)
        # earlier sessions' last lines, so the UI does not open on an empty log
        if path and Path(path).is_file():
            try:
                with open(path, "rb") as fh:
                    fh.seek(max(0, Path(path).stat().st_size - 200_000))
                    tail = fh.read().decode("utf-8", "replace").splitlines()[1:]
                for ln in tail[-300:]:
                    self.seq += 1
                    self.lines.append((self.seq, ln))
            except OSError:
                pass

    def since(self, seq: int) -> List[tuple]:
        """(seq, line) entries newer than ``seq``."""
        with self._lock:
            return [e for e in self.lines if e[0] > seq]

    def __call__(self, msg: str):
        line = _log_line(msg)
        with self._lock:
            self.seq += 1
            self.lines.append((self.seq, line))
            try:
                print(line, flush=True)
            except UnicodeEncodeError:
                print(line.encode("ascii", "replace").decode(), flush=True)
            if self.path:
                try:
                    with open(self.path, "a", encoding="utf-8") as fh:
                        fh.write(line + "\n")
                except OSError:
                    pass


@dataclass
class LiveInfo:
    live: bool
    title: str = ""
    category: str = ""
    author: str = ""
    error: str = ""


TRANSIENT = ("Unable to open URL", "Connection aborted", "RemoteDisconnected", "timed out",
             "Max retries exceeded", "Read timed out")


def check_live(channel: Channel, timeout: float = 60, retries: int = 1) -> LiveInfo:
    """Is the channel live, with what title/category. A dropped connection is
    retried once before it is reported (YouTube resets connections now and then)."""
    info = _check_live_once(channel, timeout)
    while retries > 0 and not info.live and any(t in info.error for t in TRANSIENT):
        retries -= 1
        time.sleep(5)
        info = _check_live_once(channel, timeout)
    return info


def _check_live_once(channel: Channel, timeout: float) -> LiveInfo:
    cmd = procs.streamlink_cmd() + ["--json", channel.url]
    try:
        r = procs.run(cmd, timeout=timeout)
    except subprocess.TimeoutExpired:
        return LiveInfo(False, error="live check timed out")
    except OSError as e:
        return LiveInfo(False, error=str(e))
    try:
        data = json.loads((r.stdout or b"").decode("utf-8", "replace") or "{}")
    except ValueError:
        return LiveInfo(False, error="unreadable streamlink answer")
    meta = data.get("metadata") or {}
    return LiveInfo(live=bool(data.get("streams")), title=str(meta.get("title") or ""),
                    category=str(meta.get("category") or ""), author=str(meta.get("author") or ""),
                    error="" if data.get("streams") else str(data.get("error") or ""))


@dataclass
class ClipJob:
    channel: Channel
    peak: Peak
    start: float
    end: float
    info: LiveInfo
    style: str
    dir: Path
    raw: Optional[Path] = None
    raw_offset: float = 0.0
    duration: float = 0.0
    created: float = field(default_factory=time.time)
    desc_style: str = "normal"
    moment: Optional[float] = None   # seconds into the clip where the peak is


class ChannelWatch:
    def __init__(self, svc: "Service", channel: Channel, info: LiveInfo):
        self.svc = svc
        self.channel = channel
        self.info = info
        s = svc.settings
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.dir = s.data_dir / "buffer" / channel.key.replace(":", "_") / stamp
        self.buffer = SegmentBuffer(s.buffer_seconds)
        self.detector = PeakDetector(threshold=s.threshold, warmup=s.warmup_seconds,
                                     cooldown=s.cooldown_seconds, min_chat_rate=s.min_chat_rate,
                                     audio_weight=s.audio_weight, max_per_hour=s.max_clips_per_hour)
        self.qualities = quality_steps(s.quality)
        self.level = 0
        self.recorder = self._new_recorder()
        self.chat = None
        if channel.platform == "twitch":
            self.chat = TwitchChatReader(channel.name, self.detector.add_message, log=svc.log)
        elif channel.platform == "youtube":
            self.chat = YouTubeChatReader(channel.chat_url, self.dir / "chat",
                                          self.detector.add_message, log=svc.log)
        self._loud_q: "queue.Queue[Segment]" = queue.Queue()
        self._loud_thread = threading.Thread(target=self._loudness_loop, daemon=True,
                                             name=f"loud-{channel.key}")
        self.pending: List[ClipJob] = []
        self.started = time.time()
        self.last_status = 0.0

    def _new_recorder(self) -> Recorder:
        return Recorder(self.channel.url, self.dir / f"q{self.level}", self.buffer,
                        self.qualities[self.level], on_segment=self._on_segment, log=self.svc.log)

    def can_step_down(self) -> bool:
        return self.recorder.exit_reason == BEHIND and self.level + 1 < len(self.qualities)

    def step_down(self):
        """Record again one quality lower; chat history and the detector carry on."""
        self.recorder.join(timeout=15)
        old_dir = self.recorder.out_dir
        self.level += 1
        # chunks of two resolutions cannot be joined without re-encoding: start a new buffer
        self.buffer = SegmentBuffer(self.svc.settings.buffer_seconds)
        self.recorder = self._new_recorder()
        self.recorder.start()
        shutil.rmtree(old_dir, ignore_errors=True)

    def start(self):
        self.recorder.start()
        if self.chat:
            self.chat.start()
        self._loud_thread.start()

    def stop(self):
        self.recorder.stop()
        if self.chat:
            self.chat.stop()
        self._loud_q.put(None)  # type: ignore[arg-type]
        if self.recorder.is_alive():
            self.recorder.join(timeout=15)
        if self._loud_thread.is_alive():
            self._loud_thread.join(timeout=30)
        # pending moments were already copied out of the buffer by now
        shutil.rmtree(self.dir, ignore_errors=True)

    @property
    def alive(self) -> bool:
        return self.recorder.is_alive()

    def _on_segment(self, seg: Segment):
        self._loud_q.put(seg)

    def _loudness_loop(self):
        while True:
            seg = self._loud_q.get()
            if seg is None:
                return
            for t, db in segment_loudness(seg.path):
                self.detector.add_loudness(seg.start + t, db)


class Service:
    def __init__(self, settings: Settings, log: Optional[Logger] = None):
        self.settings = settings
        settings.data_dir.mkdir(parents=True, exist_ok=True)
        self.log = log or Logger(settings.data_dir / "livewatch.log")
        self.watches: Dict[str, ChannelWatch] = {}
        self.next_check: Dict[str, float] = {}
        self.checking: Dict[str, object] = {}
        self.pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="live-check")
        self.jobs: "queue.Queue[Optional[ClipJob]]" = queue.Queue()
        self.stop_event = threading.Event()
        self.clip_counter = 0
        self.ledger = poster.PostLedger(settings.data_dir / "post_ledger.json")
        self.worker = threading.Thread(target=self._worker, daemon=True, name="clip-worker")
        self._post_queue: List[tuple] = []
        self._post_lock = threading.Lock()
        self._last_post_at: Optional[datetime] = None
        self.poster = threading.Thread(target=self._poster, daemon=True, name="poster")
        self.index_path = settings.data_dir / "clips.csv"
        # for the web UI
        self.started_at: Optional[float] = None
        self.finished = threading.Event()
        self.state: Dict[str, str] = {c.key: "not checked yet" for c in settings.channels}
        self.last_info: Dict[str, LiveInfo] = {}
        self.history: Dict[str, deque] = {}
        self.current_step = ""
        self.counts = {"peaks": 0, "clips": 0, "posted": 0, "failed": 0}

    # ------------------------------------------------------------- control
    def stop(self):
        """Ask the loop to stop; queued clips are still finished."""
        self.stop_event.set()

    def check_now(self):
        for key in self.next_check:
            self.next_check[key] = 0

    def status(self) -> dict:
        """A snapshot for the web UI (read from other threads; values only)."""
        s = self.settings
        chans = []
        for ch in s.channels:
            key = ch.key
            w = self.watches.get(key)
            info = w.info if w else self.last_info.get(key)
            item = {
                "key": key, "platform": ch.platform, "name": ch.name, "display": ch.display,
                "url": ch.url, "state": self.state.get(key, ""),
                "live": bool(info and info.live), "recording": w is not None,
                "category": info.category if info else "", "title": info.title if info else "",
                "game_filter": ch.game_filter,
            }
            if w:
                sc, cz, az = w.detector.last_score
                warm = max(0.0, s.warmup_seconds - (time.time() - (w.detector._first or time.time())))
                item.update({
                    "quality": w.qualities[w.level].split(",")[0],
                    "buffer_seconds": round(len(w.buffer) * 10),
                    "chat_messages": w.chat.count if w.chat else None,
                    "score": round(sc, 2), "chat_z": round(cz, 2), "audio_z": round(az, 2),
                    "warming_up": round(warm), "pending": len(w.pending),
                    "since": w.started,
                    "history": [round(v, 2) for _, v in list(self.history.get(key, []))[-180:]],
                })
            chans.append(item)
        return {
            "running": not self.finished.is_set() and self.started_at is not None,
            "stopping": self.stop_event.is_set() and not self.finished.is_set(),
            "started_at": self.started_at, "channels": chans,
            "queue": self.jobs.qsize(), "current_step": self.current_step,
            "counts": dict(self.counts), "threshold": s.threshold,
            "encoder": render.pick_encoder(s.encoder),
            "ai": "gemini" if s.gemini_api_key else "offline",
            "posting": bool(s.post_enabled),
            "posts_queued": len(self._post_queue),
        }

    # ------------------------------------------------------------ lifecycle
    def run(self, duration: Optional[float] = None):
        s = self.settings
        if not s.channels:
            self.log("no channels in the config: nothing to watch")
            self.finished.set()
            return
        self.log(f"livewatch watching {len(s.channels)} channel(s): "
                 + ", ".join(c.key for c in s.channels))
        self.log(f"encoder={render.pick_encoder(s.encoder)} whisper={s.whisper_model} "
                 f"ai={'gemini' if s.gemini_api_key else 'offline templates'} "
                 f"posting={'on' if s.post_enabled else 'off (review)'}")
        if s.post_enabled:
            routes = publish.route(s.post_platforms, s.youtube_via, s.instagram_via, s.others_via)
            self.log("posting: " + ", ".join(f"{n} via {v}" for n, v in routes.items()))
            for why in publish.missing(routes, publish.credentials(), s.upload_post_user):
                self.log(f"posting not ready: {why}")
        self._cleanup_old_buffers()
        self.started_at = time.time()
        self.worker.start()
        self._requeue_from_disk()
        self.poster.start()
        t_end = time.time() + duration if duration else None
        try:
            keep_awake(True)
            last_error = 0.0
            while not self.stop_event.is_set():
                if t_end and time.time() >= t_end:
                    break
                try:
                    self._tick()
                except Exception as e:  # one bad tick must never end a day of watching
                    if time.time() - last_error > 60:
                        self.log(f"watcher error (it keeps running): {type(e).__name__}: {e}")
                        self.log(traceback.format_exc(limit=4))
                    last_error = time.time()
                self.stop_event.wait(1.0)
        except KeyboardInterrupt:
            self.log("stopping (Ctrl+C)")
        finally:
            keep_awake(False)
            self.shutdown()
            self.log("stopped")
            self.finished.set()

    def shutdown(self):
        self.stop_event.set()
        for w in list(self.watches.values()):
            self._flush_pending(w, force=True)
            w.stop()
        self.watches.clear()
        self.pool.shutdown(wait=False, cancel_futures=True)
        self.jobs.put(None)
        if self.worker.is_alive():
            self.log("finishing the clips in the queue...")
            self.worker.join()

    def _cleanup_old_buffers(self):
        root = self.settings.data_dir / "buffer"
        if root.is_dir():
            for d in root.glob("*/*"):
                if d.is_dir() and time.time() - d.stat().st_mtime > 600:
                    shutil.rmtree(d, ignore_errors=True)

    # ----------------------------------------------------------------- tick
    def _tick(self):
        now = time.time()
        s = self.settings
        for ch in s.channels:
            key = ch.key
            fut = self.checking.get(key)
            if fut is not None and fut.done():
                del self.checking[key]
                try:
                    info = fut.result()
                except Exception as e:
                    info = LiveInfo(False, error=str(e))
                self._on_live_info(ch, info)
            if key not in self.checking and now >= self.next_check.get(key, 0):
                self.next_check[key] = now + s.check_interval
                self.checking[key] = self.pool.submit(check_live, ch)

        for key, w in list(self.watches.items()):
            peak = w.detector.evaluate(now)
            self.history.setdefault(key, deque(maxlen=600)).append((now, w.detector.last_score[0]))
            if peak:
                self._on_peak(w, peak)
            self._flush_pending(w)
            if now - w.last_status > 300:
                w.last_status = now
                sc, cz, az = w.detector.last_score
                chat_n = w.chat.count if w.chat else 0
                self.log(f"{key}: recording {len(w.buffer)} chunks at "
                         f"{w.qualities[w.level].split(',')[0]}, {chat_n} chat msgs, "
                         f"{len(w.detector._loud)}s of audio levels, "
                         f"score {sc:.1f} (chat {cz:.1f} / audio {az:.1f})")
            if not w.alive and w.can_step_down():
                self._flush_pending(w, force=True)
                w.step_down()
                self.log(f"{key}: {BEHIND}; now recording at {w.qualities[w.level].split(',')[0]}")
                continue
            if not w.alive:
                self.log(f"{key}: recorder stopped ({w.recorder.exit_reason})")
                self._flush_pending(w, force=True)
                w.stop()
                del self.watches[key]
                self.state[key] = "stream ended"
                self.next_check[key] = now + 20

    def _set_state(self, key: str, state: str, log: bool = True):
        """Remember a channel's state; log it only when it changes."""
        if self.state.get(key) != state:
            self.state[key] = state
            if log:
                self.log(f"{key}: {state}")

    def _on_live_info(self, ch: Channel, info: LiveInfo):
        key = ch.key
        self.last_info[key] = info
        w = self.watches.get(key)
        game_ok = ch.platform == "youtube" or matches_game(info.category, ch.game_filter)
        if w:
            if info.live:
                w.info = info
                if not game_ok:
                    self._flush_pending(w, force=True)
                    w.stop()
                    del self.watches[key]
                    self._set_state(key, f"paused: now playing {info.category or '?'}")
            return
        if not info.live:
            err = info.error or ""
            if err and "No playable streams" not in err and "No plugin" not in err:
                self._set_state(key, f"check failed: {err[:100]}")
            else:
                self._set_state(key, "offline", log=False)
            return
        if not game_ok:
            self._set_state(key, f"live, waiting for {', '.join(ch.game_filter)} "
                                 f"(playing {info.category or '?'})")
            return
        if len(self.watches) >= self.settings.max_live_channels:
            self._set_state(key, f"live, not recording: already {len(self.watches)} "
                                 f"recordings (max_live_channels)")
            return
        free_gb = shutil.disk_usage(self.settings.data_dir).free / 1e9
        if free_gb < MIN_FREE_GB:
            self._set_state(key, f"live, not recording: only {free_gb:.1f} GB free on disk")
            return
        self.state[key] = "recording"
        self.log(f"{key}: LIVE - {info.category or '?'} - {info.title[:80]}")
        w = ChannelWatch(self, ch, info)
        self.watches[key] = w
        w.start()

    def _on_peak(self, w: ChannelWatch, peak: Peak):
        s = self.settings
        start = peak.ts - s.pre_seconds
        end = peak.ts + s.post_seconds
        if end - start < s.min_seconds:
            end = start + s.min_seconds
        self.clip_counter += 1
        style = s.style_for(w.channel, self.clip_counter)
        stamp = datetime.fromtimestamp(peak.ts).strftime("%Y%m%d-%H%M%S")
        job_dir = s.data_dir / "clips" / w.channel.key.replace(":", "_") / stamp
        job = ClipJob(w.channel, peak, start, end, w.info, style, job_dir,
                      desc_style=s.description_for(self.clip_counter), moment=peak.ts - start)
        w.pending.append(job)
        self.counts["peaks"] += 1
        self.log(f"{w.channel.key}: PEAK score {peak.score} ({peak.reason}) "
                 f"chat: {' | '.join(peak.messages[:3])[:120]}")

    def _flush_pending(self, w: ChannelWatch, force: bool = False):
        """Cut each pending moment from the buffer once the seconds after it exist."""
        now = time.time()
        keep = []
        for job in w.pending:
            ready = w.buffer.latest_end >= job.end + 0.5
            waited = now - job.created > self.settings.post_seconds + 60
            if not (ready or force or waited):
                keep.append(job)
                continue
            segs, offset, real = w.buffer.select(job.start, job.end)
            if real < self.settings.min_seconds:
                self.log(f"{w.channel.key}: moment dropped, only {real:.0f}s recorded around it")
                continue
            try:
                job.raw = render.extract_raw(segs, job.dir / "raw.ts")
            except render.RenderError as e:
                self.log(f"{w.channel.key}: {e}")
                continue
            job.raw_offset, job.duration = offset, real
            self.jobs.put(job)
            self.log(f"{w.channel.key}: clip queued ({real:.0f}s, {self.jobs.qsize()} in queue)")
        w.pending = keep

    # --------------------------------------------------------------- worker
    def _worker(self):
        while True:
            job = self.jobs.get()
            if job is None:
                return
            try:
                self.process(job)
            except Exception as e:
                self.counts["failed"] += 1
                self.log(f"{job.channel.key}: clip failed: {e}")
                self.log(traceback.format_exc(limit=3))
            finally:
                self.current_step = ""

    def process(self, job: ClipJob) -> Path:
        s = self.settings
        t0 = time.time()
        d = job.dir
        d.mkdir(parents=True, exist_ok=True)
        words, lang = [], ""
        who = job.channel.key
        if s.whisper_model != "off" and (s.captions or s.gemini_api_key):
            self.current_step = f"{who}: transcribing"
            try:
                wav = render.extract_audio(job.raw, d / "audio.wav", job.raw_offset, job.duration)
                words, lang = transcribe.transcribe(str(wav), s.whisper_model, job.channel.language)
                wav.unlink(missing_ok=True)
            except Exception as e:
                self.log(f"{job.channel.key}: transcription skipped: {e}")
        frames = []
        if s.gemini_api_key:  # the writer looks at the clip, so its hook matches it
            moment = job.moment if job.moment is not None else s.pre_seconds
            frames = render.extract_frames(
                job.raw, render.frame_times(job.raw_offset, job.duration, moment))
        ctx = viral.ClipContext(
            streamer=job.channel.display, platform=job.channel.platform,
            category=job.info.category, stream_title=job.info.title,
            transcript=" ".join(w.text for w in words), chat=job.peak.messages,
            language=lang, seed=f"{job.channel.key}{job.peak.ts}", frames=frames)
        self.current_step = f"{who}: writing hook, title and hashtags"
        text = viral.write_text(ctx, job.style, s.gemini_api_key, s.ai_model,
                                base_tags=s.base_hashtags, channel_tags=job.channel.tags,
                                desc_style=job.desc_style)
        layout = s.layout if s.layout == "card" else ("facecam" if job.channel.facecam else "blur")
        card_spec, cap_y = None, None
        if layout == "card":
            info = render.probe(job.raw)
            aspect = (info["width"] / info["height"]) if info["width"] and info["height"] else 16 / 9
            brand = card.Brand(name=s.brand_name or job.channel.display, handle=s.brand_handle,
                               verified=s.brand_verified,
                               logo=s.logo_path if s.logo_path.is_file() else None)
            card_layout = card.render_card(text.hook if s.hook else "", brand, d / "card.png", aspect)
            card_spec, cap_y = (d / "card.png", card_layout), card_layout.caption_y
        ass = d / "captions.ass"
        ass.write_text(captions.build_ass(words, text.hook, job.duration, job.style, layout,
                                          s.captions, s.hook and layout != "card", cap_y=cap_y),
                       encoding="utf-8")
        mix = None
        if music.wants_music(s.music_mode, len(words), job.duration) and s.music_dir:
            mix = music.Library(s.music_dir).pick(job.duration, s.music_volume_db,
                                                   seed=f"{job.channel.key}{job.peak.ts}")
            if mix is None:
                self.log(f"{who}: music is on but the library is empty ({s.music_dir})")
        cta_spec = None
        if s.cta:
            try:
                card.render_cta(s.cta_text, d / "cta.png",
                                s.logo_path if s.logo_path.is_file() else None)
                cta_spec = (d / "cta.png", card.CTA_Y)
            except Exception as e:  # noqa: BLE001 - a badge never costs a clip
                self.log(f"{who}: follow badge skipped: {e}")
        self.current_step = f"{who}: rendering the Short"
        out = render.render_short(job.raw, d / "short.mp4", job.raw_offset, job.duration,
                                  layout=layout, facecam=job.channel.facecam,
                                  ass_path=ass if (s.captions or s.hook) else None,
                                  fps=s.fps, encoder=s.encoder, music=mix, card=card_spec,
                                  cta=cta_spec)
        if card_spec:
            card_spec[0].unlink(missing_ok=True)
        if cta_spec:
            cta_spec[0].unlink(missing_ok=True)
        if not s.keep_raw_clips:
            job.raw.unlink(missing_ok=True)
        posts = viral.platform_posts(text, s.post_platforms)
        meta = {
            "channel": job.channel.key, "streamer": job.channel.display,
            "url": job.channel.url, "moment": datetime.fromtimestamp(job.peak.ts, timezone.utc).isoformat(),
            "duration": round(job.duration, 1), "score": job.peak.score, "reason": job.peak.reason,
            "chat": job.peak.messages, "category": job.info.category,
            "stream_title": job.info.title, "language": lang,
            "transcript": ctx.transcript, "text": text.to_dict(), "posts": posts,
            "music": mix.path.name if mix else None,
            "posted": None,
        }
        self.log(f"{job.channel.key}: Short ready in {time.time() - t0:.0f}s -> {out}")
        self.log(f"   [{text.style}/{text.source}] hook: {text.hook} | title: {text.title}")
        if text.social_description:
            self.log(f"   [{text.description_style} description] {text.social_description[:100]}")
        self.counts["clips"] += 1
        meta["posted"] = None
        write_meta(d, meta)  # visible in the UI before the upload finishes
        self.current_step = f"{who}: posting"
        meta["posted"] = self.maybe_post(d, text)
        if (meta["posted"] or {}).get("networks"):
            meta["posted"] = self._schedule_retry(d, meta["posted"])
        if (meta["posted"] or {}).get("status") in ("posted", "partial"):
            self.counts["posted"] += 1
        write_meta(d, meta)
        self._index(meta, out)
        return out

    def maybe_post(self, clip_dir: Path, text: viral.ClipText) -> Optional[dict]:
        """Post now, queue for later (spacing), or say why the clip is held."""
        s = self.settings
        if not s.post_enabled:
            return {"status": "held", "why": "automatic posting is off (Settings > Posting)"}
        if s.skip_spoilers and text.spoiler:
            self.log("   held: looks like a story spoiler (Rockstar removes those)")
            return {"status": "held", "why": "spoiler"}
        if text.source == "gemini" and text.postworthy < 4:
            self.log(f"   held: rated {text.postworthy}/10 by the AI")
            return {"status": "held", "why": f"postworthy {text.postworthy}"}
        now = datetime.now(timezone.utc)
        when = self.ledger.slot(now, timedelta(minutes=s.min_gap_minutes), s.daily_cap)
        if when is None:
            self.log("   held: daily_cap reached")
            return {"status": "held", "why": "daily cap"}
        if when > now + timedelta(minutes=1):
            self.ledger.record(when)  # the slot is taken now, so the next clip queues after it
            self._enqueue(clip_dir, when)
            self.log(f"   queued: posts at {when.astimezone():%H:%M} (min_gap_minutes)")
            return {"status": "queued", "at": when.isoformat()}
        self.ledger.record(now)
        return deliver(clip_dir, s, self.log)

    # ----------------------------------------------------------- post queue
    def _enqueue(self, clip_dir: Path, when: datetime, only: Optional[List[str]] = None):
        with self._post_lock:
            self._post_queue.append((when, Path(clip_dir), only))
            self._post_queue.sort(key=lambda x: x[0])

    def _schedule_retry(self, clip_dir: Path, posted: dict) -> dict:
        """Networks that hit a temporary upload limit are retried later, alone."""
        later = sorted(n for n, r in (posted.get("networks") or {}).items()
                       if r.get("status") == "deferred")
        if later:
            when = datetime.now(timezone.utc) + timedelta(hours=publish.RETRY_LATER_HOURS)
            posted["retry_at"], posted["retry_only"] = when.isoformat(), later
            self._enqueue(clip_dir, when, later)
        else:
            posted.pop("retry_at", None)
            posted.pop("retry_only", None)
        return posted

    def _requeue_from_disk(self):
        """Clips that were waiting for their slot (or for a retry) when livewatch last stopped."""
        root = self.settings.data_dir / "clips"
        for m in root.glob("*/*/meta.json") if root.is_dir() else []:
            try:
                posted = json.loads(m.read_text(encoding="utf-8")).get("posted") or {}
                if posted.get("status") == "queued":
                    self._enqueue(m.parent, datetime.fromisoformat(posted["at"]))
                elif posted.get("retry_at") and posted.get("retry_only"):
                    self._enqueue(m.parent, datetime.fromisoformat(posted["retry_at"]),
                                  list(posted["retry_only"]))
            except (OSError, ValueError, KeyError, TypeError):
                continue

    def _next_free_slot(self, now: datetime, own: Optional[datetime] = None) -> datetime:
        """The earliest a queued post may go out: ``min_gap_minutes`` after the last
        one (whichever is later, the ledger's last past slot or the last real post).
        ``own`` is the slot this post reserved itself, which does not count."""
        gap = timedelta(minutes=self.settings.min_gap_minutes)
        past = [t for t in self.ledger.times
                if t <= now and not (own and abs((t - own).total_seconds()) < 1)]
        last = max(past + ([self._last_post_at] if self._last_post_at else []), default=None)
        return now if last is None else max(now, last + gap)

    def _poster(self):
        while not self.stop_event.is_set():
            due = None
            with self._post_lock:
                now = datetime.now(timezone.utc)
                if self._post_queue and self._post_queue[0][0] <= now:
                    due = self._post_queue.pop(0)
                    # posts that piled up while livewatch was off are spaced out,
                    # not fired together (a burst trips YouTube's upload limit)
                    slot = self._next_free_slot(now, own=due[0])
                    if slot > now + timedelta(seconds=30):
                        self._post_queue.append((slot, due[1], due[2]))
                        self._post_queue.sort(key=lambda x: x[0])
                        due = None
            if due is not None:
                self._last_post_at = datetime.now(timezone.utc)
                _, clip_dir, only = due
                try:
                    meta = json.loads((clip_dir / "meta.json").read_text(encoding="utf-8"))
                    res = deliver(clip_dir, self.settings, self.log, only=only)
                    meta["posted"] = merge_posted(meta.get("posted"), res) if only else res
                    meta["posted"] = self._schedule_retry(clip_dir, meta["posted"])
                    write_meta(clip_dir, meta)
                    if res.get("status") in ("posted", "partial"):
                        self.counts["posted"] += 1
                except Exception as e:  # noqa: BLE001 - logged, the queue goes on
                    self.log(f"queued post failed: {e}")
                continue
            self.stop_event.wait(10)

    def _index(self, meta: dict, out: Path):
        new = not self.index_path.exists()
        with open(self.index_path, "a", newline="", encoding="utf-8") as fh:
            wr = csv.writer(fh)
            if new:
                wr.writerow(["moment", "channel", "score", "style", "title", "file", "post_status",
                             "request_id"])
            p = meta.get("posted") or {}
            wr.writerow([meta["moment"], meta["channel"], meta["score"], meta["text"]["style"],
                         meta["text"]["title"], str(out), p.get("status", ""), p.get("request_id", "")])


def merge_posted(old: Optional[dict], new: dict) -> dict:
    """A retry's result folded into the clip's earlier one: networks that were
    already posted keep their result, the retried ones get the new one."""
    if not isinstance(old, dict) or not old.get("networks"):
        return new
    nets = dict(old["networks"])
    nets.update(new.get("networks") or {})
    states = {r.get("status") for r in nets.values()}
    ok = {"posted", "sent to phone"}
    if states <= ok:
        status = "posted"
    elif states & (ok | {"private"}):
        status = "partial"
    elif states <= {"deferred"}:
        status = "deferred"
    else:
        status = "failed"
    return {**old, **new, "status": status, "networks": nets}


def write_meta(d: Path, meta: dict) -> None:
    """meta.json written whole, through a temp file, so a reader never sees half."""
    tmp = d / "meta.json.tmp"
    tmp.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(d / "meta.json")


def deliver(clip_dir: Path, s: Settings, log, only: Optional[List[str]] = None) -> dict:
    """Post a finished clip to its networks the free way (see ``publish``).

    ``only``: post to these networks (the dashboard's ticks); default all of the
    clip's networks. Networks whose account is not set up are reported as
    skipped with the reason, never silently dropped.
    """
    meta = json.loads((Path(clip_dir) / "meta.json").read_text(encoding="utf-8"))
    posts = meta.get("posts") or {}
    nets = [p for p in posts if only is None or p in only]
    routes = publish.route(nets, s.youtube_via, s.instagram_via, s.others_via)
    creds = publish.credentials()
    ready, skipped = {}, {}
    for net, via in routes.items():
        why = publish.missing({net: via}, creds, s.upload_post_user)
        if why:
            skipped[net] = {"status": "skipped", "why": why[0]}
        else:
            ready[net] = via
    for net, r in skipped.items():
        log(f"   {net}: skipped, {r['why']}")
    if not ready:
        return {"status": "held", "why": "; ".join(sorted({r["why"] for r in skipped.values()}))
                or "no network selected", "networks": skipped,
                "at": datetime.now(timezone.utc).isoformat()}
    tags = (meta.get("text") or {}).get("hashtags") or []
    res = publish.publish(Path(clip_dir) / "short.mp4", posts, ready, tags=tags,
                          upload_post_user=s.upload_post_user, log=log)
    res["networks"].update(skipped)
    if skipped and res["status"] == "posted":
        res["status"] = "partial"
    res["at"] = datetime.now(timezone.utc).isoformat()
    return res
