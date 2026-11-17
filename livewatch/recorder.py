"""Recording: streamlink -> ffmpeg ``-c copy`` into 10 s chunks, kept as a rolling buffer.

Nothing is re-encoded while watching, so a 2-core laptop can follow several
streams at once; the cost is network (about 3-4 Mbit/s per 720p60 stream) and
disk (``buffer_seconds`` of chunks per channel, older ones are deleted).

Each chunk gets wall-clock times from the moment ffmpeg finished it (the
segment list line appears), not from the stream clock: ad breaks and stalls
pause the stream clock but not chat, and chat is what moments are found in.
"""
from __future__ import annotations

import re
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Tuple

from . import procs

SEGMENT_SECONDS = 10

# streamlink skipping HLS segments ("Sequence gap") means the connection cannot
# carry this quality: that many gaps within the window restart one step lower.
LAG_GAPS = 3
LAG_WINDOW = 90.0
BEHIND = "falling behind (connection too slow for this quality)"

# Quality steps, best first. A channel that falls behind moves one step down.
QUALITY_LADDER = ("720p60,720p,best", "720p,480p,best", "480p,360p,best", "360p,worst")


def quality_steps(configured: str) -> List[str]:
    """The configured quality first, then the ladder steps below it."""
    steps = [configured]
    for q in QUALITY_LADDER:
        first = configured.split(",")[0]
        if q != configured and _rank(q.split(",")[0]) < _rank(first):
            steps.append(q)
    return steps


def _rank(name: str) -> int:
    m = re.match(r"^(\d+)p(\d+)?$", name)
    if not m:
        return 10_000 if name == "best" else 0
    return int(m.group(1)) * 10 + (1 if m.group(2) and int(m.group(2)) > 30 else 0)


@dataclass(frozen=True)
class Segment:
    path: Path
    start: float   # wall clock
    end: float

    @property
    def duration(self) -> float:
        return self.end - self.start


class SegmentBuffer:
    """Thread-safe list of recorded chunks, oldest first."""

    def __init__(self, keep_seconds: float):
        self.keep_seconds = keep_seconds
        self._segs: List[Segment] = []
        self._lock = threading.Lock()

    def add(self, seg: Segment) -> List[Segment]:
        """Add a chunk; returns the chunks dropped from the buffer (caller deletes)."""
        with self._lock:
            self._segs.append(seg)
            cutoff = seg.end - self.keep_seconds
            dropped = [s for s in self._segs if s.end < cutoff]
            self._segs = [s for s in self._segs if s.end >= cutoff]
        return dropped

    @property
    def latest_end(self) -> float:
        with self._lock:
            return self._segs[-1].end if self._segs else 0.0

    @property
    def earliest_start(self) -> float:
        with self._lock:
            return self._segs[0].start if self._segs else 0.0

    def __len__(self):
        with self._lock:
            return len(self._segs)

    def select(self, start: float, end: float) -> Tuple[List[Segment], float, float]:
        """Chunks covering [start, end], the offset into the first, and the real span.

        The window is clamped to what the buffer holds. Returns ([], 0, 0) when
        nothing overlaps.
        """
        with self._lock:
            segs = [s for s in self._segs if s.end > start and s.start < end]
        if not segs:
            return [], 0.0, 0.0
        real_start = max(start, segs[0].start)
        real_end = min(end, segs[-1].end)
        return segs, real_start - segs[0].start, max(0.0, real_end - real_start)


def parse_segment_list_line(line: str) -> Optional[Tuple[str, float, float]]:
    """``seg_00012.ts,120.000000,129.999667`` -> (name, start, end)."""
    parts = line.strip().split(",")
    if len(parts) != 3 or not re.match(r"^seg_\d{5,}\.ts$", parts[0]):
        return None
    try:
        return parts[0], float(parts[1]), float(parts[2])
    except ValueError:
        return None


def parse_rms_lines(text: str) -> List[Tuple[float, float]]:
    """ffmpeg ``ametadata=print`` output -> [(pts_time, rms_db)]."""
    out = []
    t = None
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("frame:"):
            m = re.search(r"pts_time:([0-9.]+)", line)
            t = float(m.group(1)) if m else None
        elif line.startswith("lavfi.astats.Overall.RMS_level=") and t is not None:
            v = line.split("=", 1)[1]
            try:
                db = float(v)
            except ValueError:
                db = float("-inf")
            out.append((t, db if db != float("-inf") else -90.0))
            t = None
    return out


def segment_loudness(path: Path) -> List[Tuple[float, float]]:
    """Per-second RMS (dB) of a chunk's audio; [] when it has none."""
    cmd = [procs.ffmpeg(), "-v", "error", "-nostdin", "-i", str(path), "-vn", "-af",
           "aresample=16000,asetnsamples=n=16000:p=0,astats=metadata=1:reset=1,"
           "ametadata=mode=print:key=lavfi.astats.Overall.RMS_level:file=-",
           "-f", "null", "-"]
    try:
        r = procs.run(cmd, low_priority=True, timeout=30, text=True)
    except (subprocess.TimeoutExpired, OSError):
        return []
    return parse_rms_lines(r.stdout or "")


class Recorder(threading.Thread):
    """Records one live channel until the stream ends or ``stop()``."""

    def __init__(self, url: str, out_dir: Path, buffer: SegmentBuffer, quality: str,
                 on_segment: Optional[Callable[[Segment], None]] = None, log=print):
        super().__init__(name=f"rec-{out_dir.name}", daemon=True)
        self.url = url
        self.out_dir = Path(out_dir)
        self.buffer = buffer
        self.quality = quality
        self.on_segment = on_segment
        self.log = log
        self.stop_event = threading.Event()
        self._sl: Optional[subprocess.Popen] = None
        self._ff: Optional[subprocess.Popen] = None
        self.exit_reason = ""

    def stop(self):
        self.stop_event.set()
        procs.terminate(self._sl)
        procs.terminate(self._ff)

    def run(self):
        self.out_dir.mkdir(parents=True, exist_ok=True)
        list_path = self.out_dir / "list.csv"
        sl_log = open(self.out_dir / "streamlink.log", "wb")
        ff_log = open(self.out_dir / "ffmpeg.log", "wb")
        try:
            self._sl = procs.popen(
                procs.streamlink_cmd() + ["--stdout", "--loglevel", "warning",
                                          "--stream-segment-threads", "2",
                                          "--retry-open", "2",
                                          self.url, self.quality],
                stdout=subprocess.PIPE, stderr=sl_log)
            self._ff = procs.popen(
                [procs.ffmpeg(), "-hide_banner", "-loglevel", "error", "-nostdin",
                 "-fflags", "+genpts+discardcorrupt", "-i", "pipe:0",
                 "-map", "0:v:0", "-map", "0:a:0?", "-c", "copy",
                 "-f", "segment", "-segment_time", str(SEGMENT_SECONDS),
                 "-segment_list", str(list_path), "-segment_list_type", "csv",
                 "-reset_timestamps", "1", str(self.out_dir / "seg_%05d.ts")],
                stdin=self._sl.stdout, stdout=subprocess.DEVNULL, stderr=ff_log)
            self._sl.stdout.close()  # ffmpeg owns the pipe now
            self._follow(list_path)
        except OSError as e:
            self.exit_reason = f"could not start recorder: {e}"
        finally:
            requested = self.stop_event.is_set()
            self.stop()
            sl_log.close()
            ff_log.close()
            if not self.exit_reason:
                rc = self._sl.returncode if self._sl else None
                self.exit_reason = "stopped" if requested else f"stream ended (streamlink exit {rc})"

    def _falling_behind(self, log_path: Path, gaps: List[float], pos: int) -> int:
        """Note streamlink's "Sequence gap" warnings; True-ish when it cannot keep up."""
        try:
            with open(log_path, "rb") as fh:
                fh.seek(pos)
                new = fh.read()
        except OSError:
            return pos
        now = time.time()
        gaps.extend([now] * new.count(b"Sequence gap"))
        while gaps and now - gaps[0] > LAG_WINDOW:
            gaps.pop(0)
        if len(gaps) >= LAG_GAPS:
            self.exit_reason = BEHIND
            self.stop_event.set()
        return pos + len(new)

    def _follow(self, list_path: Path):
        seen = 0
        gaps: List[float] = []
        log_pos = 0
        last_lag_check = 0.0
        while not self.stop_event.is_set():
            if time.time() - last_lag_check >= 5:
                last_lag_check = time.time()
                log_pos = self._falling_behind(self.out_dir / "streamlink.log", gaps, log_pos)
            if list_path.exists():
                try:
                    lines = list_path.read_text(encoding="utf-8", errors="replace").splitlines()
                except OSError:
                    lines = []
                now = time.time()
                for line in lines[seen:]:
                    parsed = parse_segment_list_line(line)
                    if not parsed:
                        continue
                    name, s, e = parsed
                    dur = max(0.1, e - s)
                    seg = Segment(self.out_dir / name, now - dur, now)
                    for old in self.buffer.add(seg):
                        try:
                            old.path.unlink()
                        except OSError:
                            pass
                    if self.on_segment:
                        try:
                            self.on_segment(seg)
                        except Exception as ex:
                            self.log(f"segment handler error: {ex}")
                seen = len(lines)
            if self._ff and self._ff.poll() is not None:
                break
            self.stop_event.wait(0.5)
