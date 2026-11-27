"""Copy timed source packets through a bounded, verified HTTP byte-range reader."""
from contextlib import contextmanager
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
from pathlib import Path
import re
import secrets
import subprocess
from threading import Lock, Thread
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener


class RangeUnavailable(RuntimeError):
    pass


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def open_media(url, byte_range):
    """Use the system trust store. Never follow a redirect or print a signed URL."""
    import ssrf_guard
    with ssrf_guard.guarded():
        return build_opener(ProxyHandler({}), _NoRedirect()).open(
            Request(url, headers={"Range": byte_range, "Accept-Encoding": "identity"}), timeout=15)


@contextmanager
def byte_ranges(urls, maximum, *, opener=open_media):
    """FFmpeg reads private loopback URLs. Only measured 206 bytes cross the network."""
    token = secrets.token_urlsafe(18)
    routes = {f"/{token}/{i}": url for i, url in enumerate(urls)}
    stats = {"bytes": 0}
    lock = Lock()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def do_GET(self):
            url = routes.get(self.path)
            byte_range = self.headers.get("Range", "bytes=0-")
            match = re.fullmatch(r"bytes=(\d+)-(\d*)", byte_range)
            if not url or not match:
                self.send_error(404)
                return
            try:
                upstream = opener(url, byte_range)
            except OSError:
                self.send_error(502)
                return
            with upstream:
                content_range = upstream.headers.get("Content-Range", "")
                received = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", content_range)
                if (upstream.status != 206 or not received
                        or int(received[1]) != int(match[1])
                        or int(received[2]) < int(received[1])
                        or int(received[2]) >= int(received[3])):
                    self.send_error(502)
                    return
                left = int(received[2]) - int(received[1]) + 1
                self.send_response(206)
                self.send_header("Content-Range", content_range)
                self.send_header("Content-Length", str(left))
                self.send_header("Content-Type", "application/octet-stream")
                self.send_header("Accept-Ranges", "bytes")
                self.end_headers()
                try:
                    while left:
                        with lock:
                            count = min(64 * 1024, left, maximum - stats["bytes"])
                            if count <= 0:
                                return
                            stats["bytes"] += count
                        block = upstream.read(count)
                        with lock:
                            stats["bytes"] -= count - len(block)
                        if not block:
                            return
                        self.wfile.write(block)
                        left -= len(block)
                except OSError:
                    return

    with ThreadingHTTPServer(("127.0.0.1", 0), Handler) as server:
        thread = Thread(target=lambda: server.serve_forever(poll_interval=0.05), daemon=True)
        thread.start()
        try:
            local = [f"http://127.0.0.1:{server.server_port}{route}" for route in routes]
            yield local, stats
        finally:
            server.shutdown()
            thread.join()


@dataclass(frozen=True)
class Span:
    path: Path
    offset: float
    end: float
    width: int
    height: int

    def local(self, seconds):
        return max(0.0, seconds - self.offset)


def _stream_stop(stream):
    duration = stream.get("duration")
    if duration not in (None, "N/A"):
        return float(stream["start_time"]) + float(duration)
    # The Matroska muxer writes each stream's last packet end as an absolute
    # timestamp. The container duration is only the latest of those ends.
    hours, minutes, seconds = stream["tags"]["DURATION"].split(":")
    return int(hours) * 3600 + int(minutes) * 60 + float(seconds)


def inspect(path, start, end, *, height=None, base=0.0):
    """Matroska duration includes the copied source PTS offset. ``base``: the
    source's own first timestamp (an HLS playlist rarely starts at 0), so the
    returned Span is on the source timeline either way."""
    try:
        probe = subprocess.run(
            ["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", str(path)],
            check=True, capture_output=True, timeout=15)
        data = json.loads(probe.stdout)
        origin = float(data["format"]["start_time"]) - base
        container_stop = float(data["format"]["duration"]) - base
        video = next(s for s in data["streams"] if s["codec_type"] == "video")
        audio = next(s for s in data["streams"] if s["codec_type"] == "audio")
        starts = (float(video["start_time"]) - base, float(audio["start_time"]) - base)
        stops = (container_stop, _stream_stop(video) - base, _stream_stop(audio) - base)
        stop = min(stops)
        width, pixels = int(video["width"]), int(video["height"])
    except (OSError, subprocess.SubprocessError, ValueError, KeyError, TypeError, StopIteration) as exc:
        raise RangeUnavailable("The downloaded source range could not be read.") from exc
    if (not all(math.isfinite(v) for v in (origin, *starts, *stops)) or origin > start + 0.08
            or origin < start - 25 or max(starts) > start + 0.12
            or stop < end - 0.12 or width < 2 or pixels < 2 or (height and pixels != height)):
        raise RangeUnavailable("The downloaded range did not match the expected source timeline or quality.")
    return Span(Path(path), origin, stop, width, pixels)


def copy_packets(inputs, output, start, end, *, http=False):
    """Keep native audio, video and original PTS. No intermediate encoder."""
    seek = max(0.0, start - 3.0)
    stop = end + 1.0
    command = ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y", "-copyts"]
    for path in inputs:
        if http:
            command += ["-protocol_whitelist", "http,tcp", "-rw_timeout", "18000000", "-f", "mov"]
        command += ["-ss", f"{seek:.4f}", "-i", str(path)]
    command += ["-to", f"{stop:.4f}", "-map", "0:v:0", "-map", f"{len(inputs) - 1}:a:0",
                "-c", "copy", "-avoid_negative_ts", "disabled", str(output)]
    try:
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=180)
    except (OSError, subprocess.SubprocessError) as exc:
        raise RangeUnavailable("This source did not serve a usable byte range.") from exc


# FFmpeg may open only these for a playlist: no plain http (a redirect or a
# playlist line cannot downgrade), no file (a playlist cannot name a local
# path). Segments come from the same HTTPS CDN as the playlist.
HLS_PROTOCOLS = "https,tls,tcp,crypto,hls"


def playlist_base(url, *, timeout=60):
    """The first timestamp of an HLS playlist (Twitch VODs start near 62 s,
    not 0): what turns its packet times into source seconds."""
    try:
        probe = subprocess.run(
            ["ffprobe", "-v", "error", "-protocol_whitelist", HLS_PROTOCOLS,
             "-show_entries", "format=start_time", "-of", "json", url],
            check=True, capture_output=True, timeout=timeout)
        base = float(json.loads(probe.stdout)["format"]["start_time"])
    except (OSError, subprocess.SubprocessError, ValueError, KeyError, TypeError) as exc:
        raise RangeUnavailable("The stream playlist could not be read.") from exc
    if not math.isfinite(base) or not 0 <= base < 3600:
        raise RangeUnavailable("The stream playlist has an unexpected timeline.")
    return base


def fetch_hls(url, path, start, end, *, base, height, timeout=180):
    """[start, end] of a muxed HLS playlist (a Twitch VOD) to ``path``: the
    segments that cover it, packets copied with their timestamps, verified
    like a byte range (inspect). FFmpeg reads only the segments it needs, so a
    30 s range of a 10-hour VOD is a few segments, not the VOD."""
    path = Path(path)
    if path.is_file():
        try:
            return inspect(path, start, end, height=height, base=base), 0
        except RangeUnavailable:
            pass
    scratch = path.with_suffix(".partial.mkv")
    command = ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y", "-copyts",
               "-protocol_whitelist", HLS_PROTOCOLS, "-rw_timeout", "18000000",
               "-ss", f"{max(0.0, start - 3.0):.4f}", "-i", url,
               # With -copyts the cut is on the packets' own clock.
               "-to", f"{base + end + 1.0:.4f}", "-map", "0:v:0", "-map", "0:a:0",
               "-c", "copy", "-avoid_negative_ts", "disabled", str(scratch)]
    try:
        for attempt in range(2):
            try:
                subprocess.run(command, check=True, stdout=subprocess.DEVNULL,
                               stderr=subprocess.DEVNULL, timeout=timeout)
                span = inspect(scratch, start, end, height=height, base=base)
                break
            except (OSError, subprocess.SubprocessError, RangeUnavailable) as exc:
                scratch.unlink(missing_ok=True)
                if attempt:
                    if isinstance(exc, RangeUnavailable):
                        raise
                    raise RangeUnavailable("This stream did not serve a usable range.") from exc
        size = scratch.stat().st_size
        scratch.replace(path)
        return Span(path, span.offset, span.end, span.width, span.height), size
    finally:
        scratch.unlink(missing_ok=True)


def fetch(tracks, path, start, end, *, bytes_limit, height):
    path = Path(path)
    if path.is_file():
        try:
            return inspect(path, start, end, height=height), 0
        except RangeUnavailable:
            pass
    scratch = path.with_suffix(".partial.mkv")
    try:
        with byte_ranges(tracks, bytes_limit) as (inputs, downloaded):
            for attempt in range(2):
                try:
                    copy_packets(inputs, scratch, start, end, http=True)
                    span = inspect(scratch, start, end, height=height)
                    break
                except RangeUnavailable:
                    if attempt or downloaded["bytes"] >= bytes_limit:
                        raise
                    scratch.unlink(missing_ok=True)
        scratch.replace(path)
        return Span(path, span.offset, span.end, span.width, span.height), downloaded["bytes"]
    finally:
        scratch.unlink(missing_ok=True)
