"""Audio-first discovery for completed public YouTube recordings with seekable media."""
from copy import deepcopy
from dataclasses import dataclass, field
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import time
from urllib.parse import urlsplit

import chat_hype
import highlight_signals
from jsonfile import write_json_atomic
import range_media
import source_cache
import twitch_signals
import youtube_io


class DiscoveryUnavailable(RuntimeError):
    pass


def enabled():
    return os.environ.get("HIGHLIGHT_DISCOVERY_ENABLED", "").strip() == "1"


def youtube_url(url):
    vid = source_cache.video_id(url)
    if vid and re.fullmatch(r"[A-Za-z0-9_-]{11}", vid):
        return f"https://www.youtube.com/watch?v={vid}"
    return None


def twitch_url(url):
    """The canonical https://www.twitch.tv/videos/<id> of a Twitch VOD link."""
    vod = twitch_signals.parse_vod_id(url)
    return f"https://www.twitch.tv/videos/{vod}" if vod and vod.isdigit() else None


def staged_url(url):
    """The canonical URL of a recording audio-first discovery supports (a
    YouTube video or a Twitch VOD), else None."""
    return youtube_url(url) or twitch_url(url)


# Twitch serves VOD playlists and segments from these CDNs (CloudFront today).
TWITCH_MEDIA_HOSTS = (".cloudfront.net", ".ttvnw.net", ".jtvnw.net", ".akamaized.net")


def _twitch_track_url(track):
    url = track.get("url", "")
    parsed = urlsplit(url)
    host = (parsed.hostname or "").lower()
    if (parsed.scheme != "https" or not host.endswith(TWITCH_MEDIA_HOSTS)
            or parsed.username or parsed.password or parsed.port not in (None, 443)
            or not parsed.path.endswith(".m3u8")):
        raise DiscoveryUnavailable("The stream's media does not use a supported HTTPS host.")
    return url


def workspace(directory):
    path = Path(directory, ".highlight")
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    return path


def preview_file(directory, candidate_id):
    return Path(directory, ".highlight", f"preview-{candidate_id}.mp4")


def original_file(directory, candidate_id):
    return Path(directory, ".highlight", f"source-{candidate_id}.mkv")


def _track_url(track):
    url = track.get("url", "")
    parsed = urlsplit(url)
    host = (parsed.hostname or "").lower()
    if (parsed.scheme != "https" or not host.endswith(".googlevideo.com")
            or parsed.username or parsed.password or parsed.port not in (None, 443)):
        raise DiscoveryUnavailable("The extracted media does not use a supported HTTPS video host.")
    return url


def _video_formats(info):
    return [f for f in info.get("formats", [])
            if isinstance(f.get("height"), int) and f["height"] > 0
            and f.get("vcodec") not in (None, "none")]


def _seekable(track):
    if track.get("protocol") not in (None, "https") or track.get("ext") not in ("mp4", "m4a"):
        return False
    try:
        _track_url(track)
    except (DiscoveryUnavailable, ValueError):
        return False
    return True


@dataclass
class Source:
    url: str
    info: dict
    best: dict
    audio: dict
    preview: dict
    duration: float
    kind: str = "youtube"
    bases: dict = field(default_factory=dict)
    requested_audio_language: str | None = None

    def base(self, track):
        """An HLS track's first timestamp (cached): Twitch packet times minus
        this are source seconds."""
        key = str(track.get("format_id"))
        if key not in self.bases:
            self.bases[key] = range_media.playlist_base(_twitch_track_url(track))
        return self.bases[key]

    @property
    def summary(self):
        data = {"id": self.info["id"], "title": str(self.info.get("title") or "")[:200],
                "channel": str(self.info.get("channel") or self.info.get("uploader") or "")[:120],
                "duration": self.duration, "height": self.best["height"],
                "video_format": str(self.best["format_id"]), "audio_format": str(self.audio["format_id"]),
                "caption_tracks": sorted((self.info.get("subtitles") or {}).keys() - {"live_chat"})[:20],
                "automatic_captions": bool(self.info.get("automatic_captions"))}
        if self.requested_audio_language:
            data["audio_language"] = self.requested_audio_language
        return data

    def section(self, directory, ident, start, end, *, preview):
        video = self.preview if preview else self.best
        # A generous upper bound includes MP4 seek tables and the GOP before
        # the interval. If the transport tries to read linearly, stop it.
        rate = max(2500, float(video.get("tbr") or 0) + float(self.audio.get("abr") or 160))
        budget = int(min(256 * 1024 ** 2, max(16 * 1024 ** 2, (end - start + 6) * rate * 375)))
        check_space(directory, budget)
        folder = workspace(directory)
        path = folder / f"proxy-{ident}.mkv" if preview else original_file(directory, ident)
        if self.kind == "twitch":
            # A muxed playlist: the variant carries its own audio.
            return range_media.fetch_hls(_twitch_track_url(video), path, start, end,
                                         base=self.base(video), height=video["height"])
        inputs = [_track_url(video), _track_url(self.audio)]
        span, transferred = range_media.fetch(inputs, path, start, end, bytes_limit=budget, height=video["height"])
        return span, transferred


def _ydl_opts():
    from yt_clients import fallback_extractor_args
    from youtube_io import runtime_options
    return {**runtime_options(), "quiet": True, "noprogress": True, "noplaylist": True,
            "cachedir": False, "socket_timeout": 15, "retries": 2, "fragment_retries": 2,
            "extractor_args": fallback_extractor_args("", "")}


def _twitch_formats(info):
    """(videos, audio-only tracks) of a Twitch VOD: muxed HLS variants."""
    hls = [f for f in info.get("formats", [])
           if str(f.get("protocol") or "").startswith("m3u8") and f.get("url")]
    videos = [f for f in hls if isinstance(f.get("height"), int) and f["height"] > 0
              and f.get("vcodec") not in (None, "none") and f.get("acodec") not in (None, "none")]
    audios = [f for f in hls if f.get("vcodec") == "none" and f.get("acodec") not in (None, "none")]
    return videos, audios


def _inspect_twitch(canonical, previous=None):
    import ssrf_guard
    import yt_dlp

    try:
        with ssrf_guard.guarded(), yt_dlp.YoutubeDL(_ydl_opts()) as ydl:
            info = ydl.extract_info(canonical, download=False, process=False)
        duration = float(info.get("duration") or 0)
    except Exception as exc:
        raise DiscoveryUnavailable("Could not retrieve the VOD metadata through yt-dlp.") from exc
    if not math.isfinite(duration) or duration <= 0 or info.get("is_live") or info.get(
            "live_status") in ("is_live", "is_upcoming"):
        raise DiscoveryUnavailable("Wait for the stream to finish and its VOD to become available.")
    if str(info.get("id") or "").lstrip("v") != twitch_signals.parse_vod_id(canonical):
        raise DiscoveryUnavailable("The downloader returned a different recording.")
    videos, audios = _twitch_formats(info)
    usable = [f for f in videos if _safe_track(f)]
    hd = [f for f in usable if f["height"] <= 1080] or usable
    if not hd:
        raise DiscoveryUnavailable("The VOD has no supported video variants. Use full-source review.")
    height = max(f["height"] for f in hd)
    best_at_height = [f for f in hd if f["height"] == height]
    order = lambda f: (float(f.get("fps") or 0), float(f.get("tbr") or 0))
    best = max(best_at_height, key=order)
    lowest = min(usable, key=lambda f: (f["height"], float(f.get("tbr") or 0)))
    audio = max((f for f in audios if _safe_track(f)), key=lambda f: float(f.get("tbr") or 0), default=lowest)
    if previous:
        if str(previous.get("id")) != str(info["id"]) or abs(duration - float(previous["duration"])) > 1:
            raise DiscoveryUnavailable("The source changed after review. Start a new review.")
        best = next((f for f in best_at_height if str(f.get("format_id")) == previous["video_format"]), None)
        if best is None or height != previous["height"]:
            raise DiscoveryUnavailable("The approved source formats are unavailable. No lower-quality file was substituted.")
    proxies = [f for f in usable if 360 <= f["height"] <= 540]
    preview = max(proxies, key=lambda f: (f["height"], order(f))) if proxies else best
    return Source(canonical, info, best, audio, preview, duration, kind="twitch")


def _safe_track(track):
    try:
        _twitch_track_url(track)
    except DiscoveryUnavailable:
        return False
    return True


def inspect_source(url, previous=None, *, audio_language=None):
    import ssrf_guard
    import yt_dlp

    vod = twitch_url(url)
    if vod:
        return _inspect_twitch(vod, previous)
    canonical = youtube_url(url)
    if not canonical:
        raise DiscoveryUnavailable("Audio-first retrieval supports completed public YouTube videos.")
    try:
        with ssrf_guard.guarded(), yt_dlp.YoutubeDL(_ydl_opts()) as ydl:
            info = ydl.extract_info(canonical, download=False, process=False)
        duration = float(info.get("duration") or 0)
    except Exception as exc:
        raise DiscoveryUnavailable("Could not retrieve the video metadata through yt-dlp.") from exc
    if (not math.isfinite(duration) or duration <= 0 or info.get("is_live")
            or info.get("live_status") in ("is_live", "is_upcoming", "post_live")):
        raise DiscoveryUnavailable("Wait for the stream to finish and its recording to become available.")
    if str(info.get("id")) != source_cache.video_id(canonical):
        raise DiscoveryUnavailable("The downloader returned a different recording.")

    videos = _video_formats(info)
    hd = [f for f in videos if f["height"] <= 1080] or videos
    if not hd:
        raise DiscoveryUnavailable("No video formats were available.")
    # Match the existing 1080p AVC preference, without accepting lower
    # resolution because a seekable format happens to be available there.
    height = max(f["height"] for f in hd)
    best_at_height = [f for f in hd if f["height"] == height]
    order = lambda f: (str(f.get("vcodec", "")).startswith("avc1"),
                       float(f.get("fps") or 0), float(f.get("tbr") or 0))
    comparable = max(order(f)[:2] for f in best_at_height)
    best = max((f for f in best_at_height if _seekable(f) and order(f)[:2] == comparable),
               key=order, default=None)
    audios = [f for f in info.get("formats", []) if f.get("vcodec") == "none"
              and f.get("acodec") not in (None, "none") and _seekable(f)]
    if previous:
        audio_language = previous.get("audio_language", audio_language)
    if audio_language:
        audios = [f for f in audios if youtube_io.track_language(f) == audio_language]
        if not audios:
            raise DiscoveryUnavailable(f"The {audio_language} audio track is unavailable for audio-first review. "
                                       "Try full-source review with that track.")
    if not audios or best is None:
        raise DiscoveryUnavailable("The best source is not seekable with this workflow. Use full-source review.")
    audio = max(audios, key=lambda f: (float(f.get("language_preference") or 0), not str(f.get("format_id")).endswith("-drc"),
                                      float(f.get("abr") or f.get("tbr") or 0)))
    if previous:
        if previous.get("id") != info["id"] or abs(duration - float(previous["duration"])) > 1:
            raise DiscoveryUnavailable("The source changed after review. Start a new review.")
        best = next((f for f in best_at_height if f.get("format_id") == previous["video_format"] and _seekable(f)), None)
        audio = next((f for f in audios if f.get("format_id") == previous["audio_format"]), None)
        if best is None or audio is None or height != previous["height"]:
            raise DiscoveryUnavailable("The approved source formats are unavailable. No lower-quality file was substituted.")
    proxies = [f for f in videos if 360 <= f["height"] <= 540 and _seekable(f)]
    preview = max(proxies, key=lambda f: (f["height"], order(f))) if proxies else best
    return Source(canonical, info, best, audio, preview, duration, requested_audio_language=audio_language)


def resources(directory):
    free = shutil.disk_usage(directory).free
    cpus = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)
    try:
        fields = Path("/proc/meminfo").read_text().splitlines()
        memory = next(int(line.split()[1]) * 1024 for line in fields if line.startswith("MemAvailable:"))
    except (OSError, StopIteration, ValueError):
        memory = None
    load = os.getloadavg()[0] if hasattr(os, "getloadavg") else 0.0
    return {"disk_free_bytes": free, "memory_available_bytes": memory, "cpus": cpus, "load_1m": load}


def render_workers(limit, directory):
    readings = resources(directory)
    memory = readings["memory_available_bytes"]
    return min(limit, 1 if ((memory is not None and memory < 6 * 1024 ** 3)
                          or readings["cpus"] < 4 or readings["load_1m"] > readings["cpus"]) else 2)


def check_space(directory, planned_bytes):
    if shutil.disk_usage(directory).free < planned_bytes + 128 * 1024 ** 2:
        raise DiscoveryUnavailable("There is insufficient free disk space for the next discovery stage.")


def _format_key(format_id):
    """A format id as yt-dlp's processing writes it (whitespace and
    ``,/+[]()`` become ``_``), so raw and processed ids compare equal."""
    return re.sub(r"[\s,/+\[\]()]", "_", str(format_id or ""))


def download_audio(source, directory):
    import ssrf_guard
    import yt_dlp

    suffix = source.audio["ext"]
    language = getattr(source, "requested_audio_language", None)
    stem = f"audio-{language}" if language else "audio"
    dest = workspace(directory) / f"{stem}.{suffix}"
    if not dest.exists():
        check_space(directory, int(source.audio.get("filesize") or source.duration * 25_000))
        print("Downloading the audio track for discovery.")
        wanted = _format_key(source.audio["format_id"])
        # A callable picks the track by id: Twitch's "Audio Only" holds a
        # space that the format-spec grammar would split (and that yt-dlp
        # rewrites to "_" while it processes the formats).
        opts = {**_ydl_opts(), "format": lambda ctx: [f for f in ctx["formats"]
                                                      if _format_key(f.get("format_id")) == wanted],
                "outtmpl": str(dest), "continuedl": True, "overwrites": False,
                # HLS (Twitch): many segments in parallel instead of one by one.
                "concurrent_fragment_downloads": 8,
                # Thousands of segments: one slow read must not end the job.
                "fragment_retries": 10, "retries": 5}
        try:
            with ssrf_guard.guarded(), yt_dlp.YoutubeDL(opts) as ydl:
                ydl.process_ie_result(deepcopy(source.info), download=True)
        except Exception as exc:
            raise DiscoveryUnavailable("Audio download stalled. Resubmit or resume the discovery job.") from exc
    if not dest.is_file():
        raise DiscoveryUnavailable("The audio track is unavailable.")
    return dest


def imported_chat(directory):
    return next((file for extension in ("csv", "json") if
                 (file := Path(directory, ".highlight", f"imported-chat.{extension}")).is_file()), None)


def _env_seconds(name, default, lo=None, hi=None):
    """A numeric setting; a typo or an empty value is the default, never a
    crash after the audio and the transcript are already paid for."""
    try:
        value = float(os.environ.get(name, "").strip() or default)
    except ValueError:
        value = default
    if not math.isfinite(value):
        value = default
    if lo is not None:
        value = max(lo, value)
    if hi is not None:
        value = min(hi, value)
    return value


def attach_recorded_chat(info, directory, duration):
    """Chat this machine recorded live (chat_recorder.py) for a Twitch VOD,
    written as the job's imported replay. Returns the message count; an
    uploaded replay always wins."""
    import chat_recorder

    if imported_chat(directory):
        return 0
    login = chat_recorder.channel_name((info or {}).get("uploader_id"))
    start = (info or {}).get("timestamp")
    if not login or type(start) not in (int, float) or not duration:
        return 0
    count = chat_recorder.attach(login, float(start), float(duration),
                                 workspace(directory) / "imported-chat.csv")
    if count:
        print(f"Recorded live chat attached: {count} messages.")
    return count


def recorded_chat_for_url(url, directory, duration):
    """attach_recorded_chat for a full-source Twitch job: one metadata call,
    made only when something was recorded at all."""
    import chat_recorder
    import ssrf_guard
    import yt_dlp

    canonical = twitch_url(url)
    folder = chat_recorder.recordings_dir()
    if not canonical or imported_chat(directory) or not folder.is_dir() or not any(folder.iterdir()):
        return 0
    try:
        with ssrf_guard.guarded(), yt_dlp.YoutubeDL(_ydl_opts()) as ydl:
            info = ydl.extract_info(canonical, download=False, process=False)
    except Exception:
        return 0
    return attach_recorded_chat(info, directory, duration)


def collect(directory, media, duration, url=None, *, start=0.0, chat=None, metadata=None, audio=True):
    path = workspace(directory) / "signals.json"
    chat = imported_chat(directory) or chat
    chat_config = {"offset": _env_seconds("HIGHLIGHT_CHAT_OFFSET", 0.0) - start,
                   "window": int(_env_seconds("HIGHLIGHT_CHAT_WINDOW", 10, 5, 30)),
                   "reaction_delay": _env_seconds("HIGHLIGHT_CHAT_DELAY", 7.0, 0, 30),
                   "emotes": tuple(e.casefold() for e in os.environ.get("HIGHLIGHT_CHAT_EMOTES", "").split(",") if e)}
    result = highlight_signals.chat_evidence(
        chat, duration, source_id=source_cache.video_id(url) or twitch_signals.parse_vod_id(url),
        **chat_config)
    if audio:
        try:
            result["audio"] = highlight_signals.audio_peaks(media, duration)
        except OSError:
            result["audio"] = []
    result["source"] = metadata or {}
    write_json_atomic(path, result)
    state = result["chat_status"]
    print(f"Chat replay available ({len(result['chat'])} spikes)." if state == "available"
          else "Chat replay unavailable." if state == "unavailable" else "Chat replay could not be aligned or read.")
    return result


def hype_enabled():
    """Stages 1-2 of the hype engine for highlight discovery (HYPE_ENGINE=0:
    the transcript-first discovery alone)."""
    return os.environ.get("HYPE_ENGINE", "1").strip() != "0"


def clip_band():
    """The clip length band, from the job's generation controls; 30-60 s."""
    low = _env_seconds("CLIP_MIN_SECONDS", 30.0, 10, 170)
    return low, _env_seconds("CLIP_MAX_SECONDS", 60.0, low + 5, 180)


def _count_setting(name, default, lo, hi):
    return int(_env_seconds(name, default, lo, hi))


def _sum_costs(costs):
    costs = [c for c in costs if isinstance(c, dict) and isinstance(c.get("total_cost"), (int, float))]
    if not costs:
        return None
    return {"total_cost": sum(c["total_cost"] for c in costs), "calls": len(costs),
            "input_tokens": sum(c.get("input_tokens") or 0 for c in costs),
            "output_tokens": sum(c.get("output_tokens") or 0 for c in costs),
            "model": costs[0].get("model"),
            "price_estimated": any(c.get("price_estimated") for c in costs)}


def hype_review(directory, media, duration, url=None, *, transcribe_ranges, proxy, gameplay,
                start=0.0, chat=None, metadata=None, need_proxy=False, viewer_clips=None):
    """Stages 1 and 2 over one recording: the hype scan of the whole of it,
    focused transcription of the best windows, then the judge.

    ``media``: a file with the recording's audio (the audio track of an
    audio-first job, or the full source). ``proxy(index, cand)`` makes the
    small video of a window (``need_proxy``: even without a judge key, for
    the review previews). Returns None when the scan finds nothing that
    stands out, so the caller falls back to transcript-first discovery."""
    import candidate_review
    import hype_engine
    import hype_judge

    started = time.monotonic()
    min_secs, max_secs = clip_band()
    chat_file = imported_chat(directory) or chat
    found = hype_engine.scan(
        media, duration, chat_path=chat_file,
        source_id=source_cache.video_id(url) or twitch_signals.parse_vod_id(url),
        chat_offset=_env_seconds("HIGHLIGHT_CHAT_OFFSET", 0.0) - start,
        min_secs=min_secs, max_secs=max_secs, limit=30, viewer_clips=viewer_clips)
    scan_seconds = time.monotonic() - started
    if not found or not found["candidates"]:
        print("The hype scan found no standout moments; using transcript discovery.")
        return None
    print(f"Hype scan: {len(found['candidates'])} candidate moment(s) in {scan_seconds:.0f}s "
          f"(chat {found['chat_status']}, chat delay {found['lag']:.0f}s).")

    shortlist = found["candidates"][:_count_setting("HYPE_TRANSCRIBE_CANDIDATES", 20, 3, 30)]
    started = time.monotonic()
    transcript = transcribe_ranges(media, [(c["start"], c["end"]) for c in shortlist])
    asr_seconds = time.monotonic() - started

    judged = hype_judge.prefilter(shortlist, transcript,
                                  _count_setting("HYPE_JUDGE_CANDIDATES", 12, 1, candidate_review.MAX_CANDIDATES))
    dropped = len(shortlist) - len(judged)
    started = time.monotonic()
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        print("   No Gemini key for the clip judge: candidates are ranked by hype for manual review.")
    shorts, costs = hype_judge.review(
        judged, transcript, duration, proxy=proxy, cache_dir=str(workspace(directory) / "judge"),
        gameplay=gameplay, min_secs=min_secs, max_secs=max_secs, api_key=key, need_proxy=need_proxy,
        # The streamer's own title and channel: what game this is, who is playing.
        context=" — ".join(str(v) for v in ((metadata or {}).get("title"), (metadata or {}).get("channel")) if v))
    judge_seconds = time.monotonic() - started
    print(f"Clip judge kept {len(shorts)} of {len(judged)} moment(s)"
          + (f"; {dropped} alert or overflow window(s) skipped." if dropped else "."))

    signals = collect(directory, media, duration, url, start=start, chat=chat, metadata=metadata,
                      audio=False)
    signals.update({"engine": "hype", "voice": found["markers"]["voice"],
                    "hype_curve": found["curve"], "chat_lag": found["lag"]})
    write_json_atomic(workspace(directory) / "signals.json", signals)
    return {"shorts": shorts[:candidate_review.MAX_CANDIDATES], "transcript": transcript,
            "signals": signals, "cost_analysis": _sum_costs(costs),
            "hype_timings": {"scan_seconds": round(scan_seconds, 2),
                             "transcription_seconds": round(asr_seconds, 2),
                             "judge_seconds": round(judge_seconds, 2)}}


def make_preview(span, dest, start, end):
    temp = Path(dest).with_suffix(".partial.mp4")
    try:
        subprocess.run([
            "ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-ss", f"{span.local(start):.4f}",
            "-i", str(span.path), "-t", f"{end - start:.4f}",
            "-map", "0:v:0", "-map", "0:a:0", "-vf", "scale=-2:min(480\\,ih),setsar=1",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "19", "-threads", "2",
            "-c:a", "aac", "-movflags", "+faststart", str(temp),
        ], check=True, stderr=subprocess.DEVNULL, timeout=180)
        temp.replace(dest)
    finally:
        temp.unlink(missing_ok=True)


def replay_track(source):
    tracks = (source.info.get("subtitles") or {}).get("live_chat") or []
    return any(track.get("protocol") == "youtube_live_chat_replay" for track in tracks)


def _capture_replay(url, dest):
    """Run in a cancellable child process with the same guarded yt-dlp runtime."""
    import ssrf_guard
    import yt_dlp

    source = inspect_source(url)
    if not replay_track(source):
        return
    options = {**_ydl_opts(), "skip_download": True, "writesubtitles": True,
               "subtitleslangs": ["live_chat"], "subtitlesformat": "json",
               "outtmpl": str(dest / "replay.%(ext)s")}
    with ssrf_guard.guarded(), yt_dlp.YoutubeDL(options) as ydl:
        ydl.process_ie_result(source.info, download=True)


def replay_limits():
    """(bytes, seconds) a provider chat replay may take. A 10-hour stream of a
    big channel is ~100-150 MB and ~20-25 minutes of sequential fetching
    (measured), far past the 32 MiB an upload may be; it runs alongside the
    audio download and transcription."""
    megabytes = _env_seconds("HIGHLIGHT_REPLAY_MAX_MB", 1024, 1, 8192)
    minutes = _env_seconds("HIGHLIGHT_REPLAY_MINUTES", 60, 1, 600)
    return int(megabytes * 1024 * 1024), minutes * 60


_OFFSET = re.compile(rb'"videoOffsetTimeMsec"\s*:\s*"?(\d+)')


def _replay_reach(path):
    """Seconds into the stream the replay at ``path`` reaches (its last
    message offset), or 0."""
    try:
        with open(path, "rb") as handle:
            handle.seek(max(0, os.path.getsize(path) - 65536))
            found = _OFFSET.findall(handle.read())
    except OSError:
        return 0.0
    return int(found[-1]) / 1000.0 if found else 0.0


def _clock(seconds):
    seconds = int(seconds)
    return f"{seconds // 3600}:{seconds // 60 % 60:02d}:{seconds % 60:02d}"


def fetch_replay(source, directory):
    """The provider's chat replay (YouTube), fetched next to the audio.

    Long and busy streams take a while: progress goes to the job log every
    minute. If the size or time limit (replay_limits) is reached first, the
    part that arrived is kept as ``replay.partial.live_chat.json``: chat
    evidence for the stretch it covers (the hype engine uses chat only up to
    the last message it has, hype_engine.scan), never a reason to drop it."""
    import sys

    if not replay_track(source) or os.environ.get("HIGHLIGHT_FETCH_CHAT", "1") == "0":
        return None
    folder = workspace(directory) / "replay"
    folder.mkdir(exist_ok=True)
    saved = folder / "replay.live_chat.json"
    partial = folder / "replay.partial.live_chat.json"
    max_bytes, max_seconds = replay_limits()
    for done in (saved, partial):
        if done.is_file() and 0 < done.stat().st_size <= max_bytes:
            return done
    print("Fetching the chat replay alongside the audio.", flush=True)
    try:
        with subprocess.Popen([sys.executable, "-m", "livestream", source.url, str(folder)],
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) as child:
            started = time.monotonic()
            next_report = started + 60
            while child.poll() is None:
                files = [e for e in folder.glob("replay.live_chat.json*") if e.is_file()]
                count = sum(e.stat().st_size for e in files)
                now = time.monotonic()
                if now >= next_report and files:
                    print(f"Chat replay: {count / 1e6:.0f} MB so far, up to "
                          f"{_clock(_replay_reach(max(files, key=lambda e: e.stat().st_size)))} "
                          f"of {_clock(source.duration)}.", flush=True)
                    next_report = now + 60
                if count > max_bytes or now - started >= max_seconds:
                    child.terminate()
                    try:
                        child.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        child.kill()
                    part = next((e for e in files if e.name.endswith(".part")), None)
                    if part and part.is_file() and part.stat().st_size > 0:
                        part.replace(partial)
                        print(f"Chat replay limit reached: using chat up to {_clock(_replay_reach(partial))} "
                              f"of {_clock(source.duration)}.", flush=True)
                        return partial
                    return None
                time.sleep(0.5)
            if child.returncode == 0 and saved.is_file() and 0 < saved.stat().st_size <= max_bytes:
                print(f"Chat replay ready ({saved.stat().st_size / 1e6:.0f} MB, "
                      f"{_clock(time.monotonic() - started)} to fetch).", flush=True)
                return saved
    except OSError:
        pass
    return None


def _staged_proxy(source, directory, transfers):
    """proxy(index, cand) for an audio-first job: the low-resolution preview
    range of the window (kept for the review page) and the judge's copy."""
    import hype_judge

    def proxy(index, cand):
        start, end = cand["start"], cand["end"]
        span, size = source.section(directory, index, start, end, preview=True)
        transfers.append(size)
        dest = preview_file(directory, index)
        make_preview(span, dest, start, end)
        return hype_judge.judge_proxy(dest, workspace(directory) / f"judge-{index}.mp4",
                                      start, end, origin=start)
    return proxy


def discover(url, directory, *, transcribe, analyze, load_transcript, save_transcript,
             require_speech, output_format, framing_method, transcribe_ranges=None,
             viewer_clips=None, audio_language=None):
    """Resolve an archive once, keep audio and scores, and build only shortlist previews.

    With the hype engine (``transcribe_ranges`` given, HYPE_ENGINE on) the
    whole recording is scanned for chat and streamer hype first and only the
    best windows are transcribed and judged; transcript-first discovery is
    the fallback when the scan finds nothing."""
    from concurrent.futures import ThreadPoolExecutor
    import candidate_review
    from transcribe_backends import media_seconds

    job_started = time.monotonic()
    source = inspect_source(url, audio_language=audio_language)
    metadata = source.summary
    metadata_seconds = time.monotonic() - job_started
    write_json_atomic(workspace(directory) / "source.json", metadata)
    print(f"Source inspected: {source.duration:.0f}s, {metadata['height']}p available for export.")
    if getattr(source, "kind", "youtube") == "twitch":
        attach_recorded_chat(source.info, directory, source.duration)
    with ThreadPoolExecutor(max_workers=1) as pool:
        replay = pool.submit(fetch_replay, source, directory) if not imported_chat(directory) else None
        started = time.monotonic()
        audio = download_audio(source, directory)
        audio_seconds = time.monotonic() - started
        if abs(media_seconds(audio) - source.duration) > 2:
            raise DiscoveryUnavailable("The audio timeline does not match the source video.")
        if transcribe_ranges is not None and hype_enabled():
            transfers = []
            hyped = hype_review(
                directory, audio, source.duration, source.url, transcribe_ranges=transcribe_ranges,
                proxy=_staged_proxy(source, directory, transfers),
                gameplay=framing_method in ("gameplay", "streamer"),
                chat=replay.result() if replay else None, metadata=metadata, need_proxy=True,
                viewer_clips=viewer_clips)
            if hyped is not None:
                for clip in hyped["shorts"]:
                    clip["preview_slot"] = clip["hype_slot"]
                data = {
                    "shorts": hyped["shorts"], "hype": True, "deferred": metadata,
                    "source_video": ".audio-first", "source_duration": source.duration,
                    "source_offset": 0, "transcript": hyped["transcript"],
                    "output_format": output_format, "framing_method": framing_method,
                    "signals": hyped["signals"], "cost_analysis": hyped["cost_analysis"],
                    "performance": {
                        "resources": resources(directory),
                        "wall_seconds": round(time.monotonic() - job_started, 2),
                        "metadata_seconds": round(metadata_seconds, 2),
                        "audio_seconds": round(audio_seconds, 2), **hyped["hype_timings"],
                        "audio_file_bytes": audio.stat().st_size,
                        "preview_transfer_bytes": sum(transfers),
                    },
                }
                candidate_review.write(directory, data)
                return data
        started = time.monotonic()
        words = load_transcript(directory, audio, source.duration)
        if words is None:
            words = transcribe(audio)
            save_transcript(directory, words, audio, source.duration)
        asr_seconds = time.monotonic() - started
        if not require_speech(words, source.duration):
            raise DiscoveryUnavailable("There is too little speech for audio-first discovery. Use full-source review for visual analysis.")
        signals = collect(directory, audio, source.duration, source.url,
                          chat=replay.result() if replay else None, metadata=metadata)
    started = time.monotonic()
    data = analyze(words, source.duration, editorial=True, checkpoint_dir=directory, signals=signals)
    if data is None:
        raise DiscoveryUnavailable("AI analysis could not finish. Completed scoring batches are retained.")
    analysis_seconds = time.monotonic() - started
    key = os.environ.get("GEMINI_API_KEY")
    shorts = []
    started = time.monotonic()
    preview_bytes = 0
    for i, clip in enumerate(candidate_review.shortlist(data["shorts"], source.duration), 1):
        start, end = clip["start"], clip["end"]
        span, size = source.section(directory, i, start, end, preview=True)
        preview_bytes += size
        dest = preview_file(directory, i)
        make_preview(span, dest, start, end)
        assessed = candidate_review.assess_clip(str(dest), clip, words, key,
                                                gameplay=framing_method in ("gameplay", "streamer"),
                                                frame_origin=start)
        # Fast events may be unclear in a reduced picture. Keep uncertainty
        # for manual review. An explicitly weak incident can still be dropped.
        if assessed["assessment"]["verdict"] != "weak":
            assessed["preview_slot"] = i
            assessed["supporting_signals"] = highlight_signals.related(start, end, signals)
            shorts.append(assessed)
    data["shorts"] = candidate_review.rank(shorts)
    data.update({
        "deferred": metadata, "source_video": ".audio-first", "source_duration": source.duration,
        "source_offset": 0, "transcript": words, "output_format": output_format,
        "framing_method": framing_method, "signals": signals,
        "performance": {
            "resources": resources(directory),
            "wall_seconds": round(time.monotonic() - job_started, 2),
            "metadata_seconds": round(metadata_seconds, 2), "audio_seconds": round(audio_seconds, 2),
            "transcription_seconds": round(asr_seconds, 2), "analysis_seconds": round(analysis_seconds, 2),
            "preview_seconds": round(time.monotonic() - started, 2),
            "audio_file_bytes": audio.stat().st_size, "preview_transfer_bytes": preview_bytes,
        },
    })
    candidate_review.write(directory, data)
    return data


if __name__ == "__main__":
    import sys
    _capture_replay(sys.argv[1], Path(sys.argv[2]))
