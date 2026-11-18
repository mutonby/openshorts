"""Rendering a moment into a 1080x1920 Short in one ffmpeg pass.

1. ``extract_raw``: the buffer chunks covering the moment are joined with
   ``-c copy`` straight away (about a second), so the rolling buffer can keep
   deleting old chunks while the clip waits for its turn to render.
2. ``render_short``: decode once, lay out, burn the ASS captions and hook,
   loudness-normalise the audio to -14 LUFS, encode. Intel QuickSync is used
   when it works (a 7th-gen laptop iGPU encodes 1080x1920 faster than real
   time); otherwise libx264 ``veryfast``.

Layouts:
- ``blur``     the gameplay, sides trimmed a little, centred over a blurred,
               zoomed copy of itself. Works on any stream.
- ``facecam``  the streamer's camera (a rectangle given in the config) on
               top, the centre of the gameplay below.
"""
from __future__ import annotations

import functools
import json
import os
import subprocess
from pathlib import Path
from typing import List, Optional, Sequence

from . import procs
from .recorder import Segment

W, H = 1080, 1920
FONTS_DIR = Path(__file__).resolve().parent.parent / "fonts"


class RenderError(RuntimeError):
    pass


def _tail(text: str, n: int = 600) -> str:
    return (text or "")[-n:].strip()


def extract_raw(segments: Sequence[Segment], out_path: Path) -> Path:
    """Join chunks into one file without re-encoding."""
    if not segments:
        raise RenderError("no recorded chunks for this moment")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lst = out_path.with_suffix(".txt")
    lines = []
    for s in segments:
        p = str(Path(s.path).resolve()).replace("\\", "/").replace("'", r"'\''")
        lines.append(f"file '{p}'")
    lst.write_text("\n".join(lines) + "\n", encoding="utf-8")
    cmd = [procs.ffmpeg(), "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
           "-f", "concat", "-safe", "0", "-i", str(lst), "-map", "0:v:0", "-map", "0:a:0?",
           "-c", "copy", str(out_path)]
    r = procs.run(cmd, timeout=120, text=True)
    try:
        lst.unlink()
    except OSError:
        pass
    if r.returncode != 0 or not out_path.exists() or out_path.stat().st_size == 0:
        raise RenderError(f"joining chunks failed: {_tail(r.stderr)}")
    return out_path


def probe(path: Path) -> dict:
    cmd = [procs.ffprobe(), "-v", "error", "-show_entries",
           "format=duration:stream=codec_type,width,height", "-of", "json", str(path)]
    r = procs.run(cmd, timeout=60, text=True)
    if r.returncode != 0:
        raise RenderError(f"ffprobe failed: {_tail(r.stderr)}")
    data = json.loads(r.stdout or "{}")
    v = next((s for s in data.get("streams", []) if s.get("codec_type") == "video"), {})
    has_audio = any(s.get("codec_type") == "audio" for s in data.get("streams", []))
    return {"duration": float((data.get("format") or {}).get("duration") or 0),
            "width": int(v.get("width") or 0), "height": int(v.get("height") or 0),
            "audio": has_audio}


@functools.lru_cache(maxsize=4)
def pick_encoder(preference: str = "auto") -> str:
    """'qsv' or 'x264': QuickSync only when a test encode really works."""
    if preference == "x264":
        return "x264"
    cmd = [procs.ffmpeg(), "-hide_banner", "-loglevel", "error", "-nostdin",
           "-f", "lavfi", "-i", "color=black:s=256x256:d=0.5", "-vf", "format=nv12",
           "-c:v", "h264_qsv", "-f", "null", "-"]
    try:
        ok = procs.run(cmd, timeout=30).returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        ok = False
    if not ok and preference == "qsv":
        raise RenderError("encoder=qsv but Intel QuickSync is not usable on this machine")
    return "qsv" if ok else "x264"


def encoder_args(encoder: str) -> List[str]:
    if encoder == "qsv":
        return ["-c:v", "h264_qsv", "-preset", "medium", "-global_quality", "22",
                "-look_ahead", "0", "-g", "60"]
    return ["-c:v", "libx264", "-preset", "veryfast", "-crf", "21", "-pix_fmt", "yuv420p",
            "-g", "60", "-threads", "2"]


def _filter_path(path: Path, cwd: Path) -> str:
    """A path usable inside a filtergraph option: relative when possible."""
    try:
        rel = os.path.relpath(path, cwd)
    except ValueError:  # another drive on Windows
        rel = str(path)
    rel = rel.replace("\\", "/")
    return rel.replace(":", r"\:").replace("'", r"\'").replace(",", r"\,")


def build_filter(layout: str, fps: int, facecam: Optional[Sequence[float]],
                 ass_name: Optional[str], fonts: Optional[str], encoder: str) -> str:
    if layout == "facecam" and facecam:
        x, y, w, h = facecam
        top_h, bot_h = 640, H - 640
        graph = (
            f"[0:v]fps={fps},split=2[a][b];"
            f"[a]crop=iw*{w:.4f}:ih*{h:.4f}:iw*{x:.4f}:ih*{y:.4f},"
            f"scale={W}:{top_h}:force_original_aspect_ratio=increase,crop={W}:{top_h},setsar=1[top];"
            f"[b]crop='min(iw,ih*{W}/{bot_h})':ih,scale={W}:{bot_h},setsar=1[bot];"
            f"[top][bot]vstack=inputs=2[v0]")
    else:
        graph = (
            f"[0:v]fps={fps},split=2[a][b];"
            f"[a]scale=270:480:force_original_aspect_ratio=increase,crop=270:480,"
            f"boxblur=8:2,scale={W}:{H},setsar=1[bg];"
            f"[b]crop='min(iw,ih*1.39)':ih,scale={W}:-2,setsar=1[fg];"
            f"[bg][fg]overlay=(W-w)/2:(H-h)/2-30[v0]")
    last = "v0"
    if ass_name:
        opt = f"ass={ass_name}" + (f":fontsdir={fonts}" if fonts else "")
        graph += f";[{last}]{opt}[v1]"
        last = "v1"
    graph += f";[{last}]format={'nv12' if encoder == 'qsv' else 'yuv420p'}[vout]"
    return graph


def render_short(raw: Path, out_path: Path, start: float, duration: float, *,
                 layout: str = "blur", facecam: Optional[Sequence[float]] = None,
                 ass_path: Optional[Path] = None, fps: int = 30, encoder: str = "auto",
                 music=None, timeout: float = 1800) -> Path:
    """``music``: a ``music.Mix`` to lay under the clip's own audio, or None."""
    from .music import audio_graph
    raw, out_path = Path(raw).resolve(), Path(out_path).resolve()
    cwd = out_path.parent
    enc = pick_encoder(encoder)
    info = probe(raw)
    if layout == "facecam" and not facecam:
        layout = "blur"
    fonts = _filter_path(FONTS_DIR, cwd) if FONTS_DIR.is_dir() else None
    ass_name = _filter_path(Path(ass_path).resolve(), cwd) if ass_path else None
    agraph = audio_graph(info["audio"], duration, music)
    graph = build_filter(layout, fps, facecam, ass_name, fonts, enc)
    cmd = [procs.ffmpeg(), "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
           "-ss", f"{max(0.0, start):.3f}", "-t", f"{duration:.3f}", "-i", str(raw)]
    if music:
        cmd += ["-stream_loop", "-1", "-ss", f"{music.offset:.2f}", "-i", str(music.path)]
    cmd += ["-filter_complex", graph + (";" + agraph if agraph else ""), "-map", "[vout]"]
    if agraph:
        cmd += ["-map", "[aout]", "-c:a", "aac", "-b:a", "160k", "-ac", "2"]
    # -t on the output too: the fps filter can carry the last frame past the
    # input cut when the source's timestamps are uneven (joined chunks)
    cmd += encoder_args(enc) + ["-r", str(fps), "-t", f"{duration:.3f}", "-movflags", "+faststart",
                                "-map_metadata", "-1", str(out_path)]
    r = procs.run(cmd, low_priority=True, timeout=timeout, text=True, cwd=str(cwd))
    if r.returncode != 0 and enc == "qsv":
        # a driver hiccup on QuickSync should not lose the clip
        cmd = cmd[:cmd.index("-c:v")] + encoder_args("x264") + cmd[cmd.index("-r"):]
        graph_x = build_filter(layout, fps, facecam, ass_name, fonts, "x264")
        cmd[cmd.index("-filter_complex") + 1] = graph_x + (";" + agraph if agraph else "")
        r = procs.run(cmd, low_priority=True, timeout=timeout, text=True, cwd=str(cwd))
    if r.returncode != 0 or not out_path.exists() or out_path.stat().st_size < 10_000:
        raise RenderError(f"render failed: {_tail(r.stderr)}")
    return out_path


def extract_audio(raw: Path, out_wav: Path, start: float, duration: float) -> Path:
    """16 kHz mono WAV of the clip, for the transcriber."""
    cmd = [procs.ffmpeg(), "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
           "-ss", f"{max(0.0, start):.3f}", "-t", f"{duration:.3f}", "-i", str(raw),
           "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(out_wav)]
    r = procs.run(cmd, low_priority=True, timeout=300, text=True)
    if r.returncode != 0 or not Path(out_wav).exists():
        raise RenderError(f"audio extraction failed: {_tail(r.stderr)}")
    return Path(out_wav)
