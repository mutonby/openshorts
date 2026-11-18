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
                 ass_name: Optional[str], fonts: Optional[str], encoder: str,
                 card=None, cta=None) -> str:
    """``card``: a ``card.CardLayout`` (the card frame is input 1).
    ``cta``: (input index, y) of the floating follow badge."""
    if layout == "card" and card is not None:
        # the whole stream frame, never cropped (facecams sit in the corners)
        # both timelines start at 0: a .ts recording cut mid-way keeps its own
        # timestamps, and overlay would otherwise wait for frames that never line up
        graph = (
            f"[1:v]setpts=PTS-STARTPTS,fps={fps},format=yuv420p,setsar=1[bg];"
            f"[0:v]setpts=PTS-STARTPTS,fps={fps},scale={W}:{card.video_h}:force_original_aspect_ratio=decrease,"
            f"setsar=1[vid];"
            f"[bg][vid]overlay=(W-w)/2:{card.video_y}:shortest=1[v0]")
    elif layout == "facecam" and facecam:
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
            f"[b]scale={W}:-2,setsar=1[fg];"  # full width: no side crop, facecams stay
            f"[bg][fg]overlay=(W-w)/2:(H-h)/2-30[v0]")
    last = "v0"
    if cta:
        idx, y = cta
        # fades in after a second, then floats gently: 10 px up and down every 2.6 s
        graph += (f";[{idx}:v]setpts=PTS-STARTPTS,format=rgba,fade=t=in:st=0.8:d=0.7:alpha=1[cta];"
                  f"[{last}][cta]overlay=x=(W-w)/2:y='{y}+10*sin(2*PI*t/2.6)':shortest=1:eval=frame[vc]")
        last = "vc"
    if ass_name:
        opt = f"ass={ass_name}" + (f":fontsdir={fonts}" if fonts else "")
        graph += f";[{last}]{opt}[v1]"
        last = "v1"
    graph += f";[{last}]format={'nv12' if encoder == 'qsv' else 'yuv420p'}[vout]"
    return graph


def render_short(raw: Path, out_path: Path, start: float, duration: float, *,
                 layout: str = "blur", facecam: Optional[Sequence[float]] = None,
                 ass_path: Optional[Path] = None, fps: int = 30, encoder: str = "auto",
                 music=None, card=None, cta=None, timeout: float = 1800) -> Path:
    """``music``: a ``music.Mix`` to lay under the clip's own audio, or None.
    ``card``: (card frame PNG, ``card.CardLayout``) for the card layout.
    ``cta``: (badge PNG, y) for the floating follow badge."""
    from .music import audio_graph
    raw, out_path = Path(raw).resolve(), Path(out_path).resolve()
    cwd = out_path.parent
    enc = pick_encoder(encoder)
    info = probe(raw)
    if layout == "facecam" and not facecam:
        layout = "blur"
    if layout == "card" and not card:
        layout = "blur"
    card_png, card_layout = card if layout == "card" else (None, None)
    fonts = _filter_path(FONTS_DIR, cwd) if FONTS_DIR.is_dir() else None
    ass_name = _filter_path(Path(ass_path).resolve(), cwd) if ass_path else None
    # inputs: 0 the stream, then the card frame, the follow badge, the music (each optional)
    nxt = 1
    if card_png:
        nxt += 1
    cta_spec = None
    if cta:
        cta_spec = (nxt, int(cta[1]))
        nxt += 1
    agraph = audio_graph(info["audio"], duration, music, music_input=nxt)
    graph = build_filter(layout, fps, facecam, ass_name, fonts, enc, card_layout, cta_spec)
    cmd = [procs.ffmpeg(), "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
           "-ss", f"{max(0.0, start):.3f}", "-t", f"{duration:.3f}", "-i", str(raw)]
    for png in ([card_png] if card_png else []) + ([cta[0]] if cta else []):
        cmd += ["-loop", "1", "-framerate", str(fps), "-t", f"{duration:.3f}", "-i", str(Path(png).resolve())]
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
        graph_x = build_filter(layout, fps, facecam, ass_name, fonts, "x264", card_layout, cta_spec)
        cmd[cmd.index("-filter_complex") + 1] = graph_x + (";" + agraph if agraph else "")
        r = procs.run(cmd, low_priority=True, timeout=timeout, text=True, cwd=str(cwd))
    if r.returncode != 0 or not out_path.exists() or out_path.stat().st_size < 10_000:
        raise RenderError(f"render failed: {_tail(r.stderr)}")
    return out_path


def extract_frames(raw: Path, times: Sequence[float], width: int = 512) -> List[bytes]:
    """JPEG stills of ``raw`` at ``times`` (seconds), for the text writer to look at."""
    out = []
    for t in times:
        cmd = [procs.ffmpeg(), "-hide_banner", "-loglevel", "error", "-nostdin",
               "-ss", f"{max(0.0, t):.2f}", "-i", str(raw), "-frames:v", "1",
               "-vf", f"scale={width}:-2", "-q:v", "5", "-f", "image2pipe", "-c:v", "mjpeg", "-"]
        try:
            r = procs.run(cmd, low_priority=True, timeout=60)
        except (OSError, subprocess.TimeoutExpired):
            continue
        if r.returncode == 0 and r.stdout and r.stdout[:2] == b"\xff\xd8":
            out.append(r.stdout)
    return out


def frame_times(offset: float, duration: float, moment: float) -> List[float]:
    """Where to look in a clip: early setup, middle, the moment itself, its aftermath.

    ``moment`` is seconds into the clip where the peak is (``pre_seconds``).
    """
    moment = min(max(0.0, moment), duration)
    picks = [duration * 0.15, duration * 0.45, moment, min(duration - 0.5, moment + 3.0)]
    return [round(offset + max(0.0, p), 2) for p in picks]


def extract_audio(raw: Path, out_wav: Path, start: float, duration: float) -> Path:
    """16 kHz mono WAV of the clip, for the transcriber."""
    cmd = [procs.ffmpeg(), "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
           "-ss", f"{max(0.0, start):.3f}", "-t", f"{duration:.3f}", "-i", str(raw),
           "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(out_wav)]
    r = procs.run(cmd, low_priority=True, timeout=300, text=True)
    if r.returncode != 0 or not Path(out_wav).exists():
        raise RenderError(f"audio extraction failed: {_tail(r.stderr)}")
    return Path(out_wav)
