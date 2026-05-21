"""Silence detection + removal via FFmpeg ``silencedetect`` + ``select``/``aselect``."""

from __future__ import annotations

import os
import re
from typing import List, Tuple

from app.video import ffmpeg as ffmpeg_wrapper


# Parsers expect lines like:
#   [silencedetect @ 0x...] silence_start: 1.234
#   [silencedetect @ 0x...] silence_end: 5.678 | silence_duration: 4.444
_RE_SILENCE_START = re.compile(r"silence_start:\s*(-?\d+(?:\.\d+)?)")
_RE_SILENCE_END = re.compile(r"silence_end:\s*(-?\d+(?:\.\d+)?)")


def parse_silence_segments(stderr_text: str) -> List[Tuple[float, float]]:
    """Pair up ``silence_start`` / ``silence_end`` lines into ``(start, end)`` tuples.

    Tolerates a trailing unmatched ``silence_start`` (silence at end of file)
    by dropping it — we cannot tell its end time without re-running detect.
    """
    starts = [float(m.group(1)) for m in _RE_SILENCE_START.finditer(stderr_text)]
    ends = [float(m.group(1)) for m in _RE_SILENCE_END.finditer(stderr_text)]
    return list(zip(starts, ends))


def invert_silences(silences: List[Tuple[float, float]], duration: float) -> List[Tuple[float, float]]:
    """Return the complement of ``silences`` on ``[0, duration]`` — the keep windows."""
    keep: List[Tuple[float, float]] = []
    cursor = 0.0
    for s, e in silences:
        if s > cursor:
            keep.append((cursor, min(s, duration)))
        cursor = max(cursor, e)
    if cursor < duration:
        keep.append((cursor, duration))
    return [(a, b) for a, b in keep if b > a]


def detect_silence_segments(
    input_path: str,
    *,
    noise_db: float = -30.0,
    min_silence_sec: float = 0.5,
) -> List[Tuple[float, float]]:
    """Run ffmpeg ``silencedetect`` and return ``[(start, end), ...]`` in seconds."""
    result = ffmpeg_wrapper.run(
        [
            "-i", input_path,
            "-af", f"silencedetect=noise={noise_db}dB:d={min_silence_sec}",
            "-f", "null", "-",
        ],
        check=False,
    )
    stderr = (result.stderr or b"").decode(errors="replace")
    return parse_silence_segments(stderr)


def _between_expr(intervals: List[Tuple[float, float]]) -> str:
    """Build a ``between(t,a1,b1)+between(t,a2,b2)+...`` expression for select filters."""
    return "+".join(f"between(t,{a:.3f},{b:.3f})" for a, b in intervals)


def cut_silence(
    input_path: str,
    output_path: str,
    *,
    noise_db: float = -30.0,
    min_silence_sec: float = 0.5,
) -> dict:
    """Cut silent segments out of ``input_path`` and write the result to ``output_path``.

    Returns a small summary dict (``segments_removed``, ``seconds_removed``).
    If no silence is detected or the keep-windows would collapse the clip,
    falls back to a stream copy so the caller still gets a valid output.
    """
    duration = ffmpeg_wrapper.probe_duration(input_path)
    silences = detect_silence_segments(
        input_path,
        noise_db=noise_db,
        min_silence_sec=min_silence_sec,
    )
    keep = invert_silences(silences, duration)

    total_silence = sum(e - s for s, e in silences)
    if not silences or not keep or total_silence < 0.05:
        # Nothing meaningful to cut — copy the input through so the file
        # exists at output_path for the caller's idempotency check.
        ffmpeg_wrapper.run([
            "-y", "-i", input_path,
            "-c", "copy",
            output_path,
        ])
        return {"segments_removed": 0, "seconds_removed": 0.0}

    expr = _between_expr(keep)
    filter_complex = (
        f"[0:v]select='{expr}',setpts=N/FRAME_RATE/TB[v];"
        f"[0:a]aselect='{expr}',asetpts=N/SR/TB[a]"
    )
    ffmpeg_wrapper.run([
        "-y",
        "-i", input_path,
        "-filter_complex", filter_complex,
        "-map", "[v]", "-map", "[a]",
        "-c:v", "libx264", "-preset", "fast", "-crf", "20",
        "-c:a", "aac",
        output_path,
    ])

    if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
        raise RuntimeError(f"Silence cut produced empty output: {output_path}")

    return {
        "segments_removed": len(silences),
        "seconds_removed": round(total_silence, 3),
    }
