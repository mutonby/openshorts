"""Single FFmpeg wrapper for the entire codebase.

Goal: every ``subprocess.run(['ffmpeg', ...])`` call in the project should funnel
through one of these helpers. This makes it possible to:

1. Inject a global timeout / progress callback / logging hook in one place.
2. Build complex ``filter_complex`` chains by composition (used by the future
   motion-graphics compositor and the audio mixer).
3. Test the FFmpeg surface by patching this module instead of patching
   ``subprocess.run`` globally.

The scaffold below is intentionally small. Migration of existing call sites
(in app.py, openshorts/video/pipeline.py, openshorts/overlays/*.py, etc.) is
done incrementally in follow-up commits to keep each change small and the
test suite green between commits. See ROADMAP.md for the migration plan.
"""

import os
import subprocess
from typing import Iterable, List, Optional, Sequence


class FFmpegError(RuntimeError):
    """Raised when an ffmpeg/ffprobe invocation exits non-zero."""

    def __init__(self, returncode: int, stderr: bytes, cmd: Sequence[str]):
        self.returncode = returncode
        self.stderr = stderr
        self.cmd = list(cmd)
        super().__init__(
            f"ffmpeg failed (rc={returncode}): {' '.join(cmd[:6])}... — "
            f"{stderr.decode(errors='replace')[:500]}"
        )


def run(
    args: Sequence[str],
    *,
    check: bool = True,
    capture_output: bool = True,
    env: Optional[dict] = None,
    timeout: Optional[float] = None,
) -> subprocess.CompletedProcess:
    """Invoke a fully-formed ffmpeg/ffprobe command.

    Use this for one-shot ffmpeg invocations (encode, mux, probe). For
    multi-input filter graphs, build the args with ``build_filter_complex``
    and pass them through here.
    """
    cmd = ["ffmpeg", *args] if not (args and args[0].endswith("ffprobe")) else list(args)
    if cmd[0] != "ffmpeg" and not cmd[0].endswith("ffprobe"):
        cmd = ["ffmpeg", *cmd]

    # Always force UTF-8 locale; ffmpeg + non-ascii filenames on minimal
    # docker images otherwise blow up with UnicodeEncodeError from subprocess.
    full_env = os.environ.copy()
    if env:
        full_env.update(env)
    full_env.setdefault("LANG", "C.UTF-8")
    full_env.setdefault("LC_ALL", "C.UTF-8")

    result = subprocess.run(
        cmd,
        check=False,
        stdout=subprocess.PIPE if capture_output else None,
        stderr=subprocess.PIPE if capture_output else None,
        env=full_env,
        timeout=timeout,
    )

    if check and result.returncode != 0:
        raise FFmpegError(result.returncode, result.stderr or b"", cmd)

    return result


def probe_resolution(video_path: str) -> tuple:
    """Return ``(width, height)`` for the first video stream of ``video_path``."""
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height",
        "-of", "csv=s=x:p=0",
        video_path,
    ]
    result = subprocess.run(
        cmd,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env={**os.environ, "LANG": "C.UTF-8"},
    )
    width, height = result.stdout.decode().strip().split("x")
    return int(width), int(height)


def probe_duration(video_path: str) -> float:
    """Return container duration in seconds."""
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        video_path,
    ]
    result = subprocess.run(
        cmd,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return float(result.stdout.decode().strip())


def cut(input_video: str, output: str, start: float, end: float,
        *, crf: int = 18, preset: str = "fast") -> None:
    """Cut ``input_video`` to ``[start, end]`` (re-encoded for accuracy)."""
    run([
        "-y",
        "-ss", str(start),
        "-to", str(end),
        "-i", input_video,
        "-c:v", "libx264", "-crf", str(crf), "-preset", preset,
        "-c:a", "aac",
        output,
    ])


def extract_audio(input_video: str, output_audio: str) -> None:
    """Copy the audio stream from ``input_video`` into ``output_audio``."""
    run([
        "-y", "-i", input_video,
        "-vn", "-acodec", "copy",
        output_audio,
    ])


def mux_video_audio(video_path: str, audio_path: str, output: str) -> None:
    """Combine a video-only file with an audio file (stream copy, no re-encode)."""
    run([
        "-y",
        "-i", video_path,
        "-i", audio_path,
        "-c:v", "copy", "-c:a", "copy",
        output,
    ])


def overlay_png(video_path: str, png_path: str, output: str,
                *, x: int, y: int, crf: int = 22) -> None:
    """Burn a single PNG overlay onto ``video_path`` at ``(x, y)``."""
    run([
        "-y",
        "-i", video_path,
        "-i", png_path,
        "-filter_complex", f"[0:v][1:v]overlay={x}:{y}",
        "-c:a", "copy",
        "-c:v", "libx264", "-preset", "fast", "-crf", str(crf),
        output,
    ])


def build_filter_complex(chains: Iterable[str]) -> str:
    """Join multiple filter chains with ``;`` (a standard FFmpeg filter_complex).

    Used by the future motion-graphics compositor and audio mixer to batch
    multiple overlay/eq/amix operations into a single ffmpeg invocation.
    """
    return ";".join(c for c in chains if c)
