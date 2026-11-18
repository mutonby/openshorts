"""Subprocess helpers: tool lookup, no console windows, low priority renders."""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from typing import List, Optional

IS_WINDOWS = os.name == "nt"


def _flags(low_priority: bool) -> int:
    if not IS_WINDOWS:
        return 0
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    if low_priority:
        flags |= getattr(subprocess, "BELOW_NORMAL_PRIORITY_CLASS", 0)
    return flags


def _preexec(low_priority: bool):
    if IS_WINDOWS or not low_priority:
        return None
    return lambda: os.nice(10)


def popen(cmd: List[str], low_priority: bool = False, **kw) -> subprocess.Popen:
    return subprocess.Popen(cmd, creationflags=_flags(low_priority),
                            preexec_fn=_preexec(low_priority), **kw)


def run(cmd: List[str], low_priority: bool = False, timeout: Optional[float] = None,
        **kw) -> subprocess.CompletedProcess:
    kw.setdefault("capture_output", True)
    return subprocess.run(cmd, creationflags=_flags(low_priority),
                          preexec_fn=_preexec(low_priority), timeout=timeout, **kw)


def kill_tree(pid: int) -> None:
    """Windows: end a process and its children (yt-dlp.exe and streamlink.exe
    are launchers whose real work runs in a child that a plain terminate()
    leaves behind, still holding its files open)."""
    if IS_WINDOWS:
        subprocess.run(["taskkill", "/T", "/F", "/PID", str(pid)], capture_output=True,
                       creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))


def terminate(proc: Optional[subprocess.Popen], grace: float = 5.0) -> None:
    if proc is None or proc.poll() is not None:
        return
    if IS_WINDOWS:
        kill_tree(proc.pid)
        try:
            proc.wait(timeout=grace)
        except subprocess.TimeoutExpired:
            pass
        if proc.poll() is not None:
            return
    try:
        proc.terminate()
        proc.wait(timeout=grace)
    except subprocess.TimeoutExpired:
        proc.kill()
        try:
            proc.wait(timeout=grace)
        except subprocess.TimeoutExpired:
            pass
    except OSError:
        pass


def ffmpeg() -> str:
    return os.environ.get("FFMPEG_BIN") or shutil.which("ffmpeg") or "ffmpeg"


def ffprobe() -> str:
    return os.environ.get("FFPROBE_BIN") or shutil.which("ffprobe") or "ffprobe"


def ytdlp() -> str:
    return shutil.which("yt-dlp") or "yt-dlp"


def streamlink_cmd() -> List[str]:
    exe = shutil.which("streamlink")
    return [exe] if exe else [sys.executable, "-m", "streamlink"]
