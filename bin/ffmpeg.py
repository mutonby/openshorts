#!/usr/bin/env python3
import os
import subprocess
import sys

REAL_FFMPEG = os.environ.get("FFMPEG_REAL_BIN", "/usr/bin/ffmpeg")


def _run_real(args):
    if os.path.splitext(REAL_FFMPEG)[1].lower() == ".py":
        cmd = [sys.executable, REAL_FFMPEG, *args]
    else:
        cmd = [REAL_FFMPEG, *args]
    try:
        completed = subprocess.run(cmd, check=False)
        raise SystemExit(completed.returncode)
    except OSError:
        raise SystemExit(127)


if __name__ == "__main__":
    argv = sys.argv[1:]
    threads = (os.environ.get("FFMPEG_THREADS") or "").strip()
    if threads in ("", "0", "auto") or not threads.isdigit():
        _run_real(argv)

    n = int(threads)
    has_filter_threads = any(arg == "-filter_threads" for arg in argv)
    has_filter_complex_threads = any(arg == "-filter_complex_threads" for arg in argv)

    rebuilt = []
    seen_threads = False
    i = 0
    while i < len(argv):
        arg = argv[i]
        if arg == "-threads":
            seen_threads = True
            rebuilt.append(arg)
            i += 1
            if i < len(argv):
                rebuilt.append(argv[i])
            i += 1
            continue
        if arg == "-i" and not seen_threads:
            rebuilt.extend(["-threads", str(n)])
        rebuilt.append(arg)
        seen_threads = False
        i += 1

    if not has_filter_complex_threads:
        rebuilt = ["-filter_complex_threads", str(n), *rebuilt]
    if not has_filter_threads:
        rebuilt = ["-filter_threads", str(n), *rebuilt]

    _run_real(rebuilt)
