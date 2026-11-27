"""Cross-platform compatibility layer for ``fcntl``.

The project uses file locking in a few places, but the Windows runtime used by
this workspace does not ship the Unix-only stdlib module. The repo also runs on
Linux/macOS, so prefer the real stdlib implementation when it exists instead of
always replacing it with a no-op stub.
"""

from __future__ import annotations

import importlib.util
import os
import sys
import sysconfig


def _load_stdlib_fcntl():
    stdlib = sysconfig.get_paths().get("stdlib")
    candidates = [
        os.path.join(stdlib, "fcntl.py"),
        os.path.join(stdlib, "lib-dynload", "fcntl" + sysconfig.get_config_var("EXT_SUFFIX")),
    ]
    for path in candidates:
        if not path or not os.path.exists(path):
            continue
        spec = importlib.util.spec_from_file_location("_stdlib_fcntl", path)
        if spec is None or spec.loader is None:
            continue
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    return None


if sys.platform.startswith("win"):
    import msvcrt

    LOCK_SH = 1
    LOCK_EX = 2
    LOCK_NB = 4
    LOCK_UN = 8

    def _fd_obj(fd):
        return fd.fileno() if hasattr(fd, "fileno") else int(fd)

    def _lock_mode(operation):
        if operation & LOCK_UN:
            return msvcrt.LK_UNLCK
        if operation & LOCK_NB:
            return msvcrt.LK_NBLCK
        return msvcrt.LK_LOCK

    def flock(fd, operation):
        handle = _fd_obj(fd)
        whence = os.SEEK_SET
        pos = os.lseek(handle, 0, whence)
        try:
            if operation & LOCK_UN:
                msvcrt.locking(handle, msvcrt.LK_UNLCK, 1)
                return None
            if operation & LOCK_EX:
                msvcrt.locking(handle, _lock_mode(operation), 1)
                return None
            if operation & LOCK_SH:
                msvcrt.locking(handle, _lock_mode(operation), 1)
                return None
        finally:
            os.lseek(handle, pos, whence)
        raise ValueError(f"Unsupported flock operation: {operation!r}")

    def lockf(fd, operation, length=0, start=0, whence=0):
        handle = _fd_obj(fd)
        current = os.lseek(handle, 0, os.SEEK_CUR)
        try:
            if whence == os.SEEK_SET:
                start_pos = start
            elif whence == os.SEEK_CUR:
                start_pos = current + start
            elif whence == os.SEEK_END:
                end_pos = os.lseek(handle, 0, os.SEEK_END)
                start_pos = end_pos + start
            else:
                raise ValueError(f"Unsupported whence: {whence!r}")
            if operation & LOCK_UN:
                os.lseek(handle, start_pos, os.SEEK_SET)
                msvcrt.locking(handle, msvcrt.LK_UNLCK, max(length, 1))
                return None
            mode = _lock_mode(operation)
            os.lseek(handle, start_pos, os.SEEK_SET)
            msvcrt.locking(handle, mode, max(length, 1))
            return None
        finally:
            os.lseek(handle, current, os.SEEK_SET)

    __all__ = ["LOCK_SH", "LOCK_EX", "LOCK_NB", "LOCK_UN", "flock", "lockf"]
else:
    _real = _load_stdlib_fcntl()
    if _real is None:
        raise ModuleNotFoundError("fcntl is not available on this platform")
    globals().update({name: getattr(_real, name) for name in dir(_real) if not name.startswith("__")})
    __all__ = list(globals().keys())
