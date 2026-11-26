"""Raise the process's open-file limit at startup.

Python, unlike Go, keeps the soft RLIMIT_NOFILE it inherits, and container
runtimes often hand out a soft limit of 1024 under a much higher hard one.
The API holds a socket per client (long video streams and range requests
included), a pipe per running job and the files it serves, so under load it
reaches 1024 and uvicorn's accept loop fails with EMFILE ("Too many open
files") for every new connection until enough of them close.

Raising the soft limit to the hard one needs no privilege. It is capped so a
huge or unlimited hard limit doesn't make children (ffmpeg, main.py), which
inherit it, walk millions of descriptors on exec.
"""

try:
    import resource
except ImportError:  # not on POSIX (Windows dev box): nothing to raise
    resource = None

NOFILE_TARGET = 65536


def raise_nofile_limit(target=NOFILE_TARGET):
    """Lift the soft open-file limit towards ``target`` (never above the hard
    limit, never lowering it). Returns ``(soft, hard)`` after the attempt, or
    None when the platform has no rlimits. Never raises."""
    if resource is None:
        return None
    try:
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    except (OSError, ValueError):
        return None
    wanted = target if hard == resource.RLIM_INFINITY else min(target, hard)
    if soft != resource.RLIM_INFINITY and soft < wanted:
        try:
            resource.setrlimit(resource.RLIMIT_NOFILE, (wanted, hard))
            soft = wanted
        except (OSError, ValueError):
            pass
    return soft, hard
