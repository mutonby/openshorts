"""Atomic JSON writes for files other requests read while they change.

A job's *_metadata.json is rewritten in place by the job process (after the
clips render) and by several editor endpoints, while the dashboard keeps
reading it (/transcript, /status...). `open(path, 'w')` truncates first, so a
reader landing mid-write saw half a document and the endpoint answered 500
with a JSONDecodeError.

Writing to a temp file in the same directory and `os.replace`-ing it over the
target is atomic on POSIX: a reader opens either the old file or the new one,
never a partial one.
"""
import json
import os
import tempfile


def write_json_atomic(path, data, **dump_kwargs):
    """json.dump `data` to `path` atomically. `dump_kwargs` go to json.dump.

    The temp file ends in `.tmp`, so neither the `*_metadata.json` globs nor
    the artifact uploader (.json/.mp4) ever pick it up.
    """
    directory = os.path.dirname(os.path.abspath(path))
    fd, tmp = tempfile.mkstemp(dir=directory,
                               prefix=f".{os.path.basename(path)}.",
                               suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(data, f, **dump_kwargs)
        # mkstemp creates the file 0600; keep the mode the file already had
        # (what open(path, 'w') would have kept). Not via os.umask: it is
        # process-wide and the API writes from several threads.
        try:
            mode = os.stat(path).st_mode & 0o777
        except FileNotFoundError:
            mode = 0o644
        os.chmod(tmp, mode)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.remove(tmp)
        except OSError:
            pass
        raise
