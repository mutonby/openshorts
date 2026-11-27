"""Per-project file leases and deletion markers shared by API processes."""
import fcntl
import re
import shutil
from pathlib import Path

JOB_ID = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")


class Busy(Exception):
    pass


def _state_path(root, job_id, suffix):
    return Path(root) / ".project-operations" / f"{job_id}.{suffix}"


def deleted(root, job_id):
    return bool(JOB_ID.fullmatch(job_id or "") and _state_path(root, job_id, "deleted").is_file())


def mark_deleted(root, job_id):
    if not JOB_ID.fullmatch(job_id or ""):
        raise ValueError("Invalid project ID")
    marker = _state_path(root, job_id, "deleted")
    marker.parent.mkdir(exist_ok=True)
    marker.touch()


def lock(root, job_id, *, exclusive=False):
    """Return an open lease. Close it to release it, even after a failed operation."""
    if not JOB_ID.fullmatch(job_id or ""):
        return None
    path = _state_path(root, job_id, "lock")
    path.parent.mkdir(exist_ok=True)
    lease = path.open("ab")
    try:
        mode = fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH
        fcntl.flock(lease, mode | fcntl.LOCK_NB)
    except BlockingIOError:
        lease.close()
        raise Busy from None
    except BaseException:
        lease.close()
        raise
    return lease


def remove_local(root, upload_dir, job_id):
    """Remove this job's working directory and the source uploads named for it."""
    if not JOB_ID.fullmatch(job_id or ""):
        raise ValueError("Invalid project ID")
    for name in (job_id, f"{job_id}.restoring"):
        directory = Path(root) / name
        if directory.is_symlink():
            directory.unlink()
        elif directory.exists():
            shutil.rmtree(directory)
    uploads = Path(upload_dir)
    for path in uploads.glob(f"{job_id}_*"):
        # These are flat uploads. A symlink is unlinked as an entry.
        if not path.is_dir() or path.is_symlink():
            path.unlink(missing_ok=True)
