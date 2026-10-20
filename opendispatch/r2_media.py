"""Stage a clip on the public R2 bucket so Instagram's servers can fetch it.

Uses the same env names as ``cloud.storage`` (R2_ENDPOINT, R2_BUCKET,
R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY, R2_PUBLIC_BASE). The object is
world-readable only for the minutes Meta is downloading it.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from urllib.parse import quote

log = logging.getLogger("open-dispatch.r2")


class StagingError(Exception):
    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


def stage_instagram_video(video_path: str, key: str) -> str:
    from opendispatch.credentials import r2_missing

    missing = r2_missing()
    if missing:
        raise StagingError("Set " + ", ".join(missing) + " so Instagram can fetch the clip")
    from cloud.storage import upload_file
    from cloud.config import settings

    upload_file(video_path, key, content_type="video/mp4")
    return f"{settings.r2_public_base}/{quote(key, safe='/')}"


def unstage_instagram_video(key: str) -> None:
    from cloud.storage import delete_key

    delete_key(key)


def sweep_staged(max_age_seconds: int = 86400) -> int:
    """Delete ``dispatch/`` objects left behind by a crash during the poll."""
    from opendispatch.credentials import r2_missing

    if r2_missing():
        return 0
    from cloud.config import settings
    from cloud.storage import client, delete_key

    cutoff = datetime.now(timezone.utc).timestamp() - max_age_seconds
    deleted = 0
    token = None
    remote = client()
    while True:
        kw = {"Bucket": settings.r2_bucket, "Prefix": "dispatch/", "MaxKeys": 1000}
        if token:
            kw["ContinuationToken"] = token
        resp = remote.list_objects_v2(**kw)
        for obj in resp.get("Contents", []):
            modified = obj.get("LastModified")
            if modified is not None and modified.timestamp() <= cutoff:
                delete_key(obj["Key"])
                deleted += 1
        if not resp.get("IsTruncated"):
            break
        token = resp.get("NextContinuationToken")
    if deleted:
        log.info("removed %d stale dispatch object(s) from R2", deleted)
    return deleted
