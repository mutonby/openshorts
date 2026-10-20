"""Post one finished short to YouTube from disk and to Instagram via R2.

``publish_now`` uploads immediately. ``schedule`` writes queue rows and stores
only ``creds_ref``, never the tokens. ``run_due`` is one pass of the worker.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from opendispatch.adapters.errors import PERMANENT, prefix_error
from opendispatch.adapters.instagram import CollaboratorError, normalize_collaborators
from opendispatch.api.schema import CAPTION_LIMITS, ContentUnit
from opendispatch.r2_media import StagingError, stage_instagram_video, unstage_instagram_video

log = logging.getLogger("open-dispatch.shorts")

PLATFORMS = ("instagram", "youtube")
YT_TITLE_MAX = 100


class DispatchError(Exception):
    """The request cannot be queued. ``message`` is safe to show the user."""

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


@dataclass
class ShortPost:
    video_path: str
    platforms: list[str]
    title: str = ""
    caption: str = ""
    collaborators: list[str] = field(default_factory=list)
    scheduled_for: str | None = None
    privacy: str = "public"
    share_to_feed: bool = True
    job_id: str = ""
    clip_index: int | None = None
    timezone_name: str | None = None


@dataclass
class PlatformResult:
    platform: str
    ok: bool
    post_id: str
    error: str
    row_id: str | None = None


def _check_platforms(platforms: list[str]) -> None:
    if not platforms:
        raise DispatchError("select at least one platform")
    unknown = [name for name in platforms if name not in PLATFORMS]
    if unknown:
        raise NotImplementedError(
            "not enabled yet: " + ", ".join(unknown) + ". This build posts to Instagram and YouTube."
        )


def _length_error(post: ShortPost, platform: str) -> str | None:
    if platform == "youtube":
        if len(post.title) > YT_TITLE_MAX:
            return prefix_error(PERMANENT, f"title is {len(post.title)} chars, over the {YT_TITLE_MAX} limit")
        if len(post.caption) > CAPTION_LIMITS["youtube"]:
            return prefix_error(
                PERMANENT,
                f"caption is {len(post.caption)} chars, over the {CAPTION_LIMITS['youtube']} limit",
            )
    if platform == "instagram":
        if len(post.caption) > CAPTION_LIMITS["instagram"]:
            return prefix_error(
                PERMANENT,
                f"caption is {len(post.caption)} chars, over the {CAPTION_LIMITS['instagram']} limit",
            )
        try:
            normalize_collaborators(post.collaborators)
        except CollaboratorError as exc:
            return prefix_error(PERMANENT, str(exc))
    return None


def to_unit(post: ShortPost, creds_ref: str = "env") -> ContentUnit:
    formats: dict = {}
    meta = {"clip_index": post.clip_index, "job_id": post.job_id}
    if "instagram" in post.platforms:
        formats["instagram_post"] = {
            "caption": post.caption,
            "video_path": post.video_path,
            "collaborators": list(post.collaborators or []),
            "share_to_feed": post.share_to_feed,
            **meta,
        }
    if "youtube" in post.platforms:
        formats["youtube_short"] = {
            "video_path": post.video_path,
            "title": post.title,
            "description": post.caption,
            "caption": post.caption,
            "privacy": post.privacy,
            **meta,
        }
    return ContentUnit(
        targets=list(post.platforms),
        formats=formats,
        scheduled_for=post.scheduled_for,
        creds_ref=creds_ref,
        timezone_name=post.timezone_name,
        category="short",
    )


def dispatch_unit(unit: ContentUnit, platform: str, creds=None) -> tuple[bool, str, str]:
    """Publish one platform. Instagram uploads to R2 first and deletes after."""
    if platform == "instagram":
        return _publish_instagram(unit, creds)
    if platform == "youtube":
        from opendispatch.adapters import youtube

        yt_creds = getattr(creds, "youtube", None) if creds is not None else None
        return youtube.publish(unit, creds=yt_creds)
    raise NotImplementedError(f"not enabled yet: {platform}")


def _publish_instagram(unit: ContentUnit, creds) -> tuple[bool, str, str]:
    fmt = dict(unit.formats.get("instagram_post") or {})
    try:
        fmt["collaborators"] = normalize_collaborators(fmt.get("collaborators"))
    except CollaboratorError as exc:
        return False, "", prefix_error(PERMANENT, str(exc))

    staged_key = None
    try:
        if not fmt.get("video_url"):
            path = fmt.get("video_path")
            if not path:
                return False, "", prefix_error(PERMANENT, "instagram needs a video file")
            filename = Path(path).name
            job = fmt.get("job_id") or unit.id
            staged_key = f"dispatch/{job}/{Path(filename).stem}-{unit.id}{Path(filename).suffix}"
            fmt["video_url"] = stage_instagram_video(path, staged_key)
        unit.formats["instagram_post"] = fmt
        ig_creds = getattr(creds, "instagram", None) if creds is not None else None
        from opendispatch.adapters import instagram

        return instagram.publish(unit, creds=ig_creds)
    except StagingError as exc:
        return False, "", prefix_error(PERMANENT, exc.message)
    finally:
        if staged_key and fmt.get("video_url"):
            try:
                unstage_instagram_video(staged_key)
            except Exception as exc:  # noqa: BLE001
                log.warning("r2 cleanup failed for %s: %s", staged_key, exc)


def publish_now(post: ShortPost, creds) -> list[PlatformResult]:
    _check_platforms(post.platforms)
    unit = to_unit(post)
    results = []
    for platform in post.platforms:
        problem = _length_error(post, platform)
        if problem:
            results.append(PlatformResult(platform, False, "", problem))
            continue
        ok, post_id, err = dispatch_unit(unit, platform, creds)
        results.append(PlatformResult(platform, ok, post_id, err))
    return results


def schedule(post: ShortPost, *, store, creds_ref: str = "env") -> list[str]:
    """Enqueue one row per platform. Raises DispatchError before any write."""
    _check_platforms(post.platforms)
    for platform in post.platforms:
        problem = _length_error(post, platform)
        if problem:
            raise DispatchError(problem)
    unit = to_unit(post, creds_ref=creds_ref)
    when = post.scheduled_for or datetime.now(tz=timezone.utc).isoformat()
    return [
        store.enqueue(unit.to_dict(), f"{platform}:default", when)
        for platform in post.platforms
    ]


def run_due(store, resolve, now=None) -> int:
    """One worker pass. ``now`` is accepted so callers can pin a clock; the
    queue itself compares ``scheduled_for`` to the current UTC time."""
    del now  # list_due reads the wall clock; tests use a future or past stamp
    from opendispatch.scheduler.worker import run_once

    return run_once(resolve=resolve, queue=store)


def cancel_row(store, row_id: str) -> bool:
    row = store.get(row_id)
    if not row or row.get("status") != "queued":
        return False
    store._update(row_id, {"status": "canceled"})
    return True
