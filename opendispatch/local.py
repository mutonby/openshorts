"""The local OpenShorts call: plan times, enqueue rows, list them, tick the worker.

Credentials are read from the environment at enqueue time and again when a
row comes due. The queue file holds ``creds_ref="env"`` and nothing else.
"""

from __future__ import annotations

import logging
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from opendispatch.credentials import config_flags, from_env, r2_missing
from opendispatch.schedule_times import ClipSlot, ScheduleError, assign
from opendispatch.shorts import DispatchError, ShortPost, cancel_row, run_due, schedule

log = logging.getLogger("open-dispatch.local")

_stop = threading.Event()
_started = False


def local_store(output_dir: str):
    """JSONL queue under ``<output_dir>/dispatch``, ignoring DATABASE_URL."""
    from opendispatch.api.queue import JsonlQueue, bind_data_dir

    bind_data_dir(Path(output_dir) / "dispatch")
    return JsonlQueue()


def _require_creds(clips: list[dict]) -> None:
    platforms = {name for clip in clips for name in clip.get("platforms") or []}
    creds = from_env()
    if "instagram" in platforms:
        if creds.instagram is None:
            raise DispatchError("Set IG_USER_ID and IG_TOKEN")
        missing = r2_missing()
        if missing:
            raise DispatchError("Set " + ", ".join(missing))
    if "youtube" in platforms:
        if creds.youtube is None:
            raise DispatchError(
                "Set YOUTUBE_CLIENT_ID, YOUTUBE_CLIENT_SECRET, and YOUTUBE_REFRESH_TOKEN"
            )


def plan_posts(payload: dict, *, resolve_file, now: datetime | None = None) -> list[dict]:
    """Resolve clip files and UTC instants. Does not enqueue."""
    clips = list(payload.get("clips") or [])
    if not clips:
        raise DispatchError("select at least one clip")
    zone_name = payload.get("timezone") or "UTC"
    try:
        zone = ZoneInfo(zone_name)
    except ZoneInfoNotFoundError as exc:
        raise DispatchError(f"unknown timezone: {zone_name}") from exc

    slots = []
    paths: dict[int, str] = {}
    for clip in clips:
        index = int(clip["index"])
        path = resolve_file(index)
        if not path or not Path(path).is_file():
            raise DispatchError(f"clip file not found for clip {index}")
        paths[index] = str(path)
        slots.append(ClipSlot(index=index, when=clip.get("when") or None))
    _require_creds(clips)
    if now is None:
        now = datetime.now(timezone.utc)
    try:
        times = assign(
            slots,
            timezone_name=zone_name,
            mode=payload.get("mode") or "now",
            start=payload.get("start"),
            every_minutes=payload.get("every_minutes"),
            now=now,
        )
    except ScheduleError as exc:
        raise DispatchError(str(exc)) from exc

    planned = []
    for clip, when in zip(clips, times):
        index = int(clip["index"])
        local = when.astimezone(zone).strftime("%Y-%m-%d %H:%M")
        for platform in clip["platforms"]:
            planned.append({
                "clip_index": index,
                "platform": platform,
                "video_path": paths[index],
                "title": clip.get("title") or "",
                "caption": clip.get("caption") or "",
                "collaborators": list(clip.get("collaborators") or []),
                "scheduled_for": when.isoformat(),
                "local_time": local,
                "timezone": zone_name,
                "job_id": payload.get("job_id") or "",
            })
    return planned


def create_posts(payload: dict, *, resolve_file, store, now: datetime | None = None) -> dict:
    planned = plan_posts(payload, resolve_file=resolve_file, now=now)
    # One ShortPost per clip, carrying every platform selected for that clip,
    # so two platforms on one clip become two rows with the same time.
    by_clip: dict[int, dict] = {}
    for item in planned:
        bucket = by_clip.setdefault(item["clip_index"], {**item, "platforms": []})
        bucket["platforms"].append(item["platform"])
    row_ids: list[str] = []
    for item in by_clip.values():
        post = ShortPost(
            video_path=item["video_path"],
            platforms=item["platforms"],
            title=item["title"],
            caption=item["caption"],
            collaborators=item["collaborators"],
            scheduled_for=item["scheduled_for"],
            job_id=item["job_id"],
            clip_index=item["clip_index"],
            timezone_name=item["timezone"],
        )
        row_ids.extend(schedule(post, store=store))
    public = [{k: v for k, v in item.items() if k != "video_path"} for item in planned]
    return {"planned": public, "row_ids": row_ids}


def list_rows(store, job_id: str | None = None) -> list[dict]:
    out = []
    for row in store.list_all():
        unit = row.get("unit") or {}
        formats = unit.get("formats") or {}
        fmt = next(iter(formats.values()), {}) if formats else {}
        if job_id and fmt.get("job_id") not in (None, "", job_id):
            continue
        zone_name = unit.get("timezone_name") or "UTC"
        try:
            zone = ZoneInfo(zone_name)
        except ZoneInfoNotFoundError:
            zone = timezone.utc
        when = row.get("scheduled_for") or ""
        local = when
        if when:
            try:
                local = datetime.fromisoformat(when).astimezone(zone).strftime("%Y-%m-%d %H:%M")
            except ValueError:
                local = when
        out.append({
            "id": row["id"],
            "platform": (row.get("platform") or "").split(":")[0],
            "status": row.get("status"),
            "post_id": row.get("post_id"),
            "last_error": row.get("last_error"),
            "clip_index": fmt.get("clip_index"),
            "job_id": fmt.get("job_id"),
            "scheduled_for": when,
            "local_time": local,
            "timezone": zone_name,
        })
    return out


def cancel(store, row_id: str) -> bool:
    return cancel_row(store, row_id)


def start_worker(output_dir: str) -> None:
    """Daemon thread. No-op when DISPATCH_WORKER=0 (the test suite)."""
    global _started
    if os.getenv("DISPATCH_WORKER", "1").strip() == "0" or _started:
        return
    _started = True
    store = local_store(output_dir)

    def loop() -> None:
        try:
            from opendispatch.r2_media import sweep_staged
            sweep_staged()
        except Exception as exc:  # noqa: BLE001
            log.warning("dispatch sweep skipped: %s", exc)
        while not _stop.is_set():
            try:
                run_due(store, lambda _ref, _platform: from_env())
            except Exception:  # noqa: BLE001
                log.exception("dispatch tick failed")
            _stop.wait(int(os.getenv("DISPATCH_POLL_SECONDS", "5")))

    threading.Thread(target=loop, name="opendispatch", daemon=True).start()


def dispatch_config() -> dict:
    return config_flags()
