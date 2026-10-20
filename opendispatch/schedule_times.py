"""Turn a local schedule request into UTC instants.

Wall times carry no offset. The IANA zone on the request is what interprets
them, so 18:30 in Kolkata and 18:30 in New York are different moments.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError


class ScheduleError(ValueError):
    """The request does not describe a usable set of post times."""


@dataclass(frozen=True)
class ClipSlot:
    index: int
    when: str | None = None


def _zone(name: str) -> ZoneInfo:
    try:
        return ZoneInfo(name)
    except ZoneInfoNotFoundError as exc:
        raise ScheduleError(f"unknown timezone: {name}") from exc


def _wall(value: str, zone: ZoneInfo) -> datetime:
    raw = (value or "").strip()
    if not raw or raw.endswith("Z") or "+" in raw[10:] or raw.count("-") > 2:
        raise ScheduleError(f"wall time must be YYYY-MM-DDTHH:MM with no offset, got {value!r}")
    if len(raw) == 16:
        raw = raw + ":00"
    try:
        parsed = datetime.fromisoformat(raw)
    except ValueError as exc:
        raise ScheduleError(f"wall time is not a date and time: {value!r}") from exc
    if parsed.tzinfo is not None:
        raise ScheduleError("wall time must not include an offset")
    return parsed.replace(tzinfo=zone)


def _not_past(instant: datetime, now: datetime, label: str) -> datetime:
    utc = instant.astimezone(timezone.utc)
    if utc < now.astimezone(timezone.utc):
        raise ScheduleError(f"{label} is in the past")
    return utc


def assign(
    clips: list[ClipSlot],
    *,
    timezone_name: str,
    mode: str,
    start: str | None = None,
    every_minutes: int | None = None,
    now: datetime | None = None,
) -> list[datetime]:
    """One aware UTC datetime per clip, in the same order as ``clips``."""
    if now is None:
        now = datetime.now(timezone.utc)
    if now.tzinfo is None:
        raise ScheduleError("now must be timezone-aware")
    zone = _zone(timezone_name)

    if mode == "now":
        instant = now.astimezone(timezone.utc)
        return [instant for _ in clips]

    if mode == "at":
        out: list[datetime] = []
        for clip in clips:
            if not clip.when:
                raise ScheduleError(f"clip {clip.index} needs a time")
            out.append(_not_past(_wall(clip.when, zone), now, f"clip {clip.index}"))
        return out

    if mode in ("interval", "mixed"):
        if not start or every_minutes is None or every_minutes <= 0:
            raise ScheduleError("interval needs a start and a positive every_minutes")
        origin = _wall(start, zone)
        out = []
        for position, clip in enumerate(clips):
            if mode == "mixed" and clip.when:
                instant = _wall(clip.when, zone)
            else:
                instant = origin + timedelta(minutes=every_minutes * position)
            out.append(_not_past(instant, now, f"clip {clip.index}"))
        return out

    raise ScheduleError(f"unknown mode: {mode}")
