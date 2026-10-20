"""Wall-clock schedules become UTC instants. Expected values are computed
from the zone offset, not by calling the function under test."""

from datetime import datetime, timezone

import pytest

from opendispatch.schedule_times import ClipSlot, ScheduleError, assign


def _utc(y, m, d, hh, mm):
    return datetime(y, m, d, hh, mm, tzinfo=timezone.utc)


def test_interval_in_kolkata_is_five_and_a_half_hours_behind_utc():
    now = _utc(2026, 10, 1, 0, 0)
    clips = [ClipSlot(index=0), ClipSlot(index=1)]

    times = assign(
        clips,
        timezone_name="Asia/Kolkata",
        mode="interval",
        start="2026-10-21T18:30",
        every_minutes=1440,
        now=now,
    )

    assert times == [_utc(2026, 10, 21, 13, 0), _utc(2026, 10, 22, 13, 0)]


def test_new_york_winter_and_summer_use_the_right_offset():
    now = _utc(2026, 1, 1, 0, 0)
    winter = assign(
        [ClipSlot(index=0, when="2026-01-15T18:30")],
        timezone_name="America/New_York",
        mode="at",
        now=now,
    )
    summer = assign(
        [ClipSlot(index=0, when="2026-07-15T18:30")],
        timezone_name="America/New_York",
        mode="at",
        now=now,
    )

    assert winter == [_utc(2026, 1, 15, 23, 30)]
    assert summer == [_utc(2026, 7, 15, 22, 30)]


def test_mixed_override_does_not_shift_the_other_slots():
    now = _utc(2026, 10, 1, 0, 0)
    clips = [
        ClipSlot(index=0),
        ClipSlot(index=1, when="2026-10-22T09:00"),
        ClipSlot(index=2),
    ]

    times = assign(
        clips,
        timezone_name="Asia/Kolkata",
        mode="mixed",
        start="2026-10-21T18:30",
        every_minutes=1440,
        now=now,
    )

    assert times[0] == _utc(2026, 10, 21, 13, 0)
    assert times[1] == _utc(2026, 10, 22, 3, 30)
    assert times[2] == _utc(2026, 10, 23, 13, 0)


def test_a_time_in_the_past_is_rejected():
    now = _utc(2026, 10, 21, 14, 0)
    with pytest.raises(ScheduleError):
        assign(
            [ClipSlot(index=0)],
            timezone_name="Asia/Kolkata",
            mode="interval",
            start="2026-10-21T18:30",
            every_minutes=60,
            now=now,
        )


def test_unknown_zone_is_rejected():
    with pytest.raises(ScheduleError):
        assign(
            [ClipSlot(index=0)],
            timezone_name="Mars/Olympus",
            mode="now",
            now=_utc(2026, 10, 1, 0, 0),
        )


def test_now_uses_the_supplied_instant_for_every_clip():
    now = _utc(2026, 10, 21, 15, 0)
    times = assign(
        [ClipSlot(index=3), ClipSlot(index=4)],
        timezone_name="UTC",
        mode="now",
        now=now,
    )
    assert times == [now, now]
