"""Free-plan eligibility and synthetic period logic (pure functions, no DB)."""
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from cloud import config, metering


def _user(google_sub="g-123"):
    return SimpleNamespace(google_sub=google_sub)


class TestFreePlanEligible:
    def test_google_user_is_eligible(self):
        assert metering.free_plan_eligible(_user()) is True

    def test_magic_link_only_user_is_not_eligible(self):
        assert metering.free_plan_eligible(_user(google_sub=None)) is False

    def test_none_user_is_not_eligible(self):
        assert metering.free_plan_eligible(None) is False

    def test_zero_minutes_disables_free_plan(self, monkeypatch):
        monkeypatch.setattr(config, "FREE_PLAN_MINUTES", 0)
        assert metering.free_plan_eligible(_user()) is False

    def test_google_gate_can_be_disabled(self, monkeypatch):
        monkeypatch.setattr(config, "FREE_REQUIRES_GOOGLE", False)
        assert metering.free_plan_eligible(_user(google_sub=None)) is True


class TestFreePeriodEnd:
    @pytest.mark.parametrize("now, expected", [
        (datetime(2026, 7, 20, 12, 30, tzinfo=timezone.utc),
         datetime(2026, 8, 1, tzinfo=timezone.utc)),
        (datetime(2026, 12, 31, 23, 59, 59, tzinfo=timezone.utc),
         datetime(2027, 1, 1, tzinfo=timezone.utc)),
        (datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc),
         datetime(2026, 2, 1, tzinfo=timezone.utc)),
    ])
    def test_first_instant_of_next_month(self, now, expected):
        assert metering.free_period_end(now) == expected

    def test_stable_within_a_month(self):
        a = metering.free_period_end(datetime(2026, 7, 1, tzinfo=timezone.utc))
        b = metering.free_period_end(datetime(2026, 7, 31, 23, 59, tzinfo=timezone.utc))
        assert a == b

    def test_rolls_over_at_month_boundary(self):
        july = metering.free_period_end(datetime(2026, 7, 31, 23, 59, tzinfo=timezone.utc))
        august = metering.free_period_end(datetime(2026, 8, 1, 0, 0, 1, tzinfo=timezone.utc))
        assert august > july
