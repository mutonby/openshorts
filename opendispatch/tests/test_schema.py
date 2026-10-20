"""Schema unit tests — no network, no env."""

from __future__ import annotations

import pytest

from opendispatch.api.schema import ContentUnit, ValidationError, parse_target, validate


def test_parse_target_simple():
    assert parse_target("telegram") == ("telegram", None)


def test_parse_target_with_account():
    assert parse_target("twitter:pol") == ("twitter", "pol")


def test_parse_target_uppercase_normalised():
    assert parse_target("TWITTER:POL") == ("twitter", "pol")


def test_parse_target_invalid_raises():
    with pytest.raises(ValidationError):
        parse_target("not a target!")


def test_validate_empty_targets():
    unit = ContentUnit(targets=[], formats={"a": 1})
    assert "targets must be non-empty" in validate(unit)


def test_validate_empty_formats():
    unit = ContentUnit(targets=["telegram"], formats={})
    assert "formats must be non-empty" in validate(unit)


def test_validate_bad_target():
    unit = ContentUnit(targets=["bogus!"], formats={"x": 1})
    assert any("invalid target" in e for e in validate(unit))


def test_validate_bad_scheduled_for():
    unit = ContentUnit(targets=["telegram"], formats={"x": 1}, scheduled_for="not-a-date")
    assert any("ISO-8601" in e for e in validate(unit))


def test_validate_ok():
    unit = ContentUnit(targets=["telegram:main"],
                       formats={"telegram_message": {"text": "hi"}})
    assert validate(unit) == []


def test_roundtrip_dict():
    unit = ContentUnit(targets=["bluesky"], formats={"bluesky_post": {"text": "yo"}})
    again = ContentUnit.from_dict(unit.to_dict())
    assert again.targets == unit.targets
    assert again.formats == unit.formats
