"""
Characterization tests for VideoEditor._sanitize_filter_string and
_enforce_zoompan_output_size in editor.py.

Both will move into openshorts/utils/filters.py in Phase 1 (made
shared so the motion-graphics and audio compositors can reuse them).
These tests lock in the conversions FFmpeg builds depend on.
"""
import pytest

from app.editing.ai_filters import VideoEditor


# --- _sanitize_filter_string --------------------------------------------


@pytest.mark.parametrize(
    "raw, expected",
    [
        # Simple comparisons.
        ("t<3", "lt(t,3)"),
        ("t<=3", "lte(t,3)"),
        ("on>3", "gt(on,3)"),
        ("on>=75", "gte(on,75)"),
        # Decimals.
        ("t<3.5", "lt(t,3.5)"),
        ("t>=2.25", "gte(t,2.25)"),
        # Negative numbers.
        ("x<-5", "lt(x,-5)"),
        # Whitespace tolerance.
        ("t < 3", "lt(t,3)"),
        ("on  >=  75", "gte(on,75)"),
        # Already function-form should be left alone.
        ("lt(t,3)", "lt(t,3)"),
        ("gte(on,75)", "gte(on,75)"),
        # Nothing to sanitize.
        ("eq=brightness=0.1", "eq=brightness=0.1"),
    ],
)
def test_sanitize_converts_comparison_operators(raw, expected):
    assert VideoEditor._sanitize_filter_string(raw) == expected


def test_sanitize_handles_multiple_operators_in_one_string():
    raw = "enable='between(t,1,3)':eval=frame:expr='if(t<2,0.5,1)'"
    out = VideoEditor._sanitize_filter_string(raw)
    assert "lt(t,2)" in out
    # The `between(t,1,3)` token has no `<` / `>` so it survives.
    assert "between(t,1,3)" in out


def test_sanitize_does_not_touch_unrelated_arithmetic():
    raw = "scale=w=iw/2:h=ih/2"
    assert VideoEditor._sanitize_filter_string(raw) == raw


# --- _enforce_zoompan_output_size ---------------------------------------


def test_enforce_replaces_existing_size():
    raw = "zoompan=z='zoom+0.001':d=125:s=640x480"
    out = VideoEditor._enforce_zoompan_output_size(raw, 1920, 1080)
    assert ":s=1920x1080" in out
    assert ":s=640x480" not in out


def test_enforce_appends_size_if_missing():
    raw = "zoompan=z='zoom+0.001':d=125"
    out = VideoEditor._enforce_zoompan_output_size(raw, 1280, 720)
    assert out.endswith(":s=1280x720")


def test_enforce_leaves_non_zoompan_filters_untouched():
    raw = "eq=brightness=0.1:contrast=1.2"
    assert VideoEditor._enforce_zoompan_output_size(raw, 1920, 1080) == raw


def test_enforce_processes_each_filter_in_a_chain():
    # Two zoompans in the same chain — both should get the size enforced.
    raw = "zoompan=z='1.1':d=10,format=yuv420p,zoompan=z='1.2':d=10:s=640x360"
    out = VideoEditor._enforce_zoompan_output_size(raw, 1920, 1080)
    parts = out.split(",")
    zoompans = [p for p in parts if p.startswith("zoompan=")]
    assert len(zoompans) == 2
    assert all(":s=1920x1080" in zp for zp in zoompans)
    # Non-zoompan stays.
    assert "format=yuv420p" in parts
