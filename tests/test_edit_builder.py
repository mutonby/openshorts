import pytest

from edit_builder import EFFECT_LIMITS, build_filter_string, normalize_edits


def _edit(type_, start, end, strength=0.0):
    return {"type": type_, "start": start, "end": end, "strength": strength}


class TestNormalizeEdits:
    def test_unknown_type_dropped(self):
        edits = [_edit("explode", 1, 3), _edit("punch_in", 1, 3)]
        result = normalize_edits(edits, duration=30)
        assert [e["type"] for e in result] == ["punch_in"]

    def test_times_clamped_to_duration(self):
        result = normalize_edits([_edit("color_pop", -2, 90)], duration=30)
        assert result[0]["start"] == 0.0
        assert result[0]["end"] == 30.0

    def test_too_short_segment_dropped(self):
        assert normalize_edits([_edit("flash", 5.0, 5.05)], duration=30) == []

    def test_strength_capped(self):
        result = normalize_edits([_edit("zoom_in", 1, 4, strength=0.9)], duration=30)
        assert result[0]["strength"] == EFFECT_LIMITS["zoom_in"]["max_strength"]

    def test_default_strength_applied(self):
        result = normalize_edits([_edit("punch_in", 1, 4)], duration=30)
        assert result[0]["strength"] == EFFECT_LIMITS["punch_in"]["default_strength"]

    def test_captions_drop_zoom_types_only(self):
        edits = [
            _edit("zoom_in", 1, 4),
            _edit("punch_in", 6, 8),
            _edit("zoom_pulse", 10, 10.8),
            _edit("color_pop", 12, 15),
            _edit("flash", 16, 17),
        ]
        result = normalize_edits(edits, duration=30, has_captions=True)
        assert [e["type"] for e in result] == ["color_pop", "flash"]

    def test_overlapping_zooms_deduplicated(self):
        edits = [_edit("zoom_in", 1, 5), _edit("punch_in", 3, 7), _edit("punch_in", 6, 9)]
        result = normalize_edits(edits, duration=30)
        assert [(e["type"], e["start"]) for e in result] == [("zoom_in", 1.0), ("punch_in", 6.0)]

    def test_long_zoom_trimmed(self):
        result = normalize_edits([_edit("zoom_in", 0, 25)], duration=30)
        assert result[0]["end"] - result[0]["start"] == pytest.approx(8.0)

    def test_max_two_flashes(self):
        edits = [_edit("flash", i, i + 1) for i in range(5)]
        result = normalize_edits(edits, duration=30)
        assert len(result) == 2

    def test_invalid_input_shapes(self):
        assert normalize_edits(None, duration=30) == []
        assert normalize_edits([{"type": "flash", "start": "x", "end": 2}], duration=30) == []
        assert normalize_edits([_edit("flash", 1, 2)], duration=0) == []


class TestBuildFilterString:
    def test_empty_edits_returns_none(self):
        filter_string, applied = build_filter_string([], duration=30, fps=30, width=1080, height=1920)
        assert filter_string is None
        assert applied == []

    def test_zoompan_locked_to_geometry(self):
        filter_string, _ = build_filter_string(
            [_edit("punch_in", 2, 4, 0.1)], duration=30, fps=30, width=1080, height=1920
        )
        assert "zoompan=" in filter_string
        assert "s=1080x1920" in filter_string
        assert "d=1" in filter_string
        assert "between(on,60,120)" in filter_string

    def test_no_bare_comparison_operators(self):
        edits = [
            _edit("zoom_in", 1, 4),
            _edit("zoom_pulse", 6, 7),
            _edit("color_pop", 8, 12),
            _edit("bw_moment", 14, 16),
            _edit("flash", 18, 19),
            _edit("vignette", 20, 24),
        ]
        filter_string, applied = build_filter_string(edits, duration=30, fps=30, width=1080, height=1920)
        assert len(applied) == 6
        stripped = filter_string.replace("<", "").replace(">", "")
        assert stripped == filter_string

    def test_color_effects_use_enable_between(self):
        filter_string, _ = build_filter_string(
            [_edit("color_pop", 3, 6, 0.5)], duration=30, fps=30, width=1080, height=1920
        )
        assert "eq=contrast=" in filter_string
        assert "enable='between(t,3.00,6.00)'" in filter_string
        assert "zoompan" not in filter_string

    def test_flash_is_short(self):
        filter_string, _ = build_filter_string(
            [_edit("flash", 5, 8)], duration=30, fps=30, width=1080, height=1920
        )
        assert "between(t,5.00,5.15)" in filter_string

    def test_captions_only_safe_effects(self):
        edits = [_edit("zoom_in", 1, 4), _edit("color_pop", 5, 9)]
        filter_string, applied = build_filter_string(
            edits, duration=30, fps=30, width=1080, height=1920, has_captions=True
        )
        assert "zoompan" not in filter_string
        assert [e["type"] for e in applied] == ["color_pop"]

    def test_fps_fallback(self):
        filter_string, _ = build_filter_string(
            [_edit("punch_in", 1, 2, 0.1)], duration=30, fps=0, width=1080, height=1920
        )
        assert "fps=30" in filter_string
