"""Tests for the Phase 2 color-grade preset module.

Covers the LUT allowlist, the FFmpeg filter chains we ship, and the
``apply_lut`` wrapper's contract (rejects unknown LUTs without calling
ffmpeg).
"""
from __future__ import annotations

from unittest.mock import patch

import pytest

from app.editing.color_grade import (
    DEFAULT_LUT,
    LUT_PRESETS,
    allowed_luts,
    apply_lut,
)


def test_default_lut_is_in_presets():
    assert DEFAULT_LUT in LUT_PRESETS


def test_allowed_luts_matches_presets_keys():
    assert sorted(allowed_luts()) == sorted(LUT_PRESETS.keys())


def test_presets_cover_the_expected_set():
    assert sorted(LUT_PRESETS.keys()) == ["cool", "noir", "teal_orange", "vivid", "warm"]


@pytest.mark.parametrize("name", sorted(LUT_PRESETS.keys()))
def test_every_preset_is_a_non_empty_string(name):
    expr = LUT_PRESETS[name]
    assert isinstance(expr, str)
    assert expr.strip() != ""


def test_teal_orange_uses_curves_filter():
    assert LUT_PRESETS["teal_orange"].startswith("curves=")


def test_noir_drops_saturation():
    assert "s=0" in LUT_PRESETS["noir"]


def test_apply_lut_rejects_unknown_name(tmp_path):
    src = tmp_path / "in.mp4"
    src.write_bytes(b"fake")
    dst = tmp_path / "out.mp4"

    with patch("app.editing.color_grade.ffmpeg_wrapper.run") as run_mock:
        with pytest.raises(ValueError) as exc:
            apply_lut(str(src), str(dst), "neon_dreams")
        assert "Unknown LUT" in str(exc.value)
        run_mock.assert_not_called()


def test_apply_lut_invokes_ffmpeg_with_expected_args(tmp_path):
    src = tmp_path / "in.mp4"
    src.write_bytes(b"fake")
    dst = tmp_path / "out.mp4"

    def fake_run(args, **_kwargs):
        # Mimic a successful ffmpeg run by creating the output file.
        dst.write_bytes(b"graded")
        return None

    with patch("app.editing.color_grade.ffmpeg_wrapper.run", side_effect=fake_run) as run_mock:
        apply_lut(str(src), str(dst), "warm")

    args = run_mock.call_args.args[0]
    assert "-vf" in args
    assert args[args.index("-vf") + 1] == LUT_PRESETS["warm"]
    assert args[-1] == str(dst)


def test_apply_lut_raises_when_ffmpeg_produces_empty_output(tmp_path):
    src = tmp_path / "in.mp4"
    src.write_bytes(b"fake")
    dst = tmp_path / "out.mp4"

    # Simulate ffmpeg "succeeding" but writing nothing.
    with patch("app.editing.color_grade.ffmpeg_wrapper.run") as run_mock:
        run_mock.return_value = None
        with pytest.raises(RuntimeError) as exc:
            apply_lut(str(src), str(dst), "vivid")
        assert "empty output" in str(exc.value)
