"""Tests for the Phase 2 silence-detection helpers.

Cover the silencedetect stderr parser, the keep-window inversion math,
and ``cut_silence``'s fallback behavior when nothing is silent.
"""
from __future__ import annotations

from unittest.mock import patch

import pytest

from app.editing.silence import (
    cut_silence,
    invert_silences,
    parse_silence_segments,
)


# --- parse_silence_segments ------------------------------------------------

def test_parse_returns_empty_when_no_silence_lines():
    assert parse_silence_segments("just some ffmpeg log output") == []


def test_parse_pairs_start_and_end():
    text = (
        "[silencedetect @ 0x55] silence_start: 1.234\n"
        "[silencedetect @ 0x55] silence_end: 2.345 | silence_duration: 1.111\n"
    )
    assert parse_silence_segments(text) == [(1.234, 2.345)]


def test_parse_multiple_segments():
    text = (
        "silence_start: 0.5\nsilence_end: 1.2 | silence_duration: 0.7\n"
        "silence_start: 3.4\nsilence_end: 4.8 | silence_duration: 1.4\n"
        "silence_start: 7.0\nsilence_end: 9.5 | silence_duration: 2.5\n"
    )
    assert parse_silence_segments(text) == [
        (0.5, 1.2),
        (3.4, 4.8),
        (7.0, 9.5),
    ]


def test_parse_drops_unmatched_trailing_start():
    text = (
        "silence_start: 1.0\nsilence_end: 2.0\n"
        "silence_start: 5.0\n"  # no matching end → file-end silence
    )
    # zip() truncates to the shorter list, so the trailing start is dropped.
    assert parse_silence_segments(text) == [(1.0, 2.0)]


# --- invert_silences -------------------------------------------------------

def test_invert_no_silence_keeps_whole_clip():
    assert invert_silences([], 10.0) == [(0.0, 10.0)]


def test_invert_silence_at_start():
    keep = invert_silences([(0.0, 2.0)], 10.0)
    assert keep == [(2.0, 10.0)]


def test_invert_silence_at_end():
    keep = invert_silences([(8.0, 10.0)], 10.0)
    assert keep == [(0.0, 8.0)]


def test_invert_middle_silence_splits_into_two():
    keep = invert_silences([(3.0, 5.0)], 10.0)
    assert keep == [(0.0, 3.0), (5.0, 10.0)]


def test_invert_three_silences_yields_correct_keep_windows():
    keep = invert_silences([(0.0, 1.0), (3.0, 4.0), (8.0, 9.0)], 10.0)
    assert keep == [(1.0, 3.0), (4.0, 8.0), (9.0, 10.0)]


def test_invert_overlapping_silences_collapse():
    keep = invert_silences([(1.0, 3.0), (2.5, 4.0)], 10.0)
    # second silence starts before first ends — cursor jumps to 4.0
    assert keep == [(0.0, 1.0), (4.0, 10.0)]


def test_invert_silence_spanning_full_clip_returns_empty():
    assert invert_silences([(0.0, 10.0)], 10.0) == []


# --- cut_silence happy paths / fallbacks -----------------------------------

def test_cut_silence_falls_back_to_copy_when_no_silence(tmp_path):
    src = tmp_path / "in.mp4"
    src.write_bytes(b"fake")
    dst = tmp_path / "out.mp4"

    def fake_run(args, **_kwargs):
        # Copy-only call must use ``-c copy``.
        assert "-c" in args and args[args.index("-c") + 1] == "copy"
        dst.write_bytes(b"copied")
        return None

    with patch("app.editing.silence.ffmpeg_wrapper.probe_duration", return_value=10.0), \
         patch("app.editing.silence.ffmpeg_wrapper.run", side_effect=fake_run), \
         patch("app.editing.silence.detect_silence_segments", return_value=[]):
        result = cut_silence(str(src), str(dst))

    assert result == {"segments_removed": 0, "seconds_removed": 0.0}


def test_cut_silence_bails_out_when_total_silence_is_negligible(tmp_path):
    src = tmp_path / "in.mp4"
    src.write_bytes(b"fake")
    dst = tmp_path / "out.mp4"

    def fake_run(args, **_kwargs):
        # Bail-out path must still copy, not run a filter graph.
        assert "-filter_complex" not in args
        dst.write_bytes(b"copied")
        return None

    with patch("app.editing.silence.ffmpeg_wrapper.probe_duration", return_value=10.0), \
         patch("app.editing.silence.ffmpeg_wrapper.run", side_effect=fake_run), \
         patch("app.editing.silence.detect_silence_segments", return_value=[(1.0, 1.02)]):
        result = cut_silence(str(src), str(dst))

    assert result["segments_removed"] == 0


def test_cut_silence_invokes_filter_complex_when_silence_present(tmp_path):
    src = tmp_path / "in.mp4"
    src.write_bytes(b"fake")
    dst = tmp_path / "out.mp4"

    captured = {}

    def fake_run(args, **_kwargs):
        captured["args"] = args
        dst.write_bytes(b"cut")
        return None

    with patch("app.editing.silence.ffmpeg_wrapper.probe_duration", return_value=10.0), \
         patch("app.editing.silence.ffmpeg_wrapper.run", side_effect=fake_run), \
         patch("app.editing.silence.detect_silence_segments", return_value=[(3.0, 5.0)]):
        result = cut_silence(str(src), str(dst))

    assert result["segments_removed"] == 1
    assert result["seconds_removed"] == 2.0
    assert "-filter_complex" in captured["args"]
    filter_expr = captured["args"][captured["args"].index("-filter_complex") + 1]
    assert "select=" in filter_expr
    assert "aselect=" in filter_expr
    assert "between(t,0.000,3.000)" in filter_expr
    assert "between(t,5.000,10.000)" in filter_expr


def test_cut_silence_raises_when_ffmpeg_output_missing(tmp_path):
    src = tmp_path / "in.mp4"
    src.write_bytes(b"fake")
    dst = tmp_path / "out.mp4"  # never written by fake_run

    with patch("app.editing.silence.ffmpeg_wrapper.probe_duration", return_value=10.0), \
         patch("app.editing.silence.ffmpeg_wrapper.run", return_value=None), \
         patch("app.editing.silence.detect_silence_segments", return_value=[(3.0, 5.0)]):
        with pytest.raises(RuntimeError) as exc:
            cut_silence(str(src), str(dst))
        assert "empty output" in str(exc.value)
