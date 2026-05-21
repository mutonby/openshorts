"""
Characterization tests for SRT generation in subtitles.py.

Targets: generate_srt, format_srt_block, hex_to_ass_color.
These move to openshorts/overlays/subtitles_generate.py in Phase 1.
"""
import re
import pytest

from app.overlays.subtitles_generate import format_srt_block, generate_srt
from app.overlays.subtitles_render import hex_to_ass_color


# --- format_srt_block ---------------------------------------------------


def test_format_srt_block_produces_well_formed_block():
    block = format_srt_block(1, 0.0, 1.5, "Hello world")
    lines = block.strip().splitlines()
    assert lines[0] == "1"
    assert re.match(r"^\d{2}:\d{2}:\d{2},\d{3} --> \d{2}:\d{2}:\d{2},\d{3}$", lines[1])
    assert lines[2] == "Hello world"


def test_format_srt_block_time_formatting():
    block = format_srt_block(2, 3661.5, 3662.001, "X")
    # 3661.5s = 1:01:01,500; 3662.001s = 1:01:02,001
    assert "01:01:01,500 --> 01:01:02,001" in block


def test_format_srt_block_pads_zeros_for_short_times():
    block = format_srt_block(3, 0.0, 0.123, "X")
    assert "00:00:00,000 --> 00:00:00,123" in block


# --- generate_srt -------------------------------------------------------


def test_generate_srt_creates_file_with_expected_word_groups(fake_transcript, tmp_path):
    out = tmp_path / "out.srt"
    ok = generate_srt(
        fake_transcript, clip_start=0.0, clip_end=10.0,
        output_path=str(out), max_chars=20, max_duration=2.0,
    )
    assert ok is True
    content = out.read_text(encoding="utf-8")
    # All six words make it in.
    for word in ("Hello", "world", "this", "is", "a", "test"):
        assert word in content
    # At least one well-formed block (index + arrow line + text + blank).
    assert re.search(
        r"^\d+\n\d{2}:\d{2}:\d{2},\d{3} --> \d{2}:\d{2}:\d{2},\d{3}\n.+",
        content, re.MULTILINE,
    )


def test_generate_srt_returns_false_when_clip_has_no_words(fake_transcript, tmp_path):
    out = tmp_path / "empty.srt"
    ok = generate_srt(
        fake_transcript, clip_start=100.0, clip_end=200.0,
        output_path=str(out), max_chars=20, max_duration=2.0,
    )
    assert ok is False
    assert not out.exists()


def test_generate_srt_times_are_relative_to_clip_start(fake_transcript, tmp_path):
    """
    Times in the SRT file must be measured from clip_start, not the
    absolute video timeline. With clip_start=1.5 the first word with
    `end > 1.5` is "is" (start=1.6, end=1.8) → block starts at 0.100s.
    """
    out = tmp_path / "rel.srt"
    generate_srt(
        fake_transcript, clip_start=1.5, clip_end=3.0,
        output_path=str(out), max_chars=200, max_duration=10.0,
    )
    content = out.read_text(encoding="utf-8")
    assert "00:00:00,100" in content
    # Sanity: the absolute timestamp 1.6s does NOT appear.
    assert "00:00:01,600" not in content


def test_generate_srt_respects_max_chars(fake_transcript, tmp_path):
    out = tmp_path / "short.srt"
    generate_srt(
        fake_transcript, clip_start=0.0, clip_end=10.0,
        output_path=str(out), max_chars=5, max_duration=10.0,
    )
    content = out.read_text(encoding="utf-8")
    # With max_chars=5, every text line must be <=5 chars after strip.
    # Pull out text lines (the ones that aren't index or arrow).
    arrow_re = re.compile(r"^\d{2}:\d{2}:\d{2},\d{3} --> ")
    for line in content.splitlines():
        if not line or line.isdigit() or arrow_re.match(line):
            continue
        assert len(line) <= 5, f"line {line!r} exceeded max_chars"


# --- hex_to_ass_color ---------------------------------------------------


def test_hex_to_ass_color_pure_red_full_opacity():
    # ASS order is &HAABBGGRR. Red (#FF0000) → AA=00, BB=00, GG=00, RR=FF.
    assert hex_to_ass_color("#FF0000", opacity=1.0) == "&H000000FF"


def test_hex_to_ass_color_pure_white_full_opacity():
    assert hex_to_ass_color("#FFFFFF", opacity=1.0) == "&H00FFFFFF"


def test_hex_to_ass_color_partial_opacity():
    # opacity 0.5 → alpha 0x80 (128)
    result = hex_to_ass_color("#000000", opacity=0.5)
    assert result == "&H80000000"


def test_hex_to_ass_color_accepts_no_hash():
    assert hex_to_ass_color("FF0000", opacity=1.0) == "&H000000FF"


def test_hex_to_ass_color_falls_back_to_white_for_bad_input():
    # Malformed → fallback to FFFFFF (white).
    assert hex_to_ass_color("XYZ", opacity=1.0) == "&H00FFFFFF"
