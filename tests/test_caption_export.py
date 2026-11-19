"""Clip captions export as SRT/VTT and honour the CAPTION_OFFSET_SECONDS nudge.

The export reuses the same block grouping as the burned-in captions, so a
downloaded .srt/.vtt lines up with what is rendered, and a global sync offset
applied at `_collect_word_blocks` shifts both.
"""
import pytest

subtitles = pytest.importorskip("subtitles")


def _transcript():
    return {
        "language": "en",
        # ASR words carry a leading space to mark a real word boundary;
        # merge_continuation_words concatenates space-less fragments.
        "segments": [{"words": [
            {"word": "Hello", "start": 10.0, "end": 10.4},
            {"word": " world", "start": 10.5, "end": 11.0},
            {"word": " later", "start": 30.0, "end": 30.5},  # outside a 10-12s clip
        ]}],
    }


def test_srt_export_groups_and_times_the_clip_words(monkeypatch):
    monkeypatch.delenv("CAPTION_OFFSET_SECONDS", raising=False)
    srt = subtitles.clip_subtitle_text(_transcript(), 10.0, 12.0, fmt="srt")
    assert "Hello world" in srt
    assert "00:00:00,000 --> 00:00:01,000" in srt
    assert "later" not in srt            # word past clip_end is excluded


def test_vtt_export_has_header_and_dotted_times(monkeypatch):
    monkeypatch.delenv("CAPTION_OFFSET_SECONDS", raising=False)
    vtt = subtitles.clip_subtitle_text(_transcript(), 10.0, 12.0, fmt="vtt")
    assert vtt.startswith("WEBVTT")
    assert "00:00:00.000 --> 00:00:01.000" in vtt
    assert "Hello world" in vtt


def test_caption_offset_shifts_every_cue(monkeypatch):
    monkeypatch.setenv("CAPTION_OFFSET_SECONDS", "0.5")
    srt = subtitles.clip_subtitle_text(_transcript(), 10.0, 12.0, fmt="srt")
    assert "00:00:00,500 --> 00:00:01,500" in srt


def test_bad_offset_env_falls_back_to_zero(monkeypatch):
    monkeypatch.setenv("CAPTION_OFFSET_SECONDS", "not-a-number")
    assert subtitles.caption_offset_seconds() == 0.0


def test_empty_clip_returns_empty_or_bare_header(monkeypatch):
    monkeypatch.delenv("CAPTION_OFFSET_SECONDS", raising=False)
    empty = {"language": "en", "segments": []}
    assert subtitles.clip_subtitle_text(empty, 0.0, 5.0, fmt="srt") == ""
    assert subtitles.clip_subtitle_text(empty, 0.0, 5.0, fmt="vtt") == "WEBVTT\n\n"
