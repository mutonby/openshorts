import json

import pytest
from starlette.testclient import TestClient

app = pytest.importorskip("app")


@pytest.fixture
def saved_clip(tmp_path, monkeypatch):
    job = tmp_path / "job-1"
    job.mkdir()
    metadata = job / "video_metadata.json"
    monkeypatch.setattr(app, "OUTPUT_DIR", str(tmp_path))
    monkeypatch.setattr(app, "jobs", {"job-1": {"status": "completed"}})

    def write(clip, words):
        metadata.write_text(json.dumps({
            "shorts": [clip],
            "transcript": {"language": "fr", "segments": [{"words": words}]},
        }), encoding="utf-8")
        return TestClient(app.app)

    return write


def test_srt_exports_original_words_with_clipped_millisecond_timestamps(saved_clip):
    client = saved_clip({"start": 10.0, "end": 11.432, "caption_style": {"max_chars": 1}}, [
        {"word": " Hors-champ", "start": 9.0, "end": 9.5},
        {"word": " Bon", "start": 9.8, "end": 10.123},
        {"word": "jour", "start": 10.123, "end": 10.472},
        {"word": " Zoë", "start": 11.051, "end": 11.802},
    ])

    response = client.get("/api/clip/job-1/0/captions")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/x-subrip")
    assert response.headers["content-disposition"] == 'attachment; filename="job-1_clip_01.srt"'
    assert "no-store" in response.headers["cache-control"]
    assert response.text == (
        "1\n00:00:00,000 --> 00:00:00,472\nBonjour\n\n"
        "2\n00:00:01,051 --> 00:00:01,432\nZoë\n\n"
    )


def test_transcript_preview_counts_a_continuation_as_part_of_its_word(saved_clip):
    client = saved_clip({"start": 10.0, "end": 12.0, "endcard": {"duration": 2}}, [
        {"word": " YouTube", "start": 9.75, "end": 10.0},
        {"word": "-Kanal", "start": 10.0, "end": 10.5},
        {"word": " Beispiel", "start": 10.75, "end": 11.25},
    ])

    response = client.get("/api/clip/job-1/0/transcript")

    assert response.status_code == 200
    assert response.json() == {
        "durationSec": 2.0,
        "language": "fr",
        "captions": [
            {"text": "YouTube-Kanal", "startMs": 0, "endMs": 500},
            {"text": "Beispiel", "startMs": 750, "endMs": 1250},
        ],
    }


def test_plain_text_follows_the_saved_cut_order(saved_clip):
    client = saved_clip({
        "start": 10, "end": 21,
        "recipe": {"segments": [{"start": 20, "end": 21}, {"start": 10, "end": 11}]},
    }, [
        {"word": " Un", "start": 10.1, "end": 10.4},
        {"word": " Hors-champ", "start": 15.1, "end": 15.4},
        {"word": " Deux", "start": 20.1, "end": 20.4},
    ])

    response = client.get("/api/clip/job-1/0/captions?format=txt")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/plain")
    assert response.text.split() == ["Deux", "Un"]


def test_export_uses_saved_caption_words_on_the_edited_timeline(saved_clip):
    client = saved_clip({
        "start": 10, "end": 31,
        "recipe": {"segments": [{"start": 10, "end": 11}, {"start": 30, "end": 31}]},
        "caption_words": [{"word": " Corrigé", "start": 1.12, "end": 2.25}],
    }, [{"word": " Brouillon", "start": 10.0, "end": 10.5}])

    response = client.get("/api/clip/job-1/0/captions")

    assert response.status_code == 200
    assert "00:00:01,120 --> 00:00:02,000\nCorrigé" in response.text
    assert "Brouillon" not in response.text


def test_empty_clip_and_unavailable_format_return_clear_errors(saved_clip):
    client = saved_clip({"start": 10, "end": 11}, [])

    assert client.get("/api/clip/job-1/0/captions").json() == {"detail": "This clip has no caption text."}
    assert client.get("/api/clip/job-1/-1/captions").status_code == 404
    assert client.get("/api/clip/job-1/0/captions?format=csv").status_code == 422


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
