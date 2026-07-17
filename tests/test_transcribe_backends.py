import json
import sys
import types
from types import SimpleNamespace

import pytest

import transcribe_backends as tb


class FakeFloat(float):
    """Stands in for numpy scalar types (float subclasses) from onnx-asr."""


def _seg(start, end, text, tokens, timestamps):
    return SimpleNamespace(
        start=FakeFloat(start), end=FakeFloat(end), text=text,
        tokens=tokens, timestamps=[FakeFloat(t) for t in timestamps],
    )


# --- token -> word reconstruction ------------------------------------------

def test_words_from_tokens_groups_by_leading_space():
    words = tb._words_from_tokens(
        [" T", "odo", " el", " mundo", "."],
        [0.0, 0.16, 0.32, 0.40, 0.60],
        seg_start=10.0, seg_end=12.0,
    )
    assert [w["word"] for w in words] == [" Todo", " el", " mundo."]
    # Absolute (segment-offset) start times.
    assert words[0]["start"] == pytest.approx(10.0)
    assert words[1]["start"] == pytest.approx(10.32)
    # End inferred from the next word's start; last word ends at segment end.
    assert words[0]["end"] == pytest.approx(10.32)
    assert words[2]["end"] == pytest.approx(11.0, abs=0.3)


def test_words_from_tokens_first_token_without_space_gets_one():
    words = tb._words_from_tokens([
        "Hola", " que", " tal"], [0.0, 0.5, 1.0], seg_start=0.0, seg_end=2.0)
    assert words[0]["word"] == " Hola"


def test_words_from_tokens_end_capped_after_long_silence():
    # Second word starts 5s later; first word's end must stay near its
    # own tokens (cap = last token + 0.6), not stretch across the gap.
    words = tb._words_from_tokens(
        [" uno", " dos"], [0.0, 5.0], seg_start=0.0, seg_end=6.0)
    assert words[0]["end"] <= 0.7


def test_words_from_tokens_all_floats_are_native():
    words = tb._words_from_tokens(
        [" a", "b", " c"], [0.0, 0.1, 0.2], seg_start=FakeFloat(1), seg_end=FakeFloat(2))
    for w in words:
        assert type(w["start"]) is float
        assert type(w["end"]) is float


# --- parakeet transcript assembly ------------------------------------------

@pytest.fixture
def fake_parakeet(monkeypatch):
    segs = [
        _seg(0.35, 2.43, "Todo el mundo habla.",
             [" Todo", " el", " mundo", " hab", "la", "."],
             [0.0, 0.32, 0.40, 0.56, 0.72, 0.80]),
        _seg(2.59, 5.57, "Pero, ¿qué es esto?",
             [" Pero", ",", " ¿", "qu", "é", " es", " esto", "?"],
             [0.0, 0.24, 0.32, 0.48, 0.64, 0.80, 1.04, 1.36]),
    ]
    model = SimpleNamespace(recognize=lambda path: iter(segs))
    monkeypatch.setattr(tb, "_get_parakeet_model", lambda: model)
    monkeypatch.setattr(tb, "_extract_wav", lambda path: "/tmp/fake.wav")
    monkeypatch.setattr(tb.os, "remove", lambda path: None)
    return segs


def test_parakeet_transcript_matches_contract(fake_parakeet, monkeypatch):
    monkeypatch.setattr(tb, "_detect_language", lambda text: "es")
    t = tb._transcribe_with_parakeet("video.mp4")

    assert t["text"] == "Todo el mundo habla. Pero, ¿qué es esto?"
    assert t["language"] == "es"
    assert len(t["segments"]) == 2

    seg = t["segments"][1]
    assert type(seg["start"]) is float and type(seg["end"]) is float
    # Continuations (",", "qu", "é") merged; leading spaces preserved.
    assert [w["word"] for w in seg["words"]] == [" Pero,", " ¿qué", " es", " esto?"]
    # Segment offset applied to word times.
    assert seg["words"][0]["start"] == pytest.approx(2.59)
    # Whole transcript is JSON-serializable (metadata json.dump path).
    json.dumps(t)


def test_parakeet_words_survive_merge_continuation_words(fake_parakeet, monkeypatch):
    from subtitles import merge_continuation_words
    monkeypatch.setattr(tb, "_detect_language", lambda text: "es")
    t = tb._transcribe_with_parakeet("video.mp4")
    for seg in t["segments"]:
        assert merge_continuation_words(seg["words"]) == seg["words"]


# --- fallback policy --------------------------------------------------------

def _valid_transcript(n_words=200, duration=120.0, language="es"):
    words = [
        {"word": f" w{i}", "start": i * duration / n_words,
         "end": (i + 1) * duration / n_words}
        for i in range(n_words)
    ]
    return {"text": "x " * n_words, "language": language,
            "segments": [{"start": 0.0, "end": duration, "text": "x", "words": words}]}


def test_fallback_reason_none_for_good_result():
    assert tb._parakeet_fallback_reason(_valid_transcript()) is None


def test_fallback_when_no_words():
    t = {"text": "", "language": "es",
         "segments": [{"start": 0, "end": 5, "text": "x", "words": []}]}
    assert "no words" in tb._parakeet_fallback_reason(t)


def test_fallback_when_language_unsupported():
    assert "outside" in tb._parakeet_fallback_reason(
        _valid_transcript(language="ja"))


def test_fallback_when_word_rate_absurdly_low():
    assert tb._parakeet_fallback_reason(
        _valid_transcript(n_words=5, duration=600.0)) is not None


def test_transcribe_media_falls_back_on_parakeet_exception(monkeypatch):
    def boom(path):
        raise RuntimeError("onnx exploded")

    sentinel = {"text": "ok", "language": "en", "segments": []}
    monkeypatch.setenv("TRANSCRIBE_BACKEND", "parakeet")
    monkeypatch.setattr(tb, "_transcribe_with_parakeet", boom)
    monkeypatch.setattr(tb, "_transcribe_with_whisper", lambda path: sentinel)
    assert tb.transcribe_media("video.mp4") is sentinel


def test_transcribe_media_default_is_whisper(monkeypatch):
    sentinel = {"text": "ok", "language": "en", "segments": []}
    monkeypatch.delenv("TRANSCRIBE_BACKEND", raising=False)
    monkeypatch.setattr(
        tb, "_transcribe_with_parakeet",
        lambda path: (_ for _ in ()).throw(AssertionError("should not run")))
    monkeypatch.setattr(tb, "_transcribe_with_whisper", lambda path: sentinel)
    assert tb.transcribe_media("video.mp4") is sentinel


# --- whisper singleton ------------------------------------------------------

@pytest.fixture
def fake_faster_whisper(monkeypatch):
    created = []

    class FakeModel:
        def __init__(self, model_size, device=None, compute_type=None):
            self.model_size = model_size
            self.device = device
            created.append(self)

        def transcribe(self, path, **params):
            segs = (s for s in [SimpleNamespace(
                start=0.0, end=1.0, text=" hola", words=None)])
            return segs, SimpleNamespace(language="es")

    fake_module = types.SimpleNamespace(WhisperModel=FakeModel)
    monkeypatch.setitem(sys.modules, "faster_whisper", fake_module)
    monkeypatch.setattr(tb, "_whisper_model", None)
    monkeypatch.setattr(tb, "_whisper_key", None)
    return created


def test_whisper_model_is_singleton(fake_faster_whisper, monkeypatch):
    monkeypatch.delenv("WHISPER_MODEL", raising=False)
    a, _ = tb._get_whisper_model()
    b, _ = tb._get_whisper_model()
    assert a is b
    assert len(fake_faster_whisper) == 1


def test_whisper_model_rebuilds_when_env_changes(fake_faster_whisper, monkeypatch):
    monkeypatch.setenv("WHISPER_MODEL", "small")
    a, _ = tb._get_whisper_model()
    monkeypatch.setenv("WHISPER_MODEL", "large-v3-turbo")
    b, _ = tb._get_whisper_model()
    assert a is not b
    assert b.model_size == "large-v3-turbo"


def test_run_whisper_transcription_materializes_segments(fake_faster_whisper, monkeypatch):
    monkeypatch.delenv("WHISPER_MODEL", raising=False)
    segments, info = tb.run_whisper_transcription("video.mp4")
    assert isinstance(segments, list)
    assert info.language == "es"
