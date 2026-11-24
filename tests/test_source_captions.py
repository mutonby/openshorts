"""The source-caption adapter for the two-pass ASR mode."""

import youtube_captions
import source_captions


def test_is_youtube_url():
    assert source_captions.is_youtube_url("https://youtu.be/x")
    assert source_captions.is_youtube_url("https://www.youtube.com/watch?v=x")
    assert source_captions.is_youtube_url("https://youtube-nocookie.com/embed/x")
    assert not source_captions.is_youtube_url("https://twitch.tv/x")
    assert not source_captions.is_youtube_url(None)


def _captions(language="ru", segments=None):
    return {
        "language": language,
        "segments": segments or [
            {"start": 1.0, "dur": 2.0, "text": " Привет"},
            {"start": 3.0, "dur": 1.0, "text": " мир"},
        ],
    }


def test_converts_captions_to_a_coarse_transcript(monkeypatch):
    monkeypatch.setattr(youtube_captions, "extract_video_id", lambda url: "abc12345678")
    seen = {}

    def fetch(video_id, lang=None):
        seen.update(video_id=video_id, lang=lang)
        return _captions()

    monkeypatch.setattr(youtube_captions, "fetch_transcript", fetch)
    monkeypatch.delenv("SOURCE_CAPTIONS_LANG", raising=False)

    transcript = source_captions.source_captions_transcript(
        "https://www.youtube.com/watch?v=abc12345678")

    assert seen == {"video_id": "abc12345678", "lang": None}
    assert transcript["language"] == "ru"
    assert transcript["segments"] == [
        {"start": 1.0, "end": 3.0, "text": "Привет"},
        {"start": 3.0, "end": 4.0, "text": "мир"},
    ]
    assert transcript["text"] == "Привет мир"


def test_requested_language_is_forwarded(monkeypatch):
    monkeypatch.setattr(youtube_captions, "extract_video_id", lambda url: "abc12345678")
    seen = {}

    def fetch(video_id, lang=None):
        seen["lang"] = lang
        return _captions(language="ru")

    monkeypatch.setattr(youtube_captions, "fetch_transcript", fetch)
    monkeypatch.setenv("SOURCE_CAPTIONS_LANG", "ru")

    transcript = source_captions.source_captions_transcript(
        "https://youtu.be/abc12345678", lang="ru-orig")

    assert seen["lang"] == "ru-orig"
    assert transcript["language"] == "ru"


def test_a_different_language_falls_back_to_asr(monkeypatch):
    monkeypatch.setattr(youtube_captions, "extract_video_id", lambda url: "abc12345678")
    monkeypatch.setattr(
        youtube_captions, "fetch_transcript",
        lambda video_id, lang=None: _captions(language="en"))
    monkeypatch.setenv("SOURCE_CAPTIONS_LANG", "ru")

    assert source_captions.source_captions_transcript(
        "https://youtu.be/abc12345678") is None


def test_no_video_id_or_a_caption_failure_returns_none(monkeypatch):
    monkeypatch.setattr(youtube_captions, "extract_video_id", lambda url: None)
    assert source_captions.source_captions_transcript("not a video") is None

    monkeypatch.setattr(youtube_captions, "extract_video_id", lambda url: "abc12345678")
    monkeypatch.setattr(
        youtube_captions, "fetch_transcript",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("403")))
    assert source_captions.source_captions_transcript(
        "https://youtu.be/abc12345678") is None


