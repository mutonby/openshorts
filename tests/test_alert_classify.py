"""Failure classification for the high-failure-rate alert."""
from cloud.alerts import _classify_failure


def test_no_audio():
    assert _classify_failure("❌ NO_AUDIO: This video has no audio track.") == "no audio"


def test_youtube_bot_block():
    assert _classify_failure("ERROR: [youtube] x: Sign in to confirm you're not a bot") == "youtube download"
    assert _classify_failure("HTTP Error 403: Forbidden") == "youtube download"
    assert _classify_failure("HTTP Error 429: Too Many Requests") == "youtube download"


def test_proxy():
    assert _classify_failure("cannot connect to proxy") == "proxy"
    assert _classify_failure("402 payment required") == "proxy"


def test_transcription():
    assert _classify_failure("File faster_whisper/audio.py IndexError") == "transcription"
    assert _classify_failure("av/container/streams.py tuple index out of range") == "transcription"


def test_gemini():
    assert _classify_failure("google.genai error 500") == "gemini"


def test_ffmpeg():
    assert _classify_failure("ffmpeg failed during reframe") == "ffmpeg/render"


def test_unknown_is_mixed():
    assert _classify_failure("something weird happened") == "mixed"
