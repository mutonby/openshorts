from transcript_utils import has_words, words_transcript


def test_has_words_detects_word_level_segments():
    assert has_words({"segments": [{"text": "hi", "words": [{"word": " hi"}]}]})
    assert not has_words({"segments": [{"text": "hi"}]})
    assert not has_words({"segments": []})
    assert not has_words(None)


def test_words_transcript_sorts_and_builds_one_segment():
    transcript = words_transcript([
        {"word": " world", "start": 1.5, "end": 2.0},
        {"word": " Hello", "start": 1.0, "end": 1.5},
    ], language="en")
    segment = transcript["segments"][0]
    assert transcript["language"] == "en"
    assert [word["word"] for word in segment["words"]] == [" Hello", " world"]
    assert segment["start"] == 1.0
    assert segment["end"] == 2.0
    assert segment["text"] == "Hello world"


def test_words_transcript_without_words_is_empty():
    assert words_transcript([]) == {"language": "en", "segments": []}
