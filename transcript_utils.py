"""Small transcript helpers shared by the job pipeline and the API.

Kept dependency-free on purpose: ``app.py`` and the tests import these without
the ML stack that ``main.py`` pulls in.
"""
from __future__ import annotations


def has_words(transcript) -> bool:
    """True when at least one segment carries word-level timings."""
    return any((segment or {}).get("words")
               for segment in (transcript or {}).get("segments", []) or [])


def words_transcript(words, language: str = "en") -> dict:
    """A one-segment, clip-relative transcript from caption words.

    The shape the caption generators and the editor expect: one segment whose
    ``words`` are the caller's words, verbatim. Used for per-clip word timings
    (two-pass ASR) and for re-styling captions from stored ``caption_words``.
    """
    words = sorted(words, key=lambda w: float(w.get("start", 0.0)))
    if not words:
        return {"language": language, "segments": []}
    return {
        "language": language,
        "segments": [{
            "start": float(words[0]["start"]),
            "end": max(float(w["end"]) for w in words),
            # Words may or may not carry the leading space faster-whisper adds;
            # the burned captions use ``words``, this text is for readers.
            "text": " ".join(str(w.get("word", "")).strip() for w in words).strip(),
            "words": words,
        }],
    }
