"""Word-timed transcription on the CPU: faster-whisper int8, openai-whisper as a fallback.

On a 2-core laptop ``base`` transcribes a 45 s clip in well under a minute.
The model is loaded once and kept (about 150 MB for ``base`` int8).
"""
from __future__ import annotations

import os
import threading
from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass(frozen=True)
class Word:
    start: float
    end: float
    text: str


_lock = threading.Lock()
_model = None
_model_name = None
_backend = None

# Whisper's favourite hallucinations on music or silence.
_HALLUCINATIONS = ("thanks for watching", "thank you for watching", "subscribe",
                   "please like", "amara.org", "字幕")


def _load(name: str):
    global _model, _model_name, _backend
    if _model is not None and _model_name == name:
        return _model, _backend
    threads = max(1, min(4, (os.cpu_count() or 2)))
    try:
        from faster_whisper import WhisperModel
        _model = WhisperModel(name, device="cpu", compute_type="int8", cpu_threads=threads)
        _backend = "faster-whisper"
    except ImportError:
        import whisper  # openai-whisper
        _model = whisper.load_model(name, device="cpu")
        _backend = "openai-whisper"
    _model_name = name
    return _model, _backend


def available() -> Optional[str]:
    for mod in ("faster_whisper", "whisper"):
        try:
            __import__(mod)
            return mod
        except ImportError:
            continue
    return None


def clean_words(words: List[Word]) -> List[Word]:
    out = []
    for w in words:
        t = w.text.strip()
        if not t or w.end <= w.start:
            continue
        out.append(Word(round(w.start, 3), round(w.end, 3), t))
    text = " ".join(w.text for w in out).lower()
    if out and len(out) <= 8 and any(h in text for h in _HALLUCINATIONS):
        return []
    return out


def read_wav(path: str):
    """A 16-bit PCM WAV as float32 samples in [-1, 1] (mono: channels averaged).

    Handing faster-whisper the samples skips its PyAV decoder, which breaks
    whenever PyAV and faster-whisper versions drift apart.
    """
    import wave
    import numpy as np
    with wave.open(path, "rb") as wf:
        if wf.getsampwidth() != 2:
            raise ValueError("expected 16-bit PCM")
        if wf.getframerate() != 16000:
            raise ValueError("expected 16 kHz audio")
        ch = wf.getnchannels()
        data = np.frombuffer(wf.readframes(wf.getnframes()), dtype="<i2").astype(np.float32)
    if ch > 1:
        data = data.reshape(-1, ch).mean(axis=1)
    return data / 32768.0


def transcribe(path: str, model: str = "base", language: str = "auto") -> Tuple[List[Word], str]:
    """(words, language) of a 16 kHz mono WAV. ([], "") when it cannot."""
    if model == "off":
        return [], ""
    lang = None if language in ("", "auto") else language
    with _lock:
        m, backend = _load(model)
        if backend == "faster-whisper":
            audio = read_wav(path) if str(path).lower().endswith(".wav") else path
            segments, info = m.transcribe(audio, language=lang, word_timestamps=True,
                                          vad_filter=True, beam_size=1,
                                          condition_on_previous_text=False)
            words = []
            for seg in segments:
                for w in seg.words or []:
                    if (w.probability or 0) >= 0.2:
                        words.append(Word(w.start, w.end, w.word))
            return clean_words(words), info.language or ""
        res = m.transcribe(path, language=lang, word_timestamps=True, fp16=False,
                           condition_on_previous_text=False)
        words = [Word(w["start"], w["end"], w["word"])
                 for seg in res.get("segments", []) for w in seg.get("words", [])]
        return clean_words(words), res.get("language", "")
