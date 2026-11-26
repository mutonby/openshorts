"""A long recording is transcribed in bounded pieces (WHISPER_CHUNK_SECONDS).

faster-whisper computes the spectrogram of the whole voiced audio at once,
about half a megabyte per second of speech: an 11-hour stream needs ~19 GB
and the job dies of SIGKILL. These tests pin the piece boundaries, the
timeline the pieces are stitched back onto, and the language they share."""
import shutil
import subprocess
from types import SimpleNamespace

import pytest

import transcribe_backends as tb


def _segment(start, end, text):
    return SimpleNamespace(start=start, end=end, text=text,
                           words=[SimpleNamespace(word=f" {text}", start=start, end=end)])


def test_chunk_seconds_default_and_overrides(monkeypatch):
    monkeypatch.delenv("WHISPER_CHUNK_SECONDS", raising=False)
    assert tb.whisper_chunk_seconds() == 1800.0
    monkeypatch.setenv("WHISPER_CHUNK_SECONDS", "0")
    assert tb.whisper_chunk_seconds() == 0.0          # never split
    monkeypatch.setenv("WHISPER_CHUNK_SECONDS", "10")
    assert tb.whisper_chunk_seconds() == 300.0        # floor: pieces stay useful
    monkeypatch.setenv("WHISPER_CHUNK_SECONDS", "ten")
    assert tb.whisper_chunk_seconds() == 1800.0       # a typo is not a crash


def test_chunk_bounds_cover_the_recording_without_gaps(monkeypatch):
    monkeypatch.setattr(tb, "_quiet_point", lambda path, target, lo, hi: target + 3.0)
    bounds = tb.chunk_bounds("x.m4a", 4 * 3600.0, 1800.0)
    assert bounds[0][0] == 0.0 and bounds[-1][1] == 4 * 3600.0
    assert all(a[1] == b[0] for a, b in zip(bounds, bounds[1:]))
    assert all(hi - lo <= 1800 * 1.25 + 3 for lo, hi in bounds)
    # A remainder shorter than a quarter piece joins the last piece.
    assert len(tb.chunk_bounds("x.m4a", 1800 * 1.2, 1800.0)) == 1


def test_short_media_keeps_the_single_call_path(monkeypatch):
    calls = []
    monkeypatch.setattr(tb, "media_seconds", lambda path: 600.0)
    monkeypatch.setattr(tb, "run_whisper_transcription", lambda media, **kw: (
        calls.append((media, kw)) or ([_segment(1.0, 2.0, "hi")], SimpleNamespace(language="en"))))
    out = tb._transcribe_with_whisper("talk.mp4")
    assert calls[0][0] == "talk.mp4" and "offset" not in calls[0][1]
    assert out["segments"][0]["start"] == 1.0 and out["language"] == "en"


def test_long_media_is_stitched_onto_the_recording_timeline(monkeypatch):
    monkeypatch.setenv("WHISPER_CHUNK_SECONDS", "1800")
    monkeypatch.setattr(tb, "media_seconds", lambda path: 5000.0)
    monkeypatch.setattr(tb, "_quiet_point", lambda path, target, lo, hi: target)
    decoded = []
    monkeypatch.setattr(tb, "_decode_whisper_audio",
                        lambda path, start=None, duration=None: decoded.append((start, duration)) or f"pcm@{start}")
    seen = []

    def fake_run(audio, *, offset, progress, **params):
        seen.append((audio, offset, params.get("language")))
        # Local times, as faster-whisper reports them for an array input.
        return [_segment(5.0, 7.5, f"w{len(seen)}")], SimpleNamespace(language="es")

    monkeypatch.setattr(tb, "run_whisper_transcription", fake_run)
    out = tb._transcribe_with_whisper("stream.m4a")

    assert decoded == [(0.0, 1800.0), (1800.0, 1800.0), (3600.0, 1400.0)]
    assert [s["start"] for s in out["segments"]] == [5.0, 1805.0, 3605.0]
    assert out["segments"][1]["words"][0]["start"] == 1805.0
    # The first piece detects the language; the others are told it.
    assert [lang for _, _, lang in seen] == [None, "es", "es"]
    assert out["language"] == "es"
    assert out["text"] == "w1 w2 w3"


def test_progress_is_one_ladder_for_the_whole_recording(monkeypatch, capsys):
    monkeypatch.setattr(tb, "media_seconds", lambda path: 4000.0)
    monkeypatch.setattr(tb, "_quiet_point", lambda path, target, lo, hi: target)
    monkeypatch.setattr(tb, "_decode_whisper_audio", lambda path, start=None, duration=None: "pcm")

    def fake_run(audio, *, offset, progress, **params):
        progress.update(offset + 1000.0)
        return [], SimpleNamespace(language="en")

    monkeypatch.setattr(tb, "run_whisper_transcription", fake_run)
    tb._transcribe_with_whisper("stream.m4a")
    ladder = [line for line in capsys.readouterr().out.splitlines() if "Transcribing…" in line]
    assert [line.split("…")[1].split("%")[0].strip() for line in ladder] == ["25", "50", "75", "100"]


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="needs ffmpeg")
def test_quiet_point_finds_the_pause(tmp_path):
    """Tone, 1 s of silence at 30-31 s, tone: the cut lands in the silence."""
    wav = tmp_path / "gap.wav"
    subprocess.run([
        "ffmpeg", "-nostdin", "-loglevel", "error", "-y",
        "-f", "lavfi", "-i", "sine=frequency=440:duration=30:sample_rate=16000",
        "-f", "lavfi", "-i", "anullsrc=r=16000:cl=mono:d=1",
        "-f", "lavfi", "-i", "sine=frequency=440:duration=29:sample_rate=16000",
        "-filter_complex", "[0][1][2]concat=n=3:v=0:a=1", "-ac", "1", str(wav),
    ], check=True)
    cut = tb._quiet_point(str(wav), 40.0, 15.0, 55.0)
    assert 30.0 <= cut <= 31.0


def test_quiet_point_falls_back_to_the_target(tmp_path):
    assert tb._quiet_point(str(tmp_path / "missing.wav"), 100.0, 50.0, 150.0) == 100.0
