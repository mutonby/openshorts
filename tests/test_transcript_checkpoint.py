"""Transcript checkpoint: a job re-run after a container restart must not
transcribe (the slow, paid stage) a second time, and must never reuse the
transcript of a DIFFERENT video.

app.py re-enqueues an interrupted job with the same command and the same
output directory (resume manifest). main.py leaves the finished transcript
there, tied to the source's name and duration, picks it up on the re-run, and
removes it once the job completes.
"""
import json
import os

import pytest

main = pytest.importorskip("main")  # needs cv2/mediapipe, absent in minimal CI

TRANSCRIPT = {"text": "hola", "language": "es",
              "segments": [{"start": 0.0, "end": 1.0, "text": "hola", "words": []}]}
SRC = "/app/output/job1/video.mp4"
DUR = 58.64


def _save(d, transcript=TRANSCRIPT, src=SRC, dur=DUR):
    main.save_transcript_checkpoint(str(d), transcript, src, dur)


def _load(d, src=SRC, dur=DUR):
    return main.load_transcript_checkpoint(str(d), src, dur)


class TestRoundTrip:
    def test_same_source_is_reused(self, tmp_path):
        _save(tmp_path)
        assert os.path.isfile(tmp_path / main.TRANSCRIPT_CHECKPOINT)
        assert _load(tmp_path) == TRANSCRIPT

    def test_resumed_cloud_job_redownloads_to_the_same_name(self, tmp_path):
        # The re-run's file lives at the same path; a slightly different
        # duration reading (container decoders differ by a frame) still matches.
        _save(tmp_path)
        assert _load(tmp_path, dur=DUR + 0.3) == TRANSCRIPT

    def test_missing_checkpoint_is_none(self, tmp_path):
        assert _load(tmp_path) is None


class TestNeverTheWrongTranscript:
    def test_another_file_in_the_same_directory_is_ignored(self, tmp_path):
        # THE RISK: CLI run on A dies after transcribing, then B is processed
        # in the same directory without -o. B must get its own transcript.
        _save(tmp_path, src="/videos/A.mp4")
        assert _load(tmp_path, src="/videos/B.mp4") is None

    def test_same_name_but_different_length_is_ignored(self, tmp_path):
        _save(tmp_path, dur=600.0)
        assert _load(tmp_path, dur=58.6) is None

    def test_legacy_shape_without_source_is_ignored(self, tmp_path):
        (tmp_path / main.TRANSCRIPT_CHECKPOINT).write_text(json.dumps(TRANSCRIPT))
        assert _load(tmp_path) is None


class TestRobustness:
    def test_empty_transcript_is_ignored(self, tmp_path):
        _save(tmp_path, transcript={"segments": []})
        assert _load(tmp_path) is None

    def test_truncated_file_is_ignored(self, tmp_path):
        (tmp_path / main.TRANSCRIPT_CHECKPOINT).write_text('{"source": {"name": "vi')
        assert _load(tmp_path) is None

    def test_unwritable_directory_does_not_raise(self, tmp_path):
        main.save_transcript_checkpoint(str(tmp_path / "missing" / "dir"), TRANSCRIPT, SRC, DUR)

    def test_clear_is_idempotent(self, tmp_path):
        _save(tmp_path)
        main.clear_transcript_checkpoint(str(tmp_path))
        main.clear_transcript_checkpoint(str(tmp_path))
        assert _load(tmp_path) is None

    def test_checkpoint_is_hidden_from_job_listings(self):
        # app.py globs *_metadata.json and *.mp4 in the job dir; a dotfile
        # named unlike either can never be mistaken for a deliverable.
        assert main.TRANSCRIPT_CHECKPOINT.startswith(".")
        assert not main.TRANSCRIPT_CHECKPOINT.endswith("_metadata.json")
        assert not main.TRANSCRIPT_CHECKPOINT.endswith(".mp4")


class _FakeCapture:
    """25 fps, DUR seconds: what main reads to tie a checkpoint to its source."""
    def __init__(self, path):
        pass

    def get(self, prop):
        return 25.0 if prop == main.cv2.CAP_PROP_FPS else DUR * 25

    def release(self):
        pass


class TestPrepareOnly:
    """--prepare-only leaves the transcript for the run that renders the job
    (the remote worker): from the download cache or transcribed on this host."""

    @pytest.fixture
    def prep(self, tmp_path, monkeypatch):
        monkeypatch.setattr(main.cv2, "VideoCapture", _FakeCapture)
        calls = []

        def transcribe(path, slot_timeout=None):
            calls.append(path)
            assert slot_timeout == 45.0          # never waits for a slot for ever
            return TRANSCRIPT
        monkeypatch.setattr(main, "transcribe_video", transcribe)
        import source_cache
        monkeypatch.setattr(source_cache, "fetch_transcript", lambda url, dur: None)
        stored = []
        monkeypatch.setattr(source_cache, "store_transcript",
                            lambda *a: stored.append(a) or True)
        return calls, stored

    def test_transcribes_here_and_leaves_the_checkpoint(self, tmp_path, prep):
        calls, stored = prep
        assert main.prepare_transcript(SRC, str(tmp_path), "https://youtu.be/x", "1")
        assert calls == [SRC] and len(stored) == 1
        assert main.load_transcript_checkpoint(str(tmp_path), SRC, DUR) == TRANSCRIPT
        # A second prepare of the same job (a retry) does not transcribe again.
        assert main.prepare_transcript(SRC, str(tmp_path), "https://youtu.be/x", "1")
        assert calls == [SRC]

    def test_cache_mode_never_transcribes(self, tmp_path, prep, monkeypatch):
        calls, stored = prep
        assert not main.prepare_transcript(SRC, str(tmp_path), "https://youtu.be/x", "cache")
        assert calls == [] and _load(tmp_path) is None
        import source_cache
        monkeypatch.setattr(source_cache, "fetch_transcript", lambda url, dur: TRANSCRIPT)
        assert main.prepare_transcript(SRC, str(tmp_path), "https://youtu.be/x", "cache")
        assert calls == [] and stored == [] and _load(tmp_path) == TRANSCRIPT

    def test_a_failure_never_fails_the_prepare(self, tmp_path, prep, monkeypatch):
        def boom(path, slot_timeout=None):
            raise RuntimeError("no card")
        monkeypatch.setattr(main, "transcribe_video", boom)
        assert not main.prepare_transcript(SRC, str(tmp_path), None, "1")
        assert _load(tmp_path) is None

    def test_busy_slots_leave_it_to_the_renderer(self, tmp_path, prep, monkeypatch):
        def busy(path, slot_timeout=None):
            raise TimeoutError("no free transcription slot")
        monkeypatch.setattr(main, "transcribe_video", busy)
        assert not main.prepare_transcript(SRC, str(tmp_path), "https://youtu.be/x", "1")
        assert _load(tmp_path) is None
