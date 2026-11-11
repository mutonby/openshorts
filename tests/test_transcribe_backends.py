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
    monkeypatch.setattr(tb, "_has_audio_stream", lambda path: True)
    monkeypatch.setattr(tb, "_transcribe_with_parakeet", boom)
    monkeypatch.setattr(tb, "_transcribe_with_whisper", lambda path: sentinel)
    assert tb.transcribe_media("video.mp4") is sentinel


def test_transcribe_media_default_is_whisper(monkeypatch):
    sentinel = {"text": "ok", "language": "en", "segments": []}
    monkeypatch.delenv("TRANSCRIBE_BACKEND", raising=False)
    monkeypatch.setattr(tb, "_has_audio_stream", lambda path: True)
    monkeypatch.setattr(
        tb, "_transcribe_with_parakeet",
        lambda path: (_ for _ in ()).throw(AssertionError("should not run")))
    monkeypatch.setattr(tb, "_transcribe_with_whisper", lambda path: sentinel)
    assert tb.transcribe_media("video.mp4") is sentinel


def test_transcribe_media_raises_on_silent_video(monkeypatch):
    monkeypatch.setattr(tb, "_has_audio_stream", lambda path: False)
    with pytest.raises(tb.NoAudioError):
        tb.transcribe_media("silent.mp4")


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


# --- release_models: the API process must not keep the models resident ------

class _FakeCT2:
    def __init__(self):
        self.unloaded = False

    def unload_model(self, to_cpu=False):
        self.unloaded = True


class _FakeWhisper:
    def __init__(self):
        self.model = _FakeCT2()


def test_release_models_drops_both_singletons_and_unloads_whisper(monkeypatch):
    whisper = _FakeWhisper()
    monkeypatch.setattr(tb, "_whisper_model", whisper)
    monkeypatch.setattr(tb, "_whisper_key", ("large-v3-turbo", "cuda", "float16"))
    monkeypatch.setattr(tb, "_parakeet_model", object())

    tb.release_models()

    assert tb._whisper_model is None and tb._whisper_key is None
    assert tb._parakeet_model is None
    assert whisper.model.unloaded  # ctranslate2 frees VRAM on unload, not on GC


def test_release_models_hands_every_gate_slot_back():
    tb.release_models()
    # Every slot is free again: a transcription can start right after.
    for _ in range(tb._ASR_SLOTS):
        assert tb._ASR_GATE.acquire(blocking=False)
    for _ in range(tb._ASR_SLOTS):
        tb._ASR_GATE.release()


def test_release_models_is_a_no_op_when_nothing_is_loaded(monkeypatch):
    monkeypatch.setattr(tb, "_whisper_model", None)
    monkeypatch.setattr(tb, "_parakeet_model", None)
    tb.release_models()
    assert tb._whisper_model is None and tb._parakeet_model is None


def test_whisper_model_is_fetched_inside_the_gate(monkeypatch):
    """release_models drains the gate before unloading; that only protects a
    transcription if the model is taken *after* the gate is held."""
    order = []

    class _Gate:
        def __enter__(self):
            order.append("gate")
            return self

        def __exit__(self, *a):
            return False

    class _Model:
        def transcribe(self, path, **params):
            order.append("transcribe")
            return iter([]), SimpleNamespace(duration=0, language="en")

    monkeypatch.setattr(tb, "_ASR_GATE", _Gate())
    monkeypatch.setattr(tb, "_whisper_device", lambda: "cuda")
    monkeypatch.setattr(tb, "_get_whisper_model",
                        lambda: (order.append("model"), (_Model(), "cuda"))[1])

    tb._run_whisper_once("x.mp4")

    assert order == ["gate", "model", "transcribe"]


def test_release_models_drops_the_transnetv2_singleton_too(monkeypatch):
    import types
    fake = types.ModuleType("scene_detection")
    fake._tn2_model = object()
    monkeypatch.setitem(sys.modules, "scene_detection", fake)
    monkeypatch.setattr(tb, "_whisper_model", None)
    monkeypatch.setattr(tb, "_parakeet_model", None)

    tb.release_models()

    assert fake._tn2_model is None


class TestHostAsrSlot:
    """Cross-process cap on GPU transcriptions (flock slots on the shared disk)."""

    def test_slots_are_exclusive_and_released(self, tmp_path):
        import transcribe_backends as tb
        a = tb.host_asr_slot(slots=2, lock_dir=str(tmp_path), poll=0.01)
        b = tb.host_asr_slot(slots=2, lock_dir=str(tmp_path), poll=0.01)
        with a, b:
            assert a._fh is not None and b._fh is not None
            assert a._fh.name != b._fh.name
        assert a._fh is None and b._fh is None

    def test_third_waits_until_one_frees(self, tmp_path):
        import threading, time
        import transcribe_backends as tb
        first = tb.host_asr_slot(slots=1, lock_dir=str(tmp_path), poll=0.01)
        first.__enter__()
        got = []

        def worker():
            with tb.host_asr_slot(slots=1, lock_dir=str(tmp_path), poll=0.01):
                got.append(time.monotonic())

        t = threading.Thread(target=worker)
        t.start()
        time.sleep(0.1)
        assert got == []          # still blocked behind the first holder
        released = time.monotonic()
        first.__exit__(None, None, None)
        t.join(2)
        assert got and got[0] >= released

    def test_disabled_is_a_noop(self, tmp_path):
        import transcribe_backends as tb
        with tb.host_asr_slot(slots=0, lock_dir=str(tmp_path)) as s:
            assert s._fh is None

    def test_the_wait_for_a_slot_is_counted(self, tmp_path, monkeypatch):
        """JOB_CPU reports it (asr_wait_s) to size ASR_HOST_SLOTS from data."""
        import threading, time
        import transcribe_backends as tb
        monkeypatch.setattr(tb, "ASR_WAIT", {"s": 0.0})
        first = tb.host_asr_slot(slots=1, lock_dir=str(tmp_path), poll=0.01)
        first.__enter__()
        threading.Timer(0.2, first.__exit__, args=(None, None, None)).start()
        with tb.host_asr_slot(slots=1, lock_dir=str(tmp_path), poll=0.01):
            pass
        assert 0.15 <= tb.ASR_WAIT["s"] < 2


class TestParakeetMemorySettings:
    def test_vad_batch_default_is_the_benchmarked_one(self):
        import transcribe_backends as tb
        assert tb.PARAKEET_VAD_BATCH == 4

    def test_cuda_provider_is_lean_with_cpu_fallback(self):
        import transcribe_backends as tb
        (name, opts), cpu = tb.parakeet_providers()
        assert name == "CUDAExecutionProvider" and cpu == "CPUExecutionProvider"
        assert opts["arena_extend_strategy"] == "kSameAsRequested"
        assert opts["cudnn_conv_use_max_workspace"] == "0"


def test_parakeet_threads_sleep_and_the_vad_runs_on_one_cpu_thread(monkeypatch):
    """Spinning ORT threads were most of a job's CPU, and Silero on CUDA most
    of a transcription's wall time."""
    class FakeOptions:
        def __init__(self):
            self.entries = {}
            self.intra_op_num_threads = 0
            self.inter_op_num_threads = 0

        def add_session_config_entry(self, key, value):
            self.entries[key] = value

    calls = {}

    class FakeModel:
        def with_vad(self, vad, batch_size):
            return self

        def with_timestamps(self):
            return self

    def load_model(model_id, **kw):
        calls["model"] = kw
        return FakeModel()

    def load_vad(*a, **kw):
        calls["vad"] = (a, kw)
        return object()

    monkeypatch.setitem(sys.modules, "onnxruntime", SimpleNamespace(SessionOptions=FakeOptions))
    monkeypatch.setitem(sys.modules, "onnx_asr",
                        SimpleNamespace(load_model=load_model, load_vad=load_vad))
    monkeypatch.setattr(tb, "_parakeet_model", None)

    tb._get_parakeet_model()

    no_spin = {"session.intra_op.allow_spinning": "0",
               "session.inter_op.allow_spinning": "0"}
    assert calls["model"]["sess_options"].entries == no_spin
    args, kw = calls["vad"]
    assert args == ("silero",)
    assert kw["providers"] == ["CPUExecutionProvider"]
    assert kw["sess_options"].entries == no_spin
    assert kw["sess_options"].intra_op_num_threads == 1


class TestOnnxruntimeDevice:
    """A CPU onnxruntime wheel on top of onnxruntime-gpu turns the CUDA request
    into a warning; the job must still be able to tell where Parakeet ran."""

    def test_cuda_build(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "onnxruntime", SimpleNamespace(
            get_available_providers=lambda: [
                "TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"]))
        assert tb.onnxruntime_device() == "cuda"

    def test_cpu_build(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "onnxruntime", SimpleNamespace(
            get_available_providers=lambda: ["AzureExecutionProvider", "CPUExecutionProvider"]))
        assert tb.onnxruntime_device() == "cpu"

    def test_broken_import_counts_as_cpu(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "onnxruntime", None)  # import raises
        assert tb.onnxruntime_device() == "cpu"

    @pytest.mark.parametrize("providers,device,warned", [
        (["CUDAExecutionProvider", "CPUExecutionProvider"], "cuda", False),
        (["AzureExecutionProvider", "CPUExecutionProvider"], "cpu", True),
    ])
    def test_model_load_records_the_device(self, monkeypatch, capsys,
                                           providers, device, warned):
        class FakeOptions:
            intra_op_num_threads = inter_op_num_threads = 0

            def add_session_config_entry(self, key, value):
                pass

        class FakeModel:
            def with_vad(self, vad, batch_size):
                return self

            def with_timestamps(self):
                return self

        monkeypatch.setitem(sys.modules, "onnxruntime", SimpleNamespace(
            SessionOptions=FakeOptions, get_available_providers=lambda: providers))
        monkeypatch.setitem(sys.modules, "onnx_asr", SimpleNamespace(
            load_model=lambda *a, **kw: FakeModel(), load_vad=lambda *a, **kw: object()))
        monkeypatch.setattr(tb, "_parakeet_model", None)
        monkeypatch.setattr(tb, "PARAKEET_DEVICE", None)

        tb._get_parakeet_model()

        assert tb.PARAKEET_DEVICE == device
        assert ("parakeet runs on the CPU" in capsys.readouterr().out) is warned


# --- script drift repair -----------------------------------------------------

def _drift_transcript():
    return {
        "text": "Hello there. Тудей Марк із гана такі сторі. Bye now.",
        "language": "en",
        "segments": [
            {"start": 0.0, "end": 1.0, "text": "Hello there.",
             "words": [{"word": " Hello", "start": 0.0, "end": 0.5},
                       {"word": " there.", "start": 0.5, "end": 1.0}]},
            {"start": 2.0, "end": 4.0, "text": "Тудей Марк із гана такі сторі.",
             "words": [{"word": " Тудей", "start": 2.0, "end": 2.5}]},
            {"start": 5.0, "end": 6.0, "text": "Bye now.",
             "words": [{"word": " Bye", "start": 5.0, "end": 5.4},
                       {"word": " now.", "start": 5.4, "end": 6.0}]},
        ],
    }


def test_segment_drifted_by_language_script():
    assert tb._segment_drifted("Тудей Марк", "en")
    assert tb._segment_drifted("My job was to unify the whole компані", "es")
    assert not tb._segment_drifted("Hello there", "en")
    # Latin brand names inside a Cyrillic-script language are not drift.
    assert not tb._segment_drifted("Мы используем YouTube", "ru")
    assert tb._segment_drifted("Καλημέρα Марк", "el")


def test_repair_replaces_only_drifted_segments(monkeypatch):
    import numpy as np
    calls = []
    monkeypatch.setattr(tb, "_read_wav_16k", lambda path: np.zeros(16000 * 7, dtype=np.float32))

    def fake_span(audio, language):
        calls.append((len(audio), language))
        return "Today Mark is gonna tell his story.", [
            {"word": " Today", "start": 0.15, "end": 0.6},
            {"word": " story.", "start": 1.5, "end": 9.0},  # runs past the next segment
        ]

    monkeypatch.setattr(tb, "_whisper_span", fake_span)
    t = _drift_transcript()
    assert tb._repair_script_drift(t, "/tmp/x.wav") == 1
    assert calls == [(int(4.15 * 16000) - int(1.85 * 16000), "en")]
    seg = t["segments"][1]
    assert seg["text"] == "Today Mark is gonna tell his story."
    assert seg["words"][0]["start"] == pytest.approx(2.0)
    # Clamped so the transcript stays chronological.
    assert seg["words"][-1]["end"] <= t["segments"][2]["start"]
    assert "Тудей" not in t["text"] and t["text"].startswith("Hello there.")
    assert t["segments"][0]["text"] == "Hello there."


def test_repair_is_a_noop_without_drift(monkeypatch):
    monkeypatch.setattr(tb, "_read_wav_16k", lambda path: pytest.fail("must not read audio"))
    t = _drift_transcript()
    t["segments"].pop(1)
    assert tb._repair_script_drift(t, "/tmp/x.wav") == 0


def test_repair_failure_keeps_parakeet_text(monkeypatch):
    import numpy as np
    monkeypatch.setattr(tb, "_read_wav_16k", lambda path: np.zeros(16000 * 7, dtype=np.float32))

    def boom(audio, language):
        raise RuntimeError("CUDA out of memory")

    monkeypatch.setattr(tb, "_whisper_span", boom)
    t = _drift_transcript()
    assert tb._repair_script_drift(t, "/tmp/x.wav") == 0
    assert t["segments"][1]["text"].startswith("Тудей")
