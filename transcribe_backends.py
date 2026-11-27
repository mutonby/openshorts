"""Transcription backends: NVIDIA Parakeet (onnx-asr) with faster-whisper fallback.

Every caller goes through transcribe_media(), which returns the transcript
contract the whole pipeline depends on:

    {
      "text": str,          # full punctuated transcript
      "language": str,      # whisper-style short code ("es", "en", ...)
      "segments": [
        {"start": float, "end": float, "text": str,
         "words": [{"word": str, "start": float, "end": float}, ...]},
      ],
    }

Invariants the consumers rely on (clip cutting, karaoke subtitles, Remotion):
  - word["word"] carries a LEADING SPACE on true word starts; continuation
    fragments are merged into their base word (merge_continuation_words).
  - all numerics are native Python floats (json.dump of the transcript).
  - words sorted by start, segments chronological, absolute file timestamps.

TRANSCRIBE_BACKEND env: "whisper" (default) | "parakeet".
The parakeet path falls back to whisper automatically when the model errors,
produces no usable words, or the detected language is outside its 25
supported European languages (e.g. Japanese/Chinese/Arabic uploads).
GPU Whisper retries accelerator failures with multilingual small INT8 on CPU.
WHISPER_BATCH_SIZE enables an optional batched decoding pass.
"""
import os
import re
import subprocess
import sys
import tempfile
import threading
import time

from subtitles import (
    DEFAULT_WHISPER_MODEL,
    get_whisper_config,
    WHISPER_TRANSCRIBE_PARAMS,
    merge_continuation_words,
)

PARAKEET_MODEL_ID = "nemo-parakeet-tdt-0.6b-v3"

# The 25 European languages parakeet-tdt-0.6b-v3 supports (ISO 639-1).
PARAKEET_LANGS = {
    "bg", "hr", "cs", "da", "nl", "en", "et", "fi", "fr", "de", "el", "hu",
    "it", "lv", "lt", "mt", "pl", "pt", "ro", "sk", "sl", "es", "sv", "ru",
    "uk",
}

# Serializes GPU transcription across concurrent jobs so N jobs can't stack
# N model contexts / decode batches in VRAM. CPU whisper stays ungated
# (CTranslate2 models are thread-safe and that matches the old behavior).
_ASR_SLOTS = int(os.environ.get("ASR_GPU_CONCURRENCY", "1"))
_ASR_GATE = threading.Semaphore(_ASR_SLOTS)


# Host-wide cap on GPU transcriptions, across job processes AND across the two
# containers of a deploy handover (the lock files live on the shared output/
# volume). _ASR_GATE above only serialises threads inside one process; every
# main.py job is its own process, so eight jobs could all load Parakeet at
# once. Peak per transcription is ~4-6 GB on a 20 GB card (prod, 22-sep-2026).
ASR_HOST_SLOTS = int(os.environ.get("ASR_HOST_SLOTS", "2"))
ASR_LOCK_DIR = os.environ.get("ASR_LOCK_DIR", "output")
# Seconds this process waited for a host slot (main.py reports it in JOB_CPU).
ASR_WAIT = {"s": 0.0}
# Media at least this long (seconds) may not take the last host slot, which
# stays for shorter sources. In prod on 18-nov-2026 two long videos (50 and
# 88 min, languages Parakeet does not cover) held both slots for an hour while
# eight jobs, one of them a 3 MB clip, slept waiting for one. 0 turns it off.
ASR_LONG_SECONDS = float(os.environ.get("ASR_LONG_SECONDS", "1800") or 0)


class GpuUnavailable(Exception):
    """The transcription would have to leave the card, and this process may
    only transcribe on it (``require_gpu``)."""


# Set by require_gpu(): a CUDA failure raises GpuUnavailable instead of
# retrying whisper on the CPU.
_gpu_only = False


def require_gpu():
    """From now on this process transcribes on the card or not at all.

    For ``main.py --prepare-only``: a transcript prepared for another host is
    only worth it on the GPU. Prepare processes that fell back to CPU
    whisper filled the host's RAM and CPU while the renderer could have
    transcribed for itself, and each kept its CUDA context on the card until
    it exited (prod, 18-nov-2026)."""
    global _gpu_only
    _gpu_only = True


class host_asr_slot:
    """``with host_asr_slot():`` holds one of ASR_HOST_SLOTS flock slots.

    Blocks (polling once a second) until a slot is free. A crashed holder
    releases its lock with its file descriptor, so a slot can never leak.
    Degrades to a no-op where flock or the directory is unavailable.
    ``timeout``: give up with TimeoutError after that many seconds without a
    slot (a caller that has another way to get the work done).
    ``media_seconds``: length of what will be transcribed. Media of at least
    ``long_seconds`` never takes the last slot (with two or more slots), so a
    short source never waits behind long ones only.
    """

    def __init__(self, slots=None, lock_dir=None, poll=1.0, timeout=None,
                 media_seconds=None, long_seconds=None):
        self.slots = ASR_HOST_SLOTS if slots is None else slots
        self.lock_dir = lock_dir or ASR_LOCK_DIR
        self.poll = poll
        self.timeout = timeout
        self.media_seconds = media_seconds
        self.long_seconds = ASR_LONG_SECONDS if long_seconds is None else long_seconds
        self._fh = None

    def usable_slots(self):
        """How many of the slots (from index 0) this transcription may take."""
        long_media = (self.long_seconds and self.media_seconds
                      and self.media_seconds >= self.long_seconds)
        if long_media and self.slots >= 2:
            return self.slots - 1
        return self.slots

    def __enter__(self):
        if self.slots <= 0:
            return self
        try:
            import fcntl
            os.makedirs(self.lock_dir, exist_ok=True)
        except Exception:
            return self
        announced = False
        started = time.monotonic()
        while True:
            for i in range(self.usable_slots()):
                path = os.path.join(self.lock_dir, f".asr-gpu-{i}.lock")
                try:
                    fh = open(path, "a+")
                except OSError:
                    return self
                try:
                    fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    self._fh = fh
                    ASR_WAIT["s"] += time.monotonic() - started
                    return self
                except OSError:
                    fh.close()
            if self.timeout is not None and time.monotonic() - started >= self.timeout:
                ASR_WAIT["s"] += time.monotonic() - started
                raise TimeoutError("no free transcription slot")
            if not announced:
                print("🎙️ Waiting for a free transcription slot…", flush=True)
                announced = True
            time.sleep(self.poll)

    def release(self):
        """Give the slot back (idempotent; ``__exit__`` calls it too)."""
        if self._fh is not None:
            try:
                import fcntl
                fcntl.flock(self._fh, fcntl.LOCK_UN)
            except Exception:
                pass
            self._fh.close()
            self._fh = None

    def __exit__(self, *exc):
        self.release()
        return False


class _NullGate:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


_NULL_GATE = _NullGate()


class _TranscribeProgress:
    """Emits '🎙️ Transcribing… NN% (Xs)' lines at 25% steps.

    These are the only transcription lines cloud users see (log_view keeps
    them), so they must stay free of technical detail.
    """

    def __init__(self, total_seconds):
        self.total = max(float(total_seconds or 0), 0.0)
        self.started = time.time()
        self.next_pct = 25

    def update(self, position_seconds):
        if self.total <= 0:
            return
        pct = min(int(position_seconds / self.total * 100), 100)
        while pct >= self.next_pct and self.next_pct <= 100:
            elapsed = int(time.time() - self.started)
            print(f"🎙️ Transcribing… {self.next_pct}% ({elapsed}s)", flush=True)
            self.next_pct += 25

# --- whisper singleton ------------------------------------------------------

_whisper_model = None
_whisper_key = None
_whisper_lock = threading.Lock()
# Set after a CUDA failure (e.g. VRAM exhausted by other models on the GPU)
# so every later transcription goes straight to CPU instead of re-failing.
# A transcription that finds the card full waits for room and tries the card
# again before it gives up on it. Each host slot may hold a Parakeet of
# several GB next to the other jobs' own contexts, so the card fills for the
# seconds those take; the next one in then got "out of memory" and decoded its
# whole source with whisper on the CPU, on several cores for as long as the
# video lasts. ``ASR_CARD_WAIT`` is how long it waits (seconds, 0 = never).
ASR_CARD_WAIT = float(os.environ.get("ASR_CARD_WAIT", "120") or 0)
PARAKEET_CARD_MB = int(os.environ.get("PARAKEET_CARD_MB", "5500") or 0)
WHISPER_CARD_MB = int(os.environ.get("WHISPER_CARD_MB", "3500") or 0)
_CARD_FULL_HINTS = ("out of memory", "failed to allocate memory")


def _card_free_mb():
    """Free memory on the card in MB (None: no card, or it cannot be read)."""
    try:
        import gpu_decode
        return gpu_decode._free_mb()
    except Exception:
        return None


def _card_full(error):
    text = str(error).lower()
    return any(hint in text for hint in _CARD_FULL_HINTS)


def wait_for_card(need_mb, seconds=None, sleep=time.sleep, poll=3.0):
    """True once the card has ``need_mb`` free, waiting up to ``seconds``.

    False when the wait is off, the time is up or the card cannot be read:
    the caller then does what it did before there was a wait."""
    seconds = ASR_CARD_WAIT if seconds is None else seconds
    if seconds <= 0 or need_mb <= 0:
        return False
    waited = 0.0
    while True:
        free = _card_free_mb()
        if free is None:
            return False
        if free >= need_mb:
            return True
        if waited >= seconds:
            return False
        if not waited:
            print(f"⏳ [ASR] the card is full ({free} MB free, {need_mb} needed) "
                  f"— waiting for room", flush=True)
        step = min(poll, seconds - waited)
        sleep(step)
        waited += step


_whisper_force_cpu = False


def whisper_cpu_threads():
    """Threads for whisper on the CPU (``WHISPER_CPU_THREADS``, default
    min(8, cores)). Job processes run with OMP_NUM_THREADS=1, and ctranslate2
    takes that as its default: measured in prod on 18-nov-2026, a fallen-back
    large-v3-turbo decoded on one thread at about real time, so a 50-minute
    video kept its job (and the GPU slot) for an hour."""
    try:
        n = int(os.environ.get("WHISPER_CPU_THREADS", "") or 0)
    except ValueError:
        n = 0
    return n if n > 0 else max(1, min(8, os.cpu_count() or 1))


def _get_whisper_model():
    """Process-wide WhisperModel singleton, rebuilt if the env config changes.

    Keeping the model resident avoids a full reload per transcription (which
    on GPU would also mean re-allocating a couple of GB of VRAM per job).
    """
    global _whisper_model, _whisper_key
    with _whisper_lock:
        cfg = get_whisper_config()
        if _whisper_force_cpu:
            cfg["device"] = "cpu"
            cfg["compute_type"] = "int8"
            cfg["model_size"] = _whisper_cpu_fallback_model()
        key = (cfg["model_size"], cfg["device"], cfg["compute_type"])
        if _whisper_model is None or _whisper_key != key:
            from faster_whisper import WhisperModel
            kwargs = {"cpu_threads": whisper_cpu_threads()} if key[1] == "cpu" else {}
            _whisper_model = WhisperModel(key[0], device=key[1], compute_type=key[2],
                                          **kwargs)
            _whisper_key = key
        return _whisper_model, cfg["device"]


def _whisper_device():
    return "cpu" if _whisper_force_cpu else get_whisper_config()["device"]


def _whisper_cpu_fallback_model():
    return os.environ.get("WHISPER_CPU_FALLBACK_MODEL", "").strip() or DEFAULT_WHISPER_MODEL


def _whisper_batch_size():
    try:
        return max(1, min(8, int(os.environ.get("WHISPER_BATCH_SIZE", "1"))))
    except ValueError:
        return 1


def _transcribe_backend():
    return os.environ.get("TRANSCRIBE_BACKEND", "whisper").strip().lower()


def uses_gpu_for_transcription():
    """Whether the configured ASR backend or its fallback can use the card."""
    return _whisper_device() != "cpu" or (
        _transcribe_backend() == "parakeet" and onnxruntime_device() == "cuda"
    )


def _decode_whisper_audio(media_path, start=None, duration=None):
    """Decode the first audio stream to the waveform faster-whisper expects.

    ``start``/``duration`` (seconds) decode only that stretch, which is how a
    long recording is transcribed in bounded pieces (_transcribe_whisper_chunked)."""
    import numpy as np

    window = []
    if start:
        window += ["-ss", f"{float(start):.3f}"]
    tail = ["-t", f"{float(duration):.3f}"] if duration else []
    result = subprocess.run(
        ["ffmpeg", "-nostdin", "-loglevel", "error", *window, "-i", media_path, *tail,
         "-map", "0:a:0", "-ac", "1", "-ar", "16000", "-f", "s16le", "-"],
        capture_output=True, timeout=1800,
    )
    if result.returncode:
        detail = result.stderr.decode(errors="replace").strip()[-500:]
        raise RuntimeError(f"Could not decode audio with FFmpeg: {detail}")
    audio = np.frombuffer(result.stdout, dtype="<i2").astype(np.float32)
    audio /= 32768.0
    return audio


def _run_whisper_once(media_path, *, batch_size=1, progress=None, offset=0.0, **params):
    """``progress``/``offset``: a shared meter for a chunked transcription,
    advanced on the whole recording's timeline (one 25 % ladder, not one per
    piece). Without them the meter covers this call alone."""
    on_cpu = _whisper_device() == "cpu"
    if _gpu_only and on_cpu:
        raise GpuUnavailable("whisper is configured for the CPU")
    gate = _NULL_GATE if on_cpu else _ASR_GATE
    # The model is fetched inside the gate: release_models() drains the gate
    # before unloading, so a transcription can never start on a model that is
    # being dropped underneath it.
    with gate:
        model, _device = _get_whisper_model()
        decoder = model
        if batch_size > 1:
            from faster_whisper import BatchedInferencePipeline
            decoder = BatchedInferencePipeline(model)
            params["batch_size"] = batch_size
            # Preserve segment cuts and avoid packing many separated utterances
            # into a single decode window.
            params.setdefault("without_timestamps", False)
            params.setdefault("chunk_length", 12)
        try:
            segments, info = decoder.transcribe(media_path, **params)
        except TypeError as e:
            # Faster-Whisper 1.2.1 uses an av.open argument removed in PyAV 19.
            if "unexpected keyword argument 'metadata_errors'" not in str(e):
                raise
            print("⚠️ [ASR] PyAV audio decode is incompatible, retrying with FFmpeg", flush=True)
            segments, info = decoder.transcribe(_decode_whisper_audio(media_path), **params)
        shared = progress is not None
        if not shared:
            progress = _TranscribeProgress(getattr(info, "duration", 0))
        materialized = []
        for segment in segments:
            materialized.append(segment)
            progress.update(offset + segment.end)
        # VAD trims trailing silence, so the last segment can end short of the
        # media duration — force the 100% line (the chunked caller does it
        # once, after its last piece).
        if not shared:
            progress.update(progress.total)
        return materialized, info


def run_whisper_transcription(media_path, *, batch_size=1, **params):
    """Transcribe and FULLY materialize the segments inside the GPU gate.

    faster-whisper returns a lazy generator — decoding happens while
    iterating, so the gate must wrap list(segments), not just transcribe().
    Returns (segments_list, info).

    A failed batch retries sequentially. GPU initialization and decode
    failures retry on CPU and pin CPU for the rest of the process.
    """
    global _whisper_model, _whisper_force_cpu
    if batch_size > 1:
        try:
            segments, info = _run_whisper_once(media_path, batch_size=batch_size, **params)
            voiced_seconds = getattr(info, "duration_after_vad", 0)
            covered_seconds = sum(max(0, seg.end - seg.start) for seg in segments)
            if not voiced_seconds or (
                segments and (voiced_seconds < 12 or covered_seconds >= 0.6 * voiced_seconds)
            ):
                return segments, info
            print("⚠️ [ASR] batched Whisper covered too little voiced audio, "
                  "retrying sequentially", flush=True)
        except (RuntimeError, ValueError) as e:
            print(f"⚠️ [ASR] batched Whisper failed ({e}), retrying sequentially", flush=True)

    try:
        return _run_whisper_once(media_path, **params)
    except RuntimeError as e:
        error = str(e).lower()
        if _whisper_force_cpu or not any(
            name in error for name in ("cuda", "cublas", "cudnn", "out of memory")
        ):
            raise
        if _gpu_only:
            with _whisper_lock:
                _whisper_model = None
            raise GpuUnavailable(f"whisper GPU failed ({e})") from e
        if _card_full(e):
            with _whisper_lock:
                failed, _whisper_model = _whisper_model, None
            try:
                # Unloaded, not only dropped: its weights would stay on the
                # card through the wait (see release_models).
                failed.model.unload_model()
            except Exception:
                pass
            del failed
            if wait_for_card(WHISPER_CARD_MB):
                print("🎙️ [ASR] room on the card again — whisper tries it once more",
                      flush=True)
                try:
                    return _run_whisper_once(media_path, **params)
                except RuntimeError as again:
                    if "cuda" not in str(again).lower():
                        raise
                    e = again
        with _whisper_lock:
            if not _whisper_force_cpu:
                _whisper_force_cpu = True
                _whisper_model = None  # drop the GPU model to release its VRAM
        print(f"⚠️ [ASR] Whisper GPU failed ({e}), retrying with "
              f"{_whisper_cpu_fallback_model()} INT8 on CPU", flush=True)
    return _run_whisper_once(media_path, **params)


def whisper_chunk_seconds():
    """Longest stretch Whisper transcribes in one call (WHISPER_CHUNK_SECONDS,
    0 = never split). faster-whisper builds the log-mel spectrogram of the
    whole voiced audio at once (STFT, magnitudes and mel arrays together), on
    the order of half a megabyte per second of speech: an 11-hour stream asks
    for ~19 GB and the kernel's OOM killer ends the job with SIGKILL. Pieces of
    30 minutes keep that near 1 GB whatever the source length."""
    try:
        value = float(os.environ.get("WHISPER_CHUNK_SECONDS", "1800"))
    except ValueError:
        return 1800.0
    return 0.0 if value <= 0 else max(300.0, value)


# Each piece boundary moves to the quietest 100 ms within this many seconds of
# its nominal place, so a cut lands between words instead of inside one.
_CHUNK_SEARCH_SECONDS = 20.0


def _quiet_point(media_path, target, lo, hi):
    """The quietest 100 ms frame of the audio within ±_CHUNK_SEARCH_SECONDS of
    ``target`` (and inside [lo, hi]); ``target`` itself when it can't be read."""
    import numpy as np

    start = max(lo, target - _CHUNK_SEARCH_SECONDS)
    end = min(hi, target + _CHUNK_SEARCH_SECONDS)
    if end - start < 1.0:
        return target
    try:
        audio = _decode_whisper_audio(media_path, start=start, duration=end - start)
    except (OSError, RuntimeError, subprocess.SubprocessError):
        return target
    frame = 1600  # 100 ms at 16 kHz
    count = len(audio) // frame
    if count < 3:
        return target
    energy = np.square(audio[:count * frame].reshape(count, frame)).mean(axis=1)
    # A 300 ms average: one quiet frame inside a word is not a pause.
    smooth = np.convolve(energy, np.ones(3) / 3, mode="same")
    quietest = int(np.argmin(smooth[1:-1])) + 1
    return start + (quietest + 0.5) * frame / 16000


def chunk_bounds(media_path, total, piece):
    """[(start, end), ...] covering 0..total in pieces of about ``piece`` s,
    each inner boundary moved to a nearby pause (_quiet_point)."""
    bounds, start = [], 0.0
    while total - start > piece * 1.25:
        cut = _quiet_point(media_path, start + piece, start + piece * 0.5, total - piece * 0.25)
        bounds.append((start, cut))
        start = cut
    bounds.append((start, total))
    return bounds


def _segment_dicts(segments, offset=0.0):
    out = []
    for segment in segments:
        words = [
            {"word": w.word, "start": float(w.start) + offset, "end": float(w.end) + offset}
            for w in (segment.words or [])
        ]
        out.append({
            "start": float(segment.start) + offset,
            "end": float(segment.end) + offset,
            "text": segment.text,
            "words": merge_continuation_words(words),
        })
    return out


def _transcript_from(out_segments, language):
    return {
        "text": " ".join(s["text"].strip() for s in out_segments if s["text"].strip()),
        "language": language,
        "segments": out_segments,
    }


def _transcribe_whisper_chunked(media_path, total, piece):
    """Whisper over a long recording, one bounded piece at a time.

    Every piece is decoded by FFmpeg on its own (so neither the waveform nor
    the spectrogram of the whole recording is ever in memory), transcribed,
    and its times shifted back onto the recording's timeline. The language the
    first piece detects is kept for the rest, so one recording is never
    transcribed half in another language."""
    bounds = chunk_bounds(media_path, total, piece)
    print(f"🎙️ [ASR] long recording ({total / 3600:.1f} h): transcribing in "
          f"{len(bounds)} pieces", flush=True)
    progress = _TranscribeProgress(total)
    params = dict(WHISPER_TRANSCRIBE_PARAMS)
    language = params.get("language")
    out_segments = []
    for lo, hi in bounds:
        audio = _decode_whisper_audio(media_path, start=lo, duration=hi - lo)
        if language:
            params["language"] = language
        segments, info = run_whisper_transcription(
            audio, batch_size=_whisper_batch_size(), progress=progress, offset=lo, **params)
        del audio
        language = language or getattr(info, "language", None)
        # A segment's padding may reach past the piece; it never starts there.
        out_segments.extend(s for s in _segment_dicts(segments, lo) if s["start"] < hi)
    progress.update(progress.total)
    return _transcript_from(out_segments, language)


def merge_ranges(ranges, pad=0.0, total=None):
    """Sorted, padded, overlap-merged [(start, end)] inside [0, total]."""
    out = []
    for lo, hi in sorted((float(a) - pad, float(b) + pad) for a, b in ranges):
        lo = max(0.0, lo)
        if total:
            hi = min(float(total), hi)
        if hi <= lo:
            continue
        if out and lo <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], hi))
        else:
            out.append((lo, hi))
    return out


def transcribe_ranges(media_path, ranges, *, pad=8.0, total=None):
    """Whisper over chosen stretches of a recording only (stage 2 of the hype
    engine): the candidate windows plus ``pad`` seconds either side, so a
    trim can still find its words. Times are on the recording's timeline.

    The returned transcript says what it covers (``coverage``); everything
    outside it is unknown, not silent. Same shared progress ladder and
    first-piece language as a chunked transcription."""
    bounds = merge_ranges(ranges, pad, total)
    covered = sum(hi - lo for lo, hi in bounds)
    print(f"🎙️ [ASR] focused transcription: {len(bounds)} stretch(es), "
          f"{covered / 60:.1f} min of audio", flush=True)
    progress = _TranscribeProgress(covered)
    params = dict(WHISPER_TRANSCRIBE_PARAMS)
    language = params.get("language")
    out_segments, done = [], 0.0
    for lo, hi in bounds:
        audio = _decode_whisper_audio(media_path, start=lo, duration=hi - lo)
        if language:
            params["language"] = language
        # The progress meter runs on the covered seconds, not the recording.
        segments, info = run_whisper_transcription(
            audio, batch_size=_whisper_batch_size(), progress=progress, offset=done, **params)
        del audio
        done += hi - lo
        language = language or getattr(info, "language", None)
        out_segments.extend(s for s in _segment_dicts(segments, lo) if s["start"] < hi)
    progress.update(progress.total)
    transcript = _transcript_from(out_segments, language)
    transcript["coverage"] = [[round(lo, 3), round(hi, 3)] for lo, hi in bounds]
    return transcript


def _transcribe_with_whisper(media_path):
    piece = whisper_chunk_seconds()
    if piece and isinstance(media_path, str):
        total = media_seconds(media_path)
        if total and total > piece * 1.25:
            return _transcribe_whisper_chunked(media_path, total, piece)

    segments, info = run_whisper_transcription(
        media_path, batch_size=_whisper_batch_size(), **WHISPER_TRANSCRIBE_PARAMS,
    )
    return _transcript_from(_segment_dicts(segments), info.language)


# --- parakeet ---------------------------------------------------------------

_parakeet_model = None
_parakeet_lock = threading.Lock()


# How many VAD segments (up to 20 s each) the encoder takes per batch. The
# library default is 8; 4 cuts the peak VRAM of a transcription from ~6.1 to
# ~4.6 GB for the same words. Benchmarked in prod on 22-sep-2026 over 10 real
# user videos (40 min of audio, EN/ES/PT/FR-AR, 4,850 words): batch 4 changed
# 1 word outside the mixed-language clip (which goes to whisper in prod anyway),
# 0.1 ms mean timestamp drift, +3 s per 20-min video. Batch 2 saved another
# 0.5 GB but dropped a whole sentence; int8 was 11x slower on this GPU with
# 11.7% of words different. Don't go below 4 without re-running that check.
PARAKEET_VAD_BATCH = int(os.environ.get("PARAKEET_VAD_BATCH", "4"))


def parakeet_providers():
    """onnxruntime providers for Parakeet. The arena grows only by what is
    asked and cuDNN gets no oversized workspace: same numerics (identical
    transcripts in the benchmark), less memory held."""
    cuda_opts = {
        "arena_extend_strategy": "kSameAsRequested",
        "cudnn_conv_use_max_workspace": "0",
        "cudnn_conv_algo_search": "HEURISTIC",
    }
    return [("CUDAExecutionProvider", cuda_opts), "CPUExecutionProvider"]


def parakeet_session_options():
    """onnxruntime SessionOptions for Parakeet: pool threads sleep, never spin.

    By default every ORT pool thread busy-waits for its next op. On the CUDA
    path the CPU has almost nothing to do, so that spinning was the work:
    ~160 CPU-s for a 9-minute video, the biggest CPU cost of a whole job
    (prod bench 25-sep-2026). Without it: ~35 CPU-s, and the transcript is
    byte-identical (text and every word timestamp, 3 real videos).

    The VAD gets its own options (vad_load_kwargs).
    """
    import onnxruntime as rt
    opts = rt.SessionOptions()
    opts.add_session_config_entry("session.intra_op.allow_spinning", "0")
    opts.add_session_config_entry("session.inter_op.allow_spinning", "0")
    return opts


def vad_load_kwargs():
    """Silero VAD on the CPU, one sleeping thread.

    load_vad's default puts Silero on CUDA, where it runs one 32 ms chunk at a
    time with a host/device copy around each: most of a transcription's wall
    time. On one CPU thread the whole transcription is 2-3x faster (9 min of
    audio: 28 s -> 9 s) and cheaper in CPU too. The VAD's numbers are not
    bit-identical across devices: over 3 real videos (3,939 words) one word
    changed ("claudio" -> "cloud", for "Claude") and a few word times moved by
    up to 48 ms (bench, 25-sep-2026).
    """
    import onnxruntime as rt
    opts = rt.SessionOptions()
    opts.intra_op_num_threads = 1
    opts.inter_op_num_threads = 1
    opts.add_session_config_entry("session.intra_op.allow_spinning", "0")
    opts.add_session_config_entry("session.inter_op.allow_spinning", "0")
    return {"sess_options": opts, "providers": ["CPUExecutionProvider"]}


# Where the last Parakeet load runs: "cuda", "cpu", or None before any load.
# main.py reports it in the job's JOB_CPU line.
PARAKEET_DEVICE = None


def onnxruntime_device():
    """"cuda" when the installed onnxruntime build has the CUDA provider.

    onnxruntime and onnxruntime-gpu are two wheels of one import package, and
    whichever was unpacked last is the build that runs. With the CPU wheel on
    top, asking for CUDAExecutionProvider only raises a UserWarning and
    Parakeet silently runs on the CPU: many times slower, on every core.
    """
    try:
        import onnxruntime as rt
        providers = rt.get_available_providers()
    except Exception:
        return "cpu"
    return "cuda" if "CUDAExecutionProvider" in providers else "cpu"


PARAKEET_HF_REPO = "istupakov/parakeet-tdt-0.6b-v3-onnx"


def _plain_parakeet_dir():
    """The model's files as ordinary files in one directory, copied once from
    the hub cache (see ``_load_parakeet``). Several job processes may get here
    together: each copies to its own temp dir and the first rename wins."""
    import shutil
    import tempfile
    from huggingface_hub import try_to_load_from_cache
    # Not snapshot_download: the cache only has the files the loader asked
    # for, and an incomplete snapshot is an error there.
    cached = try_to_load_from_cache(PARAKEET_HF_REPO, "encoder-model.onnx")
    if not isinstance(cached, str):
        raise RuntimeError("Parakeet is not in the hub cache")
    snap = os.path.dirname(cached)
    root = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(snap))),
                        "openshorts-plain")
    target = os.path.join(root, f"parakeet-{os.path.basename(snap)}")
    if os.path.isdir(target):
        return target
    os.makedirs(root, exist_ok=True)
    tmp = tempfile.mkdtemp(dir=root, prefix=".copy-")
    try:
        for name in os.listdir(snap):
            if "int8" in name:
                continue          # the quantised twin is never loaded
            shutil.copyfile(os.path.join(snap, name), os.path.join(tmp, name))
        try:
            os.rename(tmp, target)
        except OSError:
            if not os.path.isdir(target):
                raise
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return target


def _load_parakeet(onnx_asr):
    """Load Parakeet from the hub cache. A cache written by a recent
    huggingface_hub keeps every file as a link into its own blob directory,
    and onnxruntime refuses a model whose external weights (``.onnx.data``)
    resolve outside the model file's directory: the load then fails on a
    fresh cache while an old one still works. Same files from a plain
    directory load fine, so that is the fallback."""
    kwargs = dict(providers=parakeet_providers(), sess_options=parakeet_session_options())
    try:
        return onnx_asr.load_model(PARAKEET_MODEL_ID, **kwargs)
    except Exception as e:
        if "External data path" not in str(e):
            raise
    return onnx_asr.load_model(PARAKEET_MODEL_ID, _plain_parakeet_dir(), **kwargs)


def _get_parakeet_model():
    global _parakeet_model, PARAKEET_DEVICE
    with _parakeet_lock:
        if _parakeet_model is None:
            import onnx_asr
            model = _load_parakeet(onnx_asr)
            vad = onnx_asr.load_vad("silero", **vad_load_kwargs())
            _parakeet_model = model.with_vad(
                vad, batch_size=PARAKEET_VAD_BATCH).with_timestamps()
            PARAKEET_DEVICE = onnxruntime_device()
            if PARAKEET_DEVICE != "cuda":
                print("⚠️ [ASR] parakeet runs on the CPU: this onnxruntime build "
                      "has no CUDAExecutionProvider", flush=True)
    return _parakeet_model


def _drop_parakeet():
    """Drop the Parakeet singleton (its onnxruntime CUDA arena) before whisper
    loads in the same process. Measured in prod on 18-nov-2026: a job whose
    video Parakeet rejected still held ~7 GB of the card when whisper loaded,
    whisper hit CUDA OOM and the whole transcription went to the CPU."""
    global _parakeet_model
    for _ in range(_ASR_SLOTS):
        _ASR_GATE.acquire()
    try:
        with _parakeet_lock:
            parakeet, _parakeet_model = _parakeet_model, None
    finally:
        for _ in range(_ASR_SLOTS):
            _ASR_GATE.release()
    if parakeet is None:
        return
    del parakeet
    import gc
    gc.collect()


def release_models():
    """Drop the resident ASR models and hand their VRAM back to the GPU.

    main.py runs one job per process, so its singletons die with the job.
    The API process is different: ``/api/subtitle`` on a dubbed clip and the
    thumbnail studio transcribe in-process, and after the first such request
    the models sit in the long-lived uvicorn process for good. Measured in
    prod on 17-sep-2026: the API held 7.7 GB of a 20 GB GPU while idle
    (ctranslate2 whisper + onnxruntime CUDA parakeet + torch), and with eight
    jobs running alongside it NVENC could not open a session ("Generic error
    in an external library", exit 187, 0 bytes) and TransNetV2 hit CUDA OOM:
    5 of 12 jobs failed. The API calls this after each in-process
    transcription; a job process never needs to.

    Drains every gate slot first, so no transcription is mid-decode on the
    model being dropped, and both loaders fetch their model inside the gate.
    """
    global _whisper_model, _whisper_key, _parakeet_model
    for _ in range(_ASR_SLOTS):
        _ASR_GATE.acquire()
    try:
        with _whisper_lock:
            whisper, whisper_key = _whisper_model, _whisper_key
            _whisper_model, _whisper_key = None, None
        with _parakeet_lock:
            parakeet, _parakeet_model = _parakeet_model, None
    finally:
        for _ in range(_ASR_SLOTS):
            _ASR_GATE.release()
    # A CPU decoder can still own this instance. Its reference keeps it
    # alive until it finishes reading the generator.
    if whisper is not None and (whisper_key is None or whisper_key[1] != "cpu"):
        try:
            # ctranslate2 frees the weights on unload, not on garbage
            # collection: the Python wrapper can outlive the last reference.
            whisper.model.unload_model()
        except Exception as e:
            print(f"⚠️ [ASR] whisper unload failed ({type(e).__name__}: {e})")
    del whisper, parakeet
    # TransNetV2 is the other torch tenant an in-process pipeline leaves
    # behind (scene_detection keeps it as a module singleton too).
    tn2 = sys.modules.get("scene_detection")
    if tn2 is not None:
        tn2._tn2_model = None
    import gc
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    if "onnxruntime" in sys.modules or "faster_whisper" in sys.modules:
        print("🧹 [ASR] resident models released")


def _extract_wav(media_path):
    """Parakeet wants 16kHz mono PCM wav; ffmpeg-extract to a temp file."""
    fd, wav_path = tempfile.mkstemp(suffix=".wav", prefix="asr_")
    os.close(fd)
    cmd = [
        "ffmpeg", "-y", "-nostdin", "-loglevel", "error", "-i", media_path,
        "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", wav_path,
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL,
                       stderr=subprocess.PIPE, timeout=1800, text=True, errors="replace")
    except (OSError, subprocess.SubprocessError) as e:
        try:
            os.remove(wav_path)
        except OSError:
            pass
        if isinstance(e, subprocess.CalledProcessError) and e.stderr:
            raise RuntimeError(f"Could not decode audio: {e.stderr.strip()[-500:]}") from e
        raise
    return wav_path


def _words_from_tokens(tokens, timestamps, seg_start, seg_end):
    """Group parakeet BPE tokens into words with absolute timestamps.

    Verified on the prod model: tokens already carry the leading-space
    word-start convention (" T", "odo", " el", ...) and timestamps are token
    START times in seconds relative to the VAD segment. A token without a
    leading space (subword continuations, punctuation like ",") belongs to
    the previous word — same semantics merge_continuation_words expects.
    Word end is inferred: next word's start, capped near the word's last
    token so a long inter-word silence doesn't stretch the highlight.
    """
    words = []
    last_token_ts = []
    for token, ts in zip(tokens, timestamps):
        if not token:
            continue
        abs_ts = float(ts) + seg_start
        if token.startswith(" ") or not words:
            words.append({
                "word": token if token.startswith(" ") else " " + token,
                "start": abs_ts,
            })
            last_token_ts.append(abs_ts)
        else:
            words[-1]["word"] += token
            last_token_ts[-1] = abs_ts

    for i, word in enumerate(words):
        next_start = words[i + 1]["start"] if i + 1 < len(words) else seg_end
        cap = last_token_ts[i] + 0.6
        word["end"] = float(max(word["start"] + 0.05, min(next_start, cap)))

    return words


# --- the spoken language, before Parakeet -----------------------------------
#
# Parakeet writes a language it does not cover as noise, the result is
# rejected and whisper transcribes the source again: such a video pays for two
# transcriptions, and the wasted one is the one that takes the most room on
# the card. With ``ASR_LANGUAGE_PROBE=1`` a small whisper listens to a few
# windows of the audio on the CPU first, and a source it is sure is in none of
# Parakeet's languages goes straight to whisper. It only ever skips Parakeet:
# an unsure answer, or any failure, leaves the order as it was.
#
# It also catches what the rejection cannot: speech in a language Parakeet
# does not cover sometimes comes out as words that read as English, and that
# transcript was accepted, captions and all.
LANGUAGE_PROBE_MODEL = os.environ.get("ASR_LANGUAGE_PROBE_MODEL", "tiny").strip() or "tiny"
LANGUAGE_PROBE_SECONDS = 30.0
LANGUAGE_PROBE_AT = (0.15, 0.5, 0.85)       # where in the audio each window starts


def language_probe_enabled():
    return os.environ.get("ASR_LANGUAGE_PROBE", "0").strip() == "1"


def language_probe_min():
    """How sure every window has to be (``ASR_LANGUAGE_PROBE_MIN``, 0.7).

    Checked against transcripts the pipeline had kept: at 0.7 the probe named
    most of the sources whisper ended up transcribing, and every source it
    named that Parakeet had passed as English was noise that happened to
    read as English. Higher floors only let more of both through."""
    try:
        return float(os.environ.get("ASR_LANGUAGE_PROBE_MIN", "") or 0.7)
    except ValueError:
        return 0.7


class ParakeetSkipped(Exception):
    """The source is spoken in a language Parakeet does not cover."""


def _probe_windows(wav_path):
    """A few 30 s stretches of a 16 kHz mono s16le wav, as float arrays."""
    import wave
    import numpy as np
    out = []
    with wave.open(wav_path, "rb") as w:
        rate, total = w.getframerate(), w.getnframes()
        size = int(LANGUAGE_PROBE_SECONDS * rate)
        if total <= size:
            starts = [0]
        else:
            starts = sorted({int((total - size) * at) for at in LANGUAGE_PROBE_AT})
        for start in starts:
            w.setpos(start)
            frames = w.readframes(size)
            if frames:
                out.append(np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0)
    return out


def _language_probe_model():
    from faster_whisper import WhisperModel
    return WhisperModel(LANGUAGE_PROBE_MODEL, device="cpu", compute_type="int8",
                        cpu_threads=2)


def _window_languages(wav_path):
    """[(language, probability)] for each window that had speech in it."""
    model = _language_probe_model()
    try:
        heard = []
        for audio in _probe_windows(wav_path):
            try:
                language, probability, _all = model.detect_language(
                    audio=audio, vad_filter=True)
            except Exception:
                continue        # a window of silence or music: no answer
            if language:
                heard.append((language, float(probability)))
        return heard
    finally:
        del model


def language_outside_parakeet(wav_path):
    """(language, probability) when the audio is surely in a language Parakeet
    does not cover, else None.

    Sure means every window that had speech named a language outside the set
    with at least ``language_probe_min()``, and more than one window did when
    the audio is long enough for several. The windows need not agree on which
    one: close languages are told apart badly, and neither is Parakeet's."""
    if not language_probe_enabled():
        return None
    try:
        expected = len(_probe_windows(wav_path))
        heard = _window_languages(wav_path)
    except Exception as e:
        print(f"⚠️ [ASR] language probe skipped ({type(e).__name__}: {e})", flush=True)
        return None
    if not heard or len(heard) < min(2, expected):
        return None
    floor = language_probe_min()
    if any(lang in PARAKEET_LANGS or prob < floor for lang, prob in heard):
        return None
    names = [lang for lang, _ in heard]
    return max(set(names), key=names.count), min(prob for _, prob in heard)


def _transcribe_with_parakeet(media_path):
    wav_path = _extract_wav(media_path)
    try:
        # 16kHz mono s16le wav -> 32000 bytes per second of audio.
        try:
            duration = os.path.getsize(wav_path) / 32000.0
        except OSError:
            duration = 0.0
        outside = language_outside_parakeet(wav_path)
        if outside:
            raise ParakeetSkipped(
                f"spoken language '{outside[0]}' ({outside[1]:.2f}) is not one of parakeet's")
        with _ASR_GATE:
            model = _get_parakeet_model()  # inside the gate: see release_models
            progress = _TranscribeProgress(duration)
            results = []
            for seg in model.recognize(wav_path):
                results.append(seg)
                progress.update(float(seg.end))
            progress.update(progress.total)
        transcript = _parakeet_transcript(results)
        if _parakeet_fallback_reason(transcript) is not None:
            # transcribe_media throws this away for whisper: no repair.
            return transcript
        # Outside the parakeet gate on purpose: the repair takes it again for
        # whisper, and the semaphore is not re-entrant.
        _repair_script_drift(transcript, wav_path)
        return transcript
    finally:
        try:
            os.remove(wav_path)
        except OSError:
            pass


def _parakeet_transcript(results):
    out_segments = []
    text_parts = []
    for seg in results:
        seg_start = float(seg.start)
        seg_end = float(seg.end)
        seg_text = str(seg.text or "").strip()
        if not seg_text:
            continue
        out_segments.append({
            "start": seg_start,
            "end": seg_end,
            "text": seg_text,
            "words": _words_from_tokens(
                list(seg.tokens or []), list(seg.timestamps or []),
                seg_start, seg_end,
            ),
        })
        text_parts.append(seg_text)

    text = " ".join(text_parts)
    return {
        "text": text,
        "language": _detect_language(text),
        "segments": out_segments,
    }


# --- script drift repair ----------------------------------------------------
#
# Parakeet v3 has no language parameter: it picks one of its 25 languages per
# VAD segment, from the audio alone. On short segments (a few words, an accent,
# crosstalk) it sometimes writes English phonetically in Ukrainian or Russian:
# "Тудей Марк із гана такі сторі" for "Today Mark is gonna tell his story". The
# file-level language still reads "en", so the whisper fallback never fires,
# and the Cyrillic words end up burned into the captions and fed to the moment
# picker. It is common on English sources, and the drifted segments are
# almost always the short ones.
#
# The repair re-transcribes only those segments with whisper, forced to the
# language of the whole file; on a real English podcast every drifted segment
# came back as the right English sentence.

_SCRIPT_RX = {
    "cyrl": re.compile("[\u0400-\u04FF]"),
    "grek": re.compile("[\u0370-\u03FF]"),
}
_LANG_SCRIPT = {"ru": "cyrl", "uk": "cyrl", "bg": "cyrl", "el": "grek"}
# A Cyrillic-script transcript can legitimately quote Latin brand names, so
# only scripts that cannot belong to the file's language count as drift.
_FOREIGN_SCRIPTS = {"latn": ("cyrl", "grek"), "grek": ("cyrl",), "cyrl": ()}
_DRIFT_PAD = 0.15  # seconds of audio around a segment handed to whisper


def _segment_drifted(text, language):
    expected = _LANG_SCRIPT.get(language, "latn")
    return any(_SCRIPT_RX[s].search(text or "") for s in _FOREIGN_SCRIPTS[expected])


def _read_wav_span_16k(wav_path, start, end):
    import wave
    import numpy as np
    with wave.open(wav_path, "rb") as w:
        first = min(w.getnframes(), int(start * 16000))
        last = min(w.getnframes(), int(end * 16000))
        w.setpos(first)
        frames = w.readframes(max(0, last - first))
    return np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0


def _whisper_span(audio, language):
    """Whisper words for one short audio span (seconds relative to the span).

    Calls the model directly so the Parakeet model can keep its GPU slot
    and the user-facing progress lines are not printed again for each span.
    """
    params = dict(WHISPER_TRANSCRIBE_PARAMS, language=language, vad_filter=False)
    if _gpu_only and _whisper_device() == "cpu":
        raise GpuUnavailable("whisper is configured for the CPU")
    gate = _ASR_GATE if _whisper_device() != "cpu" else _NULL_GATE
    with gate:
        model, _device = _get_whisper_model()
        segments, _info = model.transcribe(audio, **params)
        segments = list(segments)
    words = [
        {"word": w.word, "start": float(w.start), "end": float(w.end)}
        for seg in segments for w in (seg.words or [])
    ]
    text = " ".join(seg.text.strip() for seg in segments if seg.text.strip())
    return text, merge_continuation_words(words)


def _repair_script_drift(transcript, wav_path):
    """Re-transcribe, in place, the segments written in the wrong script.

    Returns how many segments were replaced. Any failure leaves the parakeet
    text as it was: a wrong-script caption is a defect, a failed job is worse.
    """
    language = transcript.get("language")
    segments = transcript.get("segments") or []
    drifted = [i for i, s in enumerate(segments) if _segment_drifted(s.get("text"), language)]
    if not drifted:
        return 0
    fixed = 0
    for i in drifted:
        seg = segments[i]
        lo = max(0.0, seg["start"] - _DRIFT_PAD)
        hi = seg["end"] + _DRIFT_PAD
        try:
            audio = _read_wav_span_16k(wav_path, lo, hi)
            if not len(audio):
                continue
            text, words = _whisper_span(audio, language)
        except Exception as e:
            print(f"⚠️ [ASR] script drift repair failed ({type(e).__name__}: {e})")
            break
        if not words or not text.strip() or _segment_drifted(text, language):
            continue
        # Keep the transcript chronological: the padded span may reach into
        # the neighbouring segments.
        floor = segments[i - 1]["end"] if i > 0 else 0.0
        ceil = segments[i + 1]["start"] if i + 1 < len(segments) else hi
        for w in words:
            w["start"] = float(min(max(w["start"] + lo, floor), ceil))
            w["end"] = float(min(max(w["end"] + lo, w["start"]), ceil))
        seg["text"] = text
        seg["words"] = words
        fixed += 1
    if fixed:
        transcript["text"] = " ".join(s["text"] for s in segments if s.get("text"))
        print(f"🎙️ [ASR] re-transcribed {fixed}/{len(drifted)} segments written in "
              f"the wrong script (language={language})")
    return fixed


def _detect_language(text):
    """Parakeet doesn't report a language; classify the transcribed text.

    py3langid is pure-Python and returns ISO 639-1 codes compatible with the
    whisper codes the pipeline expects (thumbnail titles, Gemini prompts).
    """
    sample = (text or "").strip()
    if len(sample) < 20:
        return "en"
    try:
        import py3langid
        lang, _score = py3langid.classify(sample[:4000])
        return lang
    except Exception:
        return "en"


def _parakeet_fallback_reason(transcript, duration_hint=None):
    """Return why the parakeet result is untrustworthy, or None if it's fine."""
    segments = transcript.get("segments") or []
    total_words = sum(len(s.get("words") or []) for s in segments)
    if total_words == 0:
        return "no words recognized"
    language = transcript.get("language")
    if language not in PARAKEET_LANGS:
        return f"language '{language}' outside parakeet's supported set"
    duration = duration_hint or (segments[-1]["end"] if segments else 0)
    # Real speech averages >100 wpm; under ~12 wpm on a long video means the
    # audio was mostly not recognized (e.g. unsupported language or music).
    if duration > 60 and total_words < duration * 0.2:
        return f"only {total_words} words in {duration:.0f}s of audio"
    return None


# --- public entry point -----------------------------------------------------

class NoAudioError(Exception):
    """The media has no audio track — nothing to transcribe."""


def _has_audio_stream(media_path) -> bool:
    """True if the file has at least one audio stream (ffprobe)."""
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "a",
             "-show_entries", "stream=index", "-of", "csv=p=0", media_path],
            capture_output=True, text=True, check=True, timeout=60,
        )
        return bool(out.stdout.strip())
    except Exception:
        return True  # probe failed — don't block, let the backend try


def media_seconds(media_path):
    """Container duration in seconds (ffprobe), or None when unknown."""
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "csv=p=0", media_path],
            capture_output=True, text=True, timeout=60,
        )
        value = float(out.stdout.strip().splitlines()[0])
        return value if value > 0 else None
    except Exception:
        return None


def _parakeet_gets_another_try(error):
    """True when Parakeet failed for lack of room on the card and the room
    came back: its arena is dropped first, so the wait sees what is left."""
    if _gpu_only or not _card_full(error) or ASR_CARD_WAIT <= 0:
        return False
    try:
        _drop_parakeet()
    except Exception:
        return False
    return wait_for_card(PARAKEET_CARD_MB)


def transcribe_media(media_path):
    """Transcribe with the configured backend, falling back to whisper."""
    # Silent videos (AI-generated clips, muted screen recordings) have no audio
    # stream; every ASR backend then crashes deep inside libav with an opaque
    # "tuple index out of range". Detect it up front and fail with a clear,
    # actionable reason instead.
    if not _has_audio_stream(media_path):
        raise NoAudioError(
            "This video has no audio track. OpenShorts finds viral moments from "
            "speech, so it needs a video with audio.")

    backend = os.environ.get("TRANSCRIBE_BACKEND", "whisper").strip().lower()

    if backend == "parakeet":
        for attempt in (1, 2):
            try:
                transcript = _transcribe_with_parakeet(media_path)
                reason = _parakeet_fallback_reason(transcript)
                if reason is None:
                    print(f"🎙️ [ASR] parakeet ok: lang={transcript['language']} "
                          f"segments={len(transcript['segments'])}")
                    return transcript
                print(f"⚠️ [ASR] parakeet result rejected ({reason}) — "
                      f"falling back to whisper")
            except ParakeetSkipped as e:
                print(f"🎙️ [ASR] {e} — straight to whisper", flush=True)
            except Exception as e:
                if attempt == 1 and _parakeet_gets_another_try(e):
                    print("🎙️ [ASR] room on the card again — parakeet tries it "
                          "once more", flush=True)
                    continue
                print(f"⚠️ [ASR] parakeet failed ({type(e).__name__}: {e}) — "
                      f"falling back to whisper")
            break
        # Whisper needs the card's memory that Parakeet's arena still holds.
        try:
            _drop_parakeet()
        except Exception as e:
            print(f"⚠️ [ASR] could not release parakeet ({type(e).__name__}: {e})")

    return _transcribe_with_whisper(media_path)
