"""Sound effects for the viral edit, synthesised here: no samples to license.

Three sounds cover what stream edits use them for:
- ``boom``: a sub-bass drop with a short transient, on the reaction peak;
- ``whoosh``: a swept noise burst, on the cold open's jump back to the setup;
- ``hit``: a short punch, on the very first frame of a cold open.

Each is generated once with numpy (deterministic: a fixed seed) into
SFX_CACHE_DIR (default ~/.cache/openshorts/sfx) as 48 kHz stereo WAV, peaking
at -3 dBFS; the mix level is set where it is placed.
"""
import math
import os
import uuid
import wave
from contextlib import suppress

VERSION = 1
RATE = 48000
SOUND_SECONDS = {"boom": 1.2, "whoosh": 0.55, "hit": 0.3}


def cache_dir():
    return os.environ.get("SFX_CACHE_DIR", "").strip() or os.path.join(
        os.path.expanduser("~"), ".cache", "openshorts", "sfx")


def _normalise(x, peak_db=-3.0):
    import numpy as np

    top = float(np.max(np.abs(x))) or 1.0
    return x * (10 ** (peak_db / 20.0) / top)


def boom(seconds=SOUND_SECONDS["boom"]):
    import numpy as np

    t = np.arange(int(RATE * seconds)) / RATE
    freq = 38.0 + 85.0 * np.exp(-t * 3.0)
    phase = 2 * np.pi * np.cumsum(freq) / RATE
    body = np.sin(phase) * np.exp(-t * 3.2) * np.minimum(1.0, t / 0.005)
    rng = np.random.default_rng(7)
    noise = rng.standard_normal(len(t))
    smooth = np.convolve(noise, np.ones(24) / 24, mode="same")       # a dull thump, not hiss
    transient = smooth * np.exp(-t * 40.0) * 0.8
    mix = np.tanh(1.8 * (body + transient)) / math.tanh(1.8)
    return _normalise(mix)


def whoosh(seconds=SOUND_SECONDS["whoosh"]):
    import numpy as np

    n = int(RATE * seconds)
    t = np.arange(n) / RATE
    rng = np.random.default_rng(11)
    noise = rng.standard_normal(n)
    # One-pole low-pass whose cutoff sweeps up and back down: the "air" moving.
    shape = np.sin(np.pi * t / seconds) ** 2
    cutoff = 400.0 + 5600.0 * shape
    alpha = 1.0 - np.exp(-2 * np.pi * cutoff / RATE)
    out = np.empty(n)
    state = 0.0
    for i in range(n):
        state += alpha[i] * (noise[i] - state)
        out[i] = state
    return _normalise(out * shape)


def hit(seconds=SOUND_SECONDS["hit"]):
    import numpy as np

    t = np.arange(int(RATE * seconds)) / RATE
    rng = np.random.default_rng(3)
    click = rng.standard_normal(len(t)) * np.exp(-t * 300.0)
    low = np.sin(2 * np.pi * 180.0 * t) * np.exp(-t * 18.0)
    tick = np.sin(2 * np.pi * 2400.0 * t) * np.exp(-t * 60.0) * 0.3
    return _normalise(np.tanh(1.5 * (0.6 * click + low + tick)))


SOUNDS = {"boom": boom, "whoosh": whoosh, "hit": hit}


def _write(path, samples):
    import numpy as np

    pcm = (np.clip(samples, -1, 1) * 32767).astype("<i2")
    stereo = np.repeat(pcm[:, None], 2, axis=1)
    tmp = f"{path}.{uuid.uuid4().hex}.partial"
    try:
        with wave.open(tmp, "wb") as f:
            f.setnchannels(2)
            f.setsampwidth(2)
            f.setframerate(RATE)
            f.writeframes(stereo.tobytes())
        os.replace(tmp, path)
    finally:
        with suppress(FileNotFoundError):
            os.remove(tmp)


def path(name):
    """The WAV for ``name``, generated on first use; None for an unknown name
    or an unwritable cache (the clip ships without that sound)."""
    if name not in SOUNDS:
        return None
    target = os.path.join(cache_dir(), f"{name}-v{VERSION}.wav")
    if os.path.isfile(target):
        return target
    try:
        os.makedirs(cache_dir(), exist_ok=True)
        _write(target, SOUNDS[name]())
    except OSError:
        return None
    return target


def placements(plan):
    """[(output second, sound, gain)] for an edit plan (viral_edit.plan)."""
    out = []
    peaks = list(plan.get("peak_out") or [])
    if plan.get("teaser"):
        out.append((0.0, "hit", 0.55))
        out.append((max(0.0, plan["teaser_length"] - 0.30), "whoosh", 0.7))
        peaks = peaks[1:]          # the teaser's own peak already opens on the hit
    for p in peaks[-1:]:
        out.append((max(0.0, p - 0.05), "boom", 0.8))
    return out
