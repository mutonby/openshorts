"""Stage 1 of highlight discovery: a hype curve from chat and the streamer.

A stream's best moments are the ones where chat goes wild and the streamer
does too, and neither shows up in a transcript. This module reads both over
the WHOLE recording in a few minutes, without speech recognition, and turns
them into ranked 30-60 s candidate windows for the slower stages (focused
transcription and the video judge, hype_judge.py).

Signals, all measured against the stream's OWN normal level:

- chat (``chat_curve``): distinct chatters per second, reaction messages
  (laughter, shock, hype emotes, "clip it") and clip requests. Commands,
  giveaways and one author repeating themselves cannot move it (the filters
  are chat_hype's). Raw messages stay on disk; only per-second counts leave.
- streamer voice (``voice_curve``): one streaming FFmpeg decode of the audio,
  0.5 s frames. Loudness in the speech band against a rolling baseline of the
  recording around it, energy at 1.5-4 kHz (yelling and screams), and the
  3-7 Hz envelope rhythm of laughter. A frame only counts as the streamer
  when the speech band dominates it, so an explosion in the game is not a
  reaction. These are leads, not speaker labels: the judge watches the clip.

Chat reacts seconds after the event. ``chat_lag`` estimates that delay per
stream by cross-correlating the two curves; without a clear answer it uses
HIGHLIGHT_CHAT_DELAY (7 s). Standard library + numpy + scipy only.
"""
import math
import os
import re
import subprocess
from collections import Counter, defaultdict

import chat_hype

SAMPLE_RATE = 16000
FRAME = 0.5                       # voice analysis step, seconds
_FRAME_SAMPLES = int(SAMPLE_RATE * FRAME)
_ENV_HOP = 320                    # 20 ms envelope for the laughter rhythm
_BLOCK_SECONDS = 60               # decode block; memory stays flat on any length

# Reactions chat types instead of words. Superset of chat_hype.REACTIONS
# with the common Twitch/YouTube emote names (matched case-insensitively,
# as whole tokens).
HYPE_TOKENS = {
    "laughter": {"kekw", "lul", "lulw", "omegalul", "icant", "lmao", "lmfao", "lol", "lool", "loool",
                 "haha", "hahaha", "rofl", "xd", "kek", "pepelaugh", "😂", "🤣", "💀"},
    "shock": {"omg", "wtf", "pog", "pogchamp", "poggers", "pogu", "monkas", "waytoodank", "nooo",
              "noway", "what", "bruh", "holy", "😱", "🤯", "😳"},
    "hype": {"w", "ww", "www", "letsgo", "lets", "ez", "clap", "gigachad", "goat", "insane",
             "clutch", "🔥", "🙌", "gg"},
}
_TOKEN = re.compile(r"[\w']+|[^\w\s]", re.UNICODE)
_CLIP_REQUEST = re.compile(r"^\s*!clip\b|\bclip (?:it|that|this)\b", re.I)
_QUESTION_SPAM = re.compile(r"^\s*\?{3,}\s*$")


def _setting(name, default, lo=None, hi=None):
    try:
        value = float(os.environ.get(name, "").strip() or default)
    except ValueError:
        value = default
    if not math.isfinite(value):
        value = default
    if lo is not None:
        value = max(lo, value)
    if hi is not None:
        value = min(hi, value)
    return value


def reaction_of(text):
    """The reaction category of one chat line, or None."""
    line = " ".join(str(text).casefold().split())
    if _QUESTION_SPAM.match(line):
        return "shock"
    tokens = set(_TOKEN.findall(line))
    for name, words in HYPE_TOKENS.items():
        if tokens & words:
            return name
    for name, pattern in chat_hype.REACTIONS.items():
        if pattern.search(line):
            return {"surprise": "shock", "cheering": "hype"}.get(name, name)
    return None


# --------------------------------------------------------------------------
# Chat
# --------------------------------------------------------------------------

def chat_curve(messages, duration, *, offset=0.0):
    """Per-second chat activity over [0, duration).

    Returns {"chatters", "reactions", "clip_requests", "categories", "messages"}:
    lists (one value per second) plus per-second category counters. Each
    author counts once per second; a non-reaction line many people type alike
    (a poll vote, a copy-paste) once per second; a command or giveaway not at
    all (chat_hype.NON_REACTION); a "!clip" request
    is counted before that filter, since asking for a clip is the signal."""
    n = max(1, int(math.ceil(duration)))
    authors = defaultdict(set)
    lines = defaultdict(set)
    reactions = [0] * n
    clip_requests = [0] * n
    raw = [0] * n
    categories = defaultdict(Counter)
    accepted = 0
    for m in messages:
        t = m.seconds + offset
        if not 0 <= t < duration:
            continue
        i = int(t)
        raw[i] += 1
        text = " ".join(m.text.casefold().split())
        if _CLIP_REQUEST.search(text):
            clip_requests[i] += 1
        if chat_hype.NON_REACTION.search(text):
            continue
        key = m.author or f"line:{text}"
        if key in authors[i]:
            continue
        kind = reaction_of(text)
        # Many people typing the same REACTION is hype and counts everyone;
        # the same non-reaction line ("1" for a poll, a copy-paste, a
        # giveaway word) counts once: on a real stream, 114 viewers voting
        # "1" otherwise ranked as a top moment.
        if not kind and text in lines[i]:
            continue
        authors[i].add(key)
        lines[i].add(text)
        accepted += 1
        if kind:
            reactions[i] += 1
            categories[i][kind] += 1
    return {"chatters": [len(authors[i]) for i in range(n)], "reactions": reactions,
            "clip_requests": clip_requests, "categories": categories, "messages": raw,
            "accepted": accepted}


def chat_intensity(curve, *, smooth=5, baseline=300):
    """Chat excitement per second, in robust units above the stream's own
    level: (activity - rolling median) / rolling spread, never negative.

    Activity is distinct chatters plus reactions plus clip requests (x3),
    summed over ``smooth`` seconds because a burst spreads over a few."""
    import numpy as np
    from scipy.ndimage import percentile_filter, uniform_filter1d

    chatters = np.asarray(curve["chatters"], dtype="float32")
    if not chatters.size or chatters.sum() == 0:
        return np.zeros_like(chatters)
    activity = (chatters + np.asarray(curve["reactions"], dtype="float32")
                + 3.0 * np.asarray(curve["clip_requests"], dtype="float32"))
    activity = uniform_filter1d(activity, size=smooth, mode="nearest") * smooth
    size = int(min(len(activity), baseline)) | 1
    base = percentile_filter(activity, 50, size=size, mode="nearest")
    high = percentile_filter(activity, 90, size=size, mode="nearest")
    spread = np.maximum(high - base, np.maximum(2.0, 0.25 * base))
    return np.clip((activity - base) / spread, 0, None)


# --------------------------------------------------------------------------
# Streamer voice
# --------------------------------------------------------------------------

def _band(freqs, lo, hi):
    return (freqs >= lo) & (freqs < hi)


def voice_features(media_path, *, timeout=None, start=None, duration=None):
    """0.5 s frames of the first audio stream: speech-band level (dB), the
    share of energy in the speech band, the 1.5-4 kHz share, and the 3-7 Hz
    rhythm of the speech-band envelope over the surrounding 2 s.

    One FFmpeg decode streamed in 60 s blocks, so memory does not grow with
    the recording; ``start``/``duration`` (file seconds) read only a stretch.
    Returns None when there is no readable audio."""
    import numpy as np
    from scipy.signal import butter, sosfilt, sosfilt_zi

    window = ["-ss", f"{float(start):.3f}"] if start else []
    length = ["-t", f"{float(duration):.3f}"] if duration else []
    command = ["ffmpeg", "-nostdin", "-v", "error", *window, "-i", str(media_path), *length,
               "-map", "0:a:0", "-ac", "1", "-ar", str(SAMPLE_RATE), "-f", "s16le", "-"]
    freqs = np.fft.rfftfreq(_FRAME_SAMPLES, 1 / SAMPLE_RATE)
    speech, high, total = _band(freqs, 300, 3400), _band(freqs, 1500, 4000), _band(freqs, 60, 7000)
    window = np.hanning(_FRAME_SAMPLES).astype("float32")
    sos = butter(4, [300, 3400], btype="bandpass", fs=SAMPLE_RATE, output="sos")
    state = sosfilt_zi(sos) * 0.0
    db, ratio, high_share, envelope = [], [], [], []
    block_bytes = _BLOCK_SECONDS * SAMPLE_RATE * 2
    carry = np.zeros(0, dtype="float32")
    with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL) as run:
        while True:
            raw = run.stdout.read(block_bytes)
            if not raw:
                break
            samples = np.frombuffer(raw[:len(raw) // 2 * 2], dtype="<i2").astype("float32") / 32768.0
            filtered, state = sosfilt(sos, samples, zi=state)
            hops = len(filtered) // _ENV_HOP
            if hops:
                env = filtered[:hops * _ENV_HOP].reshape(hops, _ENV_HOP)
                envelope.append(np.sqrt(np.mean(env * env, axis=1)))
            data = np.concatenate([carry, samples])
            frames = len(data) // _FRAME_SAMPLES
            carry = data[frames * _FRAME_SAMPLES:]
            if not frames:
                continue
            chunk = data[:frames * _FRAME_SAMPLES].reshape(frames, _FRAME_SAMPLES) * window
            power = np.abs(np.fft.rfft(chunk, axis=1)) ** 2
            e_total = power[:, total].sum(axis=1) + 1e-12
            e_speech = power[:, speech].sum(axis=1) + 1e-12
            db.append(10 * np.log10(e_speech / _FRAME_SAMPLES))
            ratio.append(e_speech / e_total)
            high_share.append(power[:, high].sum(axis=1) / e_speech)
        if run.wait(timeout=timeout) != 0 or not db:
            return None
    db, ratio, high_share = (np.concatenate(x).astype("float32") for x in (db, ratio, high_share))
    envelope = np.concatenate(envelope) if envelope else np.zeros(0, dtype="float32")
    return {"db": db, "speech_ratio": ratio, "high_share": high_share,
            "laugh_rhythm": _laugh_rhythm(envelope, len(db)),
            "duration": len(db) * FRAME}


def _laugh_rhythm(envelope, frames):
    """Share of the envelope's modulation at 3-7 Hz (the ha-ha-ha rate) over
    the 2 s around each 0.5 s frame. 0..1; speech sits lower, laughter higher."""
    import numpy as np

    rate = SAMPLE_RATE / _ENV_HOP              # 50 Hz
    win = int(2 * rate)                        # 100 envelope samples
    step = int(FRAME * rate)                   # 25
    out = np.zeros(frames, dtype="float32")
    if len(envelope) < win:
        return out
    starts = np.arange(0, len(envelope) - win + 1, step)
    segs = np.lib.stride_tricks.sliding_window_view(envelope, win)[starts]
    segs = segs - segs.mean(axis=1, keepdims=True)
    spec = np.abs(np.fft.rfft(segs * np.hanning(win), axis=1)) ** 2
    freqs = np.fft.rfftfreq(win, 1 / rate)
    band = spec[:, (freqs >= 3) & (freqs <= 7)].sum(axis=1)
    whole = spec[:, (freqs >= 0.5) & (freqs <= 15)].sum(axis=1) + 1e-12
    share = band / whole
    # Window i covers frames centred at its middle.
    centre = (starts + win // 2) // step
    keep = centre < frames
    out[centre[keep]] = share[keep]
    return out


def voice_excitement(features, *, baseline_seconds=120):
    """Streamer excitement per 0.5 s frame, in robust units above their own
    normal: loudness over the rolling speaking level, plus scream energy and
    laughter rhythm, gated on the speech band dominating the frame."""
    import numpy as np
    from scipy.ndimage import percentile_filter, uniform_filter1d

    db = features["db"]
    if not len(db):
        return np.zeros(0, dtype="float32")
    size = int(min(len(db), baseline_seconds / FRAME)) | 1
    p50 = percentile_filter(db, 50, size=size, mode="nearest")
    p70 = percentile_filter(db, 70, size=size, mode="nearest")
    p95 = percentile_filter(db, 95, size=size, mode="nearest")
    spread = np.maximum(p95 - p50, 4.0)
    loud = np.clip((db - p70) / spread, 0, 3)
    # The noise floor: a frame this far under the normal speaking level is silence.
    voiced = (db > p50 - 3) & (features["speech_ratio"] >= 0.45)
    hs = features["high_share"]
    hs_base = percentile_filter(hs, 50, size=size, mode="nearest")
    scream = np.clip((hs - hs_base) / np.maximum(hs_base, 0.05), 0, 2) * (loud > 0.5)
    # Ordinary speech already has some 3-7 Hz rhythm (a Part 8 measurement:
    # median 0.26, 90th percentile 0.51), so laughter is judged against this
    # recording's own voiced frames: only its top 3 % count.
    rhythm = features["laugh_rhythm"]
    laugh_threshold = laughter_threshold(features, voiced)
    laugh = np.clip((rhythm - laugh_threshold) / 0.12, 0, 1.5) * (db > p50)
    score = (loud + 0.6 * scream + 0.8 * laugh) * voiced
    # A shout is a second or two; keep its neighbourhood.
    return uniform_filter1d(score.astype("float32"), size=3, mode="nearest")


def laughter_threshold(features, voiced=None):
    """The laugh-rhythm value only this recording's top 3 % of voiced frames
    reach (never under 0.45), cached on ``features``."""
    import numpy as np

    if "laugh_threshold" not in features:
        rhythm = features["laugh_rhythm"]
        pool = rhythm[voiced] if voiced is not None and np.any(voiced) else rhythm
        features["laugh_threshold"] = max(0.45, float(np.percentile(pool, 97))) if len(pool) else 1.0
    return features["laugh_threshold"]


def per_second(frames, seconds):
    """0.5 s frame values to one value per second (the max of the pair)."""
    import numpy as np

    out = np.zeros(seconds, dtype="float32")
    n = min(len(frames) // 2, seconds)
    if n:
        out[:n] = np.asarray(frames[:2 * n]).reshape(n, 2).max(axis=1)
    return out


# --------------------------------------------------------------------------
# Fusion
# --------------------------------------------------------------------------

def chat_lag(chat, voice, *, max_lag=20, default=None):
    """Seconds chat trails the streamer in this recording.

    The lag between 2 and ``max_lag`` s at which the chat curve best matches
    the voice curve, if that match is clearly better than none; otherwise
    ``default`` (HIGHLIGHT_CHAT_DELAY, 7 s)."""
    import numpy as np

    if default is None:
        default = _setting("HIGHLIGHT_CHAT_DELAY", 7.0, 0, 30)
    n = min(len(chat), len(voice))
    if n < 120 or not np.any(chat[:n]) or not np.any(voice[:n]):
        return float(default), None
    c = np.asarray(chat[:n], dtype="float64")
    v = np.asarray(voice[:n], dtype="float64")
    c = (c - c.mean()) / (c.std() + 1e-9)
    v = (v - v.mean()) / (v.std() + 1e-9)
    scores = [float(np.dot(v[:n - lag], c[lag:]) / (n - lag)) for lag in range(0, max_lag + 1)]
    best = int(np.argmax(scores[2:])) + 2
    if scores[best] < 0.05 or scores[best] < scores[0] + 0.02:
        return float(default), round(scores[best], 3)
    return float(best), round(scores[best], 3)


def _unit(x):
    """Robust units to 0..1: 1 unit ~0.39, 2 ~0.63, 4 ~0.86."""
    import numpy as np
    return 1.0 - np.exp(-np.asarray(x, dtype="float32") / 2.0)


def fuse(chat, voice, lag, chat_until=None):
    """The hype curve per second on the SOURCE timeline: chat shifted back by
    its lag, plus the streamer, plus a bonus where both agree. Past
    ``chat_until`` (a replay that stops early) the streamer alone counts, so a
    stretch without chat data is not ranked below one with it."""
    import numpy as np

    n = max(len(chat), len(voice))
    c = np.zeros(n, dtype="float32")
    v = np.zeros(n, dtype="float32")
    k = int(round(lag))
    if len(chat):
        shifted = np.asarray(chat, dtype="float32")[k:]
        c[:len(shifted)] = shifted
    v[:len(voice)] = voice
    cn, vn = _unit(c), _unit(v)
    has_chat, has_voice = bool(np.any(c)), bool(np.any(v))
    if has_chat and has_voice:
        hype = 0.5 * cn + 0.35 * vn + 0.6 * np.sqrt(cn * vn)
        if chat_until is not None and chat_until < n:
            hype[int(max(0, chat_until)):] = vn[int(max(0, chat_until)):]
    else:
        hype = cn if has_chat else vn
    return hype, cn, vn


def windows(hype, duration, *, min_secs=30, max_secs=60, setup=18.0, tail=4.0, limit=30,
            floor=None):
    """Candidate clip windows around the hype curve's peaks, best first.

    The event is the stretch around a peak that stays above half its height;
    the window adds ``setup`` seconds before it (what led there) and ``tail``
    after (the reaction settling), then fits [min_secs, max_secs] by trimming
    the setup first. Peaks closer than ``min_secs`` keep the higher one."""
    import numpy as np
    from scipy.signal import find_peaks

    h = np.asarray(hype, dtype="float32")
    if not len(h) or float(h.max()) <= 0:
        return []
    if floor is None:
        floor = max(0.25, float(np.percentile(h, 96)))
    peaks, props = find_peaks(h, height=floor, distance=max(1, int(min_secs)), prominence=0.08)
    out = []
    for p in sorted(peaks, key=lambda i: -h[i]):
        top = float(h[p])
        lo = p
        while lo > 0 and h[lo - 1] >= 0.5 * top and p - lo < max_secs:
            lo -= 1
        hi = p
        while hi < len(h) - 1 and h[hi + 1] >= 0.5 * top and hi - p < max_secs:
            hi += 1
        start, end = lo - setup, hi + 1 + tail
        if end - start > max_secs:
            start = max(start, end - max_secs)
            if end - start > max_secs:
                end = start + max_secs
        if end - start < min_secs:
            start = end - min_secs
        start, end = max(0.0, start), min(float(duration), end)
        if end - start < min_secs:
            # Against an edge of the recording: grow on the side with room.
            end = min(float(duration), start + min_secs)
            start = max(0.0, end - min_secs)
        if end - start < min(10.0, min_secs) or any(
                start < o["end"] and end > o["start"] for o in out):
            continue
        area = float(h[int(start):int(math.ceil(end))].sum())
        out.append({"start": round(start, 2), "end": round(end, 2), "peak": int(p),
                    "hype": round(top, 3), "area": round(area, 2)})
        if len(out) >= limit:
            break
    return out


def _cue(name, value):
    return round(float(value), 2) if value is not None else None


def describe(window, chat_raw, chat_z, voice_z, features, lag):
    """What the curve saw inside one window, for the judge, the ranking and
    the review card: aggregates only, never chat lines or authors."""
    import numpy as np

    a, b = int(window["start"]), int(math.ceil(window["end"]))
    k = int(round(lag))
    out = {"hype": window["hype"], "chat_lag": lag}
    if chat_raw is not None and any(chat_raw["chatters"]):
        lo, hi = min(len(chat_raw["chatters"]), a + k), min(len(chat_raw["chatters"]), b + k)
        chatters = chat_raw["chatters"][lo:hi]
        cats = Counter()
        for i in range(lo, hi):
            cats.update(chat_raw["categories"].get(i, {}))
        before = chat_raw["chatters"][max(0, lo - 300):lo] or [0]
        normal = float(np.median(before)) if before else 0.0
        out["chat"] = {
            "peak_chatters_per_second": int(max(chatters, default=0)),
            "normal_chatters_per_second": round(normal, 2),
            "excitement": _cue("chat", float(np.max(chat_z[a:b])) if b > a else 0),
            "reactions": [name for name, _ in cats.most_common(3)],
            "clip_requests": int(sum(chat_raw["clip_requests"][lo:hi])),
        }
    if voice_z is not None and len(voice_z):
        seg = voice_z[a:b]
        out["streamer"] = {"excitement": _cue("voice", float(np.max(seg)) if len(seg) else 0)}
        if features is not None:
            fa, fb = int(a / FRAME), int(b / FRAME)
            laugh = features["laugh_rhythm"][fa:fb]
            out["streamer"]["laughing"] = bool(
                len(laugh) and float(np.max(laugh)) > laughter_threshold(features) + 0.06)
            db = features["db"][fa:fb]
            if len(db):
                out["streamer"]["loudest_db_over_median"] = round(
                    float(np.max(db) - np.median(features["db"])), 1)
    return out


def viewer_curve(viewer_clips, seconds):
    """Per-second weight of Twitch viewer clips (twitch_signals): people
    clipped exactly this. 1 + log(views)/3 per clip over its own span."""
    import numpy as np

    out = np.zeros(seconds, dtype="float32")
    for clip in viewer_clips or []:
        try:
            a, b = max(0, int(clip["start"])), min(seconds, int(math.ceil(clip["end"])))
            views = max(0.0, float(clip.get("views") or clip.get("view_count") or 0))
        except (KeyError, TypeError, ValueError):
            continue
        if b > a:
            out[a:b] += 1.0 + math.log1p(views) / 3.0
    return out


def scan(media_path, duration=None, *, chat_path=None, source_id=None, chat_offset=0.0,
         min_secs=30, max_secs=60, limit=30, viewer_clips=None):
    """The whole stage: voice + chat -> hype curve -> candidate windows.

    Returns {"candidates": [...], "lag": s, "lag_confidence": r, "chat_status",
    "voice": bool, "markers": {"voice": [...]}, "curve": [...]} where
    ``curve`` is the hype curve at 5 s resolution for the timeline."""
    import numpy as np

    features = voice_features(media_path)
    total = float(duration or (features or {}).get("duration") or 0)
    if total <= 0:
        return None
    seconds = int(math.ceil(total))
    voice = per_second(voice_excitement(features), seconds) if features else np.zeros(seconds, "float32")

    chat_raw, chat_z, status = None, np.zeros(seconds, dtype="float32"), "unavailable"
    if chat_path:
        try:
            messages = chat_hype.read_messages(chat_path, source_id=source_id)
            chat_raw = chat_curve(messages, total, offset=chat_offset)
            if chat_raw["accepted"]:
                chat_z = chat_intensity(chat_raw)
                status = "available"
        except (OSError, ValueError, TypeError, AttributeError):
            status = "invalid"
            chat_raw = None
    lag, confidence = chat_lag(chat_z, voice)
    chat_until = None
    if chat_raw is not None and any(chat_raw["messages"]):
        last = max(i for i, n in enumerate(chat_raw["messages"]) if n)
        if last < seconds - 300:
            # The replay stops well before the recording does (a limit hit).
            chat_until = last - int(round(lag))
    hype, _cn, _vn = fuse(chat_z, voice, lag, chat_until)
    viewers = viewer_curve(viewer_clips, len(hype))
    if viewers.any():
        hype = hype + 0.5 * _unit(viewers)
    found = windows(hype, total, min_secs=min_secs, max_secs=max_secs, limit=limit)
    for w in found:
        w["signals"] = describe(w, chat_raw, chat_z[int(round(lag)):] if len(chat_z) else chat_z,
                                voice, features, lag)
        clipped = sum(1 for c in viewer_clips or []
                      if isinstance(c, dict) and c.get("start", 0) < w["end"] and c.get("end", 0) > w["start"])
        if clipped:
            w["signals"]["viewer_clips"] = clipped
    markers = []
    if features is not None:
        for p in np.argsort(-voice)[:400]:
            if voice[p] < 1.0 or any(abs(int(p) - m["time"]) < 20 for m in markers):
                continue
            markers.append({"kind": "voice", "time": int(p), "start": max(0, int(p) - 2),
                            "end": min(seconds, int(p) + 4), "excitement": round(float(voice[p]), 2)})
            if len(markers) >= 60:
                break
    step = 5
    coarse = [round(float(hype[i:i + step].max()), 3) for i in range(0, len(hype), step)]
    return {"candidates": found, "lag": lag, "lag_confidence": confidence, "chat_status": status,
            "chat_until": chat_until,
            "voice": features is not None,
            "markers": {"voice": sorted(markers, key=lambda m: m["time"])},
            "curve": {"step": step, "values": coarse}}
