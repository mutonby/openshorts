"""Independent evidence for discovery, review and editorial ranking."""
from collections import defaultdict
import math
from statistics import median
import subprocess

import chat_hype


def audio_peaks(path, duration):
    """Find changes relative to nearby mixed audio. These are leads, not speaker labels."""
    import numpy as np

    command = ["ffmpeg", "-nostdin", "-v", "error", "-threads", "1", "-i", str(path),
               "-map", "0:a:0", "-ac", "1", "-ar", "4000", "-f", "s16le", "-"]
    levels = []
    with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL) as run:
        while block := run.stdout.read(16000):
            samples = np.frombuffer(block[:len(block) // 2 * 2], dtype="<i2").astype("float32")
            if not len(samples):
                break
            mean_square = float(np.mean(samples * samples)) / (32768 ** 2)
            levels.append(round(10 * math.log10(max(1e-10, mean_square)), 1))
        if run.wait():
            return []
    peaks = []
    for i in range(15, len(levels)):
        last = levels[max(0, i - 30):i]
        base = median(last)
        db = levels[i]
        if db >= -35 and db - base >= 8 and db - last[-1] >= 4:
            start = i * 2
            if start < duration:
                peaks.append({"kind": "audio", "time": start, "start": max(0, start - 2),
                              "end": min(duration, start + 4), "change_db": round(db - base, 1)})
    ranked = sorted(peaks, key=lambda p: p["change_db"], reverse=True)[:60]
    return sorted(ranked, key=lambda p: p["time"])


def chat_evidence(path, duration, *, source_id=None, offset=0, window=10, baseline=90,
                  reaction_delay=7.0, emotes=()):
    if not path:
        return {"chat_status": "unavailable", "chat": []}
    try:
        peaks, messages = chat_hype.spikes(
            chat_hype.read_messages(path, source_id=source_id), duration,
            offset=offset, window=window, baseline_seconds=baseline,
            reaction_delay=reaction_delay, emotes=emotes)
    except (OSError, ValueError, TypeError, AttributeError):
        return {"chat_status": "invalid", "chat": []}
    return {"chat_status": "available" if messages else "unavailable", "chat": peaks,
            "chat_messages": messages}


def related(start, end, evidence):
    if not evidence:
        return {}
    out = {}
    for kind, key in (("chat", "ratio"), ("audio", "change_db")):
        points = [m for m in evidence.get(kind, []) if m["start"] < end and m["end"] > start]
        if points:
            out[kind] = max(points, key=lambda m: m[key])
    return out


def additional_windows(windows, shortlist, evidence, limit=6):
    """Add evidence-led windows without taking places from transcript-led discovery."""
    chosen = {w["id"] for w in shortlist}
    groups = defaultdict(list)
    for w in windows:
        cues = related(w["start"], w["end"], evidence)
        for name, cue in cues.items():
            strength = cue.get("ratio") if name == "chat" else cue.get("change_db")
            groups[name].append((strength, w))
    for items in groups.values():
        items.sort(key=lambda row: -row[0])

    added = []
    while any(groups.values()) and len(added) < limit:
        for name in ("chat", "audio"):
            if len(added) >= limit:
                break
            while groups[name]:
                _, w = groups[name].pop(0)
                if w["id"] not in chosen:
                    chosen.add(w["id"])
                    added.append(w)
                    break
    return added
