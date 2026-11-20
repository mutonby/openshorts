"""Helpers for an audio-first YouTube clip workflow.

The full source timeline is analysed from audio. Only the selected video ranges
are fetched, then those ranges are concatenated into a compact source file for
the existing editor/render pipeline.
"""
from __future__ import annotations

import copy
import json
import os
import subprocess
from typing import Iterable


def probe_duration(path: str) -> float:
    """Read media duration with ffprobe; raise a useful error if unavailable."""
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", path],
        check=True, capture_output=True, text=True, timeout=60,
    )
    try:
        duration = float(result.stdout.strip())
    except (TypeError, ValueError):
        raise RuntimeError(f"Could not read media duration: {path}")
    if duration <= 0:
        raise RuntimeError(f"Media has no usable duration: {path}")
    return duration


def plan_ranges(clips: Iterable[dict], duration: float, padding: float = 1.5):
    """Return merged source ranges and a mapping for each input clip.

    Each range is [start, end] in source seconds. Overlapping padded ranges are
    merged so the same part of a stream is never downloaded twice.
    """
    prepared = []
    for index, clip in enumerate(clips or []):
        try:
            start = max(0.0, float(clip["start"]) - padding)
            end = min(float(duration), float(clip["end"]) + padding)
        except (KeyError, TypeError, ValueError):
            continue
        if end <= start:
            continue
        prepared.append((start, end, index))
    prepared.sort()
    merged = []
    for start, end, index in prepared:
        if merged and start <= merged[-1]["end"]:
            merged[-1]["end"] = max(merged[-1]["end"], end)
            merged[-1]["clip_indexes"].append(index)
        else:
            merged.append({"start": start, "end": end, "clip_indexes": [index]})
    return merged


def remap_clips(clips: list[dict], ranges: list[dict]):
    """Map source-timeline clip times onto the concatenated selected-source file."""
    mapped = copy.deepcopy(clips)
    offset = 0.0
    for item in ranges:
        source_start, source_end = item["start"], item["end"]
        for index in item["clip_indexes"]:
            clip = mapped[index]
            original_start, original_end = float(clip["start"]), float(clip["end"])
            clip["source_start"] = original_start
            clip["source_end"] = original_end
            clip["start"] = round(offset + original_start - source_start, 3)
            clip["end"] = round(offset + original_end - source_start, 3)
            clip["source_range"] = [round(source_start, 3), round(source_end, 3)]
        item["concat_start"] = offset
        item["concat_end"] = offset + source_end - source_start
        offset = item["concat_end"]
    return mapped, offset


def remap_transcript(transcript: dict | None, ranges: list[dict]):
    """Copy transcript words/segments into the concatenated selected-source timeline."""
    if not transcript:
        return transcript
    out = {k: copy.deepcopy(v) for k, v in transcript.items() if k != "segments"}
    out_segments = []
    for item in ranges:
        source_start, source_end = item["start"], item["end"]
        offset = item["concat_start"] - source_start
        for original in transcript.get("segments") or []:
            try:
                seg_start = float(original.get("start", 0))
                seg_end = float(original.get("end", seg_start))
            except (TypeError, ValueError):
                continue
            if seg_end <= source_start or seg_start >= source_end:
                continue
            seg = copy.deepcopy(original)
            seg["start"] = round(max(seg_start, source_start) + offset, 3)
            seg["end"] = round(min(seg_end, source_end) + offset, 3)
            if isinstance(seg.get("words"), list):
                words = []
                for word in seg["words"]:
                    try:
                        ws = float(word.get("start", word.get("s", 0)))
                        we = float(word.get("end", word.get("e", ws)))
                    except (AttributeError, TypeError, ValueError):
                        continue
                    if we <= source_start or ws >= source_end:
                        continue
                    w = copy.deepcopy(word)
                    if "start" in w:
                        w["start"] = round(max(ws, source_start) + offset, 3)
                    if "end" in w:
                        w["end"] = round(min(we, source_end) + offset, 3)
                    if "s" in w:
                        w["s"] = round(max(ws, source_start) + offset, 3)
                    if "e" in w:
                        w["e"] = round(min(we, source_end) + offset, 3)
                    words.append(w)
                seg["words"] = words
            out_segments.append(seg)
    out_segments.sort(key=lambda s: float(s.get("start", 0)))
    out["segments"] = out_segments
    out["text"] = " ".join(str(s.get("text") or "").strip() for s in out_segments).strip()
    return out


def concat_segments(paths: list[str], output_path: str) -> str:
    """Concatenate selected ranges into one compact source file.

    Try stream-copy first. If segments differ in container/stream parameters,
    normalize only the selected footage to H.264/AAC; the main renderer still
    performs its usual framing, hooks, captions, and final encoding.
    """
    if not paths:
        raise ValueError("No selected video segments were downloaded")
    if len(paths) == 1:
        import shutil
        shutil.copy2(paths[0], output_path)
        return output_path

    list_path = output_path + ".concat.txt"
    try:
        with open(list_path, "w", encoding="utf-8") as handle:
            for path in paths:
                absolute = os.path.abspath(path).replace("\\", "/")
                escaped = absolute.replace("'", "'\\''")
                handle.write(f"file '{escaped}'\n")
        base = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                "-f", "concat", "-safe", "0", "-i", list_path]
        try:
            subprocess.run(base + ["-c", "copy", "-movflags", "+faststart", output_path],
                           check=True, capture_output=True, timeout=1800)
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
            try:
                if os.path.exists(output_path):
                    os.remove(output_path)
            except OSError:
                pass
            subprocess.run(
                base + ["-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
                        "-c:a", "aac", "-b:a", "160k", "-movflags", "+faststart",
                        output_path],
                check=True, capture_output=True, timeout=3600,
            )
    finally:
        try:
            os.remove(list_path)
        except OSError:
            pass
    return output_path
