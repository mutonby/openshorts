import time
import cv2
import scenedetect
import subprocess
import argparse
import glob
import re
import sys
import math
from scenedetect import VideoManager, SceneManager
from scenedetect.detectors import ContentDetector
from ultralytics import YOLO
import torch
import os
import numpy as np
from tqdm import tqdm
import yt_dlp
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
# import whisper (replaced by faster_whisper inside function)
from google import genai
from google.genai import types as genai_types
from dotenv import load_dotenv
import json
import shutil
from typing import List, Optional
from pydantic import BaseModel
from clip_selection import build_transcript_windows, snap_clip_to_words

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module='google.protobuf')

# Load environment variables
load_dotenv()

# --- Constants ---
ASPECT_RATIO = 9 / 16
OUTPUT_FORMATS = ("auto", "vertical", "horizontal", "square")
# Watermark: subtle centered overlay so rendered clips can't be re-uploaded as
# someone else's work. Configure via env; WATERMARK_TEXT wins over the image.
WATERMARK_ENABLED = os.environ.get("WATERMARK_ENABLED", "1").strip().lower() not in ("0", "false", "off", "no")
WATERMARK_TEXT = os.environ.get("WATERMARK_TEXT", "").strip()
WATERMARK_IMAGE = os.environ.get("WATERMARK_IMAGE", "").strip()
try:
    WATERMARK_OPACITY = min(0.5, max(0.01, float(os.environ.get("WATERMARK_OPACITY", "0.08"))))
except ValueError:
    WATERMARK_OPACITY = 0.08
try:
    WATERMARK_WIDTH_FRACTION = min(0.9, max(0.1, float(os.environ.get("WATERMARK_WIDTH_FRACTION", "0.45"))))
except ValueError:
    WATERMARK_WIDTH_FRACTION = 0.45
GEMINI_MAX_ATTEMPTS = 3
EVENT_PREFIX = "__JOB_EVENT__"
HEARTBEAT_INTERVAL_SECONDS = int(os.environ.get("JOB_HEARTBEAT_INTERVAL_SECONDS", "5"))
GEMINI_SLOW_WARNING_SECONDS = int(os.environ.get("GEMINI_SLOW_WARNING_SECONDS", "180"))
GEMINI_REQUEST_TIMEOUT_SECONDS = int(os.environ.get("GEMINI_REQUEST_TIMEOUT_SECONDS", "600"))
GEMINI_MAX_TIMEOUT_SECONDS = int(os.environ.get("GEMINI_MAX_TIMEOUT_SECONDS", "900"))
GEMINI_REQUEST_TIMEOUT_SECONDS = min(GEMINI_REQUEST_TIMEOUT_SECONDS, GEMINI_MAX_TIMEOUT_SECONDS)
GEMINI_WINDOW_SECONDS = int(os.environ.get("GEMINI_WINDOW_SECONDS", "90"))
GEMINI_WINDOW_OVERLAP_SECONDS = int(os.environ.get("GEMINI_WINDOW_OVERLAP_SECONDS", "30"))
# Analysis model, overridable per task (GEMINI_MODEL_ANALYSIS) or globally (GEMINI_MODEL).
GEMINI_ANALYSIS_MODEL = (
    os.environ.get("GEMINI_MODEL_ANALYSIS")
    or os.environ.get("GEMINI_MODEL")
    or "gemini-3-flash-preview"
)
GEMINI_SCORE_BATCH_SIZE = int(os.environ.get("GEMINI_SCORE_BATCH_SIZE", "8"))
GEMINI_DETAIL_BATCH_SIZE = int(os.environ.get("GEMINI_DETAIL_BATCH_SIZE", "4"))
GEMINI_SHORTLIST_LIMIT = int(os.environ.get("GEMINI_SHORTLIST_LIMIT", "10"))
GEMINI_WORKER_SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gemini_worker.py")
PHASE_RANGES = {
    "queued": (0.0, 2.0),
    "download": (2.0, 20.0),
    "transcribe": (20.0, 55.0),
    "analyze": (55.0, 78.0),
    "render": (78.0, 95.0),
    "finalize": (95.0, 100.0),
    "completed": (100.0, 100.0),
}

# ETA model: learned from THIS machine's finished jobs (.phase_stats.json).
# Factors are processing-seconds per second of source video; analyze is
# roughly constant per job. Defaults seed the estimate until data exists.
PHASE_STATS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".phase_stats.json")
PHASE_ETA_ORDER = ["download", "transcribe", "analyze", "render", "finalize"]
PHASE_DEFAULT_FACTORS = {"download": 0.05, "transcribe": 0.85, "render": 0.35, "finalize": 0.005}
ANALYZE_DEFAULT_SECONDS = 60.0


def _load_phase_factors():
    """Median factor per phase from the last finished jobs; defaults otherwise."""
    factors = dict(PHASE_DEFAULT_FACTORS)
    factors["analyze"] = ANALYZE_DEFAULT_SECONDS
    try:
        with open(PHASE_STATS_FILE, "r", encoding="utf-8") as f:
            history = json.load(f)
        for phase, samples in history.items():
            values = sorted(float(v) for v in samples if v is not None)
            if values:
                factors[phase] = values[len(values) // 2]
    except Exception:
        pass
    return factors


def _record_phase_stats(phase_durations, video_duration):
    """Persist per-phase timing of a finished job (keeps the last 10 samples)."""
    if not video_duration or video_duration <= 0:
        return
    try:
        history = {}
        if os.path.exists(PHASE_STATS_FILE):
            with open(PHASE_STATS_FILE, "r", encoding="utf-8") as f:
                history = json.load(f)
        for phase, seconds in phase_durations.items():
            if phase not in PHASE_ETA_ORDER or seconds <= 0:
                continue
            value = seconds if phase == "analyze" else seconds / float(video_duration)
            history.setdefault(phase, []).append(round(value, 4))
            history[phase] = history[phase][-10:]
        with open(PHASE_STATS_FILE, "w", encoding="utf-8") as f:
            json.dump(history, f)
    except Exception:
        pass  # stats are best-effort; never break a job over them


class JobReporter:
    def __init__(self, job_id: Optional[str] = None):
        self.job_id = job_id
        self.started_at = time.time()
        self.phase = "queued"
        self.phase_label = "Queued"
        self.phase_started_at = self.started_at
        self.phase_progress_percent = 0.0
        self.progress_percent = 0.0
        self.last_heartbeat_at = 0.0
        self.video_duration = None
        self.phase_durations = {}
        self.phase_factors = _load_phase_factors()
        self._estimate_announced = False

    def _overall_progress(self, phase: Optional[str] = None, phase_progress_percent: Optional[float] = None) -> float:
        current_phase = phase or self.phase
        phase_percent = self.phase_progress_percent if phase_progress_percent is None else phase_progress_percent
        start, end = PHASE_RANGES.get(current_phase, (self.progress_percent, self.progress_percent))
        phase_percent = max(0.0, min(100.0, float(phase_percent)))
        if end <= start:
            return max(0.0, min(100.0, end))
        return round(start + ((end - start) * (phase_percent / 100.0)), 2)

    def _phase_total_estimate(self, phase: str) -> Optional[float]:
        """Expected total duration of a phase on this machine, in seconds."""
        if phase == "analyze":
            return float(self.phase_factors.get("analyze", ANALYZE_DEFAULT_SECONDS))
        factor = self.phase_factors.get(phase)
        if factor is None or not self.video_duration:
            return None
        return float(factor) * float(self.video_duration)

    def _estimate_remaining_seconds(self) -> Optional[int]:
        """Per-phase ETA: live measured rate for the current phase plus learned
        estimates for the phases still ahead. Far more accurate than the old
        linear extrapolation over the (arbitrarily weighted) global percent."""
        if self.phase not in PHASE_ETA_ORDER:
            return None
        now = time.time()
        in_phase = max(0.0, now - self.phase_started_at)
        phase_percent = max(0.0, min(100.0, self.phase_progress_percent))

        if phase_percent >= 3.0 and in_phase >= 5.0:
            # Real measured speed of the running phase.
            remaining = max(0.0, (in_phase * 100.0 / phase_percent) - in_phase)
        else:
            estimate = self._phase_total_estimate(self.phase)
            if estimate is None:
                return None
            remaining = max(0.0, estimate - in_phase)

        for upcoming in PHASE_ETA_ORDER[PHASE_ETA_ORDER.index(self.phase) + 1:]:
            estimate = self._phase_total_estimate(upcoming)
            if estimate:
                remaining += estimate
        return int(remaining)

    def _default_eta_seconds(self, overall_progress: float) -> Optional[int]:
        estimated = self._estimate_remaining_seconds()
        if estimated is not None:
            return estimated
        elapsed = time.time() - self.started_at
        if overall_progress <= 0.0:
            return None
        remaining = elapsed * ((100.0 - overall_progress) / overall_progress)
        return max(0, int(remaining))

    def _maybe_announce_total_estimate(self):
        """Once the video duration is known, tell the user the expected total."""
        if self._estimate_announced or not self.video_duration:
            return
        self._estimate_announced = True
        total = 0.0
        for phase in PHASE_ETA_ORDER:
            estimate = self._phase_total_estimate(phase)
            if estimate:
                total += estimate
        if total <= 0:
            return
        minutes = max(1, int(round(total / 60.0)))
        self.emit(
            "estimate",
            f"⏱️ Estimated total processing time: ~{minutes} min for this video.",
            important=True,
            total_estimate_seconds=int(total),
        )

    def emit(self, event_type: str, message: Optional[str] = None, **extra):
        if extra.get("video_duration_seconds"):
            self.video_duration = float(extra["video_duration_seconds"])
        payload = {
            "type": event_type,
            "timestamp": time.time(),
            "job_id": self.job_id,
            "phase": extra.pop("phase", self.phase),
            "phase_label": extra.pop("phase_label", self.phase_label),
            "phase_progress_percent": extra.pop("phase_progress_percent", self.phase_progress_percent),
            "progress_percent": extra.pop("progress_percent", self.progress_percent),
        }
        eta_seconds = extra.pop("eta_seconds", None)
        if eta_seconds is None:
            eta_seconds = self._default_eta_seconds(payload["progress_percent"])
        payload["eta_seconds"] = eta_seconds
        if message:
            payload["message"] = message
        payload.update(extra)
        print(f"{EVENT_PREFIX}{json.dumps(payload, ensure_ascii=False)}", flush=True)
        if event_type != "estimate":
            self._maybe_announce_total_estimate()

    def set_phase(self, phase: str, label: str, *, message: Optional[str] = None, phase_progress_percent: float = 0.0, **extra):
        # Record how long the finished phase actually took (feeds the ETA model).
        if self.phase in PHASE_ETA_ORDER:
            self.phase_durations[self.phase] = time.time() - self.phase_started_at
        self.phase = phase
        self.phase_label = label
        self.phase_started_at = time.time()
        self.phase_progress_percent = phase_progress_percent
        self.progress_percent = self._overall_progress(phase=phase, phase_progress_percent=phase_progress_percent)
        self.emit(
            "phase",
            message or label,
            phase=phase,
            phase_label=label,
            phase_progress_percent=phase_progress_percent,
            progress_percent=self.progress_percent,
            **extra,
        )

    def progress(self, phase_progress_percent: float, *, message: Optional[str] = None, important: bool = False, **extra):
        self.phase_progress_percent = max(0.0, min(100.0, float(phase_progress_percent)))
        self.progress_percent = self._overall_progress(phase_progress_percent=self.phase_progress_percent)
        self.emit(
            "progress",
            message,
            phase_progress_percent=self.phase_progress_percent,
            progress_percent=self.progress_percent,
            important=important,
            **extra,
        )

    def heartbeat(self, *, message: Optional[str] = None, force: bool = False, **extra):
        now = time.time()
        if not force and now - self.last_heartbeat_at < HEARTBEAT_INTERVAL_SECONDS:
            return
        self.last_heartbeat_at = now
        self.emit("heartbeat", message, **extra)

    def warning(self, message: str, **extra):
        self.emit("warning", message, important=True, **extra)

    def error(self, message: str, *, resumable: bool = True, **extra):
        self.emit("error", message, important=True, resumable=resumable, **extra)

    def artifact(self, kind: str, path: str, *, message: Optional[str] = None, important: bool = True, **extra):
        self.emit(
            "artifact",
            message or f"Saved {kind} to {path}",
            important=important,
            artifact={"kind": kind, "path": path},
            **extra,
        )

    def summary(self, status: str, message: str, *, resumable: bool = False, **extra):
        if status == "completed":
            # Close the timing of the final phase and persist what we measured,
            # so the next job's ETA is calibrated to THIS machine.
            if self.phase in PHASE_ETA_ORDER:
                self.phase_durations[self.phase] = time.time() - self.phase_started_at
            _record_phase_stats(self.phase_durations, self.video_duration)
        phase = "completed" if status == "completed" else self.phase
        progress_percent = 100.0 if status == "completed" else self.progress_percent
        self.emit(
            "summary",
            message,
            status=status,
            phase=phase,
            phase_label=self.phase_label if status != "completed" else "Completed",
            progress_percent=progress_percent,
            phase_progress_percent=100.0 if status == "completed" else self.phase_progress_percent,
            resumable=resumable,
            **extra,
        )


JOB_REPORTER = JobReporter()


def set_job_reporter(reporter: JobReporter):
    global JOB_REPORTER
    JOB_REPORTER = reporter


def _start_keepalive(interval_seconds=25):
    """Emit a heartbeat every few seconds regardless of pipeline progress.

    Whisper/FFmpeg can crunch for minutes without emitting an event; without
    this, the server's 90s stall detector flags perfectly healthy jobs as
    stalled. With it, "stalled" means the process is truly frozen or asleep."""
    import threading

    def _beat():
        while True:
            time.sleep(interval_seconds)
            try:
                JOB_REPORTER.heartbeat("Worker alive.")
            except Exception:
                pass

    threading.Thread(target=_beat, daemon=True, name="keepalive").start()


def _prevent_windows_sleep():
    """Keep Windows awake while a job runs.

    Lid-close / idle standby freezes the worker mid-transcription and the job
    looks hung for hours. The execution-state flag clears automatically when
    the process exits, so no cleanup is needed. The display may still turn off."""
    if sys.platform != 'win32':
        return
    try:
        import ctypes
        ES_CONTINUOUS = 0x80000000
        ES_SYSTEM_REQUIRED = 0x00000001
        ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)
        print("🔒 Standby disabled while this job is running (display may still turn off).")
    except Exception as e:
        print(f"⚠️ Could not disable standby: {e}")

def _save_json_file(path, payload):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def _save_text_file(path, text):
    with open(path, 'w', encoding='utf-8') as f:
        f.write(text or "")


def _extract_words_for_analysis(transcript_result):
    # Rounded to 2 decimals: full float precision only wastes prompt tokens.
    words = []
    for segment in transcript_result.get('segments', []):
        for word in segment.get('words', []):
            words.append({
                'w': word.get('word', ''),
                's': round(float(word.get('start', 0)), 2),
                'e': round(float(word.get('end', 0)), 2),
            })
    return words


def _build_analysis_input_payload(transcript_result, video_duration, source_url=None, input_video=None):
    return {
        "schema_version": 1,
        "source_url": source_url,
        "input_video": input_video,
        "video_duration": round(float(video_duration), 3),
        "transcript": transcript_result,
    }


def _save_json_checkpoint(output_dir, video_title, suffix, payload):
    path = os.path.join(output_dir, f"{video_title}_{suffix}.json")
    _save_json_file(path, payload)
    JOB_REPORTER.artifact(suffix, path)
    return path


def _save_text_checkpoint(output_dir, video_title, suffix, text):
    path = os.path.join(output_dir, f"{video_title}_{suffix}.txt")
    _save_text_file(path, text)
    JOB_REPORTER.artifact(suffix, path, important=False)
    return path


def _extract_text_for_range(transcript_result, start, end):
    parts = []
    for segment in transcript_result.get("segments", []):
        seg_start = float(segment.get("start", 0))
        seg_end = float(segment.get("end", 0))
        if seg_end <= start or seg_start >= end:
            continue
        text = str(segment.get("text") or "").strip()
        if text:
            parts.append(text)
    return " ".join(parts).strip()


def _extract_words_for_range(words, start, end):
    extracted = []
    for word in words:
        word_start = float(word.get("s", 0))
        word_end = float(word.get("e", 0))
        if word_end <= start or word_start >= end:
            continue
        extracted.append(word)
    return extracted


def _build_transcript_windows(transcript_result, video_duration, window_seconds=GEMINI_WINDOW_SECONDS, overlap_seconds=GEMINI_WINDOW_OVERLAP_SECONDS):
    # Segment-aligned windowing lives in clip_selection so it stays unit-testable.
    return build_transcript_windows(transcript_result, video_duration, window_seconds, overlap_seconds)


def _iter_batches(items, batch_size):
    for index in range(0, len(items), batch_size):
        yield index // batch_size, items[index:index + batch_size]


def _merge_cost_analyses(cost_analyses):
    valid = [cost for cost in cost_analyses if cost]
    if not valid:
        return None
    return {
        "input_tokens": sum(cost.get("input_tokens", 0) for cost in valid),
        "output_tokens": sum(cost.get("output_tokens", 0) for cost in valid),
        "thinking_tokens": sum(cost.get("thinking_tokens", 0) for cost in valid),
        "input_cost": sum(cost.get("input_cost", 0.0) for cost in valid),
        "output_cost": sum(cost.get("output_cost", 0.0) for cost in valid),
        "total_cost": sum(cost.get("total_cost", 0.0) for cost in valid),
        "model": valid[-1].get("model", GEMINI_ANALYSIS_MODEL),
        "price_estimated": any(cost.get("price_estimated") for cost in valid),
    }


def _normalize_scored_windows(payload, video_duration):
    if not isinstance(payload, dict):
        raise ValueError("Scoring payload was not a JSON object.")
    windows = payload.get("windows")
    if not isinstance(windows, list):
        raise ValueError("Scoring payload did not contain a valid 'windows' array.")
    normalized = []
    for item in windows:
        if not isinstance(item, dict):
            continue
        try:
            start = round(max(0.0, float(item["start"])), 3)
            end = round(min(float(video_duration), float(item["end"])), 3)
            score = int(item.get("score", 0))
        except (KeyError, TypeError, ValueError):
            continue
        if end <= start:
            continue
        normalized.append({
            "id": str(item.get("id") or ""),
            "start": start,
            "end": end,
            "score": max(0, min(100, score)),
            "reason": str(item.get("reason") or "").strip(),
        })
    return normalized


def _call_gemini_worker(mode, payload, *, output_dir, video_title, strategy, batch_index, total_batches, attempt, timeout_seconds=GEMINI_REQUEST_TIMEOUT_SECONDS):
    request_path = os.path.join(output_dir, f"{video_title}_{mode}_batch_{batch_index + 1}_attempt_{attempt}.request.json")
    response_path = os.path.join(output_dir, f"{video_title}_{mode}_batch_{batch_index + 1}_attempt_{attempt}.response.json")
    _save_json_file(request_path, payload)

    worker_cmd = [
        sys.executable,
        "-u",
        GEMINI_WORKER_SCRIPT,
        "--mode",
        mode,
        "--input",
        request_path,
        "--output",
        response_path,
        "--strategy",
        strategy,
        "--model",
        GEMINI_ANALYSIS_MODEL,
    ]

    process = subprocess.Popen(worker_cmd)
    start_time = time.time()
    slow_warning_emitted = False

    try:
        while process.poll() is None:
            elapsed = time.time() - start_time
            if not slow_warning_emitted and elapsed >= GEMINI_SLOW_WARNING_SECONDS:
                JOB_REPORTER.emit(
                    "slow",
                    f"Gemini request is taking longer than expected ({int(elapsed)}s).",
                    important=True,
                    category="gemini",
                    attempt=attempt,
                    batch_index=batch_index + 1,
                    total_batches=total_batches,
                )
                slow_warning_emitted = True
            JOB_REPORTER.heartbeat(
                message=f"Gemini {mode} batch {batch_index + 1}/{total_batches} is still running...",
                category="gemini",
                attempt=attempt,
                batch_index=batch_index + 1,
                total_batches=total_batches,
            )
            if elapsed >= timeout_seconds:
                raise TimeoutError(
                    f"Gemini {mode} batch {batch_index + 1}/{total_batches} exceeded the timeout of {timeout_seconds}s."
                )
            time.sleep(1)
    finally:
        # On timeout/exception, don't leave the Gemini worker subprocess orphaned.
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()

    if process.returncode != 0:
        raise RuntimeError(
            f"Gemini worker failed for {mode} batch {batch_index + 1}/{total_batches} with exit code {process.returncode}."
        )
    if not os.path.exists(response_path):
        raise RuntimeError(f"Gemini worker did not produce a response file for {mode} batch {batch_index + 1}/{total_batches}.")
    with open(response_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _strip_code_fences(text):
    text = (text or "").strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines:
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return text


def _extract_json_candidate(text):
    cleaned = _strip_code_fences(text)
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end != -1 and end > start:
        return cleaned[start:end + 1]
    return cleaned


def _escape_invalid_unicode_escapes(text):
    chars = []
    i = 0
    while i < len(text):
        if text[i] == "\\" and i + 1 < len(text) and text[i + 1] == "u":
            hex_digits = text[i + 2:i + 6]
            if len(hex_digits) < 4 or any(ch not in "0123456789abcdefABCDEF" for ch in hex_digits):
                chars.append("\\\\u")
                i += 2
                continue
        chars.append(text[i])
        i += 1
    return "".join(chars)


def _parse_json_response_text(text):
    if not text:
        raise ValueError("Gemini returned an empty response body.")

    candidate = _extract_json_candidate(text).replace("\x00", "").strip()
    if not candidate:
        raise ValueError("Gemini response did not contain a JSON object.")

    parse_attempts = [candidate]
    sanitized_candidate = _escape_invalid_unicode_escapes(candidate)
    if sanitized_candidate != candidate:
        parse_attempts.append(sanitized_candidate)

    last_error = None
    for parse_candidate in parse_attempts:
        try:
            return json.loads(parse_candidate)
        except json.JSONDecodeError as e:
            last_error = e

    raise ValueError(f"Failed to parse Gemini JSON response: {last_error}")


def _get_response_text(response):
    try:
        text = response.text
        if text:
            return text
    except Exception:
        pass

    parts = []
    for candidate in getattr(response, "candidates", []) or []:
        content = getattr(candidate, "content", None)
        for part in getattr(content, "parts", []) or []:
            part_text = getattr(part, "text", None)
            if part_text:
                parts.append(part_text)

    return "\n".join(parts).strip()


def _calculate_cost_analysis(response, model_name):
    usage = getattr(response, "usage_metadata", None)
    if not usage:
        return None

    input_price_per_million = 0.10
    output_price_per_million = 0.40

    prompt_tokens = usage.prompt_token_count
    output_tokens = usage.candidates_token_count

    input_cost = (prompt_tokens / 1_000_000) * input_price_per_million
    output_cost = (output_tokens / 1_000_000) * output_price_per_million
    total_cost = input_cost + output_cost

    return {
        "input_tokens": prompt_tokens,
        "output_tokens": output_tokens,
        "input_cost": input_cost,
        "output_cost": output_cost,
        "total_cost": total_cost,
        "model": model_name,
    }


def _normalize_shorts_payload(payload, video_duration, words=None):
    if isinstance(payload, BaseModel):
        payload = payload.model_dump()

    if not isinstance(payload, dict):
        raise ValueError("Gemini response was not a JSON object.")

    shorts = payload.get("shorts")
    if not isinstance(shorts, list):
        raise ValueError("Gemini response did not contain a valid 'shorts' array.")

    normalized_shorts = []
    seen_ranges = set()
    max_end = round(float(video_duration), 3)

    for clip in shorts:
        if not isinstance(clip, dict):
            continue

        try:
            start = round(max(0.0, float(clip["start"])), 3)
            end = round(min(max_end, float(clip["end"])), 3)
        except (KeyError, TypeError, ValueError):
            continue

        if end <= start:
            continue

        duration = round(end - start, 3)
        if duration > 60.0:
            end = round(min(max_end, start + 60.0), 3)
            duration = round(end - start, 3)

        if duration < 15.0:
            extended_end = round(min(max_end, start + 15.0), 3)
            if round(extended_end - start, 3) >= 15.0:
                end = extended_end
                duration = round(end - start, 3)

        if duration < 15.0 or duration > 60.0:
            continue

        # Snap onto word boundaries + surrounding silence: the LLM's second
        # guesses are approximate, the Whisper timestamps are ground truth.
        if words:
            start, end = snap_clip_to_words(start, end, words, max_end)
            duration = round(end - start, 3)

        clip_key = (start, end)
        if clip_key in seen_ranges:
            continue
        seen_ranges.add(clip_key)

        normalized_shorts.append({
            "start": start,
            "end": end,
            "video_description_for_tiktok": str(clip.get("video_description_for_tiktok") or "").strip() or "AI-selected highlight clip.",
            "video_description_for_instagram": str(clip.get("video_description_for_instagram") or "").strip() or "AI-selected highlight clip.",
            "video_title_for_youtube_short": str(clip.get("video_title_for_youtube_short") or "").strip()[:100] or "AI-selected highlight",
            "viral_hook_text": str(clip.get("viral_hook_text") or "").strip()[:120] or "Watch this",
            "source_window_id": str(clip.get("source_window_id") or "").strip() or None,
            "predicted_score": int(clip.get("predicted_score", 0) or 0),
        })

    if not normalized_shorts:
        raise ValueError("Gemini did not return any valid clips after validation.")

    return {"shorts": normalized_shorts[:15]}


def _build_fallback_metadata(video_title, transcript, duration, output_filename, analysis_error, attempts, cost_analysis=None):
    title = video_title.replace("_", " ").strip() or "Fallback video"

    metadata = {
        "schema_version": 1,
        "processing_mode": "full_video_fallback",
        "analysis_status": "fallback",
        "analysis_error": analysis_error,
        "analysis_attempts": attempts,
        "transcript": transcript,
        "shorts": [
            {
                "start": 0.0,
                "end": round(float(duration), 3),
                "video_description_for_tiktok": "Automatic full-video fallback because AI clip detection failed.",
                "video_description_for_instagram": "Automatic full-video fallback because AI clip detection failed.",
                "video_title_for_youtube_short": title[:100],
                "viral_hook_text": "Automatic fallback",
                "output_filename": os.path.basename(output_filename),
            }
        ],
    }

    if cost_analysis:
        metadata["cost_analysis"] = cost_analysis

    return metadata

# Load the YOLO model once (Keep for backup or scene analysis if needed)
model = YOLO('yolov8n.pt')

# --- MediaPipe Setup (Tasks API for mediapipe >= 0.10.21) ---
# Auto-download the face detection model if not present
_FACE_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'blaze_face_short_range.tflite')
if not os.path.exists(_FACE_MODEL_PATH):
    print("📥 Downloading MediaPipe face detection model...")
    import urllib.request
    urllib.request.urlretrieve(
        'https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/latest/blaze_face_short_range.tflite',
        _FACE_MODEL_PATH
    )
    print("✅ Model downloaded.")

_face_detector_options = mp_vision.FaceDetectorOptions(
    base_options=mp_python.BaseOptions(model_asset_path=_FACE_MODEL_PATH),
    running_mode=mp_vision.RunningMode.IMAGE,
    min_detection_confidence=0.5
)
face_detector = mp_vision.FaceDetector.create_from_options(_face_detector_options)

class SmoothedCameraman:
    """
    Handles smooth camera movement.
    Simplified Logic: "Heavy Tripod"
    Only moves if the subject leaves the center safe zone.
    Moves slowly and linearly.
    """
    def __init__(self, output_width, output_height, video_width, video_height, aspect_ratio=ASPECT_RATIO):
        self.output_width = output_width
        self.output_height = output_height
        self.video_width = video_width
        self.video_height = video_height

        # Initial State
        self.current_center_x = video_width / 2
        self.target_center_x = video_width / 2

        # Calculate crop dimensions once
        self.crop_height = video_height
        self.crop_width = int(self.crop_height * aspect_ratio)
        if self.crop_width > video_width:
             self.crop_width = video_width
             self.crop_height = int(self.crop_width / aspect_ratio)
             
        # Safe Zone: 20% of the video width
        # As long as the target is within this zone relative to current center, DO NOT MOVE.
        self.safe_zone_radius = self.crop_width * 0.25

    def update_target(self, face_box):
        """
        Updates the target center based on detected face/person.
        """
        if face_box:
            x, y, w, h = face_box
            self.target_center_x = x + w / 2
    
    def get_crop_box(self, force_snap=False):
        """
        Returns the (x1, y1, x2, y2) for the current frame.
        """
        if force_snap:
            self.current_center_x = self.target_center_x
        else:
            diff = self.target_center_x - self.current_center_x
            
            # SIMPLIFIED LOGIC:
            # 1. Is the target outside the safe zone?
            if abs(diff) > self.safe_zone_radius:
                # 2. If yes, move towards it slowly (Linear Speed)
                # Determine direction
                direction = 1 if diff > 0 else -1
                
                # Speed: 2 pixels per frame (Slow pan)
                # If the distance is HUGE (scene change or fast movement), speed up slightly
                if abs(diff) > self.crop_width * 0.5:
                    speed = 15.0 # Fast re-frame
                else:
                    speed = 3.0  # Slow, steady pan
                
                self.current_center_x += direction * speed
                
                # Check if we overshot (prevent oscillation)
                new_diff = self.target_center_x - self.current_center_x
                if (direction == 1 and new_diff < 0) or (direction == -1 and new_diff > 0):
                    self.current_center_x = self.target_center_x
            
            # If inside safe zone, DO NOTHING (Stationary Camera)
                
        # Clamp center
        half_crop = self.crop_width / 2
        
        if self.current_center_x - half_crop < 0:
            self.current_center_x = half_crop
        if self.current_center_x + half_crop > self.video_width:
            self.current_center_x = self.video_width - half_crop
            
        x1 = int(self.current_center_x - half_crop)
        x2 = int(self.current_center_x + half_crop)
        
        x1 = max(0, x1)
        x2 = min(self.video_width, x2)
        
        y1 = 0
        y2 = self.video_height
        
        return x1, y1, x2, y2

class SpeakerTracker:
    """
    Tracks speakers over time to prevent rapid switching and handle temporary obstructions.
    """
    def __init__(self, stabilization_frames=15, cooldown_frames=30):
        self.active_speaker_id = None
        self.speaker_scores = {}  # {id: score}
        self.last_seen = {}       # {id: frame_number}
        self.locked_counter = 0   # How long we've been locked on current speaker
        
        # Hyperparameters
        self.stabilization_threshold = stabilization_frames # Frames needed to confirm a new speaker
        self.switch_cooldown = cooldown_frames              # Minimum frames before switching again
        self.last_switch_frame = -1000
        
        # ID tracking
        self.next_id = 0
        self.known_faces = [] # [{'id': 0, 'center': x, 'last_frame': 123}]

    def get_target(self, face_candidates, frame_number, width):
        """
        Decides which face to focus on.
        face_candidates: list of {'box': [x,y,w,h], 'score': float}
        """
        current_candidates = []
        
        # 1. Match faces to known IDs (simple distance tracking)
        for face in face_candidates:
            x, y, w, h = face['box']
            center_x = x + w / 2
            
            best_match_id = -1
            min_dist = width * 0.15 # Reduced matching radius to avoid jumping in groups
            
            # Try to match with known faces seen recently
            for kf in self.known_faces:
                if frame_number - kf['last_frame'] > 30: # Forgot faces older than 1s (was 2s)
                    continue
                    
                dist = abs(center_x - kf['center'])
                if dist < min_dist:
                    min_dist = dist
                    best_match_id = kf['id']
            
            # If no match, assign new ID
            if best_match_id == -1:
                best_match_id = self.next_id
                self.next_id += 1
            
            # Update known face
            self.known_faces = [kf for kf in self.known_faces if kf['id'] != best_match_id]
            self.known_faces.append({'id': best_match_id, 'center': center_x, 'last_frame': frame_number})
            
            current_candidates.append({
                'id': best_match_id,
                'box': face['box'],
                'score': face['score']
            })

        # 2. Update Scores with decay
        for pid in list(self.speaker_scores.keys()):
             self.speaker_scores[pid] *= 0.85 # Faster decay (was 0.9)
             if self.speaker_scores[pid] < 0.1:
                 del self.speaker_scores[pid]

        # Add new scores
        for cand in current_candidates:
            pid = cand['id']
            # Score is purely based on size (proximity) now that we don't have mouth
            raw_score = cand['score'] / (width * width * 0.05)
            self.speaker_scores[pid] = self.speaker_scores.get(pid, 0) + raw_score

        # 3. Determine Best Speaker
        if not current_candidates:
            # If no one found, maintain last active speaker if cooldown allows
            # to avoid black screen or jump to 0,0
            return None 
            
        best_candidate = None
        max_score = -1
        
        for cand in current_candidates:
            pid = cand['id']
            total_score = self.speaker_scores.get(pid, 0)
            
            # Hysteresis: HUGE Bonus for current active speaker
            if pid == self.active_speaker_id:
                total_score *= 3.0 # Sticky factor
                
            if total_score > max_score:
                max_score = total_score
                best_candidate = cand

        # 4. Decide Switch
        if best_candidate:
            target_id = best_candidate['id']
            
            if target_id == self.active_speaker_id:
                self.locked_counter += 1
                return best_candidate['box']
            
            # New person
            if frame_number - self.last_switch_frame < self.switch_cooldown:
                old_cand = next((c for c in current_candidates if c['id'] == self.active_speaker_id), None)
                if old_cand:
                    return old_cand['box']
            
            self.active_speaker_id = target_id
            self.last_switch_frame = frame_number
            self.locked_counter = 0
            return best_candidate['box']
            
        return None

def detect_face_candidates(frame):
    """
    Returns list of all detected faces using MediaPipe Tasks FaceDetector.
    """
    height, width, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    results = face_detector.detect(mp_image)
    
    candidates = []
    
    if not results.detections:
        return []
        
    for detection in results.detections:
        bbox = detection.bounding_box
        x = bbox.origin_x
        y = bbox.origin_y
        w = bbox.width
        h = bbox.height
        
        candidates.append({
            'box': [x, y, w, h],
            'score': w * h  # Area as score
        })
            
    return candidates

def detect_person_yolo(frame):
    """
    Fallback: Detect largest person using YOLO when face detection fails.
    Returns [x, y, w, h] of the person's 'upper body' approximation.
    """
    # Use the globally loaded model
    results = model(frame, verbose=False, classes=[0]) # class 0 is person
    
    if not results:
        return None
        
    best_box = None
    max_area = 0
    
    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = [int(i) for i in box.xyxy[0]]
            w = x2 - x1
            h = y2 - y1
            area = w * h
            
            if area > max_area:
                max_area = area
                # Focus on the top 40% of the person (head/chest) for framing
                # This approximates where the face is if we can't detect it directly
                face_h = int(h * 0.4)
                best_box = [x1, y1, w, face_h]
                
    return best_box

def create_general_frame(frame, output_width, output_height):
    """
    Creates a 'General Shot' frame: 
    - Background: Blurred zoom of original
    - Foreground: Original video scaled to fit width, centered vertically.
    """
    orig_h, orig_w = frame.shape[:2]
    
    # 1. Background (Fill Height)
    # Crop center to aspect ratio
    bg_scale = output_height / orig_h
    bg_w = int(orig_w * bg_scale)
    bg_resized = cv2.resize(frame, (bg_w, output_height))
    
    # Crop center of background
    start_x = (bg_w - output_width) // 2
    if start_x < 0: start_x = 0
    background = bg_resized[:, start_x:start_x+output_width]
    if background.shape[1] != output_width:
        background = cv2.resize(background, (output_width, output_height))
        
    # Blur background
    background = cv2.GaussianBlur(background, (51, 51), 0)
    
    # 2. Foreground (Fit Width)
    scale = output_width / orig_w
    fg_h = int(orig_h * scale)
    foreground = cv2.resize(frame, (output_width, fg_h))
    
    # 3. Overlay
    y_offset = (output_height - fg_h) // 2
    
    # Clone background to avoid modifying it
    final_frame = background.copy()
    final_frame[y_offset:y_offset+fg_h, :] = foreground
    
    return final_frame


_WATERMARK_FONT_CANDIDATES = [
    "C:\\Windows\\Fonts\\arialbd.ttf",
    "C:\\Windows\\Fonts\\segoeuib.ttf",
    "/mnt/c/Windows/Fonts/arialbd.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",
]


def _find_default_watermark_image():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    candidate = os.path.join(script_dir, "dashboard", "public", "logo-openshorts.png")
    return candidate if os.path.exists(candidate) else None


def _render_watermark_rgba(output_width):
    """Render the watermark as a PIL RGBA image, opacity already baked into
    the alpha channel. WATERMARK_TEXT wins; otherwise the configured/default
    logo PNG; otherwise a plain text fallback."""
    from PIL import Image as PILImage, ImageDraw, ImageFont

    wm_width = max(2, int(output_width * WATERMARK_WIDTH_FRACTION))
    image_path = WATERMARK_IMAGE or _find_default_watermark_image()

    if not WATERMARK_TEXT and image_path:
        img = PILImage.open(image_path).convert("RGBA")
        scale = wm_width / max(img.width, 1)
        img = img.resize((wm_width, max(1, int(img.height * scale))), PILImage.LANCZOS)
    else:
        text = WATERMARK_TEXT or "OpenShorts"
        font = None
        for path in _WATERMARK_FONT_CANDIDATES:
            if os.path.exists(path):
                try:
                    font = ImageFont.truetype(path, 120)
                    break
                except Exception:
                    continue
        if font is None:
            font = ImageFont.load_default()
        probe = PILImage.new("RGBA", (4, 4))
        draw = ImageDraw.Draw(probe)
        bbox = draw.textbbox((0, 0), text, font=font)
        text_w = max(bbox[2] - bbox[0], 1)
        text_h = max(bbox[3] - bbox[1], 1)
        img = PILImage.new("RGBA", (text_w + 8, text_h + 8), (0, 0, 0, 0))
        ImageDraw.Draw(img).text((4 - bbox[0], 4 - bbox[1]), text, font=font, fill=(255, 255, 255, 255))
        scale = wm_width / img.width
        img = img.resize((wm_width, max(1, int(img.height * scale))), PILImage.LANCZOS)

    # Bake the global opacity into the alpha channel so every consumer
    # (frame blender, FFmpeg overlay) renders it equally subtle.
    r, g, b, a = img.split()
    a = a.point(lambda v: int(v * WATERMARK_OPACITY))
    img.putalpha(a)
    return img


def _build_watermark_blender(output_width, output_height):
    """Precompute the centered watermark blend for the OpenCV frame loop.
    Returns (premultiplied_bgr, inverse_alpha, x, y, w, h) or None."""
    if not WATERMARK_ENABLED:
        return None
    try:
        rgba = _render_watermark_rgba(output_width)
    except Exception as e:
        print(f"⚠️ Watermark disabled (render failed): {e}")
        return None
    arr = np.array(rgba, dtype=np.float32)
    h, w = arr.shape[:2]
    x = (output_width - w) // 2
    y = (output_height - h) // 2
    if x < 0 or y < 0 or h < 1 or w < 1:
        return None
    alpha = arr[:, :, 3:4] / 255.0
    bgr = arr[:, :, [2, 1, 0]]
    return (bgr * alpha, 1.0 - alpha, x, y, w, h)


def _apply_watermark(frame, blender):
    premul, inv_alpha, x, y, w, h = blender
    roi = frame[y:y + h, x:x + w].astype(np.float32)
    frame[y:y + h, x:x + w] = (roi * inv_alpha + premul).astype(np.uint8)
    return frame


def analyze_scenes_strategy(video_path, scenes):
    """
    Analyzes each scene to determine if it should be TRACK (Single person) or GENERAL (Group/Wide).
    Returns list of strategies corresponding to scenes.
    """
    cap = cv2.VideoCapture(video_path)
    strategies = []
    
    if not cap.isOpened():
        return ['TRACK'] * len(scenes)
        
    for start, end in tqdm(scenes, desc="   Analyzing Scenes"):
        # Sample 3 frames (start, middle, end)
        frames_to_check = [
            start.get_frames() + 5,
            int((start.get_frames() + end.get_frames()) / 2),
            end.get_frames() - 5
        ]
        
        face_counts = []
        for f_idx in frames_to_check:
            cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
            ret, frame = cap.read()
            if not ret: continue
            
            # Detect faces
            candidates = detect_face_candidates(frame)
            face_counts.append(len(candidates))
            
        # Decision Logic
        if not face_counts:
            avg_faces = 0
        else:
            avg_faces = sum(face_counts) / len(face_counts)
            
        # Strategy:
        # 0 faces -> GENERAL (Landscape/B-roll)
        # 1 face -> TRACK
        # > 1.2 faces -> GENERAL (Group)
        
        if avg_faces > 1.2 or avg_faces < 0.5:
            strategies.append('GENERAL')
        else:
            strategies.append('TRACK')
            
    cap.release()
    return strategies

def detect_scenes(video_path):
    video_manager = VideoManager([video_path])
    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector())
    video_manager.set_downscale_factor()
    video_manager.start()
    scene_manager.detect_scenes(frame_source=video_manager)
    scene_list = scene_manager.get_scene_list()
    fps = video_manager.get_framerate()
    video_manager.release()
    return scene_list, fps

def get_video_resolution(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Could not open video file {video_path}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return width, height


def sanitize_filename(filename):
    """Remove invalid characters from filename."""
    filename = re.sub(r'[<>:"/\\|?*]', '', filename)
    filename = filename.replace(' ', '_')
    return filename[:100]


def download_youtube_video(url, output_dir="."):
    """
    Downloads a YouTube video using yt-dlp.
    Returns the path to the downloaded video and the video title.
    """
    print(f"🔍 Debug: yt-dlp version: {yt_dlp.version.__version__}")
    print("📥 Downloading video from YouTube...")
    step_start_time = time.time()
    progress_state = {"last_emit": 0.0, "last_bucket": -1}

    def download_progress_hook(data):
        if data.get("status") != "downloading":
            return
        total_bytes = data.get("total_bytes") or data.get("total_bytes_estimate") or 0
        downloaded = data.get("downloaded_bytes") or 0
        if not total_bytes:
            JOB_REPORTER.heartbeat(message="Downloading video...", category="download")
            return
        percent = max(0.0, min(100.0, (downloaded / total_bytes) * 100.0))
        now = time.time()
        bucket = int(percent // 5)
        if now - progress_state["last_emit"] < 1.0 and bucket == progress_state["last_bucket"]:
            return
        progress_state["last_emit"] = now
        progress_state["last_bucket"] = bucket
        JOB_REPORTER.progress(
            percent,
            message=f"Downloading video... {percent:.1f}%",
            important=bucket % 2 == 0,
            eta_seconds=data.get("eta"),
            category="download",
        )

    # Look for cookies in the project directory (multiple common filenames)
    _script_dir = os.path.dirname(os.path.abspath(__file__))
    _cookie_candidates = [
        os.path.join(_script_dir, 'www.youtube.com_cookies.txt'),
        os.path.join(_script_dir, 'cookies.txt'),
        '/app/cookies.txt',  # Docker fallback
    ]
    cookies_path = None
    for _cp in _cookie_candidates:
        if os.path.exists(_cp):
            cookies_path = _cp
            print(f"🍪 Found cookies file: {cookies_path}")
            break

    cookies_env = os.environ.get("YOUTUBE_COOKIES")
    if cookies_env:
        print("🍪 Found YOUTUBE_COOKIES env var, writing cookies file...")
        if not cookies_path:
            cookies_path = os.path.join(_script_dir, 'cookies.txt')
        try:
            with open(cookies_path, 'w') as f:
                f.write(cookies_env)
            if os.path.exists(cookies_path):
                 print(f"   Debug: Cookies file created. Size: {os.path.getsize(cookies_path)} bytes")
                 with open(cookies_path, 'r') as f:
                     content = f.read(100)
                     print(f"   Debug: First 100 chars of cookie file: {content}")
        except Exception as e:
            print(f"⚠️ Failed to write cookies file: {e}")
            cookies_path = None
    else:
        if not cookies_path:
            print("⚠️ No cookies file found and YOUTUBE_COOKIES env var not set.")
    
    js_runtimes = None
    if shutil.which('deno'):
        js_runtimes = {'deno': {}}
        print("🧠 Using Deno for yt-dlp JS challenges")
    elif shutil.which('node'):
        js_runtimes = {'node': {}}
        print("🧠 Using Node.js for yt-dlp JS challenges")
    else:
        print("⚠️ No supported JS runtime found for yt-dlp (Deno/Node). Download quality may be limited.")

    # yt-dlp options: adapt based on whether cookies are available
    # WITH cookies: use 'web' client, DO NOT skip webpage (needed for cookie auth)
    # WITHOUT cookies: use more reliable non-auth clients and skip webpage

    # Do NOT override player_client: yt-dlp's curated defaults (tv_downgraded/
    # web_safari with cookies, android_vr/web_safari anonymous) are exactly the
    # clients that still serve HD without PO tokens. Our old hardcoded list
    # (web_safari/ios/web/mweb/android) hit PO-token/SABR walls and silently
    # degraded downloads to the token-free 360p format.
    _extractor_args = {}
    if cookies_path:
        print("🔧 Using cookie-authenticated yt-dlp config (default clients)")
    else:
        print("🔧 Using anonymous yt-dlp config (default clients)")

    _COMMON_YDL_OPTS = {
        'quiet': False,
        'verbose': True,
        'no_warnings': False,
        'cookiefile': cookies_path if cookies_path else None,
        'socket_timeout': 30,
        'retries': 10,
        'fragment_retries': 10,
        'nocheckcertificate': True,
        'cachedir': False,
        'extractor_args': _extractor_args,
        'http_headers': {
            'User-Agent': (
                'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                'AppleWebKit/537.36 (KHTML, like Gecko) '
                'Chrome/120.0.0.0 Safari/537.36'
            ),
        },
        'progress_hooks': [download_progress_hook],
    }
    if js_runtimes:
        _COMMON_YDL_OPTS['js_runtimes'] = js_runtimes

    # --- Smart retry: probe both configs and keep the one offering the best
    # --- quality. YouTube silently limits some clients/cookie states to the
    # --- muxed 360p format; without this check we'd process mush without warning.
    probe_results = []  # (attempt_name, attempt_opts, info, max_height)
    last_error = None
    for attempt_name, attempt_opts in [
        ("cookie-auth", _COMMON_YDL_OPTS),
        ("anonymous-fallback", {
            **_COMMON_YDL_OPTS,
            'cookiefile': None,
            # Default clients here too (android_vr/web_safari) — see note above.
            'extractor_args': {},
        }),
    ]:
        try:
            print(f"🔄 Trying download mode: {attempt_name}...")
            with yt_dlp.YoutubeDL(attempt_opts) as ydl:
                probe = ydl.extract_info(url, download=False)
            # Check if we actually got video formats (not just images/storyboards)
            formats = probe.get('formats') or []
            video_formats = [
                f for f in formats
                if f.get('vcodec', 'none') != 'none'
                and f.get('protocol') != 'mhtml'
                and f.get('ext') != 'mhtml'
            ]
            if not video_formats:
                print(f"⚠️ {attempt_name}: No video formats found, trying next mode...")
                continue
            max_height = max((f.get('height') or 0) for f in video_formats)
            print(f"✅ {attempt_name}: Found {len(video_formats)} video formats (best: {max_height}p)")
            probe_results.append((attempt_name, attempt_opts, probe, max_height))
            if max_height >= 1080:
                break
            print(f"⚠️ {attempt_name}: best offered is only {max_height}p — probing next mode for higher quality...")
        except Exception as e:
            last_error = e
            print(f"⚠️ {attempt_name} failed: {e}")

    if not probe_results:
        if last_error is not None:
            print("🚨 YOUTUBE DOWNLOAD ERROR 🚨", file=sys.stderr)

            error_msg = f"""

❌ ================================================================= ❌
❌ FATAL ERROR: YOUTUBE DOWNLOAD FAILED
❌ ================================================================= ❌

REASON: YouTube has blocked the download request (Error 429/Unavailable).
        This is likely a temporary IP ban on this server.

👇 SOLUTION FOR USER 👇
---------------------------------------------------------------------
1. Download the video manually to your computer.
2. Use the 'Upload Video' tab in this app to process it.
---------------------------------------------------------------------

Technical Details: {str(last_error)}
            """
            print(error_msg, file=sys.stdout)
            print(error_msg, file=sys.stderr)
            sys.stdout.flush()
            sys.stderr.flush()
            time.sleep(0.5)
            raise last_error
        print("❌ All download modes failed. No video formats available.")
        raise SystemExit(1)

    attempt_name, attempt_opts, info, best_height = max(probe_results, key=lambda item: item[3])
    _COMMON_YDL_OPTS.update(attempt_opts)
    print(f"🎯 Downloading via {attempt_name} (best available: {best_height}p)")
    if 0 < best_height < 720:
        JOB_REPORTER.warning(
            f"YouTube only offered {best_height}p for this video — clips will look soft. "
            "Fix: re-export cookies from a FRESH incognito session (log in, open "
            "youtube.com/robots.txt, export, close the window and never reuse it) "
            "and update yt-dlp (pip install -U yt-dlp), then re-process.",
            category="download",
        )

    video_title = info.get('title', 'youtube_video')
    sanitized_title = sanitize_filename(video_title)
    
    output_template = os.path.join(output_dir, f'{sanitized_title}.%(ext)s')
    expected_file = os.path.join(output_dir, f'{sanitized_title}.mp4')
    if os.path.exists(expected_file):
        os.remove(expected_file)
        print("🗑️  Removed existing file to re-download video")
    
    ydl_opts = {
        **_COMMON_YDL_OPTS,
        'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/bestvideo[ext=mp4]+bestaudio/bestvideo+bestaudio/best[ext=mp4]/best',
        'outtmpl': output_template,
        'merge_output_format': 'mp4',
        'overwrites': True,
    }
    
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])
    
    downloaded_file = os.path.join(output_dir, f'{sanitized_title}.mp4')
    
    if not os.path.exists(downloaded_file):
        for f in os.listdir(output_dir):
            if f.startswith(sanitized_title) and not f.endswith(('.part', '.ytdl', '.json', '.description')):
                downloaded_file = os.path.join(output_dir, f)
                break
    
    step_end_time = time.time()
    JOB_REPORTER.progress(100.0, message="Download complete.", important=True, category="download")
    print(f"✅ Video downloaded in {step_end_time - step_start_time:.2f}s: {downloaded_file}")
    
    return downloaded_file, sanitized_title

def _finalize_clip_passthrough(input_video, final_output_video, progress_callback=None):
    """Finish a clip without reframing (horizontal output, or source already in
    the target aspect). Applies the watermark via FFmpeg overlay if enabled;
    otherwise remuxes with faststart only."""
    if os.path.exists(final_output_video):
        os.remove(final_output_video)
    if progress_callback:
        progress_callback(5.0, "Finalizing clip without reframing...")

    wm_png = None
    if WATERMARK_ENABLED:
        try:
            width, _height = get_video_resolution(input_video)
            wm_png = f"{os.path.splitext(final_output_video)[0]}_wm.png"
            _render_watermark_rgba(width).save(wm_png)
        except Exception as e:
            print(f"⚠️ Watermark skipped (render failed): {e}")
            wm_png = None

    if wm_png:
        command = [
            'ffmpeg', '-y', '-i', input_video, '-i', wm_png,
            '-filter_complex', '[0:v][1:v]overlay=(W-w)/2:(H-h)/2:format=auto',
            '-c:v', 'libx264', '-preset', 'fast', '-crf', '20',
            '-c:a', 'copy', '-movflags', '+faststart', final_output_video,
        ]
    else:
        command = [
            'ffmpeg', '-y', '-i', input_video,
            '-c', 'copy', '-movflags', '+faststart', final_output_video,
        ]

    try:
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, timeout=1800)
    except subprocess.TimeoutExpired:
        print("\n   ❌ Passthrough finalize timed out after 1800s.")
        return False
    except subprocess.CalledProcessError as e:
        print("\n   ❌ Passthrough finalize failed.")
        print("   Stderr:", (e.stderr or b"").decode(errors="replace"))
        return False
    finally:
        if wm_png and os.path.exists(wm_png):
            os.remove(wm_png)

    print(f"   ✅ Clip saved to {final_output_video}")
    if progress_callback:
        progress_callback(100.0, f"Saved {os.path.basename(final_output_video)}")
    return True


def _render_clip(input_video, final_output_video, output_format="auto", progress_callback=None):
    """Route a cut clip through the right renderer for the chosen output format.
    'auto' behaves like 'vertical'; the vertical renderer itself detects sources
    that already match the target aspect and skips reframing for them."""
    if output_format == "horizontal":
        return _finalize_clip_passthrough(input_video, final_output_video, progress_callback)
    aspect = 1.0 if output_format == "square" else ASPECT_RATIO
    return process_video_to_vertical(input_video, final_output_video, progress_callback, aspect_ratio=aspect)


def process_video_to_vertical(input_video, final_output_video, progress_callback=None, aspect_ratio=ASPECT_RATIO):
    """
    Core logic to convert horizontal video to vertical using scene detection and Active Speaker Tracking (MediaPipe).
    """
    script_start_time = time.time()

    # Define temporary file paths based on the output name
    base_name = os.path.splitext(final_output_video)[0]
    temp_video_output = f"{base_name}_temp_video.mp4"
    temp_audio_output = f"{base_name}_temp_audio.aac"

    # Clean up previous temp files if they exist
    if os.path.exists(temp_video_output): os.remove(temp_video_output)
    if os.path.exists(temp_audio_output): os.remove(temp_audio_output)
    if os.path.exists(final_output_video): os.remove(final_output_video)

    # Smart source detection: if the source is already in the target aspect
    # (e.g. phone footage for 9:16), reframing would only crop it — pass it
    # through untouched instead.
    src_width, src_height = get_video_resolution(input_video)
    if src_width and src_height and abs((src_width / src_height) - aspect_ratio) < 0.02:
        print(f"🎬 Source already matches target aspect ({src_width}x{src_height}) — skipping reframing.")
        return _finalize_clip_passthrough(input_video, final_output_video, progress_callback)

    print(f"🎬 Processing clip: {input_video}")
    print("   Step 1: Detecting scenes...")
    if progress_callback:
        progress_callback(2.0, "Detecting scenes...")
    scenes, fps = detect_scenes(input_video)
    
    if not scenes:
        print("   ❌ No scenes were detected. Using full video as one scene.")
        # If scene detection fails or finds nothing, treat whole video as one scene
        cap = cv2.VideoCapture(input_video)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        from scenedetect import FrameTimecode
        scenes = [(FrameTimecode(0, fps), FrameTimecode(total_frames, fps))]

    print(f"   ✅ Found {len(scenes)} scenes.")

    print("\n   🧠 Step 2: Preparing Active Tracking...")
    if progress_callback:
        progress_callback(8.0, "Preparing active tracking...")
    original_width, original_height = src_width, src_height

    OUTPUT_HEIGHT = original_height
    OUTPUT_WIDTH = int(OUTPUT_HEIGHT * aspect_ratio)
    if OUTPUT_WIDTH > original_width:
        # Never upscale beyond the source width (e.g. square target from a
        # narrow portrait source) — shrink the output instead.
        OUTPUT_WIDTH = original_width
        OUTPUT_HEIGHT = int(OUTPUT_WIDTH / aspect_ratio)
    if OUTPUT_WIDTH % 2 != 0:
        OUTPUT_WIDTH += 1
    if OUTPUT_HEIGHT % 2 != 0:
        OUTPUT_HEIGHT += 1

    # Initialize Cameraman
    cameraman = SmoothedCameraman(OUTPUT_WIDTH, OUTPUT_HEIGHT, original_width, original_height, aspect_ratio=aspect_ratio)
    watermark = _build_watermark_blender(OUTPUT_WIDTH, OUTPUT_HEIGHT)
    
    # --- New Strategy: Per-Scene Analysis ---
    print("\n   🤖 Step 3: Analyzing Scenes for Strategy (Single vs Group)...")
    if progress_callback:
        progress_callback(14.0, "Analyzing scene strategy...")
    scene_strategies = analyze_scenes_strategy(input_video, scenes)
    # scene_strategies is a list of 'TRACK' or 'General' corresponding to scenes
    
    print("\n   ✂️ Step 4: Processing video frames...")
    if progress_callback:
        progress_callback(20.0, "Processing video frames...")
    
    command = [
        'ffmpeg', '-y', '-f', 'rawvideo', '-vcodec', 'rawvideo',
        '-s', f'{OUTPUT_WIDTH}x{OUTPUT_HEIGHT}', '-pix_fmt', 'bgr24',
        '-r', str(fps), '-i', '-', '-c:v', 'libx264',
        '-preset', 'fast', '-crf', '23', '-an', temp_video_output
    ]

    ffmpeg_process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)

    cap = cv2.VideoCapture(input_video)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    frame_number = 0
    current_scene_index = 0
    
    # Pre-calculate scene boundaries
    scene_boundaries = []
    for s_start, s_end in scenes:
        scene_boundaries.append((s_start.get_frames(), s_end.get_frames()))

    # Global tracker for single-person shots
    speaker_tracker = SpeakerTracker(cooldown_frames=30)

    last_progress_emit = time.time()
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Update Scene Index
            if current_scene_index < len(scene_boundaries):
                start_f, end_f = scene_boundaries[current_scene_index]
                if frame_number >= end_f and current_scene_index < len(scene_boundaries) - 1:
                    current_scene_index += 1

            # Determine Strategy for current frame based on scene
            current_strategy = scene_strategies[current_scene_index] if current_scene_index < len(scene_strategies) else 'TRACK'

            # Apply Strategy
            if current_strategy == 'GENERAL':
                # "Plano General" -> Blur Background + Fit Width
                output_frame = create_general_frame(frame, OUTPUT_WIDTH, OUTPUT_HEIGHT)

                # Reset cameraman/tracker so they don't drift while inactive
                cameraman.current_center_x = original_width / 2
                cameraman.target_center_x = original_width / 2

            else:
                # "Single Speaker" -> Track & Crop
                if frame_number % 2 == 0:
                    candidates = detect_face_candidates(frame)
                    target_box = speaker_tracker.get_target(candidates, frame_number, original_width)
                    if target_box:
                        cameraman.update_target(target_box)
                    else:
                        person_box = detect_person_yolo(frame)
                        if person_box:
                            cameraman.update_target(person_box)

                is_scene_start = (frame_number == scene_boundaries[current_scene_index][0])
                x1, y1, x2, y2 = cameraman.get_crop_box(force_snap=is_scene_start)
                if y2 > y1 and x2 > x1:
                    cropped = frame[y1:y2, x1:x2]
                    output_frame = cv2.resize(cropped, (OUTPUT_WIDTH, OUTPUT_HEIGHT))
                else:
                    output_frame = cv2.resize(frame, (OUTPUT_WIDTH, OUTPUT_HEIGHT))

            if watermark is not None:
                output_frame = _apply_watermark(output_frame, watermark)

            ffmpeg_process.stdin.write(output_frame.tobytes())
            frame_number += 1
            if progress_callback and (time.time() - last_progress_emit >= 2.0 or frame_number == total_frames):
                frame_progress = 20.0 + (70.0 * (frame_number / max(total_frames, 1)))
                progress_callback(frame_progress, f"Processing video frames... {frame_number}/{total_frames}")
                last_progress_emit = time.time()

        ffmpeg_process.stdin.close()
        stderr_output = ffmpeg_process.stderr.read().decode()
        ffmpeg_process.wait()
    finally:
        cap.release()
        # If the frame loop aborted (exception/cancel), don't leak the encoder subprocess.
        if ffmpeg_process.poll() is None:
            ffmpeg_process.terminate()
            try:
                ffmpeg_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                ffmpeg_process.kill()

    if ffmpeg_process.returncode != 0:
        print("\n   ❌ FFmpeg frame processing failed.")
        print("   Stderr:", stderr_output)
        return False

    print("\n   🔊 Step 5: Extracting audio...")
    if progress_callback:
        progress_callback(92.0, "Extracting audio...")
    audio_extract_command = [
        'ffmpeg', '-y', '-i', input_video, '-vn', '-acodec', 'copy', temp_audio_output
    ]
    try:
        subprocess.run(audio_extract_command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, timeout=1800)
    except subprocess.TimeoutExpired:
        print("\n   ❌ Audio extraction timed out after 1800s. Proceeding without audio.")
    except subprocess.CalledProcessError:
        print("\n   ❌ Audio extraction failed (maybe no audio?). Proceeding without audio.")
        pass

    print("\n   ✨ Step 6: Merging...")
    if progress_callback:
        progress_callback(97.0, "Merging output...")
    if os.path.exists(temp_audio_output):
        merge_command = [
            'ffmpeg', '-y', '-i', temp_video_output, '-i', temp_audio_output,
            '-c:v', 'copy', '-c:a', 'copy', '-movflags', '+faststart', final_output_video
        ]
    else:
         merge_command = [
            'ffmpeg', '-y', '-i', temp_video_output,
            '-c:v', 'copy', '-movflags', '+faststart', final_output_video
        ]
        
    try:
        subprocess.run(merge_command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, timeout=1800)
        print(f"   ✅ Clip saved to {final_output_video}")
        if progress_callback:
            progress_callback(100.0, f"Saved {os.path.basename(final_output_video)}")
    except subprocess.TimeoutExpired:
        print("\n   ❌ Final merge timed out after 1800s.")
        return False
    except subprocess.CalledProcessError as e:
        print("\n   ❌ Final merge failed.")
        print("   Stderr:", e.stderr.decode())
        return False

    # Clean up temp files
    if os.path.exists(temp_video_output): os.remove(temp_video_output)
    if os.path.exists(temp_audio_output): os.remove(temp_audio_output)
    
    return True

def transcribe_video(video_path, video_duration=None):
    print("🎙️  Transcribing video with Faster-Whisper (CPU Optimized)...")
    from faster_whisper import WhisperModel
    from subtitles import get_whisper_config, WHISPER_TRANSCRIBE_PARAMS, merge_continuation_words

    cfg = get_whisper_config()
    model = WhisperModel(cfg["model_size"], device=cfg["device"], compute_type=cfg["compute_type"])

    segments, info = model.transcribe(video_path, **WHISPER_TRANSCRIBE_PARAMS)
    
    print(f"   Detected language '{info.language}' with probability {info.language_probability:.2f}")
    
    # Convert to openai-whisper compatible format
    transcript_segments = []
    full_text = ""
    last_emit = time.time()
    last_bucket = -1

    for segment in segments:
        seg_dict = {
            'text': segment.text,
            'start': segment.start,
            'end': segment.end,
            'words': []
        }
        
        if segment.words:
            # Merge continuation fragments (tokens without a leading space belong
            # to the previous word) so compound words stay intact downstream.
            raw_words = [
                {
                    'word': word.word,
                    'start': word.start,
                    'end': word.end,
                    'probability': word.probability
                }
                for word in segment.words
            ]
            seg_dict['words'] = merge_continuation_words(raw_words)
        
        transcript_segments.append(seg_dict)
        full_text += segment.text + " "
        if video_duration:
            progress_percent = min(100.0, (float(segment.end) / max(video_duration, 1.0)) * 100.0)
            bucket = int(progress_percent // 2)
            if time.time() - last_emit >= 1.0 or bucket != last_bucket:
                JOB_REPORTER.progress(
                    progress_percent,
                    message=f"Transcribing audio... {progress_percent:.1f}%",
                    important=bucket % 5 == 0,
                    category="transcribe",
                )
                last_emit = time.time()
                last_bucket = bucket
        
    JOB_REPORTER.progress(100.0, message="Transcription complete.", important=True, category="transcribe")
    return {
        'text': full_text.strip(),
        'segments': transcript_segments,
        'language': info.language
    }

def get_viral_clips(transcript_result, video_duration, output_dir=None, video_title=None):
    print("🤖  Analyzing with Gemini...")

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        error_message = "GEMINI_API_KEY not found in environment variables."
        print(f"❌ Error: {error_message}")
        JOB_REPORTER.error(error_message)
        return {
            "clips_data": None,
            "error": error_message,
            "attempts": [],
            "cost_analysis": None,
        }

    if not os.path.exists(GEMINI_WORKER_SCRIPT):
        error_message = f"Gemini worker script not found: {GEMINI_WORKER_SCRIPT}"
        print(f"❌ Error: {error_message}")
        JOB_REPORTER.error(error_message)
        return {
            "clips_data": None,
            "error": error_message,
            "attempts": [],
            "cost_analysis": None,
        }

    print(f"🤖  Initializing Gemini with model: {GEMINI_ANALYSIS_MODEL}")

    words = _extract_words_for_analysis(transcript_result)
    transcript_language = str(transcript_result.get("language") or "unknown")
    windows = _build_transcript_windows(transcript_result, video_duration)
    if output_dir and video_title:
        _save_json_checkpoint(output_dir, video_title, "analysis_windows", {"windows": windows})

    attempt_specs = [
        {"name": "structured-schema", "strategy": "structured-schema"},
        {"name": "strict-json", "strategy": "strict-json"},
        {"name": "json-text-recovery", "strategy": "json-text-recovery"},
    ]

    attempts = []
    all_costs = []
    scored_windows = []
    total_score_batches = max(1, math.ceil(len(windows) / GEMINI_SCORE_BATCH_SIZE))

    for batch_index, batch_windows in _iter_batches(windows, GEMINI_SCORE_BATCH_SIZE):
        JOB_REPORTER.progress(
            (batch_index / max(total_score_batches, 1)) * 45.0,
            message=f"Scoring transcript windows... batch {batch_index + 1}/{total_score_batches}",
            important=True,
            category="analyze",
        )
        batch_payload = {
            "video_duration": round(float(video_duration), 3),
            "language": transcript_language,
            "windows": batch_windows,
        }
        batch_result = None
        last_error = None
        for attempt_number, attempt in enumerate(attempt_specs[:GEMINI_MAX_ATTEMPTS], start=1):
            print(f"🤖  Gemini scoring attempt {attempt_number}/{GEMINI_MAX_ATTEMPTS}: batch {batch_index + 1}/{total_score_batches} ({attempt['name']})")
            try:
                worker_result = _call_gemini_worker(
                    "score",
                    batch_payload,
                    output_dir=output_dir or ".",
                    video_title=video_title or "analysis",
                    strategy=attempt["strategy"],
                    batch_index=batch_index,
                    total_batches=total_score_batches,
                    attempt=attempt_number,
                )
                normalized_scores = _normalize_scored_windows(worker_result.get("payload", {}), video_duration)
                batch_result = normalized_scores
                cost_analysis = worker_result.get("cost_analysis")
                if cost_analysis:
                    all_costs.append(cost_analysis)
                attempts.append({
                    "stage": "score",
                    "batch": batch_index + 1,
                    "attempt": attempt_number,
                    "name": attempt["name"],
                    "status": "success",
                })
                break
            except Exception as e:
                last_error = str(e)
                JOB_REPORTER.warning(
                    f"Gemini scoring attempt {attempt_number} failed for batch {batch_index + 1}/{total_score_batches}: {last_error}",
                    category="gemini",
                    attempt=attempt_number,
                )
                attempts.append({
                    "stage": "score",
                    "batch": batch_index + 1,
                    "attempt": attempt_number,
                    "name": attempt["name"],
                    "status": "failed",
                    "error": last_error,
                })
        if batch_result:
            scored_windows.extend(batch_result)
        elif last_error:
            JOB_REPORTER.warning(
                f"Skipping score batch {batch_index + 1}/{total_score_batches} after repeated Gemini failures.",
                category="gemini",
            )

    if not scored_windows:
        error_message = "Gemini could not score any transcript windows."
        print(f"❌ Gemini Error: {error_message}")
        JOB_REPORTER.error(error_message)
        return {
            "clips_data": None,
            "error": error_message,
            "attempts": attempts,
            "cost_analysis": _merge_cost_analyses(all_costs),
        }

    by_id = {}
    for window in sorted(scored_windows, key=lambda item: item.get("score", 0), reverse=True):
        if window["id"] not in by_id:
            by_id[window["id"]] = window
    shortlisted = list(by_id.values())[:GEMINI_SHORTLIST_LIMIT]
    shortlist_lookup = {window["id"]: window for window in windows}
    if output_dir and video_title:
        _save_json_checkpoint(output_dir, video_title, "analysis_shortlist", {"windows": shortlisted})

    detailed_windows = []
    for item in shortlisted:
        full_window = shortlist_lookup.get(item["id"], item)
        detail_start = max(0.0, float(full_window["start"]) - 20.0)
        detail_end = min(float(video_duration), float(full_window["end"]) + 20.0)
        detail_words = _extract_words_for_range(words, detail_start, detail_end)
        detailed_windows.append({
            "id": full_window["id"],
            "start": detail_start,
            "end": detail_end,
            "candidate_score": item.get("score", 0),
            "text": _extract_text_for_range(transcript_result, detail_start, detail_end),
            "words": detail_words,
        })

    collected_clips = []
    total_detail_batches = max(1, math.ceil(len(detailed_windows) / GEMINI_DETAIL_BATCH_SIZE))
    for batch_index, batch_windows in _iter_batches(detailed_windows, GEMINI_DETAIL_BATCH_SIZE):
        batch_progress = 45.0 + ((batch_index / max(total_detail_batches, 1)) * 45.0)
        JOB_REPORTER.progress(
            batch_progress,
            message=f"Detail-analyzing shortlisted windows... batch {batch_index + 1}/{total_detail_batches}",
            important=True,
            category="analyze",
        )
        batch_payload = {
            "video_duration": round(float(video_duration), 3),
            "language": transcript_language,
            "windows": batch_windows,
        }
        batch_result = None
        last_error = None
        for attempt_number, attempt in enumerate(attempt_specs[:GEMINI_MAX_ATTEMPTS], start=1):
            print(f"🤖  Gemini detail attempt {attempt_number}/{GEMINI_MAX_ATTEMPTS}: batch {batch_index + 1}/{total_detail_batches} ({attempt['name']})")
            try:
                worker_result = _call_gemini_worker(
                    "detail",
                    batch_payload,
                    output_dir=output_dir or ".",
                    video_title=video_title or "analysis",
                    strategy=attempt["strategy"],
                    batch_index=batch_index,
                    total_batches=total_detail_batches,
                    attempt=attempt_number,
                )
                batch_result = worker_result.get("payload", {})
                cost_analysis = worker_result.get("cost_analysis")
                if cost_analysis:
                    all_costs.append(cost_analysis)
                attempts.append({
                    "stage": "detail",
                    "batch": batch_index + 1,
                    "attempt": attempt_number,
                    "name": attempt["name"],
                    "status": "success",
                })
                break
            except Exception as e:
                last_error = str(e)
                JOB_REPORTER.warning(
                    f"Gemini detail attempt {attempt_number} failed for batch {batch_index + 1}/{total_detail_batches}: {last_error}",
                    category="gemini",
                    attempt=attempt_number,
                )
                attempts.append({
                    "stage": "detail",
                    "batch": batch_index + 1,
                    "attempt": attempt_number,
                    "name": attempt["name"],
                    "status": "failed",
                    "error": last_error,
                })
        if batch_result and isinstance(batch_result.get("shorts"), list):
            collected_clips.extend(batch_result["shorts"])
        elif last_error:
            JOB_REPORTER.warning(
                f"Skipping detail batch {batch_index + 1}/{total_detail_batches} after repeated Gemini failures.",
                category="gemini",
            )

    if not collected_clips:
        error_message = "Gemini did not produce any valid clips from the shortlisted windows."
        print(f"❌ Gemini Error: {error_message}")
        JOB_REPORTER.error(error_message)
        return {
            "clips_data": None,
            "error": error_message,
            "attempts": attempts,
            "cost_analysis": _merge_cost_analyses(all_costs),
        }

    normalized_payload = _normalize_shorts_payload({"shorts": collected_clips}, video_duration, words=words)
    cost_analysis = _merge_cost_analyses(all_costs)
    if cost_analysis:
        normalized_payload["cost_analysis"] = cost_analysis
    JOB_REPORTER.progress(95.0, message=f"Gemini analysis complete. Found {len(normalized_payload['shorts'])} candidate clips.", important=True, category="analyze")
    return {
        "clips_data": normalized_payload,
        "error": None,
        "attempts": attempts,
        "cost_analysis": cost_analysis,
    }

def _ensure_dir(path: str) -> str:
    if path:
        os.makedirs(path, exist_ok=True)
    return path


def _get_video_duration(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    if fps <= 0:
        return 0.0
    return frame_count / fps


def _find_source_video(resume_dir: str):
    """Locate the downloaded/uploaded source video in a job directory."""
    skip_prefixes = ("temp_", "subtitled_", "hooked_", "edited_", "translated_")
    candidates = []
    for path in sorted(glob.glob(os.path.join(resume_dir, "*.mp4"))):
        name = os.path.basename(path)
        if name.startswith(skip_prefixes):
            continue
        if re.search(r"_clip_\d+\.mp4$", name) or name.endswith("_vertical.mp4"):
            continue
        candidates.append(path)
    # If several remain, the source is by far the largest one.
    return max(candidates, key=os.path.getsize) if candidates else None


def _load_render_config(output_dir: str) -> dict:
    """Per-job render settings (e.g. output format), persisted so resumes keep
    rendering the same way the job was started."""
    path = os.path.join(output_dir, "render_config.json")
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                return data
        except Exception:
            pass
    return {}


def _load_resume_context(resume_dir: str):
    analysis_input_files = sorted(glob.glob(os.path.join(resume_dir, "*_analysis_input.json")))
    if not analysis_input_files:
        # The job died before the transcript/analysis checkpoints were written
        # (e.g. frozen mid-transcription). If the source video survived,
        # resume from the transcription phase instead of failing — the
        # pipeline handles transcript=None by transcribing again.
        source_video = _find_source_video(resume_dir)
        if not source_video:
            raise FileNotFoundError(f"No analysis input file found in {resume_dir}")

        video_title = os.path.splitext(os.path.basename(source_video))[0]
        source_url = None
        state_file = os.path.join(resume_dir, "job_state.json")
        if os.path.exists(state_file):
            try:
                with open(state_file, "r", encoding="utf-8") as f:
                    source_url = json.load(f).get("source_url")
            except Exception:
                pass

        print(f"🔁 No checkpoints found — resuming from source video: {os.path.basename(source_video)}")
        return {
            "output_dir": resume_dir,
            "video_title": video_title,
            "analysis_input_file": os.path.join(resume_dir, f"{video_title}_analysis_input.json"),
            "analysis_result_file": os.path.join(resume_dir, f"{video_title}_analysis_result.json"),
            "metadata_file": os.path.join(resume_dir, f"{video_title}_metadata.json"),
            "transcript_file": os.path.join(resume_dir, f"{video_title}_transcript.json"),
            "words_file": os.path.join(resume_dir, f"{video_title}_words.json"),
            "input_video": source_video,
            "source_url": source_url,
            "duration": 0.0,
            "transcript": None,
            "analysis_result": None,
            "metadata": None,
        }

    analysis_input_file = analysis_input_files[0]
    video_title = os.path.basename(analysis_input_file).replace("_analysis_input.json", "")
    with open(analysis_input_file, "r", encoding="utf-8") as f:
        analysis_input_payload = json.load(f)

    analysis_result_file = os.path.join(resume_dir, f"{video_title}_analysis_result.json")
    metadata_file = os.path.join(resume_dir, f"{video_title}_metadata.json")
    transcript_file = os.path.join(resume_dir, f"{video_title}_transcript.json")
    words_file = os.path.join(resume_dir, f"{video_title}_words.json")

    transcript = analysis_input_payload.get("transcript")
    if not transcript and os.path.exists(transcript_file):
        with open(transcript_file, "r", encoding="utf-8") as f:
            transcript = json.load(f)

    analysis_result = None
    if os.path.exists(analysis_result_file):
        with open(analysis_result_file, "r", encoding="utf-8") as f:
            analysis_result = json.load(f)

    metadata = None
    if os.path.exists(metadata_file):
        with open(metadata_file, "r", encoding="utf-8") as f:
            metadata = json.load(f)

    return {
        "output_dir": resume_dir,
        "video_title": video_title,
        "analysis_input_file": analysis_input_file,
        "analysis_result_file": analysis_result_file,
        "metadata_file": metadata_file,
        "transcript_file": transcript_file,
        "words_file": words_file,
        "input_video": analysis_input_payload.get("input_video"),
        "source_url": analysis_input_payload.get("source_url"),
        "duration": float(analysis_input_payload.get("video_duration") or 0.0),
        "transcript": transcript,
        "analysis_result": analysis_result,
        "metadata": metadata,
    }


def _analysis_result_has_valid_clips(analysis_result) -> bool:
    if not isinstance(analysis_result, dict):
        return False
    clips_data = analysis_result.get("clips_data")
    if not isinstance(clips_data, dict):
        return False
    shorts = clips_data.get("shorts")
    return isinstance(shorts, list) and len(shorts) > 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="AutoCrop-Vertical with Viral Clip Detection.")
    input_group = parser.add_mutually_exclusive_group(required=False)
    input_group.add_argument('-i', '--input', type=str, help="Path to the input video file.")
    input_group.add_argument('-u', '--url', type=str, help="YouTube URL to download and process.")
    parser.add_argument('-o', '--output', type=str, help="Output directory or file (if processing whole video).")
    parser.add_argument('--keep-original', action='store_true', help="Keep the downloaded YouTube video.")
    parser.add_argument('--skip-analysis', action='store_true', help="Skip AI analysis and convert the whole video.")
    parser.add_argument('--resume-dir', type=str, help="Resume a previous job from its output directory.")
    parser.add_argument('--resume-phase', choices=['transcribe', 'analyze', 'render'], help="Force resume from a specific phase.")
    parser.add_argument('--format', dest='output_format', choices=list(OUTPUT_FORMATS), default='auto',
                        help="Output format: auto (smart), vertical (9:16), horizontal (original), square (1:1).")
    parser.add_argument('--job-id', type=str, help="Optional job id for structured worker events.")
    args = parser.parse_args()

    if not args.resume_dir and not args.input and not args.url:
        parser.error("You must provide --input, --url, or --resume-dir.")

    script_start_time = time.time()
    reporter = JobReporter(job_id=args.job_id)
    set_job_reporter(reporter)
    _start_keepalive()
    _prevent_windows_sleep()

    try:
        transcript = None
        analysis_result = None
        clips_data = None
        duration = 0.0
        source_url = args.url

        if args.resume_dir:
            output_dir = _ensure_dir(args.resume_dir)
            resume_context = _load_resume_context(output_dir)
            output_format = _load_render_config(output_dir).get("output_format") or args.output_format
            if output_format not in OUTPUT_FORMATS:
                output_format = "auto"
            input_video = resume_context["input_video"]
            video_title = resume_context["video_title"]
            source_url = resume_context["source_url"]
            duration = resume_context["duration"]
            transcript = resume_context["transcript"]
            analysis_result = resume_context["analysis_result"]
            force_analyze = args.resume_phase == "analyze"
            if force_analyze or not _analysis_result_has_valid_clips(analysis_result):
                analysis_result = None
            reporter.emit("resume", "Resuming previous job from saved checkpoints.", important=True, resumable=True)
        else:
            output_format = args.output_format
            if args.url:
                if args.output and not args.skip_analysis:
                    output_dir = _ensure_dir(args.output)
                else:
                    if args.output and os.path.isdir(args.output):
                        output_dir = args.output
                    elif args.output and not os.path.isdir(args.output):
                        output_dir = os.path.dirname(args.output) or "."
                    else:
                        output_dir = "."
                # Persist the format before the download so a crash-resume keeps it.
                _save_json_file(os.path.join(output_dir, "render_config.json"), {"output_format": output_format})
                reporter.set_phase("download", "Downloading source video", message="Starting YouTube download...")
                input_video, video_title = download_youtube_video(args.url, output_dir)
            else:
                input_video = args.input
                video_title = os.path.splitext(os.path.basename(input_video))[0]
                if args.output and not args.skip_analysis:
                    output_dir = _ensure_dir(args.output)
                else:
                    if args.output and os.path.isdir(args.output):
                        output_dir = args.output
                    elif args.output and not os.path.isdir(args.output):
                        output_dir = os.path.dirname(args.output) or os.path.dirname(input_video)
                    else:
                        output_dir = os.path.dirname(input_video)

        if not input_video or not os.path.exists(input_video):
            raise FileNotFoundError(f"Input file not found: {input_video}")

        # Persist render settings so a resume renders exactly like the original run.
        _save_json_file(os.path.join(output_dir, "render_config.json"), {"output_format": output_format})
        print(f"🖼️  Output format: {output_format}")

        reporter.artifact("source_video", input_video, message=f"Source video ready: {input_video}")
        duration = duration or _get_video_duration(input_video)
        reporter.emit("heartbeat", message="Source video loaded.", video_duration_seconds=round(float(duration), 3), important=False)

        transcript_file = os.path.join(output_dir, f"{video_title}_transcript.json")
        words_file = os.path.join(output_dir, f"{video_title}_words.json")
        analysis_input_file = os.path.join(output_dir, f"{video_title}_analysis_input.json")
        analysis_result_file = os.path.join(output_dir, f"{video_title}_analysis_result.json")
        metadata_file = os.path.join(output_dir, f"{video_title}_metadata.json")

        if args.skip_analysis and not args.resume_dir:
            reporter.set_phase("render", "Rendering full video", message="Skipping AI analysis and rendering the full video.")
            output_file = args.output if args.output else os.path.join(output_dir, f"{video_title}_vertical.mp4")
            success = _render_clip(
                input_video,
                output_file,
                output_format=output_format,
                progress_callback=lambda percent, message: reporter.progress(percent, message=message, category="render"),
            )
            if not success:
                raise RuntimeError("Full-video render failed.")
        else:
            if not transcript or args.resume_phase == "transcribe":
                reporter.set_phase("transcribe", "Transcribing audio", message="Starting transcription...")
                transcript = transcribe_video(input_video, duration)
                _save_json_file(transcript_file, transcript)
                reporter.artifact("transcript", transcript_file)
                words_payload = {"words": _extract_words_for_analysis(transcript)}
                _save_json_file(words_file, words_payload)
                reporter.artifact("words", words_file)

            analysis_input_payload = _build_analysis_input_payload(
                transcript_result=transcript,
                video_duration=duration,
                source_url=source_url,
                input_video=input_video,
            )
            _save_json_file(analysis_input_file, analysis_input_payload)
            print(f"📝 Saved analysis input to {analysis_input_file}")
            reporter.artifact("analysis_input", analysis_input_file)

            if not analysis_result or args.resume_phase == "analyze":
                reporter.set_phase("analyze", "Analyzing with Gemini", message="Starting Gemini analysis...")
                analysis_result = get_viral_clips(
                    transcript,
                    duration,
                    output_dir=output_dir,
                    video_title=video_title,
                )
                _save_json_file(analysis_result_file, analysis_result)
                reporter.artifact("analysis_result", analysis_result_file)

            clips_data = analysis_result["clips_data"]

            reporter.set_phase("render", "Rendering clips", message="Starting vertical render...")
            if not clips_data or 'shorts' not in clips_data:
                print("❌ Failed to identify clips. Converting whole video as fallback.")
                output_file = os.path.join(output_dir, f"{video_title}_vertical.mp4")
                success = _render_clip(
                    input_video,
                    output_file,
                    output_format=output_format,
                    progress_callback=lambda percent, message: reporter.progress(percent, message=message, category="render"),
                )
                if not success:
                    raise RuntimeError("Full-video fallback rendering failed.")

                fallback_metadata = _build_fallback_metadata(
                    video_title=video_title,
                    transcript=transcript,
                    duration=duration,
                    output_filename=output_file,
                    analysis_error=analysis_result["error"],
                    attempts=analysis_result["attempts"],
                    cost_analysis=analysis_result["cost_analysis"],
                )
                _save_json_file(metadata_file, fallback_metadata)
                reporter.artifact("metadata", metadata_file, message=f"Saved fallback metadata to {metadata_file}")
                reporter.warning("Fallback video rendered after AI analysis failed.")
            else:
                print(f"🔥 Found {len(clips_data['shorts'])} viral clips!")
                clips_data['schema_version'] = 1
                clips_data['processing_mode'] = 'clips'
                clips_data['analysis_status'] = 'success'
                clips_data['analysis_attempts'] = analysis_result["attempts"]
                clips_data['transcript'] = transcript
                for i, clip in enumerate(clips_data['shorts']):
                    clip['output_filename'] = f"{video_title}_clip_{i+1}.mp4"
                _save_json_file(metadata_file, clips_data)
                reporter.artifact("metadata", metadata_file)

                total_clips = len(clips_data['shorts'])
                for i, clip in enumerate(clips_data['shorts']):
                    start = clip['start']
                    end = clip['end']
                    print(f"\n🎬 Processing Clip {i+1}: {start}s - {end}s")
                    print(f"   Title: {clip.get('video_title_for_youtube_short', 'No Title')}")
                    clip_filename = clip.get('output_filename', f"{video_title}_clip_{i+1}.mp4")
                    clip_temp_path = os.path.join(output_dir, f"temp_{clip_filename}")
                    clip_final_path = os.path.join(output_dir, clip_filename)

                    reporter.progress(
                        (i / max(total_clips, 1)) * 100.0,
                        message=f"Preparing clip {i + 1}/{total_clips}",
                        important=True,
                        category="render",
                    )

                    cut_command = [
                        'ffmpeg', '-y',
                        '-ss', str(start),
                        '-to', str(end),
                        '-i', input_video,
                        '-c:v', 'libx264', '-crf', '18', '-preset', 'fast',
                        '-c:a', 'aac',
                        clip_temp_path
                    ]
                    try:
                        subprocess.run(cut_command, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, check=True, timeout=1800)
                    except subprocess.TimeoutExpired:
                        raise RuntimeError(f"FFmpeg clip cut timed out after 1800s for {clip_filename}")

                    def _clip_progress(inner_percent, message, clip_index=i, clip_count=total_clips):
                        render_percent = ((clip_index / max(clip_count, 1)) * 100.0) + ((inner_percent / 100.0) * (100.0 / max(clip_count, 1)))
                        reporter.progress(
                            render_percent,
                            message=f"Rendering clip {clip_index + 1}/{clip_count}: {message}",
                            important=inner_percent >= 100.0,
                            category="render",
                        )

                    success = _render_clip(clip_temp_path, clip_final_path, output_format=output_format, progress_callback=_clip_progress)
                    if not success:
                        raise RuntimeError(f"Clip render failed for {clip_filename}")
                    reporter.artifact(f"clip_{i + 1}", clip_final_path, message=f"Clip {i + 1} ready: {clip_final_path}")
                    if os.path.exists(clip_temp_path):
                        os.remove(clip_temp_path)

        reporter.set_phase("finalize", "Finalizing output", message="Wrapping up artifacts...")
        if args.url and not args.keep_original:
            print(f"🗂️  Keeping downloaded source for resume/retention: {input_video}")
            reporter.warning("Downloaded source video kept for resume support and delayed cleanup.")

        total_time = time.time() - script_start_time
        print(f"\n⏱️  Total execution time: {total_time:.2f}s")
        reporter.summary("completed", f"Job completed in {total_time:.2f}s.", resumable=False)
    except Exception as e:
        print(f"❌ Fatal pipeline error: {e}")
        reporter.error(str(e), resumable=True)
        raise
