import os
import sys
import uuid
import subprocess
import threading
import queue
import json
import shutil
import glob
import time
import asyncio
import re
import zipfile
from datetime import datetime, timezone
from dotenv import load_dotenv
from typing import Dict, Optional, List
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request, Header, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from starlette.background import BackgroundTask
from pydantic import BaseModel
from s3_uploader import upload_job_artifacts, list_all_clips, upload_actor_to_s3, list_actor_gallery, upload_video_to_gallery, list_video_gallery

load_dotenv()

# Constants
UPLOAD_DIR = "uploads"
OUTPUT_DIR = "output"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Configuration
# Default to 1 if not set, but user can set higher for powerful servers
MAX_CONCURRENT_JOBS = int(os.environ.get("MAX_CONCURRENT_JOBS", "5"))
MAX_FILE_SIZE_MB = 2048  # 2GB limit
JOB_RETENTION_SECONDS = int(os.environ.get("JOB_RETENTION_SECONDS", str(24 * 3600)))
HEARTBEAT_STALL_WARNING_SECONDS = int(os.environ.get("JOB_STALL_WARNING_SECONDS", "30"))
HEARTBEAT_STALLED_SECONDS = int(os.environ.get("JOB_STALLED_SECONDS", "90"))
JOB_LOG_LIMIT = int(os.environ.get("JOB_LOG_LIMIT", "4000"))
IMPORTANT_LOG_LIMIT = int(os.environ.get("JOB_IMPORTANT_LOG_LIMIT", "1000"))
EVENT_PREFIX = "__JOB_EVENT__"
JOB_STATE_FILENAME = "job_state.json"
JOB_TOMBSTONE_DIR = os.path.join(OUTPUT_DIR, ".job_tombstones")
ACTIVE_JOB_STATUSES = {"queued", "processing"}
TERMINAL_JOB_STATUSES = {"completed", "failed", "stalled", "archived"}
os.makedirs(JOB_TOMBSTONE_DIR, exist_ok=True)

# Application State
job_queue = asyncio.Queue()
jobs: Dict[str, Dict] = {}
thumbnail_sessions: Dict[str, Dict] = {}
publish_jobs: Dict[str, Dict] = {}  # {publish_id: {status, result, error, created_at}}
# Semester to limit concurrency to MAX_CONCURRENT_JOBS
concurrency_semaphore = asyncio.Semaphore(MAX_CONCURRENT_JOBS)

# Serializes writes to job_state.json across the log thread and the run_job coroutine.
_state_write_lock = threading.Lock()

# Per-job registry of subprocesses we control, so cancellation can terminate them.
# job_id -> set[subprocess.Popen]; guarded by job_processes_lock.
job_processes: Dict[str, set] = {}
job_processes_lock = threading.Lock()

# TTLs for in-memory dicts that would otherwise grow unbounded.
THUMBNAIL_SESSION_TTL_SECONDS = int(os.environ.get("THUMBNAIL_SESSION_TTL_SECONDS", str(2 * 3600)))
PUBLISH_JOB_TTL_SECONDS = int(os.environ.get("PUBLISH_JOB_TTL_SECONDS", str(3600)))


def _register_job_process(job_id: str, proc: "subprocess.Popen") -> None:
    with job_processes_lock:
        job_processes.setdefault(job_id, set()).add(proc)


def _unregister_job_process(job_id: str, proc: "subprocess.Popen") -> None:
    with job_processes_lock:
        procs = job_processes.get(job_id)
        if procs:
            procs.discard(proc)
            if not procs:
                job_processes.pop(job_id, None)


def _stop_process(proc: "subprocess.Popen") -> None:
    """Terminate a process, waiting up to 5s, then kill. No-op if already exited."""
    try:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
    except Exception as e:
        print(f"⚠️ Failed to stop process: {e}")


def _terminate_job_processes(job_id: str) -> None:
    """Terminate (then kill) any subprocesses registered for a job. Blocking; run off the loop."""
    with job_processes_lock:
        procs = list(job_processes.get(job_id, set()))
    for proc in procs:
        _stop_process(proc)


def _now_ts() -> float:
    return time.time()


def _isoformat(ts: Optional[float] = None) -> str:
    return datetime.fromtimestamp(ts or _now_ts(), tz=timezone.utc).isoformat()


def _job_state_path(job_id: str, output_dir: Optional[str] = None) -> str:
    job_dir = output_dir or os.path.join(OUTPUT_DIR, job_id)
    return os.path.join(job_dir, JOB_STATE_FILENAME)


def _job_tombstone_path(job_id: str) -> str:
    return os.path.join(JOB_TOMBSTONE_DIR, f"{job_id}.json")


def _safe_write_json(path: str, payload: dict) -> None:
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    # Unique temp name per write + a lock so the log thread and the run_job
    # coroutine can't clobber the same ".tmp" file and corrupt the state.
    temp_path = f"{path}.{uuid.uuid4().hex}.tmp"
    with _state_write_lock:
        try:
            with open(temp_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, ensure_ascii=False)
            os.replace(temp_path, path)
        except Exception:
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except OSError:
                    pass
            raise


def _read_json(path: str) -> Optional[dict]:
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _maybe_fix_mojibake_text(text: str) -> str:
    if not isinstance(text, str):
        return text
    if not any(marker in text for marker in ("Ã", "Â", "â", "Ð", "Ñ")):
        return text
    for source_encoding in ("cp1252", "latin-1"):
        try:
            repaired = text.encode(source_encoding, errors="strict").decode("utf-8", errors="strict")
        except Exception:
            continue
        if repaired != text:
            return repaired
    return text


def _repair_mojibake(payload):
    if isinstance(payload, str):
        return _maybe_fix_mojibake_text(payload)
    if isinstance(payload, list):
        return [_repair_mojibake(item) for item in payload]
    if isinstance(payload, dict):
        return {key: _repair_mojibake(value) for key, value in payload.items()}
    return payload


def _load_transcript_for_job(output_dir: str, metadata: Optional[dict] = None):
    transcript_files = sorted(glob.glob(os.path.join(output_dir, "*_transcript.json")))
    if transcript_files:
        transcript_payload = _read_json(transcript_files[0])
        if transcript_payload:
            return _repair_mojibake(transcript_payload)
    if metadata and metadata.get("transcript"):
        return _repair_mojibake(metadata.get("transcript"))
    return None


def _trim_list(items: List, limit: int) -> List:
    if len(items) <= limit:
        return items
    return items[-limit:]


def _make_log_entry(message: str, level: str = "info", category: str = "general", important: bool = False, ts: Optional[float] = None) -> dict:
    timestamp = ts or _now_ts()
    return {
        "timestamp": timestamp,
        "iso_timestamp": _isoformat(timestamp),
        "level": level,
        "category": category,
        "important": important,
        "message": message,
    }


def _serialize_job(job: dict) -> dict:
    persisted = {}
    allowed_keys = {
        "job_id",
        "status",
        "phase",
        "phase_label",
        "progress_percent",
        "phase_progress_percent",
        "created_at",
        "started_at",
        "updated_at",
        "last_heartbeat_at",
        "elapsed_seconds",
        "eta_seconds",
        "attempt",
        "resume_count",
        "stall_state",
        "error_summary",
        "warnings",
        "artifacts",
        "source_type",
        "source_url",
        "input_path",
        "input_filename",
        "output_dir",
        "video_duration_seconds",
        "total_estimate_seconds",
        "is_resumable",
        "raw_logs",
        "important_logs",
        "result",
        "processing_mode",
        "analysis_error",
        "analysis_status",
        "archive_status",
        "job_type",
    }
    for key in allowed_keys:
        if key in job:
            persisted[key] = job[key]
    persisted["logs"] = [entry.get("message", "") for entry in persisted.get("raw_logs", [])]
    return persisted


def _persist_job_state(job_id: str) -> None:
    job = jobs.get(job_id)
    if not job:
        return
    output_dir = job.get("output_dir")
    if not output_dir:
        return
    try:
        _safe_write_json(_job_state_path(job_id, output_dir), _serialize_job(job))
    except Exception as e:
        print(f"⚠️ Failed to persist job state for {job_id}: {e}")


def _build_job_state(job_id: str, *, output_dir: str, source_type: Optional[str] = None, source_url: Optional[str] = None,
                     input_path: Optional[str] = None, input_filename: Optional[str] = None, status: str = "queued") -> dict:
    now = _now_ts()
    return {
        "job_id": job_id,
        "job_type": "clip_generator",
        "status": status,
        "phase": "queued",
        "phase_label": "Queued",
        "progress_percent": 0.0,
        "phase_progress_percent": 0.0,
        "created_at": now,
        "started_at": None,
        "updated_at": now,
        "last_heartbeat_at": now,
        "elapsed_seconds": 0.0,
        "eta_seconds": None,
        "attempt": 0,
        "resume_count": 0,
        "stall_state": "healthy",
        "error_summary": None,
        "warnings": [],
        "artifacts": {},
        "source_type": source_type,
        "source_url": source_url,
        "input_path": input_path,
        "input_filename": input_filename,
        "output_dir": output_dir,
        "video_duration_seconds": None,
        "is_resumable": False,
        "raw_logs": [],
        "important_logs": [],
        "logs": [],
        "result": None,
        "analysis_status": None,
        "analysis_error": None,
        "processing_mode": None,
    }


def _append_log(job_id: str, message: str, *, level: str = "info", category: str = "general", important: bool = False, ts: Optional[float] = None) -> None:
    job = jobs.get(job_id)
    if not job:
        return
    entry = _make_log_entry(message, level=level, category=category, important=important, ts=ts)
    job.setdefault("raw_logs", []).append(entry)
    job["raw_logs"] = _trim_list(job["raw_logs"], JOB_LOG_LIMIT)
    if important:
        job.setdefault("important_logs", []).append(entry)
        job["important_logs"] = _trim_list(job["important_logs"], IMPORTANT_LOG_LIMIT)
    job["logs"] = [log_entry["message"] for log_entry in job["raw_logs"]]
    job["updated_at"] = entry["timestamp"]
    _persist_job_state(job_id)


def _set_job_result(job_id: str, result: dict) -> None:
    job = jobs.get(job_id)
    if not job:
        return
    job["result"] = result
    job["updated_at"] = _now_ts()
    _persist_job_state(job_id)


def _mark_job_status(job_id: str, status: str, *, error_summary: Optional[str] = None, resumable: Optional[bool] = None) -> None:
    job = jobs.get(job_id)
    if not job:
        return
    now = _now_ts()
    job["status"] = status
    job["updated_at"] = now
    if status == "processing" and not job.get("started_at"):
        job["started_at"] = now
    if error_summary is not None:
        job["error_summary"] = error_summary
    if resumable is not None:
        job["is_resumable"] = resumable
    _persist_job_state(job_id)


def _event_log_defaults(event_type: str) -> tuple[str, bool]:
    if event_type in {"error"}:
        return "error", True
    if event_type in {"warning", "slow", "timeout", "stalled"}:
        return "warning", True
    if event_type in {"phase", "artifact", "resume", "summary"}:
        return "info", True
    return "info", False


def _apply_job_event(job_id: str, event: dict) -> None:
    job = jobs.get(job_id)
    if not job:
        return

    now = float(event.get("timestamp") or _now_ts())
    event_type = str(event.get("type") or "log")
    message = str(event.get("message") or "").strip()
    category = str(event.get("category") or event.get("phase") or "general")
    level, default_important = _event_log_defaults(event_type)
    important = bool(event.get("important", default_important))

    job["updated_at"] = now
    if event_type in {"heartbeat", "phase", "progress", "slow", "resume"}:
        job["last_heartbeat_at"] = now
    if event_type == "stalled":
        job["stall_state"] = "stalled"
        job["status"] = "stalled"
        job["is_resumable"] = True
    elif event_type == "slow":
        job["stall_state"] = "slow"
    elif event_type in {"heartbeat", "progress", "phase", "resume"}:
        job["stall_state"] = "healthy"

    if event.get("status"):
        job["status"] = event["status"]
    if event.get("phase"):
        job["phase"] = event["phase"]
    if event.get("phase_label"):
        job["phase_label"] = event["phase_label"]
    if event.get("progress_percent") is not None:
        job["progress_percent"] = max(0.0, min(100.0, float(event["progress_percent"])))
    if event.get("phase_progress_percent") is not None:
        job["phase_progress_percent"] = max(0.0, min(100.0, float(event["phase_progress_percent"])))
    if event.get("eta_seconds") is not None:
        job["eta_seconds"] = max(0, int(event["eta_seconds"]))
    if event.get("attempt") is not None:
        job["attempt"] = int(event["attempt"])
    if event.get("resume_count") is not None:
        job["resume_count"] = int(event["resume_count"])
    if event.get("video_duration_seconds") is not None:
        job["video_duration_seconds"] = float(event["video_duration_seconds"])
    if event.get("total_estimate_seconds") is not None:
        job["total_estimate_seconds"] = max(0, int(event["total_estimate_seconds"]))
    if event.get("resumable") is not None:
        job["is_resumable"] = bool(event["resumable"])
    if event.get("processing_mode") is not None:
        job["processing_mode"] = event["processing_mode"]
    if event.get("analysis_status") is not None:
        job["analysis_status"] = event["analysis_status"]
    if event.get("analysis_error") is not None:
        job["analysis_error"] = event["analysis_error"]
        job["error_summary"] = event["analysis_error"]

    artifact = event.get("artifact")
    if isinstance(artifact, dict):
        kind = artifact.get("kind")
        path = artifact.get("path")
        if kind and path:
            job.setdefault("artifacts", {})[kind] = path

    if event_type in {"warning", "slow", "timeout"} and message:
        warnings = job.setdefault("warnings", [])
        warnings.append(message)
        job["warnings"] = _trim_list(warnings, 100)

    if event_type == "error" and message:
        job["error_summary"] = message
        job["status"] = "failed"
        job["is_resumable"] = True

    if event_type == "summary" and event.get("status") == "completed":
        job["status"] = "completed"
        job["is_resumable"] = False

    if message:
        _append_log(job_id, message, level=level, category=category, important=important, ts=now)
    else:
        _persist_job_state(job_id)


def _classify_raw_log(message: str) -> tuple[str, str, bool]:
    stripped = message.strip()
    lower = stripped.lower()

    if re.match(r"^\s*\[\d+(\.\d+)?s\s*->\s*\d+(\.\d+)?s\]", stripped):
        return "info", "transcript", False
    if "processing:" in lower or "tqdm" in lower:
        return "info", "progress", False

    # Treat yt-dlp/debug output as low-signal unless it clearly contains a fatal marker.
    if stripped.startswith("[debug]"):
        if any(token in lower for token in ("traceback", "fatal", "exception")):
            return "error", "error", True
        return "info", "debug", False

    if stripped.startswith("[download]") and "%" in stripped:
        return "info", "progress", False
    if stripped.startswith("[download] Destination:") or "destination:" in lower:
        return "info", "download", True

    if stripped.startswith("WARNING:") or "⚠️" in stripped:
        return "warning", "warning", True
    if (
        stripped.startswith(("ERROR:", "Error:", "FATAL:", "Fatal:", "Traceback"))
        or "❌" in stripped
        or re.search(r"\b(exception|traceback|fatal)\b", lower)
    ):
        return "error", "error", True
    if any(token in stripped for token in ("✅", "🔥", "📝 Saved", "⏱️", "🔄", "🤖", "🎬", "🧹", "⏩")):
        return "info", "system", True
    return "info", "raw", False


def _display_log_entry(entry: dict) -> dict:
    message = entry.get("message", "")
    stripped = message.strip()
    looks_like_worker_output = (
        stripped.startswith("[")
        or stripped.startswith(("WARNING:", "ERROR:", "Error:", "FATAL:", "Fatal:", "Traceback"))
        or any(token in stripped for token in ("✅", "🔥", "📝", "⏱️", "🔄", "🤖", "🎬", "🧹", "⏩", "❌", "⚠️"))
    )
    if not looks_like_worker_output:
        return entry

    level, category, important = _classify_raw_log(message)
    normalized = dict(entry)
    normalized["level"] = level
    normalized["category"] = category
    normalized["important"] = important
    return normalized


def _hydrate_job_from_disk(job_id: str) -> Optional[dict]:
    path = _job_state_path(job_id)
    payload = _read_json(path)
    if not payload:
        return None
    payload.setdefault("job_id", job_id)
    payload.setdefault("output_dir", os.path.join(OUTPUT_DIR, job_id))
    payload.setdefault("raw_logs", [])
    payload.setdefault("important_logs", [])
    payload["logs"] = [entry.get("message", "") for entry in payload.get("raw_logs", [])]
    jobs[job_id] = payload
    return payload


def _get_job(job_id: str) -> Optional[dict]:
    return jobs.get(job_id) or _hydrate_job_from_disk(job_id)


def _build_archive_payload(job: dict) -> dict:
    return {
        "job_id": job.get("job_id"),
        "status": "archived",
        "archived_at": _isoformat(),
        "last_status": job.get("status"),
        "error_summary": job.get("error_summary"),
        "source_type": job.get("source_type"),
        "source_url": job.get("source_url"),
        "phase": job.get("phase"),
        "progress_percent": job.get("progress_percent"),
        "video_duration_seconds": job.get("video_duration_seconds"),
        "is_resumable": False,
    }


def _write_tombstone(job_id: str, job: dict) -> None:
    _safe_write_json(_job_tombstone_path(job_id), _build_archive_payload(job))


def _get_job_tombstone(job_id: str) -> Optional[dict]:
    return _read_json(_job_tombstone_path(job_id))


def _recover_jobs_from_disk() -> None:
    for entry in os.listdir(OUTPUT_DIR):
        if entry.startswith(".") or entry == "thumbnails":
            continue
        output_dir = os.path.join(OUTPUT_DIR, entry)
        if not os.path.isdir(output_dir):
            continue
        state = _read_json(_job_state_path(entry, output_dir))
        if not state:
            continue
        state.setdefault("job_id", entry)
        state.setdefault("output_dir", output_dir)
        state.setdefault("raw_logs", [])
        state.setdefault("important_logs", [])
        state["logs"] = [log_entry.get("message", "") for log_entry in state.get("raw_logs", [])]
        if state.get("status") in ACTIVE_JOB_STATUSES:
            state["status"] = "stalled"
            state["stall_state"] = "stalled"
            state["is_resumable"] = True
            state["error_summary"] = state.get("error_summary") or "Server restart detected while the job was still running."
            state["updated_at"] = _now_ts()
            state["last_heartbeat_at"] = state.get("last_heartbeat_at") or state["updated_at"]
            jobs[entry] = state
            _append_log(entry, "Server restart detected. Job marked as stalled and resumable.", category="resume", important=True)
        else:
            jobs[entry] = state

def _relocate_root_job_artifacts(job_id: str, job_output_dir: str) -> bool:
    """
    Backward-compat rescue:
    If main.py accidentally wrote metadata/clips into OUTPUT_DIR root (e.g. output/<jobid>_...),
    move them into output/<job_id>/ so the API can find and serve them.
    """
    try:
        os.makedirs(job_output_dir, exist_ok=True)
        root = OUTPUT_DIR
        pattern = os.path.join(root, f"{job_id}_*_metadata.json")
        meta_candidates = sorted(glob.glob(pattern), key=lambda p: os.path.getmtime(p), reverse=True)
        if not meta_candidates:
            return False

        # Move the newest metadata and its associated clips.
        metadata_path = meta_candidates[0]
        base_name = os.path.basename(metadata_path).replace("_metadata.json", "")

        # Move metadata
        dest_metadata = os.path.join(job_output_dir, os.path.basename(metadata_path))
        if os.path.abspath(metadata_path) != os.path.abspath(dest_metadata):
            shutil.move(metadata_path, dest_metadata)

        # Move any clips that match the same base_name into the job folder
        clip_pattern = os.path.join(root, f"{base_name}_clip_*.mp4")
        for clip_path in glob.glob(clip_pattern):
            dest_clip = os.path.join(job_output_dir, os.path.basename(clip_path))
            if os.path.abspath(clip_path) != os.path.abspath(dest_clip):
                shutil.move(clip_path, dest_clip)

        # Also move any temp_ clips that might remain
        temp_clip_pattern = os.path.join(root, f"temp_{base_name}_clip_*.mp4")
        for clip_path in glob.glob(temp_clip_pattern):
            dest_clip = os.path.join(job_output_dir, os.path.basename(clip_path))
            if os.path.abspath(clip_path) != os.path.abspath(dest_clip):
                shutil.move(clip_path, dest_clip)

        return True
    except Exception:
        return False


def _resolve_clip_filename(clip: dict, base_name: str, clip_index: int) -> str:
    video_url = clip.get("video_url")
    if video_url:
        filename = os.path.basename(video_url)
        if filename:
            return filename

    output_filename = clip.get("output_filename")
    if output_filename:
        return os.path.basename(output_filename)

    return f"{base_name}_clip_{clip_index + 1}.mp4"


def _build_result_from_metadata(job_id: str, metadata_path: str, output_dir: str, ready_only: bool = False) -> Optional[dict]:
    with open(metadata_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    base_name = os.path.basename(metadata_path).replace('_metadata.json', '')
    clips = []

    for i, clip in enumerate(data.get('shorts', [])):
        if not isinstance(clip, dict):
            continue

        clip_filename = _resolve_clip_filename(clip, base_name, i)
        clip_path = os.path.join(output_dir, clip_filename)
        if not os.path.exists(clip_path) or os.path.getsize(clip_path) <= 0:
            continue

        clip_data = dict(clip)
        clip_data['output_filename'] = clip_filename
        clip_data['video_url'] = f"/videos/{job_id}/{clip_filename}"
        clips.append(clip_data)

    if not clips:
        return None

    result = {'clips': clips, 'cost_analysis': data.get('cost_analysis')}
    for extra_key in ('analysis_status', 'analysis_error', 'processing_mode'):
        if extra_key in data:
            result[extra_key] = data.get(extra_key)

    return result


def _build_result_from_video_artifacts(job_id: str, output_dir: str) -> Optional[dict]:
    fallback_candidates = sorted(
        glob.glob(os.path.join(output_dir, "*_vertical.mp4")),
        key=lambda p: os.path.getmtime(p),
        reverse=True,
    )
    if not fallback_candidates:
        return None

    fallback_path = fallback_candidates[0]
    fallback_filename = os.path.basename(fallback_path)
    fallback_title = os.path.splitext(fallback_filename)[0].replace("_vertical", "").replace("_", " ").strip()

    return {
        'clips': [
            {
                'start': 0.0,
                'end': 0.0,
                'video_title_for_youtube_short': fallback_title or "Fallback video",
                'video_description_for_tiktok': "Automatic fallback output generated without clip metadata.",
                'video_description_for_instagram': "Automatic fallback output generated without clip metadata.",
                'viral_hook_text': "Automatic fallback",
                'output_filename': fallback_filename,
                'video_url': f"/videos/{job_id}/{fallback_filename}",
            }
        ],
        'analysis_status': 'fallback_missing_metadata',
        'analysis_error': 'No metadata file was generated, but a fallback video was created successfully.',
        'processing_mode': 'full_video_fallback',
    }

async def cleanup_jobs():
    """Background task to remove old jobs and files."""
    print("🧹 Cleanup task started.")
    while True:
        try:
            await asyncio.sleep(300) # Check every 5 minutes
            now = time.time()

            protected_uploads = set()
            for job_id, job in list(jobs.items()):
                status = job.get("status")
                input_path = job.get("input_path")
                if input_path and (status in ACTIVE_JOB_STATUSES or job.get("is_resumable")):
                    protected_uploads.add(os.path.abspath(input_path))

                if status == "processing":
                    last_heartbeat_at = job.get("last_heartbeat_at") or job.get("updated_at") or job.get("created_at") or now
                    heartbeat_age = now - float(last_heartbeat_at)
                    if heartbeat_age >= HEARTBEAT_STALLED_SECONDS:
                        job["status"] = "stalled"
                        job["stall_state"] = "stalled"
                        job["is_resumable"] = True
                        job["error_summary"] = job.get("error_summary") or "No heartbeat received for too long."
                        _append_log(job_id, "Job heartbeat timed out. Marked as stalled.", level="warning", category="stall", important=True)
                    elif heartbeat_age >= HEARTBEAT_STALL_WARNING_SECONDS and job.get("stall_state") != "slow":
                        job["stall_state"] = "slow"
                        _append_log(job_id, "Job is slower than expected but still waiting for activity.", level="warning", category="stall", important=True)

            for job_id in os.listdir(OUTPUT_DIR):
                if job_id.startswith(".") or job_id == "thumbnails":
                    continue
                job_path = os.path.join(OUTPUT_DIR, job_id)
                if not os.path.isdir(job_path):
                    continue

                job = _get_job(job_id)
                if job and job.get("status") in ACTIVE_JOB_STATUSES:
                    continue

                state_path = _job_state_path(job_id, job_path)
                if os.path.exists(state_path):
                    state = _read_json(state_path) or {}
                    status = state.get("status")
                    if status in ACTIVE_JOB_STATUSES:
                        continue

                try:
                    mtime = os.path.getmtime(job_path)
                except FileNotFoundError:
                    continue

                if now - mtime <= JOB_RETENTION_SECONDS:
                    continue

                archived_job = job or _read_json(state_path) or {"job_id": job_id, "status": "archived"}
                print(f"🧹 Purging old job: {job_id}")
                try:
                    _write_tombstone(job_id, archived_job)
                except Exception as e:
                    print(f"⚠️ Failed to write tombstone for {job_id}: {e}")
                shutil.rmtree(job_path, ignore_errors=True)
                jobs.pop(job_id, None)

            # Cleanup SaaSShorts jobs from memory
            try:
                saas_expired = [
                    jid for jid, jdata in list(saas_jobs.items())
                    if jdata.get("status") in ("completed", "failed")
                    and jdata.get("output_dir")
                    and os.path.isdir(jdata["output_dir"])
                    and now - os.path.getmtime(jdata["output_dir"]) > JOB_RETENTION_SECONDS
                ]
                for jid in saas_expired:
                    del saas_jobs[jid]
            except NameError:
                pass

            # Cleanup thumbnail sessions (TTL-based; created_at added at creation).
            for sid, session in list(thumbnail_sessions.items()):
                created = session.get("created_at") or 0
                if now - float(created) > THUMBNAIL_SESSION_TTL_SECONDS:
                    thumbnail_sessions.pop(sid, None)

            # Cleanup finished publish jobs (TTL-based).
            for pid, pjob in list(publish_jobs.items()):
                created = pjob.get("created_at") or 0
                if now - float(created) > PUBLISH_JOB_TTL_SECONDS:
                    publish_jobs.pop(pid, None)

            # Cleanup Uploads
            for filename in os.listdir(UPLOAD_DIR):
                file_path = os.path.join(UPLOAD_DIR, filename)
                try:
                    if os.path.abspath(file_path) in protected_uploads:
                        continue
                    if now - os.path.getmtime(file_path) > JOB_RETENTION_SECONDS:
                        os.remove(file_path)
                except Exception: pass

        except Exception as e:
            print(f"⚠️ Cleanup error: {e}")

async def process_queue():
    """Background worker to process jobs from the queue with concurrency limit."""
    print(f"🚀 Job Queue Worker started with {MAX_CONCURRENT_JOBS} concurrent slots.")
    while True:
        try:
            # Wait for a job
            job_id = await job_queue.get()
            
            # Acquire semaphore slot (waits if max jobs are running)
            await concurrency_semaphore.acquire()
            print(f"🔄 Acquired slot for job: {job_id}")

            # Process in background task to not block the loop (allowing other slots to fill)
            asyncio.create_task(run_job_wrapper(job_id))
            
        except Exception as e:
            print(f"❌ Queue dispatch error: {e}")
            await asyncio.sleep(1)

async def run_job_wrapper(job_id):
    """Wrapper to run job and release semaphore"""
    try:
        job = jobs.get(job_id)
        if job:
            await run_job(job_id, job)
    except Exception as e:
         print(f"❌ Job wrapper error {job_id}: {e}")
    finally:
        # Always release semaphore and mark queue task done
        concurrency_semaphore.release()
        job_queue.task_done()
        print(f"✅ Released slot for job: {job_id}")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Start worker and cleanup
    _recover_jobs_from_disk()
    worker_task = asyncio.create_task(process_queue())
    cleanup_task = asyncio.create_task(cleanup_jobs())
    yield
    # Cleanup (optional: cancel worker)

app = FastAPI(lifespan=lifespan)

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Compress JSON responses: a real 249 KB status-poll payload measures 29.6 KB
# gzipped (-88%). Video files are served via /videos (already compressed
# codecs, > minimum_size guard not relevant since StaticFiles streams).
app.add_middleware(GZipMiddleware, minimum_size=1024)

# Mount static files for serving videos
app.mount("/videos", StaticFiles(directory=OUTPUT_DIR), name="videos")

# Mount static files for serving thumbnails
THUMBNAILS_DIR = os.path.join(OUTPUT_DIR, "thumbnails")
os.makedirs(THUMBNAILS_DIR, exist_ok=True)
app.mount("/thumbnails", StaticFiles(directory=THUMBNAILS_DIR), name="thumbnails")

class ProcessRequest(BaseModel):
    url: str


class ResumeRequest(BaseModel):
    phase: Optional[str] = None

def enqueue_output(out, job_id):
    """Reads output from a subprocess and appends it to jobs logs."""
    try:
        for line in iter(out.readline, b''):
            decoded_line = line.decode('utf-8', errors='replace').strip()
            if decoded_line:
                print(f"📝 [Job Output] {decoded_line}")
                if job_id in jobs:
                    if decoded_line.startswith(EVENT_PREFIX):
                        try:
                            event = json.loads(decoded_line[len(EVENT_PREFIX):].strip())
                            _apply_job_event(job_id, event)
                            continue
                        except Exception as e:
                            _append_log(job_id, f"Failed to parse worker event: {e}", level="warning", category="event", important=True)
                    level, category, important = _classify_raw_log(decoded_line)
                    _append_log(job_id, decoded_line, level=level, category=category, important=important)
    except Exception as e:
        print(f"Error reading output for job {job_id}: {e}")
    finally:
        out.close()


def _refresh_job_result(job_id: str, output_dir: str) -> Optional[dict]:
    json_files = glob.glob(os.path.join(output_dir, "*_metadata.json"))
    if not json_files:
        if _relocate_root_job_artifacts(job_id, output_dir):
            json_files = glob.glob(os.path.join(output_dir, "*_metadata.json"))

    result = None
    if json_files:
        result = _build_result_from_metadata(
            job_id=job_id,
            metadata_path=json_files[0],
            output_dir=output_dir,
            ready_only=False,
        )
    if not result:
        result = _build_result_from_video_artifacts(job_id, output_dir)
    if result:
        _set_job_result(job_id, result)
    return result


def _build_status_payload(job: dict) -> dict:
    now = _now_ts()
    started_at = job.get("started_at")
    last_heartbeat_at = job.get("last_heartbeat_at")
    elapsed_seconds = None
    if started_at:
        elapsed_seconds = max(0, int(now - float(started_at)))
    seconds_since_heartbeat = None
    if last_heartbeat_at:
        seconds_since_heartbeat = max(0, int(now - float(last_heartbeat_at)))

    display_raw_logs = [_display_log_entry(entry) for entry in job.get("raw_logs", [])][-JOB_LOG_LIMIT:]
    display_important_logs = [entry for entry in display_raw_logs if entry.get("important")][-IMPORTANT_LOG_LIMIT:]

    return {
        "job_id": job.get("job_id"),
        "status": job.get("status"),
        "phase": job.get("phase"),
        "phase_label": job.get("phase_label"),
        "progress_percent": job.get("progress_percent"),
        "phase_progress_percent": job.get("phase_progress_percent"),
        "eta_seconds": job.get("eta_seconds"),
        "elapsed_seconds": elapsed_seconds,
        "created_at": job.get("created_at"),
        "updated_at": job.get("updated_at"),
        "last_heartbeat_at": last_heartbeat_at,
        "seconds_since_heartbeat": seconds_since_heartbeat,
        "attempt": job.get("attempt"),
        "resume_count": job.get("resume_count"),
        "stall_state": job.get("stall_state"),
        "error_summary": job.get("error_summary"),
        "warnings": job.get("warnings", []),
        "is_resumable": job.get("is_resumable", False),
        "source_type": job.get("source_type"),
        "source_url": job.get("source_url"),
        "video_duration_seconds": job.get("video_duration_seconds"),
        "total_estimate_seconds": job.get("total_estimate_seconds"),
        "important_logs": display_important_logs,
        "raw_logs": display_raw_logs,
        "logs": [entry.get("message", "") for entry in display_raw_logs],
        "result": job.get("result"),
        "analysis_status": job.get("analysis_status"),
        "analysis_error": job.get("analysis_error"),
        "processing_mode": job.get("processing_mode"),
        "artifacts": job.get("artifacts", {}),
    }


async def run_job(job_id, job_data):
    """Executes the subprocess for a specific job."""

    cmd = job_data['cmd']
    env = job_data['env']
    output_dir = job_data['output_dir']

    _mark_job_status(job_id, 'processing', resumable=True)
    jobs[job_id]['phase'] = 'queued'
    jobs[job_id]['phase_label'] = 'Queued'
    jobs[job_id]['last_heartbeat_at'] = _now_ts()
    jobs[job_id]['updated_at'] = jobs[job_id]['last_heartbeat_at']
    _append_log(job_id, "Job started by worker.", category="queue", important=True)
    print(f"🎬 [run_job] Executing command for {job_id}: {' '.join(cmd)}")

    # If the job was cancelled while still queued, don't spawn anything.
    if jobs.get(job_id, {}).get("cancel_requested"):
        _mark_job_status(job_id, 'failed', error_summary="Cancelled by user", resumable=False)
        _append_log(job_id, "Job cancelled before start.", level="warning", category="cancel", important=True)
        return

    process = None
    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, # Merge stderr to stdout
            env=env,
            cwd=os.getcwd()
        )
        _register_job_process(job_id, process)

        # We need to capture logs in a thread because Popen isn't async
        t_log = threading.Thread(target=enqueue_output, args=(process.stdout, job_id))
        t_log.daemon = True
        t_log.start()

        # Async wait for process with incremental updates
        start_wait = time.time()
        while process.poll() is None:
            await asyncio.sleep(2)

            # Stop promptly if the job was cancelled via the cancel endpoint.
            if jobs.get(job_id, {}).get("cancel_requested"):
                break

            # Check for partial results every 2 seconds
            # Look for metadata file
            try:
                json_files = glob.glob(os.path.join(output_dir, "*_metadata.json"))
                if json_files:
                    target_json = json_files[0]
                    if os.path.getsize(target_json) > 0:
                        partial_result = _build_result_from_metadata(
                            job_id=job_id,
                            metadata_path=target_json,
                            output_dir=output_dir,
                            ready_only=True,
                        )
                        if partial_result:
                            jobs[job_id]['result'] = partial_result
            except Exception as e:
                # Ignore read errors during processing
                pass

        # Cancellation path: kill the worker and mark the job accordingly.
        if jobs.get(job_id, {}).get("cancel_requested"):
            _stop_process(process)
            _mark_job_status(job_id, 'failed', error_summary="Cancelled by user", resumable=False)
            _append_log(job_id, "Job cancelled by user.", level="warning", category="cancel", important=True)
            return

        returncode = process.returncode

        if returncode == 0:
            _mark_job_status(job_id, 'completed', resumable=False)
            _append_log(job_id, "Process finished successfully.", category="summary", important=True)

            # Start S3 upload in background (silent, non-blocking)
            loop = asyncio.get_event_loop()
            loop.run_in_executor(None, upload_job_artifacts, output_dir, job_id)

            final_result = _refresh_job_result(job_id, output_dir)
            if final_result:
                if final_result.get("analysis_status"):
                    jobs[job_id]["analysis_status"] = final_result.get("analysis_status")
                if final_result.get("analysis_error"):
                    jobs[job_id]["analysis_error"] = final_result.get("analysis_error")
                if final_result.get("processing_mode"):
                    jobs[job_id]["processing_mode"] = final_result.get("processing_mode")
                if final_result.get("processing_mode") == "full_video_fallback":
                    _append_log(job_id, "Metadata missing, but fallback video artifacts were recovered.", level="warning", category="fallback", important=True)
            else:
                _mark_job_status(job_id, 'failed', error_summary="No metadata file generated.", resumable=True)
                _append_log(job_id, "No metadata file generated.", level="error", category="result", important=True)
        else:
            _mark_job_status(job_id, 'failed', error_summary=f"Process failed with exit code {returncode}", resumable=True)
            _append_log(job_id, f"Process failed with exit code {returncode}", level="error", category="process", important=True)

    except Exception as e:
        _mark_job_status(job_id, 'failed', error_summary=f"Execution error: {str(e)}", resumable=True)
        _append_log(job_id, f"Execution error: {str(e)}", level="error", category="process", important=True)
    finally:
        # Never leave the worker subprocess running or registered after we return.
        if process is not None:
            _stop_process(process)
            _unregister_job_process(job_id, process)

async def _save_upload_with_limit(file: UploadFile, dest_path: str, cleanup_dirs: Optional[List[str]] = None):
    """Stream an UploadFile to dest_path in 1MB chunks, enforcing MAX_FILE_SIZE_MB.

    On overflow, removes the partial file plus any cleanup_dirs and raises HTTP 413.
    """
    size = 0
    limit_bytes = MAX_FILE_SIZE_MB * 1024 * 1024

    with open(dest_path, "wb") as buffer:
        while content := await file.read(1024 * 1024):  # Read 1MB chunks
            size += len(content)
            if size > limit_bytes:
                if os.path.exists(dest_path):
                    os.remove(dest_path)
                for d in (cleanup_dirs or []):
                    shutil.rmtree(d, ignore_errors=True)
                raise HTTPException(status_code=413, detail=f"File too large. Max size {MAX_FILE_SIZE_MB}MB")
            buffer.write(content)

# Below this height the quality gate asks the user before starting (0 = off).
QUALITY_GATE_MIN_HEIGHT = int(os.environ.get("QUALITY_GATE_MIN_HEIGHT", "720"))
QUALITY_PROBE_SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "quality_probe.py")


async def _probe_youtube_quality(url: str) -> dict:
    """Run quality_probe.py in a worker thread; {} on any failure (fail-open)."""
    def _run():
        try:
            proc = subprocess.run(
                [sys.executable, QUALITY_PROBE_SCRIPT, "--url", url],
                capture_output=True, timeout=75,
            )
            return json.loads(proc.stdout.decode(errors="replace").strip() or "{}")
        except Exception as e:
            print(f"⚠️ Quality probe failed ({e}); starting job without gate.")
            return {}

    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _run)


@app.post("/api/process")
async def process_endpoint(
    request: Request,
    file: Optional[UploadFile] = File(None),
    url: Optional[str] = Form(None),
    output_format: Optional[str] = Form(None)
):
    api_key = request.headers.get("X-Gemini-Key")
    if not api_key:
        raise HTTPException(status_code=400, detail="Missing X-Gemini-Key header")

    # Handle JSON body manually for URL payload
    force_low_quality = False
    content_type = request.headers.get("content-type", "")
    if "application/json" in content_type:
        body = await request.json()
        url = body.get("url")
        force_low_quality = bool(body.get("force_low_quality"))
        output_format = body.get("output_format")

    if output_format not in ("vertical", "horizontal", "square"):
        output_format = "auto"

    if not url and not file:
        raise HTTPException(status_code=400, detail="Must provide URL or File")

    # Pre-flight quality gate: probe the offered resolution BEFORE starting the
    # job, so the user can abort (refresh cookies / update yt-dlp) instead of
    # burning 20 minutes of processing on a 360p-only source. Fail-open: any
    # probe error just starts the job normally.
    if url and not force_low_quality and QUALITY_GATE_MIN_HEIGHT > 0:
        probe = await _probe_youtube_quality(url)
        max_height = int(probe.get("max_height") or 0)
        if 0 < max_height < QUALITY_GATE_MIN_HEIGHT:
            print(f"⚠️ Quality gate: only {max_height}p available for {url} — asking user before starting.")
            return JSONResponse({
                "needs_confirmation": True,
                "quality_check": {
                    "max_height": max_height,
                    "min_height": QUALITY_GATE_MIN_HEIGHT,
                    "cookies_invalid": bool(probe.get("cookies_invalid")),
                },
            })

    job_id = str(uuid.uuid4())
    job_output_dir = os.path.join(OUTPUT_DIR, job_id)
    os.makedirs(job_output_dir, exist_ok=True)

    # Prepare Command
    cmd = [sys.executable, "-u", "main.py"] # -u for unbuffered, use same Python as server
    env = os.environ.copy()
    env["GEMINI_API_KEY"] = api_key # Override with key from request

    source_type = "url" if url else "file"
    input_path = None
    input_filename = None
    if url:
        cmd.extend(["-u", url])
    else:
        # Save uploaded file with size limit check
        input_path = os.path.join(UPLOAD_DIR, f"{job_id}_{file.filename}")
        input_filename = file.filename

        await _save_upload_with_limit(file, input_path, cleanup_dirs=[job_output_dir])

        cmd.extend(["-i", input_path])

    cmd.extend(["-o", job_output_dir])
    cmd.extend(["--format", output_format])

    # Enqueue Job
    jobs[job_id] = _build_job_state(
        job_id,
        output_dir=job_output_dir,
        source_type=source_type,
        source_url=url,
        input_path=input_path,
        input_filename=input_filename,
        status="queued",
    )
    jobs[job_id].update({
        'cmd': cmd,
        'env': env,
    })
    _append_log(job_id, f"Job {job_id} queued.", category="queue", important=True)

    await job_queue.put(job_id)

    return {"job_id": job_id, "status": "queued"}

@app.get("/api/status/{job_id}")
async def get_status(job_id: str):
    job = _get_job(job_id)
    if job:
        if not job.get("result"):
            _refresh_job_result(job_id, job.get("output_dir", os.path.join(OUTPUT_DIR, job_id)))
        return _build_status_payload(job)

    tombstone = _get_job_tombstone(job_id)
    if tombstone:
        return JSONResponse(status_code=410, content=tombstone)

    raise HTTPException(status_code=404, detail="Job not found")


def _build_support_log_text(job: dict) -> str:
    lines = [
        f"Job ID: {job.get('job_id')}",
        f"Status: {job.get('status')}",
        f"Phase: {job.get('phase_label') or job.get('phase')}",
        f"Progress: {job.get('progress_percent')}",
        f"ETA Seconds: {job.get('eta_seconds')}",
        f"Elapsed Seconds: {job.get('elapsed_seconds')}",
        f"Last Heartbeat Age: {job.get('seconds_since_heartbeat')}",
        f"Stall State: {job.get('stall_state')}",
        f"Attempt: {job.get('attempt')}",
        f"Resume Count: {job.get('resume_count')}",
        f"Video Duration Seconds: {job.get('video_duration_seconds')}",
        f"Source Type: {job.get('source_type')}",
        f"Source URL: {job.get('source_url') or ''}",
        f"Processing Mode: {job.get('processing_mode') or ''}",
        f"Analysis Status: {job.get('analysis_status') or ''}",
        f"Error Summary: {job.get('error_summary') or ''}",
        "Warnings:",
    ]
    warnings = job.get("warnings") or []
    if warnings:
        lines.extend(f"- {warning}" for warning in warnings[-20:])
    else:
        lines.append("- none")

    lines.append("Artifacts:")
    artifacts = job.get("artifacts") or {}
    if artifacts:
        lines.extend(f"- {kind}: {path}" for kind, path in sorted(artifacts.items()))
    else:
        lines.append("- none")

    lines.append("Important Logs:")
    important_logs = job.get("important_logs") or []
    if important_logs:
        for entry in important_logs[-200:]:
            lines.append(f"[{entry.get('iso_timestamp')}] [{entry.get('level')}] {entry.get('message')}")
    else:
        lines.append("- none")
    return "\n".join(lines)


@app.get("/api/jobs/{job_id}/support-log")
async def get_support_log(job_id: str):
    job = _get_job(job_id)
    if not job:
        tombstone = _get_job_tombstone(job_id)
        if tombstone:
            return JSONResponse(
                status_code=410,
                content={
                    "job_id": job_id,
                    "status": "archived",
                    "support_log": f"Job {job_id} was already archived.",
                },
            )
        raise HTTPException(status_code=404, detail="Job not found")

    payload = _build_status_payload(job)
    return {
        "job_id": job_id,
        "status": job.get("status"),
        "support_log": _build_support_log_text(payload),
    }


@app.post("/api/jobs/{job_id}/cancel")
async def cancel_job(job_id: str):
    """Cancel a queued/running job: flag it, kill its worker subprocess, free the slot."""
    job = jobs.get(job_id)
    if not job:
        # Not live in memory: exists on disk => already finished; otherwise unknown.
        if _get_job(job_id):
            return {"success": False, "detail": "already finished"}
        raise HTTPException(status_code=404, detail="Job not found")

    if job.get("status") in TERMINAL_JOB_STATUSES:
        return {"success": False, "detail": "already finished"}

    job["cancel_requested"] = True
    # Killing the registered subprocess unblocks run_job's wait loop; its wrapper's
    # finally then releases the concurrency semaphore slot, so no slot is leaked.
    await asyncio.get_running_loop().run_in_executor(None, _terminate_job_processes, job_id)
    _mark_job_status(job_id, "failed", error_summary="Cancelled by user", resumable=False)
    _append_log(job_id, "Job cancelled by user.", level="warning", category="cancel", important=True)
    return {"success": True}


@app.post("/api/jobs/{job_id}/resume")
async def resume_job(job_id: str, request: Request, body: Optional[ResumeRequest] = None):
    api_key = request.headers.get("X-Gemini-Key")
    if not api_key:
        raise HTTPException(status_code=400, detail="Missing X-Gemini-Key header")

    job = _get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.get("status") in ACTIVE_JOB_STATUSES:
        raise HTTPException(status_code=409, detail="Job is already active")
    if not job.get("is_resumable") and job.get("status") != "stalled":
        raise HTTPException(status_code=409, detail="Job is not resumable")

    output_dir = job.get("output_dir") or os.path.join(OUTPUT_DIR, job_id)
    if not os.path.isdir(output_dir):
        raise HTTPException(status_code=410, detail="Job artifacts are no longer available")

    cmd = [sys.executable, "-u", "main.py", "--resume-dir", output_dir, "--job-id", job_id]
    if body and body.phase:
        cmd.extend(["--resume-phase", body.phase])

    env = os.environ.copy()
    env["GEMINI_API_KEY"] = api_key
    job["cmd"] = cmd
    job["env"] = env
    job["status"] = "queued"
    job["phase"] = "queued"
    job["phase_label"] = "Queued"
    job["progress_percent"] = min(float(job.get("progress_percent") or 0.0), 99.0)
    job["phase_progress_percent"] = 0.0
    job["attempt"] = 0
    job["stall_state"] = "healthy"
    job["error_summary"] = None
    job["warnings"] = []
    job["resume_count"] = int(job.get("resume_count") or 0) + 1
    job["updated_at"] = _now_ts()
    job["last_heartbeat_at"] = job["updated_at"]
    # Restart the elapsed clock: counting from the original start (incl. a
    # possible multi-hour freeze) makes runtime and ETA meaningless.
    job["started_at"] = job["updated_at"]
    job["is_resumable"] = True
    _append_log(job_id, "Job re-queued for resume.", category="resume", important=True)
    await job_queue.put(job_id)
    return {"job_id": job_id, "status": "queued", "resume_count": job["resume_count"]}

from editor import VideoEditor
from subtitles import generate_srt, generate_ass, burn_subtitles, generate_srt_from_video
from hooks import add_hook_to_video
from translate import translate_video, get_supported_languages
from thumbnail import analyze_video_for_titles, refine_titles, generate_thumbnail, generate_youtube_description

class EditRequest(BaseModel):
    job_id: str
    clip_index: int
    api_key: Optional[str] = None
    input_filename: Optional[str] = None

@app.post("/api/edit")
async def edit_clip(
    req: EditRequest,
    x_gemini_key: Optional[str] = Header(None, alias="X-Gemini-Key")
):
    # Determine API Key
    final_api_key = req.api_key or x_gemini_key or os.environ.get("GEMINI_API_KEY")
    
    if not final_api_key:
        raise HTTPException(status_code=400, detail="Missing Gemini API Key (Header or Body)")

    if req.job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = jobs[req.job_id]
    if 'result' not in job or 'clips' not in job['result']:
        raise HTTPException(status_code=400, detail="Job result not available")
        
    try:
        # Resolve Input Path: Prefer explict input_filename from frontend (chaining edits)
        if req.input_filename:
            # Security: Ensure just a filename, no paths
            safe_name = os.path.basename(req.input_filename)
            input_path = os.path.join(OUTPUT_DIR, req.job_id, safe_name)
            filename = safe_name
        else:
            # Fallback to original clip
            clip = job['result']['clips'][req.clip_index]
            filename = clip['video_url'].split('/')[-1]
            input_path = os.path.join(OUTPUT_DIR, req.job_id, filename)
        
        if not os.path.exists(input_path):
             raise HTTPException(status_code=404, detail=f"Video file not found: {input_path}")

        # Define output path for edited video
        edited_filename = f"edited_{filename}"
        output_path = os.path.join(OUTPUT_DIR, req.job_id, edited_filename)
        
        # Run editing in a thread to avoid blocking main loop
        # Since VideoEditor uses blocking calls (subprocess, API wait)
        def run_edit():
            editor = VideoEditor(api_key=final_api_key)
            
            # SAFE FILE RENAMING STRATEGY (Avoid UnicodeEncodeError in Docker)
            # Create a safe ASCII filename in the same directory
            safe_filename = f"temp_input_{req.job_id}.mp4"
            safe_input_path = os.path.join(OUTPUT_DIR, req.job_id, safe_filename)
            
            # Copy original file to safe path
            # (Copy is safer than rename if something crashes, we keep original)
            shutil.copy(input_path, safe_input_path)
            
            try:
                # 1. Upload (using safe path)
                vid_file = editor.upload_video(safe_input_path)
                
                # 2. Get duration
                import cv2
                cap = cv2.VideoCapture(safe_input_path)
                fps = cap.get(cv2.CAP_PROP_FPS)
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                duration = frame_count / fps if fps else 0
                cap.release()
                
                # Load transcript from metadata
                transcript = None
                try:
                    meta_files = glob.glob(os.path.join(OUTPUT_DIR, req.job_id, "*_metadata.json"))
                    if meta_files:
                        with open(meta_files[0], 'r', encoding='utf-8') as f:
                            data = json.load(f)
                            transcript = _load_transcript_for_job(os.path.join(OUTPUT_DIR, req.job_id), metadata=data)
                except Exception as e:
                    print(f"⚠️ Could not load transcript for editing context: {e}")

                # 3. Get Plan (Filter String)
                # Burned-in captions/hooks must survive the edit: zooming would
                # crop or shift them, so tell the editor to avoid zoom effects.
                has_captions = ("subtitled_" in filename) or ("hooked_" in filename)
                filter_data = editor.get_ffmpeg_filter(vid_file, duration, fps=fps, width=width, height=height, transcript=transcript, has_captions=has_captions)
                
                # 4. Apply
                # Use safe output name first
                safe_output_path = os.path.join(OUTPUT_DIR, req.job_id, f"temp_output_{req.job_id}.mp4")
                editor.apply_edits(safe_input_path, safe_output_path, filter_data)
                
                # Move result to final destination (rename works even if dest name has unicode if filesystem supports it, 
                # but python might still struggle if locale is broken? No, os.rename usually handles it better than subprocess args)
                # Actually, output_path is defined above: f"edited_{filename}"
                # If filename has unicode, output_path has unicode.
                # Let's hope shutil.move / os.rename works.
                if os.path.exists(safe_output_path):
                    shutil.move(safe_output_path, output_path)
                
                return filter_data
            finally:
                # Cleanup temp safe input
                if os.path.exists(safe_input_path):
                    os.remove(safe_input_path)

        # Run in thread pool
        loop = asyncio.get_event_loop()
        plan = await loop.run_in_executor(None, run_edit)
        
        # Update clip URL in the job result? 
        # Or return new URL and let frontend handle it?
        # Updating job result allows persistence if page refreshes.
        
        new_video_url = f"/videos/{req.job_id}/{edited_filename}"
        
        # Start a new "edited" clip entry or just update the current one?
        # Let's update the current one's video_url but keep backup?
        # Or return the new URL to the frontend to display.
        
        return {
            "success": True, 
            "new_video_url": new_video_url,
            "edit_plan": plan
        }

    except Exception as e:
        print(f"❌ Edit Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

class SubtitleRequest(BaseModel):
    job_id: str
    clip_index: int
    position: str = "bottom" # top, middle, bottom
    font_size: int = 16
    font_name: str = "Verdana"
    font_color: str = "#FFFFFF"
    border_color: str = "#000000"
    border_width: int = 2
    bg_color: str = "#000000"
    bg_opacity: float = 0.0
    style: str = "classic"  # classic (uniform color) or karaoke (word highlight)
    highlight_color: str = "#FFD700"
    effect: str = "none"  # none | glow | pop | box (karaoke only)
    base_opacity: float = 1.0  # opacity of non-active words (dimmed modern look)
    uppercase: bool = False
    input_filename: Optional[str] = None

@app.post("/api/subtitle")
async def add_subtitles(req: SubtitleRequest):
    if req.job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    # Reload job data from disk just in case metadata was updated
    job = jobs[req.job_id]
    
    # We need to access metadata.json to get the transcript
    output_dir = os.path.join(OUTPUT_DIR, req.job_id)
    json_files = glob.glob(os.path.join(output_dir, "*_metadata.json"))
    
    if not json_files:
        raise HTTPException(status_code=404, detail="Metadata not found")
        
    with open(json_files[0], 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    transcript = _load_transcript_for_job(output_dir, metadata=data)
    if not transcript:
        raise HTTPException(status_code=400, detail="Transcript not found in metadata. Please process a new video.")
    data['transcript'] = transcript
        
    clips = data.get('shorts', [])
    if req.clip_index >= len(clips):
        raise HTTPException(status_code=404, detail="Clip not found")
        
    clip_data = clips[req.clip_index]
    
    # Video Path
    if req.input_filename:
        # Use chained file
        filename = os.path.basename(req.input_filename)
    else:
        # Fallback to standard naming
        filename = clip_data.get('video_url', '').split('/')[-1]
        if not filename:
             base_name = os.path.basename(json_files[0]).replace('_metadata.json', '')
             filename = f"{base_name}_clip_{req.clip_index+1}.mp4"

    # Re-subtitling must replace previous subtitles instead of burning over
    # them — in BOTH paths (bulk picks the file itself, the single-clip modal
    # sends its current, possibly already-subtitled file explicitly): walk
    # subtitled_<ts>_ prefixes back to the pre-subtitle file.
    while True:
        m = re.match(r'^subtitled_\d+_(.+)$', filename)
        if not m or not os.path.exists(os.path.join(output_dir, m.group(1))):
            break
        filename = m.group(1)
         
    input_path = os.path.join(output_dir, filename)
    if not os.path.exists(input_path):
        # Try looking for edited version if url implied it?
        # Just fail if not found.
        raise HTTPException(status_code=404, detail=f"Video file not found: {input_path}")
        
    # Define outputs
    generation_id = int(time.time())
    is_karaoke = req.style == "karaoke"
    srt_filename = f"subs_{req.clip_index}_{generation_id}.{'ass' if is_karaoke else 'srt'}"
    srt_path = os.path.join(output_dir, srt_filename)

    # Style options shared by the karaoke ASS generator paths.
    karaoke_opts = dict(
        alignment=req.position, fontsize=req.font_size, font_name=req.font_name,
        font_color=req.font_color, border_color=req.border_color,
        border_width=req.border_width, highlight_color=req.highlight_color,
        bg_color=req.bg_color, bg_opacity=req.bg_opacity,
        effect=req.effect, base_opacity=req.base_opacity, uppercase=req.uppercase,
    )

    # Output video
    # We create a new file "subtitled_..."
    output_filename = f"subtitled_{generation_id}_{filename}"
    output_path = os.path.join(output_dir, output_filename)

    try:
        # 1. Generate subtitle file (SRT, or karaoke ASS with word highlight)
        # Check if this is a dubbed video - if so, transcribe it fresh
        is_dubbed = filename.startswith("translated_")

        if is_dubbed:
            print(f"🎙️ Dubbed video detected, transcribing audio for subtitles...")
            def run_transcribe_srt():
                if is_karaoke:
                    return generate_srt_from_video(input_path, srt_path, style="karaoke", **karaoke_opts)
                return generate_srt_from_video(input_path, srt_path)

            loop = asyncio.get_event_loop()
            success = await loop.run_in_executor(None, run_transcribe_srt)
        elif is_karaoke:
            success = generate_ass(transcript, clip_data['start'], clip_data['end'], srt_path, **karaoke_opts)
        else:
            success = generate_srt(transcript, clip_data['start'], clip_data['end'], srt_path)

        if not success:
             raise HTTPException(status_code=400, detail="No words found for this clip range.")

        # 2. Burn Subtitles
        # Run in thread pool
        def run_burn():
             burn_subtitles(input_path, srt_path, output_path,
                           alignment=req.position, fontsize=req.font_size,
                           font_name=req.font_name, font_color=req.font_color,
                           border_color=req.border_color, border_width=req.border_width,
                           bg_color=req.bg_color, bg_opacity=req.bg_opacity)
        
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, run_burn)
        
    except Exception as e:
        print(f"❌ Subtitle Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
        
    # 3. Update Result and Metadata
    # Update InMemory Jobs
    if req.clip_index < len(job['result']['clips']):
         job['result']['clips'][req.clip_index]['video_url'] = f"/videos/{req.job_id}/{output_filename}"
    
    # Update Metadata on Disk (Persistence)
    try:
        if req.clip_index < len(clips):
            clips[req.clip_index]['video_url'] = f"/videos/{req.job_id}/{output_filename}"
            # Update the main data structure
            data['shorts'] = clips
            
            # Write back
            with open(json_files[0], 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=4, ensure_ascii=False)
                print(f"✅ Metadata updated with subtitled video for clip {req.clip_index}")
    except Exception as e:
        print(f"⚠️ Failed to update metadata.json: {e}")
        # Non-critical, but good for persistence

    return {
        "success": True,
        "new_video_url": f"/videos/{req.job_id}/{output_filename}"
    }


@app.get("/api/jobs/{job_id}/download-all")
async def download_all_clips(job_id: str):
    """Bundle the current version of every clip of a job into one ZIP."""
    output_dir = os.path.join(OUTPUT_DIR, job_id)
    json_files = glob.glob(os.path.join(output_dir, "*_metadata.json"))
    if not json_files:
        raise HTTPException(status_code=404, detail="Job not found")

    with open(json_files[0], 'r', encoding='utf-8') as f:
        data = json.load(f)

    files = []
    for i, clip in enumerate(data.get('shorts', [])):
        filename = os.path.basename(clip.get('video_url', '').split('/')[-1])
        path = os.path.join(output_dir, filename)
        if filename and os.path.exists(path):
            files.append((i, path))

    if not files:
        raise HTTPException(status_code=404, detail="No clip files found for this job")

    zip_path = os.path.join(output_dir, f"clips_{int(time.time())}.zip")

    def build_zip():
        # Videos are already compressed; store instead of deflate for speed.
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_STORED) as zf:
            for i, path in files:
                zf.write(path, arcname=f"clip_{i + 1:02d}_{os.path.basename(path)}")

    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, build_zip)

    return FileResponse(
        zip_path,
        media_type="application/zip",
        filename=f"openshorts_clips_{job_id[:8]}.zip",
        background=BackgroundTask(os.remove, zip_path),
    )


class HookRequest(BaseModel):
    job_id: str
    clip_index: int
    text: str
    input_filename: Optional[str] = None
    position: Optional[str] = "top" # top, center, bottom
    size: Optional[str] = "M" # S, M, L

@app.post("/api/hook")
async def add_hook(req: HookRequest):
    if req.job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = jobs[req.job_id]
    output_dir = os.path.join(OUTPUT_DIR, req.job_id)
    json_files = glob.glob(os.path.join(output_dir, "*_metadata.json"))
    
    if not json_files:
        raise HTTPException(status_code=404, detail="Metadata not found")
        
    with open(json_files[0], 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    clips = data.get('shorts', [])
    if req.clip_index >= len(clips):
        raise HTTPException(status_code=404, detail="Clip not found")
        
    clip_data = clips[req.clip_index]
    
    # Video Path
    if req.input_filename:
        filename = os.path.basename(req.input_filename)
    else:
        filename = clip_data.get('video_url', '').split('/')[-1]
        if not filename:
             base_name = os.path.basename(json_files[0]).replace('_metadata.json', '')
             filename = f"{base_name}_clip_{req.clip_index+1}.mp4"
         
    input_path = os.path.join(output_dir, filename)
    if not os.path.exists(input_path):
        raise HTTPException(status_code=404, detail=f"Video file not found: {input_path}")
        
    # Output video
    output_filename = f"hook_{filename}"
    output_path = os.path.join(output_dir, output_filename)
    
    # Map Size to Scale
    size_map = {"S": 0.8, "M": 1.0, "L": 1.3}
    font_scale = size_map.get(req.size, 1.0)
    
    try:
        # Run in thread pool
        def run_hook():
             add_hook_to_video(input_path, req.text, output_path, position=req.position, font_scale=font_scale)
        
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, run_hook)
        
    except Exception as e:
        print(f"❌ Hook Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
        
    # Update Persistence (Same logic as subtitles)
    # Update InMemory Jobs
    if req.clip_index < len(job['result']['clips']):
         job['result']['clips'][req.clip_index]['video_url'] = f"/videos/{req.job_id}/{output_filename}"
    
    # Update Metadata on Disk
    try:
        if req.clip_index < len(clips):
            clips[req.clip_index]['video_url'] = f"/videos/{req.job_id}/{output_filename}"
            data['shorts'] = clips
            with open(json_files[0], 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=4, ensure_ascii=False)
                print(f"✅ Metadata updated with hook video for clip {req.clip_index}")
    except Exception as e:
        print(f"⚠️ Failed to update metadata.json: {e}")

    return {
        "success": True,
        "new_video_url": f"/videos/{req.job_id}/{output_filename}"
    }

class TranslateRequest(BaseModel):
    job_id: str
    clip_index: int
    target_language: str
    source_language: Optional[str] = None
    input_filename: Optional[str] = None

@app.get("/api/translate/languages")
async def get_languages():
    """Return supported languages for translation."""
    return {"languages": get_supported_languages()}

@app.post("/api/translate")
async def translate_clip(
    req: TranslateRequest,
    x_elevenlabs_key: Optional[str] = Header(None, alias="X-ElevenLabs-Key")
):
    """Translate a video clip to a different language using ElevenLabs dubbing."""
    if not x_elevenlabs_key:
        raise HTTPException(status_code=400, detail="Missing X-ElevenLabs-Key header")

    if req.job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    job = jobs[req.job_id]
    output_dir = os.path.join(OUTPUT_DIR, req.job_id)
    json_files = glob.glob(os.path.join(output_dir, "*_metadata.json"))

    if not json_files:
        raise HTTPException(status_code=404, detail="Metadata not found")

    with open(json_files[0], 'r', encoding='utf-8') as f:
        data = json.load(f)

    clips = data.get('shorts', [])
    if req.clip_index >= len(clips):
        raise HTTPException(status_code=404, detail="Clip not found")

    clip_data = clips[req.clip_index]

    # Video Path
    if req.input_filename:
        filename = os.path.basename(req.input_filename)
    else:
        filename = clip_data.get('video_url', '').split('/')[-1]
        if not filename:
             base_name = os.path.basename(json_files[0]).replace('_metadata.json', '')
             filename = f"{base_name}_clip_{req.clip_index+1}.mp4"

    input_path = os.path.join(output_dir, filename)
    if not os.path.exists(input_path):
        raise HTTPException(status_code=404, detail=f"Video file not found: {input_path}")

    # Output video with language suffix
    base, ext = os.path.splitext(filename)
    output_filename = f"translated_{req.target_language}_{base}{ext}"
    output_path = os.path.join(output_dir, output_filename)

    try:
        # Run translation in thread pool (blocking API calls)
        def run_translate():
            return translate_video(
                video_path=input_path,
                output_path=output_path,
                target_language=req.target_language,
                api_key=x_elevenlabs_key,
                source_language=req.source_language,
            )

        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, run_translate)

    except Exception as e:
        print(f"❌ Translation Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    # Update InMemory Jobs
    if req.clip_index < len(job['result']['clips']):
         job['result']['clips'][req.clip_index]['video_url'] = f"/videos/{req.job_id}/{output_filename}"

    # Update Metadata on Disk
    try:
        if req.clip_index < len(clips):
            clips[req.clip_index]['video_url'] = f"/videos/{req.job_id}/{output_filename}"
            data['shorts'] = clips
            with open(json_files[0], 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=4, ensure_ascii=False)
                print(f"✅ Metadata updated with translated video for clip {req.clip_index}")
    except Exception as e:
        print(f"⚠️ Failed to update metadata.json: {e}")

    return {
        "success": True,
        "new_video_url": f"/videos/{req.job_id}/{output_filename}"
    }

class SocialPostRequest(BaseModel):
    job_id: str
    clip_index: int
    api_key: str
    user_id: str
    platforms: List[str] # ["tiktok", "instagram", "youtube"]
    # Optional overrides if frontend wants to edit them
    title: Optional[str] = None
    description: Optional[str] = None
    scheduled_date: Optional[str] = None # ISO-8601 string
    timezone: Optional[str] = "UTC"

import httpx

@app.post("/api/social/post")
async def post_to_socials(req: SocialPostRequest):
    if req.job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = jobs[req.job_id]
    if 'result' not in job or 'clips' not in job['result']:
        raise HTTPException(status_code=400, detail="Job result not available")
        
    try:
        clip = job['result']['clips'][req.clip_index]
        # Video URL is relative /videos/..., we need absolute file path
        # clip['video_url'] is like "/videos/{job_id}/{filename}"
        # We constructed it as: f"/videos/{job_id}/{clip_filename}"
        # And file is at f"{OUTPUT_DIR}/{job_id}/{clip_filename}"
        
        filename = clip['video_url'].split('/')[-1]
        file_path = os.path.join(OUTPUT_DIR, req.job_id, filename)
        
        if not os.path.exists(file_path):
             raise HTTPException(status_code=404, detail=f"Video file not found: {file_path}")

        # Construct parameters for Upload-Post API
        # Fallbacks
        final_title = req.title or clip.get('title', 'Viral Short')
        final_description = req.description or clip.get('video_description_for_instagram') or clip.get('video_description_for_tiktok') or "Check this out!"
        
        # Prepare form data
        url = "https://api.upload-post.com/api/upload"
        headers = {
            "Authorization": f"Apikey {req.api_key}"
        }
        
        # Prepare data as dict (httpx handles lists for multiple values)
        data_payload = {
            "user": req.user_id,
            "title": final_title,
            "platform[]": req.platforms, # Pass list directly
            "async_upload": "true"  # Enable async upload
        }

        # Add scheduling if present
        if req.scheduled_date:
            data_payload["scheduled_date"] = req.scheduled_date
            if req.timezone:
                data_payload["timezone"] = req.timezone
        
        # Add Platform specifics
        if "tiktok" in req.platforms:
             data_payload["tiktok_title"] = final_description
             
        if "instagram" in req.platforms:
             data_payload["instagram_title"] = final_description
             data_payload["media_type"] = "REELS"

        if "youtube" in req.platforms:
             yt_title = req.title or clip.get('video_title_for_youtube_short', final_title)
             data_payload["youtube_title"] = yt_title
             data_payload["youtube_description"] = final_description
             data_payload["privacyStatus"] = "public"

        # Send File. Stream the open file handle (no full read into RAM) and run the
        # blocking multipart upload off the event loop so one post can't freeze the server.
        def _do_upload():
            with open(file_path, "rb") as fh:
                files = {"video": (filename, fh, "video/mp4")}
                with httpx.Client(timeout=120.0) as client:
                    print(f"📡 Sending to Upload-Post for platforms: {req.platforms}")
                    return client.post(url, headers=headers, data=data_payload, files=files)

        response = await asyncio.get_running_loop().run_in_executor(None, _do_upload)

        if response.status_code not in [200, 201, 202]: # Added 201
             print(f"❌ Upload-Post Error: {response.text}")
             raise HTTPException(status_code=response.status_code, detail=f"Vendor API Error: {response.text}")

        return response.json()

    except Exception as e:
        print(f"❌ Social Post Exception: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/social/user")
async def get_social_user(api_key: str = Header(..., alias="X-Upload-Post-Key")):
    """Proxy to fetch user ID from Upload-Post"""
    if not api_key:
         raise HTTPException(status_code=400, detail="Missing X-Upload-Post-Key header")
         
    url = "https://api.upload-post.com/api/uploadposts/users"
    print(f"🔍 Fetching User ID from: {url}")
    headers = {"Authorization": f"Apikey {api_key}"}
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            resp = await client.get(url, headers=headers)
            if resp.status_code != 200:
                print(f"❌ Upload-Post User Fetch Error: {resp.text}")
                raise HTTPException(status_code=resp.status_code, detail=f"Failed to fetch user: {resp.text}")
            
            data = resp.json()
            print(f"🔍 Upload-Post User Response: {data}")
            
            user_id = None
            # The structure is {'success': True, 'profiles': [{'username': '...'}, ...]}
            profiles_list = []
            if isinstance(data, dict):
                 raw_profiles = data.get('profiles', [])
                 if isinstance(raw_profiles, list):
                     for p in raw_profiles:
                         username = p.get('username')
                         if username:
                             # Determine connected platforms
                             socials = p.get('social_accounts', {})
                             connected = []
                             # Check typical platforms
                             for platform in ['tiktok', 'instagram', 'youtube']:
                                 account_info = socials.get(platform)
                                 # If it's a dict and typically has data, or just not empty string
                                 if isinstance(account_info, dict):
                                     connected.append(platform)
                             
                             profiles_list.append({
                                 "username": username,
                                 "connected": connected
                             })
            
            if not profiles_list:
                # Fallback if no profiles found
                return {"profiles": [], "error": "No profiles found"}
                
            return {"profiles": profiles_list}
            
            
        except Exception as e:
             raise HTTPException(status_code=500, detail=str(e))

# --- Thumbnail Studio Endpoints ---

@app.post("/api/thumbnail/upload")
async def thumbnail_upload(
    file: Optional[UploadFile] = File(None),
    url: Optional[str] = Form(None),
):
    """Upload video and start background Whisper transcription immediately."""
    if not url and not file:
        raise HTTPException(status_code=400, detail="Must provide URL or File")

    session_id = str(uuid.uuid4())
    transcript_event = asyncio.Event()

    # Save file if uploaded directly
    video_path = None
    if file:
        video_path = os.path.join(UPLOAD_DIR, f"thumb_{session_id}_{file.filename}")
        await _save_upload_with_limit(file, video_path)

    # Initialize session
    thumbnail_sessions[session_id] = {
        "created_at": _now_ts(),
        "video_path": video_path,
        "transcript_event": transcript_event,
        "transcript_ready": False,
        "transcript": None,
        "transcript_segments": [],
        "video_duration": 0,
        "language": "en",
        "context": "",
        "titles": [],
        "conversation": [],
        "_url": url,  # Store URL for deferred download
    }

    async def run_background_whisper():
        try:
            vpath = video_path
            # Download YouTube video if URL was provided
            if not vpath and url:
                from main import download_youtube_video
                loop = asyncio.get_event_loop()
                vpath, _ = await loop.run_in_executor(None, download_youtube_video, url, UPLOAD_DIR)
                thumbnail_sessions[session_id]["video_path"] = vpath

            from main import transcribe_video
            loop = asyncio.get_event_loop()
            transcript = await loop.run_in_executor(None, transcribe_video, vpath)
            segments = transcript.get("segments", [])
            duration = segments[-1]["end"] if segments else 0

            thumbnail_sessions[session_id].update({
                "transcript_ready": True,
                "transcript": transcript,
                "transcript_segments": segments,
                "video_duration": duration,
                "language": transcript.get("language", "en"),
            })
            print(f"✅ [Thumbnail] Background Whisper complete for session {session_id}")
        except Exception as e:
            print(f"❌ [Thumbnail] Background Whisper failed: {e}")
            thumbnail_sessions[session_id]["transcript_error"] = str(e)
        finally:
            transcript_event.set()

    asyncio.create_task(run_background_whisper())

    return {"session_id": session_id}


@app.post("/api/thumbnail/analyze")
async def thumbnail_analyze(
    request: Request,
    file: Optional[UploadFile] = File(None),
    url: Optional[str] = Form(None),
    session_id: Optional[str] = Form(None),
    x_gemini_key: Optional[str] = Header(None, alias="X-Gemini-Key")
):
    """Analyze a video and suggest viral YouTube titles."""
    api_key = x_gemini_key
    if not api_key:
        raise HTTPException(status_code=400, detail="Missing X-Gemini-Key header")

    pre_transcript = None

    # Check for pre-existing session with background Whisper
    if session_id and session_id in thumbnail_sessions:
        session = thumbnail_sessions[session_id]

        # Wait for background Whisper to complete
        transcript_event = session.get("transcript_event")
        if transcript_event:
            print(f"⏳ [Thumbnail] Waiting for background Whisper to finish...")
            await transcript_event.wait()

        if session.get("transcript_error"):
            raise HTTPException(status_code=500, detail=f"Transcription failed: {session['transcript_error']}")

        video_path = session["video_path"]
        if not video_path or not os.path.exists(video_path):
            raise HTTPException(status_code=404, detail="Video file not found in session")

        if session.get("transcript_ready"):
            pre_transcript = session["transcript"]
    else:
        # No pre-existing session — need file or URL
        if not url and not file:
            raise HTTPException(status_code=400, detail="Must provide URL, File, or session_id")

        session_id = str(uuid.uuid4())

        if url:
            from main import download_youtube_video
            video_path, _ = download_youtube_video(url, UPLOAD_DIR)
        else:
            video_path = os.path.join(UPLOAD_DIR, f"thumb_{session_id}_{file.filename}")
            with open(video_path, "wb") as buffer:
                content = await file.read()
                buffer.write(content)

    try:
        # Run analysis in thread pool (skips Whisper if pre_transcript is available)
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, analyze_video_for_titles, api_key, video_path, pre_transcript)

        # Store/update session context
        if session_id not in thumbnail_sessions:
            thumbnail_sessions[session_id] = {"created_at": _now_ts()}

        thumbnail_sessions[session_id].update({
            "context": result.get("transcript_summary", ""),
            "titles": result.get("titles", []),
            "language": result.get("language", "en"),
            "conversation": thumbnail_sessions[session_id].get("conversation", []),
            "video_path": video_path,
            "transcript_segments": result.get("segments", []),
            "video_duration": result.get("video_duration", 0)
        })

        return {
            "session_id": session_id,
            "titles": result.get("titles", []),
            "context": result.get("transcript_summary", ""),
            "language": result.get("language", "en"),
            "recommended": result.get("recommended", [])
        }

    except Exception as e:
        print(f"❌ Thumbnail Analyze Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


class ThumbnailTitlesRequest(BaseModel):
    session_id: Optional[str] = None
    message: Optional[str] = None
    title: Optional[str] = None

@app.post("/api/thumbnail/titles")
async def thumbnail_titles(
    req: ThumbnailTitlesRequest,
    x_gemini_key: Optional[str] = Header(None, alias="X-Gemini-Key")
):
    """Refine title suggestions or accept a manual title."""
    api_key = x_gemini_key
    if not api_key:
        raise HTTPException(status_code=400, detail="Missing X-Gemini-Key header")

    # Manual title mode - just create a session with the user's title
    if req.title:
        session_id = req.session_id or str(uuid.uuid4())
        if session_id not in thumbnail_sessions:
            thumbnail_sessions[session_id] = {
                "created_at": _now_ts(),
                "context": "",
                "titles": [req.title],
                "language": "en",
                "conversation": []
            }
        return {"session_id": session_id, "titles": [req.title]}

    # Refinement mode
    if not req.session_id or req.session_id not in thumbnail_sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    if not req.message:
        raise HTTPException(status_code=400, detail="Must provide message or title")

    session = thumbnail_sessions[req.session_id]

    # Add user message to conversation history
    session["conversation"].append({"role": "user", "content": req.message})

    try:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            refine_titles,
            api_key,
            session["context"],
            req.message,
            session["conversation"]
        )

        new_titles = result.get("titles", [])
        session["titles"] = new_titles
        session["conversation"].append({"role": "assistant", "content": json.dumps(new_titles)})

        return {"titles": new_titles}

    except Exception as e:
        print(f"❌ Thumbnail Titles Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/thumbnail/generate")
async def thumbnail_generate(
    request: Request,
    session_id: str = Form(...),
    title: str = Form(...),
    extra_prompt: str = Form(""),
    count: int = Form(3),
    face: Optional[UploadFile] = File(None),
    background: Optional[UploadFile] = File(None),
    x_gemini_key: Optional[str] = Header(None, alias="X-Gemini-Key")
):
    """Generate YouTube thumbnails with Gemini image generation."""
    api_key = x_gemini_key
    if not api_key:
        raise HTTPException(status_code=400, detail="Missing X-Gemini-Key header")

    # Clamp count
    count = min(max(1, count), 6)

    # Save optional uploaded images
    face_path = None
    bg_path = None
    thumb_upload_dir = os.path.join(UPLOAD_DIR, f"thumb_{session_id}")
    os.makedirs(thumb_upload_dir, exist_ok=True)

    try:
        if face and face.filename:
            face_path = os.path.join(thumb_upload_dir, f"face_{face.filename}")
            with open(face_path, "wb") as f:
                f.write(await face.read())

        if background and background.filename:
            bg_path = os.path.join(thumb_upload_dir, f"bg_{background.filename}")
            with open(bg_path, "wb") as f:
                f.write(await background.read())

        # Get video context from session (transcript summary from analysis step)
        video_context = ""
        if session_id in thumbnail_sessions:
            video_context = thumbnail_sessions[session_id].get("context", "")

        # Run generation in thread pool
        loop = asyncio.get_event_loop()
        thumbnails = await loop.run_in_executor(
            None,
            generate_thumbnail,
            api_key,
            title,
            session_id,
            face_path,
            bg_path,
            extra_prompt,
            count,
            video_context
        )

        if not thumbnails:
            raise HTTPException(status_code=500, detail="Thumbnail generation failed. Please check your Gemini API key has access to image generation (gemini-3.1-flash-image-preview model).")

        return {"thumbnails": thumbnails}

    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Thumbnail Generate Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


class ThumbnailDescribeRequest(BaseModel):
    session_id: str
    title: str

@app.post("/api/thumbnail/describe")
async def thumbnail_describe(
    req: ThumbnailDescribeRequest,
    x_gemini_key: Optional[str] = Header(None, alias="X-Gemini-Key")
):
    """Generate a YouTube description with chapters from the transcript."""
    api_key = x_gemini_key
    if not api_key:
        raise HTTPException(status_code=400, detail="Missing X-Gemini-Key header")

    if req.session_id not in thumbnail_sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    session = thumbnail_sessions[req.session_id]
    segments = session.get("transcript_segments", [])
    if not segments:
        raise HTTPException(status_code=400, detail="No transcript segments available. Please analyze a video first.")

    try:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            generate_youtube_description,
            api_key,
            req.title,
            segments,
            session.get("language", "en"),
            session.get("video_duration", 0)
        )
        return {"description": result.get("description", "")}

    except Exception as e:
        print(f"❌ Thumbnail Describe Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/thumbnail/publish")
async def thumbnail_publish(
    background_tasks: BackgroundTasks,
    session_id: str = Form(...),
    title: str = Form(...),
    description: str = Form(...),
    thumbnail_url: str = Form(...),
    api_key: str = Form(...),
    user_id: str = Form(...),
):
    """Kick off a background upload to YouTube via Upload-Post and return immediately."""
    if session_id not in thumbnail_sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    session = thumbnail_sessions[session_id]
    video_path = session.get("video_path")
    if not video_path or not os.path.exists(video_path):
        raise HTTPException(status_code=404, detail="Original video file not found")

    # Resolve thumbnail path from URL
    thumb_relative = thumbnail_url.lstrip("/")
    if thumb_relative.startswith("thumbnails/"):
        thumb_path = os.path.join(OUTPUT_DIR, thumb_relative)
    else:
        thumb_path = os.path.join(THUMBNAILS_DIR, thumb_relative)

    if not os.path.exists(thumb_path):
        raise HTTPException(status_code=404, detail=f"Thumbnail file not found: {thumb_path}")

    # Generate a unique ID for this publish job so the frontend can poll
    publish_id = str(uuid.uuid4())
    publish_jobs[publish_id] = {"status": "uploading", "result": None, "error": None, "created_at": _now_ts()}

    def do_upload():
        """Runs in a thread via BackgroundTasks — does the actual multipart upload."""
        try:
            upload_url = "https://api.upload-post.com/api/upload"
            headers = {"Authorization": f"Apikey {api_key}"}
            data_payload = {
                "user": user_id,
                "platform[]": ["youtube"],
                "title": title,          # required base field (fallback)
                "async_upload": "true",
                "youtube_title": title,
                "youtube_description": description,
                "privacyStatus": "public",
            }
            video_filename = os.path.basename(video_path)
            thumb_filename = os.path.basename(thumb_path)

            print(f"📡 [Thumbnail] Publishing to YouTube via Upload-Post... (publish_id={publish_id})")
            # Stream the open handles (video can be up to 2GB) instead of reading into RAM.
            # Use a long timeout — video uploads can take several minutes.
            with open(video_path, "rb") as vf, open(thumb_path, "rb") as tf:
                files = {
                    "video": (video_filename, vf, "video/mp4"),
                    "thumbnail": (thumb_filename, tf, "image/jpeg"),
                }
                with httpx.Client(timeout=600.0) as client:
                    response = client.post(upload_url, headers=headers, data=data_payload, files=files)

            if response.status_code not in [200, 201, 202]:
                err = f"Upload-Post API Error ({response.status_code}): {response.text}"
                print(f"❌ {err}")
                publish_jobs[publish_id]["status"] = "failed"
                publish_jobs[publish_id]["error"] = err
            else:
                print(f"✅ [Thumbnail] Published successfully (publish_id={publish_id})")
                publish_jobs[publish_id]["status"] = "done"
                publish_jobs[publish_id]["result"] = response.json()

        except Exception as e:
            err = str(e)
            print(f"❌ Thumbnail Publish Background Error: {err}")
            publish_jobs[publish_id]["status"] = "failed"
            publish_jobs[publish_id]["error"] = err

    background_tasks.add_task(do_upload)
    return {"publish_id": publish_id, "status": "uploading"}


@app.get("/api/thumbnail/publish/status/{publish_id}")
async def thumbnail_publish_status(publish_id: str):
    """Poll the status of a background publish job."""
    if publish_id not in publish_jobs:
        raise HTTPException(status_code=404, detail="Publish job not found")
    return publish_jobs[publish_id]


# @app.get("/api/gallery/clips")
# async def get_gallery_clips(limit: int = 20, offset: int = 0, refresh: bool = False):
#     """
#     Fetch clips from S3 for the gallery with pagination.
#
#     Args:
#         limit: Number of clips to return (default 20, max 100)
#         offset: Starting position for pagination
#         refresh: Force refresh cache
#     """
#     try:
#         # Clamp limit to reasonable values
#         limit = min(max(1, limit), 100)
#
#         # Get clips (uses cache internally)
#         all_clips = list_all_clips(limit=limit + offset, force_refresh=refresh)
#
#         # Apply offset for pagination
#         clips = all_clips[offset:offset + limit]
#
#         return {
#             "clips": clips,
#             "total": len(all_clips),
#             "limit": limit,
#             "offset": offset,
#             "has_more": len(all_clips) > offset + limit
#         }
#     except Exception as e:
#         print(f"❌ Gallery Error: {e}")
#         raise HTTPException(status_code=500, detail=str(e))


# ═══════════════════════════════════════════════════════════════════════
# SaaSShorts: AI UGC Video Generator for SaaS Products
# ═══════════════════════════════════════════════════════════════════════

from saasshorts import (
    scrape_website,
    research_saas_online,
    analyze_saas,
    generate_scripts,
    generate_full_video,
    generate_actor_images,
    get_elevenlabs_voices,
    DEFAULT_VOICES,
)

# State for SaaSShorts jobs (separate from video processing jobs)
saas_jobs: Dict[str, Dict] = {}


class SaaSAnalyzeRequest(BaseModel):
    url: Optional[str] = None
    description: Optional[str] = None  # Manual product/business description
    num_scripts: int = 3
    style: str = "ugc"
    language: str = "en"
    actor_gender: str = "female"


@app.post("/api/saasshorts/analyze")
async def saasshorts_analyze(
    req: SaaSAnalyzeRequest,
    x_gemini_key: Optional[str] = Header(None, alias="X-Gemini-Key"),
):
    """Analyze a URL or manual description and generate video scripts."""
    gemini_key = x_gemini_key or os.environ.get("GEMINI_API_KEY")
    if not gemini_key:
        raise HTTPException(status_code=400, detail="Missing Gemini API Key")

    if not req.url and not req.description:
        raise HTTPException(status_code=400, detail="Provide a URL or a product description")

    try:
        loop = asyncio.get_event_loop()

        def run_analysis():
            web_research = None

            if req.url and req.url.strip():
                # URL provided: full scrape + research pipeline
                scraped = scrape_website(req.url)
                web_research = research_saas_online(req.url, gemini_key)
                analysis = analyze_saas(scraped, gemini_key, web_research=web_research)
            else:
                # Manual description: build analysis from description
                analysis = {
                    "product_name": req.description.split(",")[0].strip()[:60] if req.description else "Product",
                    "description": req.description,
                    "value_proposition": req.description,
                    "target_audience": "general audience",
                    "key_features": [req.description],
                    "pain_points": [],
                    "tone": "casual and authentic",
                }

            scripts = generate_scripts(analysis, gemini_key, req.num_scripts, req.style, req.language, req.actor_gender)
            return {
                "analysis": analysis,
                "scripts": scripts,
                "web_research": web_research,
            }

        result = await loop.run_in_executor(None, run_analysis)
        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class SaaSActorRequest(BaseModel):
    actor_description: str
    num_options: int = 3
    product_description: Optional[str] = None


@app.post("/api/saasshorts/actor-upload")
async def saasshorts_actor_upload(file: UploadFile = File(...)):
    """Upload a custom actor image (stored locally only, not S3)."""
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    try:
        content = await file.read()

        # Validate minimum size
        if len(content) < 1000:
            raise HTTPException(status_code=400, detail="File too small to be a valid image")

        upload_id = uuid.uuid4().hex[:8]
        upload_dir = os.path.join(OUTPUT_DIR, "actor_uploads")
        os.makedirs(upload_dir, exist_ok=True)
        filename = f"custom_{upload_id}.png"
        file_path = os.path.join(upload_dir, filename)

        with open(file_path, "wb") as f:
            f.write(content)

        return {"url": f"/videos/actor_uploads/{filename}"}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/saasshorts/actor-options")
async def saasshorts_actor_options(
    req: SaaSActorRequest,
    x_fal_key: Optional[str] = Header(None, alias="X-Fal-Key"),
):
    """Generate multiple actor image options for the user to choose from."""
    fal_key = x_fal_key
    if not fal_key:
        raise HTTPException(status_code=400, detail="Missing fal.ai API Key")

    try:
        job_id = str(uuid.uuid4())
        out_dir = os.path.join(OUTPUT_DIR, f"saas_actors_{job_id}")
        os.makedirs(out_dir, exist_ok=True)

        loop = asyncio.get_running_loop()
        import functools
        paths = await loop.run_in_executor(
            None,
            functools.partial(
                generate_actor_images,
                req.actor_description, fal_key, out_dir, "actor", req.num_options,
                product_description=req.product_description,
            ),
        )

        # Upload each actor image to public S3 with description
        desc = req.actor_description
        if req.product_description:
            desc += f" (holding {req.product_description})"
        urls = []
        for p in paths:
            s3_url = upload_actor_to_s3(p, description=desc)
            if s3_url:
                urls.append(s3_url)
            else:
                # Fallback to local URL if S3 fails
                urls.append(f"/videos/saas_actors_{job_id}/{os.path.basename(p)}")

        return {"images": urls}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/saasshorts/gallery")
async def saasshorts_video_gallery(limit: int = 50):
    """List all UGC videos from the public gallery."""
    try:
        loop = asyncio.get_running_loop()
        videos = await loop.run_in_executor(None, list_video_gallery, limit)
        return {"videos": videos, "total": len(videos)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class SaaSPostRequest(BaseModel):
    job_id: str
    api_key: str
    user_id: str
    platforms: List[str]
    title: Optional[str] = None
    description: Optional[str] = None
    scheduled_date: Optional[str] = None
    timezone: Optional[str] = "UTC"


@app.post("/api/saasshorts/post")
async def saasshorts_post_to_socials(req: SaaSPostRequest):
    """Post an AI Shorts video to social media via Upload-Post."""
    if req.job_id not in saas_jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    job = saas_jobs[req.job_id]
    result = job.get("result")
    if not result or not result.get("video_url"):
        raise HTTPException(status_code=400, detail="No video available for this job")

    try:
        # Resolve video file path
        video_url = result["video_url"]  # e.g. /videos/saas_xxx/slug_final.mp4
        rel_path = video_url.replace("/videos/", "")
        file_path = os.path.join(OUTPUT_DIR, rel_path)

        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail=f"Video file not found")

        script = result.get("script", {})
        final_title = req.title or script.get("title", "AI Short")
        final_description = req.description or script.get("caption", "")
        if not final_description:
            final_description = script.get("full_narration", "Check this out!")

        url = "https://api.upload-post.com/api/upload"
        headers = {"Authorization": f"Apikey {req.api_key}"}

        data_payload = {
            "user": req.user_id,
            "title": final_title,
            "platform[]": req.platforms,
            "async_upload": "true",
        }

        if req.scheduled_date:
            data_payload["scheduled_date"] = req.scheduled_date
            if req.timezone:
                data_payload["timezone"] = req.timezone

        if "tiktok" in req.platforms:
            data_payload["tiktok_title"] = final_description
        if "instagram" in req.platforms:
            data_payload["instagram_title"] = final_description
            data_payload["media_type"] = "REELS"
        if "youtube" in req.platforms:
            data_payload["youtube_title"] = final_title
            data_payload["youtube_description"] = final_description
            data_payload["privacyStatus"] = "public"

        filename = os.path.basename(file_path)

        # Stream the open file handle and run the blocking upload off the event loop.
        def _do_upload():
            with open(file_path, "rb") as fh:
                files = {"video": (filename, fh, "video/mp4")}
                with httpx.Client(timeout=120.0) as client:
                    print(f"📡 [AI Shorts] Sending to Upload-Post: {req.platforms}")
                    return client.post(url, headers=headers, data=data_payload, files=files)

        response = await asyncio.get_running_loop().run_in_executor(None, _do_upload)

        if response.status_code not in [200, 201, 202]:
            raise HTTPException(status_code=response.status_code, detail=f"Upload-Post Error: {response.text}")

        return response.json()

    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ [AI Shorts] Post Exception: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/gallery", response_class=HTMLResponse)
async def gallery_html_page():
    """SEO gallery page with all generated UGC videos."""
    import html as html_mod
    loop = asyncio.get_running_loop()
    videos = await loop.run_in_executor(None, list_video_gallery, 100)

    cards_html = ""
    ld_items = []
    for i, v in enumerate(videos):
        title = html_mod.escape(v.get("title", "Untitled"))
        video_url = v.get("video_url", "")
        actor_url = v.get("actor_url", "")
        video_id = v.get("video_id", "")
        duration = v.get("duration", 0)
        mode = v.get("video_mode", "")
        product = html_mod.escape(v.get("product_name", ""))
        caption = html_mod.escape(v.get("caption", "")[:120])

        mode_badge = '<span style="background:#22c55e;color:#000;padding:2px 8px;border-radius:9999px;font-size:10px;font-weight:700">LOW COST</span>' if mode == "lowcost" else '<span style="background:#8b5cf6;color:#fff;padding:2px 8px;border-radius:9999px;font-size:10px;font-weight:700">PREMIUM</span>'

        cards_html += f'''
        <a href="/video/{video_id}" style="text-decoration:none;color:inherit">
          <div style="background:#18181b;border-radius:16px;overflow:hidden;border:1px solid #27272a;transition:transform 0.2s" onmouseover="this.style.transform='scale(1.02)'" onmouseout="this.style.transform='scale(1)'">
            <div style="position:relative;aspect-ratio:9/16;background:#000">
              <video src="{video_url}" poster="{actor_url}" muted playsinline preload="metadata"
                     onmouseenter="this.play()" onmouseleave="this.pause();this.currentTime=0"
                     style="width:100%;height:100%;object-fit:cover"></video>
              <div style="position:absolute;top:8px;right:8px">{mode_badge}</div>
            </div>
            <div style="padding:12px">
              <h2 style="font-size:14px;font-weight:600;margin:0 0 4px 0;white-space:nowrap;overflow:hidden;text-overflow:ellipsis">{title}</h2>
              <p style="font-size:11px;color:#71717a;margin:0">{duration:.0f}s · {product}</p>
            </div>
          </div>
        </a>'''

        ld_items.append(f'{{"@type":"ListItem","position":{i+1},"url":"https://openshorts.app/video/{video_id}","name":"{title}"}}')

    ld_json = f'{{"@context":"https://schema.org","@type":"CollectionPage","name":"AI UGC Video Gallery","mainEntity":{{"@type":"ItemList","numberOfItems":{len(videos)},"itemListElement":[{",".join(ld_items)}]}}}}'

    return f'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>AI UGC Video Gallery | OpenShorts</title>
<meta name="description" content="Browse {len(videos)} AI-generated UGC marketing videos. Create viral TikTok and Instagram Reels for your SaaS product.">
<meta name="robots" content="index, follow">
<meta property="og:title" content="AI UGC Video Gallery | OpenShorts">
<meta property="og:type" content="website">
<meta property="og:description" content="Browse AI-generated UGC marketing videos for SaaS products.">
<script type="application/ld+json">{ld_json}</script>
<style>
*{{margin:0;padding:0;box-sizing:border-box}}
body{{background:#0a0a0c;color:#e4e4e7;font-family:-apple-system,BlinkMacSystemFont,sans-serif}}
.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(220px,1fr));gap:20px;padding:20px;max-width:1400px;margin:0 auto}}
nav{{padding:20px 40px;border-bottom:1px solid #27272a;display:flex;align-items:center;justify-content:space-between}}
h1{{font-size:28px;font-weight:700;padding:40px 20px 0;text-align:center}}
.subtitle{{text-align:center;color:#71717a;font-size:14px;padding:8px 20px 20px}}
.cta{{display:inline-block;background:#8b5cf6;color:#fff;padding:10px 24px;border-radius:12px;text-decoration:none;font-weight:600;font-size:14px}}
</style>
</head>
<body>
<nav><strong style="font-size:18px">OpenShorts</strong><a href="/" class="cta">Create Your Video</a></nav>
<h1>AI-Generated UGC Videos</h1>
<p class="subtitle">{len(videos)} videos generated · Low Cost & Premium modes</p>
<div class="grid">{cards_html}</div>
<div style="text-align:center;padding:40px"><a href="/" class="cta">Create Your Own UGC Video</a></div>
</body></html>'''


@app.get("/video/{video_id}", response_class=HTMLResponse)
async def video_html_page(video_id: str):
    """SEO individual video page with og:video meta tags."""
    import html as html_mod
    loop = asyncio.get_running_loop()
    videos = await loop.run_in_executor(None, list_video_gallery, 200)
    meta = next((v for v in videos if v.get("video_id") == video_id), None)
    if not meta:
        raise HTTPException(status_code=404, detail="Video not found")

    title = html_mod.escape(meta.get("title", "Untitled"))
    caption = html_mod.escape(meta.get("caption", ""))
    narration = html_mod.escape(meta.get("full_narration", ""))
    video_url = meta.get("video_url", "")
    actor_url = meta.get("actor_url", "")
    duration = meta.get("duration", 0)
    mode = meta.get("video_mode", "")
    product = html_mod.escape(meta.get("product_name", ""))
    product_url = html_mod.escape(meta.get("product_url", ""))
    language = meta.get("language", "en")
    hashtags = " ".join(meta.get("hashtags", []))
    cost = meta.get("cost_estimate", {}).get("total", 0)
    created = meta.get("created_at", "")
    actor_desc = html_mod.escape(meta.get("actor_description", ""))

    ld_json = f'{{"@context":"https://schema.org","@type":"VideoObject","name":"{title}","description":"{caption}","thumbnailUrl":"{actor_url}","contentUrl":"{video_url}","uploadDate":"{created}","duration":"PT{int(duration)}S","width":1080,"height":1920,"inLanguage":"{language}"}}'

    mode_label = "Low Cost" if mode == "lowcost" else "Premium"

    return f'''<!DOCTYPE html>
<html lang="{language}">
<head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{title} - AI UGC Video | OpenShorts</title>
<meta name="description" content="{caption} {hashtags}">
<meta property="og:type" content="video.other">
<meta property="og:title" content="{title}">
<meta property="og:description" content="{caption}">
<meta property="og:video" content="{video_url}">
<meta property="og:video:type" content="video/mp4">
<meta property="og:video:width" content="1080">
<meta property="og:video:height" content="1920">
<meta property="og:image" content="{actor_url}">
<meta name="twitter:card" content="player">
<meta name="twitter:title" content="{title}">
<meta name="twitter:image" content="{actor_url}">
<script type="application/ld+json">{ld_json}</script>
<style>
*{{margin:0;padding:0;box-sizing:border-box}}
body{{background:#0a0a0c;color:#e4e4e7;font-family:-apple-system,BlinkMacSystemFont,sans-serif}}
nav{{padding:20px 40px;border-bottom:1px solid #27272a;display:flex;align-items:center;gap:16px}}
nav a{{color:#a1a1aa;text-decoration:none;font-size:14px}}
.container{{max-width:1000px;margin:0 auto;padding:40px 20px;display:grid;grid-template-columns:1fr 1fr;gap:40px}}
@media(max-width:768px){{.container{{grid-template-columns:1fr}}}}
video{{width:100%;border-radius:16px;background:#000}}
h1{{font-size:22px;font-weight:700;margin-bottom:8px}}
.meta{{color:#71717a;font-size:13px;margin-bottom:20px}}
.section{{margin-bottom:20px}}
.section h2{{font-size:13px;color:#71717a;text-transform:uppercase;letter-spacing:1px;margin-bottom:6px}}
.section p{{font-size:14px;line-height:1.6}}
.badge{{display:inline-block;padding:3px 10px;border-radius:9999px;font-size:11px;font-weight:700}}
.cta{{display:inline-block;background:#8b5cf6;color:#fff;padding:10px 24px;border-radius:12px;text-decoration:none;font-weight:600;font-size:14px;margin-top:20px}}
</style>
</head>
<body>
<nav><strong>OpenShorts</strong><a href="/gallery">Gallery</a><span style="color:#3f3f46">›</span><span style="color:#e4e4e7;font-size:14px">{title}</span></nav>
<div class="container">
<div><video src="{video_url}" poster="{actor_url}" controls autoplay playsinline style="aspect-ratio:9/16;object-fit:cover"></video></div>
<div>
<h1>{title}</h1>
<p class="meta">{duration:.0f}s · {mode_label} · ${cost:.2f} · {product}</p>
<div class="section"><h2>Caption</h2><p>{caption}</p><p style="color:#8b5cf6;margin-top:4px">{hashtags}</p></div>
<div class="section"><h2>Script</h2><p>{narration}</p></div>
<div class="section"><h2>Actor</h2><p>{actor_desc}</p></div>
{f'<div class="section"><h2>Product</h2><p><a href="{product_url}" style="color:#8b5cf6" target="_blank">{product}</a></p></div>' if product_url else ''}
<a href="/gallery">← Back to Gallery</a>
<br><a href="/" class="cta">Create Your Own</a>
</div>
</div>
</body></html>'''


@app.get("/api/saasshorts/actor-gallery")
async def saasshorts_actor_gallery():
    """List all previously generated actor images from public S3."""
    try:
        loop = asyncio.get_running_loop()
        images = await loop.run_in_executor(None, list_actor_gallery)
        return {"images": images}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class SaaSGenerateRequest(BaseModel):
    script: dict
    voice_id: Optional[str] = None
    actor_description: Optional[str] = None
    selected_actor_url: Optional[str] = None  # Pre-selected actor image URL
    retry_job_id: Optional[str] = None
    video_mode: str = "lowcost"  # "lowcost" or "premium"


@app.post("/api/saasshorts/generate")
async def saasshorts_generate(
    req: SaaSGenerateRequest,
    x_fal_key: Optional[str] = Header(None, alias="X-Fal-Key"),
    x_elevenlabs_key: Optional[str] = Header(None, alias="X-ElevenLabs-Key"),
):
    """Generate a SaaS UGC video from a script. Returns a job_id for polling."""
    fal_key = x_fal_key
    elevenlabs_key = x_elevenlabs_key

    if not fal_key:
        raise HTTPException(status_code=400, detail="Missing fal.ai API Key (X-Fal-Key header)")
    if not elevenlabs_key:
        raise HTTPException(status_code=400, detail="Missing ElevenLabs API Key (X-ElevenLabs-Key header)")

    # Support retry: reuse output_dir so cached assets (image, voice, head, broll) are kept
    reused = False
    if req.retry_job_id:
        # Check memory first, then disk
        old_dir = os.path.join(OUTPUT_DIR, f"saas_{req.retry_job_id}")
        if req.retry_job_id in saas_jobs:
            old_dir = saas_jobs[req.retry_job_id]["output_dir"]

        if os.path.isdir(old_dir):
            job_id = req.retry_job_id
            job_output_dir = old_dir
            reused = True
            # Clear the 0-byte final video so pipeline re-generates it
            for f in os.listdir(old_dir):
                fp = os.path.join(old_dir, f)
                if f.endswith("_final.mp4") and os.path.getsize(fp) == 0:
                    os.remove(fp)
            saas_jobs[job_id] = {
                "status": "processing",
                "logs": [f"Retrying job {job_id[:8]}... reusing cached assets from disk."],
                "result": None,
                "output_dir": job_output_dir,
            }

    if not reused:
        job_id = str(uuid.uuid4())
        job_output_dir = os.path.join(OUTPUT_DIR, f"saas_{job_id}")
        os.makedirs(job_output_dir, exist_ok=True)
        saas_jobs[job_id] = {
            "status": "processing",
            "logs": ["SaaSShorts job started."],
            "result": None,
            "output_dir": job_output_dir,
        }

    # If user selected a pre-generated actor, resolve it to a local path
    selected_actor_path = None
    if req.selected_actor_url:
        if req.selected_actor_url.startswith("http"):
            # Download from S3 public URL to job output dir (off the event loop).
            import httpx

            def _download_actor():
                actor_local = os.path.join(job_output_dir, "selected_actor.png")
                with httpx.Client(timeout=30.0) as client:
                    resp = client.get(req.selected_actor_url)
                    if resp.status_code == 200:
                        with open(actor_local, "wb") as f:
                            f.write(resp.content)
                        return actor_local
                return None

            try:
                selected_actor_path = await asyncio.get_running_loop().run_in_executor(None, _download_actor)
            except Exception:
                pass
        else:
            src = os.path.join(OUTPUT_DIR, req.selected_actor_url.replace("/videos/", ""))
            if os.path.exists(src):
                selected_actor_path = src

    config = {
        "fal_key": fal_key,
        "elevenlabs_key": elevenlabs_key,
        "voice_id": req.voice_id or "21m00Tcm4TlvDq8ikWAM",
        "actor_description": req.actor_description,
        "selected_actor_path": selected_actor_path,
        "video_mode": req.video_mode,
    }

    async def run_generation():
        await concurrency_semaphore.acquire()
        try:
            loop = asyncio.get_running_loop()

            def log_msg(msg):
                print(f"[SaaSShorts Job {job_id[:8]}] {msg}")
                if job_id in saas_jobs:
                    saas_jobs[job_id]["logs"].append(msg)

            def run():
                return generate_full_video(req.script, config, job_output_dir, log_msg)

            result = await loop.run_in_executor(None, run)

            if job_id in saas_jobs:
                video_filename = result["video_filename"]
                saas_jobs[job_id]["status"] = "completed"
                saas_jobs[job_id]["result"] = {
                    "video_url": f"/videos/saas_{job_id}/{video_filename}",
                    "video_filename": video_filename,
                    "duration": result.get("duration", 0),
                    "cost_estimate": result.get("cost_estimate", {}),
                    "script": req.script,
                }
                saas_jobs[job_id]["logs"].append("Video generation completed!")

                # Upload to public gallery (non-blocking)
                try:
                    gallery_meta = {
                        "title": req.script.get("title", "Untitled"),
                        "hook_text": req.script.get("hook_text", ""),
                        "caption": req.script.get("caption", ""),
                        "hashtags": req.script.get("hashtags", []),
                        "full_narration": req.script.get("full_narration", ""),
                        "actor_description": req.script.get("actor_description", ""),
                        "style": req.script.get("style", "ugc"),
                        "language": req.script.get("language", "en"),
                        "duration": result.get("duration", 0),
                        "video_mode": req.video_mode,
                        "product_name": req.script.get("_product_name", ""),
                        "product_url": req.script.get("_product_url", ""),
                        "segments": req.script.get("segments", []),
                        "cost_estimate": result.get("cost_estimate", {}),
                    }
                    gallery_result = upload_video_to_gallery(
                        video_path=result["video_path"],
                        actor_image_path=result.get("actor_image", ""),
                        metadata=gallery_meta,
                        video_id=job_id[:8],
                    )
                    if gallery_result:
                        saas_jobs[job_id]["result"]["gallery_video_id"] = gallery_result["video_id"]
                        log_msg("📤 Uploaded to public gallery.")
                except Exception as gallery_err:
                    log_msg(f"⚠️ Gallery upload skipped: {gallery_err}")

        except Exception as e:
            print(f"[SaaSShorts] ❌ Job {job_id} failed: {e}")
            if job_id in saas_jobs:
                saas_jobs[job_id]["status"] = "failed"
                saas_jobs[job_id]["logs"].append(f"Error: {str(e)}")
        finally:
            concurrency_semaphore.release()

    asyncio.create_task(run_generation())

    return {"job_id": job_id, "status": "processing"}


@app.get("/api/saasshorts/status/{job_id}")
async def saasshorts_status(job_id: str):
    """Poll SaaSShorts job status."""
    if job_id not in saas_jobs:
        raise HTTPException(status_code=404, detail="SaaSShorts job not found")

    job = saas_jobs[job_id]
    return {
        "status": job["status"],
        "logs": job["logs"],
        "result": job.get("result"),
    }


@app.get("/api/saasshorts/voices")
async def saasshorts_voices(
    x_elevenlabs_key: Optional[str] = Header(None, alias="X-ElevenLabs-Key"),
):
    """List available ElevenLabs voices."""
    if x_elevenlabs_key:
        try:
            loop = asyncio.get_event_loop()
            voices = await loop.run_in_executor(
                None, get_elevenlabs_voices, x_elevenlabs_key
            )
            if voices:
                return {"voices": voices, "source": "elevenlabs"}
        except Exception:
            pass

    # Fallback to default voices
    return {
        "voices": [
            {"voice_id": vid, "name": name, "category": "default"}
            for name, vid in DEFAULT_VOICES.items()
        ],
        "source": "defaults",
    }
