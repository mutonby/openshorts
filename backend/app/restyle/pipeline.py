"""AI Restyle pipeline orchestrator.

7-step async flow:
  1. (Caller validates MIME + ftyp + size; we trust that.)
  2. Probe duration; reject if > MAX_DURATION_SEC
  3. Extract first frame to PNG
  4. Nano Banana relight of that frame
  5. fal.ai v2v with source video + relit frame as reference
  6. Mux original audio back onto the restyled video
  7. Persist result on the jobs dict; mark status='completed'

Mutates ``jobs[job_id]`` in place so the route handler + frontend
polling see live progress. Any unhandled exception flips status to
'failed' and appends the exception message to logs.
"""
from __future__ import annotations

import asyncio
import os
from functools import partial
from typing import Any, Dict, Optional

from app.ml.frame_extract import extract_first_frame
from app.ml.frame_relight import relight_frame
from app.ml.video_restyle import restyle_video
from app.video.ffmpeg import mux_video_audio, probe_duration


OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "output")
MAX_DURATION_SEC = 30.0


def _ensure_job(jobs: Dict[str, Any], job_id: str) -> Dict[str, Any]:
    """Create a baseline entry if the caller forgot to seed one."""
    if job_id not in jobs:
        jobs[job_id] = {
            "status": "processing",
            "logs": [],
            "progress_pct": 0,
            "result": None,
        }
    return jobs[job_id]


def _log(jobs: Dict[str, Any], job_id: str, line: str, pct: Optional[int] = None) -> None:
    job = _ensure_job(jobs, job_id)
    job["logs"].append(line)
    if pct is not None:
        job["progress_pct"] = pct


async def run_restyle_job(
    jobs: Dict[str, Any],
    job_id: str,
    input_path: str,
    background_prompt: str,
    lighting_prompt: str,
    gemini_key: str,
    fal_key: str,
) -> None:
    """Drive the full restyle pipeline for ``job_id``. Mutates
    ``jobs[job_id]`` in place; never raises."""
    _ensure_job(jobs, job_id)
    output_dir = os.path.join(OUTPUT_DIR, job_id)
    os.makedirs(output_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(input_path))[0]

    loop = asyncio.get_event_loop()

    try:
        _log(jobs, job_id, "🔎 Probing video duration…", pct=5)
        duration = await loop.run_in_executor(None, partial(probe_duration, input_path))
        if duration > MAX_DURATION_SEC:
            raise ValueError(
                f"Video duration {duration:.1f}s exceeds the {MAX_DURATION_SEC:.0f}s "
                f"cap for AI Restyle v1"
            )

        _log(jobs, job_id, "🎞️ Extracting first frame…", pct=10)
        frame_path = os.path.join(output_dir, f"{base}_frame.png")
        await loop.run_in_executor(
            None, partial(extract_first_frame, input_path, frame_path),
        )

        _log(jobs, job_id, "🪄 Relighting frame with Nano Banana…", pct=20)
        relit_path = os.path.join(output_dir, f"{base}_relit.png")
        await loop.run_in_executor(
            None,
            partial(
                relight_frame,
                api_key=gemini_key,
                frame_path=frame_path,
                background_prompt=background_prompt,
                lighting_prompt=lighting_prompt,
                out_path=relit_path,
            ),
        )

        _log(jobs, job_id, "🎬 Restyling video via fal.ai (~30-90s)…", pct=40)
        restyled_noaudio = os.path.join(output_dir, f"{base}_restyled_noaudio.mp4")
        await loop.run_in_executor(
            None,
            partial(
                restyle_video,
                api_key=fal_key,
                video_path=input_path,
                reference_frame_path=relit_path,
                out_path=restyled_noaudio,
            ),
        )

        _log(jobs, job_id, "🔊 Muxing original audio back…", pct=90)
        final_out = os.path.join(output_dir, f"restyled_{os.path.basename(input_path)}")
        await loop.run_in_executor(
            None,
            partial(mux_video_audio, restyled_noaudio, input_path, final_out),
        )

        job = jobs[job_id]
        job["result"] = {
            "video_url": f"/videos/{job_id}/{os.path.basename(final_out)}",
            "original_url": f"/videos/{job_id}/{os.path.basename(input_path)}",
            "duration_sec": duration,
        }
        job["status"] = "completed"
        job["progress_pct"] = 100
        _log(jobs, job_id, "✅ AI Restyle complete.")

    except Exception as exc:
        job = _ensure_job(jobs, job_id)
        job["status"] = "failed"
        job["logs"].append(f"❌ {exc}")
