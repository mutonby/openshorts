"""AI Restyle FastAPI router.

Endpoints:
- POST /api/restyle          start a restyle job
- GET  /api/restyle/{job_id} poll status

Job state is the same in-memory dict as the rest of main.py (``jobs``);
imports from ``app.main`` are deferred to avoid a circular import at
module load time. The pipeline itself lives in ``app.restyle.pipeline``.
"""
from __future__ import annotations

import os
import shutil
import uuid
from typing import Any, Dict, Optional

from fastapi import APIRouter, BackgroundTasks, File, Form, HTTPException, Request, UploadFile
from pydantic import BaseModel, Field


router = APIRouter()


MAX_PROMPT_LEN = 500
MAX_FILE_SIZE_MB = 2048  # match main.py
_CHUNK = 1024 * 1024


class RestyleStatus(BaseModel):
    status: str
    logs: list[str]
    progress_pct: int = Field(default=0, ge=0, le=100)
    result: Optional[dict] = None


@router.post("/api/restyle")
async def start_restyle(
    request: Request,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    background_prompt: str = Form(...),
    lighting_prompt: str = Form(...),
):
    """Start a restyle job. Returns ``{job_id}`` immediately; poll
    ``GET /api/restyle/{job_id}`` for status."""
    # Deferred imports — main.py imports the router, so a top-level
    # import would cycle.
    from app.main import jobs, _ensure_video_upload, OUTPUT_DIR, UPLOAD_DIR

    # C1 auth (matches /api/process: auth check before any other work)
    gemini_key = request.headers.get("X-Gemini-Key")
    fal_key = request.headers.get("X-Fal-Key")
    if not gemini_key:
        raise HTTPException(status_code=401, detail="X-Gemini-Key header required")
    if not fal_key:
        raise HTTPException(status_code=401, detail="X-Fal-Key header required")

    # C3 input validation: prompt-length cap
    if len(background_prompt) > MAX_PROMPT_LEN:
        raise HTTPException(
            status_code=413,
            detail=f"background_prompt exceeds {MAX_PROMPT_LEN} chars",
        )
    if len(lighting_prompt) > MAX_PROMPT_LEN:
        raise HTTPException(
            status_code=413,
            detail=f"lighting_prompt exceeds {MAX_PROMPT_LEN} chars",
        )

    job_id = str(uuid.uuid4())
    job_output_dir = os.path.join(OUTPUT_DIR, job_id)
    os.makedirs(job_output_dir, exist_ok=True)

    safe_name = os.path.basename(file.filename or f"{job_id}.mp4")
    input_path = os.path.join(UPLOAD_DIR, f"{job_id}_{safe_name}")
    limit_bytes = MAX_FILE_SIZE_MB * 1024 * 1024

    # Read first chunk, validate signature before persisting anything.
    first_chunk = await file.read(_CHUNK)
    if not first_chunk:
        shutil.rmtree(job_output_dir, ignore_errors=True)
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    try:
        _ensure_video_upload(safe_name, first_chunk)
    except HTTPException:
        shutil.rmtree(job_output_dir, ignore_errors=True)
        raise

    size = len(first_chunk)
    with open(input_path, "wb") as buf:
        buf.write(first_chunk)
        while chunk := await file.read(_CHUNK):
            size += len(chunk)
            if size > limit_bytes:
                buf.close()
                os.remove(input_path)
                shutil.rmtree(job_output_dir, ignore_errors=True)
                raise HTTPException(
                    status_code=413,
                    detail=f"File too large. Max size {MAX_FILE_SIZE_MB}MB",
                )
            buf.write(chunk)

    jobs[job_id] = {
        "status": "processing",
        "logs": [f"📥 Received {safe_name} ({size / 1024 / 1024:.1f} MB)"],
        "progress_pct": 0,
        "result": None,
        "product": "ai-restyle",
    }

    # Schedule the async pipeline. FastAPI BackgroundTasks awaits async
    # callables natively, so no asyncio.create_task wrapper needed.
    from app.restyle.pipeline import run_restyle_job
    background_tasks.add_task(
        run_restyle_job,
        jobs=jobs,
        job_id=job_id,
        input_path=input_path,
        background_prompt=background_prompt,
        lighting_prompt=lighting_prompt,
        gemini_key=gemini_key,
        fal_key=fal_key,
    )

    return {"job_id": job_id}


@router.get("/api/restyle/{job_id}", response_model=RestyleStatus)
async def restyle_status(job_id: str):
    """Poll the status of a restyle job."""
    from app.main import jobs
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return RestyleStatus(
        status=job["status"],
        logs=job.get("logs", []),
        progress_pct=job.get("progress_pct", 0),
        result=job.get("result"),
    )
