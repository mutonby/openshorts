"""fal.ai REST client: submit_and_poll a queued job, upload a file to CDN.

A thin, dependency-free wrapper around fal.ai's queue API
(``https://queue.fal.run``) and storage API
(``https://rest.alpha.fal.ai/storage/upload``). We deliberately do not
depend on the ``fal-client`` SDK — httpx is already a project dep, and
the surface we need is small enough to own.

The legacy SaaSShorts pipeline has its own private equivalents at
``app.saas.pipeline._fal_run`` / ``._fal_upload_file``; they will be
DRY'd onto this module in a separate refactor (cross-product change kept
out of the AI Restyle PR scope).
"""
from __future__ import annotations

import json
import os
import time
from typing import Any, Dict

import httpx


FAL_QUEUE_BASE = "https://queue.fal.run"
FAL_STORAGE_INITIATE_URL = "https://rest.alpha.fal.ai/storage/upload/initiate"

DEFAULT_TIMEOUT = 600
DEFAULT_POLL_INTERVAL = 5


_CONTENT_TYPES = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".mp3": "audio/mpeg",
    ".wav": "audio/wav",
    ".mp4": "video/mp4",
}


class FalError(RuntimeError):
    """Raised when a fal.ai call fails (HTTP error, job failure, timeout)."""


def submit_and_poll(
    model_id: str,
    input_data: Dict[str, Any],
    fal_key: str,
    *,
    timeout: int = DEFAULT_TIMEOUT,
    poll_interval: float = DEFAULT_POLL_INTERVAL,
) -> Dict[str, Any]:
    """POST input_data to ``queue.fal.run/{model_id}``, poll for COMPLETED,
    return the final result JSON.

    If the submit response has no ``request_id`` it's treated as a
    synchronous result and returned directly (fal occasionally does this
    for cached or fast models).
    """
    headers = {
        "Authorization": f"Key {fal_key}",
        "Content-Type": "application/json",
    }

    submit_url = f"{FAL_QUEUE_BASE}/{model_id}"

    with httpx.Client(timeout=120.0) as client:
        resp = client.post(submit_url, headers=headers, json=input_data)

    if resp.status_code >= 400:
        raise FalError(f"fal.ai submit failed ({resp.status_code}): {resp.text[:300]}")

    try:
        submit_data = resp.json()
    except json.JSONDecodeError:
        raise FalError(f"fal.ai returned non-JSON: {resp.text[:300]}")

    request_id = submit_data.get("request_id")
    if not request_id:
        return submit_data

    status_url = submit_data.get(
        "status_url", f"{FAL_QUEUE_BASE}/{model_id}/requests/{request_id}/status"
    )
    response_url = submit_data.get(
        "response_url", f"{FAL_QUEUE_BASE}/{model_id}/requests/{request_id}"
    )

    poll_headers = {"Authorization": f"Key {fal_key}"}
    start = time.time()

    while True:
        if time.time() - start > timeout:
            raise FalError(f"fal.ai job timed out after {timeout}s for {model_id}")

        with httpx.Client(timeout=30.0) as client:
            poll_resp = client.get(status_url, headers=poll_headers)

        try:
            status_data = poll_resp.json()
        except json.JSONDecodeError:
            # Treat malformed poll as transient; sleep + retry.
            time.sleep(poll_interval)
            continue

        status = status_data.get("status", "UNKNOWN")

        if status == "COMPLETED":
            with httpx.Client(timeout=120.0) as client:
                result_resp = client.get(response_url, headers=poll_headers)
            return result_resp.json()

        if status in ("FAILED", "CANCELLED"):
            error = status_data.get("error", "unknown error")
            raise FalError(f"fal.ai job {status}: {error}")

        time.sleep(poll_interval)


def upload_file(file_path: str, fal_key: str) -> str:
    """Upload a local file to fal.ai CDN storage. Returns the public URL.

    Two-step protocol: POST initiate → get signed PUT url, then PUT bytes.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Upload source not found: {file_path}")

    headers = {"Authorization": f"Key {fal_key}"}
    filename = os.path.basename(file_path)
    ext = os.path.splitext(filename)[1].lower()
    content_type = _CONTENT_TYPES.get(ext, "application/octet-stream")

    with httpx.Client(timeout=30.0) as client:
        resp = client.post(
            FAL_STORAGE_INITIATE_URL,
            headers={**headers, "Content-Type": "application/json"},
            json={"file_name": filename, "content_type": content_type},
        )
    resp.raise_for_status()
    upload_info = resp.json()

    upload_url = upload_info["upload_url"]
    file_url = upload_info["file_url"]

    with open(file_path, "rb") as f:
        file_bytes = f.read()

    with httpx.Client(timeout=120.0) as client:
        put_resp = client.put(
            upload_url,
            content=file_bytes,
            headers={"Content-Type": content_type},
        )
    put_resp.raise_for_status()

    return file_url
