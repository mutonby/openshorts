"""fal.ai video-to-video restyle: send source video + reference frame,
download the restyled output.

Model defaults to ``fal-ai/wan/v2.5/turbo/video-to-video`` — chosen
ahead of the Phase 0 spike based on the plan's pre-research; revisit
once the spike runs against the real demo clip and a relit reference.

Wraps ``app.integrations.fal.submit_and_poll`` + ``.upload_file`` so the
network surface and auth header live in one place.
"""
from __future__ import annotations

import os

import httpx

from app.integrations.fal import require_fal_download_url, submit_and_poll, upload_file


MODEL_ID = "fal-ai/wan/v2.5/turbo/video-to-video"


_RESTYLE_PROMPT = (
    "Match the lighting and background of the reference image. "
    "Preserve all motion, subject, and camera angle from the source video."
)


def restyle_video(
    api_key: str,
    video_path: str,
    reference_frame_path: str,
    out_path: str,
) -> str:
    """Run fal.ai v2v with the source video and reference frame. Writes
    the restyled MP4 to ``out_path``.

    Returns ``out_path`` on success. Raises ``FileNotFoundError`` if
    either input is missing, ``FalError`` from the integration layer on
    API errors, ``RuntimeError`` if the response has no video URL.
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Input video not found: {video_path}")
    if not os.path.exists(reference_frame_path):
        raise FileNotFoundError(f"Reference frame not found: {reference_frame_path}")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    video_url = upload_file(video_path, api_key)
    ref_url = upload_file(reference_frame_path, api_key)

    response = submit_and_poll(
        MODEL_ID,
        {
            "video_url": video_url,
            "image_url": ref_url,
            "prompt": _RESTYLE_PROMPT,
        },
        api_key,
    )

    out_url = (
        response.get("video", {}).get("url") if isinstance(response, dict) else None
    )
    if not out_url:
        raise RuntimeError(f"fal.ai v2v response carried no video URL: {response!r}")

    # SSRF defense: the URL came from the model response, so we don't
    # trust it. Reject anything that isn't a fal-controlled host before
    # opening a connection. (Codex HIGH-1)
    out_url = require_fal_download_url(out_url)

    with httpx.Client(timeout=300.0) as client:
        with client.stream("GET", out_url) as r:
            r.raise_for_status()
            with open(out_path, "wb") as f:
                for chunk in r.iter_bytes():
                    f.write(chunk)

    return out_path
