"""Extract the first video frame to a PNG file via the FFmpeg wrapper."""
from __future__ import annotations

import os

from app.video.ffmpeg import run as ffmpeg_run


def extract_first_frame(video_path: str, out_path: str) -> str:
    """Write the frame at t=0 of ``video_path`` to ``out_path`` as PNG.

    Returns ``out_path`` on success. Raises ``FileNotFoundError`` if the
    source is missing; ``FFmpegError`` if encoding fails.
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Input video not found: {video_path}")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    ffmpeg_run(
        ["-y", "-ss", "0", "-i", video_path, "-frames:v", "1", "-update", "1", out_path],
    )
    return out_path
