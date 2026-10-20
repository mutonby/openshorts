"""Media transcoding helpers for Open-Dispatch.

Right now: image resize per-platform spec. Video transcoding (ffmpeg) is
the next obvious step but not in this module yet.
"""

from .transcode import (
    PLATFORM_IMAGE_SPECS,
    PLATFORM_VIDEO_SPECS,
    TranscodeError,
    transcode_image,
    transcode_image_bytes,
)

__all__ = [
    "PLATFORM_IMAGE_SPECS",
    "PLATFORM_VIDEO_SPECS",
    "TranscodeError",
    "transcode_image",
    "transcode_image_bytes",
]
