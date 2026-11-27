"""Fixed picture looks applied before editable B-roll, captions, and titles."""

import os
from pathlib import Path
import subprocess
import uuid

from ffmpeg_utils import METADATA_SCRUB, QUALITY_FAST, video_encode_args


LOOKS = {
    "original": ("Original", None),
    "vivid": ("Vivid", "eq=contrast=1.06:saturation=1.18:brightness=0.01"),
    "warm": ("Warm", "colorbalance=rs=0.04:bs=-0.04,eq=saturation=1.06"),
    "cool": ("Cool", "colorbalance=rs=-0.035:bs=0.05"),
    "cinematic": ("Cinematic", "eq=contrast=1.08:saturation=0.85:brightness=-0.015"),
    "monochrome": ("Monochrome", "hue=s=0,eq=contrast=1.08"),
}


def render(path, look):
    filters = LOOKS[look][1]
    if filters is None:
        return
    work = Path(path).with_name(f"temp_look_{uuid.uuid4().hex}.mp4")
    command = ["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", str(path), "-vf", filters,
               "-map", "0:v:0", "-map", "0:a?", *video_encode_args(QUALITY_FAST),
               "-c:a", "copy", *METADATA_SCRUB, "-movflags", "+faststart", str(work)]
    try:
        subprocess.run(command, check=True, capture_output=True, timeout=600)
        os.replace(work, path)
    finally:
        work.unlink(missing_ok=True)
