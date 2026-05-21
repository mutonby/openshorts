"""Color-grade presets implemented as FFmpeg filter chains (no external .cube files)."""

from __future__ import annotations

import os
from typing import Dict

from app.video import ffmpeg as ffmpeg_wrapper


# Each preset is a single ffmpeg -vf expression. Built from filters that ship
# in every modern ffmpeg (curves, colorbalance, eq, hue) so there is no
# dependency on a separate LUT asset that would need licensing review.
LUT_PRESETS: Dict[str, str] = {
    "teal_orange": (
        "curves="
        "red='0/0 0.25/0.2 0.5/0.55 0.75/0.85 1/1':"
        "green='0/0 0.5/0.5 1/1':"
        "blue='0/0.1 0.25/0.35 0.5/0.5 0.75/0.55 1/0.85'"
    ),
    "warm":  "colorbalance=rs=0.15:gs=0.05:bs=-0.15:rm=0.1:gm=0.05:bm=-0.1",
    "cool":  "colorbalance=rs=-0.1:gs=0.0:bs=0.15:rm=-0.05:bm=0.1",
    "vivid": "eq=saturation=1.4:contrast=1.08:brightness=0.02",
    "noir":  "hue=s=0,eq=contrast=1.25:brightness=-0.03",
}

DEFAULT_LUT = "teal_orange"


def allowed_luts() -> tuple:
    return tuple(LUT_PRESETS.keys())


def apply_lut(input_path: str, output_path: str, lut_name: str) -> None:
    """Apply a named LUT preset via FFmpeg ``-vf``. Re-encodes video; copies audio.

    Raises ``ValueError`` for unknown LUTs and ``FFmpegError`` for ffmpeg failures.
    """
    if lut_name not in LUT_PRESETS:
        raise ValueError(
            f"Unknown LUT {lut_name!r}. Allowed: {sorted(LUT_PRESETS.keys())}"
        )

    ffmpeg_wrapper.run([
        "-y",
        "-i", input_path,
        "-vf", LUT_PRESETS[lut_name],
        "-c:v", "libx264", "-preset", "fast", "-crf", "20",
        "-c:a", "copy",
        output_path,
    ])

    if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
        raise RuntimeError(f"Color grade produced empty output: {output_path}")
