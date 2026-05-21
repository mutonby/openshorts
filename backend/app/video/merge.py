"""Concat multiple short-form clips into a single MP4 via FFmpeg filter_complex.

Each input is normalized to 1080x1920@30fps + AAC 48 kHz stereo so the concat
filter can stitch sources whose resolution/fps/sample-rate diverge (e.g. mixing
a graded clip and a subtitled clip). All FFmpeg invocations funnel through
``app.video.ffmpeg`` per project convention.
"""

from __future__ import annotations

import contextlib
import os
import secrets
from typing import List, Sequence

from app.video import ffmpeg as ffmpeg_wrapper


# Letterbox-pad to 9:16 if source has a different aspect, force 30 fps + yuv420p
# so concat is happy regardless of input resolution/fps.
NORMALIZE_FILTER = (
    "scale=1080:1920:force_original_aspect_ratio=decrease,"
    "pad=1080:1920:(ow-iw)/2:(oh-ih)/2,"
    "fps=30,setsar=1,format=yuv420p"
)


def build_concat_args(input_paths: Sequence[str], output_path: str) -> List[str]:
    """Build the ffmpeg argv (without the leading ``ffmpeg``) for a concat pass.

    Caller is responsible for path validation; this builder only composes args.
    """
    args: List[str] = ["-y"]
    for path in input_paths:
        args.extend(["-i", path])

    n = len(input_paths)
    filter_parts: List[str] = []
    concat_inputs: List[str] = []
    for i in range(n):
        filter_parts.append(
            f"[{i}:v]{NORMALIZE_FILTER}[v{i}]"
        )
        filter_parts.append(
            f"[{i}:a]aresample=48000,aformat=channel_layouts=stereo[a{i}]"
        )
        concat_inputs.append(f"[v{i}][a{i}]")
    filter_parts.append(
        f"{''.join(concat_inputs)}concat=n={n}:v=1:a=1[outv][outa]"
    )

    args.extend([
        "-filter_complex", ";".join(filter_parts),
        "-map", "[outv]",
        "-map", "[outa]",
        "-c:v", "libx264", "-preset", "fast", "-crf", "20",
        "-c:a", "aac", "-b:a", "128k", "-ar", "48000", "-ac", "2",
        output_path,
    ])
    return args


def concat_clips(input_paths: Sequence[str], output_path: str) -> str:
    """Concat ``input_paths`` into a single MP4 at ``output_path``.

    Writes to a unique ``{output_path}.partial-{nonce}.mp4`` file first, then
    ``os.replace()``s it onto ``output_path``. Two concurrent merges with the
    same indices (the public URL is filename-as-idempotency-key) get distinct
    partial paths, so neither clobbers the other mid-write; readers see the
    pre-merge file or the post-merge file, never a partial.

    Raises ``ValueError`` if fewer than 2 inputs are provided,
    ``FileNotFoundError`` if any input is missing, ``FFmpegError`` if the
    ffmpeg invocation fails, and ``RuntimeError`` if the resulting file is
    missing or empty after a "successful" run.
    """
    if len(input_paths) < 2:
        raise ValueError(
            f"concat_clips needs at least 2 inputs (got {len(input_paths)})"
        )
    for path in input_paths:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Merge input not found: {path}")

    partial_path = f"{output_path}.partial-{secrets.token_hex(6)}.mp4"
    args = build_concat_args(list(input_paths), partial_path)
    try:
        ffmpeg_wrapper.run(args)
        if not os.path.exists(partial_path) or os.path.getsize(partial_path) == 0:
            raise RuntimeError(f"Merge produced empty output: {output_path}")
        os.replace(partial_path, output_path)
    except Exception:
        with contextlib.suppress(OSError):
            os.remove(partial_path)
        raise
    return output_path
