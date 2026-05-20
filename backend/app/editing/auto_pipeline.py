"""Reusable post-processing helpers called by /api/process (auto-pipeline) and per-clip routes."""

from __future__ import annotations

import os
import shutil
import time
from typing import Any, Dict, Optional

from app.editing.ai_filters import VideoEditor
from app.editing.color_grade import DEFAULT_LUT, apply_lut, allowed_luts
from app.editing.silence import cut_silence
from app.overlays.subtitles_generate import generate_srt, generate_srt_from_video
from app.overlays.subtitles_render import burn_subtitles

OUTPUT_DIR = "output"


def _job_dir(job_id: str) -> str:
    return os.path.join(OUTPUT_DIR, job_id)


def apply_ai_edit(
    *,
    api_key: str,
    job_id: str,
    input_filename: str,
    transcript: Optional[Dict[str, Any]] = None,
) -> str:
    """Apply Gemini-driven FFmpeg effects to a clip.

    Writes ``edited_{input_filename}`` next to the input. Idempotent: returns
    the existing output when one is present and non-empty. Raises on any
    Gemini / FFmpeg / cv2 failure — caller decides whether to log and skip
    or to mark the whole job as failed.
    """
    job_dir = _job_dir(job_id)
    input_path = os.path.join(job_dir, input_filename)
    edited_filename = f"edited_{input_filename}"
    output_path = os.path.join(job_dir, edited_filename)

    if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
        return edited_filename

    editor = VideoEditor(api_key=api_key)

    # ASCII-safe temp paths mirror the per-clip /api/edit handler — avoids
    # subprocess UnicodeEncodeError on filesystems with mixed encodings.
    stamp = int(time.time() * 1000)
    safe_input_path = os.path.join(job_dir, f"temp_input_auto_{stamp}.mp4")
    shutil.copy(input_path, safe_input_path)

    try:
        vid_file = editor.upload_video(safe_input_path)

        import cv2  # local import keeps module-load cheap

        cap = cv2.VideoCapture(safe_input_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = frame_count / fps if fps else 0
        cap.release()

        filter_data = editor.get_ffmpeg_filter(
            vid_file, duration,
            fps=fps, width=width, height=height,
            transcript=transcript,
        )

        safe_output = os.path.join(job_dir, f"temp_output_auto_{stamp}.mp4")
        editor.apply_edits(safe_input_path, safe_output, filter_data)

        if os.path.exists(safe_output):
            shutil.move(safe_output, output_path)
    finally:
        if os.path.exists(safe_input_path):
            os.remove(safe_input_path)

    return edited_filename


def apply_color_grade(
    *,
    job_id: str,
    input_filename: str,
    lut_name: Optional[str] = None,
) -> str:
    """Apply a named LUT preset to a clip. Writes ``graded_{input_filename}``.

    Idempotent: returns the existing output when present and non-empty.
    Raises ``ValueError`` for unknown LUTs (caller should log and skip).
    """
    job_dir = _job_dir(job_id)
    input_path = os.path.join(job_dir, input_filename)
    output_filename = f"graded_{input_filename}"
    output_path = os.path.join(job_dir, output_filename)

    if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
        return output_filename

    name = (lut_name or DEFAULT_LUT).strip().lower()
    if name not in allowed_luts():
        raise ValueError(f"lut_name must be one of {sorted(allowed_luts())}, got {name!r}")

    apply_lut(input_path, output_path, name)
    return output_filename


def apply_silence_cut(
    *,
    job_id: str,
    input_filename: str,
    noise_db: float = -30.0,
    min_silence_sec: float = 0.5,
) -> str:
    """Cut silent segments from a clip. Writes ``silencecut_{input_filename}``.

    Idempotent: returns the existing output when present and non-empty.
    A clip with no detectable silence is stream-copied so the output still
    exists (matches the legacy per-clip flow's expectations).
    """
    job_dir = _job_dir(job_id)
    input_path = os.path.join(job_dir, input_filename)
    output_filename = f"silencecut_{input_filename}"
    output_path = os.path.join(job_dir, output_filename)

    if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
        return output_filename

    cut_silence(
        input_path,
        output_path,
        noise_db=noise_db,
        min_silence_sec=min_silence_sec,
    )
    return output_filename


def apply_subtitles(
    *,
    job_id: str,
    clip_index: int,
    input_filename: str,
    transcript: Dict[str, Any],
    clip_start: float,
    clip_end: float,
    style: Dict[str, Any],
) -> str:
    """Burn subtitles onto a clip with brand-kit styling.

    Writes ``subtitled_{input_filename}``. Idempotent — returns the existing
    output when present and non-empty.

    ``style`` keys (all optional, defaults applied): ``position``,
    ``font_size``, ``font_name``, ``font_color``, ``border_color``,
    ``border_width``, ``bg_color``, ``bg_opacity``, ``words_per_line``,
    ``text_case``. Mirrors the SubtitleRequest schema in main.py.
    """
    job_dir = _job_dir(job_id)
    input_path = os.path.join(job_dir, input_filename)
    output_filename = f"subtitled_{input_filename}"
    output_path = os.path.join(job_dir, output_filename)

    if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
        return output_filename

    srt_path = os.path.join(job_dir, f"subs_{clip_index}_{int(time.time())}.srt")

    max_words = style.get('words_per_line')
    if max_words is not None and max_words <= 0:
        max_words = None
    text_case = style.get('text_case') or 'original'

    # Mirror /api/subtitle: dubbed clips need fresh transcription because
    # the original transcript no longer matches the audio track.
    is_dubbed = input_filename.startswith("translated_")
    if is_dubbed:
        success = generate_srt_from_video(input_path, srt_path, max_words=max_words, text_case=text_case)
    else:
        success = generate_srt(transcript, clip_start, clip_end, srt_path, max_words=max_words, text_case=text_case)

    if not success:
        raise RuntimeError(f"No words found for clip range [{clip_start}, {clip_end}]")

    burn_subtitles(
        input_path, srt_path, output_path,
        alignment=style.get('position', 'bottom'),
        fontsize=int(style.get('font_size', 16)),
        font_name=style.get('font_name', 'Verdana'),
        font_color=style.get('font_color', '#FFFFFF'),
        border_color=style.get('border_color', '#000000'),
        border_width=int(style.get('border_width', 2)),
        bg_color=style.get('bg_color', '#000000'),
        bg_opacity=float(style.get('bg_opacity', 0.0)),
    )

    return output_filename
