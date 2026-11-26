"""Uploaded B-roll and its placement on the clip's edited timeline."""

import json
import math
import os
from pathlib import Path
import re
import subprocess
import uuid

import layout_ranges
from ffmpeg_utils import METADATA_SCRUB, QUALITY_FAST, video_encode_args
from jsonfile import write_json_atomic


MAX_UPLOAD_BYTES = 40 * 1024 * 1024
MAX_ASSET_SECONDS = 60.0
MIN_LAYER_SECONDS = 0.1
MAX_ASSETS = 20
MAX_LAYERS = 12
ASSET_ID = r"[a-f0-9]{32}"
_ASSET_FILE = re.compile(rf"broll_asset_{ASSET_ID}\.(?:mp4|json)")
_FORMATS = {".mp4": "mov", ".mov": "mov", ".webm": "matroska"}


class BrollError(ValueError):
    pass


def input_format(filename):
    return _FORMATS.get(Path(filename or "").suffix.lower())


def is_asset_file(filename):
    return bool(_ASSET_FILE.fullmatch(filename))


def asset_files(directory, asset_id):
    if not isinstance(asset_id, str) or not re.fullmatch(ASSET_ID, asset_id):
        raise BrollError("Invalid B-roll asset ID.")
    stem = f"broll_asset_{asset_id}"
    return Path(directory, f"{stem}.mp4"), Path(directory, f"{stem}.json")


def _probe(path, *, demuxer=None):
    command = ["ffprobe", "-v", "error", "-protocol_whitelist", "file,pipe"]
    if demuxer:
        command += ["-f", demuxer]
    command += ["-select_streams", "v:0", "-show_entries", "stream=width,height,duration:format=duration",
                "-of", "json", str(path)]
    try:
        data = json.loads(subprocess.check_output(command, stderr=subprocess.DEVNULL, timeout=20))
        stream = data["streams"][0]
        width, height = int(stream["width"]), int(stream["height"])
        seconds = float(stream.get("duration") or data["format"]["duration"])
        if min(width, height) < 2 or not math.isfinite(seconds) or seconds <= 0:
            raise ValueError
        return width, height, seconds
    except (IndexError, KeyError, TypeError, ValueError, OSError, subprocess.SubprocessError) as exc:
        raise BrollError("The video could not be read.") from exc


def prepare_asset(raw_path, filename, directory):
    """Decode only the chosen container and retain a muted, browser-safe MP4."""
    demuxer = input_format(filename)
    if not demuxer:
        raise BrollError("Upload a video in MP4, MOV, or WebM format.")
    width, height, duration = _probe(raw_path, demuxer=demuxer)
    if not MIN_LAYER_SECONDS <= duration <= MAX_ASSET_SECONDS:
        raise BrollError(f"B-roll video must be between {MIN_LAYER_SECONDS:g}s and {MAX_ASSET_SECONDS:g}s.")
    if max(width, height) > 4096:
        raise BrollError("B-roll video dimensions must be at most 4096 pixels per side.")
    ident = uuid.uuid4().hex
    video, manifest = asset_files(directory, ident)
    work = video.with_name(f"temp_{video.name}")
    command = [
        "ffmpeg", "-nostdin", "-v", "error", "-y", "-threads", "2", "-filter_threads", "1",
        "-protocol_whitelist", "file,pipe", "-f", demuxer, "-i", str(raw_path),
        "-map", "0:v:0", "-an", "-t", str(MAX_ASSET_SECONDS),
        "-vf", "scale=1080:1080:force_original_aspect_ratio=decrease:force_divisible_by=2,setsar=1,fps=30",
        "-c:v", "libx264", "-preset", "fast", "-crf", "20", "-threads", "2", "-pix_fmt", "yuv420p",
        *METADATA_SCRUB, "-movflags", "+faststart", str(work),
    ]
    try:
        subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, check=True, timeout=180)
        _, _, actual_duration = _probe(work)
        if work.stat().st_size > MAX_UPLOAD_BYTES or actual_duration < MIN_LAYER_SECONDS:
            raise BrollError("The processed B-roll video is too large or too short.")
        label = Path(filename.replace("\\", "/")).name.strip()[:120] or "B-roll video"
        asset = {"id": ident, "name": label, "duration": round(actual_duration, 3)}
        os.replace(work, video)
        try:
            write_json_atomic(manifest, asset)
        except BaseException:
            video.unlink(missing_ok=True)
            raise
        return asset
    except (OSError, subprocess.SubprocessError) as exc:
        raise BrollError("The uploaded video could not be prepared for editing.") from exc
    finally:
        work.unlink(missing_ok=True)


def assets(directory):
    result = {}
    for manifest in Path(directory).glob("broll_asset_*.json"):
        if not is_asset_file(manifest.name) or manifest.is_symlink():
            continue
        try:
            ident = manifest.stem.removeprefix("broll_asset_")
            video, _ = asset_files(directory, ident)
            if video.is_symlink() or not video.is_file():
                continue
            data = json.loads(manifest.read_text())
            duration = float(data["duration"])
            if not MIN_LAYER_SECONDS <= duration <= MAX_ASSET_SECONDS:
                continue
            result[ident] = {"id": ident, "name": str(data["name"])[:120], "duration": duration}
            if isinstance(data.get("attribution"), dict):
                result[ident]["attribution"] = data["attribution"]
        except (OSError, ValueError, TypeError, KeyError):
            continue
    return result


def normalize_layers(layers, available):
    if len(layers) > MAX_LAYERS:
        raise BrollError(f"A clip can use up to {MAX_LAYERS} B-roll placements.")
    result = []
    for layer in layers:
        try:
            asset_id = layer["asset_id"]
            start = float(layer["start"])
            duration = float(layer["duration"])
            offset = float(layer.get("offset", 0))
            source_time = layer.get("source_time")
            source_time = float(source_time) if source_time is not None else None
            if not (math.isfinite(start) and start >= 0 and MIN_LAYER_SECONDS <= duration <= MAX_ASSET_SECONDS
                    and math.isfinite(offset) and offset >= 0
                    and (source_time is None or (math.isfinite(source_time) and source_time >= 0))):
                raise ValueError
            asset = available.get(asset_id)
        except (KeyError, TypeError, ValueError) as exc:
            raise BrollError("Invalid B-roll timeline placement.") from exc
        if asset is None:
            raise BrollError("A B-roll asset is unavailable. Remove that placement or upload the video again.")
        if duration + offset > asset["duration"] + 0.001:
            raise BrollError("B-roll duration and in-point exceed the asset length.")
        result.append({"asset_id": asset_id, "start": round(start, 3), "offset": round(offset, 3),
                       "duration": round(min(duration, asset["duration"] - offset), 3),
                       "source_time": round(source_time, 3) if source_time is not None else None})
    return result


def active_layers(layers, segments):
    """Trim overlays at the end of the cut and follow caption anchors through recuts."""
    total = sum(segment["end"] - segment["start"] for segment in segments)
    for layer in layers:
        start = layer["start"]
        anchor = layer.get("source_time")
        if anchor is not None:
            offset = 0.0
            for segment in segments:
                if segment["start"] <= anchor < segment["end"]:
                    start = offset + anchor - segment["start"]
                    break
                offset += segment["end"] - segment["start"]
            else:
                continue
        duration = round(min(layer["duration"], total - start), 3)
        if duration >= MIN_LAYER_SECONDS:
            yield {**layer, "start": round(start, 3), "duration": duration}


def render(path, layers):
    """Replace picture only for each active span and keep the clip's audio."""
    width, height, _ = _probe(path)
    command = ["ffmpeg", "-nostdin", "-v", "error", "-y", "-filter_complex_threads", "1",
               "-protocol_whitelist", "file,pipe", "-i", str(path)]
    filters = ["[0:v]setpts=PTS-STARTPTS[base0]"]
    directory = Path(path).parent
    for index, layer in enumerate(layers, 1):
        video, _ = asset_files(directory, layer["asset_id"])
        start, duration = layer["start"], layer["duration"]
        command += ["-ss", str(layer["offset"]), "-t", str(duration), "-protocol_whitelist", "file,pipe",
                    "-i", str(video)]
        filters.append(f"[{index}:v]scale={width}:{height}:force_original_aspect_ratio=increase,"
                       f"crop={width}:{height},setsar=1,setpts=PTS-STARTPTS+{start:.3f}/TB[asset{index}]")
        filters.append(f"[base{index - 1}][asset{index}]overlay=repeatlast=0:eof_action=pass:"
                       f"enable='gte(t,{start:.3f})*lt(t,{start + duration:.3f})'[base{index}]")
    _replace_picture(path, command, filters, f"base{len(layers)}", "The B-roll timeline could not be rendered.")


def render_bottom_video(path, asset_id):
    """Fit the main picture above a looping, silent secondary video."""
    width, height, duration = _probe(path)
    upper = height // 4 * 2
    lower = height - upper
    if upper < 2 or width % 2 or lower % 2:
        raise BrollError("The clip dimensions do not support a bottom video.")

    video, _ = asset_files(Path(path).parent, asset_id)
    command = ["ffmpeg", "-nostdin", "-v", "error", "-y", "-filter_complex_threads", "1",
               "-protocol_whitelist", "file,pipe", "-i", str(path),
               "-stream_loop", "-1", "-protocol_whitelist", "file,pipe", "-i", str(video)]
    filters = [
        "[0:v]setpts=PTS-STARTPTS,split[main][backdrop]",
        f"[main]scale={width}:{upper}:force_original_aspect_ratio=decrease:force_divisible_by=2,"
        "setsar=1[foreground]",
        f"[backdrop]scale={width}:{upper}:force_original_aspect_ratio=increase:force_divisible_by=2,"
        f"crop={width}:{upper},gblur=sigma=14,setsar=1[blur]",
        f"[blur][foreground]overlay=(W-w)/2:(H-h)/2:shortest=1,pad={width}:{height}:0:0[top]",
        f"[1:v]setpts=PTS-STARTPTS,scale={width}:{lower}:force_original_aspect_ratio=increase:force_divisible_by=2,"
        f"crop={width}:{lower},setsar=1[bottom]",
        f"[top][bottom]overlay=0:{upper}:shortest=1[out]",
    ]
    _replace_picture(path, command, filters, "out", "The looping bottom video could not be rendered.")
    seam = upper / height
    layout_ranges.write(str(path), [(0, duration, "stack", None, {"seam": seam, "caption_y": seam})])


def _replace_picture(path, command, filters, output_label, error):
    work = Path(path).parent / f"temp_broll_{uuid.uuid4().hex}.mp4"
    command += ["-filter_complex", ";".join(filters), "-map", f"[{output_label}]", "-map", "0:a?",
                *video_encode_args(QUALITY_FAST), "-c:a", "copy", *METADATA_SCRUB, "-movflags", "+faststart", str(work)]
    try:
        subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, check=True, timeout=600)
        os.replace(work, path)
    except (OSError, subprocess.SubprocessError) as exc:
        raise BrollError(error) from exc
    finally:
        work.unlink(missing_ok=True)
