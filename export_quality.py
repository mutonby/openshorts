"""Validate the published rendition before the API announces it."""
import json
import math
import subprocess


def _time(data, key, fallback=None):
    try:
        value = float(data.get(key))
        return value if math.isfinite(value) else fallback
    except (TypeError, ValueError):
        return fallback


def inspect(path):
    outcome = subprocess.run(
        ["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", str(path)],
        check=True, capture_output=True, timeout=15)
    return json.loads(outcome.stdout)


def has_audio(path):
    """Whether ``path`` has an audio stream; False when it cannot be probed.
    Only an expectation for validate(): a failed probe of the source must not
    fail the export being checked."""
    try:
        streams = inspect(path).get("streams") or []
    except (OSError, ValueError, AttributeError, subprocess.SubprocessError):
        return False
    return any(isinstance(s, dict) and s.get("codec_type") == "audio" for s in streams)


def validate_metadata(data, *, expected_duration, output_format, audio_expected):
    video = next((s for s in data.get("streams", []) if s["codec_type"] == "video"), None)
    audio = next((s for s in data.get("streams", []) if s["codec_type"] == "audio"), None)
    if not video:
        return ["The export has no video track."], {}
    width, height = int(video.get("width") or 0), int(video.get("height") or 0)
    runtime = _time(data.get("format", {}), "duration", 0)
    sar = video.get("sample_aspect_ratio")
    problems = []
    if width <= 0 or height <= 0:
        problems.append("The export has invalid dimensions.")
    expected = (1, 1) if output_format == "square" else (9, 16) if output_format in ("vertical", "auto") else None
    if expected and abs(width * expected[1] - height * expected[0]) > max(expected) * 2:
        problems.append("The output shape does not match the selected aspect ratio.")
    if sar and sar not in ("1:1", "N/A", "0:1"):
        problems.append("The video has non-square pixels.")
    if abs(runtime - expected_duration) > max(0.55, expected_duration * 0.01):
        problems.append("The export duration differs from the approved cut.")
    if any(_time(side, "rotation", 0) for side in video.get("side_data_list", [])):
        problems.append("The video carries unexpected rotation metadata.")
    if audio_expected and not audio:
        problems.append("The source has audio but the export has none.")
    if audio:
        v_start, a_start = _time(video, "start_time", 0), _time(audio, "start_time", 0)
        v_end = v_start + _time(video, "duration", runtime)
        a_end = a_start + _time(audio, "duration", runtime)
        if abs(v_start - a_start) > 0.25 or abs(v_end - a_end) > 0.5:
            problems.append("The audio and video tracks do not align on the export timeline.")
    return problems, {"width": width, "height": height, "sample_aspect_ratio": sar or "unspecified",
                      "duration": runtime, "audio": audio is not None}


def validate(path, *, expected_duration, output_format, audio_expected, decode=True):
    try:
        result = inspect(path)
        issues, measured = validate_metadata(
            result, expected_duration=expected_duration, output_format=output_format,
            audio_expected=audio_expected)
    except (OSError, ValueError, KeyError, subprocess.SubprocessError):
        return {"valid": False, "issues": ["FFprobe could not read the rendered MP4."]}

    black = 0
    sampled = 0
    if decode and not issues:
        # One whole-file decode would repeat the most expensive part of a
        # render. Sample beginning, middle, and end and record that coverage.
        for share in (0.03, 0.50, 0.93):
            start = expected_duration * share
            chunk = min(0.30, expected_duration - start)
            command = ["ffmpeg", "-nostdin", "-hide_banner", "-nostats", "-progress", "pipe:1", "-v", "info", "-xerror",
                       "-ss", f"{start:.4f}", "-t", f"{chunk:.4f}", "-i", str(path),
                       "-map", "0:v:0", "-an", "-vf",
                       "scale=128:-2,signalstats,metadata=print:key=lavfi.signalstats.YMAX", "-f", "null", "-"]
            try:
                decoded = subprocess.run(command, capture_output=True, text=True, timeout=20)
            except (OSError, subprocess.SubprocessError):
                issues.append("A validation decode could not finish.")
                break
            decoded_frames = [int(line.partition("=")[2]) for line in decoded.stdout.splitlines()
                              if line.startswith("frame=") and line.partition("=")[2].strip().isdigit()]
            if decoded.returncode or not any(decoded_frames):
                issues.append("An export sample could not be decoded.")
                break
            luminance = [int(line.rpartition("=")[2]) for line in decoded.stderr.splitlines()
                         if "lavfi.signalstats.YMAX=" in line and line.rpartition("=")[2].isdigit()]
            sampled += 1
            black += bool(luminance and max(luminance) <= 24)
        if black == 3:
            issues.append("Every sampled stretch is blank.")
    return {"valid": not issues, "issues": issues, "video": measured, "decode_samples": sampled}
