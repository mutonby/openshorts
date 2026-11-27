"""User music assets and duration-preserving background audio mixing."""

import json
import os
from pathlib import Path
import re
import subprocess
import uuid

import export_quality
from jsonfile import write_json_atomic

MAX_BYTES = 40 * 1024 * 1024
MAX_SECONDS = 600
ID = r"[a-f0-9]{32}"
FORMATS = {".mp3": "mp3", ".wav": "wav", ".m4a": "mov", ".ogg": "ogg", ".flac": "flac"}


def asset_paths(directory, ident):
    if not re.fullmatch(ID, ident):
        raise ValueError("Invalid music asset.")
    return Path(directory, f"music_asset_{ident}.m4a"), Path(directory, f"music_asset_{ident}.json")


def assets(directory):
    result = {}
    for manifest in Path(directory).glob("music_asset_*.json"):
        try:
            if manifest.is_symlink():
                continue
            ident = manifest.stem.removeprefix("music_asset_")
            audio, _ = asset_paths(directory, ident)
            item = json.loads(manifest.read_text())
            if audio.is_file() and not audio.is_symlink() and 0 < float(item["duration"]) <= MAX_SECONDS:
                result[ident] = {"id": ident, "name": str(item["name"])[:120], "duration": float(item["duration"])}
        except (OSError, ValueError, KeyError, TypeError):
            continue
    return result


def prepare(raw, filename, directory):
    demuxer = FORMATS.get(Path(filename or "").suffix.lower())
    if not demuxer:
        raise ValueError("Choose MP3, WAV, M4A, Ogg, or FLAC music.")
    ident = uuid.uuid4().hex
    audio, manifest = asset_paths(directory, ident)
    work = audio.with_name(f"temp_{audio.name}")
    command = ["ffmpeg", "-nostdin", "-y", "-v", "error", "-protocol_whitelist", "file,pipe",
               "-f", demuxer, "-i", str(raw), "-vn", "-t", str(MAX_SECONDS),
               "-map", "0:a:0", "-c:a", "aac", "-b:a", "160k", "-ar", "48000",
               "-map_metadata", "-1", "-movflags", "+faststart", str(work)]
    try:
        subprocess.run(command, check=True, capture_output=True, timeout=180)
        duration = float(export_quality.inspect(work)["format"]["duration"])
        if not 0 < duration <= MAX_SECONDS or work.stat().st_size > MAX_BYTES:
            raise ValueError("The music file is too long or too large.")
        asset = {"id": ident, "name": Path(filename.replace("\\", "/")).name[:120], "duration": duration}
        os.replace(work, audio)
        try:
            write_json_atomic(manifest, asset)
        except BaseException:
            audio.unlink(missing_ok=True)
            raise
        return asset
    except subprocess.SubprocessError as exc:
        raise ValueError("The music file could not be read.") from exc
    finally:
        work.unlink(missing_ok=True)


def render(path, config):
    asset, _ = asset_paths(Path(path).parent, config["asset_id"])
    info = export_quality.inspect(path)
    duration = float(info["format"]["duration"])
    has_audio = any(s.get("codec_type") == "audio" for s in info["streams"])
    work = Path(path).with_name(f"temp_music_{uuid.uuid4().hex}.mp4")
    fade = min(0.5, duration / 2)
    filters = [f"[1:a]atrim=duration={duration:.6f},asetpts=PTS-STARTPTS,volume={config['volume']},"
               f"afade=t=in:d={fade:.6f},afade=t=out:st={duration-fade:.6f}:d={fade:.6f}[music]"]
    if has_audio:
        # The voice can start late or finish before the picture. Keep its
        # timestamps and give the mixer a full-length silence-padded track.
        voice = ("[0:a:0]aresample=48000:async=1:first_pts=0,apad,"
                 f"atrim=duration={duration:.6f},asetpts=N/SR/TB")
        if config["ducking"]:
            filters.append(f"{voice},asplit[voice][control]")
            filters.append("[music][control]sidechaincompress=threshold=0.02:ratio=8:attack=15:release=300[bed]")
        else:
            filters.extend([f"{voice}[voice]", "[music]anull[bed]"])
        filters.append("[voice][bed]amix=inputs=2:duration=first:normalize=0,alimiter=limit=0.95:level=0:latency=1[a]")
    else:
        filters.append("[music]alimiter=limit=0.95:level=0:latency=1[a]")
    command = ["ffmpeg", "-nostdin", "-y", "-v", "error", "-i", str(path), "-stream_loop", "-1",
               "-ss", str(config["offset"]), "-i", str(asset), "-filter_complex", ";".join(filters),
               "-map", "0:v:0", "-map", "[a]", "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
               "-t", f"{duration:.6f}", "-map_metadata", "-1", "-movflags", "+faststart", str(work)]
    try:
        subprocess.run(command, check=True, capture_output=True, timeout=600)
        os.replace(work, path)
    finally:
        work.unlink(missing_ok=True)
