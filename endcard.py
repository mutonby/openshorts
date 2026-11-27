"""An end card is a separate derivative of the fully layered clip body."""

from bisect import bisect_right
from fractions import Fraction
import json
import math
import os
from pathlib import Path
import re
import subprocess
import tempfile
import textwrap
import uuid

from ffmpeg_utils import METADATA_SCRUB, QUALITY_FAST, video_encode_args
from jsonfile import write_json_atomic


ASSET_ID = r"[a-f0-9]{32}"
MAX_ASSETS = 20
MAX_ASSET_BYTES = 10 * 1024 * 1024
MAX_FRAMES = 120
_FORMATS = {"PNG", "JPEG", "GIF", "WEBP"}
_ASSET_FILE = re.compile(rf"endcard_asset_{ASSET_ID}\.(?:png|json)")
_SERVED = re.compile(r"endcard_[a-f0-9]{8}_(.+\.mp4)")
_FONT = Path(__file__).with_name("fonts") / "Montserrat-ExtraBold.ttf"
_CARD_FPS = 12


class EndCardError(ValueError):
    pass


def generation_config(raw):
    """Read a bounded text-only card setting shared by server and workers."""
    if raw is None or raw == "":
        return None
    try:
        if isinstance(raw, str):
            if len(raw) > 2048:
                raise ValueError
            raw = json.loads(raw)
        if raw is None:
            return None
        if not isinstance(raw, dict) or set(raw) - {"text", "handle", "duration", "sound"}:
            raise ValueError
        text, handle = raw.get("text", "Follow for more"), raw.get("handle", "")
        duration, sound = raw.get("duration", 2), raw.get("sound", "chime")
        if (not isinstance(text, str) or len(text) > 96 or not isinstance(handle, str) or len(handle) > 40
                or type(duration) is not int or duration not in (1, 2, 3) or sound not in ("silent", "chime")):
            raise ValueError
        return {"text": text.strip(), "handle": handle.strip(), "duration": duration, "sound": sound}
    except (ValueError, TypeError) as exc:
        raise EndCardError("Choose an end-card message up to 96 characters, a handle up to 40 characters, and 1–3 seconds.") from exc


def body_name(filename):
    """Return the editable body name under an end-card derivative."""
    match = _SERVED.fullmatch(filename)
    return match.group(1) if match else filename


def is_asset_file(filename):
    return bool(_ASSET_FILE.fullmatch(filename))


def asset_files(directory, asset_id):
    if not isinstance(asset_id, str) or not re.fullmatch(ASSET_ID, asset_id):
        raise EndCardError("Invalid end-card image ID.")
    stem = f"endcard_asset_{asset_id}"
    return Path(directory, f"{stem}.png"), Path(directory, f"{stem}.json")


def prepare_asset(raw_path, filename, directory):
    """Decode user images into small PNGs, retaining safe GIF animation."""
    from PIL import Image, ImageOps

    ident = uuid.uuid4().hex
    image_path, manifest = asset_files(directory, ident)
    tmp = image_path.with_name(f"temp_{image_path.name}")
    try:
        with Image.open(raw_path) as source:
            if source.format not in _FORMATS:
                raise EndCardError("Choose a PNG, JPEG, WebP, or GIF image.")
            count = getattr(source, "n_frames", 1)
            if (count > MAX_FRAMES or source.width * source.height > 16_000_000
                    or count * source.width * source.height > 90_000_000):
                raise EndCardError("The end-card image has too many pixels or frames.")
            frames, durations = [], []
            for index in range(count):
                source.seek(index)
                frame = ImageOps.exif_transpose(source.convert("RGBA"))
                frame.thumbnail((640, 640), Image.Resampling.LANCZOS)
                frames.append(frame)
                durations.append(max(20, min(10_000, int(source.info.get("duration") or 100))))
            frames[0].save(tmp, format="PNG", save_all=count > 1,
                           append_images=frames[1:], duration=durations, loop=0)
        if tmp.stat().st_size > MAX_ASSET_BYTES:
            raise EndCardError("The processed end-card image is larger than 10 MB.")
        label = Path((filename or "image").replace("\\", "/")).name.strip()[:120] or "Image"
        record = {"id": ident, "name": label, "animated": count > 1}
        os.replace(tmp, image_path)
        try:
            write_json_atomic(manifest, record)
        except BaseException:
            image_path.unlink(missing_ok=True)
            raise
        return record
    except EndCardError:
        raise
    except (EOFError, OSError, ValueError, Image.DecompressionBombError) as exc:
        raise EndCardError("The end-card image could not be opened.") from exc
    finally:
        tmp.unlink(missing_ok=True)


def assets(directory):
    result = {}
    for manifest in Path(directory).glob("endcard_asset_*.json"):
        if not is_asset_file(manifest.name) or manifest.is_symlink():
            continue
        ident = manifest.stem.removeprefix("endcard_asset_")
        image, _ = asset_files(directory, ident)
        if image.is_symlink() or not image.is_file():
            continue
        try:
            data = json.loads(manifest.read_text())
            result[ident] = {"id": ident, "name": str(data["name"])[:120], "animated": bool(data["animated"])}
        except (OSError, ValueError, TypeError, KeyError):
            continue
    return result


def _probe(path):
    command = ["ffprobe", "-v", "error", "-protocol_whitelist", "file,pipe", "-show_entries",
               "stream=codec_type,width,height,avg_frame_rate,duration,start_time:format=duration", "-of", "json", str(path)]
    try:
        data = json.loads(subprocess.check_output(command, stderr=subprocess.DEVNULL, timeout=20))
        video = next(stream for stream in data["streams"] if stream["codec_type"] == "video")
        width, height = int(video["width"]), int(video["height"])
        duration = float(video.get("duration") or data["format"]["duration"])
        start = float(video.get("start_time") or 0)
        fps = Fraction(video.get("avg_frame_rate") or "30")
        if min(width, height) < 2 or not math.isfinite(duration) or duration <= 0 or not math.isfinite(start):
            raise ValueError
        audio = next((stream for stream in data["streams"] if stream["codec_type"] == "audio"), None)
        audio_start = float((audio or {}).get("start_time") or 0)
        audio_end = audio_start + float((audio or {}).get("duration") or data["format"]["duration"])
        overlap = audio is not None and audio_start < start + duration and audio_end > start
        return width, height, duration, fps if 1 <= fps <= 120 else Fraction(30), audio is not None, overlap, start
    except (KeyError, OSError, ValueError, StopIteration, ZeroDivisionError, subprocess.SubprocessError) as exc:
        raise EndCardError("The clip body could not be read for the end card.") from exc


def _card_base(width, height, config):
    from PIL import Image, ImageDraw, ImageFont

    image = Image.new("RGB", (width, height), "#181812")
    draw = ImageDraw.Draw(image)
    scale = min(width, height)
    center = width // 2
    margin = max(1, round(scale * 0.085))
    card_top = round(height * 0.56) if config.get("asset_id") else round(height * 0.40)
    accent = "#e1bf6f"
    draw.rectangle((margin, margin, width - margin, height - margin), outline="#393529", width=max(1, scale // 320))
    bar = max(1, round(scale * 0.06))
    draw.rectangle((center - bar, card_top - margin // 2, center + bar, card_top - margin // 2 + max(1, scale // 200)), fill=accent)

    message = textwrap.fill(config.get("text") or "Follow for more", max(15, min(40, round(25 * width / height))),
                            max_lines=4, placeholder="…")
    font_size = max(2, round(scale * 0.102))
    font = ImageFont.truetype(str(_FONT), font_size)
    room = width - margin * 3
    while font_size > 2 and draw.multiline_textbbox((0, 0), message, font=font)[2] > room:
        font_size -= 1
        font = ImageFont.truetype(str(_FONT), font_size)
    draw.multiline_text((center, card_top), message, font=font, fill="#f6f0df",
                        anchor="ma", align="center", spacing=max(1, scale // 100))
    handle = config.get("handle")
    if handle:
        _, _, _, footer = draw.multiline_textbbox((center, card_top), message, font=font,
                                                   anchor="ma", align="center", spacing=max(1, scale // 100))
        handle_size = max(2, round(scale * 0.042))
        font = ImageFont.truetype(str(_FONT), handle_size)
        while handle_size > 2 and draw.textbbox((0, 0), handle, font=font)[2] > room:
            handle_size -= 1
            font = ImageFont.truetype(str(_FONT), handle_size)
        draw.text((center, footer + margin // 4), handle, font=font, fill=accent, anchor="mt")
    return image


def _card_frames(workdir, width, height, config, directory):
    from PIL import Image

    scale = min(1.0, 960 / max(width, height))
    size = max(2, round(width * scale)), max(2, round(height * scale))
    base = _card_base(*size, config)
    if not config.get("asset_id"):
        base.save(workdir / "card.png")
        return ["-loop", "1", "-framerate", str(_CARD_FPS), "-i", str(workdir / "card.png")]

    image, _ = asset_files(directory, config["asset_id"])
    frames, ends = [], []
    with Image.open(image) as source:
        for index in range(getattr(source, "n_frames", 1)):
            source.seek(index)
            frame = source.convert("RGBA")
            frame.thumbnail((round(size[0] * 0.58), round(size[1] * 0.37)), Image.Resampling.LANCZOS)
            frames.append(frame)
            ends.append((ends[-1] if ends else 0) + float(source.info.get("duration") or 100))
    for index in range(config["duration"] * _CARD_FPS):
        frame = frames[bisect_right(ends, (index * 1000 / _CARD_FPS) % ends[-1])]
        composed = base.copy()
        composed.paste(frame, ((size[0] - frame.width) // 2, round(size[1] * 0.30 - frame.height / 2)), frame)
        composed.save(workdir / f"card{index:02d}.png")
    return ["-framerate", str(_CARD_FPS), "-i", str(workdir / "card%02d.png")]


def append(body, config):
    """Write one end card after the hooked/captioned body and leave that body intact."""
    body = Path(body)
    if body_name(body.name) != body.name:
        raise EndCardError("The clip body is needed before an end card can be added.")
    duration = config["duration"]
    if duration not in (1, 2, 3):
        raise EndCardError("Choose a one-, two-, or three-second end card.")
    width, height, seconds, fps, has_audio, audio_overlap, start = _probe(body)
    target = body.with_name(f"endcard_{uuid.uuid4().hex[:8]}_{body.name}")
    try:
        with tempfile.TemporaryDirectory(prefix="temp_endcard_", dir=body.parent) as name:
            workdir = Path(name)
            card_inputs = _card_frames(workdir, width, height, config, body.parent)
            filters = [
                "[0:v:0]setpts=PTS-STARTPTS,setsar=1,format=yuv420p[body]",
                f"[1:v:0]scale={width}:{height},setsar=1,format=yuv420p,fps={fps},"
                f"tpad=stop_mode=clone:stop_duration=1,trim=duration={duration},fade=t=in:d=0.18[card]",
                "[body][card]concat=n=2:v=1:a=0[video]",
            ]
            audio_out = []
            if has_audio or config.get("sound") == "chime":
                audio_format = "aformat=sample_rates=48000:channel_layouts=stereo"
                source = (f"[0:a:0]atrim=start={start:.6f}:end={start + seconds:.6f},"
                          f"asetpts=PTS-({start:.6f})/TB,aresample=async=1:first_pts=0,"
                          f"{audio_format},apad,atrim=duration={seconds:.6f}"
                          if audio_overlap else f"anullsrc=r=48000:cl=stereo:d={seconds:.6f}")
                filters.append(f"{source}[bodyaudio]")
                if config.get("sound") == "chime":
                    tail = ("aevalsrc=0.15*(sin(2*PI*523.25*t)+0.35*sin(2*PI*783.99*t))*exp(-8*t)"
                            f":s=48000:d={duration},afade=t=in:d=0.01,{audio_format}")
                else:
                    tail = f"anullsrc=r=48000:cl=stereo:d={duration}"
                filters.extend([f"{tail}[tailaudio]", "[bodyaudio][tailaudio]concat=n=2:v=0:a=1[audio]"])
                audio_out = ["-map", "[audio]", "-c:a", "aac", "-b:a", "192k", "-ar", "48000"]
            output = workdir / "result.mp4"
            command = [
                "ffmpeg", "-nostdin", "-v", "error", "-y", "-filter_complex_threads", "1",
                "-copyts", "-protocol_whitelist", "file,pipe", "-i", str(body), *card_inputs,
                "-filter_complex", ";".join(filters), "-map", "[video]", *audio_out,
                *video_encode_args(QUALITY_FAST), *METADATA_SCRUB, "-movflags", "+faststart", str(output),
            ]
            subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, timeout=600)
            os.replace(output, target)
    except (OSError, subprocess.SubprocessError) as exc:
        raise EndCardError("The end card could not be rendered.") from exc
    return target
