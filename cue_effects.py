"""Source-timed editor cues rendered before a clip's captions and titles."""
import math
import os
from pathlib import Path
import subprocess
from typing import Literal
import uuid
import wave

import edit_builder
import export_quality
import recut
import sfx
from ffmpeg_utils import METADATA_SCRUB, QUALITY_FAST, video_encode_args

MAX_CUES = 12
MIN_SECONDS = 0.2
MAX_SECONDS = 2.0
DEFAULT_VOLUME = 0.35
CueKind = Literal["hit", "whoosh", "boom", "dip_to_black", "color_pop", "bw_moment", "vignette"]

CATALOG = {
    "hit": ("Hit", "Sound effect", sfx.SOUND_SECONDS["hit"]),
    "whoosh": ("Whoosh", "Sound effect", sfx.SOUND_SECONDS["whoosh"]),
    "boom": ("Boom", "Sound effect", sfx.SOUND_SECONDS["boom"]),
    "dip_to_black": ("Dip to black", "Transition", 0.5),
    "color_pop": ("Color pop", "Video effect", 0.8),
    "bw_moment": ("Black and white", "Video effect", 0.8),
    "vignette": ("Vignette", "Video effect", 0.8),
}


def catalog():
    return [
        {"kind": kind, "label": label, "category": category, "duration": duration,
         "min_duration": MIN_SECONDS, "max_duration": sfx.SOUND_SECONDS.get(kind, MAX_SECONDS)}
        for kind, (label, category, duration) in CATALOG.items()
    ]


def normalize(cues):
    """Keep only safe scalar values, including when loading an older project."""
    if not isinstance(cues, list):
        return []
    result = []
    for cue in cues[:MAX_CUES]:
        if not isinstance(cue, dict):
            continue
        kind = cue.get("kind")
        if not isinstance(kind, str) or kind not in CATALOG:
            continue
        try:
            source_time = float(cue["source_time"])
            duration = float(cue.get("duration", CATALOG[kind][2]))
            volume = float(cue.get("volume", DEFAULT_VOLUME))
        except (KeyError, ValueError, OverflowError, TypeError):
            continue
        if not math.isfinite(source_time) or source_time < 0 or not math.isfinite(duration) or duration < MIN_SECONDS:
            continue
        if not math.isfinite(volume) or not 0 <= volume <= 1:
            continue
        result.append({
            "kind": kind,
            "source_time": round(source_time, 3),
            "duration": round(min(duration, sfx.SOUND_SECONDS.get(kind, MAX_SECONDS)), 3),
            "volume": round(volume, 3),
        })
    return result


def placements(cues, segments):
    """Map source anchors to output time. Each occurrence in a replay gets its cue."""
    cues = normalize(cues)
    result = []
    offset = 0.0
    for segment in segments:
        first, last = segment["start"], segment["end"]
        for cue in cues:
            when = cue["source_time"]
            if first <= when < last:
                duration = round(min(cue["duration"], last - when), 3)
                if duration >= MIN_SECONDS:
                    result.append({**cue, "start": round(offset + when - first, 3),
                                   "duration": duration})
        offset += last - first
    return sorted(result, key=lambda cue: cue["start"])


def _picture_filters(placed, duration):
    filters = []
    for cue in placed:
        kind = cue["kind"]
        if kind == "dip_to_black":
            start = cue["start"]
            half = cue["duration"] / 2
            mid, end = start + half, start + 2 * half
            filters.extend([
                f"fade=t=out:st={start:.6f}:d={half:.6f}:enable='gte(t,{start:.6f})*lt(t,{mid:.6f})'",
                f"fade=t=in:st={mid:.6f}:d={half:.6f}:enable='between(t,{mid:.6f},{end:.6f})'",
            ])
        elif kind not in sfx.SOUNDS:
            start = cue["start"]
            effects, _ = edit_builder.build_filter_string(
                [{"type": kind, "start": start, "end": start + cue["duration"]}],
                duration, fps=None, width=None, height=None, has_captions=True,
            )
            if effects:
                filters.append(effects)
    return ",".join(filters)


def _write_sound_track(placed, duration, target):
    sounds = [cue for cue in placed if cue["kind"] in sfx.SOUNDS and cue["volume"] > 0]
    if not sounds:
        return False
    # Write one mono mix. Delaying every cue in its own FFmpeg branch would
    # allocate a separate delay buffer for each replay of a late sound.
    import numpy as np
    track = np.zeros(math.ceil(duration * sfx.RATE), dtype="<f4")
    samples = {}
    for cue in sounds:
        name = cue["kind"]
        if name not in samples:
            path = sfx.path(name)
            if path is None:
                raise RuntimeError("The built-in sound could not be prepared.")
            with wave.open(path, "rb") as wav:
                channels = wav.getnchannels()
                if wav.getsampwidth() != 2 or wav.getframerate() != sfx.RATE or channels not in (1, 2):
                    raise RuntimeError("The built-in sound has an unsupported format.")
                frames = wav.readframes(round(sfx.SOUND_SECONDS[name] * sfx.RATE))
                pcm = np.frombuffer(frames, dtype="<i2").reshape(-1, channels)
                samples[name] = pcm[:, 0].astype(np.float32) / 32768
        start = round(cue["start"] * sfx.RATE)
        length = min(round(cue["duration"] * sfx.RATE), len(samples[name]), len(track) - start)
        sample = samples[name][:length] * cue["volume"]
        fade = min(round(0.04 * sfx.RATE), length // 4)
        if fade:
            sample[-fade:] *= np.linspace(1, 0, fade)
        track[start:start + length] += sample
    track.tofile(target)
    return True


def _sound_graph(duration, has_audio):
    seconds = f"{duration:.6f}"
    source = (
        "[0:a:0]aresample=48000,aformat=channel_layouts=stereo,apad"
        if has_audio else "anullsrc=r=48000:cl=stereo"
    )
    return (f"{source},atrim=duration={seconds},asetpts=PTS-STARTPTS[body];"
            "[1:a]pan=stereo|c0=c0|c1=c0[effects];"
            "[body][effects]amix=inputs=2:duration=first:normalize=0,"
            "alimiter=limit=0.95:level=0:latency=1[a]")


def render(path, placed):
    """Mix sounds and apply timed video effects without changing the clip length."""
    info = export_quality.inspect(path)
    duration = float(info["format"]["duration"])
    if not 0 < duration <= recut.MAX_TOTAL_SECONDS + 1:
        raise RuntimeError("The clip is outside the timed effect duration limit.")
    has_audio = any(stream.get("codec_type") == "audio" for stream in info["streams"])
    visible = [
        {**cue, "duration": min(cue["duration"], duration - cue["start"])}
        for cue in placed if round(duration - cue["start"], 3) >= MIN_SECONDS
    ]
    picture = _picture_filters(visible, duration)
    target = Path(path).with_name(f"temp_cue_{uuid.uuid4().hex}.mp4")
    audio_track = target.with_suffix(".f32")
    try:
        sound = _write_sound_track(visible, duration, audio_track)
        if not picture and not sound:
            return
        command = ["ffmpeg", "-nostdin", "-y", "-v", "error", "-filter_threads", "1",
                   "-filter_complex_threads", "1", "-i", str(path)]
        if sound:
            command.extend(["-f", "f32le", "-ar", str(sfx.RATE), "-ac", "1", "-i", str(audio_track)])
        command.extend(["-map", "0:v:0"])
        if picture:
            command.extend(["-vf", picture, *video_encode_args(QUALITY_FAST)])
        else:
            command.extend(["-c:v", "copy"])
        if sound:
            command.extend(["-filter_complex", _sound_graph(duration, has_audio),
                            "-map", "[a]", "-c:a", "aac", "-b:a", "192k"])
        else:
            command.extend(["-map", "0:a:0?", "-c:a", "copy"])
        command.extend(["-t", f"{duration:.6f}", *METADATA_SCRUB, "-movflags", "+faststart", str(target)])
        subprocess.run(command, check=True, capture_output=True, timeout=600)
        os.replace(target, path)
    except subprocess.CalledProcessError as error:
        raise RuntimeError("The timed effects could not be rendered.") from error
    finally:
        target.unlink(missing_ok=True)
        audio_track.unlink(missing_ok=True)
