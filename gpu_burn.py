"""Burn a clip's hook, captions and free-plan mark on the card (``osrender burn``).

The CPU burn is one ffmpeg (hooks.add_hook_to_video): the clip is decoded,
``overlay`` lays the hook image on it for its first seconds, ``ass`` draws the
captions, a second ``overlay`` adds the mark on the served copy, and every
picture is encoded once per output. ``osrender burn`` does the same on the
card: NVDEC decodes, CUDA kernels repeat the three blends with the filters'
own integer arithmetic (libass is the same library, set up the same way) and
NVENC encodes each output straight from device memory; the audio is copied.
The decoded pictures are the ones the ffmpeg pass writes
(``tests/test_gpu_burn_equivalence.py`` checks it frame by frame).

The static images are converted to yuva420p by ffmpeg itself, through the
same filters the burn graph applies to them, so their colour conversion is
swscale's and not a second implementation.

Used only for what it reproduces exactly: an ``ass=filename=..:fontsdir=..``
caption filter, the hook as an image overlay and the mark chain
``main.watermark_filter`` builds. Anything else, a source NVDEC cannot decode
or any failure burns with ffmpeg as before. ``GPU_BURN=0`` turns it off.
"""
import contextlib
import json
import os
import re
import subprocess
import tempfile
import time

import gpu_render
from ffmpeg_utils import QUALITY

ENABLED = os.environ.get("GPU_BURN", "1").strip() != "0"
# A stuck burn falls back to ffmpeg after this, not after ffmpeg's 30 min.
MIN_TIMEOUT = 120
TIMEOUT_PER_SECOND = 6

_ASS_VF = re.compile(r"^ass=filename='((?:[^'\\]|\\.)*)':fontsdir='((?:[^'\\]|\\.)*)'$")
_MARK = re.compile(r"^\[l\](?P<chain>[^\[\];]+)\[wm\];\[v\]\[wm\]"
                   r"overlay=x=(?P<x>\d+):y=(?P<y>\d+)\[o\]$")
_STAGES = ("hook", "ass", "mark")


def _unescape(value):
    """Undo ffmpeg_utils._escape_ffmpeg_filter_value (a backslash before a char)."""
    return re.sub(r"\\(.)", r"\1", value)


def parse_ass(vf):
    """(ass file, fonts dir) of the caption filter subtitles_filter builds for
    an .ass file, or None for any other filter."""
    m = _ASS_VF.match(vf or "")
    if not m:
        return None
    return _unescape(m.group(1)), _unescape(m.group(2))


def parse_mark(make_filter, video_w, video_h):
    """(logo chain, x, y) of the mark ``make_filter`` (main.watermark_filter)
    draws on a video_w x video_h picture, or None if it is not the plain
    scale/format/colorchannelmixer chain + overlay this module reproduces."""
    try:
        graph = make_filter(video_w, video_h, video="[v]", logo="[l]", out="[o]")
    except Exception:
        return None
    m = _MARK.match(graph or "")
    if not m:
        return None
    chain = m.group("chain")
    for f in chain.split(","):
        if not re.match(r"^(scale=\d+:-1|format=rgba|colorchannelmixer=aa=[0-9.]+)$", f):
            return None
    return chain, int(m.group("x")), int(m.group("y"))


def usable(video_path):
    return ENABLED and gpu_render.usable(video_path)


# ffprobe's colour space names that ffmpeg's scale filter takes as a matrix.
_MATRICES = {"bt709": "bt709", "smpte170m": "smpte170m", "bt470bg": "bt470"}


def _planes(image, chain, out, matrix=None):
    """``image`` through ``chain`` (the burn graph's own filters for it), as
    the yuva420p picture ``overlay`` blends, in a one-frame NUT file. In the
    burn graph ffmpeg converts it to the clip's colour space, so a clip that
    names one (``matrix``) gets that conversion here too."""
    convert = f"scale=out_color_matrix={matrix}:out_range=tv," if matrix else ""
    graph = f"[0:v]{chain + ',' if chain else ''}{convert}format=yuva420p[o]"
    cmd = ["ffmpeg", "-v", "error", "-y", "-i", image, "-filter_complex", graph,
           "-map", "[o]", "-frames:v", "1", "-c:v", "rawvideo", "-f", "nut", out]
    subprocess.run(cmd, check=True, capture_output=True, timeout=60)


def _probe(path):
    """(frames by the header, duration, colour matrix for _planes) of the
    first video stream. Frames is None when unknown or when the clip is in a
    colour space or range the GPU burn does not reproduce."""
    try:
        out = subprocess.check_output(
            ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
             "stream=nb_frames,duration,color_space,color_range", "-of", "json", path],
            stderr=subprocess.DEVNULL, timeout=60, text=True)
        stream = (json.loads(out).get("streams") or [{}])[0]
        n = int(stream.get("nb_frames") or 0)
        duration = float(stream.get("duration") or 0)
    except (OSError, ValueError, subprocess.SubprocessError):
        return None, 0.0, None
    space = str(stream.get("color_space") or "unknown")
    if str(stream.get("color_range") or "unknown") not in ("unknown", "tv"):
        return None, duration, None
    if space in ("unknown", "unspecified"):
        return n or None, duration, None
    if space not in _MATRICES:
        return None, duration, None
    return n or None, duration, _MATRICES[space]


def burn(video_path, outputs, hook=None, ass_vf=None, mark=None, video_size=None):
    """Write ``outputs`` ([(stage, path)], stage one of hook / ass / mark) from
    ``video_path`` on the card. ``hook`` = (image, x, y, seconds or None);
    ``ass_vf`` = the caption filter; ``mark`` = (logo image, make_filter).
    True when every output was written; False (nothing left behind) when the
    ffmpeg pass should run instead."""
    if not outputs or not usable(video_path):
        return False
    stages = [s for s, _ in outputs]
    if any(s not in _STAGES for s in stages):
        return False
    ass = None
    if ass_vf:
        ass = parse_ass(ass_vf)
        if ass is None:
            return False
    elif any(s in ("ass", "mark") for s in stages) and mark is None and hook is None:
        return False
    mark_spec = None
    if mark is not None:
        w, h = video_size or (0, 0)
        mark_spec = parse_mark(mark[1], w, h)
        if mark_spec is None:
            return False
    elif "mark" in stages:
        return False
    frames, duration, matrix = _probe(video_path)
    if not frames:
        return False

    paths = [p for _, p in outputs]
    try:
        with tempfile.TemporaryDirectory(prefix="osburn_") as tmp:
            spec = {"input": video_path,
                    "outputs": [{"stage": s, "path": p} for s, p in outputs],
                    "encoder": gpu_render.encoder_options(QUALITY)}
            threads = os.environ.get("FFMPEG_THREADS", "2")
            spec["threads"] = int(threads) if threads.isdigit() else 2
            if hook is not None:
                image, x, y, seconds = hook
                nut = os.path.join(tmp, "hook.nut")
                _planes(image, "", nut, matrix)
                spec["hook"] = {"planes": nut, "x": int(x), "y": int(y),
                                "until": float(seconds) if seconds else None}
            if ass is not None:
                spec["ass"] = {"file": ass[0], "fonts_dir": ass[1]}
            if mark_spec is not None:
                chain, x, y = mark_spec
                nut = os.path.join(tmp, "mark.nut")
                _planes(mark[0], chain, nut, matrix)
                spec["mark"] = {"planes": nut, "x": x, "y": y, "until": None}
            spec_path = os.path.join(tmp, "spec.json")
            with open(spec_path, "w") as f:
                json.dump(spec, f)
            timeout = max(MIN_TIMEOUT, TIMEOUT_PER_SECOND * duration)
            with gpu_render.nvenc_slot():
                started = time.time()
                result = subprocess.run([gpu_render.BIN, "burn", spec_path],
                                        capture_output=True, text=True, timeout=timeout)
        m = re.search(r"osrender: (\d+) frames", result.stderr or "")
        written = int(m.group(1)) if m else 0
        if result.returncode != 0 or written != frames:
            why = ((result.stderr or "").strip().splitlines()[-1:] or ["no output"])[0][:200]
            if result.returncode == 0:
                why = f"{written} frames, the clip has {frames}"
            print(f"   ⚠️ GPU burn failed ({why}); burning with ffmpeg")
            _remove(paths)
            return False
        if not all(os.path.isfile(p) and os.path.getsize(p) > 0 for p in paths):
            print("   ⚠️ GPU burn left an output missing; burning with ffmpeg")
            _remove(paths)
            return False
        print(f"   ⚡ GPU burn: {written} frames x {len(paths)} in {time.time() - started:.1f}s")
        return True
    except (OSError, subprocess.SubprocessError) as e:
        print(f"   ⚠️ GPU burn failed ({type(e).__name__}); burning with ffmpeg")
        _remove(paths)
        return False


def _remove(paths):
    for p in paths:
        with contextlib.suppress(OSError):
            os.remove(p)
