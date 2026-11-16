"""Central video-encoder selection for every ffmpeg encode call site.

FFMPEG_ENCODER env values:
  x264  (default) — CPU libx264, exact pre-GPU behavior
  nvenc           — force h264_nvenc; probed once and falls back to x264
                    (with a warning) if the GPU/driver is unavailable
  auto            — h264_nvenc when the probe succeeds, else x264

Only the codec/quality args live here; surrounding args (-movflags, -pix_fmt,
audio codecs, filters) stay at each call site.
"""
import os
import subprocess
import threading
import time

# Quality tiers pinning the historical libx264 settings.
QUALITY = "quality"            # was: -preset medium -crf 18
QUALITY_FAST = "quality_fast"  # was: -preset fast -crf 18
DELIVERY = "delivery"          # was: -preset fast -crf 22
# A file the pipeline decodes again straight away (the cut that every render
# reads). libx264 keeps QUALITY_FAST's settings; on NVENC it trades encoder
# effort for bits instead, see _intermediate_nvenc_args.
INTERMEDIATE = "intermediate"

_X264_ARGS = {
    QUALITY: ["-c:v", "libx264", "-preset", "medium", "-crf", "18"],
    QUALITY_FAST: ["-c:v", "libx264", "-preset", "fast", "-crf", "18"],
    DELIVERY: ["-c:v", "libx264", "-preset", "fast", "-crf", "22"],
    INTERMEDIATE: ["-c:v", "libx264", "-preset", "fast", "-crf", "18"],
}

# NVENC -cq is not 1:1 with x264 CRF: benchmarked on the prod GPU (RTX 4000
# Ada), cq ≈ crf + 7 lands in the same file-size ballpark, with vbr + AQ for
# quality. Presets p1-p7: p5 ≈ "medium", p4 ≈ "fast". QUALITY ran p5 until
# oct-2026, then p4, then p3: same hq tune and AQs, ~25% less engine time per
# frame for files ~13% larger at an SSIM to the source within 0.001 of p4. On
# a shared card the engine time matters more than the small gain.
# -pix_fmt yuv420p is REQUIRED: with RGB input (the bgr24 rawvideo pipe from
# OpenCV) nvenc otherwise emits H.264 in gbrp/GBR colorspace, which ffmpeg
# reads fine but web players render as a magenta/green mess.
_NVENC_ARGS = {
    QUALITY: ["-c:v", "h264_nvenc", "-preset", "p3", "-tune", "hq",
              "-rc", "vbr", "-cq", "25", "-b:v", "0",
              "-spatial-aq", "1", "-temporal-aq", "1", "-pix_fmt", "yuv420p"],
    QUALITY_FAST: ["-c:v", "h264_nvenc", "-preset", "p3", "-tune", "hq",
                   "-rc", "vbr", "-cq", "25", "-b:v", "0", "-spatial-aq", "1",
                   "-pix_fmt", "yuv420p"],
    DELIVERY: ["-c:v", "h264_nvenc", "-preset", "p3",
               "-rc", "vbr", "-cq", "29", "-b:v", "0", "-spatial-aq", "1",
               "-pix_fmt", "yuv420p"],
}

# Output args that drop container/stream metadata carried over from the source
# — most notably YouTube's "produced by Google Inc." stream handler, which
# otherwise survives every re-encode (ffmpeg copies input metadata by default)
# and rides into the published clip. The per-stream specifiers are required:
# global -map_metadata -1 alone leaves the audio handler_name intact on a
# stream copy. Empty audio/video specifiers are harmless when a clip lacks that
# stream (ffmpeg ignores them, verified). Spliced in before the output filename
# at each final-artifact producer; kept out of video_encode_args() so that
# stays purely codec/quality args.
METADATA_SCRUB = ["-map_metadata", "-1", "-map_chapters", "-1",
                  "-map_metadata:s:v", "-1", "-map_metadata:s:a", "-1"]

# Loudness normalisation for the delivered clip.
#
# Without this the clip inherits whatever the source was mastered at, so a
# user's clips land anywhere: measured across real delivered clips on
# 26-jul-2026, from -13.8 LUFS on a loud upload down to -28 LUFS on a quiet
# talk. TikTok, Reels and Shorts all normalise playback to roughly -14 LUFS,
# which means the quiet ones just sound thin next to everything else in the
# feed — the loud ones aren't rewarded, the quiet ones are punished.
#
# I=-14 matches the platforms' target, LRA=11 is the usual allowance for speech.
# Applied at the clip cut, where the audio is being encoded to AAC anyway, so it
# costs nothing extra. AUDIO_NORMALIZE=0 turns it off.
#
# TP=-2.0, not the -1.5 that matches the platforms' own advice, because the
# ceiling is enforced BEFORE the AAC encode and the encoder then adds
# inter-sample peaks on top. Measured over 14 corpus clips (31-jul-2026):
#
#   TP=-1.5   peak reached +0.2 dBTP, 1 clip clipping,  8 above -1.0
#   TP=-2.0   peak reached -0.3 dBTP, 0 clipping,       5 above -1.0
#   TP=-3.0   peak reached -0.9 dBTP, 0 clipping,       1 above -1.0
#
# -3.0 also costs level: only 8 of 14 stayed inside -15..-13 LUFS versus 12 at
# -2.0, and level is what the listener notices. Two other fixes were tried and
# do NOT work, so don't reach for them again: an `alimiter` after loudnorm
# (limits sample peaks, not inter-sample, and measured WORSE at +0.7), and
# two-pass loudnorm with linear=true (+0.4, still clipping). The overshoot is
# the codec's, so the only lever is headroom.
LOUDNORM_FILTER = "loudnorm=I=-14:TP=-2.0:LRA=11"


# AI Act art. 50(2): a provider whose system generates synthetic audio or video
# has to mark the output in a machine-readable way. The regulation asks for the
# marking, not for a particular standard, and the cheap one that survives every
# player and every upload is a container tag: `ffprobe -show_format` reads it
# back, and TikTok/Reels/YouTube ignore it. Stamped by a stream copy, so it is
# a remux (no quality loss, ~0.2 s) and never a re-encode.
#
# Deliberately narrow: it goes on outputs where a machine actually synthesised
# voice or a person (dubbing, AI actors), not on an ordinary clip, whose audio
# and pixels are the user's own footage. Marking everything would make the tag
# mean nothing.
AI_DISCLOSURE = "AI-generated content produced with OpenShorts (openshorts.app)"


def mark_ai_generated(path, detail=""):
    """Stamp AI Act art. 50(2) machine-readable tags on a finished file.

    Returns True when the file now carries the tags. Never raises: a missing
    tag must not lose the user the video they just paid minutes for.
    """
    note = f"{AI_DISCLOSURE}: {detail}" if detail else AI_DISCLOSURE
    tmp = f"{path}.aitag.mp4"
    # `comment` and not a custom `ai_generated` key: mp4 only carries the
    # standard iTunes-style tags, and ffmpeg drops anything else without
    # warning (verified with ffprobe -show_entries format_tags).
    # -map 0 because ffmpeg's default picks one stream per type: a dubbed file
    # that ever ships two audio tracks would come back with one.
    cmd = ["ffmpeg", "-y", "-i", path, "-map", "0", "-c", "copy",
           "-metadata", f"comment={note}",
           "-movflags", "+faststart", tmp]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if r.returncode != 0 or not os.path.exists(tmp) or os.path.getsize(tmp) == 0:
            print(f"[ai-tag] skipped for {os.path.basename(path)}: {r.stderr[-300:]}")
            if os.path.exists(tmp):
                os.remove(tmp)
            return False
        os.replace(tmp, path)
        return True
    except Exception as e:
        print(f"[ai-tag] skipped for {os.path.basename(path)}: {e}")
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass
        return False


def audio_encode_args():
    """AAC encode args for a delivered clip, with loudness normalisation."""
    args = ["-c:a", "aac"]
    if os.environ.get("AUDIO_NORMALIZE", "1").strip() != "0":
        args = ["-af", LOUDNORM_FILTER] + args
    return args

_probe_lock = threading.Lock()
_nvenc_ok = None  # None = not probed yet
_announced = False


def _probe_nvenc():
    """One tiny lavfi encode to prove h264_nvenc works end-to-end.

    NVENC rejects frames smaller than ~145px, so the probe uses 256x256.
    Any failure (no ffmpeg binary, no GPU, no driver libs) means False.
    Never waits for an encode slot (bin/ffmpeg): queued behind busy encodes it
    would hit the timeout and turn the whole job to libx264.
    """
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-f", "lavfi", "-i", "color=black:s=256x256:d=0.1",
        "-c:v", "h264_nvenc", "-f", "null", "-",
    ]
    env = {**os.environ, "NVENC_SLOTS": "0"}
    try:
        result = subprocess.run(
            cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=30,
            env=env,
        )
        return result.returncode == 0
    except Exception:
        return False


def nvenc_available():
    """Probe h264_nvenc once and cache the verdict (thread-safe)."""
    global _nvenc_ok
    if _nvenc_ok is None:
        with _probe_lock:
            if _nvenc_ok is None:
                _nvenc_ok = _probe_nvenc()
    return _nvenc_ok


def reset_encoder_cache():
    """Test hook: forget the cached probe result."""
    global _nvenc_ok, _announced
    with _probe_lock:
        _nvenc_ok = None
        _announced = False


def video_encode_args(tier=QUALITY):
    """Return the codec/quality args for one encode, honoring FFMPEG_ENCODER."""
    global _announced
    if tier not in _X264_ARGS:
        raise ValueError(f"Unknown encode tier: {tier!r}")

    mode = os.environ.get("FFMPEG_ENCODER", "x264").strip().lower()
    use_nvenc = False
    if mode in ("nvenc", "auto"):
        use_nvenc = nvenc_available()
        if mode == "nvenc" and not use_nvenc:
            print("⚠️ [Encoder] FFMPEG_ENCODER=nvenc but h264_nvenc is not "
                  "usable here — falling back to libx264")

    if not _announced:
        _announced = True
        print(f"🎞️ [Encoder] video encoder: {'h264_nvenc' if use_nvenc else 'libx264'} "
              f"(FFMPEG_ENCODER={mode})")

    if use_nvenc and tier == INTERMEDIATE:
        return _intermediate_nvenc_args()
    if use_nvenc:
        return list(_NVENC_ARGS[tier])
    return _x264_with_preset(_X264_ARGS[tier])


# libx264 presets a deployment may pick with X264_PRESET (same CRF, less CPU
# per frame for slightly larger files). Unset keeps each tier's own preset.
_X264_PRESETS = ("ultrafast", "superfast", "veryfast", "faster", "fast",
                 "medium", "slow")


def _x264_with_preset(args):
    args = list(args)
    preset = os.environ.get("X264_PRESET", "").strip().lower()
    if preset in _X264_PRESETS:
        args[args.index("-preset") + 1] = preset
    return args


def _intermediate_nvenc_args():
    """NVENC args for INTERMEDIATE: the cheapest preset at a lower -cq.

    The cut is never delivered: the reframe decodes it and encodes the clip
    again. Spending NVENC time on it (p4, hq tune, spatial AQ) bought quality
    the next encode throws away, and on a busy card the cuts were the largest
    share of the encoder's sessions. p1 is several times cheaper per frame; the
    lower -cq spends bits instead so the clip that comes out of the reframe
    stays the same. CUT_NVENC_PRESET / CUT_NVENC_CQ override both.
    """
    preset = os.environ.get("CUT_NVENC_PRESET", "p1").strip() or "p1"
    cq = os.environ.get("CUT_NVENC_CQ", "19").strip() or "19"
    return ["-c:v", "h264_nvenc", "-preset", preset, "-rc", "vbr", "-cq", cq,
            "-b:v", "0", "-pix_fmt", "yuv420p"]


def finished_before_crash(result, outputs):
    """True when ffmpeg died on a signal AFTER finishing every output.

    The multi-output passes (hook, captions, the wm_ copy) sometimes segfault
    while tearing down, after the last trailer is written: each output's final
    stats line ("muxing overhead", printed once its trailer and faststart
    rewrite are done) is already on stderr and the files are complete, byte
    for byte what a clean exit writes. Treating that as a failure re-ran the whole thing as separate
    passes for nothing. Anything short of every output finished still fails.
    """
    if result.returncode >= 0:
        return False
    if (result.stderr or b"").count(b"muxing overhead") < len(outputs):
        return False
    if not all(os.path.exists(p) and os.path.getsize(p) > 0 for p in outputs):
        return False
    print(f"   ℹ️ ffmpeg exited on signal {-result.returncode} after writing all "
          f"{len(outputs)} outputs — keeping them.")
    return True


def blurred_backdrop(out_w, out_h, sigma):
    """Filter chain (no labels) filling out_w x out_h with a blurred copy.

    The blur runs at a quarter of the output size and is scaled up afterwards:
    a blurred picture has no detail to lose, the result looks the same (SSIM
    0.994-0.998 against the full-size gblur on prod clips) and the segment
    encode needs ~45% less CPU. ``sigma`` is the full-size one.
    """
    small_w = max(2, out_w // 4 - (out_w // 4) % 2)
    small_h = max(2, out_h // 4 - (out_h // 4) % 2)
    return (
        f"scale=-2:{small_h},crop=w=min(iw\\,{small_w}):h={small_h},"
        f"gblur=sigma={sigma / 4:g},scale={out_w}:{out_h}"
    )


def escape_filter_value(value):
    r"""Escape a path/value for use inside a quoted FFmpeg filter argument.

    Windows absolute paths are why this exists: ``:`` separates filter options,
    so an interpolated ``C:/x/y.txt`` makes the parser look for an option named
    ``/x/y.txt`` and the whole filtergraph fails to build.

    NOTE: an apostrophe in the path cannot be made safe here. ffmpeg's
    filtergraph parser is not a shell -- the shell idiom ``'\''`` was tried on
    29-jul-2026 and is worse than doing nothing: it drops the apostrophe AND
    swallows the following option, so ``ass='...Earth'\''s.ass':fontsdir='...'``
    resolved to a filename of "...Earths.ass:fontsdir=..." and failed to open.

    The only reliable answer is to keep apostrophes OUT of any path that is
    interpolated into a filter. Callers generate their own filenames, so they
    control this: use a neutral name, never one derived from a video title.
    """
    return value.replace('\\', '/').replace(':', '\\:').replace("'", "\\'")


# An mp4 that ffmpeg abandoned before writing the moov atom is a few dozen
# bytes of ftyp, or nothing at all. Anything a real cut produces is orders of
# magnitude larger, so this only ever catches a failed encode.
MIN_CUT_BYTES = 1024

# Retries of a failed cut, on the SAME encoder, and how long to wait before
# each one. Deliberately not a fallback to libx264: the clip has to come out
# of the GPU like every other one, and a CPU re-encode of a 1080p cut on a box
# that is already busy enough to have failed the first attempt is the wrong
# trade. Waiting is the whole mechanism — whatever the GPU could not give this
# encode, another job finishes and gives back.
CUT_RETRY_WAITS = (3, 9)

# The cut can stay on the card end to end: NVDEC decodes into GPU memory and
# NVENC encodes from it, so the CPU never touches a frame. Measured on a real
# 1080p source: byte-identical decoded frames to the CPU-decoded cut, 64% less
# CPU (25-sep-2026). Only for 8-bit 4:2:0 in codecs the card decodes: anything
# else would reach nvenc in a pixel format a delivered H.264 must not have, so
# it keeps the CPU decode. FFMPEG_GPU_CUT=0 turns it off.
_GPU_CUT_CODECS = {"h264", "hevc", "vp9", "av1"}
_gpu_cut_sources = {}


def _source_format(path):
    """(codec, pix_fmt) of the first video stream, or None."""
    if not os.path.exists(path):
        return None
    try:
        out = subprocess.check_output(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=codec_name,pix_fmt", "-of", "csv=p=0", path],
            stderr=subprocess.DEVNULL, timeout=60, text=True).strip()
        codec, pix_fmt = out.split(",")[:2]
        return codec, pix_fmt
    except Exception:
        return None


def _gpu_cut_ok(input_video):
    if os.environ.get("FFMPEG_GPU_CUT", "1").strip() == "0":
        return False
    if os.environ.get("FFMPEG_ENCODER", "x264").strip().lower() not in ("nvenc", "auto"):
        return False
    if input_video not in _gpu_cut_sources:
        fmt = _source_format(input_video) if nvenc_available() else None
        _gpu_cut_sources[input_video] = bool(
            fmt and fmt[0] in _GPU_CUT_CODECS and fmt[1] == "yuv420p")
    return _gpu_cut_sources[input_video]


def cut_clip(input_video, clip_temp_path, start, end, clip_number):
    """Cut [start, end] out of the source into ``clip_temp_path``.

    Raises RuntimeError with ffmpeg's own stderr when the cut does not produce
    a playable file. That report is the point: since dec-2025 the cut ran with
    its return code ignored and stderr captured into a pipe nobody read, so a
    failed cut handed an empty file to the reframer and the job's only visible
    error was "moov atom not found" three layers downstream — from ffmpeg,
    TransNetV2 and PySceneDetect in turn, each naming the temp file rather
    than the encode that never wrote it (prod, 9-sep-2026: three clips of one
    job lost, cause unrecoverable because the stderr had been discarded).

    A failed cut is retried on the same encoder after a wait. The failure this
    exists for is transient: the same command, on the same source file, cut
    fine by hand minutes later, and the nvenc probe is a 256x256 lavfi frame
    cached for the life of the process, so it stays true while a real 1080p
    session cannot allocate on a GPU that other jobs are filling.
    """
    encode_args = video_encode_args(INTERMEDIATE)
    command = [
        'ffmpeg', '-y',
        '-ss', str(start),
        '-to', str(end),
        '-i', input_video,
        *encode_args,
        *audio_encode_args(),
        clip_temp_path
    ]
    gpu_command = None
    if _gpu_cut_ok(input_video):
        # CUDA frames are already 4:2:0 (nv12); -pix_fmt would force a copy
        # back to system memory.
        gpu_args = list(encode_args)
        if "-pix_fmt" in gpu_args:
            i = gpu_args.index("-pix_fmt")
            del gpu_args[i:i + 2]
        gpu_command = ['ffmpeg', '-y', '-hwaccel', 'cuda', '-hwaccel_output_format', 'cuda',
                       '-ss', str(start), '-to', str(end), '-i', input_video,
                       *gpu_args, *audio_encode_args(), clip_temp_path]

    def _run_one(cmd):
        result = subprocess.run(cmd, stdout=subprocess.DEVNULL,
                                stderr=subprocess.PIPE, text=True, errors="replace")
        # ffmpeg has been seen exiting 0 having written nothing, so the file
        # itself is the verdict, not just the return code.
        size = os.path.getsize(clip_temp_path) if os.path.exists(clip_temp_path) else 0
        ok = result.returncode == 0 and size >= MIN_CUT_BYTES
        return ok, f"exit {result.returncode}, {size} bytes\n{(result.stderr or '').strip()[-800:]}"

    def _run():
        nonlocal gpu_command
        if gpu_command:
            ok, report = _run_one(gpu_command)
            if ok:
                return ok, report
            # Not worth a second try on this source: back to the CPU decode.
            print(f"   ⚠️ GPU cut of clip {clip_number} failed — decoding on the CPU.")
            gpu_command = None
            _gpu_cut_sources[input_video] = False
        return _run_one(command)

    for attempt, wait in enumerate(CUT_RETRY_WAITS + (None,), start=1):
        ok, report = _run()
        if ok:
            if attempt > 1:
                print(f"   ✅ Clip {clip_number} cut on attempt {attempt}.")
            return
        if wait is None:
            break
        print(f"   ⚠️ Cut of clip {clip_number} failed ({report}) — "
              f"retrying in {wait}s.")
        time.sleep(wait)

    raise RuntimeError(
        f"ffmpeg could not cut clip {clip_number} ({start}s-{end}s) from "
        f"{os.path.basename(input_video)} in {len(CUT_RETRY_WAITS) + 1} "
        f"attempts: {report}")


