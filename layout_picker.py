"""Ask Gemini which layout a video needs, once per video.

The four previous attempts at this problem asked the model (or a pixel
heuristic) to MEASURE something — edge density, MSER text density, what fraction
of the duration had content on screen, what fraction of the width it spanned —
and let thresholds turn that number into a routing decision. All four failed the
same way: they could not separate a spreadsheet from a corner scoreboard.

This asks for the decision itself. Measured over the 48-clip corpus against
hand-checked labels (`labels-layout.json` in the reframe-testing skill), three
runs:

    run 1   45/48  94%   18/20 content found   1 false positive of 28
    run 2   44/48  92%   17/20                 1 of 28
    run 3   46/48  96%   18/20                 0 of 28

Two clips out of 48 changed answer between runs. That matters because the
earlier note in this repo said Gemini was too non-deterministic to build on —
the same video scored 1% and 97% coverage on consecutive runs. The variance was
in asking for a continuous measurement, not in the model: a categorical choice
between closed options is stable.

It samples FRAMES instead of uploading the video. Gemini bills video at ~300
tokens per second, so an hour of source is ~1.08M tokens — past a 1M context
window before it starts — and means pushing a 1-2GB upload to get back one word.
This pipeline ingests hour-long podcasts, so that scales badly in exactly the
normal case. Twelve stills cost ~3k tokens whatever the source runs to.

Measured on the same 48 clips (whole video vs sampled frames):

    whole video          94% / 92% / 96%    18,17,18 of 20 found
    12 frames @ 640px    90% / 90%          15,16 of 20
    12 frames @ 1024px   92%                17 of 20
    24 frames @ 1024px   90%                16 of 20

Resolution was the gap, not frame count: at 640px a spreadsheet is unreadable,
and doubling the frames made it slightly worse rather than better. At 1024px the
difference from sending the whole video sits inside the run-to-run variance the
whole-video mode already has, at 2.2s per clip instead of ~15s.

Off by default (``AUTO_LAYOUT=1``). A caller that already switched layouts on
by hand wins: this only ever ADDS, so an explicit choice is never overridden.
"""
import json
import os
import threading

# AUTO_LAYOUT=1 decides and applies. AUTO_LAYOUT=shadow decides, logs, and
# applies NOTHING: the render is byte-for-byte what it would have been.
#
# Shadow exists because everything measured about this picker was measured on 48
# YouTube clips chosen by hand, and the material users actually upload is a
# different distribution nobody has looked at. A week of shadow answers "what
# does it say about OUR videos" for 0.002 USD and ~2s per video, with no way to
# damage a clip somebody paid for.
_MODE = os.environ.get("AUTO_LAYOUT", "0").strip().lower()
SHADOW = _MODE == "shadow"
ENABLED = _MODE == "1" or SHADOW

# 12 frames at 1024px wide. Both numbers are measured, not guessed: see above.
SAMPLE_FRAMES = int(os.environ.get("LAYOUT_SAMPLE_FRAMES", "12"))
SAMPLE_WIDTH = int(os.environ.get("LAYOUT_SAMPLE_WIDTH", "1024"))

# What each decision turns on. Keys match the layout names in the prompt.
DECISION_FLAGS = {
    "none": [],
    "screencast": ["screencast_layout"],
    "split": ["split_layout", "active_speaker"],
}

VALID = set(DECISION_FLAGS)

# On for every video the picker sees, whatever it answers. These modules decide
# per SCENE with their own guards (split_layout: only scenes already GENERAL, two
# faces of a real size side by side in the same frame for half the samples, at
# least 2.5 s; active_speaker: both mouths move). The video-level "split" answer
# is the wrong gate for them: an edited podcast cuts between close-ups and a wide
# two-shot, the sampled frames are mostly close-ups, the model rightly answers
# "none", and every two-shot rendered GENERAL, two small people floating in a
# blurred strip. On such a podcast the per-scene check stacks the 39 s two-shot
# and leaves the close-ups on TRACK.
PER_SCENE_FLAGS = ["split_layout", "active_speaker"]


def _module_flags(decision):
    """Modules to enable for a decision, ignoring anything unrecognised."""
    return DECISION_FLAGS.get(str(decision or "none").strip().lower(), [])


def _flags_to_enable(decision):
    """PER_SCENE_FLAGS plus whatever this decision adds, without repeats."""
    flags = list(PER_SCENE_FLAGS)
    flags += [f for f in _module_flags(decision) if f not in flags]
    return flags


def apply(decision):
    """Switch on the modules a decision needs. Returns the modules touched.

    Deliberately additive: an operator who set SPLIT_LAYOUT=1 for a job wants
    stacking regardless of what the model thinks, and a model that says "none"
    must not quietly undo that. PER_SCENE_FLAGS go on for every decision,
    including "none" and a failed call.
    """
    import active_speaker
    import screencast_layout
    import split_layout

    modules = {"split_layout": split_layout,
               "screencast_layout": screencast_layout,
               "active_speaker": active_speaker}

    touched = []
    for name in _flags_to_enable(decision):
        module = modules.get(name)
        if module is not None and not getattr(module, "ENABLED", False):
            module.ENABLED = True
            touched.append(name)
    return touched


def _encode_frame(frame, width):
    import cv2

    h, w = frame.shape[:2]
    scaled = cv2.resize(frame, (width, max(2, int(h * width / w))),
                        interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", scaled, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return buf.tobytes() if ok else None


def _ffmpeg_frames(video_path, n):
    """The same evenly spread frames, decoded by the ffmpeg CLI.

    OpenCV's bundled FFmpeg has no software AV1 decoder, so on the AV1 sources
    YouTube often serves every read failed ("Failed to get pixel format") and
    the picker fell back to the default layout for the whole video (prod,
    25-sep-2026). The system ffmpeg decodes AV1 with libdav1d.
    """
    import subprocess

    import cv2
    import numpy as np

    try:
        probe = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=nw=1:nk=1", video_path],
            capture_output=True, text=True, timeout=60)
        duration = float(probe.stdout.strip())
    except (OSError, ValueError, subprocess.SubprocessError):
        return []
    frames = []
    for i in range(n):
        try:
            r = subprocess.run(
                ["ffmpeg", "-v", "error", "-ss", f"{i * duration / n:.3f}",
                 "-i", video_path, "-frames:v", "1", "-c:v", "png",
                 "-f", "image2pipe", "-"],
                capture_output=True, timeout=120)
        except (OSError, subprocess.SubprocessError):
            continue
        frame = cv2.imdecode(np.frombuffer(r.stdout, np.uint8), cv2.IMREAD_COLOR)
        if frame is not None:
            frames.append(frame)
    return frames


_sampled = {"key": None, "frames": None}
_sampled_lock = threading.Lock()


def _sample_key(video_path, n, width):
    try:
        st = os.stat(video_path)
    except OSError:
        return None
    return (os.path.realpath(video_path), st.st_size, st.st_mtime_ns, n, width)


def sample_frames(video_path, n=None, width=None):
    """JPEG bytes for ``n`` frames spread evenly across the video.

    The last answer is kept: the layout choice and the burned-subtitle check
    ask for the same frames of the same source one after the other, and each
    is a seek into a long-GOP file (12 samples of a 1080p60 source: 15 CPU-s).
    """
    import cv2

    n = n or SAMPLE_FRAMES
    width = width or SAMPLE_WIDTH
    key = _sample_key(video_path, n, width)
    with _sampled_lock:
        if key is not None and _sampled["key"] == key:
            return list(_sampled["frames"])
    out = _sample_frames(video_path, n, width, cv2)
    if key is not None and out:
        with _sampled_lock:
            _sampled["key"], _sampled["frames"] = key, list(out)
    return out


def _sample_frames(video_path, n, width, cv2):
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    out = []
    try:
        if total > 0:
            for i in range(n):
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(i * total / n))
                ok, frame = cap.read()
                if not ok:
                    continue
                jpg = _encode_frame(frame, width)
                if jpg:
                    out.append(jpg)
    finally:
        cap.release()
    if not out and os.path.exists(video_path):
        out = [j for j in (_encode_frame(f, width) for f in _ffmpeg_frames(video_path, n)) if j]
    return out


def pick(video_path, video_duration):
    """The layout Gemini picks for this video, or "none" on any failure.

    Never raises: a missing answer has to degrade to today's routing rather
    than break the job.
    """
    if not ENABLED:
        return "none"
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return "none"

    model_name = os.environ.get("GEMINI_MODEL") or 'gemini-3.1-flash-lite'
    print("🎛️  Choosing a layout for this video…")
    try:
        # Inside the try on purpose: the contract above is that this never
        # raises, and an unimportable SDK is just one more reason to fall back.
        from google import genai
        from google.genai import types as genai_types
        import gemini_worker

        frames = sample_frames(video_path)
        if not frames:
            print("   ⚠️ No readable frames — keeping the default layout.")
            return "none"

        client = genai.Client(api_key=api_key)
        parts = [genai_types.Part.from_bytes(data=b, mime_type="image/jpeg")
                 for b in frames]
        response = client.models.generate_content(
            model=model_name,
            contents=parts + [gemini_worker.LAYOUT_CHOICE_PROMPT],
            config=genai_types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=gemini_worker.LayoutChoice,
            ))
        gemini_worker.raise_if_blocked(response)
        answer = json.loads(response.text) or {}
    except Exception as e:
        print(f"   ⚠️ Layout choice failed ({e}) — keeping the default layout.")
        return "none"

    decision = str(answer.get("layout", "none")).strip().lower()
    if decision not in VALID:
        print(f"   ⚠️ Unknown layout '{decision}' — keeping the default layout.")
        return "none"

    why = str(answer.get("why", ""))[:80]
    confidence = answer.get("confidence")
    print(f"   🎬 Layout: {decision} (confianza {confidence}) — {why}")
    return decision


def pick_and_apply(video_path, video_duration):
    """Decide, switch on (unless shadowing), report what changed."""
    decision = pick(video_path, video_duration)

    if SHADOW:
        # One greppable line per job. Deliberately not routed through the
        # analytics module: that one is opt-in and host-scoped, and a shadow
        # run has to work on any deployment, including self-hosted.
        would = _flags_to_enable(decision)
        print(f"[layout-shadow] decision={decision} "
              f"would_enable={','.join(would) if would else 'none'} "
              f"duration={video_duration:.0f}s")
        return decision

    touched = apply(decision)
    if touched:
        print(f"   ✅ Enabled: {', '.join(touched)}")
    return decision
