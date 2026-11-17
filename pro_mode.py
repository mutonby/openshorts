"""Pro mode: a talking-head clip re-edited the way a short-form editor would.

It works on a clip that is ALREADY framed (the clean 9:16 file) and does three
things to it:

- the dead air goes: silences are measured in the audio and cut, never inside
  a word;
- every cut changes the scale a little, the frame sits tighter on the face and
  follows it, so the jump of the cut reads as a camera change;
- the picture is graded and the background behind the speaker is dimmed
  (person segmentation), so the face is what is lit.

That only makes sense when the clip is one person talking to camera. On a
screen recording, a split or a wide shot there is no face to frame, and
cutting inside the picture pulls what is shown away from what is said.
``suggested()`` is the cheap test for that (the layouts the render recorded
plus a face check on the clip's keyframes).

Cost. The render walks every frame through Python, which the GPU pipeline
never does: it is CPU time of the same order as the whole rest of a job on a
GPU host. So ``PRO_MODE`` decides what the job pipeline does on its own
(``off``, ``suggest`` = only flag the clips it fits, ``auto`` = also render
them), ``PRO_SLOTS`` caps the renders running on one host, and a render is
always its own process (``run``), never a thread of the caller.

The clean pro file is a recut like any other (``recut_<ts>_...``, with the
kept stretches in the clip's ``recipe``), so captions, hook, watermark and the
clip editor treat it as they treat a manual trim.
"""
import json
import os
import re
import subprocess
import sys
import threading
import time
import wave

import numpy as np

SAMPLE_RATE = 48000

# --- cuts ---------------------------------------------------------------------
MIN_SILENCE = 0.36       # a pause shorter than this stays
KEEP_BEFORE = 0.13       # air left after the last word
KEEP_AFTER = 0.10        # air left before the next one
MIN_KEEP = 0.30          # a kept stretch shorter than this is not worth a cut
MIN_RESULT_SECONDS = 5.0  # never cut a clip down to less than this

# --- shots --------------------------------------------------------------------
TRACK_FPS = 5            # face samples per second
LEVELS = (1.00, 1.12, 1.05, 1.16)   # scale of consecutive shots, cycling
FACE_HEIGHT = 0.30       # face box / frame height a shot starts from
FACE_HEIGHT_MAX = 0.37   # ... and never exceeds
HEAD_TOP = 0.135         # where the top of the head sits, as a frame fraction
PUSH_PER_SECOND = 0.008  # slow push-in inside a shot
PUSH_MAX = 0.05
MIN_BEAT = 1.6           # never two scale changes closer than this
MAX_BEAT = 4.5           # nor a shot held longer than this

# --- look ---------------------------------------------------------------------
BACKGROUND_GAIN = 0.24   # how much light the background keeps
SEGMENT_EVERY = 2        # person mask refreshed every N frames
MASK_SIZE = (288, 512)

# --- titles -------------------------------------------------------------------
TITLE_COLOR = "#FF5A1F"
TITLE_FONT = "Archivo Black"     # fonts/ArchivoBlack-Regular.ttf
TITLE_SECONDS = 1.45
TITLE_GAP = 2.5          # seconds between two titles
TITLE_LIMIT = 5
TITLE_MAX_CHARS = 14

# --- suggestion ---------------------------------------------------------------
KEYFRAME_LIMIT = 16
MIN_TRACK_SHARE = 0.9
MIN_FACE_SHARE = 0.7
FACE_RANGE = (0.07, 0.60)   # face box height / frame height

RENDER_TIMEOUT_SECONDS = 1800


class Busy(RuntimeError):
    """Every pro render slot of this host stayed taken for the whole wait."""


class NotApplicable(RuntimeError):
    """The clip is not something pro mode can be rendered on."""


def mode():
    """What the job pipeline does on its own: off | suggest | auto."""
    value = os.environ.get("PRO_MODE", "suggest").strip().lower()
    return value if value in ("off", "suggest", "auto") else "suggest"


def _run(cmd, **kw):
    return subprocess.run(cmd, check=True, capture_output=True,
                          stdin=subprocess.DEVNULL, **kw)


def probe(path):
    """(width, height, fps, rate string, duration) of the first video stream."""
    out = _run(["ffprobe", "-v", "error", "-select_streams", "v:0",
                "-show_entries", "stream=width,height,r_frame_rate:format=duration",
                "-of", "json", path])
    data = json.loads(out.stdout)
    stream = data["streams"][0]
    rate = str(stream["r_frame_rate"])
    num, _, den = rate.partition("/")
    fps = float(num) / float(den or 1)
    if not re.fullmatch(r"\d+(/\d+)?", rate) or fps <= 0:
        raise NotApplicable("unreadable frame rate")
    return (int(stream["width"]), int(stream["height"]), fps, rate,
            float(data["format"]["duration"]))


# ------------------------------------------------------------------ suggestion
def track_share(ranges, duration):
    """Fraction of the clip the render framed on one person (TRACK), or None
    when it recorded no layouts at all."""
    total = track = 0.0
    for r in ranges or []:
        try:
            span = max(0.0, float(r["end"]) - float(r["start"]))
        except (KeyError, TypeError, ValueError):
            continue
        total += span
        if str(r.get("layout", "")).lower() == "track":
            track += span
    if total <= 0:
        return None
    return track / max(total, float(duration or 0) or total)


def faces_look_like_one_speaker(samples):
    """``samples``: per frame, None or (faces, face height / frame height)."""
    if len(samples) < 2:
        return False
    single = [s for s in samples if s and s[0] == 1]
    if len(single) < MIN_FACE_SHARE * len(samples):
        return False
    height = float(np.median([s[1] for s in single]))
    return FACE_RANGE[0] <= height <= FACE_RANGE[1]


def _count_faces(detections):
    """(real faces, largest box) — a face under 40 % of the largest one's area
    is someone in the background, not a second speaker."""
    boxes = [d.location_data.relative_bounding_box for d in detections or []]
    if not boxes:
        return 0, None
    best = max(boxes, key=lambda b: b.width * b.height)
    floor = 0.4 * best.width * best.height
    return sum(1 for b in boxes if b.width * b.height >= floor), best


# A job checks its clips from several threads; one detector at a time.
_DETECT_LOCK = threading.Lock()


def _keyframe_faces(path):
    """Face check on the clip's keyframes only: decoding them costs a fraction
    of a second, where sampling evenly would decode the whole clip."""
    with _DETECT_LOCK:
        return _keyframe_faces_locked(path)


def _keyframe_faces_locked(path):
    import mediapipe as mp
    sw, sh = 270, 480
    proc = subprocess.Popen(
        ["ffmpeg", "-v", "error", "-skip_frame", "nokey", "-i", path, "-an",
         "-vf", f"scale={sw}:{sh}", "-fps_mode", "passthrough",
         "-frames:v", str(KEYFRAME_LIMIT), "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, stdin=subprocess.DEVNULL)
    samples = []
    size = sw * sh * 3
    detector = mp.solutions.face_detection.FaceDetection(
        model_selection=1, min_detection_confidence=0.5)
    try:
        while True:
            buf = proc.stdout.read(size)
            if len(buf) < size:
                break
            frame = np.frombuffer(buf, dtype=np.uint8).reshape(sh, sw, 3)
            count, best = _count_faces(detector.process(frame).detections)
            samples.append((count, best.height) if best else None)
    finally:
        detector.close()
        proc.stdout.close()
        proc.wait()
    return samples


def suggested(path, ranges=None):
    """True when the clip is one person talking to camera, which is what pro
    mode is for. Never raises: a clip it cannot judge is simply not flagged."""
    try:
        width, height, _fps, _rate, duration = probe(path)
        if height <= width:
            return False
        share = track_share(ranges, duration)
        if share is not None and share < MIN_TRACK_SHARE:
            return False
        return faces_look_like_one_speaker(_keyframe_faces(path))
    except Exception as e:
        print(f"   ℹ️ Pro mode check skipped ({type(e).__name__}: {e})")
        return False


# ------------------------------------------------------------------------ cuts
def load_audio(path):
    """Stereo int16 samples at SAMPLE_RATE, shape (n, 2); empty without audio."""
    try:
        out = _run(["ffmpeg", "-v", "error", "-i", path, "-vn", "-ac", "2",
                    "-ar", str(SAMPLE_RATE), "-f", "s16le", "-"])
    except subprocess.CalledProcessError:
        return np.zeros((0, 2), dtype=np.int16)
    data = np.frombuffer(out.stdout, dtype=np.int16)
    return data[:len(data) // 2 * 2].reshape(-1, 2)


def envelope_db(audio, hop=0.01):
    """RMS level in dB every ``hop`` seconds."""
    mono = audio.astype(np.float32).mean(axis=1) / 32768.0
    n = int(SAMPLE_RATE * hop)
    frames = len(mono) // n
    if frames == 0:
        return np.zeros(0, dtype=np.float32)
    rms = np.sqrt((mono[:frames * n].reshape(frames, n) ** 2).mean(axis=1) + 1e-10)
    return (20 * np.log10(rms + 1e-10)).astype(np.float32)


def silences(env, hop=0.01):
    """(start, end) of the stretches that are really silent, measured in the
    audio: a transcript stretches its words over the pauses."""
    if len(env) == 0:
        return []
    speech = np.percentile(env, 92)
    floor = np.percentile(env, 5)
    quiet = env < max(speech - 29.0, floor + 4.0)
    runs, start = [], None
    for i, q in enumerate(quiet):
        if q and start is None:
            start = i
        elif not q and start is not None:
            runs.append((start * hop, i * hop))
            start = None
    if start is not None:
        runs.append((start * hop, len(quiet) * hop))
    return [(a, b) for a, b in runs if b - a >= MIN_SILENCE]


def keep_frames(silent, words, duration, fps):
    """[(first frame, end frame)] of what survives once ``silent`` is cut.

    The threshold is strict and the air generous, so a trailing "s" or "f" is
    never eaten. Falls back to the whole clip when nothing is worth cutting or
    the result would be too short to be a clip."""
    whole = [(0, max(1, int(round(duration * fps))))]
    first = words[0]["s"] if words else 0.0
    last = words[-1]["e"] if words else duration
    removed = []
    for a, b in silent:
        if a <= 0.02:
            cut = (0.0, max(0.0, min(b, first + 0.05) - KEEP_AFTER))
        elif b >= duration - 0.02:
            cut = (max(a, last) + 0.25, duration)
        else:
            cut = (a + KEEP_BEFORE, b - KEEP_AFTER)
        if cut[1] - cut[0] >= 0.12:
            removed.append(cut)
    removed.sort()
    keep, cursor = [], 0.0
    for a, b in removed:
        if a - cursor >= MIN_KEEP:
            keep.append((cursor, a))
        cursor = max(cursor, b)
    if duration - cursor >= MIN_KEEP:
        keep.append((cursor, duration))
    frames = []
    for a, b in keep:
        fa, fb = int(round(a * fps)), int(round(b * fps))
        if fb - fa >= int(MIN_KEEP * fps):
            frames.append((fa, fb))
    if not frames or sum(b - a for a, b in frames) < MIN_RESULT_SECONDS * fps:
        return whole
    return frames


def to_output_time(frames, fps):
    """Map a time on the uncut clip to the cut one (a time inside a removed
    stretch lands on the start of the next kept one)."""
    spans, out = [], 0.0
    for fa, fb in frames:
        spans.append((fa / fps, fb / fps, out))
        out += (fb - fa) / fps
    total = out

    def convert(t):
        for a, b, o in spans:
            if t < a:
                return o
            if t <= b:
                return o + (t - a)
        return total
    return convert


def source_segments(keep, segments):
    """Carry kept stretches (seconds on the clip's own timeline) back to the
    source timeline the clip's ``segments`` were cut from: the pro file is a
    recut of the same source, and its recipe has to say so."""
    out = []
    for a, b in keep:
        offset = 0.0
        for seg in segments:
            start, end = float(seg["start"]), float(seg["end"])
            length = end - start
            lo, hi = max(a, offset), min(b, offset + length)
            if hi - lo > 0.001:
                piece = {"start": round(start + lo - offset, 3),
                         "end": round(start + hi - offset, 3)}
                if out and abs(out[-1]["end"] - piece["start"]) < 0.002:
                    out[-1]["end"] = piece["end"]
                else:
                    out.append(piece)
            offset += length
    return out


# ----------------------------------------------------------------------- shots
def face_track(path, width, height):
    """Per sample (TRACK_FPS a second): None, or (faces, centre x, centre y,
    box height) in pixels of the clip."""
    import mediapipe as mp
    sw, sh = 270, 480
    proc = subprocess.Popen(
        ["ffmpeg", "-v", "error", "-i", path, "-an",
         "-vf", f"fps={TRACK_FPS},scale={sw}:{sh}", "-f", "rawvideo",
         "-pix_fmt", "rgb24", "-"],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, stdin=subprocess.DEVNULL)
    detector = mp.solutions.face_detection.FaceDetection(
        model_selection=1, min_detection_confidence=0.5)
    out = []
    size = sw * sh * 3
    try:
        while True:
            buf = proc.stdout.read(size)
            if len(buf) < size:
                break
            frame = np.frombuffer(buf, dtype=np.uint8).reshape(sh, sw, 3)
            count, box = _count_faces(detector.process(frame).detections)
            out.append((count, (box.xmin + box.width / 2) * width,
                        (box.ymin + box.height / 2) * height,
                        box.height * height) if box else None)
    finally:
        detector.close()
        proc.stdout.close()
        proc.wait()
    return out


def plan_shots(frames, fps, words, faces, frame_height, non_track=()):
    """One shot per cut, and a long one split where an editor would (the start
    of a sentence, or after a pause). ``words`` are on the OUTPUT timeline.
    Each shot: {f0, f1 (output frames), z, zmax, push, anchor, face_h}; a shot
    without a single clear face keeps the frame as it is (no ``anchor``)."""
    starts, out_f = [], 0
    for fa, fb in frames:
        starts.append(out_f)
        out_f += fb - fa
    total = out_f

    weight = {}
    for k, word in enumerate(words):
        f = int(word["s"] * fps)
        if k and re.search(r"[.!?…]$", words[k - 1]["w"]):
            weight[f] = 2
        elif k and word["s"] - words[k - 1]["e"] > 0.22:
            weight[f] = max(weight.get(f, 0), 1)
        else:
            weight.setdefault(f, 0)
    points = set(starts)
    for a, b in zip(starts, starts[1:] + [total]):
        last = a
        while (b - last) / fps > MAX_BEAT:
            lo = last + int(2.0 * fps)
            hi = min(last + int(MAX_BEAT * fps), b - int(MIN_BEAT * fps))
            window = [f for f in weight if lo <= f <= hi]
            if not window:
                break
            last = max(window, key=lambda f: (weight[f], -abs((f - last) / fps - 3.0)))
            points.add(last)
    # A cut that removes half a second or more shows: it always changes scale.
    big, o = set(), 0
    for k, (fa, fb) in enumerate(frames):
        if k and (fa - frames[k - 1][1]) / fps >= 0.45:
            big.add(o)
        o += fb - fa
    change = [0]
    for p in sorted(points):
        if p == 0:
            continue
        gap = (p - change[-1]) / fps
        if gap >= MIN_BEAT or (p in big and gap >= 0.7):
            change.append(p)

    def source_time(of):
        o = 0
        for fa, fb in frames:
            if of < o + (fb - fa):
                return (fa + (of - o)) / fps
            o += fb - fa
        return frames[-1][1] / fps

    shots, level = [], 0
    for k, f0 in enumerate(change):
        f1 = change[k + 1] if k + 1 < len(change) else total
        shot = {"f0": f0, "f1": f1, "z": 1.0, "zmax": 1.0, "push": 0.0, "anchor": None}
        t0, t1 = source_time(f0), source_time(max(f0, f1 - 1))
        samples = faces[int(t0 * TRACK_FPS):int(t1 * TRACK_FPS) + 1]
        found = [s for s in samples if s]
        framed = not any(a < t1 and b > t0 for a, b in non_track)
        if framed and samples and len(found) >= 0.6 * len(samples) \
                and max(s[0] for s in found) == 1:
            cx = float(np.median([s[1] for s in found]))
            cy = float(np.median([s[2] for s in found]))
            fh = float(np.median([s[3] for s in found]))
            base = min(1.65, max(1.0, FACE_HEIGHT * frame_height / fh))
            top = max(1.0, FACE_HEIGHT_MAX * frame_height / fh)
            shot.update(anchor=(cx, max(0.0, cy - 1.10 * fh)), face_h=fh,
                        z=min(base * LEVELS[level % len(LEVELS)], top),
                        zmax=min(1.95, top + PUSH_MAX), push=PUSH_PER_SECOND)
            level += 1
        shots.append(shot)
    return shots, total


def zoom_at(shot, frame, fps):
    if not shot["anchor"]:
        return 1.0
    seconds = (frame - shot["f0"]) / fps
    return min(shot["z"] + min(PUSH_MAX, shot["push"] * seconds), shot["zmax"])


def smooth_track(faces, width):
    """The face 5 times a second, smoothed so the frame can follow it inside a
    shot. Smoothed per stretch: a big jump is a camera change, not a move."""
    n = len(faces)
    cx = np.array([f[1] if f else np.nan for f in faces], dtype=np.float64)
    top = np.array([f[2] - 1.10 * f[3] if f else np.nan for f in faces], dtype=np.float64)
    stretch = np.zeros(n, dtype=np.int32)
    k, last, gap = 0, None, 0
    for i in range(n):
        if np.isnan(cx[i]):
            gap += 1
        else:
            if last is not None and (abs(cx[i] - cx[last]) > 0.12 * width or gap > 5):
                k += 1
            last, gap = i, 0
        stretch[i] = k
    kernel = np.exp(-0.5 * (np.arange(-7, 8) / 2.8) ** 2)
    kernel /= kernel.sum()
    for s in range(k + 1):
        idx = np.where(stretch == s)[0]
        ok = idx[~np.isnan(cx[idx])]
        if len(ok) < 2:
            continue
        span = np.arange(ok[0], ok[-1] + 1)
        for arr in (cx, top):
            values = np.interp(span, ok, arr[ok])
            arr[span] = np.convolve(np.pad(values, 7, mode="edge"), kernel, mode="valid")
    return cx, top, stretch


def track_at(track, t):
    """(centre x, top of the head) at second ``t`` of the uncut clip, or None."""
    cx, top, stretch = track
    x = t * TRACK_FPS
    i = int(x)
    if i < 0 or i >= len(cx) or np.isnan(cx[i]):
        return None
    j = min(i + 1, len(cx) - 1)
    if np.isnan(cx[j]) or stretch[j] != stretch[i]:
        return float(cx[i]), max(0.0, float(top[i]))
    f = x - i
    return (float(cx[i] * (1 - f) + cx[j] * f),
            max(0.0, float(top[i] * (1 - f) + top[j] * f)))


def frame_matrix(anchor, z, width, height):
    """Affine for scale ``z`` with the face centred and the top of the head at
    HEAD_TOP. Up to 5 % of invented headroom (the edge row repeated) when the
    source frame has none."""
    cx, top = anchor
    tx = min(0.0, max((1 - z) * width, 0.5 * width - z * cx))
    ty = min(0.05 * height, max((1 - z) * height, HEAD_TOP * height - z * top))
    return np.float32([[z, 0, tx], [0, z, ty]])


# ---------------------------------------------------------------------- titles
TITLES_PROMPT = """You are a senior short-form video editor (TikTok / Reels / Shorts).
Below is the word-by-word transcript of a vertical clip, one word per line as
`index|start_seconds|word`. The clip's language is whatever the words are in.

Clip title: {title}

Return ONLY JSON: {{"titles": [{{"word_index": 3, "text": "CHAMPION"}}]}}

titles: 3 to 5 on-screen keyword titles (fewer if the clip is under 15 seconds).
Each is ONE word (two at most) that is actually spoken at word_index: the
concept the sentence lands on, a noun or a strong verb (CHAMPION, MINDSET,
MONEY), never a filler or a function word. In the clip's language, UPPERCASE,
keep accents. Spread them across the clip, at least 3 seconds apart.

Transcript:
{words}
"""


def _ask_model(prompt, api_key):
    """One text call that must answer {"titles": [...]}. Split out so tests
    can stub it; a local LLM (llm_backend) takes it when one is configured."""
    from typing import List

    from pydantic import BaseModel

    class Title(BaseModel):
        word_index: int
        text: str

    class Titles(BaseModel):
        titles: List[Title]

    import llm_backend
    if llm_backend.active():
        return llm_backend.generate_json(prompt, Titles)[0]
    from google import genai
    from google.genai import types as genai_types
    # Keep the client in a name: one built inline is closed before its call
    # is sent ("client has been closed").
    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model=os.environ.get("GEMINI_MODEL") or "gemini-3.1-flash-lite",
        contents=[prompt],
        config=genai_types.GenerateContentConfig(
            response_mime_type="application/json", response_schema=Titles))
    return json.loads(response.text) or {}


def _plain(text):
    """Letters and digits only, lowercase, accents kept: what two spellings of
    the same word have in common."""
    return re.sub(r"[\W_]+", "", str(text), flags=re.UNICODE).lower()


def sanitize_titles(raw, words):
    """The model proposes, the code decides. Keeps [{"t", "text"}] for the
    proposals that name a word really spoken at that index, as one or two
    words of at most TITLE_MAX_CHARS; a proposal that drifted from the word
    (translated, inflected) falls back to the spoken word itself."""
    out = []
    for item in (raw or {}).get("titles") or []:
        try:
            i = int(item["word_index"])
        except (KeyError, TypeError, ValueError):
            continue
        if not 0 <= i < len(words):
            continue
        spoken = _plain(words[i]["w"])
        text = " ".join(re.sub(r"[^\w\s\-]", "", str(item.get("text", "")),
                               flags=re.UNICODE).split()[:2]).upper()
        first = _plain(text.split(" ")[0]) if text else ""
        if not first or not (first.startswith(spoken[:4]) or spoken.startswith(first[:4])):
            text = spoken.upper()
        if not 3 <= len(text) <= TITLE_MAX_CHARS:
            continue
        out.append({"t": float(words[i]["s"]), "text": text})
    out.sort(key=lambda t: t["t"])
    return out


def pick_titles(words, clip_title="", api_key=None):
    """Keyword titles for the clip, [{"t" (seconds on the clip), "text"}].
    Empty without a model to ask, and on any failure: a clip without titles
    is still a pro clip."""
    import llm_backend
    if not words or not (api_key or llm_backend.active()):
        return []
    listing = "\n".join(f"{i}|{w['s']:.2f}|{w['w']}" for i, w in enumerate(words))
    try:
        answer = _ask_model(TITLES_PROMPT.format(title=clip_title, words=listing[:12000]),
                            api_key)
        return sanitize_titles(answer, words)
    except Exception as e:
        print(f"   ℹ️ Pro titles skipped ({type(e).__name__}: {e})")
        return []


def place_titles(titles, frames, shots, fps, total, not_before=0.0):
    """Put the titles on the CUT timeline: [{"t", "text"}] with ``t`` in
    output seconds. A title goes only where its word survived the cuts, on a
    shot framed on the face (the top of the frame is the headroom there; over
    a screen or a board it would cover content), after ``not_before`` (the
    hook's time on screen) and TITLE_GAP after the previous one."""
    convert = to_output_time(frames, fps)
    placed, last = [], -TITLE_GAP
    for title in titles or []:
        t = float(title["t"])
        if not any(fa / fps <= t < fb / fps for fa, fb in frames):
            continue
        at = convert(t)
        frame = int(at * fps)
        shot = next((s for s in shots if s["f0"] <= frame < s["f1"]), None)
        if not shot or not shot["anchor"]:
            continue
        if at < not_before or at - last < TITLE_GAP or at > total - 0.8:
            continue
        placed.append({"t": round(at, 3), "text": str(title["text"])})
        last = at
        if len(placed) >= TITLE_LIMIT:
            break
    return placed


def _ass_time(t):
    cs = int(round(max(0.0, t) * 100))
    return f"{cs // 360000}:{cs // 6000 % 60:02d}:{cs // 100 % 60:02d}.{cs % 100:02d}"


def write_titles(path, titles, total, width, height):
    """The titles as an ASS file: one huge word at the top of the frame that
    pops in and fades, sized to span most of the width whatever its length."""
    k = width / 1080.0
    colour = TITLE_COLOR.lstrip("#")
    colour = f"&H00{colour[4:6]}{colour[2:4]}{colour[0:2]}".upper()
    lines = [
        "[Script Info]", "ScriptType: v4.00+", f"PlayResX: {width}",
        f"PlayResY: {height}", "WrapStyle: 2", "ScaledBorderAndShadow: yes", "",
        "[V4+ Styles]",
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, "
        "BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, "
        "BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding",
        f"Style: Title,{TITLE_FONT},{int(200 * k)},{colour},{colour},&H00000000,&H00000000,"
        "0,0,0,0,100,100,0,0,1,0,0,8,20,20,0,1", "",
        "[Events]",
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
    ]
    for i, title in enumerate(titles):
        following = titles[i + 1]["t"] if i + 1 < len(titles) else total
        end = min(title["t"] + TITLE_SECONDS, following - 0.05, total)
        text = title["text"].replace("\\", "").replace("{", "(").replace("}", ")")
        size = int(min(232, 0.88 * 1080 / (0.735 * max(4, len(text)))) * k)
        lines.append(
            f"Dialogue: 2,{_ass_time(title['t'])},{_ass_time(end)},Title,,0,0,0,,"
            f"{{\\an8\\pos({width // 2},{int(64 * k)})\\fs{size}\\blur0.5\\fad(50,160)"
            f"\\fscx112\\fscy112\\t(0,130,\\fscx100\\fscy100)}}{text}")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


# ------------------------------------------------------------------------ look
def grade_lut():
    """Contrast curve, crushed blacks, warm skin over slightly cold shadows."""
    x = np.linspace(0, 1, 256)
    y = 0.55 * x + 0.45 * (x * x * (3 - 2 * x))
    y = np.clip((y - 0.045) / (1 - 0.045), 0, 1)
    r, g, b = y ** 0.93, y, y ** 1.09 + 0.012 * (1 - y) ** 3
    lut = np.stack([b, g, r], axis=1)   # BGR
    return np.clip(lut * 255 + 0.5, 0, 255).astype(np.uint8).reshape(256, 1, 3)


def static_gain(width, height):
    """Vignette plus a darker bottom, at half resolution."""
    gw, gh = width // 2, height // 2
    yy, xx = np.mgrid[0:gh, 0:gw].astype(np.float32)
    d = np.sqrt(((xx - gw / 2) / (gw / 2)) ** 2 * 0.9
                + ((yy - gh * 0.45) / (gh / 2)) ** 2 * 0.55)
    vignette = 1.0 - 0.30 * np.clip(d - 0.55, 0, 1) ** 1.5 / (0.45 ** 1.5 + 1e-6) * 0.45
    bottom = np.clip((yy / gh - 0.80) / 0.20, 0, 1)
    return (np.clip(vignette, 0.55, 1.0) * (1.0 - 0.38 * bottom)).astype(np.float32)


class Look:
    """Grade + dimmed background + vignette, one frame at a time."""

    def __init__(self, width, height):
        import cv2
        import mediapipe as mp
        self.cv2 = cv2
        self.size = (width, height)
        self.lut = grade_lut()
        self.static = static_gain(width, height)
        self.segmenter = mp.solutions.selfie_segmentation.SelfieSegmentation(model_selection=0)
        self.mask = None
        self.gain = None
        self.age = 0

    def close(self):
        self.segmenter.close()

    def _refresh(self, frame, cut, person):
        cv2 = self.cv2
        small = cv2.resize(frame[::2, ::2], MASK_SIZE, interpolation=cv2.INTER_LINEAR)
        mask = self.segmenter.process(cv2.cvtColor(small, cv2.COLOR_BGR2RGB)).segmentation_mask
        mask = np.clip((mask - 0.30) / 0.45, 0, 1)
        if self.mask is not None and not cut:
            mask = 0.55 * mask + 0.45 * self.mask
        self.mask = mask
        # No dimming without a person big enough to be the subject: on a shot
        # of something else the "background" is the picture.
        strength = min(1.0, max(0.0, (float(mask.mean()) - 0.08) / 0.10)) if person else 0.0
        floor = 1.0 - (1.0 - BACKGROUND_GAIN) * strength
        soft = cv2.GaussianBlur(mask, (0, 0), 3.0)
        gh, gw = self.static.shape
        gain = self.static * (floor + (1 - floor) * cv2.resize(
            soft, (gw, gh), interpolation=cv2.INTER_LINEAR))
        gain = cv2.resize((gain * 255).astype(np.uint8), self.size,
                          interpolation=cv2.INTER_LINEAR)
        self.gain = cv2.merge([gain, gain, gain])

    def __call__(self, frame, cut, person):
        if cut or self.gain is None or self.age >= SEGMENT_EVERY:
            self._refresh(frame, cut, person)
            self.age = 0
        self.age += 1
        return self.cv2.multiply(self.cv2.LUT(frame, self.lut), self.gain, scale=1 / 255.0)


def cut_audio(audio, frames, fps):
    """The kept stretches back to back, each with a 6 ms fade at both ends so
    a join never clicks."""
    fade = int(SAMPLE_RATE * 0.006)
    ramp = np.linspace(0, 1, fade, dtype=np.float32)[:, None]
    parts = []
    for fa, fb in frames:
        a, b = int(round(fa / fps * SAMPLE_RATE)), int(round(fb / fps * SAMPLE_RATE))
        part = audio[a:b].astype(np.float32)
        if len(part) > 2 * fade:
            part[:fade] *= ramp
            part[-fade:] *= ramp[::-1]
        parts.append(part)
    if not parts:
        return audio[:0]
    return np.clip(np.concatenate(parts), -32768, 32767).astype(np.int16)


def _write_wav(path, samples):
    with wave.open(path, "wb") as f:
        f.setnchannels(2)
        f.setsampwidth(2)
        f.setframerate(SAMPLE_RATE)
        f.writeframes(samples.tobytes())


# Saturation and sharpening ride the encoder's filter chain: native code, and
# the frame is on its way to YUV there anyway.
_FINISH = ("scale=out_color_matrix=bt709:out_range=tv,format=yuv420p,"
           "eq=saturation=1.04,unsharp=7:7:0.2:7:7:0.0")


def render(clip_path, out_path, words=(), ranges=(), titles=(), hook_seconds=0.0):
    """Render the pro version of ``clip_path`` (a clean 9:16 clip) to
    ``out_path``. ``words``: [{"w", "s", "e"}] on the clip's timeline;
    ``ranges``: its layout ranges; ``titles``: [{"t", "text"}] keyword titles
    on the same timeline, burned at the top (none while a hook is on screen,
    its first ``hook_seconds``). Returns {"keep": [[start, end], ...] (seconds
    of the clip that survive), "duration", "shots", "framed", "titles"}."""
    import cv2
    from ffmpeg_utils import (METADATA_SCRUB, QUALITY_FAST, escape_filter_value,
                              video_encode_args)

    try:
        cv2.setNumThreads(max(1, int(os.environ.get("PRO_THREADS", "2"))))
    except ValueError:
        cv2.setNumThreads(2)
    width, height, fps, rate, duration = probe(clip_path)
    if height <= width:
        raise NotApplicable("pro mode needs a vertical clip")
    words = sorted(({"w": str(w.get("w", "")), "s": float(w["s"]), "e": float(w["e"])}
                    for w in words), key=lambda w: w["s"])

    audio = load_audio(clip_path)
    frames = keep_frames(silences(envelope_db(audio)), words, duration, fps)
    convert = to_output_time(frames, fps)
    out_words = [{"w": w["w"], "s": convert(w["s"]), "e": convert(w["e"])} for w in words]
    non_track = [(float(r["start"]), float(r["end"])) for r in ranges or []
                 if str(r.get("layout", "track")).lower() != "track"]
    faces = face_track(clip_path, width, height)
    shots, total = plan_shots(frames, fps, out_words, faces, height, non_track)
    track = smooth_track(faces, width) if faces else None

    # Neutral temp names next to the output: the titles file goes inside an
    # ffmpeg filter string, where a clip title's apostrophe would end the quote.
    work = os.path.join(os.path.dirname(out_path) or ".",
                        f"temp_pro_{os.getpid()}_{int(time.time() * 1000)}")
    work_audio = None
    if len(audio):
        work_audio = work + ".wav"
        _write_wav(work_audio, cut_audio(audio, frames, fps))
    del audio
    placed = place_titles(titles, frames, shots, fps, total / fps,
                          not_before=float(hook_seconds or 0) + 0.3 if hook_seconds else 0.0)
    finish, work_titles = _FINISH, None
    if placed:
        work_titles = work + ".ass"
        write_titles(work_titles, placed, total / fps, width, height)
        fonts = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fonts")
        finish += (f",ass=filename='{escape_filter_value(work_titles)}'"
                   f":fontsdir='{escape_filter_value(fonts)}'")

    size = width * height * 3
    decoder = subprocess.Popen(
        ["ffmpeg", "-v", "error", "-threads", "2", "-i", clip_path, "-an",
         "-f", "rawvideo", "-pix_fmt", "bgr24", "-"],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, stdin=subprocess.DEVNULL,
        bufsize=size)
    command = ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24",
               "-s", f"{width}x{height}", "-r", rate, "-i", "-"]
    if work_audio:
        command += ["-i", work_audio]
    command += ["-vf", finish, "-map", "0:v"]
    if work_audio:
        command += ["-map", "1:a", "-c:a", "aac"]
    command += [*video_encode_args(QUALITY_FAST), "-colorspace", "bt709",
                "-color_primaries", "bt709", "-color_trc", "bt709",
                *METADATA_SCRUB, "-movflags", "+faststart", out_path]
    encoder = subprocess.Popen(command, stdin=subprocess.PIPE,
                               stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    look = Look(width, height)
    written = 0
    try:
        pending = iter(frames)
        current = next(pending, None)
        src_f = shot_i = 0
        while current is not None:
            buf = decoder.stdout.read(size)
            if len(buf) < size:
                break
            if src_f >= current[1]:
                current = next(pending, None)
                if current is None:
                    break
            if current[0] <= src_f:
                while shot_i + 1 < len(shots) and written >= shots[shot_i]["f1"]:
                    shot_i += 1
                shot = shots[shot_i]
                frame = np.frombuffer(buf, dtype=np.uint8).reshape(height, width, 3)
                if shot["anchor"]:
                    live = track_at(track, src_f / fps) if track else None
                    frame = cv2.warpAffine(
                        frame, frame_matrix(live or shot["anchor"],
                                            zoom_at(shot, written, fps), width, height),
                        (width, height), flags=cv2.INTER_CUBIC,
                        borderMode=cv2.BORDER_REPLICATE)
                cut = src_f == current[0] or written == shot["f0"]
                encoder.stdin.write(look(frame, cut, bool(shot["anchor"])).tobytes())
                written += 1
            src_f += 1
        encoder.stdin.close()
        err = encoder.stderr.read()   # -v error: a few lines at most
        encoder.wait()
        if encoder.returncode != 0 or written == 0:
            tail = (err or b"").decode("utf-8", "replace")[-300:]
            raise RuntimeError(f"pro render failed after {written} frames: {tail}")
    except BaseException:
        encoder.kill()
        if os.path.exists(out_path):
            os.remove(out_path)
        raise
    finally:
        decoder.kill()
        decoder.stdout.close()
        decoder.wait()
        look.close()
        for temp in (work_audio, work_titles):
            if temp and os.path.exists(temp):
                os.remove(temp)

    # A short last read can leave the picture a few frames under the plan:
    # report what was really written.
    keep, left = [], written
    for fa, fb in frames:
        if left <= 0:
            break
        fb = min(fb, fa + left)
        left -= fb - fa
        keep.append([round(fa / fps, 3), round(fb / fps, 3)])
    shown = written / fps
    return {"keep": keep, "duration": round(shown, 3), "shots": len(shots),
            "framed": sum(1 for s in shots if s["anchor"]),
            "titles": [t["text"] for t in placed if t["t"] < shown]}


# ------------------------------------------------------------- host-wide slots
class slot:
    """``with slot(wait):`` holds one of PRO_SLOTS flock slots of this host.

    Raises Busy when none frees up within ``wait`` seconds: a caller with a
    person waiting says so, and the job pipeline ships the clip as it is. A
    crashed holder releases its lock with its file descriptor. A no-op where
    flock or the directory is unavailable, or with PRO_SLOTS=0."""

    def __init__(self, wait=0.0, slots=None, lock_dir=None, poll=1.0):
        self.wait, self.poll = float(wait), poll
        self.slots = int(os.environ.get("PRO_SLOTS", "1")) if slots is None else slots
        self.lock_dir = lock_dir or os.environ.get("PRO_LOCK_DIR", "output")
        self._fh = None

    def __enter__(self):
        if self.slots <= 0:
            return self
        try:
            import fcntl
            os.makedirs(self.lock_dir, exist_ok=True)
        except Exception:
            return self
        deadline = time.monotonic() + self.wait
        while True:
            for i in range(self.slots):
                try:
                    fh = open(os.path.join(self.lock_dir, f".pro-{i}.lock"), "a+")
                except OSError:
                    return self
                try:
                    fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    self._fh = fh
                    return self
                except OSError:
                    fh.close()
            if time.monotonic() >= deadline:
                raise Busy("every pro render slot is taken")
            time.sleep(self.poll)

    def __exit__(self, *exc):
        if self._fh is not None:
            try:
                import fcntl
                fcntl.flock(self._fh, fcntl.LOCK_UN)
            except Exception:
                pass
            self._fh.close()
            self._fh = None
        return False


def words_for(transcript, segments):
    """The clip's words on its own timeline, for a clip made of ``segments``
    of the source ({"start", "end"}, in order)."""
    import recut
    words = []
    for segment in recut.virtual_transcript(transcript or {}, segments)["segments"]:
        for w in segment["words"]:
            words.append({"w": w["word"].strip(), "s": w["start"], "e": w["end"]})
    return words


def run(clip_path, out_path, words=(), ranges=(), wait=0.0, titles=(), hook_seconds=0.0):
    """Render in a process of its own, under a host slot. Returns render()'s
    dict. Raises Busy, NotApplicable or RuntimeError."""
    with slot(wait):
        try:
            result = subprocess.run(
                [sys.executable, os.path.abspath(__file__), clip_path, out_path],
                input=json.dumps({"words": list(words), "ranges": list(ranges or []),
                                  "titles": list(titles or []),
                                  "hook_seconds": float(hook_seconds or 0)}).encode(),
                capture_output=True, timeout=RENDER_TIMEOUT_SECONDS)
        except subprocess.TimeoutExpired:
            if os.path.exists(out_path):
                os.remove(out_path)
            raise RuntimeError(f"pro render timed out after {RENDER_TIMEOUT_SECONDS}s")
    for line in reversed(result.stdout.decode("utf-8", "replace").splitlines()):
        if line.startswith("PRO_RESULT "):
            payload = json.loads(line[len("PRO_RESULT "):])
            if payload.get("error") == "not_applicable":
                raise NotApplicable(payload.get("detail", ""))
            if result.returncode == 0 and payload.get("keep"):
                return payload
    tail = result.stderr.decode("utf-8", "replace").strip().splitlines()[-3:]
    raise RuntimeError(f"pro render failed (exit {result.returncode}): {' | '.join(tail)[-300:]}")


# What a conversion replaces on the clip, kept under ``pro_before`` so going
# back to normal is putting these back.
_REPLACED = ("start", "end", "recipe", "layout_ranges", "crop_overrides", "caption_words")


def convert(base_path, clean_name, clip, transcript, wait=0.0, api_key=None,
            hook_seconds=0.0, caption_style=None):
    """Render the pro version of a clip from its clean file ``base_path``.

    Returns (clean pro filename, updates for the clip's metadata). The pro
    file is a recut of the same source: the updates carry its ``recipe`` (the
    kept stretches, on the source timeline), so captions and the clip editor
    follow the new cut the way they follow a manual trim.

    ``api_key`` (or a configured local LLM) adds the keyword titles;
    ``hook_seconds`` keeps them off the hook's time on screen. ``caption_style``
    becomes the clip's caption style unless the user already picked one: going
    back to normal undoes it only if it is still the one set here."""
    import uuid

    import layout_ranges
    recipe = clip.get("recipe") or {}
    whole = {"start": float(clip.get("start", 0) or 0), "end": float(clip.get("end", 0) or 0)}
    segments = recipe.get("segments") or [dict(whole)]
    ranges = clip.get("layout_ranges") or layout_ranges.read(base_path)
    out_name = f"recut_{int(time.time())}_{uuid.uuid4().hex[:6]}_{clean_name}"
    out_path = os.path.join(os.path.dirname(base_path), out_name)
    words = words_for(transcript, segments)
    titles = pick_titles(words, clip.get("video_title_for_youtube_short") or "", api_key)
    result = run(base_path, out_path, words, ranges, wait, titles=titles,
                 hook_seconds=hook_seconds)

    kept = [{"start": a, "end": b} for a, b in result["keep"]]
    new_segments = source_segments(result["keep"], segments)
    new_ranges = layout_ranges.write(out_path, [
        (r["start"], r["end"], r["layout"], r.get("content"))
        for r in layout_ranges.remap(ranges, kept)])
    new_recipe = {"v": 1, "segments": new_segments,
                  "canonical_range": recipe.get("canonical_range") or dict(whole)}
    if recipe.get("framing"):
        new_recipe["framing"] = recipe["framing"]
    before = {key: clip.get(key) for key in _REPLACED}
    before["base"] = os.path.basename(base_path)
    updates = {
        "mode": "pro", "pro_before": before, "recipe": new_recipe,
        "pro_titles": result.get("titles") or [],
        "start": min(s["start"] for s in new_segments),
        "end": max(s["end"] for s in new_segments),
        "layout_ranges": new_ranges,
        # Both are keyed to the cut they were made on (same as a rerender).
        "crop_overrides": None, "caption_words": None,
    }
    if caption_style and not clip.get("caption_style"):
        updates["caption_style"] = dict(caption_style)
        before["caption_style_set"] = dict(caption_style)
    return out_name, updates


def restore(clip):
    """Updates that put a pro clip's metadata back to what it was. The caller
    points it at ``pro_before['base']`` (the clean file from before)."""
    before = clip.get("pro_before") or {}
    updates = {key: before.get(key) for key in _REPLACED if key in before}
    updates.update(mode="normal", pro_before=None, pro_titles=None)
    # The caption style pro mode brought goes with it, unless the user has
    # changed it since: then it is theirs.
    style = before.get("caption_style_set")
    if style and clip.get("caption_style") == style:
        updates["caption_style"] = None
    return updates


def _main(argv):
    # No renice: the render keeps its caller's priority. From the API that is
    # a person waiting on a spinner (ahead of the jobs, which run niced, and
    # held to one render by the host slot); inside a job it is the job's own.
    clip_path, out_path = argv[1], argv[2]
    payload = json.loads(sys.stdin.read() or "{}")
    try:
        result = render(clip_path, out_path, payload.get("words") or (),
                        payload.get("ranges") or (), payload.get("titles") or (),
                        payload.get("hook_seconds") or 0.0)
    except NotApplicable as e:
        print("PRO_RESULT " + json.dumps({"error": "not_applicable", "detail": str(e)}))
        return 2
    print("PRO_RESULT " + json.dumps(result))
    return 0


if __name__ == "__main__":
    sys.exit(_main(sys.argv))
