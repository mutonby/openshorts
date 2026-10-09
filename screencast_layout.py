"""SCREENCAST layout: full-width content on top, the speaker underneath.

The fourth layout. It targets the failure this repo has now attacked three
times: a screen recording that happens to contain a face gets classified TRACK,
the 9:16 crop keeps a centre strip, and the chart or headline the shot is
actually about comes out sliced mid-word.

The two previous attempts tried to find that content in pixels and both failed
(see the note above analyze_scenes_strategy in main.py): edge density and MSER
text density BOTH score ordinary detailed footage higher than a clean panel of
text, because they measure visual busyness rather than meaning. The third
attempt asked Gemini and detected well, but decided badly: it flagged corner
tickers and demoted well-framed talking heads to the blurred layout.

What is different here is the question asked and what the answer is used for.

  - The question is how much of the WIDTH the content spans, not how much of the
    duration it covers. Coverage did not separate the cases (screencasts ran
    88-97% of the video, a corner-ticker clip 2%, but the failure was on the
    ticker anyway). Width is the quantity that actually decides whether a 9:16
    crop destroys information: a corner bug spans ~15% of the frame and survives
    any crop, a spreadsheet spans ~100% and cannot.
  - The answer routes to THIS layout, not to GENERAL. The old wiring's worst
    case was shrinking a subject into blurred filler to preserve a corner
    counter. Here the worst case is showing the content full width above the
    speaker, which is a reasonable frame even when the trigger was wrong.

Off by default (``SCREENCAST_LAYOUT=1``, set per job by ``layouts=["screencast"]``
or by the layout picker). Which shots show a screen is asked of Gemini per clip
(``detect_content_ranges``, one still per shot); without an answer the scenes
the face classifier sent to GENERAL are taken as the screen (``fallback_ranges``),
because the job was declared a screencast and GENERAL is the one layout that
shrinks a screen into an unreadable strip.
"""
import json
import os

import numpy as np

ENABLED = os.environ.get("SCREENCAST_LAYOUT", "0") == "1"

# Fraction of the frame width the content must span. A corner ticker, logo or
# channel bug sits far below this; a screen recording, slide or spreadsheet sits
# near 1.0. This is the axis the previous attempt did not ask about.
MIN_WIDTH_FRACTION = 0.5

# Above this the content fills the frame, so any presenter is composited ON TOP
# of it rather than sitting beside it. Stacking then shows the same content
# twice: measured on an Excel walkthrough where the speaker is keyed into the
# corner, the bottom band came out as a zoomed crop of the same spreadsheet.
# Those scenes get the full-width GENERAL layout instead, which is the fix they
# actually needed — the default GENERAL ratio crops ~24% off the sides, and on a
# spreadsheet the discarded columns are the point.
STACK_MAX_WIDTH_FRACTION = 0.85

# Seconds of overlap before a scene counts as showing the content.
MIN_OVERLAP_SECONDS = 0.25

# The speaker crop below the content needs a face of at least this width
# (fraction of frame width). Smaller than this and the bottom half is mostly
# desktop with a stamp-sized webcam in it, which is worse than GENERAL.
MIN_FACE_WIDTH = 0.05


def content_bands(orig_w, orig_h, out_w, out_h):
    """(content_height, speaker_height) for the stacked screencast frame.

    The content keeps its full width, which is the entire point of this layout,
    so its height follows from the source aspect: a 16:9 source gives 608px of a
    1920px frame. The speaker takes the rest.
    """
    content_h = int(round(out_w * orig_h / float(orig_w)))
    content_h -= content_h % 2
    content_h = max(2, min(content_h, out_h - 2))
    speaker_h = out_h - content_h
    return content_h, speaker_h


def speaker_crop(orig_w, orig_h, out_w, speaker_h, face_centre):
    """Crop box (w, h, x, y) for the speaker band, framed on the face."""
    aspect = out_w / float(speaker_h)

    crop_h = orig_h
    crop_w = int(round(crop_h * aspect))
    if crop_w > orig_w:
        crop_w = orig_w
        crop_h = int(round(crop_w / aspect))

    crop_w -= crop_w % 2
    crop_h -= crop_h % 2

    cx, cy = face_centre
    x = int(round(cx - crop_w / 2.0))
    x = max(0, min(x, orig_w - crop_w))
    y = int(round(cy - crop_h * 0.42))
    y = max(0, min(y, orig_h - crop_h))

    return crop_w, crop_h, x - (x % 2), y - (y % 2)


def screencast_filtergraph(orig_w, orig_h, out_w, out_h, face_centre):
    """Full-width content above, face-framed speaker below."""
    content_h, speaker_h = content_bands(orig_w, orig_h, out_w, out_h)
    cw, ch, cx, cy = speaker_crop(orig_w, orig_h, out_w, speaker_h, face_centre)

    return (
        f"[0:v]split=2[ca][sa];"
        # The content band is the WHOLE frame scaled down. Nothing is cropped
        # off the sides, which is the one thing this layout exists to guarantee.
        f"[ca]scale={out_w}:{content_h}[content];"
        f"[sa]crop=w={cw}:h={ch}:x={cx}:y={cy},scale={out_w}:{speaker_h}[speaker];"
        f"[content][speaker]vstack=inputs=2,"
        f"pad={out_w}:{out_h}:0:0,setsar=1[v]"
    )


def detect_faces_full_res(frame):
    """Face boxes from the UNSCALED frame, in original coordinates.

    Runs YOLO person detection at full frame resolution to capture presenter insets
    in screencasts, extracting the upper-body / face candidate area.
    """
    import main as m

    h, w, _ = frame.shape
    with m.DETECT_LOCK:
        results = m.model(frame, verbose=False, classes=[0])
    if not results or not results[0].boxes:
        return []

    out = []
    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        pw = x2 - x1
        ph = y2 - y1
        fh = int(ph * 0.45)
        fw = int(pw * 0.8)
        fx = x1 + int(pw * 0.1)
        b = [fx, y1, fw, fh]
        out.append({'box': b, 'score': fw * fh})
    return out


def _face_centre(candidates, frame_w):
    """Centre of the biggest usable face in a frame, or None."""
    big = [c for c in candidates if c['box'][2] >= MIN_FACE_WIDTH * frame_w]
    if not big:
        return None
    box = max(big, key=lambda c: c['score'])['box']
    return box[0] + box[2] / 2.0, box[1] + box[3] / 2.0


def overlapping_width(scene_start, scene_end, ranges):
    """Widest content the scene overlaps, as a fraction of frame width.

    0.0 when the scene overlaps nothing, which leaves its routing untouched.
    """
    return overlapping_range(scene_start, scene_end, ranges)[0]


def presenter_cam(scene_start, scene_end, ranges):
    """Whether the shot check saw a live webcam window over this scene's screen.

    The sixth element of a range (``presenter_cam``). Ranges without it (the
    fallback, a caller that pinned its own) answer False: the per-scene inset
    detector only runs where the model said a presenter is laid over the
    screen, because a small still face on a screen is just as often a photo on
    a slide, cover art or a game character (6-oct-2026 corpus run).
    """
    for r in ranges:
        if len(r) > 5 and r[5] and \
                min(scene_end, r[1]) - max(scene_start, r[0]) > MIN_OVERLAP_SECONDS:
            return True
    return False


def overlapping_range(scene_start, scene_end, ranges):
    """(width_fraction, focus) of the widest range the scene overlaps.

    ``focus`` is the (left, right) reading area a range carries as its fifth
    element, or None. (0.0, None) when the scene overlaps nothing.
    """
    widest, focus = 0.0, None
    for r in ranges:
        start, end = r[0], r[1]
        width = r[3] if len(r) > 3 else 1.0
        if min(scene_end, end) - max(scene_start, start) > MIN_OVERLAP_SECONDS:
            if width > widest:
                widest = width
                focus = r[4] if len(r) > 4 else None
    return widest, focus


# --- which shots show a screen ------------------------------------------------
#
# Until 6-oct-2026 nothing called the detector: SCREENCAST_LAYOUT=1 (a user's
# explicit `layouts=["screencast"]`, or the layout picker's "screencast") only
# flipped ENABLED, reframe_v2.render was never handed any content ranges, and
# every scene of a screen tutorial went through the face classifier like any
# other video. A face-less screen came out of that as GENERAL (the desktop
# shrunk into a strip over a blurred copy of itself), a face that only appears
# inside the app as TRACK (a centre crop through the middle of the screen).
#
# The detector this module used to carry uploaded the whole SOURCE to Gemini
# and asked for time ranges. That cannot be wired into a job as it was: a
# 37-min 1440p tutorial is ~670k tokens and a GB-sized upload, per job, to get
# back a handful of numbers. This asks per CLIP instead, with one still per
# shot, which is the layout picker's recipe (frames at 1024px, a closed choice)
# and lines the answer up with the scenes the renderer routes, with no
# timestamps for the model to get wrong.

# Kinds the shot prompt may answer, and the width each one routes as. A screen
# fills the frame, so it lands past STACK_MAX_WIDTH_FRACTION (WIDE or INSET); a
# chart beside a person sits between the two gates (SCREENCAST stacking).
KIND_WIDTH = {"screen": 1.0, "beside": 0.7}

# Stills per clip. One per shot; past this the longest shots are asked and the
# rest borrow the answer of the nearest asked shot.
MAX_SHOTS = int(os.environ.get("SCREENCAST_MAX_SHOTS", "24"))
SHOT_SAMPLE_WIDTH = 1024

# The reading area is cropped out of the screen when it is narrower than this
# (fraction of the frame width, after padding). Wider than that a crop only
# trims chrome while costing columns that may matter: show the whole screen.
FOCUS_MAX_WIDTH = 0.9
FOCUS_PADDING = 0.04

# The cropped screen may fill at most this share of the output height. It sets
# the narrowest crop allowed (about half of a 16:9 screen in a 9:16 frame), so a
# tight focus box never becomes a smear of upscaled pixels.
FOCUS_MAX_HEIGHT_RATIO = 0.6


def _parse_shots(raw, n):
    """The model's answer as {position: (kind, focus, cam)}; junk is dropped.

    ``cam`` (presenter_cam) only means something on a screen shot, so it is
    forced False on the other kinds, and only a real true counts: anything
    else the model might send is a no.
    """
    out = {}
    for item in raw or []:
        try:
            idx = int(item.get("shot"))
            kind = str(item.get("kind", "")).strip().lower()
            left = float(item.get("focus_left", 0.0))
            right = float(item.get("focus_right", 1.0))
            cam = item.get("presenter_cam") is True
        except (AttributeError, TypeError, ValueError):
            continue
        if not 0 <= idx < n or kind not in ("screen", "beside", "camera"):
            continue
        left, right = max(0.0, min(left, 1.0)), max(0.0, min(right, 1.0))
        out[idx] = (kind, (left, right) if right - left >= 0.05 else None,
                    cam and kind == "screen")
    return out


def shots_to_ask(scenes, limit=None):
    """Indices of the scenes to sample, the longest ones when there are too many."""
    limit = limit or MAX_SHOTS
    order = list(range(len(scenes)))
    if len(order) <= limit:
        return order

    def length(i):
        return scenes[i][1].get_frames() - scenes[i][0].get_frames()
    return sorted(sorted(order, key=length, reverse=True)[:limit])


def ranges_from_verdicts(scenes, fps, verdicts):
    """Content ranges (start_s, end_s, kind, width, focus, cam) in the clip
    timeline.

    ``verdicts`` maps scene index -> (kind, focus, cam). A scene that was not
    asked takes the verdict of the nearest asked one.
    """
    asked = sorted(verdicts)
    if not asked:
        return []
    ranges = []
    for i, (start, end) in enumerate(scenes):
        src = i if i in verdicts else min(asked, key=lambda a: abs(a - i))
        kind, focus, cam = verdicts[src]
        width = KIND_WIDTH.get(kind)
        if not width:
            continue
        ranges.append((start.get_frames() / fps, end.get_frames() / fps,
                       kind, width, focus, cam))
    return ranges


def _shot_frames(video_path, scenes, indices):
    """JPEG bytes of each asked scene's middle frame (None where unreadable)."""
    import cv2
    import frame_sampler
    from layout_picker import _encode_frame

    mids = [(scenes[i][0].get_frames() + scenes[i][1].get_frames()) // 2
            for i in indices]
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []
    try:
        frames = list(frame_sampler.read_at(cap, mids))
    finally:
        cap.release()
    return [_encode_frame(f, SHOT_SAMPLE_WIDTH) if f is not None else None
            for f in frames]


def detect_content_ranges(video_path, scenes, fps):
    """Which of this clip's shots show a screen, and where its reading area is.

    Returns a list of (start_s, end_s, kind, width_fraction, focus, cam), [] when no
    shot shows one or the module is off, or None when the question could not be
    asked or answered (no key, no frames, an API error). Callers must treat None
    differently from []: the user asked for this layout, so a failed check must
    not quietly hand the clip back to the face classifier (see fallback_ranges).
    """
    if not ENABLED:
        return []
    if not scenes:
        return None
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("   ⚠️ Screen check needs GEMINI_API_KEY.")
        return None

    model_name = os.environ.get("GEMINI_MODEL") or 'gemini-3.1-flash-lite'
    asked = shots_to_ask(scenes)
    print(f"   🔎 Screen check on {len(asked)} shot(s)…")
    try:
        from google import genai
        from google.genai import types as genai_types
        import gemini_worker

        frames = _shot_frames(video_path, scenes, asked)
        keep = [(i, jpg) for i, jpg in zip(asked, frames) if jpg]
        if not keep:
            print("   ⚠️ No readable frames for the screen check.")
            return None
        parts = []
        for n, (_, jpg) in enumerate(keep):
            parts.append(f"Shot {n}:")
            parts.append(genai_types.Part.from_bytes(data=jpg, mime_type="image/jpeg"))
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model=model_name,
            contents=parts + [gemini_worker.SHOT_CONTENT_PROMPT],
            config=genai_types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=gemini_worker.ShotContentResponse,
            ))
        gemini_worker.raise_if_blocked(response)
        raw = (json.loads(response.text) or {}).get("shots") or []
    except Exception as e:
        print(f"   ⚠️ Screen check failed ({e}).")
        return None

    by_position = _parse_shots(raw, len(keep))
    if not by_position:
        print("   ⚠️ Screen check returned nothing usable.")
        return None
    verdicts = {keep[n][0]: v for n, v in by_position.items()}
    print("   📊 Shots: " + ", ".join(
        f"{i}={k}" + (f"[{f[0]:.2f}-{f[1]:.2f}]" if f and k != "camera" else "")
        + ("+cam" if c else "")
        for i, (k, f, c) in sorted(verdicts.items())))
    return ranges_from_verdicts(scenes, fps, verdicts)


def fallback_ranges(scenes, strategies, fps):
    """Ranges to use when the screen check failed on a screencast job.

    The user (or the layout picker) said this video is built around a screen,
    so a scene the face classifier found no subject in is the screen: it gets
    the full-width layout instead of GENERAL's side crop. Scenes with a face
    keep the classifier's verdict.
    """
    out = []
    for (start, end), strategy in zip(scenes, strategies):
        if strategy == 'GENERAL':
            out.append((start.get_frames() / fps, end.get_frames() / fps,
                        "screen", 1.0, None, False))
    return out


def focus_crop(orig_w, orig_h, out_w, out_h, focus):
    """(x, w) of the source columns to show for a screen shot, or None.

    None means show the whole width (no focus, or one so wide that a crop would
    only trim chrome). Otherwise the reading area plus a little padding, never
    narrower than the crop that fills FOCUS_MAX_HEIGHT_RATIO of the output.
    """
    if not focus:
        return None
    left = max(0.0, focus[0] - FOCUS_PADDING)
    right = min(1.0, focus[1] + FOCUS_PADDING)
    if right <= left or right - left >= FOCUS_MAX_WIDTH:
        return None
    min_w = out_w * orig_h / (FOCUS_MAX_HEIGHT_RATIO * out_h)
    crop_w = min(orig_w, int(round(max((right - left) * orig_w, min_w))))
    crop_w -= crop_w % 2
    if crop_w >= orig_w * FOCUS_MAX_WIDTH:
        return None
    x = int(round((left + right) / 2.0 * orig_w - crop_w / 2.0))
    x = max(0, min(x, orig_w - crop_w))
    return x - (x % 2), crop_w


def focus_filtergraph(orig_w, orig_h, out_w, out_h, crop):
    """A screen shot cut down to its reading area, over a blurred backdrop.

    ``crop`` is focus_crop()'s (x, w). Those columns are scaled to the full
    output width at the full source height: a document or a web page reads top
    to bottom, so nothing is cut vertically, and the text comes out up to ~1.6x
    the size it has when the whole 16:9 screen is squeezed into 1080px.
    """
    from ffmpeg_utils import blurred_backdrop

    x, w = crop
    fg_h = min(out_h, int(round(out_w * orig_h / float(w))))
    fg_h -= fg_h % 2
    return (
        f"[0:v]split=2[bga][fga];"
        f"[bga]{blurred_backdrop(out_w, out_h, 12)}[bg];"
        f"[fga]crop=w={w}:h={orig_h}:x={x}:y=0,scale={out_w}:{fg_h}[fg];"
        f"[bg][fg]overlay=x=0:y=(H-h)/2,setsar=1[v]"
    )


def detect_screencast_scenes(video_path, scenes, strategies, ranges, samples=6):
    """Route scenes that show wide on-screen content.

    Returns ``{scene_index: ('SCREENCAST', centre) | ('WIDE', focus)}``:

      - SCREENCAST stacks the content over the presenter, for content that
        leaves room beside itself (width below STACK_MAX_WIDTH_FRACTION) and
        where a presenter is actually found.
      - WIDE shows the screen without GENERAL's side crop, for content that
        fills the frame or has no presenter to stack. ``focus`` is the reading
        area (left, right) when the shot check found one, else None.
    """
    if not ENABLED or not ranges:
        return {}
    # Below the gate on purpose: main pulls torch/mediapipe, and the disabled
    # path (the default, and what CI exercises) must not pay that import.
    import cv2
    import main as m

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {}

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    found = {}

    try:
        for i, (start, end) in enumerate(scenes):
            s_f, e_f = start.get_frames(), end.get_frames()
            width, focus = overlapping_range(s_f / fps, e_f / fps, ranges)
            if not width:
                continue

            # Content that fills the frame has the presenter on top of it, so
            # there is nothing to stack — just stop cropping the sides.
            if width > STACK_MAX_WIDTH_FRACTION:
                found[i] = ('WIDE', focus)
                continue
            last_f = e_f - 1
            if total_frames:
                last_f = min(last_f, total_frames - 1)
            if last_f < s_f:
                continue

            centres = []
            for f_idx in np.linspace(s_f, last_f, samples):
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(f_idx)))
                ok, frame = cap.read()
                if not ok:
                    continue
                centre = _face_centre(detect_faces_full_res(frame), frame_w)
                if centre is None:
                    # A presenter keyed into the corner of a screen recording is
                    # often too small for BlazeFace even at full resolution
                    # (measured: zero detections across an Excel walkthrough
                    # where the person is plainly visible). YOLO finds the body
                    # in the same frames, and a body centre frames the speaker
                    # just as well for this layout.
                    person = m.detect_person_yolo(frame)
                    if person:
                        centre = (person[0] + person[2] / 2.0,
                                  person[1] + person[3] / 2.0)
                if centre:
                    centres.append(centre)

            # Half the samples: a webcam inset is static and easy to find, so a
            # weaker signal than this means there is no presenter to stack, and
            # the content still deserves its full width.
            if len(centres) < samples / 2.0:
                found[i] = ('WIDE', focus)
                continue

            found[i] = ('SCREENCAST',
                        (float(np.median([c[0] for c in centres])),
                         float(np.median([c[1] for c in centres]))))
    finally:
        cap.release()

    return found
