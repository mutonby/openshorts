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

# The content shown above the speaker is what lies beside them: this much of
# the frame width at least (else there is nothing to stack and the scene is
# left to the face classifier), cut this far from their body.
MIN_CONTENT_SIDE = 0.3
PERSON_GAP = 0.02

# The content band never takes more than this share of the output height, so
# the speaker below keeps a usable size.
MAX_CONTENT_BAND = 0.55

# A person this tall (fraction of the frame height) in a "screen" shot is
# standing in a filmed room, not laid over a screen (_filmed_person).
# main.detect_person_yolo keeps the top 40% of the YOLO box (head and chest).
FILMED_PERSON_HEIGHT = 0.6
PERSON_BOX_SHARE = 0.4

# A face this many face widths wide stands in for the body when YOLO finds no
# person (head and shoulders).
BODY_FACE_WIDTHS = 3.0


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


def content_side(frame_w, person_left, person_right):
    """(x, w) of the columns beside the presenter, or None when there are none.

    The content is on the side away from the person, from the frame edge up to
    PERSON_GAP short of their body. Until 17-oct-2026 the top band was the
    whole frame, so the presenter showed twice, once small with the chart and
    once enlarged below; on a person in front of a whiteboard that is all the
    layout produced.
    """
    gap = frame_w * PERSON_GAP
    centre = (person_left + person_right) / 2.0
    if centre >= frame_w / 2.0:
        x, w = 0, int(person_left - gap)
    else:
        x = int(person_right + gap)
        w = frame_w - x
    if w < frame_w * MIN_CONTENT_SIDE:
        return None
    x += x % 2
    w -= w % 2
    return x, w


def screencast_filtergraph(orig_w, orig_h, out_w, out_h, face_centre, content):
    """The content columns above, the face-framed speaker below.

    ``content`` is content_side()'s (x, w). It is fitted whole into the top
    band (never cropped further) over a blurred backdrop; the speaker takes
    the rest of the height.
    """
    from ffmpeg_utils import blurred_backdrop

    cx_, cw_ = content
    max_h = int(out_h * MAX_CONTENT_BAND)
    scale = min(out_w / float(cw_), max_h / float(orig_h))
    sw = min(out_w, int(round(cw_ * scale)))
    sh = min(max_h, int(round(orig_h * scale)))
    sw -= sw % 2
    sh -= sh % 2
    speaker_h = out_h - sh
    cw, ch, cx, cy = speaker_crop(orig_w, orig_h, out_w, speaker_h, face_centre)

    return (
        f"[0:v]split=3[bga][ca][sa];"
        f"[bga]{blurred_backdrop(out_w, out_h, 14)}[bg];"
        f"[ca]crop=w={cw_}:h={orig_h}:x={cx_}:y=0,scale={sw}:{sh}[content];"
        f"[sa]crop=w={cw}:h={ch}:x={cx}:y={cy},scale={out_w}:{speaker_h}[speaker];"
        f"[bg][content]overlay=x={(out_w - sw) // 2}:y=0[top];"
        f"[top][speaker]overlay=x=0:y={sh},setsar=1[v]"
    )


def detect_faces_full_res(frame):
    """Face boxes from the UNSCALED frame, in original coordinates.

    main.detect_face_candidates() runs detection on a 640px copy, which is the
    right trade for a talking head whose face spans a third of the frame. A
    presenter inset into a screen recording does not survive it: measured on a
    1920x1080 Excel walkthrough, the presenter's ~110px face becomes ~37px at
    640 and BlazeFace returned zero detections on every sample. At full width
    the same frames detect fine. Six samples per scene, so the cost is paid on
    screencast candidates only.
    """
    import cv2
    import main as m

    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    with m.DETECT_LOCK:
        results = m.face_detection.process(rgb)
    if not results.detections:
        return []

    out = []
    for detection in results.detections:
        b = detection.location_data.relative_bounding_box
        box = [int(b.xmin * w), int(b.ymin * h),
               int(b.width * w), int(b.height * h)]
        out.append({'box': box, 'score': box[2] * box[3]})
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
        frames = list(frame_sampler.read_at_path(video_path, cap, mids))
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
    fg_h = crop_content_height(orig_h, out_w, out_h, w)
    return (
        f"[0:v]split=2[bga][fga];"
        f"[bga]{blurred_backdrop(out_w, out_h, 12)}[bg];"
        f"[fga]crop=w={w}:h={orig_h}:x={x}:y=0,scale={out_w}:{fg_h}[fg];"
        f"[bg][fg]overlay=x=0:y=(H-h)/2,setsar=1[v]"
    )


def crop_content_height(orig_h, out_w, out_h, crop_w):
    """Height (even) that ``crop_w`` source columns fill at the output width."""
    fg_h = min(out_h, int(round(out_w * orig_h / float(crop_w))))
    return fg_h - fg_h % 2


def content_band(out_h, fg_h):
    """(top, bottom) of a vertically centred foreground, as frame fractions.
    Recorded in the layout sidecar so the hook can stay off the picture."""
    top = (out_h - fg_h) / 2.0 / out_h
    return round(top, 3), round(1.0 - top, 3)


# --- WIDE without a reading area ----------------------------------------------
#
# A 16:9 screen kept at its full width in a 9:16 frame is 608 px tall out of
# 1920: 32% of the frame, a strip in a sea of blur. Fine text and small game
# objects do not survive that on a phone (visual review, 17-oct-2026), and
# gameplay (chases, heists, pileups) is exactly what lands here.
#
# So WIDE trades a little width for size, the way GENERAL does, but sized by
# where the hook and the captions sit rather than by presence: the default hook
# spot (top at 20%, a two-line hook ends near 29%) and the bottom captions
# (from ~75%) leave a centred window of ~41% of the height. Filling it takes a
# 16:9 screen from 32% to 41% of the frame (x1.3: text and objects 30% larger)
# and keeps 77% of the width, trimming at most ~11% off each side: the HUD
# corners, sidebars and browser chrome, not the middle of the action. The
# horizontal position follows the motion in the shot (wide_action_centre), so
# a pileup on the right keeps the right edge and loses the left one instead.
#
# WIDE_MAX_ZOOM caps it for wider-than-16:9 sources (ultrawide gameplay would
# otherwise lose half its width); a source that already fills the window
# (4:3 and narrower) is left whole. WIDE_CONTENT_HEIGHT_RATIO=0.32 restores the
# old full-width WIDE. The explicit "full frame" framing (force_strategy WIDE)
# never goes through here: the user asked for the whole width.
WIDE_CONTENT_HEIGHT_RATIO = float(
    os.environ.get("WIDE_CONTENT_HEIGHT_RATIO", "0.41"))
WIDE_MAX_ZOOM = 1.4
# Below this the crop would only shave slivers: keep the whole width.
WIDE_MIN_ZOOM = 1.05

# The crop moves off centre only when the moving pixels say so: the best
# window must hold this much more of the shot's motion than the centred one.
WIDE_ACTION_MIN_GAIN = 0.1
# Windows within this share of the best count as tied; the most central wins.
WIDE_ACTION_TIE = 0.005
WIDE_ACTION_SAMPLES = 8
WIDE_ACTION_WIDTH = 160      # columns of the motion profile
WIDE_ACTION_NOISE = 16       # grey levels of difference ignored (compression)


def wide_crop(orig_w, orig_h, out_w, out_h, centre=0.5):
    """(x, w) of the source columns a WIDE scene shows, or None for the
    whole width. ``centre`` is the crop centre as a fraction of the width."""
    full_h = out_w * orig_h / float(orig_w)
    zoom = min(WIDE_MAX_ZOOM, WIDE_CONTENT_HEIGHT_RATIO * out_h / full_h)
    if zoom < WIDE_MIN_ZOOM:
        return None
    w = int(round(orig_w / zoom))
    w -= w % 2
    x = int(round(centre * orig_w - w / 2.0))
    x = max(0, min(x, orig_w - w))
    return x - (x % 2), w


def action_window_centre(profile, frac):
    """Centre (fraction of the width) of the window of width ``frac`` that
    holds the most of ``profile`` (motion per column), or None when the motion
    gives no reason to leave the centre: no motion at all, or the best window
    beats the centred one by less than WIDE_ACTION_MIN_GAIN of the total.
    Among near-ties the window closest to the centre wins, so motion spread
    over the whole frame (a camera pan, a third-person chase) stays centred."""
    profile = np.asarray(profile, dtype=np.float64)
    n = len(profile)
    total = float(profile.sum())
    if n == 0 or total <= 0 or frac >= 1.0:
        return None
    win = max(1, min(n, int(round(frac * n))))
    sums = np.convolve(profile, np.ones(win), mode="valid") / total
    centred = (n - win) // 2
    best = float(sums.max())
    if best - float(sums[centred]) < WIDE_ACTION_MIN_GAIN:
        return None
    near = np.flatnonzero(sums >= best - WIDE_ACTION_TIE)
    start = int(near[np.argmin(np.abs(near - centred))])
    return (start + win / 2.0) / n


def wide_action_centre(video_path, start_f, end_f, frac,
                       samples=WIDE_ACTION_SAMPLES):
    """Where the action is in a WIDE scene, for wide_crop's ``centre``.

    Motion, not detail: a HUD, a minimap or a pillarbox bar is busy or dark
    but still, and must not pull the crop; the cars of a pileup move. Each
    sample differences two frames ~0.1 s apart at low resolution and adds the
    changed pixels per column. None when nothing can be read or the motion is
    spread evenly (action_window_centre)."""
    import cv2

    cap = cv2.VideoCapture(video_path)
    try:
        if not cap.isOpened():
            return None
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        gap = max(1, int(round(fps * 0.1)))
        last = end_f - 1 - gap
        if total:
            last = min(last, total - 1 - gap)
        if last < start_f:
            return None
        profile = None
        for f_idx in np.linspace(start_f, last, samples):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(f_idx)))
            ok, a = cap.read()
            for _ in range(gap - 1):
                ok = ok and cap.grab()
            ok2, b = cap.read() if ok else (False, None)
            if not (ok and ok2):
                continue
            h = max(1, int(round(a.shape[0] * WIDE_ACTION_WIDTH / a.shape[1])))
            ga = cv2.cvtColor(cv2.resize(a, (WIDE_ACTION_WIDTH, h)), cv2.COLOR_BGR2GRAY)
            gb = cv2.cvtColor(cv2.resize(b, (WIDE_ACTION_WIDTH, h)), cv2.COLOR_BGR2GRAY)
            diff = cv2.absdiff(ga, gb) > WIDE_ACTION_NOISE
            col = diff.sum(axis=0).astype(np.float64)
            profile = col if profile is None else profile + col
        if profile is None:
            return None
        return action_window_centre(profile, frac)
    finally:
        cap.release()


def _filmed_person(cap, m, s_f, last_f, samples=3):
    """True when most samples show a person standing in the shot at full size.

    A presenter laid over a screen recording is small: 22-40% of the frame
    height across the corpus's webcam clips, keyed full-body cut-outs
    included. A person filmed in a room in front of a whiteboard fills most
    of the height.
    """
    import cv2
    import numpy as np

    frame_h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0
    if not frame_h:
        return False
    hits = 0
    for f_idx in np.linspace(s_f, last_f, samples):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(f_idx)))
        ok, frame = cap.read()
        if not ok:
            continue
        person = m.detect_person_yolo(frame)
        # detect_person_yolo returns the top PERSON_BOX_SHARE of the body.
        if person and person[3] / PERSON_BOX_SHARE >= FILMED_PERSON_HEIGHT * frame_h:
            hits += 1
    return hits * 2 > samples


def detect_screencast_scenes(video_path, scenes, strategies, ranges, samples=6):
    """Route scenes that show wide on-screen content.

    Returns ``{scene_index: ('SCREENCAST', (centre, content)) | ('WIDE', focus)}``:

      - SCREENCAST stacks the content over the presenter, for content that
        leaves room beside itself (width below STACK_MAX_WIDTH_FRACTION) and
        where a presenter is actually found. ``content`` is the (x, w) of the
        columns beside them (content_side); a scene whose presenter leaves no
        such room is not returned at all.
      - WIDE shows the screen over a blurred backdrop, for content that
        fills the frame or has no presenter to stack: cut to its reading area
        (focus_crop) or, without one, slightly zoomed around the action
        (wide_crop). ``focus`` is the reading area (left, right) when the shot
        check found one, else None.
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

            last_f = e_f - 1
            if total_frames:
                last_f = min(last_f, total_frames - 1)
            if last_f < s_f:
                continue

            # A whiteboard or TV in a filmed room is not a screen layout: the
            # shot check called a presenter standing at his chart "beside" in
            # two runs and "screen" in another (17-oct-2026), whatever the
            # prompt says, and stacking showed him twice while WIDE shrank
            # him into a strip. The face classifier frames that shot best.
            # An edited split screen with a full-size presenter goes the same
            # way and loses its slide, which is the lesser harm.
            if _filmed_person(cap, m, s_f, last_f):
                continue

            # Content that fills the frame has the presenter on top of it, so
            # there is nothing to stack — just stop cropping the sides.
            if width > STACK_MAX_WIDTH_FRACTION:
                found[i] = ('WIDE', focus)
                continue

            centres, extents = [], []
            for f_idx in np.linspace(s_f, last_f, samples):
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(f_idx)))
                ok, frame = cap.read()
                if not ok:
                    continue
                faces = detect_faces_full_res(frame)
                centre = _face_centre(faces, frame_w)
                # The body's horizontal extent, so the content band can leave
                # the person out: YOLO's box when it finds one, else head and
                # shoulders around the face.
                person = m.detect_person_yolo(frame)
                if centre is None and person:
                    # A presenter keyed into the corner of a screen recording is
                    # often too small for BlazeFace even at full resolution
                    # (measured: zero detections across an Excel walkthrough
                    # where the person is plainly visible). YOLO finds the body
                    # in the same frames, and a body centre frames the speaker
                    # just as well for this layout.
                    centre = (person[0] + person[2] / 2.0,
                              person[1] + person[3] / 2.0)
                if not centre:
                    continue
                centres.append(centre)
                if person:
                    extents.append((person[0], person[0] + person[2]))
                else:
                    face_w = max(c['box'][2] for c in faces)
                    half = face_w * BODY_FACE_WIDTHS / 2.0
                    extents.append((centre[0] - half, centre[0] + half))

            # Half the samples: a webcam inset is static and easy to find, so a
            # weaker signal than this means there is no presenter to stack, and
            # the content still deserves its full width.
            if len(centres) < samples / 2.0:
                found[i] = ('WIDE', focus)
                continue

            content = content_side(
                frame_w, float(np.median([e[0] for e in extents])),
                float(np.median([e[1] for e in extents])))
            if content is None:
                # The person takes the frame: nothing beside them to stack.
                # The face classifier's verdict stands.
                continue
            found[i] = ('SCREENCAST',
                        ((float(np.median([c[0] for c in centres])),
                          float(np.median([c[1] for c in centres]))), content))
    finally:
        cap.release()

    return found
