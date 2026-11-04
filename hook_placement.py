"""Where the hook text goes when nobody picked a spot (``position="auto"``).

The hook used to sit at a fixed 20% of the frame height. On a 9:16 clip that
is exactly where the speaker's face is: a face-tracked TRACK crop puts the
eyes around a third of the way down, and on a SPLIT clip the top speaker's
face fills the upper half (6-oct-2026: hooks drawn over the forehead of a
TRACK speaker and over the face of the top SPLIT speaker while the captions
sat correctly on the seam).

The rule, kept deliberately simple:

- The hook may go anywhere between the platform's top UI (``TOP_MARGIN``) and
  its bottom UI (``BOTTOM_UI``), but never on a caption band. Captions are
  known without looking: at the bottom on every layout, on the seam
  (mid-frame) on SPLIT stretches (``layout_ranges``, ``subtitles.generate_ass``).
- On a WIDE stretch the picture floats in blur (its band is in the layout
  sidecar): the hook stays off it too when it fits above or below it
  (``content_bands``), and ignores it when it does not.
- Inside those bounds, every candidate height is scored by how much of the
  faces it would cover, averaged over a few frames of the hook's own time
  window. Faces are MediaPipe boxes grown upwards to include the forehead
  and hair, since the hook on a forehead reads as "over the face" too.
- The lowest score wins; ties go to the candidate closest to the old 20%
  spot (``MOVE_PENALTY``), so a clip with no face anywhere (gameplay, a
  slide) renders exactly as before.

The scoring is pure (normalised boxes in, a height out) so it is tested with
synthetic faces; ``detect_faces`` is the only part that touches pixels.
"""
import threading

# All heights are fractions of the frame height (0 = top edge).
TOP_MARGIN = 0.06       # status bar / "Following | For you" tabs
BOTTOM_UI = 0.80        # caption + username block of TikTok / Reels
LEGACY_TOP = 0.20       # where position="top" has always drawn the hook

# Caption bands, measured on burned prod clips (1080x1920, AUTO_CAPTION_STYLE:
# Anton 44, MarginV 43 of PlayResY 288): bottom captions span ~0.75-0.83,
# seam captions ~0.47-0.54. Padded so a two-line block or the pop scale-up
# still clears the hook.
BOTTOM_CAPTIONS = (0.70, 0.88)
SEAM_CAPTIONS = (0.42, 0.58)

# Score units: share of a face covered. 0.05 means moving the hook the whole
# frame height away from 20% costs as much as covering 5% of a face, so the
# hook moves for a face and does not wander for nothing.
MOVE_PENALTY = 0.05
STEP = 0.005
# Below this much face covered (after shrinking, see best_scale) we stop
# trying smaller text.
GOOD_ENOUGH = 0.08
SHRINK_SCALES = (1.0, 0.85, 0.72)

SAMPLES = 6
# A hook shown for the whole clip (duration None) is placed from its opening
# seconds: that is where it is read.
DEFAULT_WINDOW = 5.0


def head_box(face):
    """MediaPipe's box runs from the brows to the chin. Grow it to the whole
    head: hair and forehead above, a little chin below and on the sides.
    ``face`` and the result are (x0, y0, x1, y1) in frame fractions."""
    x0, y0, x1, y1 = face
    w, h = x1 - x0, y1 - y0
    return (x0 - 0.15 * w, y0 - 0.6 * h, x1 + 0.15 * w, y1 + 0.2 * h)


def _covered(rect, box):
    """Share of ``box`` that ``rect`` covers (both x0, y0, x1, y1)."""
    ix = min(rect[2], box[2]) - max(rect[0], box[0])
    iy = min(rect[3], box[3]) - max(rect[1], box[1])
    if ix <= 0 or iy <= 0:
        return 0.0
    area = (box[2] - box[0]) * (box[3] - box[1])
    return (ix * iy) / area if area > 0 else 0.0


def face_cost(rect, frames):
    """Mean over sampled frames of the face area the hook would cover, each
    face counted as a share of its own head box and summed (two half-covered
    faces score like one fully covered). Frames with no face add 0."""
    if not frames:
        return 0.0
    total = 0.0
    for faces in frames:
        total += sum(_covered(rect, head_box(f)) for f in faces or [])
    return total / len(frames)


def caption_bands(ranges, window_start, window_end, has_captions=True):
    """The caption bands the hook must stay off during its window.
    ``ranges`` is the clip's layout_ranges: SPLIT stretches put captions on
    the seam, everything else (or no sidecar at all) at the bottom."""
    if not has_captions:
        return []
    bands = set()
    covered = 0.0
    for r in ranges or []:
        try:
            s, e = max(float(r["start"]), window_start), min(float(r["end"]), window_end)
            layout = str(r.get("layout", "")).lower()
        except (KeyError, TypeError, ValueError):
            continue
        if e <= s:
            continue
        covered += e - s
        bands.add(SEAM_CAPTIONS if layout == "split" else BOTTOM_CAPTIONS)
    # Any part of the window no range describes keeps the default captions.
    if covered < (window_end - window_start) - 1e-3:
        bands.add(BOTTOM_CAPTIONS)
    return sorted(bands)


def content_bands(ranges, window_start, window_end):
    """The pictures of the WIDE stretches inside the hook's window, as
    (top, bottom) bands. The hook goes in the blur above them when it fits;
    a WIDE picture is the whole point of the shot, and on gameplay it has no
    face for the face cost to protect."""
    bands = set()
    for r in ranges or []:
        try:
            s, e = max(float(r["start"]), window_start), min(float(r["end"]), window_end)
            top, bottom = (float(v) for v in r["content"])
        except (KeyError, TypeError, ValueError):
            continue
        if e > s and 0.0 <= top < bottom <= 1.0:
            bands.add((top, bottom))
    return sorted(bands)


def candidates(box_h, bands, top=TOP_MARGIN, bottom=BOTTOM_UI, step=STEP):
    """Every hook top (frame fraction) that keeps the box inside the safe
    area and off every caption band."""
    out = []
    n = int(round((bottom - box_h - top) / step))
    for i in range(n + 1):
        y = round(top + i * step, 4)
        if any(y < b1 and y + box_h > b0 for b0, b1 in bands):
            continue
        out.append(y)
    return out


def choose_y(box_w, box_h, frames, bands, preferred=LEGACY_TOP):
    """(y, cost) for a box of ``box_w`` x ``box_h`` (frame fractions,
    horizontally centred), or (None, inf) when no spot clears the bands.

    ``frames``: one list of face boxes (x0, y0, x1, y1) per sampled frame."""
    x0 = (1.0 - box_w) / 2.0
    best = (None, float("inf"))
    for y in candidates(box_h, bands):
        cost = face_cost((x0, y, x0 + box_w, y + box_h), frames)
        score = cost + MOVE_PENALTY * abs(y - preferred)
        if score < best[1] - 1e-9:
            best = (y, score)
    if best[0] is None:
        return best
    # Report the face cost alone: the move penalty is a tie-breaker.
    y = best[0]
    return y, face_cost((x0, y, x0 + box_w, y + box_h), frames)


def best_scale(measure, frames, bands, scales=SHRINK_SCALES):
    """Try the hook at its own size first and only shrink the text when it
    still covers a face. ``measure(scale) -> (box_w, box_h)`` in frame
    fractions. Returns (scale, y, cost); y None means nothing fits anywhere
    and the caller keeps the legacy spot."""
    tried = []
    for scale in scales:
        box_w, box_h = measure(scale)
        y, cost = choose_y(box_w, box_h, frames, bands)
        if y is not None and cost <= GOOD_ENOUGH:
            return scale, y, cost
        tried.append((cost, -scale, y))
    fits = [t for t in tried if t[2] is not None]
    if not fits:
        return scales[0], None, float("inf")
    cost, neg_scale, y = min(fits)
    return -neg_scale, y, cost


# ---------------------------------------------------------------- pixels

_DETECTOR = None
_DETECTOR_LOCK = threading.Lock()


def _detector():
    global _DETECTOR
    if _DETECTOR is None:
        import mediapipe as mp
        # model_selection=1: the full-range model, so the small faces of a
        # GENERAL (blurred-backdrop) layout are found too.
        _DETECTOR = mp.solutions.face_detection.FaceDetection(
            model_selection=1, min_detection_confidence=0.5)
    return _DETECTOR


def detect_faces(video_path, window_start=0.0, window_end=None, samples=SAMPLES):
    """Face boxes (x0, y0, x1, y1, frame fractions) for ``samples`` frames
    spread over the hook's window, one list per frame. None when the video
    cannot be read or MediaPipe is missing: the caller keeps the old spot."""
    try:
        import cv2
        from frame_sampler import read_at
    except Exception:
        return None
    cap = cv2.VideoCapture(video_path)
    try:
        if not cap.isOpened():
            return None
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        end = window_end if window_end else DEFAULT_WINDOW
        if total:
            end = min(end, total / fps)
        start = max(0.0, min(window_start, end))
        span = max(end - start, 0.0)
        times = [start + span * (i + 0.5) / samples for i in range(samples)]
        indices = sorted({int(t * fps) for t in times})
        if total:
            indices = [min(i, total - 1) for i in indices]
        frames = []
        with _DETECTOR_LOCK:
            detector = _detector()
            for frame in read_at(cap, indices):
                if frame is None:
                    continue
                h, w = frame.shape[:2]
                if w > 720:
                    frame = cv2.resize(frame, (720, int(h * 720 / w)))
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = detector.process(rgb)
                boxes = []
                for d in result.detections or []:
                    b = d.location_data.relative_bounding_box
                    boxes.append((b.xmin, b.ymin, b.xmin + b.width, b.ymin + b.height))
                frames.append(boxes)
        return frames or None
    except Exception as e:
        print(f"⚠️ Hook placement: face detection failed ({type(e).__name__}: {e})")
        return None
    finally:
        cap.release()
