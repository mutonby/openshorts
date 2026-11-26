"""Find a streamer's webcam overlay in gameplay footage, and the stacked
"streamer" layout built from it: facecam on top, gameplay below.

The stacked format is how stream highlights are posted. camera_inset.py does
the opposite stack (screen on top, person below) and needs MediaPipe or YOLO,
which a CPU-only install does not have. This module needs only OpenCV:

- YuNet (OpenCV's own face detector, MIT, 230 KB, vendored in assets/models)
  runs on frames sampled across the clip. The webcam face is the one that
  stays put: it is found in most samples at nearly the same place, while game
  characters' faces come and go and move.
- The overlay's border is found from edges that persist over time. An OBS
  inset has hard straight borders that stay still while the game behind them
  moves, so the per-pixel median of the edge strength over the samples keeps
  the borders and averages the gameplay away. Without a clear border the box
  is the face grown by fixed proportions.

``detect`` returns None whenever this is not clearly a webcam over gameplay:
callers fall back to the full-picture gameplay layout, never to a guess.
"""
import hashlib
import math
import os
import subprocess

MODEL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "models",
                     "face_detection_yunet_2023mar.onnx")
MODEL_SHA256 = "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4"
ANALYSIS_WIDTH = 1280
SAMPLES = 16
# Seconds looked at either side of a clip to find its webcam overlay.
CONTEXT_SECONDS = 45.0
# Frame pairs inside the clip measuring where the game's action is.
MOTION_SAMPLES = 8
# The same face must be in this share of the samples, within this distance
# (fraction of the frame width) of its median position.
MIN_PRESENCE = 0.5
MAX_DRIFT = 0.04
# A webcam overlay is a minority of the frame.
MIN_BOX_WIDTH, MAX_BOX_WIDTH = 0.06, 0.45
CAM_BAND = 0.40          # legacy default seam, for clips rendered before composition.py
FACE_POSITION = 0.55     # face centre this far down the band: room for the hook above

_model_ok = None


def model_available():
    """The vendored model is present and is the file we pinned."""
    global _model_ok
    if _model_ok is None:
        try:
            with open(MODEL, "rb") as f:
                _model_ok = hashlib.sha256(f.read()).hexdigest() == MODEL_SHA256
        except OSError:
            _model_ok = False
        if _model_ok:
            import cv2
            _model_ok = hasattr(cv2, "FaceDetectorYN")
    return _model_ok


def _frame_at(source, seconds, width):
    """One BGR frame at ``seconds`` (file time), scaled to ``width``, or None."""
    import numpy as np

    try:
        out = subprocess.run(
            ["ffmpeg", "-nostdin", "-v", "error", "-ss", f"{max(0.0, seconds):.3f}", "-i", str(source),
             "-frames:v", "1", "-vf", f"scale={width}:-2", "-f", "rawvideo", "-pix_fmt", "bgr24", "-"],
            capture_output=True, timeout=60)
    except (OSError, subprocess.SubprocessError):
        return None
    data = out.stdout
    if out.returncode or not data:
        return None
    height = len(data) // (width * 3)
    if height < 2:
        return None
    return np.frombuffer(data[:width * height * 3], dtype=np.uint8).reshape(height, width, 3)


def _faces(detector, frame):
    detector.setInputSize((frame.shape[1], frame.shape[0]))
    _, found = detector.detect(frame)
    out = []
    for row in found if found is not None else []:
        x, y, w, h = (float(v) for v in row[:4])
        if w > 4 and h > 4:
            out.append((x, y, w, h, float(row[-1])))
    return out


def _persistent_face(per_frame, width):
    """(x, y, w, h) median box of the face found at the same place in most
    samples, or None. Several stable faces (a duo on camera): the largest."""
    anchors = []
    for faces in per_frame:
        for face in faces:
            cx, cy = face[0] + face[2] / 2, face[1] + face[3] / 2
            group = next((g for g in anchors if math.hypot(g["cx"] - cx, g["cy"] - cy) < MAX_DRIFT * width
                          and 0.6 < face[2] / g["w"] < 1.6), None)
            if group is None:
                anchors.append({"cx": cx, "cy": cy, "w": face[2], "faces": [face]})
            else:
                group["faces"].append(face)
    stable = _stable_faces(anchors, len(per_frame))
    return stable[0] if stable else None


def _stable_faces(anchors, samples):
    """Median boxes of every face group found in most samples, largest first."""
    out = []
    for group in anchors:
        if len(group["faces"]) < max(3, MIN_PRESENCE * samples):
            continue
        boxes = sorted(group["faces"], key=lambda f: f[2])
        out.append(boxes[len(boxes) // 2][:4])
    return sorted(out, key=lambda b: -b[2] * b[3])


def _persistent_faces(per_frame, width):
    """Every face that stays put across the samples, largest first (a duo on
    one webcam is two)."""
    anchors = []
    for faces in per_frame:
        for face in faces:
            cx, cy = face[0] + face[2] / 2, face[1] + face[3] / 2
            group = next((g for g in anchors if math.hypot(g["cx"] - cx, g["cy"] - cy) < MAX_DRIFT * width
                          and 0.6 < face[2] / g["w"] < 1.6), None)
            if group is None:
                anchors.append({"cx": cx, "cy": cy, "w": face[2], "faces": [face]})
            else:
                group["faces"].append(face)
    return _stable_faces(anchors, len(per_frame))


def _walk(static_line, start, step, stop, tolerance=4):
    """Walk from ``start`` toward ``stop`` while the strips stay mostly still;
    returns the last still position (the moving game begins past it). Up to
    ``tolerance`` moving lines in a row are crossed: the streamer's own body
    moves inside the webcam picture."""
    pos, misses, last = start, 0, start
    while (step < 0 and pos > stop) or (step > 0 and pos < stop):
        pos += step
        if static_line(pos) >= 0.5:
            last, misses = pos, 0
        else:
            misses += 1
            if misses > tolerance:
                break
    return last


def _sharpness(still_line, limit, step):
    """How abruptly stillness ends at ``limit``: still inside, moving outside
    (0..1). An overlay border over moving gameplay is close to 1."""
    inside = [still_line(limit - step * k) for k in range(0, 3)]
    outside = [still_line(limit + step * k) for k in range(2, 6)]
    return max(0.0, min(inside) - max(outside))


def _candidates(profile, near, far, limit, sharp):
    """Border positions for one side. A sharp still-to-moving transition, or
    the frame edge, IS the border. Only a soft one (a dark, static game behind
    the webcam) lets persistent edges between it and the face compete."""
    import numpy as np

    if int(limit) == int(far) or sharp >= 0.6:
        return {int(limit): 1.0}
    lo, hi = sorted((int(near), int(limit)))
    out = {int(limit): 0.3}
    if hi - lo >= 3:
        seg = profile[lo:hi + 1]
        floor = 3.5 * float(np.median(profile)) + 1e-6
        for i in range(1, len(seg) - 1):
            if seg[i] >= floor and seg[i] >= seg[i - 1] and seg[i] >= seg[i + 1]:
                out[lo + i] = min(1.0, float(seg[i]) / (2.0 * floor))
    return out


def _refine(frames, face, width, height):
    """The overlay rectangle around ``face``.

    Three cues, because none is enough alone (a real room behind the streamer
    is as still and as edgy as the overlay border): the webcam picture stays
    still while the game moves (temporal spread per pixel), the border is a
    straight edge that persists over time, and webcam overlays are about 16:9.
    Each side gets a few candidate positions; the combination with the best
    edges and the most webcam-like shape wins. Nothing convincing: the face
    grown by fixed proportions."""
    import itertools
    import cv2
    import numpy as np

    x, y, w, h = face
    grays = np.stack([cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype("float32") for f in frames])
    spread = grays.std(axis=0)
    still = spread < max(6.0, 0.5 * float(np.median(spread)))
    edges_v = np.median(np.stack([np.abs(cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)) for g in grays]), axis=0)
    edges_h = np.median(np.stack([np.abs(cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)) for g in grays]), axis=0)

    r0, r1 = int(max(0, y - 0.5 * h)), int(min(height, y + 1.5 * h))
    col_profile = edges_v[r0:r1].mean(axis=0)
    col_still = lambda c: float(still[r0:r1, min(width - 1, max(0, c))].mean())
    left_limit = _walk(col_still, int(x), -1, 0, tolerance=int(w))
    right_limit = _walk(col_still, int(x + w), 1, width - 1, tolerance=int(w))
    c0, c1 = int(max(0, left_limit)), int(min(width, right_limit + 1))
    row_profile = edges_h[:, c0:c1].mean(axis=1)
    row_still = lambda r: float(still[min(height - 1, max(0, r)), c0:c1].mean())
    top_limit = _walk(row_still, int(y), -1, 0, tolerance=int(0.5 * h))
    bottom_limit = _walk(row_still, int(y + h), 1, height - 1, tolerance=int(h))

    sides = [
        _candidates(col_profile, x, 0, left_limit, _sharpness(col_still, left_limit, -1)),
        _candidates(col_profile, x + w, width - 1, right_limit, _sharpness(col_still, right_limit, 1)),
        _candidates(row_profile, y, 0, top_limit, _sharpness(row_still, top_limit, -1)),
        _candidates(row_profile, y + h, height - 1, bottom_limit, _sharpness(row_still, bottom_limit, 1)),
    ]
    best, best_score = None, -1e9
    for (l, sl), (r, sr), (t, st), (b, sb) in itertools.product(*(s.items() for s in sides)):
        bw, bh = r - l, b - t
        if bw < 1.5 * w or bh < 1.3 * h or bw > MAX_BOX_WIDTH * width or bh > 0.6 * height:
            continue
        shape = abs(math.log((bw / bh) / (16 / 9)))
        score = sl + sr + st + sb - 2.0 * shape
        if score > best_score:
            best, best_score = (l, t, bw, bh), score
    if best is not None and best_score > 0.6:
        return tuple(float(v) for v in best), True
    # Fallback proportions: a head-and-shoulders webcam about 3 face-widths
    # wide, the face in its upper half.
    left, top = max(0.0, x - 1.1 * w), max(0.0, y - 0.9 * h)
    right, bottom = min(float(width), x + 2.1 * w), min(float(height), y + 2.2 * h)
    return (left, top, right - left, bottom - top), False


def _motion_columns(frames, pairs):
    """Changed pixels per column between frames ~0.1 s apart (the action),
    at analysis width. None without usable pairs."""
    import cv2
    import numpy as np

    profile = None
    for a, b in zip(frames, pairs):
        if b is None or a.shape != b.shape:
            continue
        diff = cv2.absdiff(cv2.cvtColor(a, cv2.COLOR_BGR2GRAY), cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)) > 18
        col = diff.sum(axis=0).astype("float64")
        profile = col if profile is None else profile + col
    return profile


def overlay_evidence(frames, box):
    """How clearly ``box`` is a picture laid over a DIFFERENT, changing one:
    the share of the frame outside the box whose pixels change across the
    samples. Gameplay behind a webcam changes almost everywhere; a full-screen
    webcam in a still room barely changes outside the streamer. 0..1."""
    import cv2
    import numpy as np

    grays = np.stack([cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype("float32") for f in frames])
    moving = grays.std(axis=0) >= 10.0
    x, y, w, h = (int(round(v)) for v in box)
    mask = np.ones(moving.shape, dtype=bool)
    mask[max(0, y):max(0, y + h), max(0, x):max(0, x + w)] = False
    outside = int(mask.sum())
    return float(moving[mask].sum()) / outside if outside else 0.0


# Below this share of changing pixels around the webcam the "overlay" may be
# a streamer filmed full screen: not stacked, and flagged as uncertain.
MIN_OVERLAY_EVIDENCE = 0.30
# A box with a clear border or flush with the frame edge (where streaming
# software puts webcams) needs less: a dark or slow game scene changes little.
MIN_OVERLAY_EVIDENCE_EDGE = 0.12


def _still_region(frames, face, width, height):
    """The webcam overlay as the still region around the face: across samples
    spread over the clip the game changes almost everywhere while the webcam's
    room stays put. The streamer moves inside it, so the head-and-shoulders
    area around the face counts as part of it. Returns (x, y, w, h) at
    analysis size, or None when the still region is not a compact block (a
    dark or paused game is still everywhere; then the edge method decides)."""
    import cv2
    import numpy as np

    grays = np.stack([cv2.cvtColor(cv2.resize(f, (width // 4, height // 4)), cv2.COLOR_BGR2GRAY)
                      .astype("float32") for f in frames])
    still = (grays.std(axis=0) < 10.0).astype("uint8")
    x, y, w, h = (v / 4.0 for v in face)
    # The streamer: head, shoulders and the chair below, never still.
    x0, x1 = int(max(0, x - 0.7 * w)), int(min(still.shape[1], x + 1.7 * w))
    y0, y1 = int(max(0, y - 0.6 * h)), int(min(still.shape[0], y + 2.6 * h))
    still[y0:y1, x0:x1] = 1
    k = max(3, int(w * 0.4)) | 1
    still = cv2.morphologyEx(still, cv2.MORPH_CLOSE, np.ones((k, k), np.uint8))
    still = cv2.morphologyEx(still, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(still, connectivity=4)
    label = labels[int(min(still.shape[0] - 1, y + h / 2)), int(min(still.shape[1] - 1, x + w / 2))]
    if label == 0:
        return None
    bx, by, bw, bh, area = (int(v) for v in stats[label])
    sh, sw = still.shape
    if bw > MAX_BOX_WIDTH * sw or bh > 0.65 * sh or bw * bh > 0.30 * sw * sh:
        return None
    if area < 0.70 * bw * bh or bw < 1.5 * w or bh < 1.3 * h:
        return None
    # The block must reach well past the streamer's own area (marked still
    # above): a region that is only that area found no overlay at all, and
    # calling it the border would let the game crop show the rest of the
    # webcam.
    if bw * bh < 1.4 * (x1 - x0) * (y1 - y0):
        return None
    return bx * 4.0, by * 4.0, bw * 4.0, bh * 4.0


def _union(boxes):
    x0 = min(b[0] for b in boxes)
    y0 = min(b[1] for b in boxes)
    x1 = max(b[0] + b[2] for b in boxes)
    y1 = max(b[1] + b[3] for b in boxes)
    return x0, y0, x1 - x0, y1 - y0


def _grow(box, k, width, height):
    x, y, w, h = box
    x0, y0 = max(0.0, x - k * w), max(0.0, y - k * h)
    x1, y1 = min(float(width), x + w + k * w), min(float(height), y + h + k * h)
    return x0, y0, x1 - x0, y1 - y0


def _head_and_shoulders(people, face, width, height):
    """Hair or a beanie above the detected face, shoulders below."""
    x, y, w, h = people
    fw, fh = face[2], face[3]
    x0, y0 = max(0.0, x - 0.8 * fw), max(0.0, y - 0.8 * fh)
    x1, y1 = min(float(width), x + w + 0.8 * fw), min(float(height), y + h + 1.0 * fh)
    return x0, y0, x1 - x0, y1 - y0


def _touches_edge(box, width, height, margin=0.03):
    x, y, w, h = box
    return x <= margin * width or y <= margin * height or \
        x + w >= (1 - margin) * width or y + h >= (1 - margin) * height


def avoid_region(box, face, bordered, width, height, face_w=None):
    """The part of the frame a game crop must stay clear of so the webcam is
    never shown twice. A box found by its border or its still region is the
    overlay itself. Otherwise the box is only proportions around the face, so
    the margin is generous (a webcam is several face widths wide) and runs
    to any frame edge close by, where streaming software pins webcams."""
    if bordered:
        return box
    fx, fy, pw, ph = face
    fw = face_w or pw
    fh = fw * 1.3
    x0, y0 = min(box[0], fx - 3.0 * fw), min(box[1], fy - 1.5 * fh)
    x1, y1 = max(box[0] + box[2], fx + pw + 3.0 * fw), max(box[1] + box[3], fy + ph + 3.0 * fh)
    if x0 < 0.15 * width:
        x0 = 0.0
    if x1 > 0.85 * width:
        x1 = float(width)
    if y0 < 0.15 * height:
        y0 = 0.0
    if y1 > 0.85 * height:
        y1 = float(height)
    x0, y0 = max(0.0, x0), max(0.0, y0)
    x1, y1 = min(float(width), x1), min(float(height), y1)
    return x0, y0, x1 - x0, y1 - y0


def _duration(source):
    try:
        out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                              "-of", "csv=p=0", str(source)], capture_output=True, text=True, timeout=30)
        return float(out.stdout.strip())
    except (OSError, subprocess.SubprocessError, ValueError):
        return None


def detect(source, start, end, *, samples=SAMPLES, with_motion=False, context=CONTEXT_SECONDS):
    """The webcam overlay in [start, end] (file seconds) of ``source``:
    {"box", "face", "people", "avoid": (x, y, w, h), "source_size",
    "evidence", "bordered", "confident"} in source pixels, or None.

    The overlay is looked for over ``context`` seconds either side of the
    clip too (a webcam does not move during a stream, and more footage is
    more game motion to tell it apart). ``with_motion`` also returns
    ``motion``: changed pixels per column between frames 0.1 s apart inside
    the clip itself (overlay included), for the game crop."""
    import cv2

    if not model_available() or end <= start:
        return None
    duration = _duration(source)
    lo = max(0.0, start - context)
    hi = min(duration, end + context) if duration else end
    if hi <= lo:
        lo, hi = start, end
    frames = []
    for i in range(samples):
        frame = _frame_at(source, lo + (hi - lo) * (i + 0.5) / samples, ANALYSIS_WIDTH)
        if frame is not None:
            frames.append(frame)
    if len(frames) < max(4, samples // 2):
        return None
    height, width = frames[0].shape[:2]
    frames = [f for f in frames if f.shape[:2] == (height, width)]
    clip_frames, pairs = [], []
    if with_motion:
        for i in range(MOTION_SAMPLES):
            at = start + (end - start) * (i + 0.5) / MOTION_SAMPLES
            a, b = _frame_at(source, at, ANALYSIS_WIDTH), _frame_at(source, at + 0.1, ANALYSIS_WIDTH)
            if a is not None and b is not None and a.shape[:2] == (height, width):
                clip_frames.append(a)
                pairs.append(b)
    detector = cv2.FaceDetectorYN.create(MODEL, "", (width, height), 0.7, 0.3, 50)
    stable = _persistent_faces([_faces(detector, f) for f in frames], width)
    if not stable:
        return None
    face = stable[0]
    # Everyone else sharing the webcam (a duo on one camera): close to the
    # main face and about its size. The overlay holds all of them.
    group = [f for f in stable if math.hypot((f[0] + f[2] / 2) - (face[0] + face[2] / 2),
                                             (f[1] + f[3] / 2) - (face[1] + face[3] / 2)) < 5 * face[2]
             and f[2] > 0.5 * face[2]]
    people = _union(group)
    region = _still_region(frames, people, width, height)
    if region is not None:
        box, bordered, method = region, True, "still"
    else:
        box, bordered = _refine(frames, people, width, height)
        method = "edges" if bordered else "proportions"
    # Whatever the border says, the facecam shows whole heads: a cut-out
    # streamer keyed onto the game has no rectangle, and an edge inside the
    # webcam (a counter, a desk) once cut the crop through a beanie.
    box = _union([box, _head_and_shoulders(people, face, width, height)])
    if not MIN_BOX_WIDTH * width <= box[2] <= MAX_BOX_WIDTH * width or box[3] > 0.6 * height:
        return None
    evidence = overlay_evidence(frames, box)
    probe = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                            "stream=width,height", "-of", "csv=p=0", str(source)],
                           capture_output=True, text=True, timeout=30)
    try:
        src_w, src_h = (int(v) for v in probe.stdout.strip().split(",")[:2])
    except ValueError:
        return None
    scale = src_w / float(width)

    def up(b):
        return tuple(int(round(v * scale)) for v in b)

    found = {"box": up(box), "face": up(face), "people": up(people), "faces": len(group),
             # Only a still region is the webcam's own edge; anything else
             # gets the generous margin, so the game crop cannot show the
             # rest of it.
             "avoid": up(avoid_region(box, people, method == "still", width, height, face_w=face[2])),
             "method": method,
             "source_size": (src_w, src_h),
             "evidence": round(evidence, 3), "bordered": bordered,
             "confident": evidence >= MIN_OVERLAY_EVIDENCE or (
                 evidence >= MIN_OVERLAY_EVIDENCE_EDGE and (bordered or _touches_edge(box, width, height)))}
    if with_motion:
        motion = _motion_columns(clip_frames, pairs)
        if motion is not None:
            found["motion"] = [float(v) for v in motion]
    return found


def present(source, cam, start, end, samples=3):
    """Whether the overlay's face is in its place during [start, end]: a
    scene where the streamer went full screen, or a cut away from the stream
    layout, is not stacked."""
    import cv2

    if not model_available() or end <= start:
        return False
    src_w = cam["source_size"][0]
    fx, fy, fw, fh = (v * ANALYSIS_WIDTH / float(src_w) for v in cam["face"])
    hits = seen = 0
    for i in range(samples):
        frame = _frame_at(source, start + (end - start) * (i + 0.5) / samples, ANALYSIS_WIDTH)
        if frame is None:
            continue
        seen += 1
        detector = cv2.FaceDetectorYN.create(MODEL, "", (frame.shape[1], frame.shape[0]), 0.6, 0.3, 50)
        for x, y, w, h, _ in _faces(detector, frame):
            if abs((x + w / 2) - (fx + fw / 2)) < 0.6 * fw and abs((y + h / 2) - (fy + fh / 2)) < 0.6 * fh:
                hits += 1
                break
    return seen > 0 and hits * 2 >= seen


def _even(v):
    v = int(round(v))
    return v - (v % 2)


def stack_layout(orig_w, orig_h, out_w, out_h, cam, *, focus="balanced", cam_position="top"):
    """Geometry of the stacked layout (composition.stack), or None when the
    gameplay crop cannot avoid showing the webcam a second time.

    The facecam band is sized by the face it shows and the overlay's real
    resolution (composition.cam_band); the gameplay crop slides toward the
    action measured by ``detect(..., with_motion=True)``, never onto the
    webcam."""
    import composition

    action = None
    motion = cam.get("motion")
    if motion:
        src_w = float(cam.get("source_size", (orig_w,))[0] or orig_w)
        bx, _, bw, _ = cam.get("avoid", cam["box"])
        fill_w = orig_h * out_w / float(out_h * (1 - composition.profile(focus)["cam_target"]))
        frac = min(1.0, max(fill_w, composition.profile(focus)["game_keep"] * orig_w) / orig_w)
        action = composition.action_centre(motion, frac, mask=(bx / src_w, (bx + bw) / src_w))
    return composition.stack(orig_w, orig_h, out_w, out_h, cam, focus=focus, action=action,
                             cam_position=cam_position)


def stack_filtergraph(geometry, out_w, out_h, *, cam_zoom_cmds=None, scale_flags=None):
    """[0:v] -> [v]: the facecam band and the gameplay band, stacked (crop,
    scale, gblur, overlay and vstack only, so the GPU renderer reads it).
    A card shows the whole overlay over a blurred copy of itself; a game
    picture shorter than its band sits on a blurred fill of the game.
    ``cam_zoom_cmds``: a sendcmd file driving ``crop@pz`` inside the facecam
    band for the punch-in. ``scale_flags`` (e.g. "lanczos") sharpens the
    upscales where the encoder is FFmpeg."""
    from ffmpeg_utils import escape_filter_value, blurred_backdrop

    flags = f":flags={scale_flags}" if scale_flags else ""
    cw, ch, cx, cy = geometry["cam_crop"]
    gw, gh, gx, gy = geometry["game_crop"]
    cam_h, game_h = geometry["cam_h"], geometry["game_h"]
    pic_h = geometry.get("game_pic_h", game_h)
    card = geometry.get("card")
    punch = ""
    if cam_zoom_cmds:
        punch = (f",sendcmd=f='{escape_filter_value(cam_zoom_cmds)}',"
                 f"crop@pz=w={out_w}:h={cam_h}:x=0:y=0,scale={out_w}:{cam_h}")
    parts = []
    if card:
        parts.append("[0:v]split=4[camsrc][cambg][gamesrc][gamebg]" if pic_h < game_h
                     else "[0:v]split=3[camsrc][cambg][gamesrc]")
        parts.append(f"[cambg]crop=w={cw}:h={ch}:x={cx}:y={cy},{blurred_backdrop(out_w, cam_h, 40)}[cb]")
        parts.append(f"[camsrc]crop=w={cw}:h={ch}:x={cx}:y={cy},scale={card['w']}:{card['h']}{flags}[cc]")
        parts.append(f"[cb][cc]overlay=x={card['x']}:y={card['y']}{punch},setsar=1[cam]")
    else:
        parts.append("[0:v]split=3[camsrc][gamesrc][gamebg]" if pic_h < game_h
                     else "[0:v]split=2[camsrc][gamesrc]")
        parts.append(f"[camsrc]crop=w={cw}:h={ch}:x={cx}:y={cy},scale={out_w}:{cam_h}{flags}{punch},setsar=1[cam]")
    if pic_h < game_h:
        top = geometry.get("cam_position", "top") == "top"
        parts.append(f"[gamesrc]crop=w={gw}:h={gh}:x={gx}:y={gy},scale={out_w}:{pic_h}{flags}[gp]")
        parts.append(f"[gamebg]crop=w={gw}:h={gh}:x={gx}:y={gy},{blurred_backdrop(out_w, game_h, 60)}[gb]")
        parts.append(f"[gb][gp]overlay=x=0:y={0 if top else game_h - pic_h},setsar=1[game]")
    else:
        parts.append(f"[gamesrc]crop=w={gw}:h={gh}:x={gx}:y={gy},scale={out_w}:{game_h}{flags},setsar=1[game]")
    order = "[cam][game]" if geometry.get("cam_position", "top") == "top" else "[game][cam]"
    parts.append(f"{order}vstack=inputs=2,setsar=1[v]")
    return ";".join(parts)
