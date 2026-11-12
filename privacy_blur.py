"""Opt-in blur of faces, license plates and on-screen personal details.

Off unless the job was started with ``BLUR_SENSITIVE=1``. The dashboard, the
API and MCP all default to off, and a value sitting in the server environment
is not a switch: every caller pops it and sets it only when this request asked.

Detection stays on this machine. Frames are never uploaded to find the regions
(the privacy policy's promise about footage and the model provider). The pass
runs after reframing, because the crop still needs the real faces, and before
captions and hooks, so burned text stays sharp.

The blur is baked into the canonical clip. It is the picture the user asked
for, not a delivery mark like the free-plan watermark, so a later download of
the clean file, a fast recut and a caption restyle all keep it. A source
re-render paints new pixels and must run this pass again.

A failure refuses the clip. Shipping the unblurred frame after the user asked
for it to be hidden is the worse outcome. Finding nothing is a success: the
file is left untouched.
"""
import os
import re
import shutil
import subprocess
import tempfile
import threading

ENV_NAME = "BLUR_SENSITIVE"

# How often the picture is scanned, in frames. Boxes are held across the
# frames in between and padded, so a face that blinks or a plate that shifts
# does not flash back into view. Text is read less often (it sits still).
SAMPLE_STRIDE = 2
DETECT_MAX_WIDTH = 640
OCR_MAX_WIDTH = 960
# COCO: car, motorcycle, bus, truck. Plates are searched inside these first.
VEHICLE_CLASSES = (2, 3, 5, 7)
_HOLD_MISSES = {"face": 5, "plate": 4, "qr": 4, "text": 3}
# Pad as a fraction of the box: sideways (each side), above, below.
_PAD = {
    "face": (0.55, 0.65, 0.45),
    "plate": (0.40, 0.45, 0.45),
    "text": (0.12, 0.45, 0.45),
    "qr": (0.08, 0.08, 0.08),
}

_EMAIL = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9][A-Za-z0-9.\-]*\.[A-Za-z]{2,}")
_SSN = re.compile(r"(?<!\d)\d{3}-\d{2}-\d{4}(?!\d)")
_IBAN = re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{11,30}\b")
# A loose number-shaped span. _is_phone / _is_card decide what it really is,
# so a year, a date or an IP address is not blurred just for containing digits.
_LOOSE_NUMBER = re.compile(r"(?<!\w)\+?\d[\d\s().\-]{6,24}\d(?!\w)")
_IPV4 = re.compile(r"^(?:\d{1,3}\.){3}\d{1,3}$")
_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")

_FACE_LOCK = threading.Lock()
_FACE = None
_FACE_ERR = None
_WARNED = set()


def job_env_value(raw):
    """``'1'`` when this request asked for the blur, else None.

    Absent, false and anything unrecognised stay off. Callers pop
    ``BLUR_SENSITIVE`` from the child environment first, so a deployment-wide
    value cannot turn the feature on for every job.
    """
    if raw is None or raw is False:
        return None
    if raw is True:
        return "1"
    if str(raw).strip().lower() in ("1", "true", "yes"):
        return "1"
    return None


def wanted(flag=None):
    """Whether this render should blur.

    ``True`` / ``False`` are an explicit decision (a recut running in the API
    process, which does not carry the job's environment). ``None`` reads the
    job environment, and only the exact value ``1`` counts.
    """
    if flag is True:
        return True
    if flag is False:
        return False
    return os.environ.get(ENV_NAME, "").strip() == "1"


def from_job(env, metadata):
    """The choice to honour on a later recut or reframe.

    The job environment is the authority. Metadata is a file in the job
    directory, so only a JSON boolean ``true`` counts, and only when the
    environment no longer says (a resumed job). A string or a ``1`` does not.
    """
    env = env or {}
    if ENV_NAME in env:
        return env.get(ENV_NAME) == "1"
    return (metadata or {}).get("blur_sensitive") is True


def pii_spans(text):
    """Character spans of emails, phone numbers, card numbers, SSNs and IBANs.

    Empty when ``text`` has none. Dates, IPv4 addresses and bare digit runs
    are not personal details.
    """
    if not text:
        return []
    spans = [(m.start(), m.end()) for m in _EMAIL.finditer(text)]
    spans += [(m.start(), m.end()) for m in _SSN.finditer(text)]
    # upper() keeps the indexes: IBAN letters are case-insensitive, emails are not.
    spans += [(m.start(), m.end()) for m in _IBAN.finditer(text.upper())]
    for m in _LOOSE_NUMBER.finditer(text):
        chunk = m.group(0)
        if _is_phone(chunk) or _is_card(chunk):
            spans.append((m.start(), m.end()))
    return _merge_spans(spans)


def boxes_for_line(words):
    """``(x, y, w, h, 'text')`` boxes for the words of one line that are personal.

    ``words`` are in reading order, each ``{text, x, y, w, h}``. A phone number
    split across words is judged on the joined line, and only the words that
    overlap the match are boxed.
    """
    if not words:
        return []
    parts = []
    spans = []
    pos = 0
    for word in words:
        token = str(word.get("text") or "")
        if not token:
            continue
        if parts:
            pos += 1
        start = pos
        parts.append(token)
        pos += len(token)
        spans.append((start, pos, word))
    hits = pii_spans(" ".join(parts))
    if not hits:
        return []
    chosen = []
    for start, end, word in spans:
        if any(end > hs and start < he for hs, he in hits):
            chosen.append(word)
    return _union_runs(chosen)


def expand_box(x, y, w, h, kind, frame_w, frame_h):
    """Pad ``(x, y, w, h)`` for ``kind`` and clamp it to the frame.

    None when the box falls outside the frame or would cover almost all of
    it (a detector glitch must not frost the whole shot). A face may fill a
    close-up; a plate or a line of text may not.
    """
    px, top, bottom = _PAD.get(kind, (0.1, 0.1, 0.1))
    nx = x - w * px
    ny = y - h * top
    nw = w * (1 + 2 * px)
    nh = h * (1 + top + bottom)
    if frame_w <= 0 or frame_h <= 0 or nw <= 0 or nh <= 0:
        return None
    if (nw * nh) / float(frame_w * frame_h) > (0.95 if kind == "face" else 0.40):
        return None
    x0 = max(0, int(round(nx)))
    y0 = max(0, int(round(ny)))
    x1 = min(int(frame_w), int(round(nx + nw)))
    y1 = min(int(frame_h), int(round(ny + nh)))
    if x1 - x0 < 2 or y1 - y0 < 2:
        return None
    return (x0, y0, x1 - x0, y1 - y0, kind)


def update_tracks(tracks, detections, searched):
    """Carry boxes from one frame to the next.

    ``detections`` is None on a frame that was not scanned: every track is
    held and none of them age. Otherwise only kinds in ``searched`` are
    matched; a face track stays put on a frame where only text was read.
    A kind that was searched and missed ages, and drops after its hold.
    """
    if detections is None:
        return [dict(t) for t in tracks]
    searched = set(searched or ())
    used = set()
    out = []
    for track in tracks:
        if track["kind"] not in searched:
            out.append(dict(track))
            continue
        best_i, best_iou = None, 0.25
        for i, det in enumerate(detections):
            if i in used or det[4] != track["kind"]:
                continue
            score = _iou(track["box"], det[:4])
            if score > best_iou:
                best_i, best_iou = i, score
        if best_i is None:
            missed = track["missed"] + 1
            if missed <= _HOLD_MISSES.get(track["kind"], 3):
                out.append({**track, "missed": missed})
        else:
            used.add(best_i)
            det = detections[best_i]
            out.append({"box": det[:4], "kind": det[4], "missed": 0})
    for i, det in enumerate(detections):
        if i not in used and det[4] in searched:
            out.append({"box": det[:4], "kind": det[4], "missed": 0})
    return out


def find_plates(bgr, vehicles=None):
    """License-plate boxes ``(x, y, w, h, 'plate')`` in ``bgr`` pixel coords.

    Classical, no extra model: a plate is a wide rectangle with a strong
    edge. Inside a detected vehicle the bar is lower; on the whole frame it
    is stricter, so a window or a title card is less likely to qualify.
    ``vehicles`` are ``(x, y, w, h)`` or empty.
    """
    import cv2
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape[:2]
    found = []
    rois = []
    for box in vehicles or []:
        x, y, w, h = [int(v) for v in box[:4]]
        # Plates sit in the lower half of a vehicle, not on the roof.
        y0 = y + int(h * 0.35)
        rois.append((max(0, x), max(0, y0), w, max(0, y + h - y0), False))
    rois.append((0, 0, width, height, True))
    for x, y, w, h, strict in rois:
        x = max(0, min(width - 1, x))
        y = max(0, min(height - 1, y))
        w = max(0, min(width - x, w))
        h = max(0, min(height - y, h))
        if w < 24 or h < 12:
            continue
        found.extend(_plates_in(gray[y:y + h, x:x + w], x, y, width, height, strict))
    return merge_boxes(found)


def parse_tsv(text, scale=1.0):
    """Word boxes from a tesseract TSV, scaled back to the original frame.

    Confidence below 35 is dropped. The text column may itself contain tabs.
    """
    lines = (text or "").splitlines()
    if len(lines) < 2:
        return []
    words = []
    for line in lines[1:]:
        cols = line.split("\t")
        if len(cols) < 12:
            continue
        try:
            if int(cols[0]) != 5:
                continue
            conf = float(cols[10])
        except ValueError:
            continue
        token = "\t".join(cols[11:]).strip()
        if conf < 35 or not token:
            continue
        try:
            left, top, bw, bh = (float(cols[i]) for i in (6, 7, 8, 9))
        except ValueError:
            continue
        words.append({
            "block": cols[2], "par": cols[3], "line": cols[4],
            "text": token,
            "x": int(left * scale), "y": int(top * scale),
            "w": max(1, int(bw * scale)), "h": max(1, int(bh * scale)),
        })
    return words


def pii_boxes(words):
    """Text boxes for personal details across an OCR word list."""
    groups = []
    key = None
    current = None
    for word in words:
        group_key = (word.get("block"), word.get("par"), word.get("line"))
        if group_key != key:
            current = []
            groups.append(current)
            key = group_key
        current.append(word)
    boxes = []
    for group in groups:
        boxes.extend(boxes_for_line(group))
    return merge_boxes(boxes)


def merge_boxes(boxes, min_iou=0.4):
    """Union boxes of the same kind that overlap."""
    out = []
    for box in boxes:
        merged = False
        for i, kept in enumerate(out):
            if kept[4] != box[4] or _iou(kept[:4], box[:4]) < min_iou:
                continue
            x1 = min(kept[0], box[0])
            y1 = min(kept[1], box[1])
            x2 = max(kept[0] + kept[2], box[0] + box[2])
            y2 = max(kept[1] + kept[3], box[1] + box[3])
            out[i] = (x1, y1, x2 - x1, y2 - y1, kept[4])
            merged = True
            break
        if not merged:
            out.append(tuple(box))
    return out


def quads_to_boxes(points):
    """Axis-aligned boxes around QR corner quads. ``points`` is array-like."""
    if points is None:
        return []
    try:
        import numpy as np
        arr = np.asarray(points, dtype=float)
    except Exception:
        return []
    if arr.size == 0:
        return []
    if arr.ndim == 2:
        arr = arr.reshape(1, -1, arr.shape[-1])
    if arr.ndim != 3 or arr.shape[-1] < 2:
        return []
    boxes = []
    for quad in arr:
        xs, ys = quad[:, 0], quad[:, 1]
        x0, y0 = int(xs.min()), int(ys.min())
        x1, y1 = int(xs.max()), int(ys.max())
        if x1 - x0 >= 8 and y1 - y0 >= 8:
            boxes.append((x0, y0, x1 - x0, y1 - y0, "qr"))
    return boxes


def apply(path, *, yolo=None, detect_lock=None, detect=None):
    """Blur ``path`` in place. True when the file is safe to deliver.

    False when the request could not be honoured (the face detector is
    missing, the file cannot be read, the encode fails). The original file
    is left as it was on failure; the caller deletes it rather than serve it.

    ``detect(frame, index)`` replaces the real detectors. Tests use it so
    this module stays runnable without MediaPipe.
    """
    try:
        import cv2
    except Exception as e:
        print(f"   ⚠️ Privacy blur: OpenCV is not available ({e}).")
        return False
    cap = cv2.VideoCapture(path)
    try:
        if not cap.isOpened():
            print(f"   ⚠️ Privacy blur: could not read {os.path.basename(path)}.")
            return False
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        if fps <= 1:
            fps = 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        if width < 2 or height < 2:
            print("   ⚠️ Privacy blur: the clip has no picture.")
            return False
        if detect is None:
            try:
                _face_detector()
            except RuntimeError as e:
                print(f"   ⚠️ Privacy blur: {e}. The clip is not delivered.")
                return False
        try:
            regions_per_frame, hits = _scan_clip(
                cap, cv2, fps, width, height, yolo, detect_lock, detect)
        except Exception as e:
            print(f"   ⚠️ Privacy blur: scan failed ({type(e).__name__}: {e}). "
                  f"The clip is not delivered.")
            return False
    finally:
        cap.release()
    if not regions_per_frame:
        print("   ⚠️ Privacy blur: no frames could be read. The clip is not delivered.")
        return False
    if hits == 0:
        print("   🔒 Privacy blur: scanned the clip, nothing to hide.")
        return True
    print(f"   🔒 Privacy blur: hiding sensitive regions in {hits} frame(s).")
    return _encode(path, regions_per_frame, width, height, fps)


# ---------------------------------------------------------------- internals


def _is_phone(chunk):
    stripped = chunk.strip()
    if _IPV4.match(stripped) or _DATE.match(stripped):
        return False
    digits = sum(c.isdigit() for c in stripped)
    if not (10 <= digits <= 15):
        return False
    return any(c in " \t.-()" for c in stripped)


def _is_card(chunk):
    digits = "".join(c for c in chunk if c.isdigit())
    if not (13 <= len(digits) <= 19) or len(set(digits)) < 2:
        return False
    return _luhn(digits)


def _luhn(digits):
    total = 0
    alt = False
    for ch in reversed(digits):
        n = ord(ch) - 48
        if alt:
            n *= 2
            if n > 9:
                n -= 9
        total += n
        alt = not alt
    return total % 10 == 0


def _merge_spans(spans):
    if not spans:
        return []
    spans = sorted(spans)
    out = [list(spans[0])]
    for start, end in spans[1:]:
        if start <= out[-1][1]:
            out[-1][1] = max(out[-1][1], end)
        else:
            out.append([start, end])
    return [(s, e) for s, e in out]


def _union_runs(words):
    if not words:
        return []
    words = sorted(words, key=lambda w: w["x"])
    runs = [[words[0]]]
    for word in words[1:]:
        prev = runs[-1][-1]
        gap = word["x"] - (prev["x"] + prev["w"])
        if gap <= max(prev["h"], word["h"]):
            runs[-1].append(word)
        else:
            runs.append([word])
    boxes = []
    for run in runs:
        x0 = min(w["x"] for w in run)
        y0 = min(w["y"] for w in run)
        x1 = max(w["x"] + w["w"] for w in run)
        y1 = max(w["y"] + w["h"] for w in run)
        boxes.append((x0, y0, max(1, x1 - x0), max(1, y1 - y0), "text"))
    return boxes


def _iou(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x1, y1 = max(ax, bx), max(ay, by)
    x2, y2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    if inter <= 0:
        return 0.0
    union = aw * ah + bw * bh - inter
    return inter / union if union else 0.0


def _plates_in(roi, origin_x, origin_y, frame_w, frame_h, strict):
    import cv2
    if roi.size == 0:
        return []
    blur = cv2.bilateralFilter(roi, 7, 40, 40)
    edges = cv2.Canny(blur, 40, 160)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 3))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(closed, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    min_w = 36 if strict else 24
    min_h = 10 if strict else 8
    found = []
    frame_area = float(frame_w * frame_h) or 1.0
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w < min_w or h < min_h or h > w:
            continue
        aspect = w / float(h)
        if not (2.0 <= aspect <= 6.5):
            continue
        area = w * h
        if area / frame_area > 0.15:
            continue
        if area < (700 if strict else 280):
            continue
        rect_area = cv2.contourArea(contour)
        if rect_area <= 0 or area / rect_area > 8:
            continue
        # A plate is busy inside (glyphs). A plain light rectangle is not.
        # The rim is a strong edge on its own, so the measure stays inset:
        # counting the border would blur every pale window and title card.
        patch = roi[y:y + h, x:x + w]
        inset = max(2, min(w, h) // 8)
        if patch.size == 0 or h <= inset * 2 or w <= inset * 2:
            continue
        inner = patch[inset:h - inset, inset:w - inset]
        if inner.size == 0 or float(inner.std()) < (18 if strict else 12):
            continue
        found.append((origin_x + x, origin_y + y, w, h, "plate"))
    return found


def _downscale(frame, max_width):
    import cv2
    height, width = frame.shape[:2]
    if width <= max_width:
        return frame, 1.0
    scale = width / float(max_width)
    small_h = max(int(round(height / scale)), 2)
    return cv2.resize(frame, (max_width, small_h), interpolation=cv2.INTER_AREA), scale


def _face_detector():
    global _FACE, _FACE_ERR
    with _FACE_LOCK:
        if _FACE_ERR:
            raise RuntimeError(_FACE_ERR)
        if _FACE is None:
            try:
                import mediapipe as mp
                _FACE = mp.solutions.face_detection.FaceDetection(
                    model_selection=1, min_detection_confidence=0.5)
            except Exception as e:
                _FACE_ERR = (f"face detector unavailable ({type(e).__name__}: {e})")
                raise RuntimeError(_FACE_ERR) from e
        return _FACE


def _warn_once(kind, exc):
    if kind in _WARNED:
        return
    _WARNED.add(kind)
    print(f"   ⚠️ Privacy blur: {kind} detection failed "
          f"({type(exc).__name__}: {exc}).")


def _faces(frame):
    import cv2
    height, width = frame.shape[:2]
    small, _scale = _downscale(frame, DETECT_MAX_WIDTH)
    rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
    rgb.flags.writeable = False
    try:
        with _FACE_LOCK:
            detector = _face_detector()
            result = detector.process(rgb)
    except RuntimeError:
        raise
    except Exception as e:
        _warn_once("face", e)
        return []
    boxes = []
    for detection in getattr(result, "detections", None) or []:
        box = detection.location_data.relative_bounding_box
        x = int(box.xmin * width)
        y = int(box.ymin * height)
        bw = int(box.width * width)
        bh = int(box.height * height)
        if bw >= 8 and bh >= 8:
            boxes.append((x, y, bw, bh, "face"))
    return boxes


def _vehicles(frame, yolo, lock):
    if yolo is None:
        return []
    small, scale = _downscale(frame, DETECT_MAX_WIDTH)
    try:
        if lock is None:
            results = yolo(small, verbose=False, classes=list(VEHICLE_CLASSES))
        else:
            with lock:
                results = yolo(small, verbose=False, classes=list(VEHICLE_CLASSES))
    except Exception as e:
        _warn_once("vehicle", e)
        return []
    boxes = []
    for result in results or []:
        for box in getattr(result, "boxes", None) or []:
            x1, y1, x2, y2 = (float(v) * scale for v in box.xyxy[0])
            bw, bh = x2 - x1, y2 - y1
            if bw >= 16 and bh >= 16:
                boxes.append((int(x1), int(y1), int(bw), int(bh)))
    return boxes


def _qrs(frame):
    import cv2
    small, scale = _downscale(frame, DETECT_MAX_WIDTH)
    try:
        detector = cv2.QRCodeDetector()
        points = None
        try:
            ok, pts = detector.detectMulti(small)
            if ok:
                points = pts
        except Exception:
            points = None
        if points is None:
            ok, pts = detector.detect(small)
            if ok:
                points = pts
    except Exception as e:
        _warn_once("qr", e)
        return []
    boxes = []
    for x, y, w, h, kind in quads_to_boxes(points):
        boxes.append((int(x * scale), int(y * scale),
                      max(1, int(w * scale)), max(1, int(h * scale)), kind))
    return boxes


_OCR_CHECKED = False
_OCR_OK = False


def _tesseract_ok():
    global _OCR_CHECKED, _OCR_OK
    if not _OCR_CHECKED:
        _OCR_CHECKED = True
        _OCR_OK = shutil.which("tesseract") is not None
        if not _OCR_OK:
            print("   🔒 Privacy blur: tesseract is not installed, so emails, "
                  "phone numbers and card numbers on screen are not blurred. "
                  "Faces, license plates and QR codes still are.")
    return _OCR_OK


def _text(frame):
    if not _tesseract_ok():
        return []
    import cv2
    small, scale = _downscale(frame, OCR_MAX_WIDTH)
    fd, name = tempfile.mkstemp(suffix=".png")
    os.close(fd)
    try:
        if not cv2.imwrite(name, small):
            return []
        proc = subprocess.run(
            ["tesseract", name, "stdout", "--psm", "11", "-l", "eng", "tsv"],
            capture_output=True, timeout=25)
        if proc.returncode != 0:
            return []
        words = parse_tsv(proc.stdout.decode("utf-8", "replace"), scale)
        return pii_boxes(words)
    except Exception as e:
        _warn_once("text", e)
        return []
    finally:
        if os.path.exists(name):
            os.remove(name)


def _scan_clip(cap, cv2, fps, width, height, yolo, detect_lock, detect):
    ocr_every = max(int(round(fps)) // 2, 1)
    tracks = []
    vehicles = []
    regions_per_frame = []
    hits = 0
    index = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if detect is not None:
            dets = [(int(d[0]), int(d[1]), int(d[2]), int(d[3]), str(d[4]))
                    for d in (detect(frame, index) or [])]
            searched = {d[4] for d in dets} or {"face", "plate", "text", "qr"}
            tracks = update_tracks(tracks, dets, searched)
        else:
            sample = index % SAMPLE_STRIDE == 0
            read_text = index % ocr_every == 0
            if not sample and not read_text:
                tracks = update_tracks(tracks, None, ())
            else:
                dets = []
                searched = set()
                if sample:
                    searched.update(("face", "plate", "qr"))
                    dets.extend(_faces(frame))
                    if yolo is not None and index % (SAMPLE_STRIDE * 3) == 0:
                        vehicles = _vehicles(frame, yolo, detect_lock)
                    if vehicles:
                        small, scale = _downscale(frame, DETECT_MAX_WIDTH)
                        scaled = [(int(x / scale), int(y / scale),
                                   max(1, int(w / scale)), max(1, int(h / scale)))
                                  for x, y, w, h in vehicles]
                        for px, py, pw, ph, kind in find_plates(small, scaled):
                            dets.append((int(px * scale), int(py * scale),
                                         max(1, int(pw * scale)),
                                         max(1, int(ph * scale)), kind))
                    else:
                        small, scale = _downscale(frame, DETECT_MAX_WIDTH)
                        for px, py, pw, ph, kind in find_plates(small, None):
                            dets.append((int(px * scale), int(py * scale),
                                         max(1, int(pw * scale)),
                                         max(1, int(ph * scale)), kind))
                    dets.extend(_qrs(frame))
                if read_text:
                    searched.add("text")
                    dets.extend(_text(frame))
                tracks = update_tracks(tracks, dets, searched)
        regions = []
        for track in tracks:
            box = expand_box(*track["box"], track["kind"], width, height)
            if box is not None:
                regions.append(box)
        if regions:
            hits += 1
        regions_per_frame.append(regions)
        index += 1
    return regions_per_frame, hits


def _encode(path, regions_per_frame, width, height, fps):
    from ffmpeg_utils import METADATA_SCRUB, QUALITY_FAST, video_encode_args
    import cv2
    import numpy as np
    # Not a .mp4 name: the remote worker uploads every mp4 in the job
    # directory, and a half-written temp would go out as a clip. The
    # container is named with -f because the temp suffix is not one ffmpeg
    # can guess.
    tmp = path + ".privacy.partial"
    enc = list(video_encode_args(QUALITY_FAST))
    if "-pix_fmt" not in enc:
        enc += ["-pix_fmt", "yuv420p"]
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "bgr24",
        "-video_size", f"{width}x{height}",
        "-framerate", f"{fps:.6f}", "-i", "pipe:0",
        "-i", path,
        "-map", "0:v:0", "-map", "1:a?",
        *enc, "-c:a", "copy",
        *METADATA_SCRUB, "-movflags", "+faststart", "-f", "mp4", tmp,
    ]
    proc = subprocess.Popen(
        cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    written = 0
    rc = 1
    err = b""
    cap = cv2.VideoCapture(path)
    try:
        for regions in regions_per_frame:
            ok, frame = cap.read()
            if not ok:
                break
            if frame.shape[1] != width or frame.shape[0] != height:
                frame = cv2.resize(frame, (width, height))
            if regions:
                frame = _blur(frame, regions)
            proc.stdin.write(np.ascontiguousarray(frame).tobytes())
            written += 1
        proc.stdin.close()
        err = proc.stderr.read()
        rc = proc.wait(timeout=1800)
    except Exception as e:
        try:
            if proc.stdin and not proc.stdin.closed:
                proc.stdin.close()
        except Exception:
            pass
        try:
            proc.kill()
        except Exception:
            pass
        try:
            proc.wait(timeout=30)
        except Exception:
            pass
        try:
            err = proc.stderr.read() if proc.stderr else b""
        except Exception:
            err = b""
        if not err:
            msg = getattr(e, "strerror", None) or str(e)
            err = msg if isinstance(msg, bytes) else str(msg).encode()
        rc = 1
    finally:
        cap.release()
        if proc.stdin and not proc.stdin.closed:
            try:
                proc.stdin.close()
            except Exception:
                pass
        if os.path.exists(tmp) and (rc != 0 or written != len(regions_per_frame)):
            os.remove(tmp)
    if rc != 0 or written != len(regions_per_frame) or not os.path.isfile(tmp) or os.path.getsize(tmp) == 0:
        tail = err.decode("utf-8", "replace")[-300:] if isinstance(err, bytes) else str(err)[-300:]
        print(f"   ⚠️ Privacy blur encode failed ({tail}). The clip is not delivered.")
        if os.path.exists(tmp):
            os.remove(tmp)
        return False
    os.replace(tmp, path)
    return True


def _blur(frame, regions):
    import cv2
    import numpy as np
    height, width = frame.shape[:2]
    painted = frame.copy()
    mask = np.zeros((height, width), np.float32)
    for x, y, bw, bh, kind in regions:
        x = max(0, min(width - 1, int(x)))
        y = max(0, min(height - 1, int(y)))
        bw = max(0, min(width - x, int(bw)))
        bh = max(0, min(height - y, int(bh)))
        if bw < 2 or bh < 2:
            continue
        roi = painted[y:y + bh, x:x + bw]
        sigma = max(6.0, min(bw, bh) * (0.25 if kind == "face" else 0.18))
        try:
            painted[y:y + bh, x:x + bw] = cv2.GaussianBlur(roi, (0, 0), sigma)
        except cv2.error:
            continue
        if kind == "face":
            ell = np.zeros((bh, bw), np.uint8)
            cv2.ellipse(ell, (bw // 2, bh // 2), (max(bw // 2, 1), max(bh // 2, 1)),
                        0, 0, 360, 255, -1)
            mask[y:y + bh, x:x + bw] = np.maximum(
                mask[y:y + bh, x:x + bw], ell.astype(np.float32) / 255.0)
        else:
            mask[y:y + bh, x:x + bw] = 1.0
    kernel = max(3, (min(height, width) // 40) | 1)
    if kernel % 2 == 0:
        kernel += 1
    if 3 <= kernel < min(height, width):
        mask = cv2.GaussianBlur(mask, (kernel, kernel), 0)
    mixed = painted.astype(np.float32) * mask[..., None] + frame.astype(np.float32) * (1.0 - mask[..., None])
    return np.clip(mixed, 0, 255).astype(np.uint8)
