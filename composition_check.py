"""Checks a stream/game clip's composition before it is delivered.

Two halves:

- ``describe``: the planned layout's numbers (composition.py) turned into
  metrics and flags: a facecam too small or too large, upscaled past what its
  pixels carry, a game cropped too hard, too much blur, a stretched picture,
  a webcam the detector was unsure of, a low-resolution source.
- ``verify``: the RENDERED file. FFprobe for size, square pixels and audio
  sync (export_quality), and frames from the beginning, middle and end
  searched for the same face twice: two faces whose pictures match at the
  same instant are one streamer shown twice (the duplicate-facecam bug),
  where two different people (a podcast SPLIT) do not match.

A clip with a ``review`` flag is still delivered, marked ``needs_review`` in
its metadata with the reasons, so a human looks before posting instead of a
broken layout going out silently. Nothing here raises: a failed check is
reported as unchecked, never as a failed clip.
"""
import math
import subprocess

# Flags that put a clip in front of a human. The others are notes.
REVIEW = {"duplicate_face", "facecam_too_small", "facecam_too_large", "facecam_upscaled",
          "distorted", "wrong_size", "non_square_pixels", "av_desync", "blank",
          "game_too_small"}

SAMPLE_SHARES = (0.04, 0.2, 0.35, 0.5, 0.65, 0.8, 0.96)


def describe(chosen, orig_w, orig_h, out_w, out_h):
    """Metrics and flags for viral_edit.choose_layout's decision."""
    import composition

    out = {"layout": chosen.get("kind"), "focus": chosen.get("focus"),
           "source": [orig_w, orig_h], "output": [out_w, out_h], "flags": [], "notes": list(chosen.get("notes") or [])}
    cam = chosen.get("cam")
    if cam:
        out["facecam"] = {"box": list(cam["box"]), "evidence": cam.get("evidence"),
                          "confident": bool(cam.get("confident"))}
    if orig_h < 720:
        out["flags"].append("low_resolution_source")
    g = chosen.get("geometry")
    if g:
        prof = composition.profile(g.get("focus") or "balanced")
        out.update({"facecam_share": round(g["cam_h"] / float(out_h), 3), "face_share": g["face_share"],
                    "facecam_upscale": g["cam_scale"], "game_keep": g["game_keep"],
                    "fill_share": g["fill_share"], "card": bool(g.get("card")),
                    "seam": round(g["seam"], 3), "caption_y": g["caption_y"]})
        if g["face_share"] < 0.75 * prof["face_min"]:
            out["flags"].append("facecam_too_small")
        if g["face_share"] > 1.2 * prof["face_max"]:
            out["flags"].append("facecam_too_large")
        if g["cam_scale"] > 1.15 * prof["max_upscale"]:
            out["flags"].append("facecam_upscaled")
        if g["game_keep"] < 0.35:
            out["flags"].append("game_cropped_hard")
        if g["fill_share"] > 0.25:
            out["flags"].append("excess_blur")
        cw, ch = g["cam_crop"][:2]
        gw, gh = g["game_crop"][:2]
        cam_box = (g["card"]["w"], g["card"]["h"]) if g.get("card") else (out_w, g["cam_h"])
        if (abs((cw / float(ch)) / (cam_box[0] / float(cam_box[1])) - 1) > 0.02
                or abs((gw / float(gh)) / (out_w / float(g["game_pic_h"])) - 1) > 0.02):
            out["flags"].append("distorted")
    else:
        crop = chosen.get("crop")
        shown_w = crop[1] if crop else orig_w
        fg_h = min(out_h, out_w * orig_h / float(shown_w))
        out.update({"game_share": round(fg_h / out_h, 3), "game_keep": round(shown_w / float(orig_w), 3),
                    "fill_share": round(1 - fg_h / out_h, 3)})
        if fg_h / out_h < 0.28 and orig_w > orig_h:
            out["flags"].append("game_too_small")
        if cam:
            face_px = cam["face"][3] * out_w / float(shown_w)
            out["face_share"] = round(face_px / out_h, 4)
            if not cam.get("confident"):
                out["flags"].append("facecam_uncertain")
            if face_px / out_h < 0.025:
                # The streamer is a thumbnail: fine for a pure gameplay clip,
                # but worth a look when the reaction matters.
                out["flags"].append("facecam_small_in_gameplay")
    return out


def _frame(path, seconds, width=720):
    import numpy as np
    try:
        run = subprocess.run(
            ["ffmpeg", "-nostdin", "-v", "error", "-ss", f"{max(0.0, seconds):.3f}", "-i", str(path),
             "-frames:v", "1", "-vf", f"scale={width}:-2", "-f", "rawvideo", "-pix_fmt", "bgr24", "-"],
            capture_output=True, timeout=60)
    except (OSError, subprocess.SubprocessError):
        return None
    if run.returncode or not run.stdout:
        return None
    h = len(run.stdout) // (width * 3)
    return np.frombuffer(run.stdout[:width * h * 3], dtype=np.uint8).reshape(h, width, 3) if h > 1 else None


def _patch(frame, box):
    import cv2
    x, y, w, h = (int(round(v)) for v in box[:4])
    x, y = max(0, x), max(0, y)
    crop = frame[y:y + h, x:x + w]
    if crop.size == 0 or min(crop.shape[:2]) < 8:
        return None
    gray = cv2.cvtColor(cv2.resize(crop, (40, 48)), cv2.COLOR_BGR2GRAY).astype("float32")
    gray -= gray.mean()
    norm = math.sqrt(float((gray * gray).sum()))
    return gray / norm if norm > 1e-6 else None


def same_face_twice(frames_faces):
    """Share of sampled frames holding two faces whose pictures match (the
    same streamer shown twice). ``frames_faces``: [(frame, [box...])]."""
    hits = usable = 0
    for frame, faces in frames_faces:
        if len(faces) < 2:
            if faces:
                usable += 1
            continue
        usable += 1
        patches = [p for p in (_patch(frame, f) for f in faces) if p is not None]
        best = 0.0
        for i in range(len(patches)):
            for j in range(i + 1, len(patches)):
                best = max(best, float((patches[i] * patches[j]).sum()))
        if best >= 0.80:
            hits += 1
    return hits / float(usable) if usable else 0.0


def duplicate_faces(path, duration):
    """(share of samples showing one face twice, samples read), or None
    without the face model."""
    import cv2
    import facecam
    if not facecam.model_available() or not duration:
        return None
    sampled = []
    for share in SAMPLE_SHARES:
        frame = _frame(path, duration * share)
        if frame is None:
            continue
        det = cv2.FaceDetectorYN.create(facecam.MODEL, "", (frame.shape[1], frame.shape[0]), 0.75, 0.3, 50)
        sampled.append((frame, [f[:4] for f in facecam._faces(det, frame)]))
    if not sampled:
        return None
    return same_face_twice(sampled), len(sampled)


def verify(path, *, expected_duration, output_format="vertical", audio_expected=True, planned=None):
    """The rendered clip's report: ``planned`` (describe's dict) plus what
    the file shows. Adds ``needs_review`` and ``review_reasons``."""
    report = dict(planned or {})
    flags = list(report.get("flags") or [])
    try:
        import export_quality
        quality = export_quality.validate(path, expected_duration=expected_duration,
                                          output_format=output_format, audio_expected=audio_expected)
        report["export"] = quality
        for issue in quality.get("issues") or []:
            text = issue.lower()
            flags.append("wrong_size" if "shape" in text or "dimension" in text else
                         "non_square_pixels" if "pixels" in text else
                         "av_desync" if "align" in text or "audio" in text else
                         "blank" if "blank" in text else "export_issue")
        if output_format in ("vertical", "auto"):
            video = quality.get("video") or {}
            if video and (video.get("width"), video.get("height")) != (1080, 1920):
                report["note_size"] = f"{video.get('width')}x{video.get('height')}"
    except Exception as e:
        report["export"] = {"unchecked": type(e).__name__}
    try:
        dup = duplicate_faces(path, expected_duration)
        if dup is not None:
            share, samples = dup
            report["duplicate_face_share"] = round(share, 3)
            report["face_samples"] = samples
            if share >= 0.3:
                flags.append("duplicate_face")
    except Exception as e:
        report["duplicate_check"] = f"unchecked ({type(e).__name__})"
    report["flags"] = sorted(set(flags))
    reasons = [f for f in report["flags"] if f in REVIEW]
    report["needs_review"] = bool(reasons)
    report["review_reasons"] = reasons
    return report
