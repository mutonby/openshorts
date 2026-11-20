"""Lightweight framing from local changes and contrast, without model weights."""

import os


FRAMING_METHODS = ("auto", "motion", "detail", "center", "fit", "legacy")
SAMPLES_PER_SECOND = 4
ANALYSIS_WIDTH = 192


def resolve(method=None):
    """Use the deployment default when a caller has no per-video setting."""
    method = method if method is not None else os.environ.get("FRAMING_METHOD", "legacy")
    return method if method in FRAMING_METHODS else "legacy"


def _body(image):
    # Ignore HUD chrome and existing captions near the edges.
    h = image.shape[0]
    return image[h // 12:h * 5 // 6].astype("float32")


def _detail(image):
    import numpy as np

    horizontal = np.abs(np.diff(image, axis=1, prepend=image[:, :1]))
    vertical = np.abs(np.diff(image, axis=0, prepend=image[:1, :]))
    return np.clip((horizontal + vertical) - 12, 0, 90).mean(axis=0)


def _motion(image, previous):
    import numpy as np

    if previous is None:
        return None
    delta = image - previous
    # A global brightness change is often a flash, fade or exposure change.
    delta = np.abs(delta - np.median(delta))
    if not 0.008 <= np.mean(delta > 12) <= 0.60:
        return None
    return np.clip(delta - 10, 0, 70).mean(axis=0)


def _focus(profile, width):
    """Centre of a crop capturing most of the evidence, or None for wide shots."""
    import numpy as np

    if profile is None or float(profile.sum()) < 12:
        return None
    profile[:2] = profile[-2:] = 0
    total = float(profile.sum())
    if total < 12:
        return None
    ends = np.concatenate(([0.0], np.cumsum(profile, dtype=float)))
    scores = ends[width:] - ends[:-width]
    best = float(scores.max())
    if best / total < max(0.57, width / len(profile) + 0.12):
        return None
    # Among near-equal crops, leave equal room on either side of the detail.
    starts = np.flatnonzero(scores >= best * 0.97)
    centre = float(np.dot(profile, np.arange(len(profile)) + 0.5) / total)
    return float(starts[np.argmin(np.abs(starts + width / 2 - centre))] + width / 2)


def scene_crop(samples, frame_count, source_width, crop_width, fps, method):
    """Horizontal crop positions, or None when auto should retain full width.

    Samples are ordered (scene-relative frame number, small grayscale image)
    pairs. Motion never compares frames across a cut. Scene-level confidence
    chooses the layout once, while a dead zone and speed limit keep the crop
    from reacting to noisy individual samples.
    """
    import numpy as np

    if method == "fit":
        return None
    max_x = max(0, source_width - crop_width)
    centered_x = (max_x // 4) * 2
    if method == "center" or not max_x:
        return [centered_x] * frame_count
    if not samples:
        return None if method == "auto" else [centered_x] * frame_count

    sample_w = samples[0][1].shape[1]
    width = max(1, min(sample_w, round(crop_width * sample_w / source_width)))
    positions = []
    previous = None
    for _index, gray in samples:
        body = _body(gray)
        chosen = (_focus(_motion(body, previous), width)
                  if method in ("auto", "motion") else None)
        if chosen is None and method != "motion":
            chosen = _focus(_detail(body), width)
        positions.append(chosen)
        previous = body

    valid = [i for i, pos in enumerate(positions) if pos is not None]
    if not valid:
        return None if method == "auto" else [centered_x] * frame_count
    if method == "auto" and len(valid) / len(samples) < 0.60:
        return None

    # Pre-read the first strong observation so a new shot opens framed.
    # Ignore isolated outliers and hold the last position on weak samples.
    current = positions[valid[0]]
    previous_index = samples[0][0]
    shot_x = []
    for i, ((index, _), position) in enumerate(zip(samples, positions)):
        if position is not None:
            nearby = [p for p in positions[max(0, i - 1):i + 2] if p is not None]
            target = float(np.median(nearby))
            delta = target - current
            if abs(delta) > width * 0.10:
                limit = 0.55 * width * (index - previous_index) / fps
                current += max(-limit, min(limit, delta))
        previous_index = index
        shot_x.append((current - width / 2) * source_width / sample_w)
    interpolated = np.interp(np.arange(frame_count), [i for i, _ in samples], shot_x)
    return [int(x) for x in (np.clip(interpolated, 0, max_x).astype(int) // 2 * 2)]
