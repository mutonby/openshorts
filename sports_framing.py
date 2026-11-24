"""Scene-local sports ball tracking with the host's existing YOLO weights."""

import logging
import math
import os
import threading
from dataclasses import dataclass
from functools import lru_cache, partial
from pathlib import Path

log = logging.getLogger(__name__)

MIN_CONFIDENCE = 0.35
MAX_GAP_SECONDS = 0.75
MAX_INPUT_WIDTH = 640
MOTION_WIDTH = 192
SAMPLES_PER_SECOND = 4
_MODEL_LOCK = threading.Lock()


@dataclass(frozen=True)
class Ball:
    x: float
    y: float
    confidence: float


def ball_detector():
    """Return a predictor only for a model already installed on the host."""
    weights = Path(os.getenv("YOLO_MODEL_PATH", "yolov8n.pt")).expanduser().absolute()
    if not weights.is_file():
        return None
    with _MODEL_LOCK:
        return _load_ball_detector(weights)


@lru_cache(maxsize=1)
def _load_ball_detector(weights):
    try:
        from ultralytics import YOLO

        model = YOLO(str(weights))
        classes = [index for index, label in model.names.items()
                   if label.strip().lower() in ("sports ball", "ball")]
    except Exception as exc:
        log.warning("Sports detector unavailable, using visual framing (%s)", type(exc).__name__)
        return None
    return partial(_predict_balls, model, classes) if classes else None


def _predict_balls(model, classes, frame):
    with _MODEL_LOCK:
        results = model(frame, classes=classes, conf=MIN_CONFIDENCE,
                        max_det=4, verbose=False, save=False)
    found = []
    for result in results:
        for box in result.boxes:
            left, top, right, bottom = map(float, box.xyxy[0])
            height, width = frame.shape[:2]
            x, y = (left + right) / (2 * width), (top + bottom) / (2 * height)
            confidence = float(box.conf[0])
            if (0 <= x <= 1 and 0 <= y <= 1 and right > left and bottom > top
                    and MIN_CONFIDENCE <= confidence <= 1):
                found.append(Ball(x, y, confidence))
    return found


def _sample_frames(video_path, indices):
    """Keep just the detections and small grayscale frames for motion fallback."""
    import cv2
    from frame_sampler import read_at_path

    detector = ball_detector()
    cap = cv2.VideoCapture(video_path)
    try:
        for index, frame in zip(indices, read_at_path(video_path, cap, indices)):
            gray = None
            balls = []
            if frame is not None:
                height, width = frame.shape[:2]
                sample_width = min(width, MOTION_WIDTH)
                sample_height = max(2, round(height * sample_width / width))
                gray = cv2.cvtColor(cv2.resize(frame, (sample_width, sample_height),
                                              interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
                if detector is not None:
                    try:
                        if width > MAX_INPUT_WIDTH:
                            frame = cv2.resize(frame, (MAX_INPUT_WIDTH, max(2, round(height * MAX_INPUT_WIDTH / width))),
                                               interpolation=cv2.INTER_AREA)
                        balls = detector(frame)
                    except Exception as exc:
                        log.warning("Sports inference failed, using motion (%s)", type(exc).__name__)
                        detector = None
            yield index, balls, gray
    except Exception as exc:
        log.warning("Sports sampling failed, using wide framing (%s)", type(exc).__name__)
    finally:
        cap.release()


def _distance(first, second):
    return math.hypot(first.x - second.x, first.y - second.y)


def _nearby(first, second, seconds):
    return seconds <= MAX_GAP_SECONDS and _distance(first, second) <= 0.08 + 0.85 * seconds


def _observations(samples, fps):
    """Start with two agreeing observations, then prefer a continuous path."""
    hits = [(index, balls) for index, balls in samples if balls]
    track = []
    for i, (index, balls) in enumerate(hits):
        if not track:
            if i == len(hits) - 1:
                break
            next_index, next_balls = hits[i + 1]
            seconds = (next_index - index) / fps
            candidates = [ball for ball in balls
                          if any(_nearby(ball, other, seconds) for other in next_balls)]
            chosen = max(candidates, key=lambda ball: ball.confidence, default=None)
        else:
            last_index, last_ball = track[-1]
            seconds = (index - last_index) / fps
            if seconds > MAX_GAP_SECONDS:
                break
            candidates = [ball for ball in balls if _nearby(last_ball, ball, seconds)]
            chosen = max(candidates, key=lambda ball: ball.confidence - _distance(ball, last_ball),
                         default=None)
        if chosen is not None:
            track.append((index, chosen))
    return track


def _pan(indices, centers, frame_count, source_width, crop_width, fps, rate):
    """Convert sampled source-pixel centers to a bounded crop per frame."""
    import numpy as np

    max_x = source_width - crop_width
    targets = np.clip(np.interp(np.arange(frame_count), indices, centers) - crop_width / 2,
                      0, max_x)
    current = targets[0]
    step = rate * crop_width / fps
    dead_zone = 0.06 * crop_width
    positions = []
    for target in targets:
        delta = target - current
        if abs(delta) > dead_zone:
            current += min(step, max(-step, delta))
        positions.append(int(current) // 2 * 2)
    return positions


def scene_crop(samples, frame_count, source_width, crop_width, fps):
    """Pan toward a reliable ball track, or let the visual camera cover the shot."""
    track = _observations(samples, fps)
    if len(track) < max(2, math.ceil(len(samples) / 2)):
        return None
    if (track[0][0] / fps > MAX_GAP_SECONDS
            or (frame_count - 1 - track[-1][0]) / fps > MAX_GAP_SECONDS):
        return None

    indices, balls = zip(*track)
    centers = [ball.x * source_width for ball in balls]
    positions = _pan(indices, centers, frame_count, source_width, crop_width, fps, rate=1.25)
    if all(positions[index] <= center <= positions[index] + crop_width
           for index, center in zip(indices, centers)):
        return positions
    return None


def _motion_center(previous, current, crop_width):
    import numpy as np

    # Ignore HUD and captions at the edges. Exposure changes and broad camera
    # movement do not identify a subject worth cropping to.
    height = current.shape[0]
    delta = current[height // 12:height * 5 // 6] - previous[height // 12:height * 5 // 6]
    delta = np.abs(delta - np.median(delta))
    if not 0.008 <= np.mean(delta > 12) <= 0.60:
        return None
    profile = np.clip(delta - 10, 0, 70).mean(axis=0)
    profile[:2] = profile[-2:] = 0
    total = float(profile.sum())
    if total < 12:
        return None
    sums = np.concatenate(([0.0], np.cumsum(profile, dtype=float)))
    scores = sums[crop_width:] - sums[:-crop_width]
    best = float(scores.max())
    if best / total < max(0.57, crop_width / len(profile) + 0.12):
        return None
    starts = np.flatnonzero(scores >= best * 0.97)
    center = float(np.dot(profile, np.arange(len(profile)) + 0.5) / total)
    return float(starts[np.argmin(np.abs(starts + crop_width / 2 - center))] + crop_width / 2)


def motion_crop(samples, frame_count, source_width, crop_width, fps):
    """Follow concentrated motion or preserve the whole shot as a wide view."""
    previous = None
    centers = []
    for index, frame in samples:
        if frame is None:
            previous = None
            continue
        sample_width = frame.shape[1]
        width = max(1, min(sample_width, round(crop_width * sample_width / source_width)))
        current = frame.astype("float32")
        if previous is not None:
            center = _motion_center(previous, current, width)
            if center is not None:
                centers.append((index, center * source_width / sample_width))
        previous = current
    if not centers or len(centers) / len(samples) < 0.60:
        return None
    indices, target = zip(*centers)
    return _pan(indices, target, frame_count, source_width, crop_width, fps, rate=0.55)


def trajectory(path, bounds, fps, source_width, crop_width):
    """Return scene-local crop positions and the upstream renderer's layouts."""
    positions = [None] * bounds[-1][1]
    if source_width <= crop_width:
        return [0] * len(positions), ["TRACK"] * len(bounds)

    stride = max(1, round(fps / SAMPLES_PER_SECOND))
    indices = sorted({*range(0, len(positions), stride), *(start for start, _ in bounds)})
    samples = iter(_sample_frames(path, indices))
    upcoming = next(samples, None)
    strategies = []
    for start, end in bounds:
        ball_samples, motion_samples = [], []
        while upcoming is not None and upcoming[0] < end:
            index, balls, gray = upcoming
            if index >= start:
                ball_samples.append((index - start, balls))
                motion_samples.append((index - start, gray))
            upcoming = next(samples, None)
        crop = scene_crop(ball_samples, end - start, source_width, crop_width, fps)
        if crop is None:
            crop = motion_crop(motion_samples, end - start, source_width, crop_width, fps)
        strategies.append("TRACK" if crop is not None else "WIDE")
        if crop is not None:
            positions[start:end] = crop
    return positions, strategies
