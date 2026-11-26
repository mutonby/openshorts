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


def sample_detections(video_path, indices):
    """Read one color frame at a time and return scene tracking observations."""
    detector = ball_detector()
    if detector is None:
        return ()

    import cv2
    from frame_sampler import read_at

    cap = cv2.VideoCapture(video_path)
    try:
        observations = []
        for index, frame in zip(indices, read_at(cap, indices)):
            balls = []
            if frame is not None:
                height, width = frame.shape[:2]
                if width > MAX_INPUT_WIDTH:
                    frame = cv2.resize(frame, (MAX_INPUT_WIDTH, round(height * MAX_INPUT_WIDTH / width)),
                                       interpolation=cv2.INTER_AREA)
                balls = detector(frame)
            observations.append((index, balls))
        return observations
    except Exception as exc:
        log.warning("Sports inference failed, using visual framing (%s)", type(exc).__name__)
        return ()
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


def scene_crop(samples, frame_count, source_width, crop_width, fps):
    """Pan toward a reliable ball track, or let the visual camera cover the shot."""
    import numpy as np

    if source_width <= crop_width:
        return [0] * frame_count
    track = _observations(samples, fps)
    if len(track) < max(2, math.ceil(len(samples) / 2)):
        return None
    if (track[0][0] / fps > MAX_GAP_SECONDS
            or (frame_count - 1 - track[-1][0]) / fps > MAX_GAP_SECONDS):
        return None

    indices, balls = zip(*track)
    centers = np.array([ball.x * source_width for ball in balls])
    max_x = source_width - crop_width
    targets = np.clip(np.interp(np.arange(frame_count), indices, centers) - crop_width / 2,
                      0, max_x)
    current = targets[0]
    step = 1.25 * crop_width / fps
    dead_zone = 0.06 * crop_width
    positions = []
    for target in targets:
        delta = target - current
        if abs(delta) > dead_zone:
            current += min(step, max(-step, delta))
        positions.append(int(current) // 2 * 2)

    visible = sum(positions[index] <= center <= positions[index] + crop_width
                  for index, center in zip(indices, centers))
    return positions if visible == len(track) else None
