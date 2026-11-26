import sys
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np

import framing
import reframe_v2
import sports_framing
from sports_framing import Ball


SOURCE_WIDTH = 1920
CROP_WIDTH = 608
FPS = 24


def test_sports_crop_follows_a_supported_track_through_a_missing_sample():
    samples = [
        (0, [Ball(0.95, 0.5, 0.99)]),
        (6, [Ball(0.22, 0.5, 0.72)]),
        (12, [Ball(0.27, 0.5, 0.73)]),
        (18, []),
        (24, [Ball(0.34, 0.5, 0.70), Ball(0.96, 0.5, 0.99)]),
        (30, [Ball(0.39, 0.5, 0.78)]),
        (36, [Ball(0.43, 0.5, 0.74)]),
        (42, [Ball(0.46, 0.5, 0.70)]),
    ]

    positions = sports_framing.scene_crop(samples, 48, SOURCE_WIDTH, CROP_WIDTH, FPS)

    assert positions is not None
    assert len(positions) == 48
    assert positions[0] < 160
    assert positions[-1] > 510
    assert max(abs(b - a) for a, b in zip(positions, positions[1:])) <= 32
    assert all(0 <= x <= SOURCE_WIDTH - CROP_WIDTH and x % 2 == 0 for x in positions)


def test_sports_crop_rejects_an_interrupted_track_with_a_long_gap():
    ball = Ball(0.4, 0.5, 0.8)
    samples = [(index, [ball] if index < 18 or index >= 36 else [])
               for index in range(0, 66, 6)]

    assert sports_framing.scene_crop(samples, 66, SOURCE_WIDTH, CROP_WIDTH, FPS) is None


def test_sports_trajectory_starts_fresh_at_each_cut_and_fits_without_evidence(monkeypatch):
    bounds = [(0, 31), (31, 63), (63, 95)]
    strategies = ["TRACK"] * len(bounds)
    frame = np.zeros((108, 192), dtype=np.uint8)
    indices, _ = reframe_v2._sample_indices(bounds, FPS)
    monkeypatch.setattr(reframe_v2, "_visual_samples", lambda *_: [(i, frame) for i in indices])

    def ball_samples(_path, requested):
        assert requested == indices
        assert 31 in requested and 63 in requested
        left = Ball(0.18, 0.5, 0.85)
        right = Ball(0.78, 0.5, 0.85)
        return [(index, [left] if index < 31 else [right] if index < 63 else [])
                for index in requested]

    monkeypatch.setattr(sports_framing, "sample_detections", ball_samples)
    positions = reframe_v2.visual_trajectory("clip.mp4", bounds, strategies, FPS,
                                             SOURCE_WIDTH, 1080, CROP_WIDTH, "sports")

    assert positions[30] < 50
    assert positions[31] > 1150
    assert positions[63:] == [None] * 32
    assert strategies == ["TRACK", "TRACK", "FIT"]


def test_sports_fallback_crops_only_when_motion_is_concentrated():
    def frame_at(left):
        frame = np.zeros((108, 192), dtype=np.uint8)
        frame[18:82, left:left + 22] = 200
        return frame

    moving = [(i * 6, frame_at(left)) for i, left in enumerate([112, 120, 127, 134, 143])]
    still = [(index, moving[0][1]) for index, _ in moving]

    positions = framing.scene_crop(moving, 30, SOURCE_WIDTH, CROP_WIDTH, FPS, "sports")

    assert positions is not None
    assert positions[-1] > positions[0] > 750
    assert framing.scene_crop(still, 30, SOURCE_WIDTH, CROP_WIDTH, FPS, "sports") is None


def test_sports_detector_requires_installed_weights(monkeypatch, tmp_path):
    monkeypatch.setenv("YOLO_MODEL_PATH", str(tmp_path / "missing.pt"))
    load_model = Mock()
    monkeypatch.setitem(sys.modules, "ultralytics", SimpleNamespace(YOLO=load_model))

    assert sports_framing.ball_detector() is None
    assert sports_framing.sample_detections("unused.mp4", [0]) == ()
    load_model.assert_not_called()


def test_sports_detector_uses_the_models_ball_class_and_confidence(monkeypatch, tmp_path):
    weights = tmp_path / "local.pt"
    weights.touch()
    monkeypatch.setenv("YOLO_MODEL_PATH", str(weights))
    frame = np.zeros((200, 400, 3), dtype=np.uint8)
    valid = SimpleNamespace(xyxy=[[220, 90, 260, 110]], conf=[0.83])
    weak = SimpleNamespace(xyxy=[[20, 20, 50, 50]], conf=[0.10])

    class FakeModel:
        names = {0: "person", 7: "sports ball"}

        def __call__(self, image, **options):
            assert image is frame
            assert options["classes"] == [7]
            assert options["conf"] == sports_framing.MIN_CONFIDENCE
            assert options["save"] is False
            return [SimpleNamespace(boxes=[valid, weak])]

    def load_model(path):
        assert path == str(weights)
        return FakeModel()

    monkeypatch.setitem(sys.modules, "ultralytics", SimpleNamespace(YOLO=load_model))

    detector = sports_framing.ball_detector()

    assert detector(frame) == [Ball(0.6, 0.5, 0.83)]
