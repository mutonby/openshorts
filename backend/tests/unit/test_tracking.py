"""
Characterization tests for SmoothedCameraman and SpeakerTracker.

These classes live in main.py today and will move to
openshorts/video/tracking.py in Phase 1. The tests target the behaviors
that must be preserved across the move:

- SmoothedCameraman: initial centering, safe-zone (no move when target
  is inside it), slow vs fast pan speeds, force_snap, clamping to video
  bounds.
- SpeakerTracker: no candidates → None, single candidate locks in,
  switch cooldown prevents rapid re-targeting.
"""
import pytest

from app.video.tracking import SmoothedCameraman, SpeakerTracker


# --- SmoothedCameraman ---------------------------------------------------


@pytest.fixture
def cameraman():
    # 9:16 output (1080x1920) cropped out of a 1920x1080 source.
    # crop_width = video_height * (9/16) = 1080 * 0.5625 = 607
    return SmoothedCameraman(
        output_width=1080, output_height=1920,
        video_width=1920, video_height=1080,
    )


def test_initial_center_is_video_midpoint(cameraman):
    assert cameraman.current_center_x == 960  # 1920 / 2
    assert cameraman.target_center_x == 960


def test_crop_dimensions_for_landscape_source(cameraman):
    # crop_height = video_height; crop_width = video_height * 9/16
    assert cameraman.crop_height == 1080
    assert cameraman.crop_width == int(1080 * (9 / 16))


def test_safe_zone_radius_is_quarter_of_crop_width(cameraman):
    assert cameraman.safe_zone_radius == cameraman.crop_width * 0.25


def test_update_target_with_face_box_sets_center(cameraman):
    cameraman.update_target((400, 100, 200, 200))  # x, y, w, h
    assert cameraman.target_center_x == 500  # 400 + 200/2


def test_update_target_with_none_leaves_target_unchanged(cameraman):
    cameraman.target_center_x = 750
    cameraman.update_target(None)
    assert cameraman.target_center_x == 750


def test_force_snap_jumps_directly_to_target(cameraman):
    cameraman.target_center_x = 1500
    cameraman.get_crop_box(force_snap=True)
    assert cameraman.current_center_x == 1500


def test_camera_does_not_move_when_target_within_safe_zone(cameraman):
    # Target 50 px away — well inside the ~152 px safe zone.
    cameraman.target_center_x = 960 + 50
    cameraman.get_crop_box(force_snap=False)
    assert cameraman.current_center_x == 960  # unchanged


def test_camera_moves_slowly_outside_safe_zone(cameraman):
    # 200 px away — outside safe zone (~152) but not "huge" (>303 = crop_width/2).
    cameraman.target_center_x = 960 + 200
    cameraman.get_crop_box(force_snap=False)
    # Slow pan = 3 px/frame
    assert cameraman.current_center_x == pytest.approx(963.0)


def test_camera_moves_fast_when_distance_is_huge(cameraman):
    # 600 px away — > crop_width/2 (303). Speed = 15 px/frame.
    cameraman.target_center_x = 960 + 600
    cameraman.get_crop_box(force_snap=False)
    assert cameraman.current_center_x == pytest.approx(975.0)


def test_get_crop_box_returns_a_9_16_window(cameraman):
    x1, y1, x2, y2 = cameraman.get_crop_box(force_snap=True)
    # y span is full video height
    assert (y1, y2) == (0, 1080)
    # x span equals crop_width
    assert (x2 - x1) == cameraman.crop_width


def test_camera_clamps_to_left_edge(cameraman):
    cameraman.target_center_x = 0
    cameraman.get_crop_box(force_snap=True)
    x1, _, x2, _ = cameraman.get_crop_box(force_snap=False)
    assert x1 == 0
    assert x2 == cameraman.crop_width


def test_camera_clamps_to_right_edge(cameraman):
    cameraman.target_center_x = 5000
    cameraman.get_crop_box(force_snap=True)
    x1, _, x2, _ = cameraman.get_crop_box(force_snap=False)
    assert x2 == cameraman.video_width


# --- SpeakerTracker -----------------------------------------------------


@pytest.fixture
def tracker():
    return SpeakerTracker(stabilization_frames=15, cooldown_frames=30)


def _face(x, y, w, h, score=10000):
    return {"box": [x, y, w, h], "score": score}


def test_no_candidates_returns_none(tracker):
    assert tracker.get_target([], frame_number=0, width=1920) is None


def test_single_candidate_returns_its_box(tracker):
    box = [100, 100, 200, 200]
    out = tracker.get_target([_face(*box)], frame_number=0, width=1920)
    assert out == box
    assert tracker.active_speaker_id == 0


def test_same_face_in_subsequent_frames_keeps_active_speaker(tracker):
    box = [100, 100, 200, 200]
    tracker.get_target([_face(*box)], frame_number=0, width=1920)
    speaker_a = tracker.active_speaker_id
    tracker.get_target([_face(*box)], frame_number=5, width=1920)
    tracker.get_target([_face(*box)], frame_number=10, width=1920)
    assert tracker.active_speaker_id == speaker_a


def test_new_dominant_speaker_does_not_switch_within_cooldown(tracker):
    # Frame 0: speaker A enters.
    tracker.get_target([_face(100, 100, 200, 200, score=10000)],
                       frame_number=0, width=1920)
    a_id = tracker.active_speaker_id
    # Frame 10 (well inside cooldown=30): BOTH A and a much bigger
    # speaker B are visible. The cooldown only protects against switching
    # when the previous speaker is still on screen — if A disappeared,
    # the tracker would correctly switch since there's no point holding
    # an absent speaker. So we keep A on screen here.
    tracker.get_target(
        [
            _face(100, 100, 200, 200, score=10000),         # A still there
            _face(1500, 100, 400, 400, score=10_000_000),    # B much bigger
        ],
        frame_number=10, width=1920,
    )
    # Hysteresis + cooldown blocks the switch while A is still visible.
    assert tracker.active_speaker_id == a_id


def test_speaker_switches_after_cooldown_when_a_new_face_dominates(tracker):
    # Frame 0: speaker A.
    tracker.get_target([_face(100, 100, 200, 200, score=10000)],
                       frame_number=0, width=1920)
    a_id = tracker.active_speaker_id
    # Frame 100 (well past cooldown=30): speaker B is the only face on
    # screen. With no competing scores and no hysteresis bonus applicable
    # (A is not present), B should take over.
    out = tracker.get_target(
        [_face(1500, 100, 400, 400, score=10_000_000)],
        frame_number=100, width=1920,
    )
    assert tracker.active_speaker_id != a_id
    assert out == [1500, 100, 400, 400]
