import numpy as np

import framing


def still(left):
    image = np.zeros((108, 192), dtype=np.uint8)
    image[18:82, left:left + 22] = 200
    return image


def test_detail_frames_a_static_off_center_subject():
    samples = [(n, still(136)) for n in range(0, 32, 8)]
    cropped = framing.scene_crop(samples, 32, 1920, 608, 30, "detail")
    assert len(cropped) == 32
    assert len(set(cropped)) == 1
    assert max(cropped) > 1000


def test_motion_frames_a_moving_subject_and_smooths_the_crop():
    samples = [(n, still(x)) for n, x in zip(range(0, 40, 8), [95, 110, 124, 133, 141])]
    cropped = framing.scene_crop(samples, 40, 1920, 608, 30, "motion")
    assert cropped[0] > 700
    assert cropped[-1] > cropped[0]
    assert max(abs(a - b) for a, b in zip(cropped, cropped[1:])) <= 12
    assert all(0 <= x <= 1312 and x % 2 == 0 for x in cropped)


def test_auto_fits_a_wide_scene_and_rejects_global_brightness_changes():
    striped = np.tile(np.arange(192, dtype=np.uint8) % 2 * 150, (108, 1))
    samples = [(0, striped), (8, striped + 30), (16, striped + 50)]
    assert framing.scene_crop(samples, 24, 1920, 608, 30, "auto") is None
    assert set(framing.scene_crop(samples, 24, 1920, 608, 30, "motion")) == {656}


def test_auto_fits_a_scene_with_two_separated_people():
    group = np.maximum(still(16), still(151))
    samples = [(i, group) for i in [0, 8, 16]]
    assert framing.scene_crop(samples, 24, 1920, 608, 30, "auto") is None


def test_auto_uses_a_confident_visual_crop():
    samples = [(i, still(18)) for i in [0, 8, 16]]
    cropped = framing.scene_crop(samples, 24, 1920, 608, 30, "auto")
    assert cropped[0] == 0


def test_static_modes_keep_their_layout_without_frames():
    assert framing.scene_crop((), 24, 1920, 608, 30, "fit") is None
    assert set(framing.scene_crop((), 24, 1920, 608, 30, "center")) == {656}
