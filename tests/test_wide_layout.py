"""WIDE without a reading area: a moderate zoom around the action.

A 16:9 screen kept at full width in a 9:16 frame was a 608 px strip (32% of
the height), illegible on a phone (visual review, 17-oct-2026). Automatic WIDE
now fills ~41% of the height, keeps ~77% of the width, and slides the crop
toward the motion; the explicit "full frame" framing keeps the whole width.
"""
import cv2
import numpy as np
import pytest

import gpu_render
import hook_placement
import layout_ranges
import reframe_v2
import screencast_layout
from screencast_layout import (
    action_window_centre,
    content_band,
    crop_content_height,
    focus_filtergraph,
    wide_action_centre,
    wide_crop,
)

OW, OH = 1080, 1920


class TestWideCrop:
    def test_16x9_fills_the_window_between_hook_and_captions(self):
        x, w = wide_crop(1920, 1080, OW, OH)
        fg_h = crop_content_height(1080, OW, OH, w)
        # Was 608 px (32%); now ~41% of the height, ~77% of the width kept.
        assert 0.40 <= fg_h / OH <= 0.42
        assert 0.75 <= w / 1920 <= 0.79
        # Centred by default, each side trimmed alike.
        assert abs(x - (1920 - x - w)) <= 2
        top, bottom = content_band(OH, fg_h)
        assert top >= 0.29 and bottom <= 0.75  # off the bottom captions

    def test_same_geometry_at_any_16x9_resolution(self):
        for src in [(1280, 720), (3840, 2160)]:
            _, w = wide_crop(*src, OW, OH)
            assert abs(w / src[0] - 0.772) < 0.005

    def test_ultrawide_is_capped(self):
        _, w = wide_crop(2560, 1080, OW, OH)
        assert w / 2560 == pytest.approx(1 / screencast_layout.WIDE_MAX_ZOOM, abs=0.005)

    def test_a_source_that_already_fills_the_window_keeps_its_width(self):
        assert wide_crop(1440, 1080, OW, OH) is None   # 4:3
        assert wide_crop(1080, 1080, OW, OH) is None

    def test_centre_moves_the_crop_but_never_out_of_the_frame(self):
        x, w = wide_crop(1920, 1080, OW, OH, centre=0.6)
        assert abs((x + w / 2) / 1920 - 0.6) < 0.01
        x, w = wide_crop(1920, 1080, OW, OH, centre=1.0)
        assert x + w == 1920
        x, w = wide_crop(1920, 1080, OW, OH, centre=0.0)
        assert x == 0

    def test_even(self):
        for c in (0.0, 0.33, 0.5, 0.71, 1.0):
            x, w = wide_crop(1920, 1080, OW, OH, centre=c)
            assert x % 2 == 0 and w % 2 == 0

    def test_old_full_width_wide_is_one_env_away(self, monkeypatch):
        monkeypatch.setattr(screencast_layout, "WIDE_CONTENT_HEIGHT_RATIO", 0.32)
        assert wide_crop(1920, 1080, OW, OH) is None

    def test_graph_stays_on_the_gpu_render_path(self):
        crop = wide_crop(1920, 1080, OW, OH, centre=0.6)
        graph = focus_filtergraph(1920, 1080, OW, OH, crop)
        layers = gpu_render.layers_from_graph(graph, 1920, 1080, OW, OH)
        fg = layers[-1]
        x, w = crop
        assert fg["src"] == [x, 0, w, 1080]
        h = crop_content_height(1080, OW, OH, w)
        # overlay=y=(H-h)/2, aligned to the chroma grid like ffmpeg does.
        assert fg["dst"][0] == 0 and fg["dst"][2:] == [OW, h]
        assert abs(fg["dst"][1] - (OH - h) / 2) <= 1


class TestActionWindow:
    def test_motion_spread_evenly_stays_centred(self):
        assert action_window_centre(np.ones(160), 0.77) is None

    def test_no_motion_stays_centred(self):
        assert action_window_centre(np.zeros(160), 0.77) is None

    def test_action_on_the_right_pulls_the_window_right(self):
        profile = np.ones(160)
        profile[115:150] += 20  # a pileup at 72-94% of the width
        c = action_window_centre(profile, 0.77)
        assert c is not None and c > 0.5
        # The whole pileup is inside the window.
        assert c - 0.385 <= 115 / 160 and 150 / 160 <= c + 0.385 + 1e-9

    def test_takes_the_most_central_of_the_tied_windows(self):
        profile = np.zeros(160)
        profile[0:20] = 1.0  # all the motion in the left edge
        c = action_window_centre(profile, 0.77)
        # Any window starting at 0 holds it all; only the leftmost does.
        assert c == pytest.approx(0.385, abs=0.01)

    def test_a_small_gain_is_not_worth_moving(self):
        profile = np.ones(160)
        profile[150:] += 0.5
        assert action_window_centre(profile, 0.77) is None


def _video(path, n=40, box_x=None, size=(320, 180)):
    w, h = size
    out = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 20, (w, h))
    for i in range(n):
        frame = np.full((h, w, 3), 90, np.uint8)
        cv2.rectangle(frame, (5, 5), (60, 30), (255, 255, 255), -1)  # static HUD
        if box_x is not None:
            x = box_x + int(15 * np.sin(i))
            # Bright, so its grey level is far from the background's on any
            # JPEG decoder (pure red is ~76 grey on 90: under the noise floor).
            cv2.rectangle(frame, (x, 80), (x + 30, 110), (0, 230, 255), -1)
        out.write(frame)
    out.release()
    return str(path)


class TestWideActionCentre:
    def test_moving_cars_on_the_right(self, tmp_path):
        path = _video(tmp_path / "pile.avi", box_x=260)
        c = wide_action_centre(path, 0, 40, 0.77)
        # The centred window (11.5-88.5%) clips the moving box (~76-96%); the
        # crop slides just far enough to hold it, no further.
        assert c is not None and c > 0.5
        assert c + 0.385 >= 292 / 320

    def test_a_still_hud_does_not_pull_it(self, tmp_path):
        path = _video(tmp_path / "still.avi")
        assert wide_action_centre(path, 0, 40, 0.77) is None

    def test_unreadable_video(self, tmp_path):
        assert wide_action_centre(str(tmp_path / "nope.mp4"), 0, 40, 0.77) is None


class TestRenderWiring:
    def test_motion_wins_then_reading_area_then_centre(self, monkeypatch):
        monkeypatch.setattr(screencast_layout, "wide_action_centre", lambda *a: 0.6)
        x, w = reframe_v2._wide_crop("c.mp4", 1920, 1080, OW, OH, 0, 90, (0.0, 0.92))
        assert abs((x + w / 2) / 1920 - 0.6) < 0.01

        monkeypatch.setattr(screencast_layout, "wide_action_centre", lambda *a: None)
        x, w = reframe_v2._wide_crop("c.mp4", 1920, 1080, OW, OH, 0, 90, (0.0, 0.92))
        assert abs((x + w / 2) / 1920 - 0.46) < 0.01
        x, w = reframe_v2._wide_crop("c.mp4", 1920, 1080, OW, OH, 0, 90, None)
        assert abs((x + w / 2) / 1920 - 0.5) < 0.01

    def test_a_failed_probe_still_renders(self, monkeypatch):
        def boom(*a):
            raise RuntimeError("decoder")
        monkeypatch.setattr(screencast_layout, "wide_action_centre", boom)
        assert reframe_v2._wide_crop("c.mp4", 1920, 1080, OW, OH, 0, 90, None)

    def test_4x3_is_not_probed(self, monkeypatch):
        def never(*a):
            raise AssertionError("probed")
        monkeypatch.setattr(screencast_layout, "wide_action_centre", never)
        assert reframe_v2._wide_crop("c.mp4", 1440, 1080, OW, OH, 0, 90, None) is None

    def test_automatic_wide_zooms_and_forced_full_frame_does_not(self):
        import inspect
        src = inspect.getsource(reframe_v2.render)
        # The automatic route asks for the zoomed crop...
        assert "_wide_crop(input_video" in src
        # ...while a WIDE scene without one (the user's "full frame") keeps
        # the whole width.
        assert "full_width_content_height(orig_w, orig_h, out_w)" in src


class TestContentBandForTheHook:
    def test_sidecar_keeps_the_band_through_a_recut(self, tmp_path):
        clip = str(tmp_path / "c.mp4")
        layout_ranges.write(clip, [(0, 4, "TRACK"), (4, 10, "WIDE", (0.295, 0.705))])
        got = layout_ranges.read(clip)
        assert got[0] == {"start": 0.0, "end": 4.0, "layout": "track"}
        assert got[1]["content"] == [0.295, 0.705]
        moved = layout_ranges.remap(got, [{"start": 5, "end": 8}])
        assert moved == [{"start": 0.0, "end": 3.0, "layout": "wide",
                          "content": [0.295, 0.705]}]

    def test_bad_bands_are_dropped(self):
        got = layout_ranges.normalise([
            {"start": 0, "end": 1, "layout": "wide", "content": [0.7, 0.3]},
            {"start": 1, "end": 2, "layout": "wide", "content": "x"},
            (2, 3, "wide", (0.2, 1.4)),
        ])
        assert all("content" not in r for r in got) and len(got) == 3

    def test_hook_goes_in_the_blur_above_the_picture(self):
        ranges = layout_ranges.normalise([(0, 10, "WIDE", (0.295, 0.705))])
        pictures = hook_placement.content_bands(ranges, 0.0, 5.0)
        assert pictures == [(0.295, 0.705)]
        bands = sorted(set(hook_placement.caption_bands(ranges, 0.0, 5.0)) | set(pictures))
        # A two-line pill hook (~12% of the height) moves up from 20%.
        y, _ = hook_placement.choose_y(0.8, 0.123, [[]], bands)
        assert y is not None and y + 0.123 <= 0.295 + 1e-6
        # A short hook already above the picture stays at the old spot.
        y, _ = hook_placement.choose_y(0.8, 0.08, [[]], bands)
        assert y == pytest.approx(hook_placement.LEGACY_TOP)

    def test_pictures_outside_the_window_do_not_count(self):
        ranges = [{"start": 6.0, "end": 9.0, "layout": "wide", "content": [0.3, 0.7]}]
        assert hook_placement.content_bands(ranges, 0.0, 5.0) == []
