"""SCREENCAST_LAYOUT=1 must reach the renderer.

Until 6-oct-2026 the flag was set (by ``layouts=["screencast"]`` or by the
layout picker) and nothing ever asked which shots showed a screen:
reframe_v2.render was never given content ranges, so screen tutorials came out
GENERAL/TRACK only, even when the user forced the layout (prod jobs 204da77a
and 62f72a44). These tests pin the call and the fallback.
"""
import sys
import types

import pytest

import reframe_v2
import screencast_layout
from screencast_layout import (
    _parse_shots,
    fallback_ranges,
    focus_crop,
    focus_filtergraph,
    overlapping_range,
    presenter_cam,
    ranges_from_verdicts,
    shots_to_ask,
)


class FT:
    """Stand-in for scenedetect's FrameTimecode."""

    def __init__(self, frames):
        self.frames = frames

    def get_frames(self):
        return self.frames


def scenes_of(*bounds):
    return [(FT(s), FT(e)) for s, e in bounds]


class _Stop(Exception):
    pass


@pytest.fixture
def fake_main(monkeypatch):
    """A `main` with just what render() touches before routing screens."""
    m = types.ModuleType("main")
    m.detect_scenes = lambda path: (scenes_of((0, 90), (90, 300)), 30.0)
    m.get_video_resolution = lambda path: (1920, 1080)
    m.analyze_scenes_strategy = lambda path, scenes: ['TRACK', 'GENERAL']
    monkeypatch.setitem(sys.modules, "main", m)
    return m


def _routed_ranges(monkeypatch):
    """Run render() up to the screencast routing and return the ranges it got."""
    seen = {}

    def capture(video, scenes, strategies, ranges):
        seen["ranges"] = ranges
        raise _Stop()

    monkeypatch.setattr(screencast_layout, "detect_screencast_scenes", capture)
    monkeypatch.setattr(reframe_v2.camera_inset, "detect", lambda path: None)
    try:
        reframe_v2.render("clip.mp4", "out.mp4", 9 / 16)
    except _Stop:
        pass
    return seen.get("ranges")


class TestRenderAsksForScreens:
    def test_enabled_layout_routes_the_detected_screens(self, monkeypatch, fake_main):
        monkeypatch.setattr(screencast_layout, "ENABLED", True)
        detected = [(3.0, 10.0, "screen", 1.0, (0.2, 0.7), False)]
        monkeypatch.setattr(screencast_layout, "detect_content_ranges",
                            lambda video, scenes, fps: detected)
        assert _routed_ranges(monkeypatch) == detected

    def test_failed_check_treats_faceless_scenes_as_the_screen(self, monkeypatch, fake_main):
        # An explicit layouts=["screencast"] must not silently render GENERAL
        # because Gemini was unreachable.
        monkeypatch.setattr(screencast_layout, "ENABLED", True)
        monkeypatch.setattr(screencast_layout, "detect_content_ranges",
                            lambda video, scenes, fps: None)
        assert _routed_ranges(monkeypatch) == [(3.0, 10.0, "screen", 1.0, None, False)]

    def test_disabled_layout_asks_nothing(self, monkeypatch, fake_main):
        monkeypatch.setattr(screencast_layout, "ENABLED", False)

        def boom(*a, **k):
            raise AssertionError("asked for screens with the layout off")

        monkeypatch.setattr(screencast_layout, "detect_content_ranges", boom)
        # No ranges -> render goes on to the trajectory pass, which needs a
        # real file; reaching it without calling the detector is the point.
        with pytest.raises(Exception) as err:
            reframe_v2.render("clip.mp4", "out.mp4", 9 / 16)
        assert "asked for screens" not in str(err.value)

    def test_forced_framing_skips_the_check(self, monkeypatch, fake_main):
        monkeypatch.setattr(screencast_layout, "ENABLED", True)

        def boom(*a, **k):
            raise AssertionError("asked for screens over a forced framing")

        monkeypatch.setattr(screencast_layout, "detect_content_ranges", boom)
        with pytest.raises(Exception) as err:
            reframe_v2.render("clip.mp4", "out.mp4", 9 / 16, force_strategy="TRACK")
        assert "asked for screens" not in str(err.value)


class TestDetectContentRanges:
    def test_disabled_module_detects_nothing(self, monkeypatch):
        monkeypatch.setattr(screencast_layout, "ENABLED", False)
        assert screencast_layout.detect_content_ranges(
            "x.mp4", scenes_of((0, 30)), 30.0) == []

    def test_no_key_is_a_failure_not_an_empty_answer(self, monkeypatch):
        monkeypatch.setattr(screencast_layout, "ENABLED", True)
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        assert screencast_layout.detect_content_ranges(
            "x.mp4", scenes_of((0, 30)), 30.0) is None


class TestShotVerdicts:
    def test_parse_keeps_known_kinds_and_clamps_focus(self):
        raw = [{"shot": 0, "kind": "Screen", "focus_left": -0.2, "focus_right": 0.6},
               {"shot": 1, "kind": "camera", "focus_left": 0, "focus_right": 1},
               {"shot": 2, "kind": "beside", "focus_left": 0.4, "focus_right": 0.42},
               {"shot": 7, "kind": "screen", "focus_left": 0, "focus_right": 1},
               {"shot": 3, "kind": "slideshow", "focus_left": 0, "focus_right": 1},
               {"shot": "x"}]
        assert _parse_shots(raw, 4) == {
            0: ("screen", (0.0, 0.6), False),
            1: ("camera", (0.0, 1.0), False),
            2: ("beside", None, False),       # a 2%-wide focus is noise
        }

    def test_presenter_cam_counts_only_on_a_screen_and_only_when_true(self):
        raw = [{"shot": 0, "kind": "screen", "focus_left": 0, "focus_right": 1,
                "presenter_cam": True},
               {"shot": 1, "kind": "camera", "focus_left": 0, "focus_right": 1,
                "presenter_cam": True},
               {"shot": 2, "kind": "beside", "focus_left": 0, "focus_right": 1,
                "presenter_cam": True},
               {"shot": 3, "kind": "screen", "focus_left": 0, "focus_right": 1,
                "presenter_cam": "true"},
               {"shot": 4, "kind": "screen", "focus_left": 0, "focus_right": 1}]
        cams = {i: v[2] for i, v in _parse_shots(raw, 5).items()}
        assert cams == {0: True, 1: False, 2: False, 3: False, 4: False}

    def test_screens_route_wide_and_cameras_do_not_move(self):
        scenes = scenes_of((0, 60), (60, 120), (120, 180))
        verdicts = {0: ("camera", None, False), 1: ("screen", (0.2, 0.7), True),
                    2: ("beside", None, False)}
        assert ranges_from_verdicts(scenes, 30.0, verdicts) == [
            (2.0, 4.0, "screen", 1.0, (0.2, 0.7), True),
            (4.0, 6.0, "beside", 0.7, None, False),
        ]

    def test_unasked_scene_borrows_the_nearest_answer(self):
        scenes = scenes_of((0, 30), (30, 33), (33, 36), (36, 90))
        verdicts = {0: ("camera", None, False), 3: ("screen", None, False)}
        ranges = ranges_from_verdicts(scenes, 30.0, verdicts)
        # Scene 1 sits next to the camera shot, scene 2 next to the screen.
        assert [r[:3] for r in ranges] == [(1.1, 1.2, "screen"), (1.2, 3.0, "screen")]

    def test_too_many_shots_asks_the_longest(self):
        scenes = scenes_of((0, 10), (10, 100), (100, 105), (105, 300))
        assert shots_to_ask(scenes, limit=2) == [1, 3]
        assert shots_to_ask(scenes, limit=10) == [0, 1, 2, 3]

    def test_fallback_only_takes_faceless_scenes(self):
        scenes = scenes_of((0, 30), (30, 90))
        assert fallback_ranges(scenes, ['TRACK', 'GENERAL'], 30.0) == [
            (1.0, 3.0, "screen", 1.0, None, False)]

    def test_overlapping_range_carries_the_focus(self):
        ranges = [(0.0, 5.0, "screen", 1.0, (0.1, 0.5))]
        assert overlapping_range(1.0, 4.0, ranges) == (1.0, (0.1, 0.5))
        assert overlapping_range(6.0, 9.0, ranges) == (0.0, None)

    def test_presenter_cam_needs_the_flag_on_an_overlapping_range(self):
        ranges = [(0.0, 5.0, "screen", 1.0, None, True),
                  (5.0, 9.0, "screen", 1.0, None, False)]
        assert presenter_cam(1.0, 4.0, ranges)
        assert not presenter_cam(5.0, 9.0, ranges)
        assert not presenter_cam(9.5, 12.0, ranges)
        # Ranges without the flag (pinned by a caller, old shape) never ask.
        assert not presenter_cam(1.0, 4.0, [(0.0, 5.0, "screen", 1.0, None)])


class TestFocusCrop:
    def test_no_focus_or_full_width_shows_the_whole_screen(self):
        assert focus_crop(1920, 1080, 1080, 1920, None) is None
        assert focus_crop(1920, 1080, 1080, 1920, (0.0, 1.0)) is None
        assert focus_crop(1920, 1080, 1080, 1920, (0.05, 0.9)) is None

    def test_reading_area_is_cropped_and_centred(self):
        x, w = focus_crop(1920, 1080, 1080, 1920, (0.25, 0.75))
        # 50% + padding, centred on the page column.
        assert abs((x + w / 2) - 960) <= 2
        assert 0.5 * 1920 <= w < 0.9 * 1920

    def test_tight_focus_never_zooms_past_the_floor(self):
        x, w = focus_crop(1920, 1080, 1080, 1920, (0.45, 0.5))
        # The screen may fill at most FOCUS_MAX_HEIGHT_RATIO of the frame.
        assert 1080 * 1080 / w <= 1920 * screencast_layout.FOCUS_MAX_HEIGHT_RATIO + 2

    def test_crop_stays_inside_the_source_and_even(self):
        for focus in [(0.0, 0.3), (0.7, 1.0), (0.3, 0.6)]:
            x, w = focus_crop(1920, 1080, 1080, 1920, focus)
            assert 0 <= x and x + w <= 1920
            assert x % 2 == 0 and w % 2 == 0

    def test_filtergraph_keeps_full_height_and_fills_the_width(self):
        graph = focus_filtergraph(1920, 1080, 1080, 1920, (480, 1000))
        assert "crop=w=1000:h=1080:x=480:y=0" in graph
        assert "scale=1080:1166" in graph
        assert graph.endswith("[v]")


class TestInsetPerScene:
    def test_detection_must_sit_in_the_inset_box(self):
        from camera_inset import _centre_inside
        box = (0, 500, 220, 220)
        assert _centre_inside((40, 540, 120, 120), box, 22)      # face in the bubble
        assert not _centre_inside((900, 300, 200, 200), box, 22)  # someone elsewhere

    def test_stable_box_needs_agreement(self):
        from camera_inset import stable_box
        still = [(100, 500, 300, 200)] * 3 + [(102, 498, 300, 200)]
        assert stable_box(still, 1920, 1080) is not None
        moving = [(100, 500, 300, 200), (700, 300, 300, 200), (1400, 100, 300, 200)]
        assert stable_box(moving, 1920, 1080) is None

    def test_overlay_box_is_centred_on_the_face(self):
        from camera_inset import overlay_box
        x, y, w, h = overlay_box((300, 400, 100, 100), 1920, 1080)
        assert abs((x + w / 2) - 350) <= 1
        assert y < 450 < y + h
        assert abs(w / h - 16 / 9) < 0.02
        # Clamped inside the frame at an edge.
        x, y, w, h = overlay_box((0, 0, 100, 100), 1920, 1080)
        assert x == 0 and y == 0


@pytest.mark.parametrize("evidence", ["flag", "geometry", "gameplay"])
def test_embedded_webcam_and_gameplay_keep_one_foreground(monkeypatch, fake_main, tmp_path, evidence):
    monkeypatch.setattr(screencast_layout, "ENABLED", True)
    kind = "gameplay" if evidence == "gameplay" else "screen"
    monkeypatch.setattr(screencast_layout, "detect_content_ranges",
                        lambda *args: [(3.0, 10.0, kind, 1.0, None, evidence == "flag")])
    monkeypatch.setattr(screencast_layout, "detect_screencast_scenes",
                        lambda *args: {1: ("WIDE", None)})
    box = (0, 0, 250, 160) if evidence == "geometry" else None
    monkeypatch.setattr(reframe_v2.camera_inset, "detect", lambda *args: box)
    monkeypatch.setattr(reframe_v2.camera_inset, "present_in_scene",
                        lambda path, box, start, end: start == 90)
    fake_main.SmoothedCameraman = lambda *args, **kwargs: types.SimpleNamespace(
        crop_width=608, crop_height=1080)
    fake_main.SpeakerTracker = lambda **kwargs: None
    monkeypatch.setattr(reframe_v2, "_analyze_trajectory", lambda *args: [0] * 300)
    monkeypatch.setattr(reframe_v2.split_layout, "detect_split_scenes", lambda *args: {})
    monkeypatch.setattr(reframe_v2.punch_in, "ENABLED", False)

    def check(_video, specs, *args):
        assert len(specs) == 2
        screen = specs[1][2]
        assert "split=2" in screen
        assert screen.count("overlay=") == 1
        assert "[cam]" not in screen
        # Gameplay Priority: the game zoomed past the 608 px strip (x1.25,
        # composition.gameplay_zoom), still one foreground with no second webcam.
        assert "crop=w=1536:h=1080" in screen and "scale=1080:758" in screen

    monkeypatch.setattr(reframe_v2, "encode_scenes", check)
    output = str(tmp_path / "out.mp4")
    assert reframe_v2.render("clip.mp4", output, 9 / 16)
    assert reframe_v2.layout_ranges.read(output)[1]["layout"] == "gameplay"
