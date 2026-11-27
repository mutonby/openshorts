from reframe_v2 import (
    DELIVERY_MIN_WIDTH,
    concat_list_content,
    dedupe_sendcmd_lines,
    delivery_size,
    general_filtergraph,
    scene_frame_ranges,
    source_already_fits,
)

VERTICAL = 9 / 16


def test_sendcmd_lines_dedupe_to_change_points():
    xs = [100, 100, 100, 104, 104, 110]
    lines = dedupe_sendcmd_lines(xs, fps=30.0)
    assert lines == [
        "0.0000 crop@c x 100;",
        "0.1000 crop@c x 104;",
        "0.1667 crop@c x 110;",
    ]


def test_sendcmd_static_camera_is_single_command():
    assert len(dedupe_sendcmd_lines([250] * 300, fps=30.0)) == 1


def test_scene_ranges_clamped_and_keep_strategy():
    ranges = scene_frame_ranges(
        [(0, 100), (100, 250), (250, 400)],
        ['TRACK', 'GENERAL', 'TRACK'],
        total_frames=200,
    )
    # Third scene starts past the decoded frame count -> dropped, and the
    # GENERAL strategy stays attached to its own range.
    assert ranges == [(0, 100, 'TRACK'), (100, 200, 'GENERAL')]


def test_scene_ranges_missing_strategy_defaults_to_track():
    assert scene_frame_ranges([(0, 10)], [], 10) == [(0, 10, 'TRACK')]


def test_concat_list_format():
    content = concat_list_content(["/tmp/a/seg_000.mp4", "/tmp/a/seg_001.mp4"])
    assert content == "file '/tmp/a/seg_000.mp4'\nfile '/tmp/a/seg_001.mp4'\n"


def test_visual_render_uses_no_face_detectors_and_keeps_scene_overrides(monkeypatch, tmp_path):
    import sys
    from types import SimpleNamespace

    import numpy as np
    import reframe_v2

    frame = np.zeros((108, 192), dtype=np.uint8)
    frame[15:80, 145:170] = 200
    other = np.fliplr(frame)
    timecode = lambda n: SimpleNamespace(get_frames=lambda: n)
    camera = lambda *_a, **_kw: SimpleNamespace(crop_width=608, crop_height=1080)
    main = SimpleNamespace(
        get_video_resolution=lambda _: (1920, 1080),
        detect_scenes=lambda _: ([(timecode(0), timecode(30)),
                                  (timecode(30), timecode(60))], 30),
        SmoothedCameraman=camera,
    )

    def no_detector(*_args, **_kwargs):
        raise AssertionError("a face model was called")

    monkeypatch.setitem(sys.modules, "main", main)
    monkeypatch.setattr(reframe_v2, "_analyze_trajectory", no_detector)
    monkeypatch.setattr(reframe_v2.split_layout, "detect_split_scenes", no_detector)
    monkeypatch.setattr(reframe_v2, "_visual_samples",
                        lambda *_: [(0, frame), (15, frame), (30, other), (45, other)])

    def verify(_path, specs, *_):
        assert len(specs) == 2
        assert specs[0][3][0][2] > 1100
        assert {box[2] for box in specs[1][3]} == {656}

    monkeypatch.setattr(reframe_v2, "encode_scenes", verify)
    assert reframe_v2.render("clip", str(tmp_path / "result.mp4"), VERTICAL,
                             framing_method="auto", crop_overrides={1: 0.5})


def test_gameplay_renders_the_selected_source_interval_with_audio_in_one_pass(monkeypatch, tmp_path):
    import sys
    from types import SimpleNamespace

    import reframe_v2

    monkeypatch.setitem(sys.modules, "main", SimpleNamespace(get_video_resolution=lambda _: (1920, 1080)))
    monkeypatch.setattr(reframe_v2, "cut_rate_filter", lambda _: [])
    calls = []

    def encode(source, start, length, extra, graph, label, path, with_audio):
        assert graph.count("overlay=") == 1
        # Zoomed past the 608 px full-width strip (composition.gameplay_zoom).
        assert "scale=1080:758" in graph and "[cam]" not in graph
        calls.append((source, start, length, extra, label, path, with_audio))

    monkeypatch.setattr(reframe_v2, "_run_segment", encode)
    import facecam
    monkeypatch.setattr(facecam, "detect", lambda *a, **k: None)
    path = str(tmp_path / "clip.mp4")
    assert reframe_v2.render_gameplay_range("source.mp4", path, VERTICAL, 620.5, 645.5)
    assert calls == [("source.mp4", 620.5, 25.0, [], "[v]", path, True)]
    assert reframe_v2.layout_ranges.read(path) == [
        {"start": 0.0, "end": 25.0, "layout": "gameplay", "content": [0.303, 0.697]},
    ]


class TestDeliverySize:
    """A 720p source must not produce a 406x720 clip (prod audit, 25-jul-2026)."""

    def test_720p_source_is_upscaled_to_the_delivery_floor(self):
        assert delivery_size(1280, 720, VERTICAL) == (1080, 1920)

    def test_1080p_source_lands_on_the_floor_exactly(self):
        assert delivery_size(1920, 1080, VERTICAL) == (1080, 1920)

    def test_tiny_source_still_reaches_the_floor(self):
        # 360x640 sources were shipping as-is; platforms treat that as junk.
        assert delivery_size(640, 360, VERTICAL) == (1080, 1920)

    def test_high_res_source_is_never_downscaled(self):
        # 4K source: the native crop is already well past the floor, so keep it.
        assert delivery_size(3840, 2160, VERTICAL) == (1216, 2160)

    def test_already_vertical_source_keeps_its_full_height(self):
        assert delivery_size(1080, 1920, VERTICAL) == (1080, 1920)

    def test_square_output_uses_the_same_floor(self):
        assert delivery_size(1280, 720, 1.0) == (1080, 1080)

    def test_dimensions_are_always_even(self):
        for w, h in [(1280, 720), (1920, 1080), (854, 480), (3840, 2160), (641, 361)]:
            out_w, out_h = delivery_size(w, h, VERTICAL)
            assert out_w % 2 == 0 and out_h % 2 == 0, (w, h)

    def test_never_returns_below_the_floor(self):
        for w, h in [(320, 180), (640, 360), (1280, 720), (1920, 1080)]:
            assert delivery_size(w, h, VERTICAL)[0] >= DELIVERY_MIN_WIDTH

    def test_aspect_ratio_is_preserved(self):
        for w, h in [(1280, 720), (640, 360), (3840, 2160)]:
            out_w, out_h = delivery_size(w, h, VERTICAL)
            assert abs(out_w / out_h - VERTICAL) < 0.01, (w, h)


def test_general_filtergraph_targets_output_geometry():
    graph = general_filtergraph(1080, 1920)
    assert "gblur" in graph
    assert "overlay=x=(W-w)/2:y=(H-h)/2" in graph


class TestGeneralLayout:
    """GENERAL used to leave the content at 32% of the frame, floating in blur."""

    def test_content_is_scaled_by_height_not_width(self):
        from reframe_v2 import GENERAL_CONTENT_HEIGHT_RATIO
        graph = general_filtergraph(1080, 1920)
        expected = int(1920 * GENERAL_CONTENT_HEIGHT_RATIO)
        expected += expected % 2
        assert f"scale=-2:{expected}" in graph

    def test_content_height_is_even(self):
        # Odd dimensions are rejected by the encoder. Read the FOREGROUND
        # scale — the background line also contains a scale=-2: term.
        for out_h in (1920, 1080, 1078, 976):
            graph = general_filtergraph(1080, out_h)
            fg = graph.split("[fga]scale=-2:")[1]
            h = int(fg.split(",")[0])
            assert h % 2 == 0, (out_h, h)

    def test_overflow_is_trimmed_never_padded(self):
        # min() keeps it a no-op for sources narrower than the frame.
        graph = general_filtergraph(1080, 1920)
        assert "crop=w=min(iw\\,1080):h=ih" in graph

    def test_content_fills_more_than_the_old_full_width_fit(self):
        # A 16:9 source fitted to full width was 9/16 of it — 0.5625 * 1080
        # = 608px, i.e. 32% of a 1920 frame. The new layout must beat that.
        from reframe_v2 import GENERAL_CONTENT_HEIGHT_RATIO
        old_ratio = (1080 * 9 / 16) / 1920
        assert GENERAL_CONTENT_HEIGHT_RATIO > old_ratio

    def test_keeps_most_of_the_source_width(self):
        # GENERAL is for groups and landscapes: cropping the sides too hard
        # cuts people out of frame, which is the whole thing it exists to avoid.
        from reframe_v2 import GENERAL_CONTENT_HEIGHT_RATIO
        scaled_w = 1920 * GENERAL_CONTENT_HEIGHT_RATIO * (16 / 9)
        assert 1080 / scaled_w > 0.70, "keeps less than 70% of the source width"


class TestVerticalSource:
    """An already-vertical upload used to come back shrunk into a blurred bed.

    The scene classifier sends face-less shots (a slide, a screen recording) to
    GENERAL, and GENERAL scaled the content to 42% of the frame height — on a
    9:16 source that is 453px of 1080, floating over a blurred copy of itself.
    """

    def test_vertical_source_needs_no_reframe(self):
        assert source_already_fits(1080, 1920, VERTICAL)
        assert source_already_fits(720, 1280, VERTICAL)
        assert source_already_fits(1080, 2400, VERTICAL)  # taller than 9:16

    def test_landscape_and_squarish_sources_still_reframe(self):
        assert not source_already_fits(1920, 1080, VERTICAL)
        assert not source_already_fits(1080, 1080, VERTICAL)
        assert not source_already_fits(1080, 1350, VERTICAL)  # 4:5 has width to cut

    def test_general_never_renders_the_source_narrower_than_the_frame(self):
        # 1080x1920 in a 1080x1920 frame: the foreground must fill it, not sit
        # at 0.42 * 1920 = 806 (453px wide).
        graph = general_filtergraph(1080, 1920, orig_w=1080, orig_h=1920)
        assert "[fga]scale=-2:1920," in graph

    def test_portrait_source_fills_the_width(self):
        # 4:5 source: full width is reached at 1350, well above the 806 default.
        graph = general_filtergraph(1080, 1920, orig_w=1080, orig_h=1350)
        assert "[fga]scale=-2:1350," in graph

    def test_landscape_keeps_the_presence_ratio(self):
        # The floor must not disturb the case the ratio was tuned for: a 16:9
        # source fills the width at 608, below the 806 the ratio asks for.
        from reframe_v2 import GENERAL_CONTENT_HEIGHT_RATIO
        expected = int(1920 * GENERAL_CONTENT_HEIGHT_RATIO)
        expected += expected % 2
        graph = general_filtergraph(1080, 1920, orig_w=1920, orig_h=1080)
        assert f"[fga]scale=-2:{expected}," in graph

    def test_floor_keeps_the_height_even(self):
        for orig_h in (1919, 1351, 1233):
            graph = general_filtergraph(1080, 1920, orig_w=1080, orig_h=orig_h)
            h = int(graph.split("[fga]scale=-2:")[1].split(",")[0])
            assert h % 2 == 0, (orig_h, h)
