from reframe_v2 import (
    concat_list_content,
    dedupe_sendcmd_lines,
    general_filtergraph,
    scene_frame_ranges,
)


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


def test_general_filtergraph_targets_output_geometry():
    graph = general_filtergraph(608, 1080)
    assert "gblur" in graph
    assert "scale=608:-2" in graph
    assert "overlay=x=(W-w)/2:y=(H-h)/2" in graph
