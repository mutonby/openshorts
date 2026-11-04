"""osrender draws every layout the way the ffmpeg path does (needs a GPU).

Each case renders the same source twice: through the scene's filtergraph in
ffmpeg (what reframe_v2 runs without osrender) and through gpu_render/osrender.
The two must match frame for frame (count) and picture for picture (SSIM).
Skipped unless the osrender binary and a working NVENC are present.
OSRENDER_TEST_VIDEO points it at a real clip; a synthetic one is used otherwise.
"""
import os
import re
import subprocess

import pytest

import gpu_render
from ffmpeg_utils import nvenc_available, _NVENC_ARGS, QUALITY_FAST

pytestmark = pytest.mark.skipif(
    not (os.path.isfile(gpu_render.BIN) and nvenc_available()),
    reason="needs the osrender binary and NVENC")

W, H, OW, OH, FPS, DUR = 1920, 1080, 1080, 1920, 30, 4.0
MIN_SSIM = 0.97


@pytest.fixture(scope="module")
def source(tmp_path_factory):
    real = os.environ.get("OSRENDER_TEST_VIDEO")
    out = str(tmp_path_factory.mktemp("src") / "src.mp4")
    inp = ["-ss", "0", "-i", real] if real else [
        "-f", "lavfi", "-i", f"testsrc2=size={W}x{H}:rate={FPS}"]
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", *inp, "-t", str(DUR),
                    "-vf", f"scale={W}:{H},fps={FPS},format=yuv420p",
                    "-c:v", "libx264", "-crf", "16", "-an", out], check=True)
    return out


def _ffmpeg(src, graph, out):
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", src,
                    "-filter_complex", graph, "-map", "[v]",
                    *_NVENC_ARGS[QUALITY_FAST], "-an", out], check=True)


def _frames(path):
    return int(subprocess.run(
        ["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v",
         "-show_entries", "stream=nb_read_frames", "-of", "csv=p=0", path],
        capture_output=True, text=True, check=True).stdout.strip())


def _ssim(a, b):
    err = subprocess.run(["ffmpeg", "-hide_banner", "-i", a, "-i", b,
                          "-lavfi", "ssim", "-f", "null", "-"],
                         capture_output=True, text=True).stderr
    return float(re.search(r"All:([0-9.]+)", err).group(1))


def _graphs():
    import reframe_v2
    import screencast_layout
    import split_layout
    import camera_inset
    return {
        "general": reframe_v2.general_filtergraph(OW, OH, orig_w=W, orig_h=H),
        "wide": reframe_v2.general_filtergraph(
            OW, OH, reframe_v2.full_width_content_height(W, H, OW)),
        "focus": screencast_layout.focus_filtergraph(W, H, OW, OH, (300, 1200)),
        "wide_zoom": screencast_layout.focus_filtergraph(
            W, H, OW, OH, screencast_layout.wide_crop(W, H, OW, OH, centre=0.6)),
        "screencast": screencast_layout.screencast_filtergraph(
            W, H, OW, OH, (1500, 400), (0, 1250)),
        "split": split_layout.split_filtergraph(W, H, OW, OH, (500, 450), (1400, 500)),
        "inset": camera_inset.inset_filtergraph(W, H, OW, OH, (1500, 760, 380, 280)),
    }


@pytest.mark.parametrize("name", ["general", "wide", "wide_zoom", "focus", "screencast", "split", "inset"])
def test_layout_matches_ffmpeg(source, tmp_path, name):
    graph = _graphs()[name]
    ref, got = str(tmp_path / "ref.mp4"), str(tmp_path / "got.mp4")
    _ffmpeg(source, graph, ref)
    assert gpu_render.render(source, [(0.0, DUR, graph, None)], W, H, OW, OH, FPS, got)
    assert _frames(got) == _frames(ref)
    assert _ssim(got, ref) >= MIN_SSIM


def _track_graph(tmp_path, lines, first):
    cmd = tmp_path / "cmd.txt"
    cmd.write_text("\n".join(lines) + "\n")
    return (f"[0:v]sendcmd=f='{cmd}',crop@c=w={first[0]}:h={first[1]}:x={first[2]}:y={first[3]},"
            f"scale={OW}:{OH},setsar=1[v]")


def test_track_matches_ffmpeg(source, tmp_path):
    import math
    import reframe_v2
    n = int(DUR * FPS)
    xs = [int(656 + 500 * math.sin(i / 20)) for i in range(n)]
    boxes = [(608, 1080, x, 0) for x in xs]
    graph = _track_graph(tmp_path, reframe_v2.dedupe_sendcmd_lines(xs, FPS), boxes[0])
    ref, got = str(tmp_path / "ref.mp4"), str(tmp_path / "got.mp4")
    _ffmpeg(source, graph, ref)
    assert gpu_render.render(source, [(0.0, DUR, graph, boxes)], W, H, OW, OH, FPS, got)
    assert _frames(got) == _frames(ref)
    assert _ssim(got, ref) >= MIN_SSIM


def _frame(path, k, vf=None):
    out = subprocess.run(["ffmpeg", "-v", "error", "-i", path,
                          "-vf", f"select=eq(n\\,{k})" + (f",{vf}" if vf else ""),
                          "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "gray", "-"],
                         capture_output=True, check=True).stdout
    return out


def test_punch_in_keeps_every_frame_and_zooms(source, tmp_path):
    """ffmpeg's crop stops the stream at the first size change sendcmd makes
    (the render ends at the first push). The GPU render keeps every frame, and
    each zoomed frame is that frame's box cropped and scaled."""
    import math
    import numpy as np
    import punch_in
    n = _frames(source)
    xs = [int(656 + 300 * math.sin(i / 20)) for i in range(n)]
    boxes = punch_in.crop_boxes(xs, punch_in.zoom_curve(n, FPS, [1.0, 2.5], max_zoom=1.25),
                                608, 1080, W, H)
    graph = _track_graph(tmp_path, punch_in.sendcmd_lines(boxes, FPS), boxes[0])
    got = str(tmp_path / "got.mp4")
    assert gpu_render.render(source, [(0.0, DUR, graph, boxes)], W, H, OW, OH, FPS, got)
    assert _frames(got) == n
    order = gpu_render.sendcmd_order(n, FPS)
    zoomed = [k for k in range(n) if boxes[order[k]][0] < 600]
    assert zoomed
    for k in zoomed[::max(1, len(zoomed) // 4)]:
        w, h, x, y = boxes[order[k]]
        ref = np.frombuffer(_frame(source, k, f"crop={w}:{h}:{x}:{y},scale={OW}:{OH}"), np.uint8)
        mine = np.frombuffer(_frame(got, k), np.uint8)
        assert np.abs(ref.astype(int) - mine.astype(int)).mean() < 3.0


def test_scenes_concatenate_like_segments(source, tmp_path):
    """Three scenes with different layouts in one pass vs three segments + concat."""
    g = _graphs()
    scenes = [(0.0, 1.5, g["general"], None), (1.5, 1.0, g["split"], None),
              (2.5, 1.5, g["inset"], None)]
    segs = []
    for k, (ss, dur, graph, _) in enumerate(scenes):
        seg = str(tmp_path / f"seg{k}.mp4")
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-ss", f"{ss:.4f}",
                        "-t", f"{dur:.4f}", "-i", source, "-filter_complex", graph,
                        "-map", "[v]", *_NVENC_ARGS[QUALITY_FAST], "-an", seg], check=True)
        segs.append(seg)
    lst = tmp_path / "list.txt"
    lst.write_text("".join(f"file '{s}'\n" for s in segs))
    ref, got = str(tmp_path / "ref.mp4"), str(tmp_path / "got.mp4")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0",
                    "-i", str(lst), "-c", "copy", ref], check=True)
    assert gpu_render.render(source, scenes, W, H, OW, OH, FPS, got)
    assert _frames(got) == _frames(ref)
    assert _ssim(got, ref) >= MIN_SSIM


def test_watermark_overlay_matches_ffmpeg(source, tmp_path):
    import main as m
    logo = tmp_path / "logo.png"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i",
                    "color=c=white:s=600x180,format=rgba", "-frames:v", "1", str(logo)], check=True)
    graph = _graphs()["general"]
    ref, got = str(tmp_path / "ref.mp4"), str(tmp_path / "got.mp4")
    full = graph + ";" + m.watermark_filter(OW, OH, video="[v]", out="[vw]")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", source, "-i", str(logo),
                    "-filter_complex", full, "-map", "[vw]", *_NVENC_ARGS[QUALITY_FAST],
                    "-an", ref], check=True)
    mark = {"path": str(logo), "x": int(OW * m.WATERMARK_MARGIN_RATIO),
            "y": int(OH * m.WATERMARK_Y_RATIO),
            "w": max(80, int(OW * m.WATERMARK_WIDTH_RATIO)), "opacity": m.WATERMARK_OPACITY}
    assert gpu_render.render(source, [(0.0, DUR, graph, None)], W, H, OW, OH, FPS, got,
                             overlay=mark)
    assert _ssim(got, ref) >= MIN_SSIM
