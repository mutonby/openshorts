"""osrender burn draws the hook, the captions and the mark exactly as ffmpeg does (needs a GPU).

The same clip goes through the ffmpeg burn graph (hooks.add_hook_to_video's,
decoded on the card like the job does) and through ``osrender burn``; both
write each stage's pictures as raw yuv420p, which must be identical byte for
byte. Then the encoded files gpu_burn writes are decoded and compared frame by
frame against the ffmpeg pass's. Skipped unless the osrender binary and a
working NVENC are present. OSRENDER_TEST_VIDEO points it at a real clip; a
synthetic one is used otherwise.
"""
import hashlib
import json
import os
import subprocess

import pytest

import gpu_burn
import gpu_render
import subtitles
from ffmpeg_utils import nvenc_available

pytestmark = pytest.mark.skipif(
    not (os.path.isfile(gpu_render.BIN) and nvenc_available()),
    reason="needs the osrender binary and NVENC")

W, H, FPS, DUR = 1080, 1920, 30, 7.0
LOGO = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "assets", "watermark.png")

ASS = """[Script Info]
ScriptType: v4.00+
PlayResX: 1080
PlayResY: 1920
WrapStyle: 0

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,DejaVu Sans,96,&H00FFFFFF,&H000000FF,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,6,3,2,40,40,420,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
Dialogue: 0,0:00:00.00,0:00:02.10,Default,,0,0,0,,{\\c&H00D7FF&}Hola{\\c&HFFFFFF&} mundo
Dialogue: 0,0:00:02.10,0:00:04.30,Default,,0,0,0,,{\\fad(200,200)}Esto {\\c&H3C14DC&\\3c&HFFFFFF&}se quema{\\r} en GPU
Dialogue: 0,0:00:04.30,0:00:07.00,Default,,0,0,0,,{\\an5\\pos(541,961)\\bord9}Centro {\\alpha&H60&}medio
"""


@pytest.fixture(scope="module")
def work(tmp_path_factory):
    d = tmp_path_factory.mktemp("burn")
    clip = str(d / "clip.mp4")
    real = os.environ.get("OSRENDER_TEST_VIDEO")
    inp = ["-ss", "0", "-i", real] if real else [
        "-f", "lavfi", "-i", f"testsrc2=size={W}x{H}:rate={FPS}",
        "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000"]
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", *inp, "-t", str(DUR),
                    "-vf", f"scale={W}:{H},fps={FPS},format=yuv420p",
                    "-c:v", "libx264", "-crf", "16", "-c:a", "aac", "-shortest", clip], check=True)
    ass = str(d / "subs.ass")
    with open(ass, "w") as f:
        f.write(ASS)
    import hooks
    png = str(d / "hook.png")
    hooks.create_hook_image("¿Probamos el gancho en GPU?", 972, png, style="pill")
    return d, clip, ass, png


def _watermark(vw, vh, video="[0:v]", logo="[1:v]", out=""):
    return (f"{logo}scale={max(80, int(vw * 0.3))}:-1,format=rgba,"
            f"colorchannelmixer=aa=0.85[wm];{video}[wm]overlay=x={int(vw * 0.05)}:y={int(vh * 0.4)}{out}")


def _frames(path):
    size = W * H * 3 // 2
    with open(path, "rb") as f:
        data = f.read()
    return [hashlib.md5(data[i:i + size]).hexdigest() for i in range(0, len(data), size)]


def _tagged(clip, d):
    """The clip with BT.709 in its headers: ffmpeg then converts the static
    images to that matrix, and the GPU burn must do the same."""
    out = str(d / "clip709.mp4")
    if not os.path.exists(out):
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", clip, "-c", "copy", "-bsf:v",
                        "h264_metadata=colour_primaries=1:transfer_characteristics=1:matrix_coefficients=1",
                        out], check=True)
    return out


@pytest.mark.parametrize("hook_xy,bt709", [((69, 1085), False), ((54, 3), False),
                                           ((0, 1700), False), ((69, 1085), True)])
def test_raw_stages_are_identical(work, hook_xy, bt709):
    d, clip, ass, png = work
    if bt709:
        clip = _tagged(clip, d)
    x, y = hook_xy
    vf = subtitles.subtitles_filter(ass)
    graph = (f"[0:v]hwdownload,format=nv12,format=yuv420p[g];[g][1:v]overlay={x}:{y}:"
             f"enable='between(t,0,5.0)',split=2[h][c];[c]{vf},split=2[s][sm];"
             + _watermark(W, H, video="[sm]", logo="[2:v]", out="[m]"))
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-hwaccel", "cuda", "-hwaccel_output_format", "cuda",
           "-i", clip, "-i", png, "-i", LOGO, "-filter_complex", graph]
    for s in "hsm":
        cmd += ["-map", f"[{s}]", "-f", "rawvideo", "-pix_fmt", "yuv420p", str(d / f"ref_{s}.yuv")]
    subprocess.run(cmd, check=True)
    chain, mx, my = gpu_burn.parse_mark(_watermark, W, H)
    frames, _, matrix = gpu_burn._probe(clip)
    assert frames and (matrix == "bt709" or not bt709)
    gpu_burn._planes(png, "", str(d / "hook.nut"), matrix)
    gpu_burn._planes(LOGO, chain, str(d / "mark.nut"), matrix)
    spec = {"input": clip,
            "hook": {"planes": str(d / "hook.nut"), "x": x, "y": y, "until": 5.0},
            "ass": {"file": ass, "fonts_dir": gpu_burn.parse_ass(vf)[1]},
            "mark": {"planes": str(d / "mark.nut"), "x": mx, "y": my, "until": None},
            "outputs": [{"stage": st, "path": str(d / f"gpu_{c}.yuv"), "raw": True}
                        for st, c in (("hook", "h"), ("ass", "s"), ("mark", "m"))]}
    with open(d / "spec.json", "w") as f:
        json.dump(spec, f)
    subprocess.run([gpu_render.BIN, "burn", str(d / "spec.json")], check=True, capture_output=True)
    for s in "hsm":
        ref, got = _frames(d / f"ref_{s}.yuv"), _frames(d / f"gpu_{s}.yuv")
        assert len(ref) == len(got) == round(DUR * FPS)
        assert ref == got, f"stage {s}: {sum(a != b for a, b in zip(ref, got))} frames differ"


def test_the_encoded_outputs_decode_to_the_same_pictures(work, monkeypatch):
    d, clip, ass, png = work
    monkeypatch.setenv("FFMPEG_ENCODER", "auto")
    vf = subtitles.subtitles_filter(ass)
    outs = [("hook", str(d / "h.mp4")), ("ass", str(d / "s.mp4")), ("mark", str(d / "m.mp4"))]
    assert gpu_burn.burn(clip, outs, hook=(png, 69, 1085, 5.0), ass_vf=vf,
                         mark=(LOGO, _watermark), video_size=(W, H))
    monkeypatch.setattr(gpu_burn, "ENABLED", False)
    import hooks
    cpu = [str(d / "cpu_h.mp4"), str(d / "cpu_s.mp4"), str(d / "cpu_m.mp4")]
    graph = (f"[0:v][1:v]overlay=69:1085:enable='between(t,0,5.0)',split=2[h][c];[c]{vf},split=2[s][sm];"
             + _watermark(W, H, video="[sm]", logo="[2:v]", out="[m]"))
    cmd = ["ffmpeg", "-y", "-i", clip, "-i", png, "-i", LOGO, "-filter_complex", graph]
    for label, out in zip("hsm", cpu):
        cmd += ["-map", f"[{label}]", "-map", "0:a?", *hooks._encode_tail(), out]
    subprocess.run(cmd, check=True, capture_output=True)

    def md5s(path, stream):
        out = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-map", f"0:{stream}",
                              "-f", "framemd5", "-"], capture_output=True, text=True, check=True).stdout
        return [l.rsplit(",", 1)[-1].strip() for l in out.splitlines() if l and not l.startswith("#")]

    for (_, gpu), ref in zip(outs, cpu):
        assert md5s(gpu, "v") == md5s(ref, "v")
        assert md5s(gpu, "a") == md5s(ref, "a")
