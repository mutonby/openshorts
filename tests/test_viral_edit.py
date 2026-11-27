"""The viral edit (cold open, punch-in, dead air, emoji, sound effects) and the
streamer stack layout. Planning and graph building are pure; one real render
on synthetic footage checks the encode end to end."""
import json
import shutil
import subprocess

import pytest

import facecam
import framing
import layout_ranges
import recut
import sfx
import subtitles
import viral_edit as ve

needs_ffmpeg = pytest.mark.skipif(not shutil.which("ffmpeg"), reason="needs ffmpeg")


def _words(*spans):
    return [{"w": f"w{i}", "s": s, "e": e} for i, (s, e) in enumerate(spans)]


# --- planning ---------------------------------------------------------------

def test_cold_open_teases_the_peak_then_plays_the_setup():
    p = ve.plan(100.0, 140.0, peak=125.0)
    assert p["teaser"] and p["segments"][0] == (124.3, 126.4)
    assert p["segments"][1] == (100.0, 140.0)
    assert p["teaser_length"] == pytest.approx(2.1)
    # The peak plays twice: in the teaser, then again with its context.
    assert p["peak_out"] == [pytest.approx(0.7), pytest.approx(2.1 + 25.0)]
    assert p["duration"] == pytest.approx(42.1)


@pytest.mark.parametrize("peak", [None, 103.0, 139.5])
def test_no_cold_open_without_room_for_it(peak):
    p = ve.plan(100.0, 140.0, peak=peak)
    assert not p["teaser"] and p["segments"] == [(100.0, 140.0)]


def test_cold_open_can_be_switched_off():
    assert not ve.plan(100.0, 140.0, peak=125.0, cold_open=False)["teaser"]


def test_dead_air_tightens_long_gaps_but_never_near_the_peak():
    words = _words((100, 101), (101.2, 102), (106, 107),      # 4 s gap at 102-106
                   (119, 120), (127, 128), (128.2, 129))      # 7 s gap at 120-127 holds the peak
    cuts = ve.dead_air(words, 100.0, 140.0, peak=124.0)
    assert cuts == [(102.35, 105.65)]
    p = ve.plan(100.0, 140.0, peak=124.0, words=words, cold_open=False)
    assert p["segments"] == [(100.0, 102.35), (105.65, 140.0)]
    assert p["peak_out"] == [pytest.approx(124.0 - 100.0 - 3.3)]


def test_dead_air_needs_words_and_respects_its_budget():
    assert ve.dead_air([], 0, 60, None) == []
    words = _words((0, 1), (10, 11), (20, 21), (30, 31), (40, 41))   # four 9 s gaps
    cuts = ve.dead_air(words, 0, 60, None)
    assert sum(b - a for a, b in cuts) <= ve.DEAD_AIR_MAX_SHARE * 60 + 1e-6
    assert len(cuts) == 1


def test_dead_air_keeps_editor_sized_segments_at_both_clip_edges():
    words = _words((10, 10.08), (12.4, 12.7), (27.3, 27.6), (29.92, 30))
    plan = ve.plan(10, 30, peak=20, words=words)
    assert plan["cuts"] == [(10.5, 12.05), (27.95, 29.5)]
    assert plan["segments"] == [(19.3, 21.4), (10, 10.5), (12.05, 27.95), (29.5, 30)]
    segments = [{"start": start, "end": end} for start, end in plan["segments"]]
    assert recut.normalize_segments(segments) == segments


def test_find_peak_prefers_the_hype_engines_peak_inside_the_clip():
    assert ve.find_peak("x.mp4", 100, 140, hint=120) == 120.0


def test_emoji_for_reaction_words_spaced_and_limited():
    vwords = [{"word": " No", "start": 1.0}, {"word": " way!", "start": 1.3},
              {"word": " haha", "start": 2.0},                 # too close to the first
              {"word": " bro", "start": 6.0}, {"word": " let's", "start": 10.0},
              {"word": " go", "start": 10.2}, {"word": " what", "start": 14.0},
              {"word": " wait", "start": 18.0}]
    cues = ve.emoji_cues(vwords, duration=30)
    assert cues == [(1.0, "😱"), (6.0, "💀"), (10.0, "🔥"), (14.0, "😳")]


def test_a_laughing_streamer_gets_a_laugh_on_the_peak():
    assert ve.emoji_cues([], peak_out=[0.7, 26.6], laughing=True, duration=30) == [(0.7, "😂"), (26.6, "😂")]


def test_punch_boxes_zoom_in_and_back_out():
    boxes = ve.punch_boxes(4.0, 30, [1.0], 1080, 768, zoom=1.2, focus_y=0.55)
    assert boxes[0] == (1080, 768, 0, 0)
    tightest = min(boxes, key=lambda b: b[0])
    assert tightest[0] == 900 and tightest[1] == 640         # 1080/1.2, 768/1.2
    assert all(b[0] % 2 == 0 and b[1] % 2 == 0 for b in boxes)
    assert boxes[-1] == (1080, 768, 0, 0)


def test_options_follow_the_flags(monkeypatch):
    monkeypatch.delenv("CLIP_VIRAL_EDIT", raising=False)
    monkeypatch.setenv("CLIP_SFX", "0")
    opts = ve.options()
    assert opts["cold_open"] and opts["punch"] and not opts["sfx"]
    monkeypatch.setenv("CLIP_VIRAL_EDIT", "0")
    assert not any(ve.options().values())


# --- the filter graph ---------------------------------------------------------

def test_build_concatenates_segments_and_mixes_sound_effects(tmp_path):
    inputs, graph = ve.build(
        source="src.mkv", base=10.0, span=40.0, segments_rel=[(25.3, 27.4), (1.0, 31.0)],
        layout_graph="[0:v]scale=1080:1920[v]", punch_full=None,
        emoji_inputs=[(tmp_path / "e.png", 455, 576, 0.7, 1.8)],
        sfx_inputs=[(tmp_path / "boom.wav", 26.55, 0.8)], audio=True, out_w=1080, out_h=1920)
    assert inputs[:6] == ["-ss", "10.0000", "-t", "40.0000", "-i", "src.mkv"]
    assert "split=2[s0][s1]" in graph and "asplit=2[r0][r1]" in graph
    assert "[v0][a0][v1][a1]concat=n=2:v=1:a=1[cat][acat]" in graph
    assert "[cat]scale=1080:1920[v]" in graph
    assert "[v][1:v]overlay=x=455:y=576:enable='between(t,0.700,1.800)'[e1]" in graph
    assert "[2:a]adelay=26550:all=1,volume=0.80[x0]" in graph
    assert "amix=inputs=2:normalize=0:duration=first" in graph and "loudnorm" in graph
    assert graph.endswith("[aout]") and "[e1]format=yuv420p[vout]" in graph


def test_build_single_segment_without_audio():
    _, graph = ve.build(source="s.mp4", base=0.0, span=31.0, segments_rel=[(0.5, 30.5)],
                        layout_graph="[0:v]null[v]", punch_full=None, emoji_inputs=[], sfx_inputs=[],
                        audio=False, out_w=1080, out_h=1920)
    assert "concat" not in graph and "atrim" not in graph and "[aout]" not in graph
    assert graph.startswith("[0:v]trim=start=0.5000:end=30.5000,setpts=PTS-STARTPTS[cat]")


# --- the streamer stack ---------------------------------------------------------

CAM = {"box": (0, 432, 390, 219), "face": (169, 522, 51, 71)}


def test_stack_layout_crops_inside_the_overlay_and_keeps_the_game_clear_of_it():
    g = facecam.stack_layout(1920, 1080, 1080, 1920, CAM)
    cw, ch, cx, cy = g["cam_crop"]
    assert g["cam_h"] + g["game_h"] == 1920
    assert 0 <= cx and cx + cw <= 390 and 432 <= cy and cy + ch <= 432 + 219 + 1
    shown = (g["card"]["w"], g["card"]["h"]) if g["card"] else (1080, g["cam_h"])
    assert abs(cw / ch - shown[0] / shown[1]) < 0.03              # no stretch
    assert g["cam_scale"] <= 2.5 * 1.12                           # never blown up past its pixels
    gw, gh, gx, gy = g["game_crop"]
    assert gx >= 390 and gx + gw <= 1920 and gh == 1080          # the webcam is not shown twice
    assert abs(gw / gh - 1080 / g["game_pic_h"]) < 0.01


def test_stack_layout_gives_up_when_the_game_crop_would_show_the_webcam():
    centre_cam = {"box": (700, 300, 520, 400), "face": (900, 380, 80, 100)}
    assert facecam.stack_layout(1920, 1080, 1080, 1920, centre_cam) is None


def test_stack_filtergraph_uses_only_crop_scale_vstack():
    g = facecam.stack_layout(1920, 1080, 1080, 1920, CAM)
    graph = facecam.stack_filtergraph(g, 1080, 1920, cam_zoom_cmds="/job/.zoom.cmd")
    assert graph.startswith("[0:v]split=") and graph.endswith("[v]")
    assert "vstack=inputs=2" in graph and f"crop@pz=w=1080:h={g['cam_h']}" in graph
    assert "sendcmd=f='/job/.zoom.cmd'" in graph


def test_persistent_face_ignores_faces_that_move():
    webcam = (100.0, 500.0, 50.0, 70.0, 0.9)
    frames = [[webcam, (300.0 + 90 * i, 200.0 + 40 * i, 60.0, 80.0, 0.95)] for i in range(10)]
    assert facecam._persistent_face(frames, 1280) == webcam[:4]
    assert facecam._persistent_face([[f[1]] for f in frames], 1280) is None


def test_the_vendored_model_is_the_pinned_one():
    assert facecam.model_available()


def test_streamer_is_a_gameplay_framing(monkeypatch):
    assert framing.gameplay_like("streamer") and framing.gameplay_like("gameplay")
    assert not framing.gameplay_like("auto") and not framing.gameplay_like("legacy")
    assert "streamer" in framing.FRAMING_METHODS


def test_stack_captions_sit_on_the_seam():
    ranges = layout_ranges.normalise([(0, 30, "STACK", None, {"seam": 0.34, "caption_y": 0.34})])
    assert layout_ranges.split_ranges(ranges) == [(0.0, 30.0, 0.34)]


# --- captions that pop ------------------------------------------------------------

def test_emphasis_words_get_their_own_colour(tmp_path):
    transcript = {"segments": [{"start": 0, "end": 3, "text": "no way bro okay", "words": [
        {"word": " no", "start": 0.0, "end": 0.4}, {"word": " way", "start": 0.4, "end": 0.8},
        {"word": " bro", "start": 0.8, "end": 1.2}, {"word": " okay", "start": 1.2, "end": 1.6}]}]}
    path = tmp_path / "c.ass"
    style = subtitles.STREAM_CAPTION_STYLE
    assert subtitles.generate_ass(transcript, 0, 3, str(path), max_chars=style["max_chars"],
                                  effect="pop", uppercase=True, highlight_color="#FFE500",
                                  emphasis_color=style["emphasis_color"])
    text = path.read_text(encoding="utf-8-sig")
    emph = subtitles._hex_to_ass_inline_color("#FF4D4D")
    assert f"{{\\c{emph}}}NO" in text and f"{{\\c{emph}}}BRO" in text
    assert f"{{\\c{emph}}}OKAY" not in text
    plain = tmp_path / "p.ass"
    subtitles.generate_ass(transcript, 0, 3, str(plain), effect="pop")
    assert emph not in plain.read_text(encoding="utf-8-sig")


# --- sound effects -------------------------------------------------------------------

def test_sound_effects_are_synthesised_once(tmp_path, monkeypatch):
    monkeypatch.setenv("SFX_CACHE_DIR", str(tmp_path))
    for name in ("boom", "whoosh", "hit"):
        path = sfx.path(name)
        assert path and path.endswith(f"{name}-v{sfx.VERSION}.wav")
    first = open(sfx.path("boom"), "rb").read()
    assert open(sfx.path("boom"), "rb").read() == first
    assert sfx.path("airhorn") is None


def test_sound_effect_placements_follow_the_edit():
    teased = ve.plan(100.0, 140.0, peak=125.0)
    assert [(round(t, 2), name) for t, name, _ in sfx.placements(teased)] == [
        (0.0, "hit"), (1.8, "whoosh"), (27.05, "boom")]
    plain = ve.plan(100.0, 140.0, peak=125.0, cold_open=False)
    assert [name for _, name, _ in sfx.placements(plain)] == ["boom"]
    assert sfx.placements(ve.plan(100.0, 140.0)) == []


# --- a real render ----------------------------------------------------------------------

@needs_ffmpeg
def test_render_end_to_end_on_synthetic_footage(tmp_path, monkeypatch):
    monkeypatch.setenv("SFX_CACHE_DIR", str(tmp_path / "sfx"))
    src = tmp_path / "game.mp4"
    subprocess.run([
        "ffmpeg", "-nostdin", "-loglevel", "error", "-y",
        "-f", "lavfi", "-i", "testsrc2=s=640x360:r=30:d=40",
        "-f", "lavfi", "-i", "sine=frequency=700:sample_rate=16000:duration=40,volume=0.05",
        "-f", "lavfi", "-i", "sine=frequency=2200:sample_rate=16000:duration=2,volume=0.9,adelay=25000,apad=whole_dur=40",
        "-filter_complex", "[1][2]amix=inputs=2:normalize=0[a]", "-map", "0:v", "-map", "[a]",
        "-c:v", "libx264", "-preset", "ultrafast", "-c:a", "aac", "-shortest", str(src)], check=True)
    out = tmp_path / "edit.mp4"
    result = ve.render(str(src), str(out), aspect=9 / 16, start=5.0, end=37.0, layout="streamer",
                       workdir=str(tmp_path))
    assert result["teaser"] and 24.0 <= result["peak"] <= 28.0
    assert result["layout"] == "gameplay"                 # no webcam in a test pattern
    assert [s["sound"] for s in result["sfx"]] == ["hit", "whoosh", "boom"]
    probe = json.loads(subprocess.run(["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json",
                                       str(out)], capture_output=True, check=True).stdout)
    video = next(s for s in probe["streams"] if s["codec_type"] == "video")
    assert (video["width"], video["height"]) == (1080, 1920)
    assert any(s["codec_type"] == "audio" for s in probe["streams"])
    assert abs(float(probe["format"]["duration"]) - result["duration"]) < 0.15
    assert layout_ranges.read(str(out))[0]["layout"] == "gameplay"
    assert not list(tmp_path.glob(".edit.*"))               # command files cleaned up
