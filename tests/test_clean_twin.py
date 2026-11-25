"""A free clip's clean twin kept as a recipe and encoded when it is needed.

With CLEAN_LAZY=1 the free burn encodes only the served ``wm_`` copy
(clean_twin.py). The twin encoded later from the recipe must be the file the
pass used to write: same pictures, same audio. Here on the CPU (ffmpeg, the
GPU burn off) so it runs anywhere ffmpeg does; tests/test_gpu_burn_equivalence
already holds the card's burn to ffmpeg's.
"""
import asyncio
import json
import os
import shutil
import subprocess

import pytest

import clean_twin
import watermarked



def _has_libass():
    if shutil.which("ffmpeg") is None:
        return False
    out = subprocess.run(["ffmpeg", "-hide_banner", "-filters"], capture_output=True, text=True)
    return any(l.split()[1:2] == ["ass"] for l in out.stdout.splitlines() if l.strip())


needs_ass = pytest.mark.skipif(not _has_libass(), reason="needs ffmpeg with libass")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOGO = os.path.join(ROOT, "assets", "watermark.png")
W, H = 360, 640

ASS = """[Script Info]
ScriptType: v4.00+
PlayResX: 360
PlayResY: 640

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,DejaVu Sans,40,&H00FFFFFF,&H000000FF,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,3,0,2,20,20,120,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
Dialogue: 0,0:00:00.00,0:00:01.50,Default,,0,0,0,,{\\c&H00D7FF&}Hola{\\c&HFFFFFF&} mundo
Dialogue: 0,0:00:01.50,0:00:03.00,Default,,0,0,0,,Se codifica al pagar
"""


def _watermark(vw, vh, video="[0:v]", logo="[1:v]", out=""):
    return (f"{logo}scale={max(80, int(vw * 0.3))}:-1,format=rgba,"
            f"colorchannelmixer=aa=0.85[wm];{video}[wm]overlay=x={int(vw * 0.05)}:y={int(vh * 0.4)}{out}")


@pytest.fixture
def job(tmp_path, monkeypatch):
    import gpu_burn
    monkeypatch.setattr(gpu_burn, "ENABLED", False)
    monkeypatch.setenv("FFMPEG_ENCODER", "libx264")
    monkeypatch.setenv("FFMPEG_THREADS", "1")
    d = tmp_path / "job"
    d.mkdir()
    clip = d / "t_clip_1.mp4"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error",
                    "-f", "lavfi", "-i", f"testsrc2=size={W}x{H}:rate=30",
                    "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000",
                    "-t", "3", "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p",
                    "-c:a", "aac", "-shortest", str(clip)], check=True)
    ass = d / "autosubs_1_ab.ass"
    ass.write_text(ASS)
    return d, clip, ass


def _md5s(path, stream):
    out = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-map", f"0:{stream}",
                          "-f", "framemd5", "-"], capture_output=True, text=True, check=True).stdout
    return [l.rsplit(",", 1)[-1].strip() for l in out.splitlines() if l and not l.startswith("#")]


def _same(a, b):
    assert _md5s(a, "v") == _md5s(b, "v")
    assert _md5s(a, "a") == _md5s(b, "a")


@needs_ass
class TestCaptionsOnly:
    def test_the_twin_from_the_recipe_is_the_one_the_pass_wrote(self, job, tmp_path):
        import subtitles
        d, clip, ass = job
        out = d / "subtitled_5_t_clip_1.mp4"
        wm = d / watermarked.marked_name(out.name)
        subtitles.burn_subtitles(str(clip), str(ass), str(out),
                                 mark=(LOGO, _watermark, str(wm)), clean=False)
        assert wm.exists() and not out.exists()
        assert clean_twin.has_recipe(str(d), out.name)
        ref = tmp_path / "ref"
        ref.mkdir()
        ref_out, ref_wm = ref / out.name, ref / wm.name
        subtitles.burn_subtitles(str(clip), str(ass), str(ref_out),
                                 mark=(LOGO, _watermark, str(ref_wm)))
        _same(wm, ref_wm)
        os.remove(ass)  # the recipe keeps its own copy of the captions
        assert clean_twin.materialize(str(d), out.name)
        _same(out, ref_out)
        assert not [n for n in os.listdir(d) if n.startswith(".part_")]

    def test_without_the_mark_the_clean_file_is_written(self, job):
        import subtitles
        d, clip, ass = job
        out = d / "subtitled_5_t_clip_1.mp4"
        subtitles.burn_subtitles(str(clip), str(ass), str(out), clean=False)
        assert out.exists() and not clean_twin.has_recipe(str(d), out.name)


@needs_ass
class TestHookAndCaptions:
    def test_hook_recipe_and_captions_give_the_pass_twin(self, job, tmp_path, monkeypatch):
        import hooks
        import subtitles
        monkeypatch.setattr(hooks, "LAZY", True)
        d, clip, ass = job
        vf = subtitles.subtitles_filter(str(ass))
        hooked = d / "hooked_2_t_clip_1.mp4"
        captioned = d / f"subtitled_3_{hooked.name}"
        wm = d / watermarked.marked_name(captioned.name)
        hooks.add_hook_to_video(str(clip), "¿Se ve igual?", str(hooked), duration=2.0,
                                also=(vf, str(captioned), str(ass)),
                                mark=(LOGO, _watermark, str(wm)), clean=False)
        assert wm.exists() and not captioned.exists() and not hooked.exists()
        assert hooks.has_recipe(str(d), hooked.name)
        assert clean_twin.has_recipe(str(d), captioned.name)

        # The pass as it was, from the same hook image and position.
        with open(hooks.recipe_paths(str(d), hooked.name)[0]) as f:
            r = json.load(f)
        png = hooks.recipe_paths(str(d), hooked.name)[1]
        ref = tmp_path / "ref"
        ref.mkdir()
        graph = (f"{hooks._overlay_graph(r['x'], r['y'], r['duration'])},{vf},split=2[s][sm];"
                 + _watermark(W, H, video="[sm]", logo="[2:v]", out="[m]"))
        tail = hooks._encode_tail()
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(clip), "-i", png, "-i", LOGO,
                        "-filter_complex", graph,
                        "-map", "[s]", "-map", "0:a?", *tail, str(ref / "s.mp4"),
                        "-map", "[m]", "-map", "0:a?", *tail, str(ref / "m.mp4")], check=True)
        _same(wm, ref / "m.mp4")
        assert clean_twin.materialize(str(d), captioned.name)
        _same(captioned, ref / "s.mp4")
        # Still from the hook's source once the hooked picture exists too.
        os.remove(captioned)
        assert hooks.materialize(str(d), hooked.name)
        assert clean_twin.materialize(str(d), captioned.name)
        _same(captioned, ref / "s.mp4")

        names = clean_twin.archive_names(str(d), captioned.name)
        assert set(names) == {os.path.basename(p) for p in
                              (*clean_twin.recipe_paths(str(d), captioned.name),
                               *hooks.recipe_paths(str(d), hooked.name))}


class TestRecipe:
    def test_a_broken_recipe_makes_nothing(self, tmp_path):
        d = str(tmp_path)
        assert not clean_twin.materialize(d, "subtitled_1_t_clip_1.mp4")
        json_path, ass_path = clean_twin.recipe_paths(d, "subtitled_1_t_clip_1.mp4")
        with open(json_path, "w") as f:
            f.write("{}")
        open(ass_path, "w").close()
        assert not clean_twin.materialize(d, "subtitled_1_t_clip_1.mp4")
        with open(json_path, "w") as f:
            json.dump({"source": "../../etc/t_clip_1.mp4"}, f)
        assert not clean_twin.materialize(d, "subtitled_1_t_clip_1.mp4")

    def test_the_source_must_sit_next_to_the_clip(self, tmp_path):
        (tmp_path / "a").mkdir()
        (tmp_path / "x.ass").write_text("x")
        with pytest.raises(ValueError):
            clean_twin.write_recipe(str(tmp_path / "subtitled_1_c.mp4"),
                                    str(tmp_path / "a" / "c.mp4"), str(tmp_path / "x.ass"))

    def test_lazy_only_when_asked(self, monkeypatch):
        monkeypatch.delenv("CLEAN_LAZY", raising=False)
        assert not clean_twin.lazy()
        monkeypatch.setenv("CLEAN_LAZY", "1")
        assert clean_twin.lazy()

    def test_never_sent_to_a_remote_worker(self):
        import remote_worker
        assert not remote_worker.allowed_option("CLEAN_LAZY", "1")
        assert remote_worker.job_options({"CLEAN_LAZY": "1", "WATERMARK": "1"}, {}) == {
            "WATERMARK": "1"}


def _recipe_job(d):
    """A free clip whose clean twin is only a recipe."""
    (d / "t_clip_1.mp4").write_bytes(b"x")
    (d / "wm_subtitled_3_t_clip_1.mp4").write_bytes(b"x")
    (d / "a.ass").write_text("x")
    clean_twin.write_recipe(str(d / "subtitled_3_t_clip_1.mp4"), str(d / "t_clip_1.mp4"),
                            str(d / "a.ass"))


class TestWalkBacks:
    def test_the_mark_strips_to_a_recipe_twin_without_encoding(self, tmp_path, monkeypatch):
        app = pytest.importorskip("app")
        _recipe_job(tmp_path)
        monkeypatch.setattr(clean_twin, "materialize", lambda *a: pytest.fail("encoded"))
        assert app._strip_watermark_copy(str(tmp_path), "wm_subtitled_3_t_clip_1.mp4") == \
            "subtitled_3_t_clip_1.mp4"
        assert app._strip_burned_captions(str(tmp_path), "wm_subtitled_3_t_clip_1.mp4") == \
            "t_clip_1.mp4"

    def test_a_walk_that_stops_on_the_twin_encodes_it(self, tmp_path, monkeypatch):
        app = pytest.importorskip("app")
        _recipe_job(tmp_path)
        os.remove(tmp_path / "t_clip_1.mp4")
        made = []
        monkeypatch.setattr(clean_twin, "materialize", lambda d, n: made.append(n) or True)
        assert app._strip_burned_captions(str(tmp_path), "wm_subtitled_3_t_clip_1.mp4") == \
            "subtitled_3_t_clip_1.mp4"
        assert made == ["subtitled_3_t_clip_1.mp4"]


class TestLibrary:
    def test_the_recipe_rides_to_r2_instead_of_the_twin(self, tmp_path):
        videos = pytest.importorskip("cloud.videos")
        _recipe_job(tmp_path)
        assert videos._archived_twins(str(tmp_path), "wm_subtitled_3_t_clip_1.mp4", "t", 0) == [
            os.path.basename(p) for p in
            clean_twin.recipe_paths(str(tmp_path), "subtitled_3_t_clip_1.mp4")]
        (tmp_path / "subtitled_3_t_clip_1.mp4").write_bytes(b"x")
        assert videos._archived_twins(str(tmp_path), "wm_subtitled_3_t_clip_1.mp4", "t", 0) == [
            "subtitled_3_t_clip_1.mp4"]

    def test_an_upgrade_encodes_a_twin_r2_does_not_hold(self, monkeypatch):
        videos = pytest.importorskip("cloud.videos")
        in_r2 = set()
        monkeypatch.setattr(videos.storage, "job_key", lambda u, j, n: f"{u}/{j}/{n}")
        monkeypatch.setattr(videos.storage, "object_size",
                            lambda key: 7 if key in in_r2 else None)
        asked = []

        async def render(user_id, proj, clean):
            asked.append(clean)
            if clean != "broken.mp4":
                in_r2.add(f"{user_id}/{proj.job_id}/{clean}")
            return clean != "broken.mp4"

        monkeypatch.setattr(videos, "_clean_render", render)

        class P:
            job_id = "j"
        assert asyncio.run(videos._clean_size("u", P, "subtitled_3_t_clip_1.mp4")) == 7
        assert asyncio.run(videos._clean_size("u", P, "subtitled_3_t_clip_1.mp4")) == 7
        assert asyncio.run(videos._clean_size("u", P, "broken.mp4")) is None
        assert asked == ["subtitled_3_t_clip_1.mp4", "broken.mp4"]
