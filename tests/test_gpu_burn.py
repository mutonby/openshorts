"""gpu_burn: what it takes over from the ffmpeg burn and when it steps back (no GPU needed)."""
import os
import subprocess

import pytest

import gpu_burn
import gpu_render
import subtitles


def _watermark(vw, vh, video="[0:v]", logo="[1:v]", out=""):
    """main.watermark_filter's shape, without importing main."""
    return (f"{logo}scale={max(80, int(vw * 0.3))}:-1,format=rgba,"
            f"colorchannelmixer=aa=0.85[wm];{video}[wm]overlay=x={int(vw * 0.05)}:y={int(vh * 0.4)}{out}")


class TestParse:
    def test_reads_back_the_ass_filter_subtitles_builds(self, tmp_path):
        ass = tmp_path / "it's: a.ass"
        vf = subtitles.subtitles_filter(str(ass))
        path, fonts = gpu_burn.parse_ass(vf)
        assert path == str(ass)
        assert fonts.endswith("fonts")

    def test_srt_with_forced_style_is_not_taken(self, tmp_path):
        assert gpu_burn.parse_ass(subtitles.subtitles_filter(str(tmp_path / "a.srt"))) is None

    def test_anything_else_is_not_taken(self):
        for vf in ("", None, "ass=filename='a.ass'", "ass=filename='a.ass':fontsdir='f',eq=gamma=2",
                   "subtitles=filename='a.ass':fontsdir='f'"):
            assert gpu_burn.parse_ass(vf) is None

    def test_reads_the_mark_chain_and_position(self):
        chain, x, y = gpu_burn.parse_mark(_watermark, 1080, 1920)
        assert chain == "scale=324:-1,format=rgba,colorchannelmixer=aa=0.85"
        assert (x, y) == (54, 768)

    def test_the_real_mark_filter_is_taken(self):
        main = pytest.importorskip("main")
        assert gpu_burn.parse_mark(main.watermark_filter, 1080, 1920) is not None

    def test_a_mark_with_other_filters_is_not_taken(self):
        def other(vw, vh, video="[0:v]", logo="[1:v]", out=""):
            return f"{logo}scale=300:-1,hue=s=0[wm];{video}[wm]overlay=x=10:y=10{out}"

        def boom(*a, **k):
            raise ValueError
        assert gpu_burn.parse_mark(other, 1080, 1920) is None
        assert gpu_burn.parse_mark(boom, 1080, 1920) is None


class TestFallback:
    def test_off_without_the_card(self, monkeypatch, tmp_path):
        monkeypatch.setattr(gpu_render, "usable", lambda p: False)
        assert gpu_burn.burn("x.mp4", [("ass", str(tmp_path / "o.mp4"))], ass_vf="x") is False

    def test_off_when_disabled(self, monkeypatch, tmp_path):
        monkeypatch.setattr(gpu_render, "usable", lambda p: True)
        monkeypatch.setattr(gpu_burn, "ENABLED", False)
        assert gpu_burn.burn("x.mp4", [("hook", str(tmp_path / "o.mp4"))],
                             hook=("h.png", 0, 0, 5)) is False

    @pytest.fixture
    def card(self, monkeypatch):
        monkeypatch.setattr(gpu_render, "usable", lambda p: True)
        monkeypatch.setattr(gpu_burn, "ENABLED", True)
        monkeypatch.setattr(gpu_burn, "_probe", lambda p: (100, 4.0, None))
        monkeypatch.setattr(gpu_burn, "_planes", lambda *a: None)

    def test_a_caption_filter_it_cannot_read_burns_with_ffmpeg(self, card, monkeypatch, tmp_path):
        ran = []
        monkeypatch.setattr(subprocess, "run", lambda *a, **k: ran.append(a))
        assert gpu_burn.burn("x.mp4", [("ass", str(tmp_path / "o.mp4"))], ass_vf="subtitles=x") is False
        assert not ran

    def test_a_mark_output_needs_the_mark(self, card, tmp_path):
        assert gpu_burn.burn("x.mp4", [("mark", str(tmp_path / "o.mp4"))],
                             hook=("h.png", 0, 0, 5)) is False

    def test_an_unknown_stage_is_refused(self, card, tmp_path):
        assert gpu_burn.burn("x.mp4", [("blur", str(tmp_path / "o.mp4"))],
                             hook=("h.png", 0, 0, 5)) is False

    def _fake_run(self, rc, frames, outputs):
        def run(cmd, **k):
            if cmd[:2] == [gpu_render.BIN, "burn"]:
                for p in outputs:
                    with open(p, "wb") as f:
                        f.write(b"x")
                return subprocess.CompletedProcess(cmd, rc, "", f"osrender: {frames} frames\n")
            raise AssertionError(cmd)
        return run

    def test_a_short_burn_is_thrown_away(self, card, monkeypatch, tmp_path):
        out = str(tmp_path / "o.mp4")
        monkeypatch.setattr(subprocess, "run", self._fake_run(0, 99, [out]))
        assert gpu_burn.burn("x.mp4", [("hook", out)], hook=("h.png", 1, 2, 5)) is False
        assert not os.path.exists(out)

    def test_a_failed_burn_leaves_nothing(self, card, monkeypatch, tmp_path):
        outs = [str(tmp_path / "s.mp4"), str(tmp_path / "m.mp4")]
        monkeypatch.setattr(subprocess, "run", self._fake_run(1, 100, outs))
        vf = subtitles.subtitles_filter(str(tmp_path / "a.ass"))
        assert gpu_burn.burn("x.mp4", [("ass", outs[0]), ("mark", outs[1])], ass_vf=vf,
                             mark=("logo.png", _watermark), video_size=(1080, 1920)) is False
        assert not any(os.path.exists(p) for p in outs)

    def test_a_whole_burn_is_kept(self, card, monkeypatch, tmp_path):
        outs = [str(tmp_path / "h.mp4"), str(tmp_path / "s.mp4"), str(tmp_path / "m.mp4")]
        monkeypatch.setattr(subprocess, "run", self._fake_run(0, 100, outs))
        vf = subtitles.subtitles_filter(str(tmp_path / "a.ass"))
        assert gpu_burn.burn("x.mp4", list(zip(("hook", "ass", "mark"), outs)),
                             hook=("h.png", 69, 1085, 5.0), ass_vf=vf,
                             mark=("logo.png", _watermark), video_size=(1080, 1920)) is True
        assert all(os.path.exists(p) for p in outs)
