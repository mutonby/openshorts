"""Free-plan jobs cut a high-frame-rate source at a capped rate, so every pass
after the cut reads half the frames; paid jobs keep the source's rate."""
from fractions import Fraction

import pytest

import ffmpeg_utils
import remote_worker


@pytest.fixture(autouse=True)
def _rates(monkeypatch):
    monkeypatch.setattr(ffmpeg_utils, "_source_rates", {})
    monkeypatch.delenv("CUT_MAX_FPS", raising=False)


def _source_at(monkeypatch, rate):
    monkeypatch.setattr(ffmpeg_utils, "_source_rate", lambda path: rate)


@pytest.mark.parametrize("rate, expected", [
    (Fraction(60), "fps=30/1"),
    (Fraction(60000, 1001), "fps=30000/1001"),
    (Fraction(50), "fps=25/1"),
    (Fraction(120), "fps=30/1"),
    (Fraction(48), "fps=24/1"),
    (Fraction(240), "fps=30/1"),
])
def test_a_fast_source_is_cut_at_an_even_fraction_of_its_rate(monkeypatch, rate, expected):
    monkeypatch.setenv("CUT_MAX_FPS", "30")
    _source_at(monkeypatch, rate)
    assert ffmpeg_utils.cut_rate_filter("v.mp4") == ["-vf", expected]


@pytest.mark.parametrize("rate", [Fraction(30), Fraction(30000, 1001), Fraction(25),
                                  Fraction(24), Fraction(40), None])
def test_a_source_at_or_near_the_cap_is_left_alone(monkeypatch, rate):
    monkeypatch.setenv("CUT_MAX_FPS", "30")
    _source_at(monkeypatch, rate)
    assert ffmpeg_utils.cut_rate_filter("v.mp4") == []


@pytest.mark.parametrize("value", [None, "0", "", "x"])
def test_no_cap_means_the_source_rate(monkeypatch, value):
    if value is not None:
        monkeypatch.setenv("CUT_MAX_FPS", value)
    _source_at(monkeypatch, Fraction(60))
    assert ffmpeg_utils.cut_rate_filter("v.mp4") == []


def test_the_cap_reaches_both_cut_commands(monkeypatch, tmp_path):
    monkeypatch.setenv("CUT_MAX_FPS", "30")
    _source_at(monkeypatch, Fraction(60))
    monkeypatch.setattr(ffmpeg_utils, "_gpu_cut_ok", lambda path: True)
    seen = []

    class _Done:
        returncode = 0
        stderr = ""
    out = tmp_path / "temp_c.mp4"

    def _run(cmd, **kw):
        seen.append(cmd)
        out.write_bytes(b"x" * (ffmpeg_utils.MIN_CUT_BYTES + 1))
        return _Done()
    monkeypatch.setattr(ffmpeg_utils.subprocess, "run", _run)
    ffmpeg_utils.cut_clip("v.mp4", str(out), 1, 5, 1)
    cmd = seen[0]
    assert cmd[cmd.index("-vf") + 1] == "fps=30/1"
    assert cmd.index("-vf") > cmd.index("-i")          # an output option


def test_only_free_jobs_get_the_cap(monkeypatch):
    import app
    monkeypatch.delenv("FREE_MAX_FPS", raising=False)
    env = {"WATERMARK": "1"}
    app._apply_rate_cap(env)
    assert env["CUT_MAX_FPS"] == "30"
    paid = {"CUT_MAX_FPS": "30"}                      # e.g. a resumed job that paid
    app._apply_rate_cap(paid)
    assert "CUT_MAX_FPS" not in paid
    monkeypatch.setenv("FREE_MAX_FPS", "0")
    env = {"WATERMARK": "1"}
    app._apply_rate_cap(env)
    assert "CUT_MAX_FPS" not in env


def test_the_cap_travels_to_the_remote_worker():
    assert remote_worker.allowed_option("CUT_MAX_FPS", "30")
