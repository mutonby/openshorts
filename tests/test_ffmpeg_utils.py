import os
import subprocess
import sys

import pytest

import ffmpeg_utils
from ffmpeg_utils import (
    AI_DISCLOSURE,
    DELIVERY,
    INTERMEDIATE,
    METADATA_SCRUB,
    QUALITY,
    QUALITY_FAST,
    blurred_backdrop,
    mark_ai_generated,
    reset_encoder_cache,
    video_encode_args,
)


@pytest.fixture(autouse=True)
def _clean_encoder_state(monkeypatch):
    monkeypatch.delenv("FFMPEG_ENCODER", raising=False)
    monkeypatch.delenv("X264_PRESET", raising=False)
    monkeypatch.setattr(ffmpeg_utils, "_gpu_cut_sources", {})
    reset_encoder_cache()
    yield
    reset_encoder_cache()


def test_default_args_pin_historical_x264_settings():
    assert video_encode_args(QUALITY) == [
        "-c:v", "libx264", "-preset", "medium", "-crf", "18"]
    assert video_encode_args(QUALITY_FAST) == [
        "-c:v", "libx264", "-preset", "fast", "-crf", "18"]
    assert video_encode_args(DELIVERY) == [
        "-c:v", "libx264", "-preset", "fast", "-crf", "22"]


def test_a_deployment_can_pick_a_faster_x264_preset(monkeypatch):
    """A CPU-only host (the remote worker) trades a little size for CPU; the
    CRF stays, NVENC and an unknown preset name are untouched."""
    monkeypatch.setenv("X264_PRESET", "veryfast")
    assert video_encode_args(QUALITY) == [
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "18"]
    assert video_encode_args(DELIVERY)[3:] == ["veryfast", "-crf", "22"]
    assert ffmpeg_utils._X264_ARGS[QUALITY][3] == "medium"      # not mutated
    monkeypatch.setenv("X264_PRESET", "placebo; rm")
    assert video_encode_args(QUALITY)[3] == "medium"
    monkeypatch.setenv("X264_PRESET", "veryfast")
    monkeypatch.setenv("FFMPEG_ENCODER", "nvenc")
    monkeypatch.setattr(ffmpeg_utils, "_probe_nvenc", lambda: True)
    assert video_encode_args(QUALITY)[:4] == ["-c:v", "h264_nvenc", "-preset", "p3"]


def test_unknown_tier_raises():
    with pytest.raises(ValueError):
        video_encode_args("ultra")


def test_nvenc_mode_uses_nvenc_when_probe_passes(monkeypatch):
    monkeypatch.setenv("FFMPEG_ENCODER", "nvenc")
    monkeypatch.setattr(ffmpeg_utils, "_probe_nvenc", lambda: True)
    for tier in (QUALITY, QUALITY_FAST, DELIVERY):
        args = video_encode_args(tier)
        assert args[:2] == ["-c:v", "h264_nvenc"]
        assert "-cq" in args
        # Without an explicit yuv420p, RGB input makes nvenc emit GBR-space
        # H.264 that web players render with wrong colors.
        assert args[-2:] == ["-pix_fmt", "yuv420p"]


def test_nvenc_mode_falls_back_to_x264_when_probe_fails(monkeypatch):
    monkeypatch.setenv("FFMPEG_ENCODER", "nvenc")
    monkeypatch.setattr(ffmpeg_utils, "_probe_nvenc", lambda: False)
    assert video_encode_args(QUALITY)[:2] == ["-c:v", "libx264"]


def test_auto_probes_only_once(monkeypatch):
    calls = []

    def fake_probe():
        calls.append(1)
        return True

    monkeypatch.setenv("FFMPEG_ENCODER", "auto")
    monkeypatch.setattr(ffmpeg_utils, "_probe_nvenc", fake_probe)
    for _ in range(3):
        video_encode_args(QUALITY_FAST)
    assert len(calls) == 1


def test_missing_ffmpeg_binary_means_x264(monkeypatch):
    def raise_missing(*args, **kwargs):
        raise FileNotFoundError("ffmpeg")

    monkeypatch.setenv("FFMPEG_ENCODER", "auto")
    monkeypatch.setattr(subprocess, "run", raise_missing)
    assert video_encode_args(DELIVERY)[:2] == ["-c:v", "libx264"]


def test_returns_a_fresh_list_each_call():
    first = video_encode_args(QUALITY)
    first.append("-mutated")
    assert "-mutated" not in video_encode_args(QUALITY)


def test_metadata_scrub_covers_global_and_per_stream():
    # Global -map_metadata -1 alone leaves the audio handler_name intact on a
    # stream copy — the per-stream specifiers are what strip YouTube's
    # "produced by Google Inc." handler out of the published clip.
    assert METADATA_SCRUB[:2] == ["-map_metadata", "-1"]
    assert "-map_metadata:s:v" in METADATA_SCRUB
    assert "-map_metadata:s:a" in METADATA_SCRUB


def test_encode_args_stay_free_of_metadata_flags():
    # The scrub is spliced at call sites, not baked into the codec args — keep
    # video_encode_args a pure codec/quality list.
    for tier in (QUALITY, QUALITY_FAST, DELIVERY):
        assert not any(a.startswith("-map_metadata") for a in video_encode_args(tier))


def _tiny_mp4(path):
    subprocess.run(
        ["ffmpeg", "-y", "-f", "lavfi", "-i", "testsrc=size=320x240:rate=10:duration=1",
         "-f", "lavfi", "-i", "sine=frequency=440:duration=1",
         "-c:v", "libx264", "-c:a", "aac", "-pix_fmt", "yuv420p", str(path)],
        capture_output=True, check=True)


def _comment_tag(path):
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format_tags=comment",
         "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
        capture_output=True, text=True, check=True)
    return r.stdout.strip()


def test_ai_disclosure_survives_in_a_tag_a_machine_can_read():
    # AI Act art. 50(2) wants the marking readable by a machine, so it has to
    # come back out of the file. `comment` is the only tag mp4 keeps: a custom
    # key is dropped by ffmpeg without any warning.
    pytest.importorskip("shutil")
    import shutil, tempfile, os
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        pytest.skip("ffmpeg/ffprobe not installed")
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "clip.mp4")
        _tiny_mp4(p)
        assert mark_ai_generated(p, "AI voice dubbing") is True
        tag = _comment_tag(p)
        assert AI_DISCLOSURE in tag
        assert "AI voice dubbing" in tag


def test_ai_disclosure_never_destroys_the_file_it_cannot_tag():
    # The tag is a nicety; the video the user just paid minutes for is not.
    import tempfile, os
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "broken.mp4")
        with open(p, "wb") as f:
            f.write(b"not a video")
        assert mark_ai_generated(p) is False
        assert open(p, "rb").read() == b"not a video"
        assert not os.path.exists(p + ".aitag.mp4")


# --- cut_clip: a failed cut must say why, not hand an empty file downstream ---

class _FakeRun:
    """Stand-in for subprocess.run that writes what a real ffmpeg would."""

    def __init__(self, outcomes):
        self.outcomes = list(outcomes)   # (returncode, bytes_written, stderr)
        self.commands = []

    def __call__(self, command, **kwargs):
        self.commands.append(command)
        returncode, size, stderr = self.outcomes.pop(0)
        with open(command[-1], "wb") as fh:
            fh.write(b"\0" * size)
        return subprocess.CompletedProcess(command, returncode, None, stderr)


@pytest.fixture
def _cut(tmp_path, monkeypatch):
    """Run cut_clip against a scripted ffmpeg, on the GPU encoder, no waiting."""
    monkeypatch.setenv("FFMPEG_ENCODER", "auto")
    monkeypatch.setattr(ffmpeg_utils, "_probe_nvenc", lambda: True)
    slept = []
    monkeypatch.setattr(ffmpeg_utils.time, "sleep", slept.append)

    def run(outcomes):
        fake = _FakeRun(outcomes)
        monkeypatch.setattr(ffmpeg_utils.subprocess, "run", fake)
        ffmpeg_utils.cut_clip("source.mp4", str(tmp_path / "temp_clip.mp4"), 1.5, 12.0, 4)
        return fake

    run.slept = slept
    return run


def test_successful_cut_runs_once(_cut):
    fake = _cut([(0, 5_000_000, "")])
    assert len(fake.commands) == 1
    assert "h264_nvenc" in fake.commands[0]


def test_retry_stays_on_the_gpu_after_a_wait(_cut):
    """No libx264 fallback: the GPU failure is transient, so wait and ask again."""
    fake = _cut([(1, 0, "OpenEncodeSessionEx failed: out of memory"),
                 (0, 5_000_000, "")])
    assert len(fake.commands) == 2
    assert all("h264_nvenc" in cmd for cmd in fake.commands)
    assert "libx264" not in fake.commands[1]
    assert _cut.slept == [ffmpeg_utils.CUT_RETRY_WAITS[0]]


def test_exit_zero_with_an_empty_file_is_still_a_failure(_cut):
    """The prod symptom: ffmpeg wrote nothing, so the reframer met a file with
    no moov atom and blamed the temp clip instead of the encode."""
    fake = _cut([(0, 0, ""), (0, 5_000_000, "")])
    assert len(fake.commands) == 2


def test_every_attempt_failing_raises_with_ffmpeg_stderr(_cut):
    attempts = len(ffmpeg_utils.CUT_RETRY_WAITS) + 1
    with pytest.raises(RuntimeError, match="Invalid data found"):
        _cut([(1, 0, "Invalid data found when processing input")] * attempts)
    assert _cut.slept == list(ffmpeg_utils.CUT_RETRY_WAITS)


def test_backdrop_blurs_at_quarter_size_and_fills_the_frame():
    chain = blurred_backdrop(1080, 1920, 12)
    assert "scale=-2:480,crop=w=min(iw\\,270):h=480" in chain
    assert "gblur=sigma=3," in chain
    assert chain.endswith("scale=1080:1920")


def _gpu_source(monkeypatch, codec_pix):
    monkeypatch.setattr(ffmpeg_utils, "_gpu_cut_sources", {})
    monkeypatch.setattr(ffmpeg_utils, "_source_format",
                        lambda path: tuple(codec_pix.split(",")))


def test_cut_stays_on_the_gpu_for_8bit_420(_cut, monkeypatch):
    _gpu_source(monkeypatch, "h264,yuv420p")
    cmd = _cut([(0, 5_000_000, "")]).commands[0]
    assert cmd[cmd.index("-hwaccel") + 1] == "cuda"
    assert cmd[cmd.index("-hwaccel_output_format") + 1] == "cuda"
    assert "-pix_fmt" not in cmd
    assert "h264_nvenc" in cmd


def test_cut_decodes_10bit_sources_on_the_cpu(_cut, monkeypatch):
    _gpu_source(monkeypatch, "hevc,yuv420p10le")
    cmd = _cut([(0, 5_000_000, "")]).commands[0]
    assert "-hwaccel" not in cmd
    assert cmd[cmd.index("-pix_fmt") + 1] == "yuv420p"


def test_failed_gpu_cut_falls_back_to_the_cpu_decode(_cut, monkeypatch):
    _gpu_source(monkeypatch, "av1,yuv420p")
    fake = _cut([(1, 0, "cuvid error"), (0, 5_000_000, "")])
    assert "-hwaccel" in fake.commands[0]
    assert "-hwaccel" not in fake.commands[1]
    assert len(fake.commands) == 2


# --- INTERMEDIATE: the cut is decoded again at once, so NVENC spends bits, not effort ---

def test_intermediate_keeps_the_cpu_settings_of_quality_fast():
    assert video_encode_args(INTERMEDIATE) == video_encode_args(QUALITY_FAST)


def test_intermediate_on_nvenc_is_the_cheap_preset(monkeypatch):
    monkeypatch.setenv("FFMPEG_ENCODER", "nvenc")
    monkeypatch.setattr(ffmpeg_utils, "_probe_nvenc", lambda: True)
    args = video_encode_args(INTERMEDIATE)
    assert args[:2] == ["-c:v", "h264_nvenc"]
    assert args[args.index("-preset") + 1] == "p1"
    assert "-spatial-aq" not in args and "-tune" not in args
    assert args[-2:] == ["-pix_fmt", "yuv420p"]
    monkeypatch.setenv("CUT_NVENC_PRESET", "p3")
    monkeypatch.setenv("CUT_NVENC_CQ", "17")
    args = video_encode_args(INTERMEDIATE)
    assert args[args.index("-preset") + 1] == "p3"
    assert args[args.index("-cq") + 1] == "17"


def test_cut_uses_the_intermediate_tier(_cut):
    cmd = _cut([(0, 5_000_000, "")]).commands[0]
    assert cmd[cmd.index("-preset") + 1] == "p1"


# --- bin/ffmpeg: the thread cap every ffmpeg of the image goes through ---

WRAPPER = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "bin", "ffmpeg.py" if os.name == "nt" else "ffmpeg")


def _wrapped(tmp_path, threads, *args):
    echo_name = "echo-ffmpeg.py" if os.name == "nt" else "echo-ffmpeg"
    echo = tmp_path / echo_name
    if os.name == "nt":
        echo.write_text('import sys\nfor a in sys.argv[1:]:\n    print(a)\n')
    else:
        echo.write_text('#!/bin/sh\nfor a in "$@"; do printf \'%s\\n\' "$a"; done\n')
    echo.chmod(0o755)
    env = {"PATH": os.environ.get("PATH", ""), "FFMPEG_REAL_BIN": str(echo)}
    if threads is not None:
        env["FFMPEG_THREADS"] = threads
    cmd = [sys.executable, WRAPPER, *args] if os.name == "nt" else ["sh", WRAPPER, *args]
    out = subprocess.run(cmd, env=env, capture_output=True,
                         text=True, check=True).stdout
    return out.splitlines()


def test_wrapper_caps_decoder_and_filter_threads(tmp_path):
    argv = _wrapped(tmp_path, "3", "-y", "-ss", "1", "-i", "a b.mp4", "-i", "logo.png",
                    "-filter_complex", "[0:v][1:v]overlay", "-c:v", "h264_nvenc",
                    "-threads", "8", "out.mp4")
    assert argv == ["-filter_threads", "3", "-filter_complex_threads", "3",
                    "-y", "-ss", "1", "-threads", "3", "-i", "a b.mp4",
                    "-threads", "3", "-i", "logo.png",
                    "-filter_complex", "[0:v][1:v]overlay", "-c:v", "h264_nvenc",
                    "-threads", "8", "out.mp4"]


def test_wrapper_keeps_what_the_caller_set(tmp_path):
    argv = _wrapped(tmp_path, "2", "-threads", "5", "-i", "a.mp4",
                    "-filter_threads", "1", "-i", "b.mp4", "out.mp4")
    assert argv == ["-filter_complex_threads", "2", "-threads", "5", "-i", "a.mp4",
                    "-filter_threads", "1", "-threads", "2", "-i", "b.mp4", "out.mp4"]


@pytest.mark.parametrize("threads", [None, "", "0", "auto"])
def test_wrapper_is_a_passthrough_when_unset(tmp_path, threads):
    assert _wrapped(tmp_path, threads, "-i", "a.mp4", "x y.mp4") == ["-i", "a.mp4", "x y.mp4"]
