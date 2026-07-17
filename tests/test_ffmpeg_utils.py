import subprocess

import pytest

import ffmpeg_utils
from ffmpeg_utils import (
    DELIVERY,
    QUALITY,
    QUALITY_FAST,
    reset_encoder_cache,
    video_encode_args,
)


@pytest.fixture(autouse=True)
def _clean_encoder_state(monkeypatch):
    monkeypatch.delenv("FFMPEG_ENCODER", raising=False)
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


def test_unknown_tier_raises():
    with pytest.raises(ValueError):
        video_encode_args("ultra")


def test_nvenc_mode_uses_nvenc_when_probe_passes(monkeypatch):
    monkeypatch.setenv("FFMPEG_ENCODER", "nvenc")
    monkeypatch.setattr(ffmpeg_utils, "_probe_nvenc", lambda: True)
    args = video_encode_args(QUALITY)
    assert args[:2] == ["-c:v", "h264_nvenc"]
    assert "-cq" in args


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
