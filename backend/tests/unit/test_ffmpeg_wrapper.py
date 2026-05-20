"""Tests for app.video.ffmpeg wrapper default-timeout safety.

Codex Phase 5 audit flagged that wrapper timeout defaulted to None,
allowing any state-mutating FFmpeg call to run forever on a hostile or
corrupt input (deferred C4 exploit). The wrapper now:

  * Applies DEFAULT_TIMEOUT (configurable via FFMPEG_TIMEOUT_SECONDS env)
    when callers don't pass an explicit timeout.
  * Wraps subprocess.TimeoutExpired into FFmpegError so callers see a
    single exception type.
  * Applies DEFAULT_PROBE_TIMEOUT to probe_resolution / probe_duration.

These tests verify the contract by patching subprocess.run.
"""
from __future__ import annotations

import subprocess
from unittest.mock import MagicMock, patch

import pytest

from app.video import ffmpeg as ffmpeg_wrapper
from app.video.ffmpeg import FFmpegError


def _fake_completed(returncode=0, stderr=b""):
    cp = MagicMock(spec=subprocess.CompletedProcess)
    cp.returncode = returncode
    cp.stderr = stderr
    cp.stdout = b""
    return cp


def test_run_applies_default_timeout_when_none_passed():
    with patch("app.video.ffmpeg.subprocess.run", return_value=_fake_completed()) as srun:
        ffmpeg_wrapper.run(["-y", "-i", "a.mp4", "out.mp4"])
    assert srun.call_count == 1
    kwargs = srun.call_args.kwargs
    assert kwargs["timeout"] is not None
    assert kwargs["timeout"] == ffmpeg_wrapper.DEFAULT_TIMEOUT


def test_run_honors_explicit_timeout():
    with patch("app.video.ffmpeg.subprocess.run", return_value=_fake_completed()) as srun:
        ffmpeg_wrapper.run(["-y", "-i", "a.mp4", "out.mp4"], timeout=12.5)
    assert srun.call_args.kwargs["timeout"] == 12.5


def test_run_wraps_timeout_into_ffmpeg_error():
    def fake_run(*args, **_kwargs):
        raise subprocess.TimeoutExpired(cmd=args[0], timeout=1.0)

    with patch("app.video.ffmpeg.subprocess.run", side_effect=fake_run):
        with pytest.raises(FFmpegError) as exc:
            ffmpeg_wrapper.run(["-y", "-i", "a.mp4", "out.mp4"], timeout=1.0)
    assert "timeout" in str(exc.value).lower()
    # Returncode -1 by convention signals a timeout-induced failure.
    assert exc.value.returncode == -1


def test_probe_resolution_has_default_probe_timeout():
    cp = _fake_completed()
    cp.stdout = b"1920x1080\n"
    with patch("app.video.ffmpeg.subprocess.run", return_value=cp) as srun:
        ffmpeg_wrapper.probe_resolution("/tmp/v.mp4")
    assert srun.call_args.kwargs.get("timeout") == ffmpeg_wrapper.DEFAULT_PROBE_TIMEOUT


def test_probe_duration_has_default_probe_timeout():
    cp = _fake_completed()
    cp.stdout = b"42.5\n"
    with patch("app.video.ffmpeg.subprocess.run", return_value=cp) as srun:
        ffmpeg_wrapper.probe_duration("/tmp/v.mp4")
    assert srun.call_args.kwargs.get("timeout") == ffmpeg_wrapper.DEFAULT_PROBE_TIMEOUT


def test_default_timeout_is_finite_and_positive():
    assert ffmpeg_wrapper.DEFAULT_TIMEOUT is not None
    assert ffmpeg_wrapper.DEFAULT_TIMEOUT > 0
    assert ffmpeg_wrapper.DEFAULT_PROBE_TIMEOUT > 0
    # Probe should be quicker than full ffmpeg.
    assert ffmpeg_wrapper.DEFAULT_PROBE_TIMEOUT < ffmpeg_wrapper.DEFAULT_TIMEOUT
