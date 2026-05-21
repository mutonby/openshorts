"""Tests for the Phase 4 merge helper.

Covers the public surface of ``app.video.merge.concat_clips``: input
validation, filter-graph composition, output path derivation, the FFmpeg
invocation contract (only via the wrapper), and the atomic-rename safety
net that prevents concurrent-identical-merge clobbering.
"""
from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

import pytest

from app.video.merge import (
    NORMALIZE_FILTER,
    build_concat_args,
    concat_clips,
)


def test_normalize_filter_targets_1080x1920_30fps():
    # Output normalized to 9:16 1080x1920 @ 30fps so concat can succeed
    # regardless of source resolution / fps.
    assert "scale=1080:1920" in NORMALIZE_FILTER
    assert "fps=30" in NORMALIZE_FILTER
    assert "setsar=1" in NORMALIZE_FILTER
    assert "format=yuv420p" in NORMALIZE_FILTER


def test_build_concat_args_two_inputs(tmp_path):
    a = tmp_path / "a.mp4"
    b = tmp_path / "b.mp4"
    a.write_bytes(b"x")
    b.write_bytes(b"x")
    out = tmp_path / "merged.mp4"

    args = build_concat_args([str(a), str(b)], str(out))

    # Both inputs declared with -i.
    assert args.count("-i") == 2
    # Filter graph references the normalize chain once per input, then concat.
    fc_index = args.index("-filter_complex")
    fc = args[fc_index + 1]
    assert "[0:v]" in fc and "[1:v]" in fc
    assert "[0:a]" in fc and "[1:a]" in fc
    assert "concat=n=2:v=1:a=1" in fc
    # Audio re-encoded to AAC 48 kHz stereo for clean concat.
    assert "-c:a" in args and args[args.index("-c:a") + 1] == "aac"
    assert "-ar" in args and args[args.index("-ar") + 1] == "48000"
    assert "-ac" in args and args[args.index("-ac") + 1] == "2"
    # Video re-encoded with libx264.
    assert "-c:v" in args and args[args.index("-c:v") + 1] == "libx264"
    # Output path is the last positional argument.
    assert args[-1] == str(out)


def test_build_concat_args_three_inputs_concat_n_matches(tmp_path):
    paths = []
    for name in ("a.mp4", "b.mp4", "c.mp4"):
        p = tmp_path / name
        p.write_bytes(b"x")
        paths.append(str(p))
    out = tmp_path / "merged.mp4"

    args = build_concat_args(paths, str(out))
    fc = args[args.index("-filter_complex") + 1]
    assert "concat=n=3:v=1:a=1" in fc
    for i in range(3):
        assert f"[{i}:v]" in fc
        assert f"[{i}:a]" in fc


def test_concat_clips_rejects_empty_list(tmp_path):
    with pytest.raises(ValueError) as exc:
        concat_clips([], str(tmp_path / "out.mp4"))
    assert "at least" in str(exc.value).lower()


def test_concat_clips_rejects_single_input(tmp_path):
    a = tmp_path / "a.mp4"
    a.write_bytes(b"x")
    with pytest.raises(ValueError) as exc:
        concat_clips([str(a)], str(tmp_path / "out.mp4"))
    assert "at least 2" in str(exc.value).lower()


def test_concat_clips_rejects_missing_input(tmp_path):
    a = tmp_path / "a.mp4"
    a.write_bytes(b"x")
    missing = tmp_path / "ghost.mp4"
    with pytest.raises(FileNotFoundError):
        concat_clips([str(a), str(missing)], str(tmp_path / "out.mp4"))


def test_concat_clips_invokes_ffmpeg_with_expected_filter(tmp_path):
    a = tmp_path / "a.mp4"
    b = tmp_path / "b.mp4"
    a.write_bytes(b"x")
    b.write_bytes(b"x")
    out = tmp_path / "merged.mp4"

    def fake_run(args, **_kwargs):
        # ffmpeg's positional out path is the last argv element; atomic-rename
        # passes a `.partial-*.mp4` path, not the public `out`.
        actual_out = args[-1]
        Path(actual_out).write_bytes(b"merged")
        return None

    with patch("app.video.merge.ffmpeg_wrapper.run", side_effect=fake_run) as run_mock:
        result = concat_clips([str(a), str(b)], str(out))

    assert result == str(out)
    args = run_mock.call_args.args[0]
    fc = args[args.index("-filter_complex") + 1]
    assert "concat=n=2:v=1:a=1" in fc


def test_concat_clips_raises_when_ffmpeg_produces_empty_output(tmp_path):
    a = tmp_path / "a.mp4"
    b = tmp_path / "b.mp4"
    a.write_bytes(b"x")
    b.write_bytes(b"x")
    out = tmp_path / "merged.mp4"

    with patch("app.video.merge.ffmpeg_wrapper.run") as run_mock:
        run_mock.return_value = None
        with pytest.raises(RuntimeError) as exc:
            concat_clips([str(a), str(b)], str(out))
        assert "empty output" in str(exc.value)


# ---------------------------------------------------------------------------
# Atomic-rename safety net (Phase 5 fix for concurrent-identical-merge race).
# ---------------------------------------------------------------------------

def test_concat_clips_writes_to_partial_then_renames(tmp_path):
    """ffmpeg writes to a `.partial-*.mp4` path; final `out` appears via rename.

    Prevents partial-read races when two clients POST the same indices: each
    merge writes to a unique partial path, then atomic-renames to the stable
    public URL. The reader either sees the old file or the new file, never
    a mid-write file.
    """
    a = tmp_path / "a.mp4"
    b = tmp_path / "b.mp4"
    a.write_bytes(b"x")
    b.write_bytes(b"x")
    out = tmp_path / "merged.mp4"
    seen_partial_paths: list[str] = []

    def fake_run(args, **_kwargs):
        actual_out = args[-1]
        seen_partial_paths.append(actual_out)
        # Final public path must not exist during ffmpeg run.
        assert not out.exists(), "final path was written to before rename"
        assert ".partial-" in actual_out, f"expected partial path, got {actual_out}"
        Path(actual_out).write_bytes(b"merged")
        return None

    with patch("app.video.merge.ffmpeg_wrapper.run", side_effect=fake_run):
        result = concat_clips([str(a), str(b)], str(out))

    assert result == str(out)
    # After concat_clips returns, public path exists, partial does not.
    assert out.exists()
    assert out.read_bytes() == b"merged"
    for partial in seen_partial_paths:
        assert not os.path.exists(partial), f"partial {partial} not cleaned up"


def test_concat_clips_cleans_up_partial_on_ffmpeg_failure(tmp_path):
    """If ffmpeg raises, the partial file is removed and `out` is not touched."""
    from app.video.ffmpeg import FFmpegError
    a = tmp_path / "a.mp4"
    b = tmp_path / "b.mp4"
    a.write_bytes(b"x")
    b.write_bytes(b"x")
    out = tmp_path / "merged.mp4"
    out.write_bytes(b"existing-stable-output")  # pretend a prior merge succeeded
    seen_partial_paths: list[str] = []

    def fake_run(args, **_kwargs):
        actual_out = args[-1]
        seen_partial_paths.append(actual_out)
        Path(actual_out).write_bytes(b"corrupt-mid-write")
        raise FFmpegError(1, b"simulated ffmpeg crash", args)

    with patch("app.video.merge.ffmpeg_wrapper.run", side_effect=fake_run):
        with pytest.raises(FFmpegError):
            concat_clips([str(a), str(b)], str(out))

    # Stable output preserved; corrupt partial cleaned up.
    assert out.read_bytes() == b"existing-stable-output"
    for partial in seen_partial_paths:
        assert not os.path.exists(partial), f"partial {partial} not cleaned up"


def test_concat_clips_partial_paths_are_unique_across_calls(tmp_path):
    """Two back-to-back merges with the same final path use different partials.

    Simulates the concurrent-merge case: with unique nonces, neither writer
    clobbers the other mid-flight.
    """
    a = tmp_path / "a.mp4"
    b = tmp_path / "b.mp4"
    a.write_bytes(b"x")
    b.write_bytes(b"x")
    out = tmp_path / "merged.mp4"
    seen_partials: list[str] = []

    def fake_run(args, **_kwargs):
        actual_out = args[-1]
        seen_partials.append(actual_out)
        Path(actual_out).write_bytes(b"merged")
        return None

    with patch("app.video.merge.ffmpeg_wrapper.run", side_effect=fake_run):
        concat_clips([str(a), str(b)], str(out))
        concat_clips([str(a), str(b)], str(out))

    assert len(seen_partials) == 2
    assert seen_partials[0] != seen_partials[1], "partial paths must be unique"
