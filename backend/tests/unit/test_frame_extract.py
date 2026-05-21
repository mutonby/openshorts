"""Tests for ml/frame_extract: extract first frame of a video to PNG."""
from __future__ import annotations

from pathlib import Path

import pytest

from app.ml.frame_extract import extract_first_frame


FIXTURE = Path(__file__).resolve().parent.parent / "fixtures" / "short-clip.mp4"


@pytest.fixture(scope="module", autouse=True)
def _ensure_fixture():
    """Generate a tiny synthetic clip with ffmpeg testsrc so the test runs
    anywhere ffmpeg is available — no dependency on a binary in git or on a
    host-side demo file leaking into the container.
    """
    if FIXTURE.exists():
        return
    FIXTURE.parent.mkdir(parents=True, exist_ok=True)
    import subprocess
    try:
        subprocess.run(
            [
                "ffmpeg", "-y",
                "-f", "lavfi", "-i", "testsrc=duration=2:size=320x240:rate=10",
                "-c:v", "libx264", "-pix_fmt", "yuv420p",
                str(FIXTURE),
            ],
            check=True, capture_output=True,
        )
    except FileNotFoundError:
        pytest.skip("ffmpeg not on PATH")


def test_extract_first_frame_writes_png(tmp_path):
    out = tmp_path / "frame.png"
    result = extract_first_frame(str(FIXTURE), str(out))
    assert result == str(out)
    assert out.exists()
    assert out.stat().st_size > 1000  # not empty


def test_extract_first_frame_missing_input(tmp_path):
    out = tmp_path / "frame.png"
    with pytest.raises(FileNotFoundError):
        extract_first_frame(str(tmp_path / "does-not-exist.mp4"), str(out))


def test_extract_first_frame_creates_output_dir(tmp_path):
    """Caller should not need to mkdir the destination; helper does it."""
    out = tmp_path / "nested" / "dir" / "frame.png"
    result = extract_first_frame(str(FIXTURE), str(out))
    assert result == str(out)
    assert out.exists()
