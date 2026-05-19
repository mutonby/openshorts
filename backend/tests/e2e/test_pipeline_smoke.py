"""
End-to-end pipeline smoke test.

Runs the real video-to-shorts pipeline against a tiny committed
fixture (tests/fixtures/smoke.mp4). Requires:
  - ffmpeg on PATH
  - The fixture file present (otherwise the test is SKIPPED)
  - All real Python deps installed (mediapipe, ultralytics, etc.)

This test is marked `e2e` and is excluded from the default `pytest -m
"not e2e"` run. It is the slowest test and the most production-faithful
one: it proves the pipeline still produces a valid vertical clip after
the restructure.

To run only this:
    pytest -m e2e
"""
import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
FIXTURE = REPO_ROOT / "tests" / "fixtures" / "smoke.mp4"


pytestmark = pytest.mark.e2e


def _ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None


def _ffprobe(path: Path) -> dict:
    out = subprocess.check_output([
        "ffprobe", "-v", "error",
        "-show_entries", "stream=width,height,codec_type:format=duration",
        "-of", "json", str(path),
    ])
    return json.loads(out)


@pytest.fixture
def real_modules_available():
    """Skip the e2e test if production deps aren't actually installed.

    The unit/api tests run with sys.modules stubs from conftest.py. The
    e2e test needs the REAL packages — if any are still stubs, bail.
    """
    import sys
    from unittest.mock import MagicMock

    required = ["cv2", "mediapipe", "ultralytics", "yt_dlp", "scenedetect"]
    for mod in required:
        installed = sys.modules.get(mod)
        if installed is None or isinstance(installed, MagicMock):
            pytest.skip(f"e2e needs real {mod} installed (got stub or missing)")


def test_pipeline_produces_vertical_clip(real_modules_available, tmp_path):
    if not _ffmpeg_available():
        pytest.skip("ffmpeg/ffprobe not on PATH")
    if not FIXTURE.exists():
        pytest.skip(
            f"missing fixture {FIXTURE} — see tests/fixtures/README.md for "
            "how to generate one"
        )

    # Run the existing main.py orchestrator directly. This is the same
    # entrypoint the FastAPI subprocess worker invokes for real jobs.
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    proc = subprocess.run(
        [
            "python", str(REPO_ROOT / "main.py"),
            "-i", str(FIXTURE),
            "-o", str(output_dir),
        ],
        capture_output=True, text=True, timeout=600,
        env={**os.environ, "PYTHONPATH": str(REPO_ROOT)},
    )
    assert proc.returncode == 0, (
        f"main.py exited non-zero.\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
    )

    # At least one clip MP4 was produced.
    clips = list(output_dir.glob("*_clip_*.mp4"))
    assert clips, f"no clips in {output_dir}\nSTDOUT:\n{proc.stdout}"

    # The first clip must be a valid MP4 that's vertical (9:16 ratio).
    probe = _ffprobe(clips[0])
    streams = probe.get("streams", [])
    video = next(s for s in streams if s.get("codec_type") == "video")
    w, h = video["width"], video["height"]
    aspect = w / h
    # 9/16 = 0.5625. Allow a 5% tolerance for encoder padding.
    assert 0.5 < aspect < 0.625, f"expected ~9:16, got {w}x{h} (aspect {aspect:.3f})"

    # Metadata JSON exists alongside the clips.
    meta_files = list(output_dir.glob("*_metadata.json"))
    assert meta_files, f"no metadata json produced in {output_dir}"
    meta = json.loads(meta_files[0].read_text(encoding="utf-8"))
    assert "shorts" in meta
    assert isinstance(meta["shorts"], list)
