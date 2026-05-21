"""Regression test for the FFmpeg wrapper invariant.

Codex Phase 5 audit (focus 2) confirmed that several modules called
``subprocess.run(['ffmpeg', ...])`` directly instead of going through
``app.video.ffmpeg``. The wrapper centralizes:

* default timeouts (Phase 5 B-2 fix)
* UTF-8 locale setup
* uniform FFmpegError surfacing
* future logging / progress / audit hooks

Phase 5 B-1 migrated the three Codex BLOCKERs:

* ``app/editing/ai_filters.py``
* ``app/overlays/subtitles_render.py``
* ``app/overlays/hooks.py``

This test pins those migrations. New code in those packages must not
re-introduce direct ``subprocess`` calls referencing the ``ffmpeg``
or ``ffprobe`` binaries.

Out of scope (documented deferred):

* ``app/video/pipeline.py`` — per-frame Popen with stdin streaming;
  needs a wrapper redesign before migration.
* ``app/cli.py`` — legacy CLI entrypoint, used by ``/api/process``
  subprocess fan-out.
* ``app/saas/pipeline.py`` — separate product line.
* ``app/main.py``  — has one ffprobe probe in /api/effects/generate
  pending a wrapper ``probe_metadata`` helper.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
PROTECTED_FILES = [
    REPO_ROOT / "app" / "editing" / "ai_filters.py",
    REPO_ROOT / "app" / "overlays" / "subtitles_render.py",
    REPO_ROOT / "app" / "overlays" / "hooks.py",
]

# Matches `subprocess.<anything>(... 'ffmpeg' or 'ffprobe' ...)` on a
# single source line. False-positives are fine — the protected files
# should have ZERO ffmpeg/ffprobe references via subprocess.*.
SUBPROCESS_FFMPEG_PATTERN = re.compile(
    r"subprocess\.\w+\([^)]*['\"](?:ffmpeg|ffprobe)['\"]",
    re.MULTILINE,
)


@pytest.mark.parametrize("path", PROTECTED_FILES, ids=lambda p: p.name)
def test_no_direct_ffmpeg_subprocess_in_protected_file(path):
    """The file MUST NOT call subprocess.* with ffmpeg/ffprobe as a literal arg."""
    assert path.exists(), f"protected file missing: {path}"
    src = path.read_text()
    matches = SUBPROCESS_FFMPEG_PATTERN.findall(src)
    assert not matches, (
        f"{path.name} has a direct ffmpeg/ffprobe subprocess call. "
        "Migrate it to app.video.ffmpeg (ffmpeg_wrapper.run / .probe_resolution / "
        "etc.) per project convention."
    )


@pytest.mark.parametrize("path", PROTECTED_FILES, ids=lambda p: p.name)
def test_protected_file_imports_wrapper(path):
    """Every protected file must import ``ffmpeg_wrapper`` from app.video."""
    src = path.read_text()
    assert "from app.video import ffmpeg as ffmpeg_wrapper" in src, (
        f"{path.name} must import the wrapper as ffmpeg_wrapper for uniform usage."
    )
