"""Tests for the AI-filter safety allowlist.

Codex Phase 5 audit (focus 1, BLOCKER) found that ``/api/edit`` and
``auto_pipeline.apply_ai_edit`` executed LLM-produced ``filter_string``
through FFmpeg ``-vf`` with only a regex-comparison-cleanup pass. A
malicious Gemini response like ``movie=/etc/passwd,scale=1:1`` would
exfiltrate / probe filesystem state.

These tests pin a strict allowlist of FFmpeg filters allowed in
LLM-generated content. The filter parser strips bracket labels
(``[0:v]``...``[v0]``), splits the chain on commas + semicolons, and
matches each node's leading filter name against the allowlist.
"""
from __future__ import annotations

import pytest

from app.utils.filters import (
    UnsafeFilterError,
    validate_filter_string,
)


# ---- happy paths ----------------------------------------------------------

@pytest.mark.parametrize(
    "good",
    [
        "zoompan=z='1.2':d=1:s=1080x1920:fps=30",
        "eq=contrast=1.2:enable='between(t,0,3)'",
        "hue=s=0:enable='between(t,10,12)'",
        "unsharp=5:5:1.0",
        "curves=preset=darker",
        "zoompan=z='1.1':d=1:s=1080x1920,eq=contrast=1.2",  # chain comma
        # Bracket labels (filter_complex style) — parser must strip them.
        "[0:v]zoompan=z='1.2':d=1:s=1080x1920[v0]",
        "scale=1080:1920,setsar=1,fps=30,format=yuv420p",
        # Whitespace tolerance
        " eq = contrast = 1.2 ",
        # Empty string is a no-op (no filter to apply) — allowed.
        "",
    ],
)
def test_allowed_filters_pass(good):
    validate_filter_string(good)  # no raise


# ---- the actual attack from Codex reproducer ------------------------------

def test_blocks_movie_filter_with_arbitrary_path():
    """The exact attack Codex flagged: LLM returns a `movie=` node."""
    with pytest.raises(UnsafeFilterError) as exc:
        validate_filter_string("movie=/etc/passwd,scale=1:1")
    assert "movie" in str(exc.value).lower()


def test_blocks_amovie_filter():
    with pytest.raises(UnsafeFilterError):
        validate_filter_string("amovie=/etc/passwd")


def test_blocks_subtitles_filter_for_arbitrary_path_read():
    with pytest.raises(UnsafeFilterError):
        validate_filter_string("subtitles=/etc/shadow")


def test_blocks_ass_filter():
    with pytest.raises(UnsafeFilterError):
        validate_filter_string("ass=/etc/secrets.ass")


def test_blocks_concat_filter():
    """concat can read arbitrary files when used as `concat=...:f=path`."""
    with pytest.raises(UnsafeFilterError):
        validate_filter_string("concat=n=2:v=1:a=1")


# ---- chain-level attacks --------------------------------------------------

def test_blocks_when_evil_filter_after_legit_one():
    """LLM tries to slip an unsafe filter after a legit one."""
    with pytest.raises(UnsafeFilterError):
        validate_filter_string("eq=contrast=1.2,movie=/etc/passwd")


def test_blocks_when_evil_filter_after_semicolon():
    """filter_complex chains-of-chains use `;`."""
    with pytest.raises(UnsafeFilterError):
        validate_filter_string("zoompan=z='1.2':d=1;movie=/etc/passwd")


def test_blocks_unknown_filter_not_in_allowlist():
    """Even non-malicious unknown filters fail closed."""
    with pytest.raises(UnsafeFilterError):
        validate_filter_string("brand_new_filter_that_does_not_exist=1")


def test_rejects_non_string_input():
    with pytest.raises(TypeError):
        validate_filter_string(123)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        validate_filter_string(None)  # type: ignore[arg-type]


def test_validate_returns_none_on_success():
    # Pure validator: no return value, just raises on bad input.
    assert validate_filter_string("eq=contrast=1.2") is None


def test_error_message_names_offending_filter():
    """Operators triaging an error need to know WHICH filter was blocked."""
    with pytest.raises(UnsafeFilterError) as exc:
        validate_filter_string("eq=contrast=1.2,movie=/etc/passwd")
    msg = str(exc.value).lower()
    assert "movie" in msg
