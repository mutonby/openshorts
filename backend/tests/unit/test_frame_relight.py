"""Tests for ml/frame_relight: Nano Banana relight call.

Gemini client is mocked — no network calls, no API costs in unit tests.
We verify the prompt template structure, the model name (matched against
the codebase's existing thumbnails/images.py choice), and the file-write
contract.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from app.ml.frame_relight import (
    MODEL_NAME,
    SAFETY_CONSTRAINTS,
    build_relight_prompt,
    relight_frame,
)


def test_model_name_matches_codebase():
    """We mirror the model id already used in thumbnails/images.py."""
    assert MODEL_NAME == "gemini-3.1-flash-image-preview"


def test_build_relight_prompt_contains_inputs():
    p = build_relight_prompt("bahamas beach", "golden hour")
    assert "bahamas beach" in p.lower()
    assert "golden hour" in p.lower()


def test_build_relight_prompt_contains_safety_constraints():
    p = build_relight_prompt("x", "y")
    for clause in SAFETY_CONSTRAINTS:
        assert clause in p


def test_relight_frame_writes_image(tmp_path):
    src = tmp_path / "src.png"
    src.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)
    out = tmp_path / "out.png"

    fake_client = MagicMock()
    fake_part = MagicMock()
    fake_part.inline_data = MagicMock(data=b"\x89PNG\r\n\x1a\n" + b"\x00" * 200)
    fake_resp = MagicMock()
    fake_resp.parts = [fake_part]
    fake_client.models.generate_content.return_value = fake_resp

    with patch("app.ml.frame_relight.genai.Client", return_value=fake_client):
        result = relight_frame(
            api_key="fake-key",
            frame_path=str(src),
            background_prompt="bahamas beach",
            lighting_prompt="golden hour",
            out_path=str(out),
        )

    assert result == str(out)
    assert out.exists()
    call = fake_client.models.generate_content.call_args
    assert call.kwargs["model"] == MODEL_NAME


def test_relight_frame_missing_input(tmp_path):
    out = tmp_path / "out.png"
    with pytest.raises(FileNotFoundError):
        relight_frame(
            api_key="x",
            frame_path=str(tmp_path / "missing.png"),
            background_prompt="x",
            lighting_prompt="y",
            out_path=str(out),
        )


def test_relight_frame_raises_when_no_image_returned(tmp_path):
    """Nano Banana sometimes returns text instead of an image (content
    policy refusal). Must raise, not silently write an empty file."""
    src = tmp_path / "src.png"
    src.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)
    out = tmp_path / "out.png"

    fake_client = MagicMock()
    fake_part = MagicMock(inline_data=None, text="sorry, can't comply")
    fake_resp = MagicMock()
    fake_resp.parts = [fake_part]
    fake_client.models.generate_content.return_value = fake_resp

    with patch("app.ml.frame_relight.genai.Client", return_value=fake_client):
        with pytest.raises(RuntimeError, match="no image"):
            relight_frame(
                api_key="x",
                frame_path=str(src),
                background_prompt="x",
                lighting_prompt="y",
                out_path=str(out),
            )


def test_relight_frame_creates_output_dir(tmp_path):
    src = tmp_path / "src.png"
    src.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)
    out = tmp_path / "nested" / "dir" / "out.png"

    fake_client = MagicMock()
    fake_part = MagicMock()
    fake_part.inline_data = MagicMock(data=b"\x89PNG\r\n\x1a\n" + b"\x00" * 50)
    fake_resp = MagicMock()
    fake_resp.parts = [fake_part]
    fake_client.models.generate_content.return_value = fake_resp

    with patch("app.ml.frame_relight.genai.Client", return_value=fake_client):
        result = relight_frame(
            api_key="x",
            frame_path=str(src),
            background_prompt="x",
            lighting_prompt="y",
            out_path=str(out),
        )

    assert result == str(out)
    assert out.exists()
