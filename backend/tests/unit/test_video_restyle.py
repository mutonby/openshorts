"""Tests for ml/video_restyle: fal.ai video-to-video call orchestration.

All network calls are mocked via patches on app.integrations.fal +
httpx.Client. We verify the model id, the argument shape, and that the
returned URL is downloaded to out_path.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from app.ml.video_restyle import MODEL_ID, restyle_video


def _make_stream_resp(payload: bytes):
    resp = MagicMock()
    resp.raise_for_status = MagicMock()
    resp.iter_bytes = MagicMock(return_value=iter([payload]))
    return resp


def test_restyle_video_calls_chosen_model(tmp_path):
    video = tmp_path / "in.mp4"
    video.write_bytes(b"\x00\x00\x00\x18ftypisom" + b"\x00" * 1000)
    ref = tmp_path / "ref.png"
    ref.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)
    out = tmp_path / "out.mp4"

    fake_submit = MagicMock(return_value={"video": {"url": "https://cdn.fal/out.mp4"}})
    fake_upload = MagicMock(side_effect=[
        "https://cdn.fal/in.mp4",
        "https://cdn.fal/ref.png",
    ])

    stream_ctx = MagicMock()
    stream_ctx.__enter__.return_value = _make_stream_resp(b"video-bytes")
    stream_ctx.__exit__.return_value = False
    client_ctx = MagicMock()
    client_ctx.__enter__.return_value.stream = MagicMock(return_value=stream_ctx)
    client_ctx.__exit__.return_value = False

    with patch("app.ml.video_restyle.submit_and_poll", fake_submit), \
         patch("app.ml.video_restyle.upload_file", fake_upload), \
         patch("app.ml.video_restyle.httpx.Client", return_value=client_ctx):
        result = restyle_video(
            api_key="fkey",
            video_path=str(video),
            reference_frame_path=str(ref),
            out_path=str(out),
        )

    assert result == str(out)
    assert out.exists()
    assert out.read_bytes() == b"video-bytes"

    fake_upload.assert_any_call(str(video), "fkey")
    fake_upload.assert_any_call(str(ref), "fkey")
    fake_submit.assert_called_once()
    call = fake_submit.call_args
    assert call.args[0] == MODEL_ID
    args = call.args[1]
    assert args["video_url"] == "https://cdn.fal/in.mp4"
    assert args["image_url"] == "https://cdn.fal/ref.png"
    assert "reference" in args["prompt"].lower() or "match" in args["prompt"].lower()


def test_restyle_video_missing_video(tmp_path):
    out = tmp_path / "out.mp4"
    with pytest.raises(FileNotFoundError, match="video"):
        restyle_video(
            api_key="x",
            video_path=str(tmp_path / "missing.mp4"),
            reference_frame_path=str(tmp_path / "ref.png"),
            out_path=str(out),
        )


def test_restyle_video_missing_reference(tmp_path):
    video = tmp_path / "in.mp4"
    video.write_bytes(b"fake")
    out = tmp_path / "out.mp4"
    with pytest.raises(FileNotFoundError, match="[Rr]eference"):
        restyle_video(
            api_key="x",
            video_path=str(video),
            reference_frame_path=str(tmp_path / "missing.png"),
            out_path=str(out),
        )


def test_restyle_video_raises_if_no_url_in_response(tmp_path):
    """Model response missing the video URL must surface as RuntimeError,
    not silently write an empty file."""
    video = tmp_path / "in.mp4"
    video.write_bytes(b"fake")
    ref = tmp_path / "ref.png"
    ref.write_bytes(b"png")
    out = tmp_path / "out.mp4"

    fake_submit = MagicMock(return_value={"unexpected": "shape"})
    fake_upload = MagicMock(side_effect=["u1", "u2"])

    with patch("app.ml.video_restyle.submit_and_poll", fake_submit), \
         patch("app.ml.video_restyle.upload_file", fake_upload):
        with pytest.raises(RuntimeError, match="no video URL"):
            restyle_video(
                api_key="x",
                video_path=str(video),
                reference_frame_path=str(ref),
                out_path=str(out),
            )


def test_restyle_video_creates_output_dir(tmp_path):
    video = tmp_path / "in.mp4"
    video.write_bytes(b"fake")
    ref = tmp_path / "ref.png"
    ref.write_bytes(b"png")
    out = tmp_path / "nested" / "dir" / "out.mp4"

    fake_submit = MagicMock(return_value={"video": {"url": "https://cdn/out.mp4"}})
    fake_upload = MagicMock(side_effect=["u1", "u2"])

    stream_ctx = MagicMock()
    stream_ctx.__enter__.return_value = _make_stream_resp(b"v")
    stream_ctx.__exit__.return_value = False
    client_ctx = MagicMock()
    client_ctx.__enter__.return_value.stream = MagicMock(return_value=stream_ctx)
    client_ctx.__exit__.return_value = False

    with patch("app.ml.video_restyle.submit_and_poll", fake_submit), \
         patch("app.ml.video_restyle.upload_file", fake_upload), \
         patch("app.ml.video_restyle.httpx.Client", return_value=client_ctx):
        result = restyle_video(
            api_key="x",
            video_path=str(video),
            reference_frame_path=str(ref),
            out_path=str(out),
        )

    assert result == str(out)
    assert out.exists()
