"""A client file name over the filesystem's 255-byte limit must not 500 the upload."""
import os

import pytest

app_module = pytest.importorskip("app")

LONG = "🎮 " + "𝙏𝙝𝙚 𝙁𝙞𝙧𝙨𝙩 " * 12 + "🔥💀 sample, ¡clip!.mp4"


def test_long_name_is_clamped_in_bytes_and_keeps_extension():
    assert len(LONG.encode("utf-8")) > 255 - 40
    out = app_module._clamp_upload_name(LONG)
    assert len(out.encode("utf-8")) <= app_module.MAX_UPLOAD_NAME_BYTES
    assert out.endswith(".mp4")
    out.encode("utf-8")  # no half characters


def test_short_name_is_untouched():
    assert app_module._clamp_upload_name("clip ñ.mp4") == "clip ñ.mp4"


def test_clamped_name_is_a_valid_file(tmp_path):
    name = app_module._clamp_upload_name(LONG)
    path = tmp_path / f"pending_{'x' * 36}_{name}"
    path.write_bytes(b"ok")
    assert path.exists()


def test_name_without_extension_and_odd_extension():
    out = app_module._clamp_upload_name("é" * 300)
    assert len(out.encode()) <= app_module.MAX_UPLOAD_NAME_BYTES
    out = app_module._clamp_upload_name("a" * 200 + "." + "b" * 100)
    assert len(out.encode()) <= app_module.MAX_UPLOAD_NAME_BYTES
