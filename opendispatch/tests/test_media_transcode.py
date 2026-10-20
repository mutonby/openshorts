"""Media transcoding tests — Pillow only, no network."""

from __future__ import annotations

import io

import pytest

from opendispatch.media import PLATFORM_IMAGE_SPECS, TranscodeError, transcode_image, transcode_image_bytes


def _png_bytes(w: int = 1000, h: int = 1000, color=(255, 0, 0)) -> bytes:
    """Build a solid-color PNG of (w,h) for use in tests."""
    from PIL import Image
    buf = io.BytesIO()
    Image.new("RGB", (w, h), color).save(buf, format="PNG")
    return buf.getvalue()


def _jpg_bytes(w: int = 1000, h: int = 1000, color=(0, 128, 0)) -> bytes:
    from PIL import Image
    buf = io.BytesIO()
    Image.new("RGB", (w, h), color).save(buf, format="JPEG")
    return buf.getvalue()


# ─── Spec table ─────────────────────────────────────────────────────────

def test_specs_table_has_expected_platforms():
    assert "instagram" in PLATFORM_IMAGE_SPECS
    assert "twitter" in PLATFORM_IMAGE_SPECS
    assert "linkedin" in PLATFORM_IMAGE_SPECS
    assert "youtube_thumbnail" in PLATFORM_IMAGE_SPECS


def test_all_specs_have_valid_dimensions():
    for name, spec in PLATFORM_IMAGE_SPECS.items():
        assert spec.max_width > 0, f"{name} has no max_width"
        assert spec.max_height > 0, f"{name} has no max_height"
        assert spec.format in ("JPEG", "PNG"), f"{name} unsupported format"


# ─── Resizing ───────────────────────────────────────────────────────────

def test_resize_oversize_input(tmp_path):
    """Large input should be scaled down to fit max_size."""
    src = tmp_path / "big.png"
    src.write_bytes(_png_bytes(4000, 4000))
    out = transcode_image(src, "twitter")
    from PIL import Image
    with Image.open(out) as img:
        # Twitter spec is 1200x675, so the long side should be ≤ 1200
        assert max(img.size) <= 1200


def test_small_input_not_upscaled(tmp_path):
    """We never enlarge images — quality matters more than filling the spec."""
    src = tmp_path / "small.png"
    src.write_bytes(_png_bytes(500, 500))
    out = transcode_image(src, "twitter")
    from PIL import Image
    with Image.open(out) as img:
        assert img.size[0] <= 1200
        # The aspect crop will trim 500x500 to ~16:9 (no enlargement)
        assert img.size[0] <= 500


def test_aspect_crop_for_instagram(tmp_path):
    """IG square spec should produce a 1:1 output even for landscape input."""
    src = tmp_path / "landscape.png"
    src.write_bytes(_png_bytes(1920, 1080))
    out = transcode_image(src, "instagram")
    from PIL import Image
    with Image.open(out) as img:
        w, h = img.size
        # ±2px tolerance for rounding during crop
        assert abs(w - h) <= 2


def test_aspect_crop_for_reels(tmp_path):
    """Reels = 9:16 portrait, even for square input."""
    src = tmp_path / "square.png"
    src.write_bytes(_png_bytes(1500, 1500))
    out = transcode_image(src, "instagram_reels")
    from PIL import Image
    with Image.open(out) as img:
        w, h = img.size
        # 9:16 = 0.5625
        assert abs((w / h) - 0.5625) < 0.02


# ─── Format handling ───────────────────────────────────────────────────

def test_rgba_png_flattened_to_jpeg(tmp_path):
    """RGBA input → opaque JPEG output (JPEG has no alpha)."""
    from PIL import Image
    rgba = Image.new("RGBA", (1000, 1000), (255, 0, 0, 128))
    buf = io.BytesIO()
    rgba.save(buf, format="PNG")

    out = transcode_image_bytes(buf.getvalue(), "twitter")
    # Output should be JPEG-decodable
    img = Image.open(io.BytesIO(out))
    assert img.format == "JPEG"
    assert img.mode == "RGB"


def test_exif_orientation_applied(tmp_path):
    """EXIF Orientation:6 = rotated 90° CW. We should rotate to upright."""
    from PIL import Image

    # Build a 1000x500 image with EXIF orientation 6 (rotate 270° CW for upright)
    img = Image.new("RGB", (1000, 500), (255, 0, 0))
    exif = img.getexif()
    exif[0x0112] = 6  # Orientation tag
    buf = io.BytesIO()
    img.save(buf, format="JPEG", exif=exif.tobytes())

    out = transcode_image_bytes(buf.getvalue(), "twitter")
    out_img = Image.open(io.BytesIO(out))
    # After EXIF transpose: a 1000x500 with orientation 6 becomes 500x1000.
    # Then Twitter spec is 16:9 landscape → we crop and downscale.
    # We just verify it's a valid image and the bytes round-trip.
    assert out_img.format == "JPEG"


# ─── Error paths ───────────────────────────────────────────────────────

def test_unknown_platform_raises():
    with pytest.raises(TranscodeError, match="no image spec"):
        transcode_image_bytes(_jpg_bytes(), "myspace")


def test_missing_source_raises(tmp_path):
    missing = tmp_path / "nope.png"
    with pytest.raises(TranscodeError, match="source image not found"):
        transcode_image(missing, "twitter")


def test_invalid_image_bytes_raises():
    with pytest.raises(TranscodeError, match="could not open image"):
        transcode_image_bytes(b"this is not a real image", "twitter")


# ─── End-to-end disk path ──────────────────────────────────────────────

def test_default_dest_path_includes_platform(tmp_path):
    src = tmp_path / "original.png"
    src.write_bytes(_png_bytes(2000, 2000))
    out = transcode_image(src, "twitter")
    # Default dest is <stem>.<platform>.<ext>
    assert "twitter" in out.name
    assert out.suffix.lower() == ".jpg"
    assert out.exists()


def test_explicit_dest_path_respected(tmp_path):
    src = tmp_path / "original.png"
    src.write_bytes(_png_bytes(2000, 2000))
    dest = tmp_path / "custom_name.jpg"
    out = transcode_image(src, "twitter", dest_path=dest)
    assert out == dest
    assert dest.exists()


# ─── API endpoint smoke tests ─────────────────────────────────────────

import importlib  # noqa: E402


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("OPEN_DISPATCH_DATA", str(tmp_path))
    import opendispatch.api.queue as q
    importlib.reload(q)
    import opendispatch.api.app as appmod
    importlib.reload(appmod)
    from fastapi.testclient import TestClient
    return TestClient(appmod.app)


def test_media_specs_endpoint(client):
    r = client.get("/media/specs")
    assert r.status_code == 200
    body = r.json()
    assert "instagram" in body
    assert body["instagram"]["max_width"] == 1080


def test_media_transcode_endpoint_raw_body(client):
    img = _png_bytes(2000, 2000)
    r = client.post("/media/transcode?platform=twitter", content=img,
                    headers={"Content-Type": "image/png"})
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("image/jpeg")
    # Output should be valid JPEG bytes
    from PIL import Image
    out_img = Image.open(io.BytesIO(r.content))
    assert out_img.format == "JPEG"


def test_media_transcode_rejects_unknown_platform(client):
    r = client.post("/media/transcode?platform=myspace",
                    content=_png_bytes(),
                    headers={"Content-Type": "image/png"})
    assert r.status_code == 400


def test_media_transcode_rejects_empty_body(client):
    r = client.post("/media/transcode?platform=twitter", content=b"",
                    headers={"Content-Type": "image/png"})
    assert r.status_code == 400
