"""
Characterization tests for hooks.create_hook_image.

Targets the real PIL rendering pipeline — no mocks. Uses the
committed fonts/NotoSerif-Bold.ttf so no font download happens at
test time.

Asserts structural properties (file written, valid PNG, RGBA, has
visible pixels) rather than pixel-perfect content (too brittle).
"""
import os
import pytest
from PIL import Image

from app.overlays.hooks import create_hook_image


def test_create_hook_image_writes_a_png(tmp_path):
    out = tmp_path / "hook.png"
    create_hook_image("Hello world", target_width=1080, output_image_path=str(out))
    assert out.exists()
    assert out.stat().st_size > 0


def test_created_image_is_rgba_with_visible_pixels(tmp_path):
    out = tmp_path / "hook.png"
    create_hook_image("Did you know?", target_width=1080, output_image_path=str(out))
    with Image.open(out) as img:
        assert img.mode == "RGBA"
        alpha = img.split()[-1]
        visible_pixels = sum(1 for px in alpha.getdata() if px > 0)
        assert visible_pixels > 0


def test_image_width_is_bounded_by_target_width(tmp_path):
    """
    `target_width` is the MAX width the card can occupy — for short
    text the image will be narrower (just text + padding + shadow),
    but it must never exceed target_width by more than the shadow
    margin (~10 px on each side).
    """
    out = tmp_path / "hook.png"
    target = 1080
    create_hook_image("Hello", target_width=target, output_image_path=str(out))
    with Image.open(out) as img:
        assert img.width > 0
        assert img.width <= target + 50  # shadow margin


def test_longer_text_pushes_width_up_to_target(tmp_path):
    short = tmp_path / "short.png"
    long_ = tmp_path / "long.png"
    target = 1080
    create_hook_image("Hi", target_width=target, output_image_path=str(short))
    create_hook_image(
        "This is a much longer hook that should fill the available width.",
        target_width=target, output_image_path=str(long_),
    )
    with Image.open(short) as a, Image.open(long_) as b:
        assert b.width > a.width


def test_longer_text_produces_taller_image(tmp_path):
    short_path = tmp_path / "short.png"
    long_path = tmp_path / "long.png"
    create_hook_image("Hi", target_width=1080, output_image_path=str(short_path))
    create_hook_image(
        "This is a much longer hook that should wrap onto multiple lines because it "
        "exceeds the pixel-based wrap width by a comfortable margin.",
        target_width=1080,
        output_image_path=str(long_path),
    )
    with Image.open(short_path) as a, Image.open(long_path) as b:
        assert b.height > a.height


def test_font_scale_increases_image_size(tmp_path):
    small = tmp_path / "small.png"
    big = tmp_path / "big.png"
    create_hook_image("Same text", target_width=1080, output_image_path=str(small),
                      font_scale=1.0)
    create_hook_image("Same text", target_width=1080, output_image_path=str(big),
                      font_scale=2.0)
    with Image.open(small) as a, Image.open(big) as b:
        # At 2x font scale, the image must be visibly bigger in at least one dimension.
        assert (b.width >= a.width and b.height > a.height) or (
            b.height >= a.height and b.width > a.width
        )
