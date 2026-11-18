"""livewatch card layout: news-page header, emoji headline, full-frame stream, logo import."""
import shutil
import subprocess
from io import BytesIO

import pytest

PIL = pytest.importorskip("PIL")
from PIL import Image  # noqa: E402

from livewatch import card, captions, config, render  # noqa: E402

HAS_FFMPEG = bool(shutil.which("ffmpeg") and shutil.which("ffprobe"))


def _png(color=(240, 110, 30), size=(300, 200)) -> bytes:
    buf = BytesIO()
    Image.new("RGB", size, color).save(buf, "PNG")
    return buf.getvalue()


def test_save_logo_validates_and_squares(tmp_path):
    dest = tmp_path / "brand" / "logo.png"
    card.save_logo(_png(), dest)
    with Image.open(dest) as im:
        assert im.size == (512, 512) and im.format == "PNG"
    for bad in (b"", b"not an image at all", b"\x89PNG broken"):
        with pytest.raises(ValueError):
            card.save_logo(bad, tmp_path / "x.png")


def test_render_card_with_and_without_logo(tmp_path):
    logo = tmp_path / "logo.png"
    card.save_logo(_png(), logo)
    with_logo = card.render_card("First look at the crash physics in GTA 6 👀",
                                 card.Brand("GTA6 Clips", "gta6clips", True, logo), tmp_path / "a.png")
    no_logo = card.render_card("A much longer headline that has to wrap over more than one line "
                               "because it keeps going and going 😭🔥",
                               card.Brand("", "gta6clips", False, None), tmp_path / "b.png")
    for lay, f in ((with_logo, "a.png"), (no_logo, "b.png")):
        with Image.open(tmp_path / f) as im:
            assert im.size == (1080, 1920)
        assert lay.video_h == 608                      # the whole 16:9 frame at full width
        assert lay.video_y >= card.TOP_SAFE and lay.video_y % 2 == 0
        assert lay.video_y + lay.video_h < lay.caption_y < 1920
    # the logo is drawn: its colour is in the header, absent without a logo
    with Image.open(tmp_path / "a.png") as im:
        px = im.getpixel((card.MARGIN + 60, with_logo.video_y - 300))
    assert px[0] > 200 and px[2] < 100
    # a vertical source is fitted in height, never cropped
    tall = card.render_card("x", card.Brand(handle="a"), tmp_path / "c.png", video_aspect=9 / 16)
    assert tall.video_y + tall.video_h <= 1920 - card.CAPTION_BAND


def test_emoji_runs_are_split():
    assert card._runs("GTA 6 👀 is here 😭🔥") == [("GTA 6 ", False), ("👀", True), (" is here ", False),
                                                  ("😭🔥", True)]


def test_card_filter_keeps_the_whole_frame():
    lay = card.CardLayout(video_y=700, video_h=608, caption_y=1420)
    g = render.build_filter("card", 30, None, "c.ass", None, "x264", lay)
    assert "[1:v]" in g and "overlay=(W-w)/2:700" in g
    assert "scale=1080:608:force_original_aspect_ratio=decrease" in g and "crop" not in g
    blur = render.build_filter("blur", 30, None, None, None, "x264")
    assert "crop='min(iw" not in blur  # the old side crop that cut facecams is gone


def test_captions_follow_the_card(tmp_path):
    from livewatch.transcribe import Word
    ass = captions.build_ass([Word(0, 0.5, "hey")], "Headline 👀", 5, "news", "card", True, False, cap_y=1420)
    assert "\\pos(540,1420)" in ass and ",Hook," not in ass


def test_config_layout_and_brand(tmp_path):
    s = config.from_dict({"style": {"layout": "card"}, "brand": {"name": "GTA6 Clips", "handle": "@gta6_clips",
                                                                  "verified": False}}, base_dir=tmp_path)
    assert (s.layout, s.brand_name, s.brand_handle, s.brand_verified) == ("card", "GTA6 Clips", "gta6_clips", False)
    assert s.logo_path == s.data_dir / "brand" / "logo.png"
    assert config.from_dict({}, base_dir=tmp_path).caption_style == "news"
    for bad in ({"style": {"layout": "tiktok"}}, {"brand": {"handle": "bad handle!"}}):
        with pytest.raises(config.ConfigError):
            config.from_dict(bad, base_dir=tmp_path)


@pytest.mark.skipif(not HAS_FFMPEG, reason="ffmpeg not installed")
def test_render_card_short_end_to_end(tmp_path):
    raw = tmp_path / "raw.ts"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc2=size=1280x720:rate=30:duration=4",
                    "-f", "lavfi", "-i", "sine=frequency=440:duration=4", "-c:v", "libx264", "-preset", "ultrafast",
                    # a keyframe every second, like a stream (a single keyframe cannot be seeked into)
                    "-g", "30", "-c:a", "aac", "-shortest", str(raw)], check=True)
    lay = card.render_card("Test headline 👀", card.Brand(handle="me"), tmp_path / "card.png")
    out = render.render_short(raw, tmp_path / "short.mp4", 0.5, 3, layout="card", encoder="x264",
                              card=(tmp_path / "card.png", lay))
    info = render.probe(out)
    assert (info["width"], info["height"]) == (1080, 1920) and info["audio"]
    assert info["duration"] == pytest.approx(3, abs=0.3)
    # the stream really sits where the layout says: a frame row inside the video band is not black
    frame = tmp_path / "f.png"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", "1", "-i", str(out), "-frames:v", "1", str(frame)], check=True)
    from PIL import ImageStat
    with Image.open(frame) as im:
        rgb = im.convert("L")
        inside = ImageStat.Stat(rgb.crop((0, lay.video_y + 10, 1080, lay.video_y + lay.video_h - 10))).mean[0]
        below = ImageStat.Stat(rgb.crop((0, lay.video_y + lay.video_h + 8, 1080,
                                         lay.video_y + lay.video_h + 60))).mean[0]
    assert inside > 40 and below < 8
