"""Arabic in burned captions and hooks, checked on rendered pixels.
Run: python tests/test_arabic_text.py"""
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from PIL import Image, features

import hooks
import subtitles


def _need_ffmpeg():
    if not shutil.which("ffmpeg"):
        raise unittest.SkipTest("needs ffmpeg with libass")
    filters = subprocess.run(["ffmpeg", "-hide_banner", "-filters"], capture_output=True, text=True).stdout
    if " ass " not in filters or not features.check("raqm"):
        raise unittest.SkipTest("needs libass and libraqm")


def _frame(words, at, tmp, **opts):
    _need_ffmpeg()
    transcript = {"segments": [{"words": [
        {"word": " " + w, "start": i * .4, "end": (i + 1) * .4} for i, w in enumerate(words)]}]}
    path = os.path.join(tmp, "t.ass")
    args = dict(max_chars=80, max_duration=20, fontsize=44, font_name="Anton",
                highlight_color="#FF0000", border_width=4, effect="pop", uppercase=True)
    args.update(opts)
    assert subtitles.generate_ass(transcript, 0, 20, path, **args)
    vf = subtitles.subtitles_filter(path)
    raw = subprocess.check_output([
        "ffmpeg", "-v", "error", "-f", "lavfi", "-i", "color=blue:s=540x960:d=6:r=10",
        "-vf", vf, "-ss", str(at), "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"])
    return np.frombuffer(raw, np.uint8).reshape(960, 540, 3)


def _active_centre(img):
    r, g, b = (img[..., i].astype(int) for i in range(3))
    ys, xs = np.nonzero((r > 180) & (g < 80) & (b < 80))
    assert len(xs), "no highlighted word on screen"
    return xs.mean(), ys.mean()


def _centres(words, **opts):
    with tempfile.TemporaryDirectory() as tmp:
        return [_active_centre(_frame(words, i * .4 + .3, tmp, **opts)) for i in range(len(words))]


def test_words_run_right_to_left():
    for effect in ("pop", "none", "glow", "highlight"):
        for words in (["بيقول", "أنا", "أذنب"], ["تعبت", "و", "أنا"], ["من", "الذنب،", "ثم"]):
            xs = [x for x, _ in _centres(words, effect=effect)]
            assert xs == sorted(xs, reverse=True), (effect, words, xs)


def test_reveal_keeps_rtl_order():
    xs = [x for x, _ in _centres(["كل", "ما", "توب"], reveal=True)]
    assert xs == sorted(xs, reverse=True), xs


def test_latin_phrase_keeps_its_order():
    x = [c[0] for c in _centres(["في", "New", "York"], fontsize=30)]
    assert x[1] < x[2] < x[0], x


def test_long_line_wraps_first_words_on_top():
    c = _centres(["الاستغفار", "والتوبة", "النصوحة", "المستمرة"])
    assert c[0][1] < c[-1][1], c


def test_arabic_blocks_are_measured_in_the_arabic_font():
    assert subtitles.line_width(subtitles.rtl_text.ARABIC_FONT_NAME, 34, "الاستغفار") > 0


def test_english_is_unchanged():
    xs = [x for x, _ in _centres(["one", "two", "six"])]
    assert xs == sorted(xs), xs


def test_classic_srt_has_arabic_glyphs():
    _need_ffmpeg()
    with tempfile.TemporaryDirectory() as tmp:
        out = []
        for text in ("سسسسس", "ممممم"):
            path = os.path.join(tmp, "t.srt")
            subtitles.generate_srt({"segments": [{"words": [{"word": text, "start": 0, "end": 2}]}]}, 0, 2, path)
            out.append(subprocess.check_output([
                "ffmpeg", "-v", "error", "-f", "lavfi", "-i", "color=blue:s=540x960:d=1:r=5",
                "-vf", subtitles.subtitles_filter(path), "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]))
        assert out[0] != out[1]


def test_hooks_have_arabic_glyphs():
    with tempfile.TemporaryDirectory() as tmp:
        for style in hooks.HOOK_STYLES:
            for font in hooks.HOOK_FONTS:
                out = []
                # Tofu boxes are identical for any two words of equal length.
                for i, text in enumerate(("سسسسس", "ممممم")):
                    path = os.path.join(tmp, f"{i}.png")
                    hooks.create_hook_image(text, 900, path, style=style, font=font)
                    out.append(np.asarray(Image.open(path)))
                assert out[0].shape != out[1].shape or (out[0] != out[1]).any(), (style, font)


if __name__ == "__main__":
    failed = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            try:
                fn()
                print("ok  ", name)
            except unittest.SkipTest as e:
                print("skip", name, e)
            except Exception as e:
                failed += 1
                print("FAIL", name, repr(e)[:260])
    sys.exit(1 if failed else 0)
