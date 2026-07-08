"""Tests for hook overlay text handling (emoji runs, long-word wrapping)."""
from PIL import Image, ImageDraw, ImageFont

from hooks import _split_emoji_runs, _break_long_word, _EMOJI_RE


def _draw_and_font():
    img = Image.new("RGBA", (1, 1))
    draw = ImageDraw.Draw(img)
    return draw, ImageFont.load_default()


class TestEmojiRuns:
    def test_split_mixed_text(self):
        assert _split_emoji_runs("Feuer 🔥 test") == [
            (False, "Feuer "),
            (True, "🔥"),
            (False, " test"),
        ]

    def test_plain_text_single_run(self):
        assert _split_emoji_runs("nur text") == [(False, "nur text")]

    def test_emoji_only(self):
        assert _split_emoji_runs("🔥🚀") == [(True, "🔥🚀")]

    def test_strip_regex(self):
        assert _EMOJI_RE.sub("", "Stop 🛑 doing this! 💯") == "Stop  doing this! "


class TestLongWordWrap:
    def test_pieces_fit_and_recombine(self):
        draw, font = _draw_and_font()
        word = "A" * 60
        max_width = 50
        pieces = _break_long_word(draw, word, font, None, max_width)
        assert len(pieces) > 1
        assert "".join(pieces) == word
        for piece in pieces:
            assert draw.textlength(piece, font=font) <= max_width

    def test_short_word_single_piece(self):
        draw, font = _draw_and_font()
        assert _break_long_word(draw, "kurz", font, None, 1000) == ["kurz"]
