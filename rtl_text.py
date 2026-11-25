"""Right-to-left text for the burned renderers (captions, hooks, thumbnails).

Two things break Arabic there:
* PIL has no font fallback, and none of the bundled display fonts has Arabic
  (tofu boxes). fonts/OpenShortsArabic-Bold.ttf carries Arabic and Latin
  (Noto Sans Arabic + Noto Sans, OFL; built by fonts/build_arabic_font.py).
* libass lays each style run out left to right on its own, so the inline tags
  of the active karaoke word reverse an RTL line. The line is written in
  visual order instead, every word bidi-isolated from its neighbours.
"""
import os
import re
from functools import lru_cache

# Hebrew, Arabic, Syriac, Thaana, NKo and presentation forms (as Subtitles.tsx).
_RTL_RE = re.compile("[\u0590-\u08FF\uFB1D-\uFDFF\uFE70-\uFEFF]")
_ARABIC_RE = re.compile("[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]")

ARABIC_FONT_NAME = "OpenShorts Arabic"
ARABIC_FONT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "fonts", "OpenShortsArabic-Bold.ttf")

RLE = "\u202B"
PDF = "\u202C"
LRM = "\u200E"
# A strong LTR mark on both sides keeps two RTL words from merging into one run.
WORD_GAP = f"{LRM} {LRM}"


def is_rtl(text):
    return bool(_RTL_RE.search(text or ""))


def is_arabic(text):
    return bool(_ARABIC_RE.search(text or ""))


def layout_opts():
    """Pillow options for RTL. Without libraqm it refuses ``direction``: the
    text is then unshaped, which beats failing the render."""
    from PIL import features
    return {"direction": "rtl"} if features.check("raqm") else {}


def embed(text):
    """RTL base direction for a whole run of text; other text is untouched."""
    if not is_rtl(text):
        return text
    return "\n".join(f"{RLE}{line}{PDF}" if line else line for line in text.split("\n"))


def visual_order(words):
    """(index, text) pairs in left-to-right screen order for an RTL line.

    Words run right to left, except a Latin phrase, which keeps its own order
    with its closing punctuation moved to its left ("!New York").
    """
    groups = []
    i = 0
    while i < len(words):
        last = None
        j = i
        while j < len(words) and not is_rtl(words[j]):
            if any(ch.isalpha() for ch in words[j]):
                last = j
            elif last is None:
                break
            j += 1
        if last is None:
            groups.append([(i, f"{RLE}{words[i]}{PDF}")])
            i += 1
            continue
        group = [(k, words[k]) for k in range(i, last + 1)]
        body = words[last].rstrip(".,!?:;…")
        tail = words[last][len(body):]
        if tail:
            group[-1] = (last, body)
            group[0] = (i, tail[::-1] + group[0][1])
        groups.append(group)
        i = last + 1
    return [item for group in reversed(groups) for item in group]


@lru_cache(maxsize=1)
def _metrics():
    from PIL import ImageFont
    font = ImageFont.truetype(ARABIC_FONT_PATH, 1000)
    ascent, descent = font.getmetrics()
    return font, ascent + descent


def wrap_words(words, font_size, max_width):
    """(lines of word indices, font size), in ASS units.

    Visual-order text cannot be left to libass to wrap (the first line would
    get the last words), so the lines are broken here. The size only shrinks
    when a single word is wider than the line.
    """
    font, cell = _metrics()  # libass sizes a font by its whole cell height

    def width(text):
        return font.getlength(text, **layout_opts()) * font_size / cell

    widest = max(width(w) for w in words)
    if widest > max_width:
        return [[i] for i in range(len(words))], max(1, int(font_size * max_width / widest))
    lines = [[0]]
    for i in range(1, len(words)):
        if width(" ".join(words[k] for k in lines[-1] + [i])) > max_width:
            lines.append([i])
        else:
            lines[-1].append(i)
    return lines, font_size
