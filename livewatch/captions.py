"""One ASS subtitle file per clip: the hook on top and word-timed pop captions.

The frame is 1080x1920. Each layout leaves a known free band for the hook and
one for the captions (``LAYOUT_BANDS``). Captions come in chunks of up to three
words; the word being spoken is highlighted and pops in, the TikTok/Reels
look. libass cannot draw colour emoji, so emoji are dropped from burned text
(they stay in the post description).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

from .transcribe import Word
from .viral import strip_emoji

W, H = 1080, 1920
HOLD = 0.6  # seconds a caption stays up after its last word

# (hook centre y, caption centre y) per layout
LAYOUT_BANDS = {"blur": (330, 1560), "facecam": (690, 1480), "fill": (300, 1500)}


@dataclass(frozen=True)
class Look:
    hook_font: str
    hook_size: int
    hook_primary: str   # ASS &HAABBGGRR
    hook_outline: str
    hook_back: str
    hook_box: bool
    cap_font: str
    cap_size: int
    cap_primary: str
    cap_highlight: str
    cap_upper: bool


WHITE, BLACK, YELLOW, GREEN, RED = "&H00FFFFFF", "&H00000000", "&H0000E5FF", "&H0000FF7F", "&H003C3CFF"

LOOKS = {
    "ooc": Look("Montserrat ExtraBold", 66, BLACK, WHITE, WHITE, True,
                "Montserrat ExtraBold", 84, WHITE, YELLOW, True),
    "clean": Look("Montserrat ExtraBold", 64, BLACK, WHITE, WHITE, True,
                  "Montserrat ExtraBold", 80, WHITE, YELLOW, False),
    "brainrot": Look("Anton", 92, WHITE, BLACK, BLACK, False,
                     "Anton", 100, WHITE, GREEN, True),
}

def ass_time(t: float) -> str:
    t = max(0.0, t)
    cs = int(round(t * 100))
    h, cs = divmod(cs, 360000)
    m, cs = divmod(cs, 6000)
    s, cs = divmod(cs, 100)
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


def escape(text: str) -> str:
    """Plain text safe inside an ASS Dialogue line (no override blocks, no breaks)."""
    t = strip_emoji(text).replace("\\", "/").replace("{", "(").replace("}", ")")
    return " ".join(t.split())


def wrap(text: str, max_chars: int) -> List[str]:
    lines, cur = [], ""
    for word in text.split():
        if cur and len(cur) + 1 + len(word) > max_chars:
            lines.append(cur)
            cur = word
        else:
            cur = f"{cur} {word}".strip()
    if cur:
        lines.append(cur)
    return lines


def chunk_words(words: Sequence[Word], max_words: int = 3, max_chars: int = 16,
                max_gap: float = 0.6) -> List[List[Word]]:
    chunks: List[List[Word]] = []
    cur: List[Word] = []
    for w in words:
        text = escape(w.text)
        if not text:
            continue
        w = Word(w.start, w.end, text)
        length = sum(len(x.text) + 1 for x in cur) + len(text)
        if cur and (len(cur) >= max_words or length > max_chars or w.start - cur[-1].end > max_gap):
            chunks.append(cur)
            cur = []
        cur.append(w)
    if cur:
        chunks.append(cur)
    return chunks


def build_ass(words: Sequence[Word], hook: str, duration: float, style: str = "ooc",
              layout: str = "blur", show_captions: bool = True, show_hook: bool = True) -> str:
    look = LOOKS.get(style, LOOKS["ooc"])
    hook_y, cap_y = LAYOUT_BANDS.get(layout, LAYOUT_BANDS["blur"])
    hook_font = look.hook_font
    border_style = 3 if look.hook_box else 1
    hook_outline_w = 14 if look.hook_box else 7
    header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {W}
PlayResY: {H}
WrapStyle: 2
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Hook,{hook_font},{look.hook_size},{look.hook_primary},{look.hook_primary},{look.hook_outline},{look.hook_back},-1,0,0,0,100,100,0,0,{border_style},{hook_outline_w},0,5,60,60,0,1
Style: Cap,{look.cap_font},{look.cap_size},{look.cap_primary},{look.cap_primary},&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,6,3,5,40,40,0,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    events = []
    hook_text = escape(hook)
    if show_hook and hook_text:
        lines = wrap(hook_text, 20)
        body = r"\N".join(lines[:3])
        pop = r"{\fscx80\fscy80\t(0,120,\fscx100\fscy100)" + rf"\pos({W // 2},{hook_y})" + "}"
        events.append(f"Dialogue: 1,{ass_time(0)},{ass_time(duration)},Hook,,0,0,0,,{pop}{body}")
    if show_captions:
        chunks = chunk_words(words)
        for ci, chunk in enumerate(chunks):
            next_start = chunks[ci + 1][0].start if ci + 1 < len(chunks) else duration
            for i, w in enumerate(chunk):
                start = w.start
                if i + 1 < len(chunk):
                    end = chunk[i + 1].start
                else:
                    # hold the last word through a short pause, never into the next chunk
                    end = min(next_start, max(w.end, w.start + 0.25) + HOLD)
                if end <= start:
                    continue
                parts = []
                for j, x in enumerate(chunk):
                    t = x.text.upper() if look.cap_upper else x.text
                    if j == i:
                        parts.append(r"{\c" + look.cap_highlight + r"&\fscx106\fscy106}" + t
                                     + r"{\c" + look.cap_primary + r"&\fscx100\fscy100}")
                    else:
                        parts.append(t)
                pos = r"{\pos(" + f"{W // 2},{cap_y}" + ")}"
                events.append(f"Dialogue: 0,{ass_time(start)},{ass_time(min(end, duration))},Cap,,0,0,0,,"
                              f"{pos}{' '.join(parts)}")
    return header + "\n".join(events) + "\n"


def shift_words(words: Sequence[Word], offset: float, duration: float) -> List[Word]:
    """Words moved by ``-offset`` and clipped to [0, duration]."""
    out = []
    for w in words:
        s, e = w.start - offset, w.end - offset
        if e <= 0 or s >= duration:
            continue
        out.append(Word(max(0.0, s), min(duration, e), w.text))
    return out
