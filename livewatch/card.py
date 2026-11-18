"""The "card" look: a news-page post on black, the way viral clip pages present GTA moments.

    [logo] Page name (verified badge)
           @handle
    First look at the motorcycle crash physics in GTA 6 👀
    ┌──────────────── the stream, full width, nothing cropped ────────────────┐
    └──────────────────────────────────────────────────────────────────────────┘
                         word-by-word captions

The whole stream frame is shown (16:9 at 1080 px wide), so a facecam in a
corner is never cut off. The header and headline are drawn with Pillow,
because the subtitle renderer (libass) cannot draw colour emoji; ffmpeg lays
the stream over this frame.

Without a logo the header is the @handle alone (and the badge, if on).
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

W, H = 1080, 1920
MARGIN = 64
TOP_SAFE = 170            # platform UI (status bar, "Reels" title) covers the very top
CAPTION_BAND = 230        # room under the video for the captions
FONT_DIRS = [Path(os.environ.get("WINDIR", r"C:\Windows")) / "Fonts",
             Path("/usr/share/fonts/truetype/dejavu"), Path("/usr/share/fonts")]
REPO_FONTS = Path(__file__).resolve().parent.parent / "fonts"
BADGE_BLUE = (29, 155, 240)


@dataclass(frozen=True)
class Brand:
    name: str = ""
    handle: str = ""
    verified: bool = True
    logo: Optional[Path] = None

    @property
    def at_handle(self) -> str:
        h = (self.handle or "").strip().lstrip("@")
        return f"@{h}" if h else ""


@dataclass(frozen=True)
class CardLayout:
    video_y: int       # where the stream's top edge goes
    video_h: int       # its height at 1080 wide (or the cap)
    caption_y: int     # centre line of the captions


# --------------------------------------------------------------------------- fonts
def _font_path(*names: str) -> Optional[str]:
    for d in [*FONT_DIRS, REPO_FONTS]:
        for n in names:
            p = d / n
            if p.is_file():
                return str(p)
    return None


def _font(size: int, weight: str = "regular"):
    from PIL import ImageFont
    names = {"bold": ("segoeuib.ttf", "DejaVuSans-Bold.ttf", "Montserrat-ExtraBold.ttf"),
             "semibold": ("seguisb.ttf", "segoeuib.ttf", "DejaVuSans-Bold.ttf"),
             "regular": ("segoeui.ttf", "DejaVuSans.ttf", "Montserrat-ExtraBold.ttf")}[weight]
    path = _font_path(*names)
    return ImageFont.truetype(path, size) if path else ImageFont.load_default(size)


def _emoji_font(size: int):
    from PIL import ImageFont
    path = _font_path("seguiemj.ttf", "NotoColorEmoji.ttf")
    if not path:
        return None
    try:
        return ImageFont.truetype(path, size)
    except OSError:
        try:
            return ImageFont.truetype(path, 109)  # Noto Color Emoji has one bitmap size
        except OSError:
            return None


_EMOJI = re.compile("([\U0001F000-\U0001FAFF\u2600-\u27BF\u2B50\u2B55\u203C\u2049"
                    "\U0001F1E6-\U0001F1FF][\uFE0F\u200D\U0001F3FB-\U0001F3FF"
                    "\U0001F000-\U0001FAFF\u2600-\u27BF]*)")


def _runs(text: str) -> List[Tuple[str, bool]]:
    """Text split into (piece, is_emoji)."""
    out = []
    for i, part in enumerate(_EMOJI.split(text)):
        if part:
            out.append((part, i % 2 == 1))
    return out


def _width(draw, text: str, font, efont) -> float:
    w = 0.0
    for piece, emo in _runs(text):
        f = efont if (emo and efont) else font
        w += draw.textlength(piece, font=f)
    return w


def _draw(img, draw, xy, text: str, font, efont, fill):
    x, y = xy
    size = getattr(font, "size", 60)
    for piece, emo in _runs(text):
        if emo and efont:
            draw.text((x, y + size * 0.08), piece, font=efont, embedded_color=True)
            x += draw.textlength(piece, font=efont)
        else:
            draw.text((x, y), piece, font=font, fill=fill)
            x += draw.textlength(piece, font=font)


def wrap(draw, text: str, font, efont, max_w: float) -> List[str]:
    lines, cur = [], ""
    for word in text.split():
        trial = f"{cur} {word}".strip()
        if cur and _width(draw, trial, font, efont) > max_w:
            lines.append(cur)
            cur = word
        else:
            cur = trial
    if cur:
        lines.append(cur)
    return lines


# ------------------------------------------------------------------------- pieces
def _badge(size: int):
    """A blue verified-style rosette with a white tick."""
    from PIL import Image, ImageDraw
    s = size * 4
    img = Image.new("RGBA", (s, s), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    import math
    pts = []
    for i in range(24):
        r = s * (0.5 if i % 2 == 0 else 0.43)
        a = math.pi * 2 * i / 24
        pts.append((s / 2 + r * math.cos(a), s / 2 + r * math.sin(a)))
    d.polygon(pts, fill=BADGE_BLUE + (255,))
    d.line([(s * 0.29, s * 0.52), (s * 0.44, s * 0.66), (s * 0.72, s * 0.37)],
           fill=(255, 255, 255, 255), width=int(s * 0.09), joint="curve")
    return img.resize((size, size))


def _logo(path: Path, size: int):
    from PIL import Image, ImageDraw, ImageOps
    img = Image.open(path).convert("RGBA")
    img = ImageOps.fit(img, (size * 2, size * 2))
    mask = Image.new("L", img.size, 0)
    ImageDraw.Draw(mask).rounded_rectangle([0, 0, img.size[0] - 1, img.size[1] - 1],
                                           radius=int(size * 2 * 0.18), fill=255)
    img.putalpha(mask)
    return img.resize((size, size))


# --------------------------------------------------------------------------- frame
def render_card(headline: str, brand: Brand, out_path: Path, video_aspect: float = 16 / 9,
                video_box: Optional[Tuple[int, int]] = None) -> CardLayout:
    """Draw the 1080x1920 background frame; returns where the stream goes.

    ``video_box`` (w, h) draws a grey placeholder there (dashboard preview).
    """
    from PIL import Image, ImageDraw
    img = Image.new("RGBA", (W, H), (0, 0, 0, 255))
    d = ImageDraw.Draw(img)
    name_f, handle_f = _font(50, "bold"), _font(42)
    logo_size = 128
    has_logo = bool(brand.logo and Path(brand.logo).is_file())
    name = brand.name.strip() if has_logo else ""
    handle = brand.at_handle
    if not has_logo and not handle:
        handle = brand.name.strip()

    # headline: as big as fits in 3 lines
    text = " ".join((headline or "").split())
    for size in (66, 60, 54, 48, 44):
        head_f, emo_f = _font(size), _emoji_font(size)
        lines = wrap(d, text, head_f, emo_f, W - 2 * MARGIN)
        if len(lines) <= 3:
            break
    lines = lines[:4]
    line_h = int(head_f.size * 1.25)
    header_h = logo_size if has_logo else (64 if handle else 0)
    head_h = len(lines) * line_h
    video_h = int(round(W / max(video_aspect, 0.3)))
    video_h = min(video_h, H - TOP_SAFE - header_h - head_h - CAPTION_BAND - 120)
    video_h -= video_h % 2
    total = header_h + 44 + head_h + 56 + video_h + CAPTION_BAND
    top = max(TOP_SAFE, (H - total) // 2)

    # header
    y = top
    if has_logo:
        img.alpha_composite(_logo(Path(brand.logo), logo_size), (MARGIN, y))
        tx = MARGIN + logo_size + 28
        if name:
            _draw(img, d, (tx, y + 6), name, name_f, _emoji_font(50), (255, 255, 255))
            if brand.verified:
                bx = int(tx + _width(d, name, name_f, _emoji_font(50)) + 12)
                img.alpha_composite(_badge(46), (bx, y + 18))
            d.text((tx, y + 70), handle, font=handle_f, fill=(139, 152, 165))
        else:
            d.text((tx, y + 36), handle, font=name_f, fill=(255, 255, 255))
            if brand.verified and handle:
                img.alpha_composite(_badge(46), (int(tx + d.textlength(handle, font=name_f) + 12), y + 48))
    elif handle:
        d.text((MARGIN, y), handle, font=name_f, fill=(255, 255, 255))
        if brand.verified:
            img.alpha_composite(_badge(46), (int(MARGIN + d.textlength(handle, font=name_f) + 12), y + 12))
    y += header_h + (44 if header_h else 0)

    # headline
    emo_f = _emoji_font(head_f.size)
    for ln in lines:
        _draw(img, d, (MARGIN, y), ln, head_f, emo_f, (255, 255, 255))
        y += line_h
    y += 56
    video_y = y - y % 2
    if video_box:
        d.rectangle([0, video_y, W - 1, video_y + video_h], fill=(45, 49, 56))
        d.text((W // 2, video_y + video_h // 2), "stream", font=_font(48), fill=(120, 126, 135), anchor="mm")
    img.convert("RGB").save(out_path)
    return CardLayout(video_y=video_y, video_h=video_h,
                      caption_y=min(H - 160, video_y + video_h + CAPTION_BAND // 2))


CTA_Y = 1690          # where the follow badge floats (under the captions, above the app's buttons)
DEFAULT_CTA = "Follow & subscribe for more GTA 6 news 🔔"


def render_cta(text: str, out_path: Path, logo: Optional[Path] = None) -> Tuple[int, int]:
    """The floating "follow" badge: a white pill with the text (emoji in colour)
    and, when there is one, the logo as a small round avatar. Returns (w, h)."""
    from PIL import Image, ImageDraw
    text = " ".join((text or DEFAULT_CTA).split())[:80]
    probe = ImageDraw.Draw(Image.new("RGBA", (10, 10)))
    for size in (40, 36, 32, 28):
        font, efont = _font(size, "semibold"), _emoji_font(size)
        tw = _width(probe, text, font, efont)
        av = int(size * 1.6) if logo and Path(logo).is_file() else 0
        w = int(tw + av + (18 if av else 0) + 2 * 34)
        if w <= W - 2 * MARGIN:
            break
    h = int(size * 2.0)
    s = 2  # draw at 2x, then scale down for smooth edges
    img = Image.new("RGBA", (w * s, h * s), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.rounded_rectangle([0, 0, w * s - 1, h * s - 1], radius=h * s // 2, fill=(255, 255, 255, 240))
    x = 34 * s
    if av:
        logo_img = _logo(Path(logo), av * s)
        mask = Image.new("L", logo_img.size, 0)
        ImageDraw.Draw(mask).ellipse([0, 0, logo_img.size[0] - 1, logo_img.size[1] - 1], fill=255)
        logo_img.putalpha(mask)
        img.alpha_composite(logo_img, (x - 14 * s, (h * s - av * s) // 2))
        x += (av + 18 - 14) * s
    big_f, big_e = _font(size * s, "semibold"), _emoji_font(size * s)
    _draw(img, d, (x, (h * s - big_f.size) // 2 - int(big_f.size * 0.12)), text, big_f, big_e, (10, 10, 10))
    img = img.resize((w, h), Image.LANCZOS)
    img.save(out_path)
    return w, h


def save_logo(data: bytes, dest: Path) -> Path:
    """Validate an uploaded image and store it as a 512 px PNG."""
    from io import BytesIO
    from PIL import Image, ImageOps, UnidentifiedImageError
    if not data or len(data) > 8 * 1024 * 1024:
        raise ValueError("the image is empty or larger than 8 MB")
    try:
        img = Image.open(BytesIO(data))
        img.load()
    except (UnidentifiedImageError, OSError):
        raise ValueError("that file is not an image (use PNG or JPG)")
    img = ImageOps.exif_transpose(img).convert("RGBA")
    img = ImageOps.fit(img, (512, 512))
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".tmp.png")
    img.save(tmp, "PNG")
    tmp.replace(dest)
    return dest
