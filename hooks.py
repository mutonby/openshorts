import json
import os
import re
import shutil
import textwrap
import subprocess
import urllib.request
import uuid
from PIL import Image, ImageDraw, ImageFont, ImageFilter

from ffmpeg_utils import video_encode_args, QUALITY, METADATA_SCRUB, finished_before_crash
import gpu_decode


def _truncate_bytes(text, max_bytes):
    """Trim to a byte budget without splitting a multi-byte character.

    Deliberately duplicated from main.truncate_bytes: this module stays free of
    main's heavy imports (cv2, mediapipe, torch) so it can be used standalone.
    """
    encoded = text.encode("utf-8")
    if len(encoded) <= max_bytes:
        return text
    return encoded[:max_bytes].decode("utf-8", "ignore")

FONT_URL = "https://github.com/googlefonts/noto-fonts/raw/main/hinted/ttf/NotoSerif/NotoSerif-Bold.ttf"
FONT_DIR = "fonts"
FONT_PATH = os.path.join(FONT_DIR, "NotoSerif-Bold.ttf")
# Bundled (SIL OFL, see fonts/Montserrat-OFL.txt): the "pill" look's sans.
PILL_FONT_PATH = os.path.join(FONT_DIR, "Montserrat-ExtraBold.ttf")

# Typefaces the hook editor offers, all bundled so the server render and the
# browser preview use the same file. The size factor (share of the box width)
# evens out their visual weight: Anton is condensed, Montserrat wide.
HOOK_FONTS = {
    "montserrat": (PILL_FONT_PATH, 0.064),
    "anton": (os.path.join(FONT_DIR, "Anton-Regular.ttf"), 0.08),
    "serif": (FONT_PATH, 0.05),
}

# Codepoint ranges NotoSerif has no glyphs for (would render as tofu boxes).
_EMOJI_RE = re.compile(
    "["
    "\U0001F000-\U0001FAFF"  # emoticons, symbols, transport, supplemental
    "\U00002600-\U000027BF"  # misc symbols + dingbats
    "\U0001F1E6-\U0001F1FF"  # regional indicators (flags)
    "\U00002B00-\U00002BFF"  # arrows, stars
    "\U0000FE0E\U0000FE0F"   # variation selectors
    "\U0000200D"             # zero-width joiner
    "\U000020E3"             # combining keycap
    "]+"
)

# Emoji-capable fonts, probed at runtime (Windows, WSL, Linux/Docker, macOS).
_EMOJI_FONT_CANDIDATES = [
    "C:\\Windows\\Fonts\\seguiemj.ttf",
    "/mnt/c/Windows/Fonts/seguiemj.ttf",
    "/usr/share/fonts/truetype/noto/NotoColorEmoji.ttf",
    "/usr/share/fonts/noto-emoji/NotoColorEmoji.ttf",
    "/System/Library/Fonts/Apple Color Emoji.ttc",
]


# Bitmap strike sizes color-emoji fonts ship with. NotoColorEmoji (the font
# the Docker image installs) ONLY loads at its strike size — asking for an
# arbitrary size raises "invalid pixel size" — so glyphs are rendered at the
# native size and rescaled at draw time.
_EMOJI_BITMAP_SIZES = [109, 128, 136, 160, 96, 72, 64, 32]


def _load_emoji_font(font_size):
    """Return (font, native_size) for an emoji-capable font, or None.

    native_size == font_size for scalable fonts (Segoe UI Emoji); for
    fixed-bitmap fonts (NotoColorEmoji) it is the strike size the font
    actually loaded at, and the renderer scales glyphs down from it."""
    for path in _EMOJI_FONT_CANDIDATES:
        if not os.path.exists(path):
            continue
        try:
            return ImageFont.truetype(path, font_size), font_size
        except Exception:
            pass
        for native in _EMOJI_BITMAP_SIZES:
            try:
                return ImageFont.truetype(path, native), native
            except Exception:
                continue
    return None


def _split_emoji_runs(text):
    """Split text into (is_emoji, chunk) runs."""
    runs = []
    pos = 0
    for m in _EMOJI_RE.finditer(text):
        if m.start() > pos:
            runs.append((False, text[pos:m.start()]))
        runs.append((True, m.group()))
        pos = m.end()
    if pos < len(text):
        runs.append((False, text[pos:]))
    return runs


def _emoji_scale(font, emoji_font):
    """Draw-time scale factor from the emoji font's native size to the text
    size. 1.0 for scalable emoji fonts; < 1 for fixed-bitmap strikes."""
    efont, native = emoji_font
    target = getattr(font, "size", native)
    return target / float(native) if native else 1.0


def _measure_width(draw, text, font, emoji_font):
    """Pixel width of a line, measuring emoji runs with the emoji font.

    emoji_font is the (font, native_size) pair from _load_emoji_font, or None."""
    width = 0.0
    for is_emoji, chunk in _split_emoji_runs(text):
        if is_emoji and emoji_font:
            width += (draw.textlength(chunk, font=emoji_font[0])
                      * _emoji_scale(font, emoji_font))
        else:
            width += draw.textlength(chunk, font=font)
    return width


def _render_emoji_chunk(chunk, emoji_font, scale, fill):
    """Rasterize an emoji run at the font's native size, scaled to the text
    size. Returns an RGBA image ready to alpha-composite, or None."""
    efont, native = emoji_font
    probe = ImageDraw.Draw(Image.new('RGBA', (1, 1)))
    w = int(probe.textlength(chunk, font=efont))
    if w <= 0:
        return None
    # Emoji glyphs can overshoot the em box slightly; 1.3x covers it.
    tmp = Image.new('RGBA', (w, int(native * 1.3)), (0, 0, 0, 0))
    d = ImageDraw.Draw(tmp)
    try:
        d.text((0, 0), chunk, font=efont, embedded_color=True)
    except TypeError:
        d.text((0, 0), chunk, font=efont, fill=fill)
    if scale == 1.0:
        return tmp
    return tmp.resize((max(int(w * scale), 1), max(int(tmp.height * scale), 1)),
                      Image.LANCZOS)


def _draw_mixed(img, draw, xy, text, font, emoji_font, fill, outline=None):
    """Draw a line onto img, rendering emoji runs with the emoji font (in
    color if supported, rescaled when the font is a fixed-size bitmap).
    outline: optional (color, px) stroke drawn under the text."""
    x, y = xy
    stroke_w = outline[1] if outline else 0
    stroke_fill = outline[0] if outline else None
    for is_emoji, chunk in _split_emoji_runs(text):
        if is_emoji and emoji_font:
            scale = _emoji_scale(font, emoji_font)
            rendered = _render_emoji_chunk(chunk, emoji_font, scale, fill)
            if rendered is not None:
                img.alpha_composite(rendered, (int(x), int(y)))
                x += draw.textlength(chunk, font=emoji_font[0]) * scale
        else:
            if stroke_w:
                draw.text((x, y), chunk, font=font, fill=fill,
                          stroke_width=stroke_w, stroke_fill=stroke_fill)
            else:
                draw.text((x, y), chunk, font=font, fill=fill)
            x += draw.textlength(chunk, font=font)


def _break_long_word(draw, word, font, emoji_font, max_width):
    """Character-level hard wrap for a single word wider than max_width."""
    pieces = []
    current = ""
    for ch in word:
        if current and _measure_width(draw, current + ch, font, emoji_font) > max_width:
            pieces.append(current)
            current = ch
        else:
            current += ch
    if current:
        pieces.append(current)
    return pieces

def download_font_if_needed():
    """Downloads a serif font for the hook text if not present."""
    if not os.path.exists(FONT_DIR):
        os.makedirs(FONT_DIR)
    if not os.path.exists(FONT_PATH):
        print(f"⬇️ Downloading font from {FONT_URL}...")
        try:
            # Add user agent to avoid 403s slightly
            req = urllib.request.Request(
                FONT_URL, 
                headers={'User-Agent': 'Mozilla/5.0'}
            )
            with urllib.request.urlopen(req) as response, open(FONT_PATH, 'wb') as out_file:
                out_file.write(response.read())
            print("✅ Font downloaded.")
        except Exception as e:
            print(f"❌ Failed to download font: {e}")

# Hook visual styles. Each maps to box fill (RGBA, alpha 0 = no box), text
# color, and an optional text outline (color, px) for box-less looks.
HOOK_STYLES = {
    # Default. TikTok-native look: every line its own white rounded box,
    # black bold sans (Montserrat ExtraBold).
    "pill":    {"box": (255, 255, 255, 250), "text": (0, 0, 0), "outline": None, "shadow": False,
                "font": PILL_FONT_PATH, "pills": True},
    # White card, black serif text (original look).
    "classic": {"box": (255, 255, 255, 240), "text": (0, 0, 0), "outline": None, "shadow": True},
    # Dark card, white text.
    "dark":    {"box": (18, 18, 20, 235),    "text": (255, 255, 255), "outline": None, "shadow": True},
    # Bright yellow card, black text (high-contrast TikTok look).
    "yellow":  {"box": (255, 214, 0, 245),   "text": (0, 0, 0), "outline": None, "shadow": True},
    # Red "breaking" card, white text.
    "red":     {"box": (220, 38, 38, 245),   "text": (255, 255, 255), "outline": None, "shadow": True},
    # No box: white text with a thick black outline (caption/MrBeast style).
    "outline": {"box": (0, 0, 0, 0),         "text": (255, 255, 255), "outline": ((0, 0, 0), 8), "shadow": False},
    # No box: yellow text with black outline.
    "outline_yellow": {"box": (0, 0, 0, 0),  "text": (255, 214, 0),   "outline": ((0, 0, 0), 8), "shadow": False},
}


def _draw_pills(lines, font, emoji_font, font_size, box_fill, text_fill, output_image_path):
    """One rounded box per line, stacked and centred (the "pill" style).

    Every box has the same height (from the font's own ascent/descent plus
    accents), so the stack reads as one block whatever the letters are.
    """
    lines = [ln for ln in lines if ln.strip()] or [" "]
    pad_x = int(font_size * 0.48)
    pad_y = int(font_size * 0.24)
    radius = int(font_size * 0.36)
    probe = ImageDraw.Draw(Image.new('RGBA', (1, 1)))
    top, bottom = probe.textbbox((0, 0), "ÁÉÍÓÚÑgjpqy", font=font)[1::2]
    line_h = (bottom - top) + 2 * pad_y
    widths = [int(_measure_width(probe, ln, font, emoji_font)) for ln in lines]
    canvas_w = max(widths) + 2 * pad_x + 40
    canvas_h = line_h * len(lines) + 40

    img = Image.new('RGBA', (canvas_w, canvas_h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    for i, (ln, w) in enumerate(zip(lines, widths)):
        box_w = w + 2 * pad_x
        x0 = (canvas_w - box_w) // 2
        y0 = 20 + i * line_h
        draw.rounded_rectangle([x0, y0, x0 + box_w, y0 + line_h], radius=radius, fill=box_fill)
        _draw_mixed(img, draw, (x0 + pad_x, y0 + pad_y - top), ln, font, emoji_font,
                    fill=text_fill, outline=None)
    img.save(output_image_path)
    return output_image_path, canvas_w, canvas_h


def create_hook_image(text, target_width, output_image_path="hook_overlay.png", font_scale=1.0, style="pill",
                      font=None):
    """
    Generates a hook overlay image using pixel-based wrapping.
    target_width: The max width the box should occupy (e.g. 85% of video)
    style: one of HOOK_STYLES (pill/classic/dark/yellow/red/outline/outline_yellow)
    """
    download_font_if_needed()

    look = HOOK_STYLES.get(style, HOOK_STYLES["pill"])
    box_fill = look["box"]
    text_fill = look["text"]
    outline = look["outline"]
    has_box = box_fill[3] > 0
    draw_shadow = look["shadow"]
    
    # Configuration
    padding_x = 30 # Balanced padding
    padding_y = 25 
    line_spacing = 20 # Increased spacing
    cornerradius = 20
    shadow_offset = (5, 5) 
    shadow_blur = 10
    
    # Font Size Calculation (approx 5% of width - tuned to match Noto Serif Bold metrics in browser)
    pills = bool(look.get("pills"))
    # A chosen typeface (HOOK_FONTS key) wins; otherwise the style's own
    # (pill -> Montserrat, the rest -> Noto Serif, as they always rendered).
    default_font = "montserrat" if look.get("font") == PILL_FONT_PATH else "serif"
    font_path, size_factor = HOOK_FONTS.get(font or default_font, HOOK_FONTS[default_font])
    base_font_size = int(target_width * size_factor)
    font_size = int(base_font_size * font_scale)
    
    try:
        font = ImageFont.truetype(font_path, font_size)
    except Exception as e:
        print(f"⚠️ Warning: Could not load font {font_path}, using default. Error: {e}")
        font = ImageFont.load_default()

    # Emoji handling: render with an emoji-capable font if one exists,
    # otherwise strip emoji instead of drawing tofu boxes.
    emoji_font = None
    if _EMOJI_RE.search(text):
        emoji_font = _load_emoji_font(font_size)
        if emoji_font is None:
            text = _EMOJI_RE.sub("", text)
            text = re.sub(r"[ \t]{2,}", " ", text).strip()

    # Wrap text logic (Pixel-based)
    dummy_img = Image.new('RGBA', (1, 1))
    draw = ImageDraw.Draw(dummy_img)

    max_text_width = target_width - (2 * padding_x)

    # Handle manual newlines first
    paragraphs = text.split('\n')
    lines = []

    for p in paragraphs:
        if not p.strip():
            lines.append("")
            continue

        words = p.split()
        current_line = []

        for word in words:
            # Test if adding word fits
            test_line = ' '.join(current_line + [word])
            w = _measure_width(draw, test_line, font, emoji_font)

            if w <= max_text_width:
                current_line.append(word)
                continue

            # Word doesn't fit on the current line
            if current_line:
                lines.append(' '.join(current_line))
                current_line = []

            if _measure_width(draw, word, font, emoji_font) <= max_text_width:
                current_line = [word]
            else:
                # Single word wider than the box: hard-wrap it character-wise
                # so it can't get cut off at the edges.
                pieces = _break_long_word(draw, word, font, emoji_font, max_text_width)
                lines.extend(pieces[:-1])
                current_line = [pieces[-1]] if pieces else []

        if current_line:
            lines.append(' '.join(current_line))

    if pills:
        return _draw_pills(lines, font, emoji_font, font_size, box_fill, text_fill,
                           output_image_path)

    # Recalculate true width/height
    max_line_width = 0
    text_heights = []

    for line in lines:
        if not line:
            text_heights.append(font_size) # Use font size for empty line height
            continue

        w = _measure_width(draw, line, font, emoji_font)
        bbox = draw.textbbox((0, 0), line, font=font)
        h = bbox[3] - bbox[1]
        max_line_width = max(max_line_width, int(w))
        text_heights.append(h)
    
    # Box dimensions
    # We want the box to fit the text exactly + padding
    # Ensure min width for aesthetic reasons if text is short (at least 30% of target)
    box_width = max(max_line_width + (2 * padding_x), int(target_width * 0.3))
    
    # Total Text Height: sum(heights) + spacing * (n-1)
    if not text_heights:
         total_text_height = font_size
    else:
         total_text_height = sum(text_heights) + (len(text_heights) - 1) * line_spacing
         
    box_height = total_text_height + (2 * padding_y)
    
    # Create Final Image with Rounded Corners and Shadow
    # 1. Canvas for Shadow (larger than box)
    canvas_w = box_width + 40
    canvas_h = box_height + 40
    
    img = Image.new('RGBA', (canvas_w, canvas_h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    # 2. Draw Shadow (only for boxed styles)
    if draw_shadow and has_box:
        shadow_box = [
            (20 + shadow_offset[0], 20 + shadow_offset[1]),
            (20 + box_width + shadow_offset[0], 20 + box_height + shadow_offset[1])
        ]
        draw.rounded_rectangle(shadow_box, radius=cornerradius, fill=(0, 0, 0, 100))
        # 3. Blur Shadow
        img = img.filter(ImageFilter.GaussianBlur(5))

    # 4. Draw Box (sharper, on top of blurred shadow)
    draw_final = ImageDraw.Draw(img)

    if has_box:
        main_box = [
            (20, 20),
            (20 + box_width, 20 + box_height)
        ]
        draw_final.rounded_rectangle(main_box, radius=cornerradius, fill=box_fill)

    # 5. Draw Text
    current_y = 20 + padding_y - 2 # Minor visual adjustment
    for i, line in enumerate(lines):
        if not line:
            current_y += font_size + line_spacing
            continue

        line_w = _measure_width(draw_final, line, font, emoji_font)
        bbox = draw_final.textbbox((0, 0), line, font=font)
        line_h = text_heights[i] if i < len(text_heights) else bbox[3] - bbox[1]

        # Center X
        x = 20 + int(box_width - line_w) // 2

        # Draw text in the style's color (emoji runs use the emoji font)
        _draw_mixed(img, draw_final, (x, current_y), line, font, emoji_font,
                    fill=text_fill, outline=outline)

        current_y += line_h + line_spacing
        
    img.save(output_image_path)
    return output_image_path, canvas_w, canvas_h

def _auto_place(video_path, text, target_box_width, hook_filename, font_scale, style, font,
                video_width, video_height, duration, ranges, has_captions, first_dims):
    """position='auto': pick the height (and, only if every height covers a
    face, a smaller text) with hook_placement. Returns {"image": (path, w, h),
    "y_px", "y", "scale"}, or None to fall back to the legacy top spot (no
    readable frames, no MediaPipe, or nothing fits between the bands)."""
    import hook_placement
    import layout_ranges as _layouts

    window_end = float(duration) if duration else hook_placement.DEFAULT_WINDOW
    if ranges is None:
        ranges = _layouts.read(video_path)
    ranges = _layouts.normalise(ranges)
    bands = hook_placement.caption_bands(ranges, 0.0, window_end, has_captions)
    # A WIDE picture floats in blur: keep the hook in the blur when it fits,
    # and only fall back to the caption-only rule when it does not.
    pictures = hook_placement.content_bands(ranges, 0.0, window_end)
    frames = hook_placement.detect_faces(video_path, 0.0, window_end)
    if frames is None:
        print("   ℹ️ Hook placement: no frames read, keeping the top spot.")
        return None

    dims = {1.0: first_dims}

    def measure(rel):
        if rel not in dims:
            _, w, h = create_hook_image(text, target_box_width, hook_filename,
                                        font_scale=font_scale * rel, style=style, font=font)
            dims[rel] = (w, h)
        w, h = dims[rel]
        return w / video_width, h / video_height

    y = None
    if pictures:
        rel, y, cost = hook_placement.best_scale(measure, frames,
                                                 sorted(set(bands) | set(pictures)))
        if y is not None:
            bands = sorted(set(bands) | set(pictures))
    if y is None:
        rel, y, cost = hook_placement.best_scale(measure, frames, bands)
    if y is None:
        print("   ℹ️ Hook placement: no spot clears the captions, keeping the top spot.")
        return None
    # The temp PNG holds the last size measured: render the chosen one again.
    _, w, h = create_hook_image(text, target_box_width, hook_filename,
                                font_scale=font_scale * rel, style=style, font=font)
    n_faces = sum(len(f) for f in frames)
    print(f"   🎯 Hook placement: y={y:.2f} scale={rel:g} face cover={cost:.2f} "
          f"({n_faces} faces in {len(frames)} frames, bands={bands})")
    return {"image": (hook_filename, w, h), "y_px": int(round(y * video_height)),
            "y": round(y, 3), "scale": rel}


def add_hook_to_video(video_path, text, output_path, position="top", font_scale=1.0, duration=None, style="pill",
                      also=None, font=None, layout_ranges=None, has_captions=True,
                      mark=None):
    """
    Overlays text hook onto video.
    position: 'top', 'center', 'bottom', or 'auto' (hook_placement: off the
      faces in the hook's window and off the caption bands; what the job
      pipeline and an editor left on "auto" use). An explicit top/center/
      bottom is drawn exactly where it always was.
    layout_ranges / has_captions: only read by 'auto', to know where the
      captions sit (seam on SPLIT stretches, bottom elsewhere). None reads
      the clip's own ``.layout.json`` sidecar.
    font_scale: float multiplier (1.0 = default)
    style: hook look (see HOOK_STYLES)
    also: optional (vf, path): ALSO write ``path`` = the hooked picture with
      ``vf`` applied (the captions), from the same decode. The editor
      re-captions from the hooked-only picture, but most clips are never
      re-captioned, so with ``also`` that file is not encoded: a recipe
      (overlay image + position) is kept instead and ``materialize`` encodes
      it the first time it is asked for (``HOOK_LAZY=0``: encode it here).
    mark: optional (logo_path, make_filter, path), only with ``also``: ALSO
      write ``path`` = the captioned picture with the free-plan mark, from the
      same decode (the served ``wm_`` copy, see main.mark_delivery).
      ``make_filter(vw, vh, video=, logo=, out=)`` is main.watermark_filter.
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video {video_path} not found")

    # 1. Probe video width to scale text properly
    try:
        cmd = ['ffprobe', '-v', 'error', '-show_entries', 'stream=width,height', '-of', 'csv=s=x:p=0', video_path]
        res = subprocess.check_output(cmd, timeout=60).decode().strip()
        # Takes first stream if multiple
        dims = res.split('\n')[0].split('x')
        video_width = int(dims[0])
        video_height = int(dims[1])
    except Exception as e:
        print(f"⚠️ FFprobe failed: {e}. Assuming 1080x1920")
        video_width = 1080
        video_height = 1920
        
    # 2. Generate Image
    # Box check: Don't let it be wider than 90% of screen
    target_box_width = int(video_width * 0.9)
    
    # Unique per invocation so parallel jobs can't overwrite each other's overlay.
    # The uuid alone guarantees that; the source name rides along only as a
    # debugging hint, so it is trimmed by BYTES (filesystems cap at 255 bytes,
    # and one Bengali or Arabic character costs three). Embedding it untrimmed
    # raised OSError 36 and killed the endpoint in prod on 26-jul-2026.
    stem = os.path.splitext(os.path.basename(video_path))[0]
    hook_filename = (f"temp_hook_{uuid.uuid4().hex[:8]}_"
                     f"{_truncate_bytes(stem, 80)}.png")
    
    try:
        img_path, box_w, box_h = create_hook_image(text, target_box_width, hook_filename, font_scale=font_scale, style=style,
                                                       font=font)
        placed = None
        if position == "auto":
            placed = _auto_place(video_path, text, target_box_width, hook_filename, font_scale,
                                 style, font, video_width, video_height, duration,
                                 layout_ranges, has_captions, (box_w, box_h))
            if placed:
                img_path, box_w, box_h = placed["image"]

        # 3. Calculate Overlay Position
        overlay_x = (video_width - box_w) // 2
        
        if placed:
            overlay_y = placed["y_px"]
        elif position == "center":
            overlay_y = (video_height - box_h) // 2
        elif position == "bottom":
             # Bottom 20% mark (approx)
             overlay_y = int(video_height * 0.70)
        else:
             # Top 20% mark
             overlay_y = int(video_height * 0.20)
        
        # 4. FFmpeg Command
        print(f"🎬 Overlaying hook: '{text}' at {overlay_x},{overlay_y}")
        
        overlay = _overlay_graph(overlay_x, overlay_y, duration)
        tail = _encode_tail()
        lazy = also is not None and LAZY
        hooked_out = [] if lazy else ['-map', '[h]', '-map', '0:a?', *tail, output_path]
        if also and mark:
            extra_vf, extra_path = also
            logo_path, make_filter, mark_path = mark
            marked = make_filter(video_width, video_height, video="[sm]",
                                 logo="[2:v]", out="[m]")
            graph = (f"{overlay},{extra_vf},split=2[s][sm];{marked}" if lazy else
                     f"{overlay},split=2[h][c];[c]{extra_vf},split=2[s][sm];{marked}")
            ffmpeg_cmd = [
                'ffmpeg', '-y',
                '-i', video_path,
                '-i', img_path,
                '-i', logo_path,
                '-filter_complex', graph,
                *hooked_out,
                '-map', '[s]', '-map', '0:a?', *tail, extra_path,
                '-map', '[m]', '-map', '0:a?', *tail, mark_path,
            ]
        elif also:
            extra_vf, extra_path = also
            graph = (f"{overlay},{extra_vf}[s]" if lazy else
                     f"{overlay},split=2[h][c];[c]{extra_vf}[s]")
            ffmpeg_cmd = [
                'ffmpeg', '-y',
                '-i', video_path,
                '-i', img_path,
                '-filter_complex', graph,
                *hooked_out,
                '-map', '[s]', '-map', '0:a?', *tail, extra_path,
            ]
        else:
            ffmpeg_cmd = _overlay_cmd(video_path, img_path, overlay, output_path)

        outputs = [] if lazy else [output_path]
        if also:
            outputs.append(also[1])
            if mark:
                outputs.append(mark[2])
        result = _run_encode(ffmpeg_cmd, video_path, outputs)
        if result.returncode != 0 and not finished_before_crash(result, outputs):
            raise subprocess.CalledProcessError(result.returncode, ffmpeg_cmd,
                                                result.stdout, result.stderr)
        if lazy:
            try:
                _write_recipe(output_path, video_path, img_path, overlay_x, overlay_y, duration)
                print(f"✅ Hook added to {also[1]} ({os.path.basename(output_path)} on demand)")
            except Exception as e:
                # No recipe, no on-demand file: encode the hooked picture now.
                print(f"⚠️ Hook recipe not kept ({e}); encoding {output_path} now")
                cmd = _overlay_cmd(video_path, img_path, overlay, output_path)
                done = _run_encode(cmd, video_path, [output_path])
                if done.returncode != 0 and not finished_before_crash(done, [output_path]):
                    raise subprocess.CalledProcessError(done.returncode, cmd,
                                                        done.stdout, done.stderr)
        else:
            print(f"✅ Hook added to {output_path}")
        if placed:
            return {"position": "auto", "y": placed["y"], "scale": placed["scale"],
                    "height": round(box_h / video_height, 3)}
        return True

    except subprocess.TimeoutExpired:
        print("❌ FFmpeg hook overlay timed out after 1800s.")
        raise RuntimeError("FFmpeg hook overlay timed out after 1800s.")
    except subprocess.CalledProcessError as e:
        print(f"❌ FFmpeg Error: {e.stderr.decode() if e.stderr else 'Unknown'}")
        raise e
    except Exception as e:
        print(f"❌ Hook Gen Error: {e}")
        raise e
    finally:
        # Cleanup temp image
        if os.path.exists(hook_filename):
            os.remove(hook_filename)


# --- the hooked-only picture, encoded on demand -------------------------------

LAZY = os.environ.get("HOOK_LAZY", "1").strip() != "0"
RECIPE_PREFIX = ".hookrecipe_"


def _overlay_graph(x, y, duration):
    return (f"[0:v][1:v]overlay={x}:{y}"
            + (f":enable='between(t,0,{float(duration)})'" if duration else ""))


def _encode_tail():
    return ['-c:a', 'copy', *video_encode_args(QUALITY), *METADATA_SCRUB,
            '-movflags', '+faststart']


def _overlay_cmd(video_path, img_path, overlay, output_path):
    return ['ffmpeg', '-y', '-i', video_path, '-i', img_path,
            '-filter_complex', overlay, *_encode_tail(), output_path]


def _run_encode(cmd, video_path, outputs):
    return gpu_decode.run(
        lambda on_gpu: gpu_decode.on_card(cmd, video_path) if on_gpu else cmd,
        lambda c: subprocess.run(c, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                 timeout=1800),
        ok=lambda r: r.returncode == 0 or finished_before_crash(r, outputs),
        path=video_path)


def recipe_paths(output_dir, hooked_name):
    """(json, png) holding how to encode ``hooked_name``, next to the clip."""
    base = os.path.join(output_dir, f"{RECIPE_PREFIX}{hooked_name}")
    return base + ".json", base + ".png"


def has_recipe(output_dir, hooked_name):
    return os.path.exists(recipe_paths(output_dir, hooked_name)[0])


def _write_recipe(output_path, video_path, img_path, x, y, duration):
    """Keep what it takes to encode ``output_path`` later: the overlay image
    and where it went. The source must sit in the same directory."""
    output_dir = os.path.dirname(output_path)
    if os.path.dirname(os.path.abspath(video_path)) != os.path.abspath(output_dir):
        raise ValueError("the hook's source must be next to its output")
    json_path, png_path = recipe_paths(output_dir, os.path.basename(output_path))
    shutil.copyfile(img_path, png_path)
    with open(json_path, "w") as f:
        json.dump({"source": os.path.basename(video_path), "x": x, "y": y,
                   "duration": duration}, f)


def materialize(output_dir, hooked_name):
    """Encode ``hooked_name`` from its recipe if it is not on disk yet.

    True when the file exists afterwards. Safe to call concurrently: each
    call encodes to its own temporary name and renames it into place."""
    path = os.path.join(output_dir, hooked_name)
    if os.path.exists(path):
        return True
    json_path, png_path = recipe_paths(output_dir, hooked_name)
    try:
        with open(json_path) as f:
            r = json.load(f)
        source = os.path.join(output_dir, os.path.basename(r["source"]))
    except (OSError, ValueError, KeyError, TypeError):
        return False
    if not (os.path.exists(source) and os.path.exists(png_path)):
        return False
    tmp = os.path.join(output_dir, f".part_{uuid.uuid4().hex[:8]}_{hooked_name}")
    cmd = _overlay_cmd(source, png_path, _overlay_graph(r["x"], r["y"], r.get("duration")), tmp)
    try:
        result = _run_encode(cmd, source, [tmp])
        if result.returncode != 0 and not finished_before_crash(result, [tmp]):
            print(f"⚠️ Could not encode {hooked_name} from its recipe")
            return False
        os.replace(tmp, path)
        return True
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
