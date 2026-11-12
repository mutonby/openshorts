import os
import re
import subprocess
import sys

from ffmpeg_utils import (video_encode_args, escape_filter_value, QUALITY,
                          METADATA_SCRUB, finished_before_crash)
import gpu_decode


_STDIO_CONFIGURED = False

# Shared faster-whisper config so both transcription paths (this module and
# main.transcribe_video) behave identically. "small" is meaningfully better at
# German than "base" without being much slower on CPU.
DEFAULT_WHISPER_MODEL = "small"


def get_whisper_config():
    """Return the faster-whisper model config, overridable via env vars."""
    return {
        "model_size": os.environ.get("WHISPER_MODEL", DEFAULT_WHISPER_MODEL),
        "device": os.environ.get("WHISPER_DEVICE", "cpu"),
        "compute_type": os.environ.get("WHISPER_COMPUTE", "int8"),
    }


# Decode params shared by both transcription paths. condition_on_previous_text
# is off to avoid repetition/hallucination loops; vad_filter drops silence.
WHISPER_TRANSCRIBE_PARAMS = {
    "beam_size": 5,
    "vad_filter": True,
    "condition_on_previous_text": False,
    "word_timestamps": True,
}


def merge_continuation_words(words):
    """Merge faster-whisper continuation fragments into their base word.

    faster-whisper marks a word boundary with a LEADING SPACE on each token.
    Compound-word fragments (e.g. "-Kanal.", ".200") arrive WITHOUT a leading
    space and belong to the preceding word. Without merging, "YouTube" and
    "-Kanal." get space-joined into "YouTube -Kanal." or split across subtitle
    blocks. We concatenate such fragments onto the previous word and extend its
    end time. Normal words keep their leading space, so real word boundaries
    (e.g. "ich habe") are never glued together.

    Returns a new list; the input dicts are not mutated.
    """
    merged = []
    for word in words:
        text = word.get("word", "")
        if merged and isinstance(text, str) and text and not text.startswith(" "):
            prev = merged[-1]
            prev["word"] = f"{prev.get('word', '')}{text}"
            if word.get("end") is not None:
                prev["end"] = word["end"]
        else:
            merged.append(dict(word))
    return merged


def _configure_stdio():
    global _STDIO_CONFIGURED
    if _STDIO_CONFIGURED:
        return
    _STDIO_CONFIGURED = True
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        if not stream or not hasattr(stream, "reconfigure"):
            continue
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


def _log(message):
    _configure_stdio()
    stream = sys.stdout
    text = str(message)
    try:
        stream.write(text + "\n")
    except UnicodeEncodeError:
        encoding = getattr(stream, "encoding", None) or "utf-8"
        safe_text = text.encode(encoding, errors="replace").decode(encoding, errors="replace")
        stream.write(safe_text + "\n")
    stream.flush()


def _escape_ffmpeg_filter_value(value):
    """Escape a path/value for use inside a quoted FFmpeg filter argument.

    NOTE: an apostrophe in the path cannot be made safe here. ffmpeg's
    filtergraph parser is not a shell — the shell idiom ``'\\''`` was tried on
    29-jul-2026 and is worse than doing nothing: it drops the apostrophe AND
    swallows the following option, so ``ass='…Earth'\\''s.ass':fontsdir='…'``
    resolved to a filename of "…Earths.ass:fontsdir=…" and failed to open.

    The only reliable answer is to keep apostrophes OUT of any path that is
    interpolated into a filter. Callers generate their own subtitle filenames,
    so they control this: use a neutral name (``subs_<i>_<ts>.ass``), never one
    derived from a video title.

    The implementation now lives in ffmpeg_utils so the reframe engine can use
    it too: it was building `sendcmd=f='<abs path>'` unescaped, which is the
    same bug this function was written for.
    """
    return escape_filter_value(value)


def _normalize_subtitle_word(value):
    return " ".join(str(value or "").split())


def transcribe_audio(video_path):
    """
    Transcribe audio from a video file via the configured ASR backend.
    Returns transcript in the same format as main.py for compatibility.
    """
    # Lazy import: transcribe_backends imports helpers from this module.
    from transcribe_backends import transcribe_media

    _log(f"🎙️  Transcribing audio from: {video_path}")
    transcript = transcribe_media(video_path)
    _log(f"✅ Transcription complete. Language: {transcript['language']}")
    return transcript


def generate_srt_from_video(video_path, output_path, max_chars=20, max_duration=2.0,
                            style="classic", **style_opts):
    """
    Transcribe a video and generate a subtitle file directly (SRT, or karaoke
    ASS when style="karaoke"). Used for dubbed videos without a transcript.
    """
    transcript = transcribe_audio(video_path)

    # Get video duration to use as clip_end
    import cv2
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps if fps else 0
    cap.release()

    if style == "karaoke":
        return generate_ass(transcript, 0, duration, output_path, max_chars, max_duration, **style_opts)
    return generate_srt(transcript, 0, duration, output_path, max_chars, max_duration)


def _collect_word_blocks(transcript, clip_start, clip_end, max_chars=20, max_duration=2.0):
    """
    Flatten transcript words for a clip range and group them into short blocks
    suitable for vertical video. Returns a list of blocks; each block is a list
    of {'word', 'start', 'end'} dicts with times relative to the clip.

    Continuation fragments are merged defensively here too, because transcripts
    from old jobs on disk store unmerged tokens (the leading space is still
    present, so the boundary signal survives).
    """
    flat_words = []
    for segment in transcript.get('segments', []):
        flat_words.extend(segment.get('words', []))
    flat_words = merge_continuation_words(flat_words)

    words = []
    for word_info in flat_words:
        if word_info.get('end', 0) > clip_start and word_info.get('start', 0) < clip_end:
            cleaned_word = _normalize_subtitle_word(word_info.get('word', ''))
            if not cleaned_word:
                continue
            words.append({
                'word': cleaned_word,
                'start': max(0, word_info['start'] - clip_start),
                'end': max(0, word_info['end'] - clip_start),
            })

    blocks = []
    current_block = []
    block_start = None

    for word in words:
        if not current_block:
            current_block = [word]
            block_start = word['start']
            continue

        current_text_len = sum(len(w['word']) + 1 for w in current_block)
        duration = word['end'] - block_start

        if current_text_len + len(word['word']) > max_chars or duration > max_duration:
            blocks.append(current_block)
            current_block = [word]
            block_start = word['start']
        else:
            current_block.append(word)

    if current_block:
        blocks.append(current_block)
    return blocks


def generate_srt(transcript, clip_start, clip_end, output_path, max_chars=20, max_duration=2.0):
    """
    Generates an SRT file from the transcript for a specific time range.
    Groups words into short lines suitable for vertical video.
    """
    blocks = _collect_word_blocks(transcript, clip_start, clip_end, max_chars, max_duration)
    if not blocks:
        return False

    srt_content = ""
    for index, block in enumerate(blocks, 1):
        text = " ".join(w['word'] for w in block).strip()
        srt_content += format_srt_block(index, block[0]['start'], block[-1]['end'], text)

    # Write UTF-8 with BOM so Windows/FFmpeg subtitle readers reliably detect Unicode text.
    with open(output_path, 'w', encoding='utf-8-sig') as f:
        f.write(srt_content)

    return True


# Vertical margin for burned captions, in PlayResY=288 units (so ~15% of the
# frame height). The old hardcoded 25 (8.7%) put captions underneath TikTok's
# and Reels' own bottom UI — the caption/username block and the music ticker —
# where they were partly covered on the platform even though the exported file
# looked fine.
SAFE_MARGIN_V = 43


# The caption look applied automatically to every generated clip. Chosen by
# rendering four candidates on a real clip and comparing them (25-jul-2026):
# white Anton uppercase with a yellow active word, heavy black outline, gentle
# pop. Yellow because it is the one colour that almost never occurs in footage,
# so the active word reads instantly on any background; the base text stays
# fully opaque (dimming it tested worse over bright scenes). This is a starting
# point, not a cage — the subtitle modal still overrides every field.
AUTO_CAPTION_STYLE = {
    "style": "karaoke",
    "alignment": "bottom",
    "font_name": "Anton",
    "font_size": 44,
    "font_color": "#FFFFFF",
    "highlight_color": "#FFE500",
    "border_color": "#000000",
    "border_width": 4,
    "effect": "pop",
    "base_opacity": 1.0,
    "uppercase": True,
    "max_chars": 16,
    "max_duration": 1.4,
}


# Named looks for /api/subtitle `preset` (and the MCP add_subtitles tool), in
# request-field names. Mirrors the dashboard's CAPTION_PRESETS in
# SubtitleModal.jsx (a test checks the ids): "default" is what every clip
# ships with, the rest are the short-form looks trending in 2026.
_PRESET_BASE = {"style": "karaoke", "font_color": "#FFFFFF", "bg_opacity": 0.0,
                "base_opacity": 1.0, "reveal": False, "shadow": 0,
                "max_duration": 1.4}
CAPTION_PRESETS = {
    "default": {**_PRESET_BASE, "font_name": "Anton", "font_size": 44,
                "highlight_color": "#FFE500", "border_width": 4, "effect": "pop",
                "uppercase": True, "max_chars": 16},
    # Words appear as they are spoken, yellow active word, shadow.
    "hormozi": {**_PRESET_BASE, "font_name": "Montserrat ExtraBold", "font_size": 44,
                "highlight_color": "#FFE500", "border_width": 4, "shadow": 2,
                "effect": "pop", "uppercase": True, "reveal": True, "max_chars": 9},
    # Solid box behind the active word (CapCut / Submagic).
    "pill": {**_PRESET_BASE, "font_name": "Montserrat ExtraBold", "font_size": 44,
             "highlight_color": "#7C3AED", "border_width": 3, "effect": "highlight",
             "uppercase": True, "max_chars": 9},
    "lime": {**_PRESET_BASE, "font_name": "Montserrat ExtraBold", "font_size": 44,
             "highlight_color": "#A3FF12", "border_width": 3, "effect": "highlight",
             "uppercase": True, "max_chars": 9},
    # One big word at a time.
    "oneword": {**_PRESET_BASE, "font_name": "Anton", "font_size": 70,
                "highlight_color": "#FFFFFF", "border_width": 5, "effect": "pop",
                "uppercase": True, "max_chars": 1},
    # No outline, soft shadow, sentence case.
    "clean": {**_PRESET_BASE, "font_name": "Montserrat ExtraBold", "font_size": 34,
              "highlight_color": "#FFFFFF", "border_width": 0, "shadow": 2,
              "effect": "none", "uppercase": False, "base_opacity": 0.7,
              "max_chars": 12},
}

# Characters per line at font size 44, per font (Anton is condensed,
# Montserrat wide). Mirrors lineBudget in SubtitleModal.jsx.
_LINE_CHARS = {"Anton": 16, "Montserrat ExtraBold": 9, "Impact": 16}


def line_budget(font_name, font_size):
    """max_chars that keeps one line inside the 9:16 frame at this size."""
    size = _clamp_number(font_size, 10, 200, 44)
    return max(6, round(_LINE_CHARS.get(font_name, 14) * 44 / size))


def _ass_time(seconds):
    """Format seconds as ASS timestamp H:MM:SS.cc (centiseconds)."""
    seconds = max(0, seconds)
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    centis = int(round((seconds - int(seconds)) * 100))
    if centis >= 100:
        centis = 99
    return f"{hours}:{minutes:02d}:{secs:02d}.{centis:02d}"


def _hex_to_ass_inline_color(hex_color, fallback="FFFFFF"):
    """Convert #RRGGBB to the &HBBGGRR& form used by inline \\c override tags."""
    hex_digits = str(hex_color or "").lstrip('#')
    if not _HEX_COLOR_RE.match(hex_digits):
        hex_digits = fallback
    r = hex_digits[0:2]
    g = hex_digits[2:4]
    b = hex_digits[4:6]
    return f"&H{b}{g}{r}&".upper()


def _escape_ass_text(text):
    """Neutralize characters that would start ASS override blocks."""
    # Line breaks too: one inside a word would end the Dialogue line and let
    # the rest be read as an event or a section of its own.
    return (str(text).replace('\\', '/').replace('{', '(').replace('}', ')')
            .replace('\r', ' ').replace('\n', ' '))


def _dim_hex_color(hex_color, opacity, fallback="FFFFFF"):
    """Fully-opaque 'dimmed' variant of a color (scaled toward black).

    Dimming via alpha looks muddy in ASS: libass draws the outline as a
    filled shape UNDER the fill, so a semi-transparent white fill blends
    with its own black outline into dark grey. Scaling the RGB instead
    keeps the text crisp on every player."""
    hex_digits = str(hex_color or "").lstrip('#')
    if not _HEX_COLOR_RE.match(hex_digits):
        hex_digits = fallback
    # Gentle curve: even strong dimming stays a readable light silver, matching
    # the airy look of browser-alpha dimming over bright video.
    factor = 0.5 + 0.5 * _clamp_number(opacity, 0.05, 1.0, 1.0)
    r = min(255, round(int(hex_digits[0:2], 16) * factor))
    g = min(255, round(int(hex_digits[2:4], 16) * factor))
    b = min(255, round(int(hex_digits[4:6], 16) * factor))
    return f"{r:02X}{g:02X}{b:02X}"


def generate_ass(transcript, clip_start, clip_end, output_path,
                 max_chars=20, max_duration=2.0, alignment='bottom',
                 fontsize=16, font_name="Verdana", font_color="#FFFFFF",
                 border_color="#000000", border_width=2,
                 highlight_color="#FFD700", bg_color="#000000", bg_opacity=0.0,
                 effect="none", base_opacity=1.0, uppercase=False,
                 margin_v=SAFE_MARGIN_V, split_ranges=None,
                 reveal=False, shadow=0):
    """
    Generates a karaoke-style ASS file: each block is shown like the SRT path,
    but the currently spoken word is rendered in highlight_color (modern
    TikTok/CapCut caption look). One dialogue event per word, back to back, so
    the highlight moves with the audio without flicker.

    effect: "none" | "glow" (neon shine around the active word) |
            "pop" (active word scales up) | "box" (thick colored outline) |
            "highlight" (active word on a solid box in highlight_color, the
            CapCut / Submagic look).
    base_opacity: opacity of the non-active words — dimmed base text is the
    modern captioneer look (e.g. 0.4).
    reveal: words not spoken yet are invisible, so the line builds up word
    by word (Hormozi style). They keep their slot (alpha, not removal), so
    the line never reflows while it fills.
    shadow: drop shadow depth in PlayRes units (0 = none). With border_width
    0 it is the soft "clean" look; a hard outline does not need it.
    max_chars=1 puts one word on screen at a time.
    """
    blocks = _collect_word_blocks(transcript, clip_start, clip_end, max_chars, max_duration)
    if not blocks:
        return False

    # Match the SRT burn path: PlayResY 288 keeps font sizes consistent.
    final_fontsize = int(_clamp_number(fontsize, 10, 200, 16) * 0.85)
    if final_fontsize < 10:
        final_fontsize = 10

    align_map = {'top': 8, 'middle': 5, 'bottom': 2}
    ass_alignment = align_map.get(str(alignment).lower(), 2)

    # On a SPLIT scene the two speakers are stacked and the seam between the
    # halves (exactly mid-frame) is the one place the text covers nobody, so
    # every word event inside such a stretch is anchored there with an inline
    # \an5, per event rather than per style: a clip mixes stacked and single
    # shots, and the text moves with the cut. ``split_ranges`` is a list of
    # (start, end) in clip seconds (layout_ranges.split_ranges); the style's
    # own alignment still rules everywhere else. Only the ASS path can do
    # this: SRT burns carry one alignment for the whole file.
    seam_ranges = [(float(a), float(b)) for a, b in (split_ranges or [])]

    def seam_prefix(t):
        return "{\\an5}" if any(a <= t < b for a, b in seam_ranges) else ""

    safe_font = _sanitize_font_name(font_name)
    base_opacity = _clamp_number(base_opacity, 0.05, 1.0, 1.0)
    # Dim inactive words via a fully-opaque scaled color (NOT alpha — see
    # _dim_hex_color); the active word overrides the color inline.
    primary_colour = hex_to_ass_color(_dim_hex_color(font_color, base_opacity), 1.0)
    bg_opacity = _clamp_number(bg_opacity, 0.0, 1.0, 0.0)
    border_width = _clamp_number(border_width, 0, 10, 2)
    shadow = int(_clamp_number(shadow, 0, 6, 0))

    if bg_opacity > 0:
        border_style = 3
        outline_colour = hex_to_ass_color(bg_color, bg_opacity, fallback="000000")
        outline_width = 1
    else:
        border_style = 1
        outline_colour = hex_to_ass_color(border_color, 1.0, fallback="000000")
        # 0 is a real choice now (the shadow-only "clean" look); before the
        # slider's "None" still drew a 1px outline. Without a shadow keep the
        # old floor, or white text on a white wall disappears.
        outline_width = int(border_width) if shadow else max(1, int(border_width))

    back_colour = hex_to_ass_color("#000000", 0.55 if shadow else 0.0)
    highlight_inline = _hex_to_ass_inline_color(highlight_color, fallback="FFD700")

    # Inline override tags for the active word; {\r} after it resets to the
    # (dimmed) style so the rest of the block stays untouched.
    if effect == "glow":
        glow_bord = max(3, int(outline_width) + 2)
        active_prefix = (f"{{\\c&HFFFFFF&\\3c{highlight_inline}"
                         f"\\bord{glow_bord}\\blur4}}")
    elif effect == "box":
        box_bord = max(4, int(outline_width) + 3)
        active_prefix = (f"{{\\c&HFFFFFF&\\3c{highlight_inline}"
                         f"\\bord{box_bord}\\blur0}}")
    elif effect == "highlight":
        # A second style with BorderStyle 3 draws an opaque box around just
        # the active word (libass boxes each style run separately). The text
        # on it flips to black when the box is light, or yellow would carry
        # white text nobody can read.
        active_prefix = "{\\rActive}"
    elif effect == "pop":
        # Gentle pop. The old 75->112 range started the word so small that any
        # frame caught mid-animation read as a sizing bug rather than a beat.
        active_prefix = (f"{{\\c{highlight_inline}"
                         f"\\fscx90\\fscy90\\t(0,110,\\fscx108\\fscy108)}}")
    else:
        active_prefix = f"{{\\c{highlight_inline}}}"

    active_style = ""
    if effect == "highlight":
        box_colour = hex_to_ass_color(highlight_color, 1.0, fallback="FFD700")
        on_box = "#000000" if _luminance(highlight_color) > 0.6 else font_color
        # Padding scales with the text so the box keeps its shape at any size.
        pad = max(2, round(final_fontsize * 0.12))
        active_style = (
            f"Style: Active,{safe_font},{final_fontsize},"
            f"{hex_to_ass_color(on_box, 1.0)},{hex_to_ass_color(on_box, 1.0)},"
            f"{box_colour},{box_colour},1,0,0,0,100,100,0,0,3,{pad},0,"
            f"{ass_alignment},10,10,{int(_clamp_number(margin_v, 0, 200, SAFE_MARGIN_V))},1\n"
        )

    header = (
        "[Script Info]\n"
        "ScriptType: v4.00+\n"
        "PlayResY: 288\n"
        "WrapStyle: 0\n"
        "ScaledBorderAndShadow: yes\n"
        "\n"
        "[V4+ Styles]\n"
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, "
        "OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, "
        "ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, "
        "Alignment, MarginL, MarginR, MarginV, Encoding\n"
        f"Style: Default,{safe_font},{final_fontsize},{primary_colour},{primary_colour},"
        f"{outline_colour},{back_colour},1,0,0,0,100,100,0,0,{border_style},"
        f"{outline_width},{shadow},{ass_alignment},10,10,{int(_clamp_number(margin_v, 0, 200, SAFE_MARGIN_V))},1\n"
        f"{active_style}"
        "\n"
        "[Events]\n"
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
    )

    events = []
    for block in blocks:
        for i, word in enumerate(block):
            # Event runs until the next word starts (no flicker in gaps);
            # the last word holds until the block ends.
            ev_start = block[0]['start'] if i == 0 else word['start']
            ev_end = block[i + 1]['start'] if i < len(block) - 1 else block[-1]['end']
            if ev_end <= ev_start:
                continue

            parts = []
            for j, other in enumerate(block):
                text = _escape_ass_text(other['word'])
                if uppercase:
                    text = text.upper()
                if j == i:
                    parts.append(f"{active_prefix}{text}{{\\r}}")
                elif reveal and j > i:
                    parts.append(f"{{\\alpha&HFF&}}{text}{{\\r}}")
                else:
                    parts.append(text)

            events.append(
                f"Dialogue: 0,{_ass_time(ev_start)},{_ass_time(ev_end)},Default,,0,0,0,,"
                f"{seam_prefix(ev_start)}{' '.join(parts)}"
            )

    if not events:
        return False

    with open(output_path, 'w', encoding='utf-8-sig') as f:
        f.write(header + "\n".join(events) + "\n")

    return True

def format_srt_block(index, start, end, text):
    def format_time(seconds):
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int((seconds - int(seconds)) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"
        
    return f"{index}\n{format_time(start)} --> {format_time(end)}\n{text}\n\n"

_HEX_COLOR_RE = re.compile(r'^[0-9A-Fa-f]{6}$')
_FONT_NAME_RE = re.compile(r'[^A-Za-z0-9 _-]')


def hex_to_ass_color(hex_color, opacity=1.0, fallback="FFFFFF"):
    """Convert #RRGGBB to ASS &HAABBGGRR format. opacity: 0.0=transparent, 1.0=opaque.

    Invalid hex (e.g. "#GGGGGG", None, wrong length) falls back to `fallback`
    instead of raising, so a bad color from the client can't 500 the request.
    """
    hex_digits = str(hex_color or "").lstrip('#')
    if not _HEX_COLOR_RE.match(hex_digits):
        hex_digits = fallback
    opacity = _clamp_number(opacity, 0.0, 1.0, 1.0)
    r = int(hex_digits[0:2], 16)
    g = int(hex_digits[2:4], 16)
    b = int(hex_digits[4:6], 16)
    alpha = round((1.0 - opacity) * 255)
    return f"&H{alpha:02X}{b:02X}{g:02X}{r:02X}"


def _luminance(hex_color):
    """Relative brightness 0-1 of #RRGGBB (Rec. 601 weights); 1.0 if invalid."""
    hex_digits = str(hex_color or "").lstrip('#')
    if not _HEX_COLOR_RE.match(hex_digits):
        return 1.0
    r, g, b = (int(hex_digits[i:i + 2], 16) for i in (0, 2, 4))
    return (0.299 * r + 0.587 * g + 0.114 * b) / 255


def _clamp_number(value, lo, hi, default):
    """Coerce value to float and clamp to [lo, hi]; use default if not numeric."""
    try:
        num = float(value)
    except (TypeError, ValueError):
        num = float(default)
    return max(lo, min(hi, num))


def _sanitize_font_name(name):
    """Strip anything but [A-Za-z0-9 _-] so the font name can't inject extra
    ASS override fields (commas/braces/backslashes) into force_style."""
    cleaned = _FONT_NAME_RE.sub('', str(name or '')).strip()
    return cleaned or "Verdana"


def subtitles_filter(srt_path, alignment=2, fontsize=16,
                     font_name="Verdana", font_color="#FFFFFF",
                     border_color="#000000", border_width=2,
                     bg_color="#000000", bg_opacity=0.0):
    """The -vf string burn_subtitles uses (also fed to hooks.add_hook_to_video)."""
    # Position mapping
    ass_alignment = 2
    align_lower = str(alignment).lower()
    if align_lower == 'top':
        ass_alignment = 6
    elif align_lower == 'middle':
        ass_alignment = 10
    elif align_lower == 'bottom':
        ass_alignment = 2

    # Font size scaling for ASS virtual resolution (PlayResY=288 default)
    # For vertical 1080x1920 video, we need larger text for readability
    final_fontsize = int(_clamp_number(fontsize, 10, 200, 16) * 0.85)
    if final_fontsize < 10:
        final_fontsize = 10

    safe_font_name = _sanitize_font_name(font_name)
    bg_opacity = _clamp_number(bg_opacity, 0.0, 1.0, 0.0)
    border_width = _clamp_number(border_width, 0, 10, 2)

    # Path handling for FFmpeg filter syntax
    safe_srt_path = _escape_ffmpeg_filter_value(srt_path)

    # Convert colors to ASS format and build style
    primary_colour = hex_to_ass_color(font_color, 1.0)

    if bg_opacity > 0:
        # Box mode: opaque background box
        border_style = 3
        outline_colour = hex_to_ass_color(bg_color, bg_opacity, fallback="000000")
        outline_width = 1
    else:
        # Outline mode: text border/outline
        border_style = 1
        outline_colour = hex_to_ass_color(border_color, 1.0, fallback="000000")
        outline_width = max(1, int(border_width))

    back_colour = hex_to_ass_color("#000000", 0.0)

    style_string = (
        f"Alignment={ass_alignment},"
        f"Fontname={safe_font_name},"
        f"Fontsize={final_fontsize},"
        f"PrimaryColour={primary_colour},"
        f"OutlineColour={outline_colour},"
        f"BackColour={back_colour},"
        f"BorderStyle={border_style},"
        f"Outline={outline_width},"
        f"Shadow=0,"
        f"MarginV={SAFE_MARGIN_V},"
        f"Bold=1"
    )

    # Let libass see the fonts bundled with the app (e.g. Anton for Impact)
    # even when the system fontconfig has no cache for them.
    fonts_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fonts")
    safe_fonts_dir = _escape_ffmpeg_filter_value(fonts_dir)

    # The first option is named explicitly (filename=) rather than positional:
    # ffmpeg 8's filtergraph parser rejects a quoted positional value that is
    # followed by more :name=value options ("No option name near ..."), while
    # the named form parses on every version back to 4.x.
    if str(srt_path).lower().endswith('.ass'):
        # ASS files (karaoke style) carry their own styles; force_style would
        # override the per-word color tags.
        vf = f"ass=filename='{safe_srt_path}':fontsdir='{safe_fonts_dir}'"
    else:
        vf = (f"subtitles=filename='{safe_srt_path}':fontsdir='{safe_fonts_dir}'"
              f":charenc=UTF-8:force_style='{style_string}'")

    return vf


def _video_size(path):
    """(width, height) of the first video stream, or None."""
    try:
        out = subprocess.check_output(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=width,height", "-of", "csv=p=0:s=x", path],
            stderr=subprocess.DEVNULL, timeout=60).decode().strip()
        w, h = out.split("\n")[0].split("x")[:2]
        return int(w), int(h)
    except Exception:
        return None


def burn_subtitles(video_path, srt_path, output_path, alignment=2, fontsize=16,
                   font_name="Verdana", font_color="#FFFFFF",
                   border_color="#000000", border_width=2,
                   bg_color="#000000", bg_opacity=0.0, mark=None):
    """
    Burns subtitles into the video using FFmpeg.
    Supports two modes:
    - Outline mode (bg_opacity=0): Text with colored outline/border
    - Box mode (bg_opacity>0): Text with semi-transparent background box

    ``mark``: optional (logo_path, make_filter, path): ALSO write ``path`` =
    the captioned picture with the free-plan mark (the served ``wm_`` copy,
    main.mark_delivery), from the same decode. ``make_filter(vw, vh, video=,
    logo=, out=)`` is main.watermark_filter.
    """
    vf = subtitles_filter(srt_path, alignment=alignment, fontsize=fontsize,
                          font_name=font_name, font_color=font_color,
                          border_color=border_color, border_width=border_width,
                          bg_color=bg_color, bg_opacity=bg_opacity)

    tail = ['-c:a', 'copy', *video_encode_args(QUALITY), *METADATA_SCRUB,
            '-movflags', '+faststart']
    outputs = [output_path]
    size = _video_size(video_path) if mark else None
    if size:
        logo_path, make_filter, mark_path = mark
        marked = make_filter(size[0], size[1], video="[sm]", logo="[1:v]", out="[m]")
        cmd = [
            'ffmpeg', '-y',
            '-i', video_path,
            '-i', logo_path,
            '-filter_complex', f"[0:v]{vf},split=2[s][sm];{marked}",
            '-map', '[s]', '-map', '0:a?', *tail, output_path,
            '-map', '[m]', '-map', '0:a?', *tail, mark_path,
        ]
        outputs.append(mark_path)
    else:
        cmd = ['ffmpeg', '-y', '-i', video_path, '-vf', vf, *tail, output_path]

    _log(f"🎬 Burning subtitles: {' '.join(cmd)}")
    result = gpu_decode.run(
        lambda on_gpu: gpu_decode.on_card(cmd, video_path) if on_gpu else cmd,
        lambda c: subprocess.run(c, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                                 timeout=1800),
        ok=lambda r: r.returncode == 0 or finished_before_crash(r, outputs),
        path=video_path)

    if result.returncode != 0 and not finished_before_crash(result, outputs):
        stderr_text = result.stderr.decode(errors='replace')
        _log(f"❌ FFmpeg Subtitle Error: {stderr_text}")
        raise Exception(f"FFmpeg failed: {stderr_text}")

    return True

