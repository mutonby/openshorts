"""9:16 TikTok/Shorts template: a hook card for the first seconds plus a permanent source credit."""
import os
import subprocess
import tempfile
import textwrap

from PIL import Image, ImageDraw, ImageFont

W, H = 1080, 1920
HOOK_SECONDS = 3.0
FONTS = (
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
)


def _font(size):
    for path in FONTS:
        if os.path.exists(path):
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def _lines(draw, text, font, max_width):
    lines, cur = [], ""
    for word in text.split():
        trial = f"{cur} {word}".strip()
        if cur and draw.textlength(trial, font=font) > max_width:
            lines.append(cur)
            cur = word
        else:
            cur = trial
    return lines + [cur] if cur else [text]


def hook_card(text, path, y=330):
    """Yellow-on-black card, big enough to read before the thumb moves on."""
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    font = _font(92)
    lines = _lines(d, text.upper(), font, W - 200)
    line_h = 118
    box_h = line_h * len(lines) + 70
    d.rounded_rectangle((50, y, W - 50, y + box_h), 36, fill=(0, 0, 0, 215))
    for i, line in enumerate(lines):
        w = d.textlength(line, font=font)
        d.text(((W - w) / 2, y + 35 + i * line_h), line, font=font, fill=(255, 221, 0, 255),
               stroke_width=3, stroke_fill=(0, 0, 0, 255))
    img.save(path)


def credit_card(text, path, y=70):
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    font = _font(34)
    w = d.textlength(text, font=font)
    d.rounded_rectangle(((W - w) / 2 - 24, y, (W + w) / 2 + 24, y + 62), 18, fill=(0, 0, 0, 150))
    d.text(((W - w) / 2, y + 10), text, font=font, fill=(255, 255, 255, 235))
    img.save(path)


def render(video, hook, credit, output, hook_seconds=HOOK_SECONDS):
    """Scale/pad to 1080x1920, show the hook card for ``hook_seconds`` (pop-in), credit throughout."""
    with tempfile.TemporaryDirectory() as tmp:
        hook_png, credit_png = os.path.join(tmp, "hook.png"), os.path.join(tmp, "credit.png")
        hook_card(hook, hook_png)
        credit_card(credit, credit_png)
        graph = (
            f"[0:v]scale={W}:{H}:force_original_aspect_ratio=decrease,"
            f"pad={W}:{H}:(ow-iw)/2:(oh-ih)/2:black,setsar=1[base];"
            f"[1:v]format=rgba,fade=t=in:st=0:d=0.25:alpha=1,"
            f"fade=t=out:st={hook_seconds - 0.3}:d=0.3:alpha=1[hook];"
            f"[base][hook]overlay=0:0:enable='lt(t,{hook_seconds})'[a];"
            f"[a][2:v]overlay=0:0[v]"
        )
        cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", video,
               "-loop", "1", "-i", hook_png, "-loop", "1", "-i", credit_png,
               "-filter_complex", graph, "-map", "[v]", "-map", "0:a?",
               "-c:v", "libx264", "-preset", "fast", "-crf", "20", "-pix_fmt", "yuv420p",
               "-c:a", "copy", "-shortest", "-movflags", "+faststart", output]
        subprocess.run(cmd, check=True)
    return output
