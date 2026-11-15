import json
import os
import subprocess
import time
import uuid

SETTINGS_PATH = os.path.join("output", ".text_watermark.json")
POSITIONS = {
    "top-left": 7, "top": 8, "top-right": 9,
    "left": 4, "center": 5, "right": 6,
    "bottom-left": 1, "bottom": 2, "bottom-right": 3,
}
DEFAULTS = {"enabled": False, "text": "", "opacity": 0.5, "size": 4.0, "position": "bottom-right"}
FONT = "Montserrat ExtraBold"
MAX_TEXT = 80


def _num(value, lo, hi, default):
    try:
        v = float(value)
    except (TypeError, ValueError):
        return default
    if v != v:
        return default
    return min(hi, max(lo, v))


def clean(raw):
    raw = raw if isinstance(raw, dict) else {}
    text = " ".join(str(raw.get("text") or "").split())[:MAX_TEXT]
    position = raw.get("position") if raw.get("position") in POSITIONS else DEFAULTS["position"]
    return {
        "enabled": bool(raw.get("enabled")) and bool(text),
        "text": text,
        "opacity": round(_num(raw.get("opacity"), 0.05, 1.0, DEFAULTS["opacity"]), 2),
        "size": round(_num(raw.get("size"), 1.0, 15.0, DEFAULTS["size"]), 1),
        "position": position,
    }


def load(path=SETTINGS_PATH):
    try:
        with open(path, encoding="utf-8") as f:
            return clean(json.load(f))
    except (OSError, ValueError):
        return dict(DEFAULTS)


def save(settings, path=SETTINGS_PATH):
    data = clean(settings)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = f"{path}.{uuid.uuid4().hex[:6]}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
    os.replace(tmp, path)
    return data


def _escape(text):
    return (str(text).replace("\\", "/").replace("{", "(").replace("}", ")")
            .replace("\r", " ").replace("\n", " "))


def ass_document(settings, width, height):
    s = clean(settings)
    font_size = max(8, round(height * s["size"] / 100))
    margin = max(8, round(min(width, height) * 0.04))
    alpha = f"{round((1 - s['opacity']) * 255):02X}"
    outline = max(1, round(font_size * 0.07))
    return "\n".join([
        "[Script Info]",
        "ScriptType: v4.00+",
        f"PlayResX: {int(width)}",
        f"PlayResY: {int(height)}",
        "WrapStyle: 2",
        "ScaledBorderAndShadow: yes",
        "",
        "[V4+ Styles]",
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, "
        "BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, "
        "BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding",
        f"Style: Mark,{FONT},{font_size},&H{alpha}FFFFFF,&H{alpha}FFFFFF,&H{alpha}000000,"
        f"&HFF000000,1,0,0,0,100,100,0,0,1,{outline},0,{POSITIONS[s['position']]},"
        f"{margin},{margin},{margin},1",
        "",
        "[Events]",
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
        f"Dialogue: 0,0:00:00.00,9:59:59.99,Mark,,0,0,0,,{_escape(s['text'])}",
        "",
    ])


def apply(video_path, settings=None):
    s = clean(settings) if settings is not None else load()
    if not s["enabled"]:
        return None
    import subtitles
    from ffmpeg_utils import video_encode_args, QUALITY, METADATA_SCRUB
    size = subtitles._video_size(video_path)
    if not size:
        return None
    out_dir = os.path.dirname(video_path)
    ass_path = os.path.join(out_dir, f"textmark_{uuid.uuid4().hex[:8]}.ass")
    out_path = os.path.join(out_dir, f"branded_{int(time.time())}_{os.path.basename(video_path)}")
    with open(ass_path, "w", encoding="utf-8") as f:
        f.write(ass_document(s, size[0], size[1]))
    try:
        vf = subtitles.subtitles_filter(ass_path)
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", video_path, "-vf", vf,
                        "-map", "0:v:0", "-map", "0:a?", "-c:a", "copy",
                        *video_encode_args(QUALITY), *METADATA_SCRUB,
                        "-movflags", "+faststart", out_path],
                       check=True, capture_output=True, timeout=900)
    except Exception as e:
        print(f"   ⚠️ Text watermark failed ({e}) — clip ships without it.")
        try:
            os.remove(out_path)
        except OSError:
            pass
        return None
    finally:
        try:
            os.remove(ass_path)
        except OSError:
            pass
    print(f"   🏷️ Text watermark: {os.path.basename(out_path)}")
    return out_path


