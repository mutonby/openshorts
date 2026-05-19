"""Shared FFmpeg filter helpers: chain splitting, sanitization, zoompan size enforcement.

These were originally private statics on editor.VideoEditor; moved here so the
motion-graphics and audio compositors can reuse them without importing the
editing module. VideoEditor still re-exposes them as static/classmethods for
backwards compatibility.
"""
import re


def split_filter_chain(filter_string: str) -> list:
    """Split a -vf filter chain on commas, respecting single-quoted substrings."""
    parts = []
    start = 0
    in_quote = False
    for i, ch in enumerate(filter_string):
        if ch == "'":
            in_quote = not in_quote
        elif ch == "," and not in_quote:
            parts.append(filter_string[start:i])
            start = i + 1
    parts.append(filter_string[start:])
    return parts


def enforce_zoompan_output_size(filter_string: str, width: int, height: int) -> str:
    """Force any zoompan filter to output the same geometry as the input clip."""
    parts = split_filter_chain(filter_string)
    out_parts = []
    for part in parts:
        if "zoompan=" in part:
            # Force s=WxH inside zoompan options (digitsxdigits only).
            if re.search(r":s=\d+x\d+", part):
                part = re.sub(r":s=\d+x\d+", f":s={width}x{height}", part)
            else:
                part = f"{part}:s={width}x{height}"
        out_parts.append(part)
    return ",".join(out_parts)


# Order matters: handle >= / <= before > / <
_COMPARISON_PATTERNS = [
    (re.compile(r"(?<![A-Za-z0-9_])([A-Za-z_]\w*)\s*>=\s*(-?\d+(?:\.\d+)?)"), r"gte(\1,\2)"),
    (re.compile(r"(?<![A-Za-z0-9_])([A-Za-z_]\w*)\s*<=\s*(-?\d+(?:\.\d+)?)"), r"lte(\1,\2)"),
    (re.compile(r"(?<![A-Za-z0-9_])([A-Za-z_]\w*)\s*>\s*(-?\d+(?:\.\d+)?)"), r"gt(\1,\2)"),
    (re.compile(r"(?<![A-Za-z0-9_])([A-Za-z_]\w*)\s*<\s*(-?\d+(?:\.\d+)?)"), r"lt(\1,\2)"),
]


def sanitize_filter_string(filter_string: str) -> str:
    """
    Best-effort sanitizer for Gemini-generated FFmpeg expressions.
    Converts comparison operators (t<3, on>=75, etc.) into FFmpeg expr functions
    (lt(), gte(), ...), which are far more reliably parsed across FFmpeg builds.
    """
    s = filter_string
    for pat, repl in _COMPARISON_PATTERNS:
        s = pat.sub(repl, s)
    return s
