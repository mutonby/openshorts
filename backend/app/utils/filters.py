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


# ---------------------------------------------------------------------------
# AI-filter safety allowlist (Codex Phase 5 focus 1 BLOCKER).
#
# /api/edit and auto_pipeline.apply_ai_edit pass LLM-produced filter strings
# to FFmpeg via ``-vf``. Without an allowlist, a malicious Gemini response
# (or a prompt-injected transcript) can include filters that read arbitrary
# files (``movie``, ``amovie``, ``subtitles``, ``ass``) or perform other
# filesystem side effects.
#
# Strategy: parse the chain on ``,`` and ``;``, strip ``[label]`` brackets,
# extract the leading filter name (chars up to ``=``/``:``), and confirm
# it's in ``_ALLOWED_FILTERS``. Fails closed for unknown filters.
# ---------------------------------------------------------------------------


class UnsafeFilterError(ValueError):
    """Raised when an LLM-produced filter string contains a disallowed filter."""


# Filters the AI-effect prompt instructs Gemini to use, plus essentials the
# pipeline injects (scale/setsar/fps/format) and a handful of common visual
# effects that are safe.
_ALLOWED_FILTERS = frozenset({
    # Geometry / colorspace (used by build_concat_args etc.)
    "scale", "crop", "pad", "format", "setsar", "fps",
    "hflip", "vflip", "transpose", "rotate",
    # Time-based visual effects from the Gemini prompt
    "zoompan", "fade",
    # Color / tone (prompt-listed)
    "eq", "hue", "curves", "vibrance",
    "colorbalance", "colorchannelmixer",
    "lutyuv", "lutrgb", "lut",
    # Sharpen / blur
    "unsharp", "smartblur", "gblur", "boxblur",
    # Misc safe primitives
    "vignette", "edgedetect", "noise", "drawbox",
    # Aspect helpers for normalization
    "null", "copy", "trim", "setpts",
    # Audio safe primitives (filter graphs may include these for muxing).
    "anull", "acopy", "aresample", "aformat", "atrim", "asetpts",
})


# Explicit deny-list, in case an allowlist gap ever appears. These filter
# names imply filesystem reads / writes from a string parameter and must
# never be invocable via LLM-generated content.
_DISALLOWED_FILTERS = frozenset({
    "movie", "amovie",      # arbitrary file read
    "subtitles", "ass",     # arbitrary subtitle file read
    "concat",               # accepts file path via :f= option
    "sendcmd", "asendcmd",  # external command channel
})


_BRACKET_LABEL_RE = re.compile(r"\[[^\]]*\]")
_FILTER_NAME_RE = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)")


def _iter_filter_nodes(filter_string: str):
    """Yield (name, raw_node_text) for each filter in the chain.

    Handles both ``,`` (chain) and ``;`` (filter_complex chains-of-chains)
    separators, and strips leading/trailing ``[label]`` brackets.
    """
    # Split on ';' first, then on ',' (respecting single-quoted strings).
    for sub_chain in filter_string.split(";"):
        for node in split_filter_chain(sub_chain):
            cleaned = _BRACKET_LABEL_RE.sub("", node).strip()
            if not cleaned:
                continue
            m = _FILTER_NAME_RE.match(cleaned)
            if not m:
                continue
            yield m.group(1).lower(), cleaned


def validate_filter_string(filter_string: str) -> None:
    """Validate ``filter_string`` against the AI-filter allowlist.

    Raises ``UnsafeFilterError`` if any node uses a filter name outside
    ``_ALLOWED_FILTERS`` (or explicitly listed in ``_DISALLOWED_FILTERS``).
    An empty string is treated as "no filter" — allowed.
    """
    if not isinstance(filter_string, str):
        raise TypeError(
            f"filter_string must be str, got {type(filter_string).__name__}"
        )
    if not filter_string.strip():
        return

    for name, _node in _iter_filter_nodes(filter_string):
        if name in _DISALLOWED_FILTERS:
            raise UnsafeFilterError(
                f"Disallowed FFmpeg filter '{name}' in AI-generated filter string "
                f"(filesystem / side-effect risk)"
            )
        if name not in _ALLOWED_FILTERS:
            raise UnsafeFilterError(
                f"FFmpeg filter '{name}' is not in the AI-filter allowlist. "
                f"Add it to app.utils.filters._ALLOWED_FILTERS if it is safe."
            )
