"""
Deterministic FFmpeg filter builder for AI edit decision lists.

The old Auto Edit asked Gemini to write a raw -vf filter string, which broke
in two ways: fragile syntax, and zooms that cropped burned-in captions/hooks
out of frame. Now Gemini only decides WHAT to do and WHEN (an edit decision
list); this module turns that list into a safe filter string with hard caps
(max zoom strength, no zooms over captioned videos, no overlapping zooms).

Stdlib-only so it stays unit-testable without FFmpeg or the Gemini SDK.
"""

# Per-type hard limits. "zoom" types re-frame the picture and are dropped
# entirely when the input already has burned-in captions/hooks.
EFFECT_LIMITS = {
    "zoom_in":    {"zoom": True,  "max_strength": 0.15, "default_strength": 0.10},
    "punch_in":   {"zoom": True,  "max_strength": 0.15, "default_strength": 0.09},
    "zoom_pulse": {"zoom": True,  "max_strength": 0.10, "default_strength": 0.07},
    "color_pop":  {"zoom": False, "max_strength": 1.0,  "default_strength": 0.5},
    "bw_moment":  {"zoom": False, "max_strength": 1.0,  "default_strength": 1.0},
    "flash":      {"zoom": False, "max_strength": 1.0,  "default_strength": 1.0},
    "vignette":   {"zoom": False, "max_strength": 1.0,  "default_strength": 1.0},
}

MAX_EDITS = 12
MIN_EFFECT_SECONDS = 0.15
MAX_ZOOM_SECONDS = 8.0
FLASH_SECONDS = 0.15

# Zoom window anchor: 45% of the height keeps headroom for faces (slightly
# above center) instead of cropping equally from top and bottom.
ZOOM_CENTER_Y = 0.45


def normalize_edits(edits, duration, has_captions=False):
    """Validate and clamp a raw AI edit list. Returns a cleaned, start-sorted
    list of dicts with keys: type, start, end, strength."""
    if not isinstance(edits, list) or duration <= 0:
        return []

    cleaned = []
    for entry in edits:
        if not isinstance(entry, dict):
            continue
        effect_type = str(entry.get("type", "")).strip().lower()
        limits = EFFECT_LIMITS.get(effect_type)
        if not limits:
            continue
        if has_captions and limits["zoom"]:
            # Burned-in captions/hooks must stay visible: no re-framing.
            continue
        try:
            start = float(entry.get("start", 0.0))
            end = float(entry.get("end", 0.0))
        except (TypeError, ValueError):
            continue
        start = max(0.0, min(start, duration))
        end = max(0.0, min(end, duration))
        if end - start < MIN_EFFECT_SECONDS:
            continue
        if limits["zoom"] and end - start > MAX_ZOOM_SECONDS:
            end = start + MAX_ZOOM_SECONDS
        try:
            strength = float(entry.get("strength") or 0.0)
        except (TypeError, ValueError):
            strength = 0.0
        if strength <= 0.0:
            strength = limits["default_strength"]
        strength = min(strength, limits["max_strength"])
        cleaned.append({"type": effect_type, "start": start, "end": end, "strength": strength})

    cleaned.sort(key=lambda e: (e["start"], e["end"]))

    # Zoom terms are summed inside one zoompan expression, so overlapping zoom
    # segments would stack into an oversized zoom — keep the first, drop the rest.
    result = []
    last_zoom_end = -1.0
    flash_count = 0
    for entry in cleaned:
        if EFFECT_LIMITS[entry["type"]]["zoom"]:
            if entry["start"] < last_zoom_end:
                continue
            last_zoom_end = entry["end"]
        if entry["type"] == "flash":
            if flash_count >= 2:
                continue
            flash_count += 1
        result.append(entry)
        if len(result) >= MAX_EDITS:
            break
    return result


def _zoom_term(entry, fps):
    """One additive term of the zoompan z expression for a zoom-type edit."""
    start_frame = int(round(entry["start"] * fps))
    end_frame = max(start_frame + 2, int(round(entry["end"] * fps)))
    span = end_frame - start_frame
    strength = entry["strength"]
    gate = f"between(on,{start_frame},{end_frame})"
    if entry["type"] == "zoom_in":
        # Linear ramp from 0 to full strength across the segment.
        return f"{strength:.4f}*clip((on-{start_frame})/{span},0,1)*{gate}"
    if entry["type"] == "zoom_pulse":
        # Triangular in-and-out peaking mid-segment.
        mid = (start_frame + end_frame) / 2.0
        half = max(span / 2.0, 1.0)
        return f"{strength:.4f}*(1-abs((on-{mid:.1f})/{half:.1f}))*{gate}"
    # punch_in: constant tighter framing for the whole segment.
    return f"{strength:.4f}*{gate}"


def build_filter_string(edits, duration, fps, width, height, has_captions=False):
    """Build a -vf filter string from an AI edit list.

    Returns (filter_string_or_None, applied_edits). None means "no edits" —
    the caller should keep the original video untouched.
    """
    applied = normalize_edits(edits, duration, has_captions=has_captions)
    if not applied:
        return None, []

    fps = float(fps) if fps and fps > 0 else 30.0
    width = int(width or 1080)
    height = int(height or 1920)

    color_filters = []
    zoom_terms = []
    for entry in applied:
        start, end, strength = entry["start"], entry["end"], entry["strength"]
        if EFFECT_LIMITS[entry["type"]]["zoom"]:
            zoom_terms.append(_zoom_term(entry, fps))
        elif entry["type"] == "color_pop":
            contrast = 1.0 + 0.15 * strength
            saturation = 1.0 + 0.6 * strength
            color_filters.append(
                f"eq=contrast={contrast:.2f}:saturation={saturation:.2f}:enable='between(t,{start:.2f},{end:.2f})'"
            )
        elif entry["type"] == "bw_moment":
            color_filters.append(f"hue=s=0:enable='between(t,{start:.2f},{end:.2f})'")
        elif entry["type"] == "flash":
            flash_end = min(start + FLASH_SECONDS, end)
            color_filters.append(f"eq=brightness=0.35:enable='between(t,{start:.2f},{flash_end:.2f})'")
        elif entry["type"] == "vignette":
            color_filters.append(f"vignette=angle=PI/4.5:enable='between(t,{start:.2f},{end:.2f})'")

    parts = list(color_filters)
    if zoom_terms:
        z_expr = "1+" + "+".join(zoom_terms)
        parts.append(
            f"zoompan=z='{z_expr}'"
            f":x='iw/2-(iw/zoom)/2'"
            f":y='ih*{ZOOM_CENTER_Y}-(ih/zoom)/2'"
            f":d=1:fps={fps:g}:s={width}x{height}"
        )

    if not parts:
        return None, []
    return ",".join(parts), applied
