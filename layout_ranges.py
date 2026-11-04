"""Which stretches of a rendered clip use which layout, kept next to the clip.

reframe_v2 decides a layout per scene (TRACK, GENERAL, SPLIT, ...) and renders
it, and nothing downstream used to know. Captions need to: on a SPLIT scene
the two speakers sit one above the other and the seam between the halves is
the emptiest place in the frame, so that is where the captions go (OpusClip
does the same). On every other layout they stay at the bottom.

A WIDE stretch also records ``content``: the (top, bottom) frame fractions its
picture fills over the blurred backdrop, so the auto-placed hook can sit in
the blur instead of over the action (hook_placement.content_bands).

The ranges travel as a sidecar ``<clip>.layout.json`` written by the render,
because the render happens in four different places (the job pipeline, the
recut/rerender, the manual reframe and the v1 fallback, which writes none)
and a return value would have to be threaded through all of them. Times are
in the CLIP's own timeline, which is also the timeline captions are laid on.
Whoever updates a clip's metadata copies them into ``layout_ranges`` so
/api/subtitle can find them after the sidecar is gone.
"""
import json
import os

SIDECAR_SUFFIX = ".layout.json"


def sidecar_path(video_path):
    return video_path + SIDECAR_SUFFIX


def write(video_path, ranges):
    """``ranges``: iterable of (start_s, end_s, strategy) or (start_s, end_s,
    strategy, content), ``content`` being the (top, bottom) frame fractions
    the picture fills on a layout that floats it over a blurred backdrop
    (WIDE), or None. Best effort: a failure here must never fail a render that
    already succeeded."""
    payload = []
    for r in ranges:
        s, e, layout = r[:3]
        entry = {"start": round(float(s), 3), "end": round(float(e), 3),
                 "layout": str(layout).lower()}
        band = _band(r[3] if len(r) > 3 else None)
        if band:
            entry["content"] = list(band)
        payload.append(entry)
    try:
        with open(sidecar_path(video_path), "w") as f:
            json.dump({"ranges": payload}, f)
    except OSError:
        pass
    return payload


def read(video_path):
    """The ranges recorded for this file, or [] (no sidecar, v1 render, or
    a file that was never reframed)."""
    try:
        with open(sidecar_path(video_path)) as f:
            data = json.load(f)
    except (OSError, ValueError):
        return []
    return normalise(data.get("ranges"))


def _band(value):
    """A valid (top, bottom) content band in frame fractions, or None."""
    try:
        top, bottom = (float(v) for v in value)
    except (TypeError, ValueError):
        return None
    if 0.0 <= top < bottom <= 1.0:
        return top, bottom
    return None


def normalise(ranges):
    """Accept the sidecar/metadata shape or bare tuples; drop anything odd
    (an old metadata file, a hand-edited value) instead of raising. A
    ``content`` band is kept when it is valid and dropped otherwise."""
    out = []
    for r in ranges or []:
        try:
            if isinstance(r, dict):
                s, e, layout = r["start"], r["end"], r.get("layout", "")
                band = r.get("content")
            else:
                s, e, layout = r[:3]
                band = r[3] if len(r) > 3 else None
            s, e = float(s), float(e)
        except (KeyError, TypeError, ValueError):
            continue
        if e > s:
            entry = {"start": s, "end": e, "layout": str(layout).lower()}
            band = _band(band)
            if band:
                entry["content"] = list(band)
            out.append(entry)
    return out


def split_ranges(ranges):
    """(start, end) pairs of the stretches where captions belong on the seam."""
    return [(r["start"], r["end"]) for r in normalise(ranges) if r["layout"] == "split"]


def in_split(t, splits):
    return any(s <= t < e for s, e in splits or [])


def remap(ranges, segments):
    """Carry ranges across a cut-and-concat. ``segments`` are the kept
    stretches of the SOURCE file in order ({"start", "end"}); the output is
    those segments back to back, so each range is clipped to every segment it
    overlaps and shifted to where that segment landed. This is what the fast
    rerender needs: it never reframes, so the only layout information it can
    have is the canonical clip's, seen through the new cut."""
    out = []
    offset = 0.0
    for seg in segments or []:
        try:
            seg_s, seg_e = float(seg["start"]), float(seg["end"])
        except (KeyError, TypeError, ValueError):
            continue
        if seg_e <= seg_s:
            continue
        for r in normalise(ranges):
            s, e = max(r["start"], seg_s), min(r["end"], seg_e)
            if e > s:
                moved = {"start": round(s - seg_s + offset, 3),
                         "end": round(e - seg_s + offset, 3),
                         "layout": r["layout"]}
                if "content" in r:
                    moved["content"] = r["content"]
                out.append(moved)
        offset += seg_e - seg_s
    return out
