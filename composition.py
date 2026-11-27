"""The layout engine for stream and game clips: how big the streamer's facecam
is, how much of the game stays visible, and where the captions go.

Everything here is geometry: numbers in, numbers out, no pixels and no
FFmpeg, so every rule is unit tested. facecam.py finds the webcam overlay and
builds the filtergraph from what ``stack`` returns; viral_edit.py and
reframe_v2.py render it; composition_check.py validates the result.

Why this exists (measured on real stream clips, 22-nov-2026):

- The streamer stack used a fixed 40% facecam band and cropped INSIDE the
  webcam overlay to that band's shape. A 16:9 webcam was cut to 1.4:1 and
  blown up ~2.8x: the streamer's face filled the top of the clip and was soft.
- Gameplay Priority showed the whole 16:9 game as a 32%-high strip in 68% of
  blur, so a corner webcam came out ~280 px wide: the reaction was invisible.
- The game band was always cropped to its full height, which on a 16:9 source
  keeps 46% of the width, centred, wherever the action was.

The fix is a sizing rule instead of a constant. The facecam band is sized by
what the viewer actually sees, the FACE (its share of the output height),
inside bounds set by a focus profile, and never upscaled past what the
overlay's real pixels can carry. When even a full-width band would need more
upscale than that, the webcam is shown at an honest size on a blurred card
instead of being stretched. The game keeps at least a profile share of its
width (the rest of its band is a blurred fill, which is also where captions
go) and its crop slides toward the action, never onto the webcam.

Focus profiles (``FRAMING_FOCUS`` per job, or the env default):

- ``streamer``: the reaction is the point; a large face, less game width.
- ``balanced``: the default stream clip.
- ``gameplay``: the game is the point; the face stays visible but smaller.
- ``auto``: ``streamer`` when the moment's measured signal is the streamer's
  own reaction (laughing, a voice peak well above the game), else ``balanced``.

Every bound can be overridden by env (``FACECAM_MIN_SHARE`` and friends, see
``profile``), which is how a campaign that needs a larger or smaller streamer
gets it without code.
"""
import math
import os

FOCUSES = ("auto", "streamer", "balanced", "gameplay")

# cam_*: facecam band as a share of the output height.
# face_*: the face's height as a share of the output height. This is the size
#   a viewer perceives; a band of the same height can hold a tiny or a huge
#   face depending on how the streamer frames their webcam.
# game_keep: the least share of the source WIDTH the game band shows.
PROFILES = {
    "streamer": {"cam_target": 0.42, "cam_min": 0.30, "cam_max": 0.48,
                 "face_min": 0.10, "face_target": 0.16, "face_max": 0.22,
                 "game_keep": 0.42},
    "balanced": {"cam_target": 0.34, "cam_min": 0.24, "cam_max": 0.40,
                 "face_min": 0.08, "face_target": 0.12, "face_max": 0.18,
                 "game_keep": 0.55},
    "gameplay": {"cam_target": 0.26, "cam_min": 0.18, "cam_max": 0.32,
                 "face_min": 0.06, "face_target": 0.09, "face_max": 0.14,
                 "game_keep": 0.72},
}

# The most a webcam overlay is enlarged. Past this the picture is visibly
# soft; the layout shows it smaller on a card instead of pretending.
MAX_CAM_UPSCALE = 2.5
# The face's place down the facecam crop (room for headroom above).
FACE_POSITION = 0.55
# Captions sit in the blurred fill under the game when at least this much of
# it (share of the output height) is above PLATFORM_UI_TOP; otherwise on the
# facecam/game seam.
CAPTION_FILL_MIN = 0.08
# TikTok / Reels / Shorts draw the username, caption and buttons below this.
PLATFORM_UI_TOP = 0.80
# Gameplay Priority without a usable facecam: zoom the full-width picture by
# this much (crop the sides), so the game is not a 32% strip in a sea of blur.
GAMEPLAY_ZOOM = 1.25
# Voice excitement (z-score against the streamer's own normal) from which
# an automatic focus treats the clip as a reaction clip.
STREAMER_EXCITEMENT = 3.0
# A game crop moves off the frame centre only when it frames this much more
# of the clip's motion.
ACTION_MIN_GAIN = 0.10


def _env_float(name, default, lo, hi):
    try:
        value = float(os.environ.get(name, "").strip() or default)
    except ValueError:
        return default
    return min(hi, max(lo, value)) if math.isfinite(value) else default


def resolve_focus(focus=None, signals=None, ratings=None):
    """The concrete profile name for a requested focus.

    ``auto`` follows the streamer when the moment IS their reaction: the
    clip judge rated the reaction strong while the visual action is not
    (``ratings``, hype_judge), or the measured signals (``signals``,
    hype_engine.describe) say they laughed or their voice peaked far above
    their own normal. A strong play gets the game; everything else is
    balanced."""
    focus = (focus if focus is not None else os.environ.get("FRAMING_FOCUS", "auto")) or "auto"
    focus = str(focus).strip().lower()
    if focus not in FOCUSES:
        focus = "auto"
    if focus != "auto":
        return focus
    ratings = ratings if isinstance(ratings, dict) else {}
    if ratings.get("reaction") == "strong" and ratings.get("visual_action") != "strong":
        return "streamer"
    if ratings.get("visual_action") == "strong" and ratings.get("reaction") == "weak":
        return "gameplay"
    streamer = (signals or {}).get("streamer") if isinstance(signals, dict) else None
    if isinstance(streamer, dict) and not ratings:
        try:
            excitement = float(streamer.get("excitement") or 0.0)
        except (TypeError, ValueError):
            excitement = 0.0
        if streamer.get("laughing") or excitement >= STREAMER_EXCITEMENT:
            return "streamer"
    return "balanced"


def profile(focus="balanced"):
    """The bounds for ``focus`` with any env overrides applied."""
    p = dict(PROFILES.get(focus, PROFILES["balanced"]))
    for key, env in (("cam_min", "FACECAM_MIN_SHARE"), ("cam_max", "FACECAM_MAX_SHARE"),
                     ("cam_target", "FACECAM_TARGET_SHARE"), ("face_min", "FACE_MIN_SHARE"),
                     ("face_max", "FACE_MAX_SHARE"), ("game_keep", "GAME_KEEP_MIN")):
        if os.environ.get(env, "").strip():
            p[key] = _env_float(env, p[key], 0.02, 0.9)
    p["cam_min"] = min(p["cam_min"], p["cam_max"])
    p["cam_target"] = min(max(p["cam_target"], p["cam_min"]), p["cam_max"])
    p["max_upscale"] = _env_float("FACECAM_MAX_UPSCALE", MAX_CAM_UPSCALE, 1.0, 6.0)
    return p


def _even(v):
    """An even SIZE (at least 2)."""
    v = int(round(v))
    return max(2, v - (v % 2))


def _even_at(v):
    """An even POSITION (0 allowed), rounded down so a crop never grows past
    the edge it was clamped to."""
    v = max(0, int(math.floor(v + 1e-6)))
    return v - (v % 2)


def _overlap(a, b):
    """Area shared by two (x, y, w, h) boxes."""
    w = min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0])
    h = min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1])
    return max(0.0, w) * max(0.0, h)


def cam_band(out_w, out_h, cam, prof):
    """Facecam band geometry, or None when the webcam cannot be shown at a
    readable size without being blown up past its pixels.

    A search over band heights (cam_min..cam_max of the output) and scales
    (up to the profile's upscale cap): at each, the crop of the overlay that
    fits the band, centred on the face. The face's resulting share of the
    output height is held between the profile's face_min and face_max and
    drawn to face_target; a band the picture does not fill (a narrow webcam
    shown at an honest scale) becomes a card over a blurred copy of itself,
    and costs a little so a full band wins when it can. The band height is
    drawn to cam_target."""
    box, face = cam["box"], cam["face"]
    # Several people on one webcam: frame all of them, size by the main face.
    people = cam.get("people") or face
    bx, by, bw, bh = (float(v) for v in box)
    fh = float(face[3])
    fw = max(float(face[2]), float(people[2]))
    ph = max(fh, float(people[3]))
    lo, hi = int(prof["cam_min"] * out_h), int(prof["cam_max"] * out_h)
    cap = prof["max_upscale"]
    full_width_scale = out_w / bw
    if cap < full_width_scale <= cap * 1.12:
        # Two slivers of blur around an almost full-width card look worse
        # than a slightly softer webcam: fill the width at its own size.
        cap = full_width_scale * 1.0001
    scales = sorted({round(cap * k / 40.0, 4) for k in range(12, 41)} | {round(min(cap, full_width_scale), 4)})
    best = None
    for cam_h in range(lo, hi + 1, 8):
        for scale in scales:
            cw = min(bw, out_w / scale)
            ch = min(bh, cam_h / scale)
            if cw < min(bw, 1.6 * fw) or ch < min(bh, 1.35 * ph):
                continue                      # the crop would cut into the head
            face_share = fh * scale / out_h
            card_w, card_h = cw * scale, ch * scale
            filled = (card_w * card_h) / float(out_w * cam_h)
            miss = 0.0
            if face_share > prof["face_max"]:
                miss = 4.0 * (face_share - prof["face_max"])
            elif face_share < prof["face_min"]:
                miss = 2.0 * (prof["face_min"] - face_share)
            score = (miss + abs(face_share - prof["face_target"])
                     + 0.25 * abs(cam_h / float(out_h) - prof["cam_target"])
                     + 0.06 * (1.0 - filled))
            if best is None or score < best[0] - 1e-9:
                best = (score, cam_h, scale, cw, ch, face_share)
    if best is None:
        return None
    _, cam_h, scale, cw, ch, face_share = best
    if face_share < 0.75 * prof["face_min"]:
        return None
    if cw * scale < out_w - 4 and cw * scale >= 0.9 * out_w and full_width_scale <= cap + 1e-6:
        # Within 10% of the width: fill it rather than leave two slivers.
        scale = full_width_scale
        cw, ch = bw, min(bh, cam_h / scale)
        face_share = fh * scale / out_h
    # A band taller than its picture only adds stripes of blur: shrink it
    # to the picture when the profile's minimum allows.
    cam_h = _even(max(lo, min(cam_h, ch * scale)))
    cx = min(max(people[0] + people[2] / 2.0 - cw / 2.0, bx), bx + bw - cw)
    cy = min(max(people[1] + people[3] / 2.0 - FACE_POSITION * ch, by), by + bh - ch)
    crop = (_even(cw), _even(ch), _even_at(cx), _even_at(cy))
    # Rounding must not carry the crop past the overlay's edge.
    crop = (min(crop[0], int(bx + bw) - crop[2] - (int(bx + bw) - crop[2]) % 2),
            min(crop[1], int(by + bh) - crop[3] - (int(by + bh) - crop[3]) % 2), crop[2], crop[3])
    card_w, card_h = _even(crop[0] * scale), _even(crop[1] * scale)
    card = None
    if card_w < out_w - 4 or card_h < cam_h - 4:
        card_w, card_h = min(card_w, out_w), min(card_h, cam_h)
        card = {"w": card_w, "h": card_h, "x": _even((out_w - card_w) / 2.0),
                "y": _even((cam_h - card_h) / 2.0)}
    return {"cam_h": cam_h, "cam_crop": crop, "cam_scale": round(scale, 3),
            "face_share": round(face_share, 4), "card": card}


def game_band(orig_w, orig_h, out_w, game_h, cam_box, prof, action=None):
    """Gameplay crop for a band ``out_w`` x ``game_h`` that never shows the
    webcam overlay a second time, or None when no such crop exists.

    The crop is full source height. It keeps at least ``game_keep`` of the
    width; when filling the band would keep less, the picture is shorter
    than the band and a blurred copy of the game fills the rest (``fill``).
    Its position: the clip's action (``action``, a centre as a share of the
    width) when given, else the frame centre, then slid clear of the
    overlay's columns."""
    bx, by, bw, bh = cam_box
    fill_w = orig_h * out_w / float(game_h)          # crop width that fills the band
    gw = min(float(orig_w), max(fill_w, prof["game_keep"] * orig_w))
    # The widest crop that can sit beside the overlay: the overlay's columns
    # are off limits because the crop is full height.
    room = max(bx, orig_w - (bx + bw))
    if gw > room:
        # Narrower than the profile wants, but still filling the band.
        gw = max(fill_w, room)
        if gw > room + 1:
            return None
    pic_h = min(game_h, out_w * orig_h / gw)
    want = (action if action is not None else 0.5) * orig_w - gw / 2.0
    spots = [want, (orig_w - gw) / 2.0, 0.0, orig_w - gw, bx + bw, bx - gw]
    spots = [min(max(s, 0.0), orig_w - gw) for s in spots]
    for gx in sorted(set(spots), key=lambda s: abs(s - want)):
        crop = (gx, 0.0, gw, orig_h)
        if _overlap(crop, cam_box) <= 0.02 * gw * orig_h:
            gx_e = _even_at(gx)
            gw_e = min(_even(gw), (orig_w - gx_e) - (orig_w - gx_e) % 2)
            return {"game_crop": (gw_e, _even(orig_h) if orig_h % 2 == 0 else orig_h - 1, gx_e, 0),
                    "game_pic_h": _even(pic_h), "game_keep": round(gw / orig_w, 3)}
    return None


def why_not(orig_w, orig_h, out_w, out_h, cam, focus="balanced"):
    """Why ``stack`` gave None, in words for the log and the review."""
    prof = profile(focus)
    if cam_band(out_w, out_h, cam, prof) is None:
        return "the webcam is too small to enlarge without blur"
    return "the game cannot be cropped clear of the webcam"


def stack(orig_w, orig_h, out_w, out_h, cam, *, focus="balanced", action=None,
          cam_position="top"):
    """The full streamer-stack geometry, or None when the game cannot be
    shown without the webcam appearing twice, or the webcam is too small to
    enlarge honestly (callers then keep the whole picture in one foreground:
    one face, always).

    Returns cam band, game band, where the seam is, where captions go and
    the metrics composition_check validates."""
    prof = profile(focus)
    cam_part = cam_band(out_w, out_h, cam, prof)
    if cam_part is None:
        return None
    cam_h = cam_part["cam_h"]
    game_h = out_h - cam_h
    game = game_band(orig_w, orig_h, out_w, game_h, cam.get("avoid", cam["box"]), prof, action)
    if game is None:
        return None
    fill_h = game_h - game["game_pic_h"]
    top = cam_position != "bottom"
    if top:
        seam = cam_h
        game_top = cam_h
        pic = (game_top, game_top + game["game_pic_h"])
        fill = (pic[1], out_h) if fill_h > 0 else None
    else:
        seam = game_h
        pic = (max(0, game_h - game["game_pic_h"]), game_h)
        fill = (0, pic[0]) if fill_h > 0 else None
    # Captions: in the blurred fill when enough of it sits above the
    # platforms' bottom UI (where the fill otherwise just keeps the game out
    # from under the username and buttons), else on the seam.
    caption_y = float(seam)
    if fill and top:
        usable = (fill[0], min(fill[1], PLATFORM_UI_TOP * out_h))
        if usable[1] - usable[0] >= CAPTION_FILL_MIN * out_h:
            caption_y = (usable[0] + usable[1]) / 2.0
    return {
        **cam_part, **game,
        "game_h": _even(game_h),
        "cam_position": "top" if top else "bottom",
        "focus": focus,
        "seam": seam / float(out_h),
        "caption_y": round(caption_y / float(out_h), 4),
        "content": (round(pic[0] / float(out_h), 4), round(pic[1] / float(out_h), 4)),
        "fill_share": round(max(0, fill_h) / float(out_h), 4),
        "out": (out_w, out_h),
    }


def gameplay_zoom(orig_w, orig_h, out_w, out_h, *, cam_box=None, action=None, zoom=None):
    """Gameplay Priority's foreground crop: (x, w) of the source columns, or
    None for the whole width.

    The full width is a 32% strip on 9:16. A moderate zoom makes the game
    readable; it never crops into an embedded facecam (the crop must hold the
    whole overlay) and slides toward the action."""
    if zoom is None:
        zoom = _env_float("GAMEPLAY_ZOOM", GAMEPLAY_ZOOM, 1.0, 1.6)
    full_h = out_w * orig_h / float(orig_w)
    zoom = min(zoom, out_h / full_h)
    if zoom <= 1.01 or orig_w / float(orig_h) < 1.2:
        return None
    w = orig_w / zoom
    lo, hi = 0.0, orig_w - w
    if cam_box:
        bx, _, bw, _ = cam_box
        if bw > w:
            return None
        lo, hi = max(lo, bx + bw - w), min(hi, bx)
        if lo > hi:
            return None
    want = (action if action is not None else 0.5) * orig_w - w / 2.0
    x = min(max(want, lo), hi)
    return _even_at(x), _even(w)


def action_centre(profile_cols, frac, mask=None):
    """Centre (share of the width) of the window of width ``frac`` holding
    the most motion, or None when it gains less than ACTION_MIN_GAIN over the
    centred window. ``mask``: (start, end) column shares to ignore (the
    webcam overlay moves too)."""
    cols = [max(0.0, float(v)) for v in profile_cols]
    n = len(cols)
    if mask:
        a, b = int(mask[0] * n), int(math.ceil(mask[1] * n))
        for i in range(max(0, a), min(n, b)):
            cols[i] = 0.0
    total = sum(cols)
    if not n or total <= 0 or frac >= 1.0:
        return None
    win = max(1, min(n, int(round(frac * n))))
    sums, acc = [], sum(cols[:win])
    sums.append(acc)
    for i in range(win, n):
        acc += cols[i] - cols[i - win]
        sums.append(acc)
    centred = (n - win) // 2
    best = max(sums)
    if (best - sums[centred]) / total < ACTION_MIN_GAIN:
        return None
    near = [i for i, s in enumerate(sums) if s >= best - 0.02 * total]
    start = min(near, key=lambda i: abs(i - centred))
    return (start + win / 2.0) / n
