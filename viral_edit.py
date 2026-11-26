"""The viral edit for stream clips, in one encode.

A highlight that plays as raw footage loses viewers in its first seconds. What
editors do by hand, this does from the measured reaction peak:

- cold open: the clip opens on ~2 s of the peak reaction, then jumps back to
  the setup and plays through (it reaches the peak again with its context);
- punch-in: a quick push toward the streamer (the facecam band in the
  streamer layout, the whole frame otherwise) on each appearance of the peak;
- dead air: long speech gaps away from the peak are tightened (conservative:
  gameplay keeps moving while nobody talks);
- emoji: stickers on reaction words ("no way" 😱, "haha" 😂, "let's go" 🔥);
- sound effects (sfx.py): a whoosh on the jump back, a boom on the punch.

The layout is the streamer stack (facecam.py) when a webcam overlay is found,
else the full-picture gameplay layout. Everything is one FFmpeg pass of crop,
scale, vstack, trim, concat and overlay. The edited timeline is returned as
source segments, so captions follow it (recut.virtual_transcript) and the
editor can open it like any recut.

Settings (all default on; CLIP_VIRAL_EDIT=0 turns every effect off and leaves
only the layout): CLIP_COLD_OPEN, CLIP_PUNCH, CLIP_DEAD_AIR, CLIP_EMOJI,
CLIP_SFX.
"""
import math
import os
import re
import subprocess

import layout_ranges

TEASER_BEFORE, TEASER_AFTER = 0.7, 1.4      # around the peak
MIN_SETUP = 6.0           # a cold open only pays off with this much lead-in
DEAD_AIR_GAP = 2.0        # a speech gap at least this long can be tightened
DEAD_AIR_KEEP = 0.35      # seconds kept on either side of a tightened gap
DEAD_AIR_MAX_SHARE = 0.20
PEAK_GUARD = (3.0, 3.5)   # never cut this close before / after the peak
CAM_ZOOM, FRAME_ZOOM = 1.22, 1.12
EMOJI_SIZE = 170
EMOJI_SHOW = 1.1
EMOJI_LIMIT, EMOJI_SPACING = 4, 3.5

_WORD = re.compile(r"[^\w']+", re.UNICODE)
# (phrase as a tuple of words, emoji). Longest phrases are tried first.
EMOJI_RULES = [
    (("no", "way"), "😱"), (("oh", "my", "god"), "😱"), (("let's", "go"), "🔥"),
    (("lets", "go"), "🔥"), (("what", "the"), "😳"),
    (("haha",), "😂"), (("hahaha",), "😂"), (("lol",), "😂"), (("lmao",), "😂"), (("lmfao",), "😂"),
    (("dead",), "💀"), (("bro",), "💀"), (("bruh",), "💀"), (("what",), "😳"), (("huh",), "😳"),
    (("wtf",), "😳"), (("omg",), "😱"), (("nooo",), "😱"), (("noooo",), "😱"), (("clutch",), "🔥"),
    (("insane",), "🔥"), (("yes",), "🔥"), (("gg",), "🏆"), (("ez",), "🏆"), (("wait",), "👀"),
]


def _flag(name, default=True):
    raw = os.environ.get(name, "").strip().lower()
    if not raw:
        return default
    return raw not in ("0", "false", "no", "off")


def options():
    """Which parts of the edit run for this job."""
    on = _flag("CLIP_VIRAL_EDIT")
    return {name: on and _flag(f"CLIP_{name.upper()}")
            for name in ("cold_open", "punch", "dead_air", "emoji", "sfx")}


# --------------------------------------------------------------------------
# Planning (pure)
# --------------------------------------------------------------------------

def find_peak(source, start, end, origin=0.0, hint=None):
    """The reaction peak inside [start, end] (source seconds): the hype
    engine's own peak when it falls inside, else the loudest streamer moment
    measured on this stretch, else None (no clear reaction: no punch)."""
    if hint is not None and start <= float(hint) <= end:
        return float(hint)
    import hype_engine

    try:
        features = hype_engine.voice_features(source, start=max(0.0, start - origin),
                                              duration=end - start)
    except (OSError, ValueError, subprocess.SubprocessError):
        return None
    if not features:
        return None
    seconds = int(math.ceil(end - start))
    excitement = hype_engine.per_second(hype_engine.voice_excitement(features), seconds)
    if not len(excitement) or float(excitement.max()) < 1.0:
        return None
    return start + int(excitement.argmax()) + 0.5


def dead_air(words, start, end, peak=None):
    """Cuts [(a, b)] that tighten long speech gaps inside [start, end].
    ``words``: [{"s", "e"}] source seconds. No words: no cuts (silence may be
    action). Never within PEAK_GUARD of the peak, never past
    DEAD_AIR_MAX_SHARE of the clip, longest gaps first."""
    inside = sorted((float(w["s"]), float(w["e"])) for w in words
                    if float(w["e"]) > start and float(w["s"]) < end)
    if len(inside) < 2:
        return []
    gaps = []
    for (_, prev_end), (next_start, _) in zip(inside, inside[1:]):
        a, b = prev_end + DEAD_AIR_KEEP, next_start - DEAD_AIR_KEEP
        if next_start - prev_end < DEAD_AIR_GAP or b - a < 0.3:
            continue
        if peak is not None and a < peak + PEAK_GUARD[1] and b > peak - PEAK_GUARD[0]:
            continue
        gaps.append((max(a, start), min(b, end)))
    budget = DEAD_AIR_MAX_SHARE * (end - start)
    cuts = []
    for a, b in sorted(gaps, key=lambda g: g[0] - g[1]):
        if b - a <= budget:
            cuts.append((round(a, 3), round(b, 3)))
            budget -= b - a
    return sorted(cuts)


def plan(start, end, *, peak=None, words=(), cold_open=True, dead_air_cuts=True):
    """The edited clip as source segments in play order, plus where its
    moments land on the output timeline."""
    cuts = dead_air(words, start, end, peak) if dead_air_cuts else []
    body, cursor = [], start
    for a, b in cuts:
        if a > cursor:
            body.append((cursor, a))
        cursor = b
    if end > cursor:
        body.append((cursor, end))

    teaser = None
    if cold_open and peak is not None and peak - start >= MIN_SETUP and end - peak >= 1.0:
        teaser = (max(start, peak - TEASER_BEFORE), min(end, peak + TEASER_AFTER))
    segments = ([teaser] if teaser else []) + body
    teaser_len = (teaser[1] - teaser[0]) if teaser else 0.0

    def to_output(t):
        """Where source second ``t`` of the body plays in the edit."""
        out = teaser_len
        for a, b in body:
            if t < a:
                return None
            if t <= b:
                return out + (t - a)
            out += b - a
        return None

    peaks = []
    if peak is not None:
        if teaser:
            peaks.append(peak - teaser[0])
        body_peak = to_output(peak)
        if body_peak is not None:
            peaks.append(body_peak)
    duration = sum(b - a for a, b in segments)
    return {"segments": [(round(a, 3), round(b, 3)) for a, b in segments],
            "teaser": teaser is not None, "teaser_length": round(teaser_len, 3),
            "peak": peak, "peak_out": [round(p, 3) for p in peaks], "cuts": cuts,
            "duration": round(duration, 3)}


def _norm(word):
    return _WORD.sub("", str(word).casefold().replace("’", "'"))


def emoji_cues(vwords, peak_out=(), laughing=False, duration=None):
    """[(output second, emoji)] for reaction words in ``vwords`` (the edited
    timeline's words, [{"word", "start"}]): at most EMOJI_LIMIT, spaced
    EMOJI_SPACING apart. A laughing streamer gets 😂 on the peak."""
    tokens = [(_norm(w.get("word")), float(w.get("start", 0))) for w in vwords]
    tokens = [(t, s) for t, s in tokens if t]
    rules = sorted(EMOJI_RULES, key=lambda r: -len(r[0]))
    found = []
    i = 0
    while i < len(tokens):
        for phrase, emoji in rules:
            n = len(phrase)
            if tuple(t for t, _ in tokens[i:i + n]) == phrase:
                found.append((tokens[i][1], emoji))
                i += n - 1
                break
        i += 1
    if laughing:
        found += [(p, "😂") for p in peak_out]
    found.sort()
    out = []
    for t, emoji in found:
        if duration is not None and t > duration - 0.5:
            continue
        if all(abs(t - u) >= EMOJI_SPACING for u, _ in out):
            out.append((round(t, 3), emoji))
        if len(out) >= EMOJI_LIMIT:
            break
    return out


def punch_boxes(duration, fps, times, band_w, band_h, *, zoom, focus_y=0.5):
    """Per-frame (w, h, x, y) crops of a band_w x band_h picture zooming to
    ``zoom`` on each of ``times`` (punch_in's curve: fast in, hold, slow out),
    about (centre, focus_y)."""
    import punch_in

    frames = int(math.ceil(duration * fps)) + 1
    zooms = punch_in.zoom_curve(frames, fps, list(times), max_zoom=zoom)
    boxes = []
    for z in zooms:
        w = int(band_w / z)
        h = int(band_h / z)
        w -= w % 2
        h -= h % 2
        x = int(round((band_w - w) / 2.0))
        y = int(round(focus_y * band_h - h * focus_y))
        x = max(0, min(x - x % 2, band_w - w))
        y = max(0, min(y - y % 2, band_h - h))
        boxes.append((w, h, x, y))
    return boxes


# --------------------------------------------------------------------------
# Assets
# --------------------------------------------------------------------------

EMOJI_FONTS = ("/usr/share/fonts/truetype/noto/NotoColorEmoji.ttf",
               "/usr/share/fonts/noto/NotoColorEmoji.ttf",
               "/usr/share/fonts/google-noto-emoji/NotoColorEmoji.ttf")


def emoji_font():
    configured = os.environ.get("EMOJI_FONT", "").strip()
    for path in ((configured,) if configured else ()) + EMOJI_FONTS:
        if path and os.path.isfile(path):
            return path
    return None


def emoji_png(emoji, path, size=EMOJI_SIZE):
    """``emoji`` drawn in colour to an RGBA PNG, or None (no colour emoji
    font: the clip ships without stickers rather than with empty boxes)."""
    font_path = emoji_font()
    if not font_path:
        return None
    try:
        from PIL import Image, ImageDraw, ImageFont

        font = ImageFont.truetype(font_path, 109)   # the bitmap size Noto Color Emoji ships
        canvas = Image.new("RGBA", (160, 160), (0, 0, 0, 0))
        ImageDraw.Draw(canvas).text((80, 80), emoji, font=font, embedded_color=True, anchor="mm")
        box = canvas.getbbox()
        if not box:
            return None
        canvas.crop(box).resize((size, size), Image.LANCZOS).save(path)
        return path
    except (OSError, ValueError):
        return None


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------

def _probe(path):
    """(width, height, fps, has_audio) of ``path``."""
    import json
    from fractions import Fraction

    out = subprocess.run(["ffprobe", "-v", "error", "-show_streams", "-of", "json", str(path)],
                         capture_output=True, timeout=60, check=True)
    streams = json.loads(out.stdout).get("streams", [])
    video = next(s for s in streams if s.get("codec_type") == "video")
    fps = 30.0
    # A range file can report "0/0" for one of the two; take the first sane one.
    for key in ("avg_frame_rate", "r_frame_rate"):
        num, _, den = str(video.get(key) or "").partition("/")
        try:
            value = float(Fraction(int(num), int(den or 1)))
        except (ValueError, ZeroDivisionError):
            continue
        if 1 <= value <= 240:
            fps = value
            break
    return (int(video["width"]), int(video["height"]), fps,
            any(s.get("codec_type") == "audio" for s in streams))


def _write_commands(path, lines):
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    return path


def build(*, source, base, span, segments_rel, layout_graph, punch_full, emoji_inputs,
          sfx_inputs, audio, out_w, out_h, rate_filter=""):
    """The ffmpeg argv input part and filter graph (pure; tested)."""
    k = len(segments_rel)
    parts = []
    vsrc = f"[0:v]{rate_filter + ',' if rate_filter else ''}"
    if k == 1:
        a, b = segments_rel[0]
        parts.append(f"{vsrc}trim=start={a:.4f}:end={b:.4f},setpts=PTS-STARTPTS[cat]")
        if audio:
            parts.append(f"[0:a]atrim=start={a:.4f}:end={b:.4f},asetpts=PTS-STARTPTS[acat]")
    else:
        parts.append(f"{vsrc}split={k}" + "".join(f"[s{i}]" for i in range(k)))
        if audio:
            parts.append("[0:a]asplit=" + str(k) + "".join(f"[r{i}]" for i in range(k)))
        for i, (a, b) in enumerate(segments_rel):
            parts.append(f"[s{i}]trim=start={a:.4f}:end={b:.4f},setpts=PTS-STARTPTS[v{i}]")
            if audio:
                parts.append(f"[r{i}]atrim=start={a:.4f}:end={b:.4f},asetpts=PTS-STARTPTS[a{i}]")
        pairs = "".join(f"[v{i}]" + (f"[a{i}]" if audio else "") for i in range(k))
        parts.append(f"{pairs}concat=n={k}:v=1:a={1 if audio else 0}[cat]" + ("[acat]" if audio else ""))
    parts.append(layout_graph.replace("[0:v]", "[cat]", 1))
    label = "[v]"
    if punch_full:
        from ffmpeg_utils import escape_filter_value
        parts.append(f"{label}sendcmd=f='{escape_filter_value(punch_full)}',"
                     f"crop@pz=w={out_w}:h={out_h}:x=0:y=0,scale={out_w}:{out_h},setsar=1[vp]")
        label = "[vp]"
    inputs = ["-ss", f"{base:.4f}", "-t", f"{span:.4f}", "-i", str(source)]
    for n, (png, x, y, t0, t1) in enumerate(emoji_inputs, 1):
        inputs += ["-i", str(png)]
        parts.append(f"{label}[{n}:v]overlay=x={x}:y={y}:enable='between(t,{t0:.3f},{t1:.3f})'[e{n}]")
        label = f"[e{n}]"
    parts.append(f"{label}format=yuv420p[vout]")
    if audio:
        from ffmpeg_utils import LOUDNORM_FILTER
        first = 1 + len(emoji_inputs)
        mix = ["[acat]"]
        for n, (wav, at, gain) in enumerate(sfx_inputs):
            inputs += ["-i", str(wav)]
            ms = max(0, int(at * 1000))
            parts.append(f"[{first + n}:a]adelay={ms}:all=1,volume={gain:.2f}[x{n}]")
            mix.append(f"[x{n}]")
        if len(mix) > 1:
            parts.append(f"{''.join(mix)}amix=inputs={len(mix)}:normalize=0:duration=first[amixed]")
            parts.append(f"[amixed]{LOUDNORM_FILTER}[aout]")
        else:
            parts.append(f"[acat]{LOUDNORM_FILTER}[aout]")
    return inputs, ";".join(parts)


def choose_layout(source, *, start, end, origin, orig_w, orig_h, out_w, out_h, layout,
                  focus=None, signals=None, ratings=None):
    """The composition for one clip (composition.py), decided once and used
    by the render, the layout sidecar and the composition check alike.

    - ``streamer``: the facecam stack, sized by the resolved focus.
    - ``gameplay``: the game, zoomed past the 32% strip, its embedded facecam
      kept whole in it; when the moment is the streamer's reaction (auto
      focus) or the user asked for the streamer, the balanced stack instead,
      so the reaction is not a thumbnail in the corner.

    A stack needs a CONFIDENT overlay (moving game around a still webcam):
    an uncertain one keeps the single full picture, so the streamer is never
    shown twice. Returns {"kind": "stack"|"gameplay", "geometry", "crop",
    "cam", "focus", "notes"}."""
    import composition

    out = {"kind": "gameplay", "geometry": None, "crop": None, "cam": None,
           "focus": None, "notes": []}
    if out_w >= out_h:
        return out
    cam = None
    try:
        import facecam
        cam = facecam.detect(source, max(0.0, start - origin), end - origin, with_motion=True)
    except Exception as e:
        out["notes"].append(f"facecam check failed ({type(e).__name__})")
    out["cam"] = cam
    requested = (focus if focus is not None else os.environ.get("FRAMING_FOCUS", "auto")) or "auto"
    focus = composition.resolve_focus(focus, signals, ratings)
    out["focus"] = focus
    # Gameplay Priority: an explicit Large or Balanced streamer is a stack of
    # that size; auto stacks (balanced) only when the moment is the reaction.
    explicit = str(requested).strip().lower() in ("streamer", "balanced")
    wants_stack = layout == "streamer" or (layout == "gameplay" and (explicit or focus == "streamer"))
    if wants_stack and cam and cam.get("confident"):
        import facecam
        prof = focus if layout == "streamer" or explicit else "balanced"
        geometry = facecam.stack_layout(orig_w, orig_h, out_w, out_h, cam, focus=prof)
        if geometry:
            out.update(kind="stack", geometry=geometry, focus=prof)
            return out
        out["notes"].append(composition.why_not(orig_w, orig_h, out_w, out_h, cam, prof))
    elif wants_stack and cam:
        out["notes"].append("webcam overlay uncertain")
    elif wants_stack:
        out["notes"].append("no webcam overlay")
    # One picture: the game with its own facecam, zoomed when that keeps the
    # whole facecam in frame.
    action = None
    frac = 1.0 / composition._env_float("GAMEPLAY_ZOOM", composition.GAMEPLAY_ZOOM, 1.0, 1.6)
    if cam and cam.get("motion"):
        src_w = float(cam["source_size"][0])
        bx, _, bw, _ = cam.get("avoid", cam["box"])
        action = composition.action_centre(cam["motion"], frac, mask=(bx / src_w, (bx + bw) / src_w))
    out["crop"] = composition.gameplay_zoom(orig_w, orig_h, out_w, out_h,
                                            cam_box=cam.get("avoid", cam["box"]) if cam else None, action=action)
    if layout == "gameplay" or not wants_stack:
        out["focus"] = "gameplay" if layout == "gameplay" else focus
    return out


def render(source, output, *, aspect, start, end, origin=0.0, transcript=None, peak_hint=None,
           layout="gameplay", laughing=False, workdir=None, settings=None, focus=None,
           signals=None, ratings=None):
    """Render [start, end] (source seconds; ``source``'s first frame is at
    ``origin``) as the edited clip. Returns what it did:
    {"segments": [{"start", "end"}], "duration", "layout", "teaser", "peak",
     "peak_out", "cuts", "emoji", "sfx", "composition"}. Raises on an FFmpeg
    failure. ``focus`` / ``signals`` / ``ratings`` choose how large the
    streamer is (composition.resolve_focus)."""
    import reframe_v2
    import screencast_layout
    from ffmpeg_utils import video_encode_args, cut_rate_filter, METADATA_SCRUB, QUALITY_FAST

    opts = settings if settings is not None else options()
    workdir = workdir or os.path.dirname(os.path.abspath(output))
    stem = os.path.splitext(os.path.basename(output))[0]
    orig_w, orig_h, fps, audio = _probe(source)
    out_w, out_h = reframe_v2.delivery_size(orig_w, orig_h, aspect)

    from recut import transcript_words
    words = [w for w in transcript_words(transcript or {}) if w["e"] > start and w["s"] < end]
    wants_peak = opts["cold_open"] or opts["punch"] or opts["sfx"]
    peak = find_peak(source, start, end, origin, peak_hint) if wants_peak else None
    p = plan(start, end, peak=peak, words=words, cold_open=opts["cold_open"],
             dead_air_cuts=opts["dead_air"])

    chosen = {"kind": "gameplay", "geometry": None, "crop": None, "cam": None, "focus": None, "notes": []}
    if aspect < 1.0:
        chosen = choose_layout(source, start=start, end=end, origin=origin, orig_w=orig_w,
                               orig_h=orig_h, out_w=out_w, out_h=out_h, layout=layout,
                               focus=focus, signals=signals, ratings=ratings)
    geometry = chosen["geometry"]
    if geometry:
        print(f"   🎥 Streamer stack ({chosen['focus']}): facecam {geometry['cam_h']}px "
              f"(face {geometry['face_share']:.0%} of the height, x{geometry['cam_scale']:.2f}"
              f"{', card' if geometry.get('card') else ''}), game keeps {geometry['game_keep']:.0%} of the width.")
    else:
        why = "; ".join(chosen["notes"])
        print("   🎮 Gameplay layout" + (f" ({why})" if why else "") +
              (f": zoomed to {chosen['crop'][1]}px of the width, facecam kept whole." if chosen["crop"]
               else ": the whole width."))

    rate = cut_rate_filter(source)
    rate_filter = rate[1] if rate else ""
    out_fps = fps
    if rate_filter:
        num, _, den = rate_filter.split("=", 1)[1].partition("/")
        out_fps = float(num) / float(den or 1)

    punch_cam = punch_full = None
    zoom_times = p["peak_out"] if opts["punch"] else []
    if zoom_times:
        import punch_in
        if geometry:
            boxes = punch_boxes(p["duration"], out_fps, zoom_times, out_w, geometry["cam_h"],
                                zoom=CAM_ZOOM, focus_y=0.55)
            punch_cam = _write_commands(os.path.join(workdir, f".{stem}.camzoom.cmd"),
                                        punch_in.sendcmd_lines(boxes, out_fps, target="crop@pz"))
        else:
            boxes = punch_boxes(p["duration"], out_fps, zoom_times, out_w, out_h, zoom=FRAME_ZOOM)
            punch_full = _write_commands(os.path.join(workdir, f".{stem}.zoom.cmd"),
                                         punch_in.sendcmd_lines(boxes, out_fps, target="crop@pz"))

    extra = {}
    if geometry:
        import facecam
        graph = facecam.stack_filtergraph(geometry, out_w, out_h, cam_zoom_cmds=punch_cam,
                                          scale_flags="lanczos")
        band = geometry["content"]
        strategy = "STACK"
        extra = {"seam": geometry["seam"], "caption_y": geometry["caption_y"]}
    else:
        graph, fg_h = reframe_v2.gameplay_filtergraph(orig_w, orig_h, out_w, out_h, chosen["crop"])
        band = screencast_layout.content_band(out_h, fg_h)
        strategy = "GAMEPLAY"

    emoji_inputs, emoji_used = [], []
    if opts["emoji"] and transcript:
        import recut
        vwords = []
        for seg in recut.virtual_transcript(transcript, [{"start": a, "end": b} for a, b in p["segments"]])["segments"]:
            vwords.extend(seg.get("words") or [])
        cues = emoji_cues(vwords, p["peak_out"], laughing, p["duration"])
        # Emoji pop on the facecam band, beside the face, or above the game.
        y = int(out_h * (min(0.30, geometry["cam_h"] / out_h * 0.7) if geometry
                         and geometry.get("cam_position") == "top" else 0.62))
        for n, (t, emoji) in enumerate(cues):
            png = emoji_png(emoji, os.path.join(workdir, f".{stem}.emoji{n}.png"))
            if png:
                x = (out_w - EMOJI_SIZE) // 2 + (140 if n % 2 else -140)
                emoji_inputs.append((png, x, y, t, min(p["duration"], t + EMOJI_SHOW)))
                emoji_used.append({"time": t, "emoji": emoji})

    sfx_inputs, sfx_used = [], []
    if opts["sfx"] and audio:
        import sfx
        for at, name, gain in sfx.placements(p):
            wav = sfx.path(name)
            if wav:
                sfx_inputs.append((wav, at, gain))
                sfx_used.append({"time": round(at, 3), "sound": name})

    base = max(0.0, min(a for a, _ in p["segments"]) - origin - 1.0)
    span = max(b for _, b in p["segments"]) - origin - base + 0.5
    rel = [(a - origin - base, b - origin - base) for a, b in p["segments"]]
    inputs, graph = build(source=source, base=base, span=span, segments_rel=rel, layout_graph=graph,
                          punch_full=punch_full, emoji_inputs=emoji_inputs, sfx_inputs=sfx_inputs,
                          audio=audio, out_w=out_w, out_h=out_h, rate_filter=rate_filter)
    audio_args = ["-map", "[aout]", "-c:a", "aac", "-b:a", "192k"] if audio else ["-an"]
    cmd = ["ffmpeg", "-nostdin", "-y", "-loglevel", "error", "-filter_complex_threads", "1",
           *inputs, "-filter_complex", graph,
           # The captions pass encodes the delivered file again: this is the
           # same tier render_gameplay_range used for its intermediate.
           "-map", "[vout]", *video_encode_args(QUALITY_FAST), *audio_args, *METADATA_SCRUB,
           "-movflags", "+faststart", str(output)]
    try:
        result = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, timeout=1800)
    finally:
        for path in (punch_cam, punch_full, *(e[0] for e in emoji_inputs)):
            if path and os.path.exists(path):
                os.remove(path)
    if result.returncode != 0:
        detail = result.stderr.decode(errors="replace").strip()[-400:]
        raise RuntimeError(f"viral edit encode failed: {detail}")
    layout_ranges.write(output, [(0, p["duration"], strategy, band, extra)])
    import composition_check
    return {"segments": [{"start": a, "end": b} for a, b in p["segments"]],
            "duration": p["duration"], "layout": "stack" if geometry else "gameplay",
            "teaser": p["teaser"], "peak": p["peak"], "peak_out": p["peak_out"],
            "cuts": [{"start": a, "end": b} for a, b in p["cuts"]],
            "emoji": emoji_used, "sfx": sfx_used,
            "composition": composition_check.describe(chosen, orig_w, orig_h, out_w, out_h)}


def layout_still(source, output, *, aspect, start, end, origin=0.0, layout="streamer", focus=None,
                 signals=None, ratings=None, at=None, width=540):
    """One frame of [start, end] composed exactly as the render will compose
    it (the same choose_layout decision, the same filtergraph), as a JPEG at
    ``width``: the candidate review's preview of the vertical clip, so what
    is approved is what is delivered. Returns composition_check.describe's
    dict. Raises on an FFmpeg failure."""
    import reframe_v2
    import composition_check

    orig_w, orig_h, _fps, _audio = _probe(source)
    out_w, out_h = reframe_v2.delivery_size(orig_w, orig_h, aspect)
    chosen = choose_layout(source, start=start, end=end, origin=origin, orig_w=orig_w, orig_h=orig_h,
                           out_w=out_w, out_h=out_h, layout=layout, focus=focus, signals=signals,
                           ratings=ratings)
    if chosen["geometry"]:
        import facecam
        graph = facecam.stack_filtergraph(chosen["geometry"], out_w, out_h)
    else:
        graph, _ = reframe_v2.gameplay_filtergraph(orig_w, orig_h, out_w, out_h, chosen["crop"])
    height = int(round(width * out_h / float(out_w) / 2)) * 2
    t = (start + end) / 2.0 if at is None else at
    cmd = ["ffmpeg", "-nostdin", "-y", "-loglevel", "error", "-ss", f"{max(0.0, t - origin):.3f}",
           "-i", str(source), "-filter_complex", f"{graph};[v]scale={width}:{height}[still]",
           "-map", "[still]", "-frames:v", "1", "-q:v", "3", str(output)]
    result = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, timeout=180)
    if result.returncode != 0:
        raise RuntimeError("layout still failed: " + result.stderr.decode(errors="replace")[-300:])
    return composition_check.describe(chosen, orig_w, orig_h, out_w, out_h)
