"""Render a reframed clip in one GPU pass (``osrender``).

The ffmpeg path renders a clip scene by scene: one ffmpeg per scene that
decodes on the card, downloads every frame, crops/scales/blurs it on the CPU
and uploads it again for NVENC, then a concat. ``osrender`` (``osrender/``,
Rust) does the whole clip in one process with the frames kept on the card:
NVDEC -> CUDA kernels -> NVENC.

The layout geometry is not re-implemented here: ``layers_from_graph`` reads
the very filtergraph each scene would hand ffmpeg (crop / scale / gblur /
overlay / vstack / pad / split) and turns it into layers, each a source
rectangle drawn into an output rectangle, optionally blurred. TRACK scenes
pass their per-frame crop boxes instead of a sendcmd file.

It is used only when everything it needs is there: the binary, NVENC, a
source NVDEC decodes and a filtergraph it can read. Anything else, or any
failure, renders through ffmpeg exactly as before. ``GPU_RENDER=0`` turns it
off. The encode takes an NVENC slot like every job ffmpeg (``bin/ffmpeg``).
"""
import ast
import contextlib
import json
import os
import re
import subprocess
import tempfile
import time

import gpu_decode
from ffmpeg_utils import QUALITY_FAST, _NVENC_ARGS, nvenc_available

ENABLED = os.environ.get("GPU_RENDER", "1").strip() != "0"
BIN = os.environ.get("OSRENDER_BIN", "/usr/local/bin/osrender")
# A stuck render falls back to ffmpeg after this, not after ffmpeg's 30 min.
MIN_TIMEOUT = 300
TIMEOUT_PER_SECOND = 10


def usable(input_video):
    """True when ``input_video`` can be rendered by osrender right now."""
    if not ENABLED or not (os.path.isfile(BIN) and os.access(BIN, os.X_OK)):
        return False
    if os.environ.get("FFMPEG_ENCODER", "x264").strip().lower() not in ("nvenc", "auto"):
        return False
    return nvenc_available() and gpu_decode.usable(input_video, render=True)


# --- filtergraph -> layers (pure, CI-testable) -------------------------------

class Unsupported(Exception):
    """The graph uses something osrender does not draw."""


_FUNCS = {"min": min, "max": max, "trunc": int, "floor": int}
_BINOPS = {ast.Add: lambda a, b: a + b, ast.Sub: lambda a, b: a - b,
           ast.Mult: lambda a, b: a * b, ast.Div: lambda a, b: a / b}


def _eval(expr, names):
    """Evaluate an ffmpeg arithmetic option (numbers, names, + - * /, min/max)."""
    def ev(node):
        if isinstance(node, ast.Expression):
            return ev(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.Name) and node.id in names:
            return names[node.id]
        if isinstance(node, ast.BinOp) and type(node.op) in _BINOPS:
            return _BINOPS[type(node.op)](ev(node.left), ev(node.right))
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            return -ev(node.operand)
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id in _FUNCS and not node.keywords):
            return _FUNCS[node.func.id](*[ev(a) for a in node.args])
        raise Unsupported(f"expression {expr!r}")
    try:
        tree = ast.parse(str(expr).strip(), mode="eval")
    except SyntaxError as e:
        raise Unsupported(f"expression {expr!r}") from e
    return ev(tree)


def _split_unescaped(s, sep):
    out, cur, i = [], [], 0
    while i < len(s):
        c = s[i]
        if c == "\\" and i + 1 < len(s):
            cur.append(s[i:i + 2])
            i += 2
            continue
        if c == sep:
            out.append("".join(cur))
            cur = []
        else:
            cur.append(c)
        i += 1
    out.append("".join(cur))
    return out


def _options(args, positional):
    """``a=1:b=2`` / ``1:2`` -> dict, positional names in ffmpeg's order."""
    opts = {}
    if not args:
        return opts
    for k, part in enumerate(_split_unescaped(args, ":")):
        part = part.replace("\\,", ",").replace("\\", "")
        if "=" in part and re.match(r"^[a-z_]+=", part):
            key, val = part.split("=", 1)
            opts[key] = val
        elif k < len(positional):
            opts[positional[k]] = part
        else:
            raise Unsupported(f"option {part!r}")
    return opts


class _View:
    """A picture made from the source: ``rect`` (x, y, w, h in source pixels)
    shown at ``w x h``; ``blur`` = (sigma, w, h) when it was blurred at that size."""

    def __init__(self, rect, w, h, blur=None):
        self.rect, self.w, self.h, self.blur = rect, w, h, blur


class _Canvas:
    """Views placed on a ``w x h`` frame: [(rect, (x, y, w, h), blur)], black elsewhere."""

    def __init__(self, w, h, layers):
        self.w, self.h, self.layers = w, h, layers


def _as_canvas(s):
    if isinstance(s, _Canvas):
        return s
    return _Canvas(s.w, s.h, [(s.rect, (0, 0, s.w, s.h), s.blur)])


def _shifted(c, dx, dy):
    return [(r, (x + dx, y + dy, w, h), b) for r, (x, y, w, h), b in c.layers]


def _even(v):
    return int(v) & ~1 if v >= 0 else -((-int(v) + 1) & ~1)


def _crop(s, opts):
    if not isinstance(s, _View) or s.blur:
        raise Unsupported("crop after a composite or a blur")
    names = {"iw": s.w, "ih": s.h, "in_w": s.w, "in_h": s.h}
    w = _even(_eval(opts.get("w", opts.get("out_w", "iw")), names))
    h = _even(_eval(opts.get("h", opts.get("out_h", "ih")), names))
    names.update(ow=w, oh=h, out_w=w, out_h=h)
    x = _eval(opts.get("x", "(iw-ow)/2"), names)
    y = _eval(opts.get("y", "(ih-oh)/2"), names)
    # ffmpeg keeps the rectangle inside the picture and on chroma-aligned
    # (even) coordinates for 4:2:0.
    x = _even(min(max(x, 0), s.w - w))
    y = _even(min(max(y, 0), s.h - h))
    sx, sy, sw, sh = s.rect
    kx, ky = sw / s.w, sh / s.h
    return _View((sx + x * kx, sy + y * ky, w * kx, h * ky), w, h)


def _scale(s, opts):
    if not isinstance(s, _View):
        raise Unsupported("scale of a composite")
    for k in opts:
        if k not in ("w", "h", "width", "height"):
            raise Unsupported(f"scale option {k}")
    names = {"iw": s.w, "ih": s.h, "in_w": s.w, "in_h": s.h}
    w = int(_eval(opts.get("w", opts.get("width", "iw")), names))
    h = int(_eval(opts.get("h", opts.get("height", "ih")), names))
    # -1 keeps the aspect ratio, -n also makes the side a multiple of n
    # (ffmpeg: av_rescale(h, iw, ih * n) * n, rounded to nearest).
    if w < 0 and h < 0:
        w, h = s.w, s.h
    elif w < 0:
        w = round(h * s.w / (s.h * -w)) * -w
    elif h < 0:
        h = round(w * s.h / (s.w * -h)) * -h
    return _View(s.rect, int(w), int(h), s.blur)


def _gblur(s, opts):
    if not isinstance(s, _View) or s.blur:
        raise Unsupported("gblur of a composite or twice")
    sigma = float(_eval(opts.get("sigma", "0.5"), {}))
    return _View(s.rect, s.w, s.h, (sigma, s.w, s.h))


def _overlay(base, top, opts):
    for k in opts:
        if k not in ("x", "y"):
            raise Unsupported(f"overlay option {k}")
    b = _as_canvas(base)
    names = {"W": b.w, "H": b.h, "main_w": b.w, "main_h": b.h,
             "w": top.w, "h": top.h, "overlay_w": top.w, "overlay_h": top.h}
    # overlay places 4:2:0 pictures on even coordinates.
    x = _even(_eval(opts.get("x", "0"), names))
    y = _even(_eval(opts.get("y", "0"), names))
    return _Canvas(b.w, b.h, b.layers + _shifted(_as_canvas(top), x, y))


def _vstack(inputs):
    width = inputs[0].w
    layers, y = [], 0
    for s in inputs:
        if s.w != width:
            raise Unsupported("vstack of different widths")
        layers += _shifted(_as_canvas(s), 0, y)
        y += s.h
    return _Canvas(width, y, layers)


def _pad(s, opts):
    color = opts.get("color", opts.get("c", "black")).lower()
    if color not in ("black", "0x000000", "#000000"):
        raise Unsupported(f"pad colour {color}")
    names = {"iw": s.w, "ih": s.h, "in_w": s.w, "in_h": s.h}
    w = int(_eval(opts.get("w", opts.get("width", "iw")), names))
    h = int(_eval(opts.get("h", opts.get("height", "ih")), names))
    names.update(ow=w, oh=h, out_w=w, out_h=h)
    x = _even(_eval(opts.get("x", "0"), names))
    y = _even(_eval(opts.get("y", "0"), names))
    return _Canvas(w, h, _shifted(_as_canvas(s), x, y))


_LABELS = re.compile(r"^((?:\[[^\]]+\])*)(.*?)((?:\[[^\]]+\])*)$", re.S)


def layers_from_graph(graph, src_w, src_h, out_w, out_h, src_label="[0:v]"):
    """The layers a ``-filter_complex`` graph draws, or raise ``Unsupported``.

    Returns [{"src": [x, y, w, h], "dst": [x, y, w, h], "blur": {...}?}] in
    drawing order, for a graph whose single output is ``out_w x out_h``."""
    streams = {src_label.strip("[]"): _View((0.0, 0.0, float(src_w), float(src_h)), src_w, src_h)}
    last = None
    for chain in _split_unescaped(graph, ";"):
        chain = chain.strip()
        if not chain:
            continue
        m = _LABELS.match(chain)
        ins = re.findall(r"\[([^\]]+)\]", m.group(1))
        outs = re.findall(r"\[([^\]]+)\]", m.group(3))
        filters = [f for f in _split_unescaped(m.group(2), ",") if f.strip()]
        try:
            cur = [streams.pop(lbl) for lbl in ins]
        except KeyError as e:
            raise Unsupported(f"unknown input {e}") from e
        for k, f in enumerate(filters):
            name, _, args = f.strip().partition("=")
            name = name.split("@")[0]
            if name in ("setsar", "null", "copy"):
                continue
            if name == "split":
                n = int(args or 2)
                if k != len(filters) - 1 or len(cur) != 1:
                    raise Unsupported("split in the middle of a chain")
                cur = [cur[0]] * n
            elif name == "crop":
                cur = [_crop(cur[0], _options(args, ["w", "h", "x", "y"]))]
            elif name == "scale":
                cur = [_scale(cur[0], _options(args, ["w", "h"]))]
            elif name == "gblur":
                cur = [_gblur(cur[0], _options(args, ["sigma"]))]
            elif name == "overlay":
                if len(cur) != 2:
                    raise Unsupported("overlay needs two inputs")
                cur = [_overlay(cur[0], cur[1], _options(args, ["x", "y"]))]
            elif name == "vstack":
                opts = _options(args, ["inputs"])
                if int(opts.get("inputs", 2)) != len(cur):
                    raise Unsupported("vstack inputs")
                cur = [_vstack(cur)]
            elif name == "pad":
                cur = [_pad(cur[0], _options(args, ["w", "h", "x", "y", "color"]))]
            else:
                raise Unsupported(f"filter {name}")
            if len(cur) > 1 and name != "split":
                raise Unsupported("chain with several streams")
        if len(outs) != len(cur):
            raise Unsupported("output labels do not match the chain")
        for lbl, s in zip(outs, cur):
            streams[lbl] = s
            last = s
    if len(streams) != 1 or last is None:
        raise Unsupported("graph must end in exactly one stream")
    canvas = _as_canvas(next(iter(streams.values())))
    if (canvas.w, canvas.h) != (out_w, out_h):
        raise Unsupported(f"graph output {canvas.w}x{canvas.h}, expected {out_w}x{out_h}")
    layers = []
    for rect, dst, blur in canvas.layers:
        layer = {"src": [round(v, 4) for v in rect], "dst": list(dst)}
        if blur:
            layer["blur"] = {"sigma": blur[0], "w": blur[1], "h": blur[2]}
        layers.append(layer)
    return layers


def sendcmd_order(n, fps):
    """Which of ``n`` per-frame values sendcmd has applied when frame k is
    filtered. The commands are written at ``f"{i / fps:.4f}"`` and sendcmd
    applies one once the frame time (microseconds) reaches it, so a time
    rounded up lands on the next frame. Mirrored here so the GPU render moves
    the crop on the very frames the ffmpeg render does."""
    order, j = [], 0
    for k in range(n):
        t_us = round(k / fps * 1e6)
        while j + 1 < n and round(float(f"{(j + 1) / fps:.4f}") * 1e6) <= t_us:
            j += 1
        order.append(j)
    return order


def track_layer(boxes, src_w, src_h, out_w, out_h, fps):
    """TRACK/punch-in: one crop box (w, h, x, y) per frame, filling the frame.
    Clamped and aligned the way ffmpeg's crop applies each sendcmd value."""
    out = []
    for k in sendcmd_order(len(boxes), fps):
        w, h, x, y = boxes[k]
        w, h = _even(min(w, src_w)), _even(min(h, src_h))
        x = _even(min(max(x, 0), src_w - w))
        y = _even(min(max(y, 0), src_h - h))
        out.append([x, y, w, h])
    return {"boxes": out, "dst": [0, 0, out_w, out_h]}


# --- running osrender --------------------------------------------------------

def encoder_options(tier=QUALITY_FAST):
    """The NVENC ffmpeg args of ``tier`` as encoder options for osrender."""
    args = _NVENC_ARGS[tier]
    opts = []
    for k in range(0, len(args) - 1, 2):
        key, val = args[k].lstrip("-"), args[k + 1]
        if key in ("c:v", "pix_fmt"):
            continue
        opts.append([{"b:v": "b"}.get(key, key), val])
    return opts


def _note_nvenc_wait(seconds):
    """Append an encode's wait for a slot to NVENC_WAIT_LOG (as bin/ffmpeg
    does), for the job's JOB_CPU line. Best effort."""
    path = os.environ.get("NVENC_WAIT_LOG", "")
    if seconds > 0 and path:
        try:
            with open(path, "a") as f:
                f.write(f"{int(seconds)}\n")
        except OSError:
            pass


@contextlib.contextmanager
def nvenc_slot():
    """Hold one of the host-wide NVENC slots (same lock files and wait as
    bin/ffmpeg) for the life of the block."""
    try:
        slots = int(os.environ.get("NVENC_SLOTS", "0") or 0)
    except ValueError:
        slots = 0
    lock_dir = os.environ.get("NVENC_LOCK_DIR", "")
    if slots <= 0 or not lock_dir or not os.path.isdir(lock_dir) or not os.access(lock_dir, os.W_OK):
        yield
        return
    import fcntl
    try:
        max_wait = int(os.environ.get("NVENC_SLOT_WAIT", "600"))
    except ValueError:
        max_wait = 600
    held, waited = None, 0
    while held is None and waited <= max_wait:
        for i in range(slots):
            try:
                f = open(os.path.join(lock_dir, f".nvenc-{i}.lock"), "a")
            except OSError:
                continue
            try:
                fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
                held = f
                break
            except OSError:
                f.close()
        if held is None:
            time.sleep(1)
            waited += 1
    _note_nvenc_wait(waited)
    try:
        yield
    finally:
        if held is not None:
            held.close()


def render(input_video, scenes, src_w, src_h, out_w, out_h, fps, output, overlay=None):
    """Render ``scenes`` of ``input_video`` into ``output`` (video only).

    ``scenes``: [(ss, dur, graph, boxes)] — ``boxes`` (per-frame crop boxes)
    for TRACK scenes, else the scene's filtergraph. Returns False, without
    raising, whenever the ffmpeg path should render instead."""
    plan_scenes = []
    expected = 0
    try:
        for ss, dur, graph, boxes in scenes:
            if boxes is not None:
                layers = [track_layer(boxes, src_w, src_h, out_w, out_h, fps)]
            else:
                layers = layers_from_graph(graph, src_w, src_h, out_w, out_h)
            plan_scenes.append({"ss": ss, "dur": dur, "layers": layers})
            expected += round(dur * fps)
    except Unsupported as e:
        print(f"   ↪️ GPU render skipped ({e}); rendering with ffmpeg")
        return False
    threads = os.environ.get("FFMPEG_THREADS", "2")
    plan = {"out_w": out_w, "out_h": out_h, "fps": fps, "encoder": encoder_options(),
            "scenes": plan_scenes, "overlay": overlay,
            "threads": int(threads) if threads.isdigit() else 2}
    fd, plan_path = tempfile.mkstemp(suffix=".json", prefix="osrender_")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(plan, f)
        timeout = max(MIN_TIMEOUT, TIMEOUT_PER_SECOND * sum(s[1] for s in scenes))
        with nvenc_slot():
            started = time.time()
            result = subprocess.run([BIN, input_video, plan_path, output],
                                    capture_output=True, text=True, timeout=timeout)
        if result.returncode != 0:
            tail = (result.stderr or "").strip().splitlines()[-1:] or ["no output"]
            print(f"   ⚠️ GPU render failed ({tail[0][:200]}); rendering with ffmpeg")
            return False
        m = re.search(r"osrender: (\d+) frames", result.stderr or "")
        frames = int(m.group(1)) if m else 0
        if abs(frames - expected) > max(2, len(scenes)):
            print(f"   ⚠️ GPU render gave {frames} frames, expected {expected}; "
                  "rendering with ffmpeg")
            return False
        print(f"   ⚡ GPU render: {frames} frames in {time.time() - started:.1f}s")
        return True
    except (OSError, subprocess.TimeoutExpired) as e:
        print(f"   ⚠️ GPU render failed ({e}); rendering with ffmpeg")
        return False
    finally:
        with contextlib.suppress(OSError):
            os.unlink(plan_path)
