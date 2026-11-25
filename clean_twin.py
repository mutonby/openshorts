"""The free plan's clean captioned clip, encoded when it is first needed.

A free clip is served as ``wm_<final>`` and the clean ``<final>`` was kept
next to it so an upgrade can drop the mark (``watermarked`` module). That
clean file was one NVENC encode per free clip that nobody watches unless the
user pays or edits the clip, on a card whose encoder is the busiest part of
the job (balrog, 25-nov-2026: encoder at 95-100 %, 87 % of the burns free).

With ``CLEAN_LAZY=1`` (this host's free jobs; never a remote worker, which
does not send recipes) the burn writes only the marked copy and keeps a
recipe for the clean one: the caption file and the picture it went on, which
is a file in the job directory or a hooked picture kept as a hook recipe
(hooks.materialize). ``materialize`` encodes the clean file from it with the
same graph the pipeline's pass applies (hook overlay, then the captions) and
the same encoder settings, from the same source.

Who asks for it: an upgrade (cloud/videos.unmark_user_library, through the
callback app.py registers), and the editor walk-backs when they need the
clean picture itself (app._strip_watermark_copy). The recipe files ride to R2
with the clip (``archive_names``), so a restored project can still make it.

Path helpers only use the standard library: cloud/videos imports this module.
"""
import json
import os
import shutil
import uuid

PREFIX = ".cleanrecipe_"


def lazy():
    return os.environ.get("CLEAN_LAZY", "0").strip() == "1"


def recipe_paths(output_dir, clean_name):
    """(json, ass) holding how to encode ``clean_name``, next to the clip."""
    base = os.path.join(output_dir, f"{PREFIX}{os.path.basename(clean_name)}")
    return base + ".json", base + ".ass"


def has_recipe(output_dir, clean_name):
    return os.path.exists(recipe_paths(output_dir, clean_name)[0])


def available(output_dir, clean_name):
    """The clean file is on disk or can be encoded from its recipe."""
    return (os.path.exists(os.path.join(output_dir, os.path.basename(clean_name)))
            or has_recipe(output_dir, clean_name))


def write_recipe(clean_path, source_path, ass_path):
    """Keep what it takes to encode ``clean_path`` later: its caption file and
    the picture they go on (``source_path``: on disk, or a hooked picture with
    a hook recipe). Both must sit in the clip's directory."""
    output_dir = os.path.dirname(os.path.abspath(clean_path))
    if os.path.dirname(os.path.abspath(source_path)) != output_dir:
        raise ValueError("the captions' source must be next to the clip")
    json_path, ass_copy = recipe_paths(output_dir, clean_path)
    shutil.copyfile(ass_path, ass_copy)
    # The JSON last and atomically: a recipe whose JSON exists is whole.
    tmp = f"{json_path}.{uuid.uuid4().hex[:8]}.part"
    with open(tmp, "w") as f:
        json.dump({"source": os.path.basename(source_path)}, f)
    os.replace(tmp, json_path)


def _source(output_dir, clean_name):
    try:
        with open(recipe_paths(output_dir, clean_name)[0]) as f:
            source = os.path.basename(str(json.load(f)["source"]))
    except (OSError, ValueError, KeyError, TypeError):
        return None
    return source or None


def archive_names(output_dir, clean_name):
    """The recipe files of ``clean_name`` to keep with the clip in R2 (the
    hook recipe too when the captions go on a hooked picture that is one),
    or [] when it has no recipe."""
    if not has_recipe(output_dir, clean_name):
        return []
    names = [os.path.basename(p) for p in recipe_paths(output_dir, clean_name)]
    source = _source(output_dir, clean_name)
    if source:
        import hooks
        if hooks.has_recipe(output_dir, source):
            names += [os.path.basename(p) for p in hooks.recipe_paths(output_dir, source)]
    return [n for n in names if os.path.exists(os.path.join(output_dir, n))]


def _hook_of(output_dir, hooked_name):
    """(source path, png, x, y, duration) of a hook recipe, read as numbers
    (the same checks hooks.materialize makes), or None."""
    import hooks
    json_path, png_path = hooks.recipe_paths(output_dir, hooked_name)
    try:
        with open(json_path) as f:
            r = json.load(f)
        source = os.path.join(output_dir, os.path.basename(r["source"]))
        x, y = int(r["x"]), int(r["y"])
        duration = float(r["duration"]) if r.get("duration") else None
    except (OSError, ValueError, KeyError, TypeError):
        return None
    if not (-20000 <= x <= 20000 and -20000 <= y <= 20000) or (
            duration is not None and not 0 < duration < 86400):
        return None
    if not (os.path.exists(source) and os.path.exists(png_path)):
        return None
    return source, png_path, x, y, duration


def materialize(output_dir, clean_name):
    """Encode ``clean_name`` from its recipe if it is not on disk yet.

    True when the file exists afterwards. Safe to call concurrently: each
    call encodes to its own temporary name and renames it into place."""
    clean_name = os.path.basename(clean_name)
    path = os.path.join(output_dir, clean_name)
    if os.path.exists(path):
        return True
    source = _source(output_dir, clean_name)
    ass_path = recipe_paths(output_dir, clean_name)[1]
    if not source or not os.path.exists(ass_path):
        return False
    import gpu_burn
    import hooks
    import subtitles
    source_path = os.path.join(output_dir, source)
    # Captions on a hooked picture kept as a recipe: the pipeline's pass drew
    # the hook and the captions from one decode of the hook's source, so that
    # is what is repeated here (even once the hooked file has been encoded:
    # captioning that file would be one generation further from the source).
    hook = _hook_of(output_dir, source) if hooks.has_recipe(output_dir, source) else None
    if hook is not None:
        source_path = hook[0]
    elif not os.path.exists(source_path):
        return False
    vf = subtitles.subtitles_filter(ass_path)
    tmp = os.path.join(output_dir, f".part_{uuid.uuid4().hex[:8]}_{clean_name}")
    try:
        if gpu_burn.burn(source_path, [("ass", tmp)], hook=hook[1:] if hook else None,
                         ass_vf=vf):
            os.replace(tmp, path)
            return True
        if hook:
            _, png, x, y, duration = hook
            cmd = ['ffmpeg', '-y', '-i', source_path, '-i', png, '-filter_complex',
                   f"{hooks._overlay_graph(x, y, duration)},{vf}", *hooks._encode_tail(), tmp]
        else:
            cmd = ['ffmpeg', '-y', '-i', source_path, '-vf', vf, *hooks._encode_tail(), tmp]
        result = hooks._run_encode(cmd, source_path, [tmp])
        from ffmpeg_utils import finished_before_crash
        if result.returncode != 0 and not finished_before_crash(result, [tmp]):
            print(f"⚠️ Could not encode {clean_name} from its recipe")
            return False
        os.replace(tmp, path)
        print(f"   🧼 Clean twin encoded: {clean_name}")
        return True
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
