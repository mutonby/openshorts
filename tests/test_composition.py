"""The layout engine for stream and game clips (composition.py) and what uses
it: facecam sizing bounds, the streamer shown once, gameplay framing, caption
and hook placement on the stack, and the composition check.

The overlay shapes below are the ones measured on real stream footage
(22-nov-2026): a small bordered 16:9 webcam, a tall webcam with a follower
counter under it, a 480p source, a 4:3 webcam, and two people on one camera.
"""
import os

import pytest

import composition
import composition_check
import facecam
import gpu_render
import hooks
import hook_placement
import layout_ranges
import subtitles

OUT_W, OUT_H = 1080, 1920

OVERLAYS = {
    # name: (source w, h, cam)
    "small_16x9_left": (1920, 1080, {"box": (0, 438, 390, 324), "face": (170, 539, 67, 87)}),
    "tall_with_counter": (1920, 1080, {"box": (0, 96, 438, 576), "face": (213, 235, 133, 169)}),
    "bordered_corner": (1920, 1080, {"box": (24, 408, 324, 384), "face": (121, 507, 86, 112)}),
    "low_res_480p": (854, 480, {"box": (0, 45, 176, 195), "face": (55, 106, 57, 63)}),
    "four_by_three_right": (1920, 1080, {"box": (1440, 0, 480, 360), "face": (1640, 90, 90, 110)}),
    "duo_on_one_camera": (1920, 1080, {"box": (0, 0, 492, 306), "face": (236, 124, 65, 70),
                                       "people": (150, 110, 160, 90)}),
    "large_full_hd_webcam": (1920, 1080, {"box": (1280, 0, 640, 360), "face": (1560, 90, 120, 150)}),
}


def _overlap(a, b):
    w = min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0])
    h = min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1])
    return max(0, w) * max(0, h)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for name in ("FRAMING_FOCUS", "FACECAM_MIN_SHARE", "FACECAM_MAX_SHARE", "FACECAM_TARGET_SHARE",
                 "FACE_MIN_SHARE", "FACE_MAX_SHARE", "GAME_KEEP_MIN", "FACECAM_MAX_UPSCALE",
                 "GAMEPLAY_ZOOM"):
        monkeypatch.delenv(name, raising=False)


# --- facecam size: never tiny, never huge, never blown up -------------------------------

@pytest.mark.parametrize("name", sorted(OVERLAYS))
@pytest.mark.parametrize("focus", ["streamer", "balanced", "gameplay"])
def test_facecam_size_stays_inside_the_profile_bounds(name, focus):
    w, h, cam = OVERLAYS[name]
    g = composition.stack(w, h, OUT_W, OUT_H, cam, focus=focus)
    if g is None:
        return  # covered by the fallback tests: the single gameplay picture
    prof = composition.profile(focus)
    assert 0.75 * prof["face_min"] <= g["face_share"] <= 1.2 * prof["face_max"], g
    assert g["cam_scale"] <= prof["max_upscale"] * 1.12 + 1e-6
    assert prof["cam_min"] * OUT_H - 8 <= g["cam_h"] <= prof["cam_max"] * OUT_H + 8
    assert g["cam_h"] + g["game_h"] == OUT_H


@pytest.mark.parametrize("name", sorted(OVERLAYS))
def test_the_streamer_appears_once_the_game_crop_never_touches_the_webcam(name):
    w, h, cam = OVERLAYS[name]
    for focus in ("streamer", "balanced", "gameplay"):
        g = composition.stack(w, h, OUT_W, OUT_H, cam, focus=focus)
        if g is None:
            continue
        gw, gh, gx, gy = g["game_crop"]
        assert _overlap((gx, gy, gw, gh), cam["box"]) <= 0.02 * gw * gh
        assert 0 <= gx and gx + gw <= w and gh <= h


@pytest.mark.parametrize("name", sorted(OVERLAYS))
def test_no_stretching_anywhere(name):
    w, h, cam = OVERLAYS[name]
    g = composition.stack(w, h, OUT_W, OUT_H, cam, focus="balanced")
    if g is None:
        return
    cw, ch, cx, cy = g["cam_crop"]
    shown = (g["card"]["w"], g["card"]["h"]) if g["card"] else (OUT_W, g["cam_h"])
    assert abs((cw / ch) / (shown[0] / shown[1]) - 1) < 0.02
    gw, gh = g["game_crop"][:2]
    assert abs((gw / gh) / (OUT_W / g["game_pic_h"]) - 1) < 0.02
    # The crop stays inside the overlay: never game pixels in the facecam band.
    bx, by, bw, bh = cam["box"]
    assert bx - 2 <= cx and cx + cw <= bx + bw + 2 and by - 2 <= cy and cy + ch <= by + bh + 2
    assert "distorted" not in composition_check.describe(
        {"kind": "stack", "geometry": g, "cam": cam, "focus": "balanced"}, w, h, OUT_W, OUT_H)["flags"]


def test_focus_profiles_order_the_streamer_and_the_game():
    w, h, cam = OVERLAYS["tall_with_counter"]
    big = composition.stack(w, h, OUT_W, OUT_H, cam, focus="streamer")
    mid = composition.stack(w, h, OUT_W, OUT_H, cam, focus="balanced")
    small = composition.stack(w, h, OUT_W, OUT_H, cam, focus="gameplay")
    assert big["face_share"] > mid["face_share"] > small["face_share"]
    assert big["cam_h"] > mid["cam_h"] > small["cam_h"]
    assert small["game_keep"] >= mid["game_keep"]


def test_the_old_fixed_band_bug_a_narrow_webcam_is_not_blown_up_to_fill_the_width():
    # Before: a fixed 40% band, cropped inside the overlay and enlarged ~2.8x
    # (KreekCraft's face filled the top of the clip). Now the face stays in
    # bounds and the webcam is shown as a card at an honest scale.
    w, h, cam = OVERLAYS["tall_with_counter"]
    g = composition.stack(w, h, OUT_W, OUT_H, cam, focus="balanced")
    assert g["face_share"] <= 0.18 * 1.2
    assert g["cam_scale"] < 2.0
    assert g["card"] is not None


def test_a_tiny_webcam_is_not_enlarged_into_mush():
    tiny = {"box": (0, 0, 120, 80), "face": (40, 10, 22, 28)}
    assert composition.stack(1920, 1080, OUT_W, OUT_H, tiny, focus="balanced") is None
    assert "too small" in composition.why_not(1920, 1080, OUT_W, OUT_H, tiny)


def test_a_webcam_in_the_middle_of_the_game_cannot_be_stacked():
    centre = {"box": (700, 300, 520, 400), "face": (900, 380, 80, 100)}
    assert composition.stack(1920, 1080, OUT_W, OUT_H, centre) is None
    assert "cropped clear" in composition.why_not(1920, 1080, OUT_W, OUT_H, centre)


def test_the_game_crop_follows_the_action_but_never_onto_the_webcam():
    w, h, cam = OVERLAYS["small_16x9_left"]
    right = composition.stack(w, h, OUT_W, OUT_H, cam, focus="balanced", action=0.85)
    left = composition.stack(w, h, OUT_W, OUT_H, cam, focus="balanced", action=0.05)
    assert right["game_crop"][2] > left["game_crop"][2]
    assert left["game_crop"][2] >= cam["box"][0] + cam["box"][2]


def test_env_overrides_make_campaign_profiles_without_code(monkeypatch):
    w, h, cam = OVERLAYS["tall_with_counter"]
    base = composition.stack(w, h, OUT_W, OUT_H, cam, focus="balanced")
    monkeypatch.setenv("FACE_MIN_SHARE", "0.16")
    monkeypatch.setenv("FACE_MAX_SHARE", "0.22")
    monkeypatch.setenv("FACECAM_MAX_SHARE", "0.46")
    bigger = composition.stack(w, h, OUT_W, OUT_H, cam, focus="balanced")
    assert bigger["face_share"] > base["face_share"]


def test_bottom_facecam_position_swaps_the_bands():
    w, h, cam = OVERLAYS["small_16x9_left"]
    g = composition.stack(w, h, OUT_W, OUT_H, cam, focus="balanced", cam_position="bottom")
    assert g["cam_position"] == "bottom"
    assert abs(g["seam"] - g["game_h"] / OUT_H) < 0.01
    assert facecam.stack_filtergraph(g, OUT_W, OUT_H).endswith("[game][cam]vstack=inputs=2,setsar=1[v]")


# --- auto focus -----------------------------------------------------------------------

def test_auto_focus_follows_the_moment(monkeypatch):
    assert composition.resolve_focus("auto") == "balanced"
    assert composition.resolve_focus("auto", ratings={"reaction": "strong", "visual_action": "clear"}) == "streamer"
    assert composition.resolve_focus("auto", ratings={"reaction": "weak", "visual_action": "strong"}) == "gameplay"
    assert composition.resolve_focus("auto", signals={"streamer": {"laughing": True}}) == "streamer"
    assert composition.resolve_focus("auto", signals={"streamer": {"excitement": 3.4}}) == "streamer"
    assert composition.resolve_focus("auto", signals={"streamer": {"excitement": 1.2}}) == "balanced"
    assert composition.resolve_focus("gameplay", ratings={"reaction": "strong"}) == "gameplay"
    assert composition.resolve_focus("nonsense") == "balanced"
    monkeypatch.setenv("FRAMING_FOCUS", "streamer")
    assert composition.resolve_focus() == "streamer"


# --- gameplay priority ------------------------------------------------------------------

def test_gameplay_priority_zooms_past_the_strip_and_keeps_the_facecam_whole():
    box = (0, 0, 492, 306)
    x, w = composition.gameplay_zoom(1920, 1080, OUT_W, OUT_H, cam_box=box)
    assert x <= box[0] and x + w >= box[0] + box[2]
    assert 1.2 < 1920 / w <= 1.3                  # the game ~25% larger than the old strip
    right = (1500, 700, 420, 380)
    x, w = composition.gameplay_zoom(1920, 1080, OUT_W, OUT_H, cam_box=right, action=0.1)
    assert x + w >= 1920 - 2                       # the action cannot pull the webcam out


def test_gameplay_priority_zoom_is_configurable_and_portrait_sources_are_left_alone(monkeypatch):
    monkeypatch.setenv("GAMEPLAY_ZOOM", "1.0")
    assert composition.gameplay_zoom(1920, 1080, OUT_W, OUT_H) is None
    monkeypatch.delenv("GAMEPLAY_ZOOM")
    assert composition.gameplay_zoom(1080, 1920, OUT_W, OUT_H) is None


def test_action_centre_ignores_the_webcam_columns():
    cols = [0.0] * 100
    for i in range(0, 20):
        cols[i] = 50.0          # the streamer moving in a left webcam
    for i in range(70, 90):
        cols[i] = 10.0          # the game's action on the right
    centre = composition.action_centre(cols, 0.5, mask=(0.0, 0.2))
    assert centre is not None and centre > 0.55
    assert composition.action_centre([1.0] * 100, 0.5) is None   # spread motion stays centred


# --- the filtergraph stays readable by the one-pass GPU renderer -------------------------

@pytest.mark.parametrize("name", sorted(OVERLAYS))
def test_stack_graph_uses_only_gpu_readable_filters(name):
    w, h, cam = OVERLAYS[name]
    g = composition.stack(w, h, OUT_W, OUT_H, cam, focus="balanced")
    if g is None:
        return
    layers = gpu_render.layers_from_graph(facecam.stack_filtergraph(g, OUT_W, OUT_H), w, h, OUT_W, OUT_H)
    assert layers


# --- captions and hook on the stack ------------------------------------------------------

def test_stack_ranges_carry_their_seam_and_caption_height(tmp_path):
    clip = str(tmp_path / "c.mp4")
    layout_ranges.write(clip, [(0, 20, "STACK", (0.34, 0.92), {"seam": 0.34, "caption_y": 0.34})])
    got = layout_ranges.read(clip)
    assert got[0]["seam"] == 0.34 and got[0]["caption_y"] == 0.34
    assert layout_ranges.split_ranges(got) == [(0.0, 20.0, 0.34)]
    assert layout_ranges.seam_of(got, 5.0) == 0.34
    # A SPLIT keeps the old mid-frame seam and its two-value shape.
    assert layout_ranges.split_ranges([(0, 5, "SPLIT", None)]) == [(0.0, 5.0)]
    # Old stack sidecars (no anchors) keep the old seam.
    assert layout_ranges.split_ranges([(0, 5, "STACK", None)]) == [(0.0, 5.0, 0.4)]


def _ass_events(tmp_path, split_ranges):
    transcript = {"segments": [{"start": 0, "end": 3, "text": "one two three", "words": [
        {"word": "one", "start": 0.0, "end": 0.8}, {"word": "two", "start": 0.9, "end": 1.6},
        {"word": "three", "start": 1.7, "end": 2.5}]}]}
    path = tmp_path / "s.ass"
    assert subtitles.generate_ass(transcript, 0, 3, str(path), split_ranges=split_ranges)
    return [line for line in path.read_text(encoding="utf-8-sig").splitlines() if line.startswith("Dialogue:")]


def test_stack_captions_sit_where_the_composition_put_them(tmp_path):
    events = _ass_events(tmp_path, [(0.0, 3.0, 0.34)])
    assert events and all("{\\an2}" in e for e in events)
    margin = int(events[0].split(",")[7])
    assert abs((288 - margin) / 288.0 - 0.34) < 0.06      # bottom of the line just under the seam
    split = _ass_events(tmp_path, [(0.0, 3.0)])
    assert all("{\\an5}" in e and e.split(",")[7] == "0" for e in split)


def test_the_hook_never_sits_on_the_seam_captions():
    ranges = [{"start": 0, "end": 20, "layout": "stack", "seam": 0.34, "caption_y": 0.34}]
    y = hooks.seam_hook_y(ranges, OUT_H, 150)
    assert y >= 0.34 * OUT_H + 0.04 * OUT_H            # under the captions, off the face
    fill = [{"start": 0, "end": 20, "layout": "stack", "seam": 0.30, "caption_y": 0.74}]
    assert hooks.seam_hook_y(fill, OUT_H, 150) == int(0.30 * OUT_H) - 75
    assert hook_placement.caption_bands(ranges, 0, 5) == [(0.28, 0.4)]


# --- the composition check ---------------------------------------------------------------

def test_describe_flags_a_blown_up_or_oversized_facecam():
    g = composition.stack(1920, 1080, OUT_W, OUT_H, OVERLAYS["small_16x9_left"][2], focus="balanced")
    ok = composition_check.describe({"kind": "stack", "geometry": g, "focus": "balanced"}, 1920, 1080, OUT_W, OUT_H)
    assert not set(ok["flags"]) & composition_check.REVIEW
    bad = dict(g, face_share=0.40, cam_scale=4.0)
    flagged = composition_check.describe({"kind": "stack", "geometry": bad, "focus": "balanced"},
                                         1920, 1080, OUT_W, OUT_H)
    assert {"facecam_too_large", "facecam_upscaled"} <= set(flagged["flags"])
    low = composition_check.describe({"kind": "gameplay", "crop": None}, 854, 480, OUT_W, OUT_H)
    assert "low_resolution_source" in low["flags"]


def test_same_face_twice_is_caught_and_two_people_are_not():
    np = pytest.importorskip("numpy")
    pytest.importorskip("cv2")
    rng = np.random.default_rng(7)
    face = rng.integers(0, 255, (60, 50, 3), dtype=np.uint8)
    other = rng.integers(0, 255, (60, 50, 3), dtype=np.uint8)
    frame = np.zeros((960, 540, 3), dtype=np.uint8)
    frame[100:160, 100:150] = face
    big = np.kron(face, np.ones((2, 2, 1), dtype=np.uint8))
    frame[500:620, 200:300] = big                    # the same face again, enlarged
    dup = [(frame, [(100, 100, 50, 60), (200, 500, 100, 120)])] * 4
    assert composition_check.same_face_twice(dup) == 1.0
    two = frame.copy()
    two[500:620, 200:300] = np.kron(other, np.ones((2, 2, 1), dtype=np.uint8))
    assert composition_check.same_face_twice([(two, [(100, 100, 50, 60), (200, 500, 100, 120)])] * 4) == 0.0


def test_verify_marks_a_duplicate_for_review(monkeypatch, tmp_path):
    import export_quality
    monkeypatch.setattr(export_quality, "validate", lambda *a, **k: {"valid": True, "issues": [],
                                                                    "video": {"width": 1080, "height": 1920}})
    monkeypatch.setattr(composition_check, "duplicate_faces", lambda path, d: (0.6, 7))
    report = composition_check.verify(str(tmp_path / "x.mp4"), expected_duration=10, planned={"flags": []})
    assert report["needs_review"] and report["review_reasons"] == ["duplicate_face"]
    monkeypatch.setattr(composition_check, "duplicate_faces", lambda path, d: (0.0, 7))
    assert not composition_check.verify(str(tmp_path / "x.mp4"), expected_duration=10)["needs_review"]


# --- choosing the layout ---------------------------------------------------------------------

def _choose(monkeypatch, cam, layout, focus=None, **kw):
    import viral_edit
    monkeypatch.setattr(facecam, "detect", lambda *a, **k: cam)
    return viral_edit.choose_layout("x.mp4", start=0, end=30, origin=0, orig_w=1920, orig_h=1080,
                                    out_w=OUT_W, out_h=OUT_H, layout=layout, focus=focus, **kw)


def test_streamer_framing_stacks_only_a_confident_overlay(monkeypatch):
    cam = dict(OVERLAYS["small_16x9_left"][2], source_size=(1920, 1080), confident=True)
    assert _choose(monkeypatch, cam, "streamer")["kind"] == "stack"
    unsure = dict(cam, confident=False)
    chosen = _choose(monkeypatch, unsure, "streamer")
    assert chosen["kind"] == "gameplay" and "uncertain" in chosen["notes"][0]
    assert _choose(monkeypatch, None, "streamer")["kind"] == "gameplay"


def test_gameplay_priority_gives_a_reaction_its_own_band(monkeypatch):
    cam = dict(OVERLAYS["small_16x9_left"][2], source_size=(1920, 1080), confident=True)
    plain = _choose(monkeypatch, cam, "gameplay")
    assert plain["kind"] == "gameplay" and plain["crop"] is not None
    reaction = _choose(monkeypatch, cam, "gameplay", ratings={"reaction": "strong", "visual_action": "clear"})
    assert reaction["kind"] == "stack" and reaction["focus"] == "balanced"


def test_remote_worker_carries_the_focus():
    import remote_worker
    assert remote_worker.allowed_option("FRAMING_FOCUS", "streamer")
    assert not remote_worker.allowed_option("FRAMING_FOCUS", "huge")


# --- legacy "Speaker and screen layouts": the streamer once -------------------------------

def test_presenter_layouts_on_a_stream_become_one_stack(monkeypatch):
    import reframe_v2
    cam = dict(OVERLAYS["small_16x9_left"][2], source_size=(1920, 1080), confident=True)
    monkeypatch.setattr(facecam, "detect", lambda *a, **k: cam)
    present = {0: True, 1: True, 2: False, 3: True}
    monkeypatch.setattr(facecam, "present", lambda src, c, s, e, **k: present[int(s // 10)])
    bounds = [(0, 300), (300, 600), (600, 900), (900, 1200)]
    strategies = ["SCREENCAST", "SPLIT", "SPLIT", "TRACK"]
    stacks = reframe_v2.stream_scenes("x.mp4", bounds, strategies, 30.0, 1920, 1080, OUT_W, OUT_H,
                                      crop_overrides={3: 0.5})
    # Scene 0/1: the webcam is there, the presenter panel would repeat it.
    assert strategies[:2] == ["STACK", "STACK"] and set(stacks) == {0, 300}
    # Scene 2: no webcam in place, but a presenter panel could still be the
    # streamer: the whole picture, one face.
    assert strategies[2] == "GAMEPLAY"
    # Scene 3: framed by hand, left alone.
    assert strategies[3] == "TRACK"


def test_an_uncertain_overlay_leaves_the_legacy_layouts_alone(monkeypatch):
    import reframe_v2
    cam = dict(OVERLAYS["small_16x9_left"][2], source_size=(1920, 1080), confident=False, evidence=0.05)
    calls = []
    monkeypatch.setattr(facecam, "detect", lambda *a, **k: calls.append(k) or cam)
    strategies = ["SPLIT", "TRACK"]
    assert reframe_v2.stream_scenes("x.mp4", [(0, 300), (300, 600)], strategies, 30.0,
                                    1920, 1080, OUT_W, OUT_H) == {}
    assert strategies == ["SPLIT", "TRACK"]
    assert len(calls) == 1 and calls[0].get("samples") == 8   # only the cheap probe ran


def test_an_explicit_streamer_size_applies_in_gameplay_priority(monkeypatch):
    cam = dict(OVERLAYS["small_16x9_left"][2], source_size=(1920, 1080), confident=True)
    assert _choose(monkeypatch, cam, "gameplay", focus="streamer")["focus"] == "streamer"
    assert _choose(monkeypatch, cam, "gameplay", focus="balanced")["kind"] == "stack"
    assert _choose(monkeypatch, cam, "gameplay", focus="gameplay")["kind"] == "gameplay"
