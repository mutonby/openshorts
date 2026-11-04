"""Hook placement (position="auto"): off the faces, off the captions, and
exactly the old spot when there is no face. Faces are synthetic boxes in
frame fractions (x0, y0, x1, y1), the shape hook_placement.detect_faces
returns."""
import os
import subprocess

import pytest

import hook_placement as hp

BOX_W, BOX_H = 0.85, 0.10   # a two-line pill hook on a 1080x1920 clip


def _off_head(y, box_h, face):
    """True when a hook at ``y`` leaves the grown head box alone."""
    hb = hp.head_box(face)
    return y + box_h <= hb[1] + 1e-9 or y >= hb[3] - 1e-9


def _off_bands(y, box_h, bands):
    return all(y + box_h <= b0 + 1e-9 or y >= b1 - 1e-9 for b0, b1 in bands)


class TestChooseY:
    def test_no_face_keeps_the_legacy_top_spot(self):
        bands = [hp.BOTTOM_CAPTIONS]
        y, cost = hp.choose_y(BOX_W, BOX_H, [[], [], []], bands)
        assert y == pytest.approx(hp.LEGACY_TOP)
        assert cost == 0

    def test_track_speaker_hook_leaves_the_face(self):
        # Single face-tracked speaker, brows-to-chin at 22-34% of the height:
        # the legacy 20% box (20-30%) sits on the eyes.
        face = (0.35, 0.22, 0.65, 0.34)
        frames = [[face]] * 6
        bands = hp.caption_bands([{"start": 0, "end": 30, "layout": "track"}], 0, 5)
        legacy = hp.face_cost((0.075, 0.20, 0.925, 0.30), frames)
        assert legacy > 0.3
        y, cost = hp.choose_y(BOX_W, BOX_H, frames, bands)
        assert cost == 0
        assert _off_head(y, BOX_H, face)
        assert _off_bands(y, BOX_H, bands)
        assert hp.TOP_MARGIN <= y and y + BOX_H <= hp.BOTTOM_UI

    def test_split_hook_goes_between_top_face_and_seam_captions(self):
        # Two speakers stacked: faces centred in each half, captions on the seam.
        top_face = (0.38, 0.13, 0.62, 0.22)
        bottom_face = (0.38, 0.65, 0.62, 0.74)
        frames = [[top_face, bottom_face]] * 6
        bands = hp.caption_bands([{"start": 0, "end": 50, "layout": "split"}], 0, 5)
        assert bands == [hp.SEAM_CAPTIONS]
        y, cost = hp.choose_y(BOX_W, BOX_H, frames, bands)
        assert cost == 0
        assert _off_head(y, BOX_H, top_face) and _off_head(y, BOX_H, bottom_face)
        assert _off_bands(y, BOX_H, bands)
        # Nearest clean spot to the old 20%: under the top face, over the seam.
        assert hp.head_box(top_face)[3] <= y and y + BOX_H <= hp.SEAM_CAPTIONS[0]

    def test_ties_go_to_the_spot_nearest_the_old_one(self):
        # A small face far from the top: 20% is already clean, keep it.
        face = (0.45, 0.60, 0.55, 0.65)
        y, cost = hp.choose_y(BOX_W, BOX_H, [[face]], [hp.BOTTOM_CAPTIONS])
        assert y == pytest.approx(hp.LEGACY_TOP) and cost == 0

    def test_never_lands_on_captions_even_when_faces_fill_the_rest(self):
        # Faces everywhere a hook may go: it still must not cover the captions.
        faces = [(0.0, y, 1.0, y + 0.05) for y in (0.1, 0.25, 0.4, 0.55)]
        bands = [hp.SEAM_CAPTIONS, hp.BOTTOM_CAPTIONS]
        y, _ = hp.choose_y(BOX_W, BOX_H, [faces], bands)
        assert y is not None and _off_bands(y, BOX_H, bands)

    def test_a_box_too_tall_for_every_gap_finds_nothing(self):
        bands = [hp.SEAM_CAPTIONS, hp.BOTTOM_CAPTIONS]
        y, cost = hp.choose_y(BOX_W, 0.40, [[]], bands)
        assert y is None and cost == float("inf")

    def test_a_face_that_comes_and_goes_counts_by_how_often_it_is_there(self):
        face = (0.3, 0.2, 0.7, 0.3)
        rect = (0.075, 0.2, 0.925, 0.3)
        always = hp.face_cost(rect, [[face]] * 4)
        half = hp.face_cost(rect, [[face], [], [face], []])
        assert half == pytest.approx(always / 2)


class TestCaptionBands:
    def test_no_sidecar_means_bottom_captions(self):
        assert hp.caption_bands([], 0, 5) == [hp.BOTTOM_CAPTIONS]

    def test_split_window_is_the_seam(self):
        ranges = [{"start": 0, "end": 10, "layout": "split"}]
        assert hp.caption_bands(ranges, 0, 5) == [hp.SEAM_CAPTIONS]

    def test_window_that_crosses_a_cut_gets_both(self):
        ranges = [{"start": 0, "end": 3, "layout": "split"},
                  {"start": 3, "end": 9, "layout": "track"}]
        assert set(hp.caption_bands(ranges, 0, 5)) == {hp.SEAM_CAPTIONS, hp.BOTTOM_CAPTIONS}

    def test_split_later_in_the_clip_does_not_matter(self):
        ranges = [{"start": 0, "end": 8, "layout": "track"},
                  {"start": 8, "end": 20, "layout": "split"}]
        assert hp.caption_bands(ranges, 0, 5) == [hp.BOTTOM_CAPTIONS]

    def test_window_past_the_ranges_falls_back_to_bottom(self):
        ranges = [{"start": 0, "end": 2, "layout": "split"}]
        assert set(hp.caption_bands(ranges, 0, 5)) == {hp.SEAM_CAPTIONS, hp.BOTTOM_CAPTIONS}

    def test_no_captions_no_bands(self):
        assert hp.caption_bands([{"start": 0, "end": 5, "layout": "split"}], 0, 5,
                                has_captions=False) == []


class TestBestScale:
    def test_keeps_full_size_when_it_fits(self):
        scale, y, cost = hp.best_scale(lambda s: (BOX_W * s, BOX_H * s), [[]],
                                       [hp.BOTTOM_CAPTIONS])
        assert scale == 1.0 and y == pytest.approx(hp.LEGACY_TOP) and cost == 0

    def test_shrinks_only_when_every_spot_covers_a_face(self):
        # Free strips: 0.06-0.135 above the head and 0.33-0.42 under it, both
        # shorter than a 0.10 box. At 0.72 scale (0.072) the hook fits.
        face = (0.3, 0.21, 0.7, 0.30)       # head box 0.156-0.318
        low = (0.3, 0.70, 0.7, 0.78)        # bottom speaker, below the seam
        frames = [[face, low]] * 3
        bands = [hp.SEAM_CAPTIONS]

        def measure(s):
            return BOX_W * s, 0.115 * s

        full_y, full_cost = hp.choose_y(BOX_W, 0.115, frames, bands)
        assert full_cost > hp.GOOD_ENOUGH
        scale, y, cost = hp.best_scale(measure, frames, bands)
        assert scale < 1.0 and cost <= hp.GOOD_ENOUGH
        assert _off_bands(y, 0.115 * scale, bands)

    def test_nothing_fits_returns_no_spot(self):
        scale, y, cost = hp.best_scale(lambda s: (BOX_W, 0.9), [[]], [hp.SEAM_CAPTIONS])
        assert y is None


class TestAddHookToVideo:
    """The plumbing: auto asks hook_placement, explicit positions do not."""

    @pytest.fixture
    def fake_ffmpeg(self, monkeypatch, tmp_path):
        import hooks
        calls = {}
        monkeypatch.setattr(hooks.subprocess, "check_output", lambda *a, **k: b"1080x1920\n")

        def run(cmd, **kw):
            calls["cmd"] = cmd
            return subprocess.CompletedProcess(cmd, 0, b"", b"")
        monkeypatch.setattr(hooks.subprocess, "run", run)
        video = tmp_path / "clip.mp4"
        video.write_bytes(b"x")
        return hooks, calls, str(video)

    @staticmethod
    def _overlay_y(cmd):
        graph = cmd[cmd.index("-filter_complex") + 1]
        return int(graph.split("overlay=")[1].split(":")[1])

    def test_explicit_top_ignores_faces(self, fake_ffmpeg, monkeypatch, tmp_path):
        hooks, calls, video = fake_ffmpeg
        import hook_placement
        monkeypatch.setattr(hook_placement, "detect_faces",
                            lambda *a, **k: pytest.fail("explicit position must not detect"))
        assert hooks.add_hook_to_video(video, "Hello there", str(tmp_path / "out.mp4"), position="top",
                                       duration=5) is True
        assert self._overlay_y(calls["cmd"]) == int(1920 * 0.20)

    def test_auto_moves_off_a_face(self, fake_ffmpeg, monkeypatch, tmp_path):
        hooks, calls, video = fake_ffmpeg
        import hook_placement
        face = (0.35, 0.20, 0.65, 0.34)
        monkeypatch.setattr(hook_placement, "detect_faces", lambda *a, **k: [[face]] * 6)
        result = hooks.add_hook_to_video(video, "Performance should always outweigh your words.",
                                         str(tmp_path / "out.mp4"), position="auto", duration=5,
                                         layout_ranges=[{"start": 0, "end": 30, "layout": "track"}])
        y = self._overlay_y(calls["cmd"])
        assert result["position"] == "auto"
        # The whole drawn box (its measured height) is off the head.
        assert result["y"] != pytest.approx(hook_placement.LEGACY_TOP)
        assert hook_placement.face_cost(
            (0.0, y / 1920, 1.0, y / 1920 + result["height"]), [[face]]) == 0
        assert y / 1920 == pytest.approx(result["y"], abs=0.001)

    def test_mark_adds_the_served_copy_to_the_same_pass(self, fake_ffmpeg, tmp_path):
        """Free plan: the wm_ copy comes out of the hook+captions ffmpeg, from
        its decode, instead of a second pass over the captioned file."""
        hooks, calls, video = fake_ffmpeg
        made = []

        def make_filter(vw, vh, video, logo, out):
            made.append((vw, vh, video, logo, out))
            return f"{logo}scale=324:-1[wm];{video}[wm]overlay=54:768{out}"

        hooked, captioned, marked = (str(tmp_path / n) for n in
                                     ("hooked.mp4", "subtitled.mp4", "wm_subtitled.mp4"))
        hooks.add_hook_to_video(video, "Hi", hooked, position="top", duration=5,
                                also=("ass=subs.ass", captioned),
                                mark=("logo.png", make_filter, marked))
        cmd = calls["cmd"]
        assert made == [(1080, 1920, "[sm]", "[2:v]", "[m]")]
        assert cmd[cmd.index("logo.png") - 1] == "-i"
        graph = cmd[cmd.index("-filter_complex") + 1]
        assert ",ass=subs.ass,split=2[s][sm];" in graph and "[h]" not in graph
        assert [cmd[i + 1] for i, a in enumerate(cmd) if a == "-map" and cmd[i + 1].startswith("[")] \
            == ["[s]", "[m]"]
        # The hooked-only picture is not encoded: its recipe is kept instead.
        assert cmd[-1] == marked and captioned in cmd and hooked not in cmd
        assert hooks.has_recipe(str(tmp_path), "hooked.mp4")

    def test_hooked_copy_is_encoded_in_the_same_pass_when_not_lazy(self, fake_ffmpeg,
                                                                  monkeypatch, tmp_path):
        hooks, calls, video = fake_ffmpeg
        monkeypatch.setattr(hooks, "LAZY", False)
        hooked, captioned = str(tmp_path / "hooked.mp4"), str(tmp_path / "subtitled.mp4")
        hooks.add_hook_to_video(video, "Hi", hooked, position="top", duration=5,
                                also=("ass=subs.ass", captioned))
        cmd = calls["cmd"]
        assert "split=2[h][c];[c]ass=subs.ass[s]" in cmd[cmd.index("-filter_complex") + 1]
        assert hooked in cmd and captioned in cmd
        assert not hooks.has_recipe(str(tmp_path), "hooked.mp4")

    def test_recipe_encodes_the_hooked_copy_on_demand(self, fake_ffmpeg, monkeypatch, tmp_path):
        hooks, calls, video = fake_ffmpeg
        hooked, captioned = str(tmp_path / "hooked.mp4"), str(tmp_path / "subtitled.mp4")
        hooks.add_hook_to_video(video, "Hi", hooked, position="top", duration=5,
                                also=("ass=subs.ass", captioned))
        burn = calls["cmd"]

        def run(cmd, **kw):
            calls["cmd"] = cmd
            open(cmd[-1], "wb").write(b"mp4")
            return subprocess.CompletedProcess(cmd, 0, b"", b"")
        monkeypatch.setattr(hooks.subprocess, "run", run)
        assert hooks.materialize(str(tmp_path), "hooked.mp4")
        cmd = calls["cmd"]
        # Same overlay, same place, same window as the burn drew it.
        overlay = burn[burn.index("-filter_complex") + 1].split(",ass=")[0]
        assert cmd[cmd.index("-filter_complex") + 1] == overlay
        assert cmd[cmd.index("-i") + 1] == video
        assert open(hooked, "rb").read() == b"mp4"
        assert not [n for n in os.listdir(tmp_path) if n.startswith(".part_")]
        # Already there: nothing to encode.
        monkeypatch.setattr(hooks.subprocess, "run", lambda *a, **k: pytest.fail("re-encoded"))
        assert hooks.materialize(str(tmp_path), "hooked.mp4")

    def test_materialize_without_recipe_or_source_is_false(self, fake_ffmpeg, monkeypatch,
                                                          tmp_path):
        hooks, calls, video = fake_ffmpeg
        assert not hooks.materialize(str(tmp_path), "hooked_1_clip.mp4")
        hooked, captioned = str(tmp_path / "hooked.mp4"), str(tmp_path / "subtitled.mp4")
        hooks.add_hook_to_video(video, "Hi", hooked, position="top", duration=5,
                                also=("ass=subs.ass", captioned))
        os.remove(video)
        monkeypatch.setattr(hooks.subprocess, "run", lambda *a, **k: pytest.fail("encoded"))
        assert not hooks.materialize(str(tmp_path), "hooked.mp4")

    def test_failed_on_demand_encode_leaves_nothing(self, fake_ffmpeg, monkeypatch, tmp_path):
        hooks, calls, video = fake_ffmpeg
        hooked, captioned = str(tmp_path / "hooked.mp4"), str(tmp_path / "subtitled.mp4")
        hooks.add_hook_to_video(video, "Hi", hooked, position="top", duration=5,
                                also=("ass=subs.ass", captioned))

        def run(cmd, **kw):
            open(cmd[-1], "wb").write(b"half")
            return subprocess.CompletedProcess(cmd, 1, b"", b"boom")
        monkeypatch.setattr(hooks.subprocess, "run", run)
        assert not hooks.materialize(str(tmp_path), "hooked.mp4")
        assert not os.path.exists(hooked)
        assert not [n for n in os.listdir(tmp_path) if n.startswith(".part_")]

    def test_a_crash_after_every_output_is_finished_keeps_them(self, monkeypatch, fake_ffmpeg,
                                                              tmp_path):
        hooks, calls, video = fake_ffmpeg
        paths = [str(tmp_path / n) for n in ("h.mp4", "s.mp4", "wm.mp4")]

        def run(cmd, **kw):
            for p in paths:
                open(p, "wb").write(b"x" * 10)
            stats = b"[out#0/mp4] video:1KiB muxing overhead: 0.3%\n" * 3
            return subprocess.CompletedProcess(cmd, -11, b"", stats)
        monkeypatch.setattr(hooks.subprocess, "run", run)
        assert hooks.add_hook_to_video(video, "Hi", paths[0], position="top",
                                       also=("ass=a.ass", paths[1]),
                                       mark=("logo.png", lambda *a, **k: "[2:v]null[m]", paths[2]))

    def test_a_crash_before_every_output_is_finished_still_fails(self, monkeypatch, fake_ffmpeg,
                                                                 tmp_path):
        hooks, calls, video = fake_ffmpeg
        out, cap = str(tmp_path / "h.mp4"), str(tmp_path / "s.mp4")

        def run(cmd, **kw):
            open(out, "wb").write(b"x")
            return subprocess.CompletedProcess(cmd, -11, b"", b"muxing overhead: 0.3%\n")
        monkeypatch.setattr(hooks.subprocess, "run", run)
        with pytest.raises(subprocess.CalledProcessError):
            hooks.add_hook_to_video(video, "Hi", out, position="top", also=("ass=a.ass", cap))

    def test_a_plain_error_exit_still_fails(self, monkeypatch, fake_ffmpeg, tmp_path):
        hooks, calls, video = fake_ffmpeg
        out = str(tmp_path / "h.mp4")

        def run(cmd, **kw):
            open(out, "wb").write(b"x")
            return subprocess.CompletedProcess(cmd, 1, b"", b"muxing overhead: 0.3%\n")
        monkeypatch.setattr(hooks.subprocess, "run", run)
        with pytest.raises(subprocess.CalledProcessError):
            hooks.add_hook_to_video(video, "Hi", out, position="top")

    def test_mark_without_captions_is_ignored(self, fake_ffmpeg, tmp_path):
        hooks, calls, video = fake_ffmpeg
        hooks.add_hook_to_video(video, "Hi", str(tmp_path / "out.mp4"), position="top",
                                mark=("logo.png", lambda *a, **k: "", "wm.mp4"))
        assert "logo.png" not in calls["cmd"]

    def test_auto_keeps_the_hook_off_a_wide_picture(self, fake_ffmpeg, monkeypatch, tmp_path):
        hooks, calls, video = fake_ffmpeg
        import hook_placement
        monkeypatch.setattr(hook_placement, "detect_faces", lambda *a, **k: [[]] * 6)
        band = [0.295, 0.705]
        result = hooks.add_hook_to_video(
            video, "The cops never saw this pileup coming, and neither did we",
            str(tmp_path / "out.mp4"), position="auto", duration=5,
            layout_ranges=[{"start": 0, "end": 30, "layout": "wide", "content": band}])
        y = self._overlay_y(calls["cmd"]) / 1920
        assert y + result["height"] <= band[0] + 1e-3

    def test_auto_ignores_a_picture_it_cannot_clear(self, fake_ffmpeg, monkeypatch, tmp_path):
        hooks, calls, video = fake_ffmpeg
        import hook_placement
        monkeypatch.setattr(hook_placement, "detect_faces", lambda *a, **k: [[]] * 6)
        # A picture from the top margin to the captions: nowhere to go, so the
        # caption-only rule (the old spot) applies.
        hooks.add_hook_to_video(
            video, "Hi there", str(tmp_path / "out.mp4"), position="auto", duration=5,
            layout_ranges=[{"start": 0, "end": 30, "layout": "wide", "content": [0.05, 0.72]}])
        assert self._overlay_y(calls["cmd"]) == int(round(1920 * 0.20))

    def test_auto_without_frames_keeps_the_top_spot(self, fake_ffmpeg, monkeypatch, tmp_path):
        hooks, calls, video = fake_ffmpeg
        import hook_placement
        monkeypatch.setattr(hook_placement, "detect_faces", lambda *a, **k: None)
        assert hooks.add_hook_to_video(video, "Hi", str(tmp_path / "out.mp4"), position="auto",
                                       layout_ranges=[]) is True
        assert self._overlay_y(calls["cmd"]) == int(1920 * 0.20)
