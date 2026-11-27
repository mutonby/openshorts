"""Captions on the seam of SPLIT scenes: the render records which stretches
are stacked (layout_ranges sidecar), and generate_ass anchors the word events
inside them mid-frame (\\an5) while everything else keeps the style's own
alignment. A clip mixes both, so the placement is per event, not per style."""
import json
import os

import pytest

import layout_ranges
import subtitles


def _transcript(words):
    return {"language": "en", "segments": [{
        "start": words[0][1], "end": words[-1][2],
        "words": [{"word": w, "start": s, "end": e} for w, s, e in words]}]}


class TestSidecar:
    def test_write_then_read_roundtrip(self, tmp_path):
        clip = str(tmp_path / "clip_1.mp4")
        layout_ranges.write(clip, [(0, 4.5, "TRACK"), (4.5, 10, "SPLIT")])
        assert os.path.exists(clip + ".layout.json")
        got = layout_ranges.read(clip)
        assert got == [{"start": 0.0, "end": 4.5, "layout": "track"},
                       {"start": 4.5, "end": 10.0, "layout": "split"}]
        assert layout_ranges.split_ranges(got) == [(4.5, 10.0)]

    def test_missing_sidecar_is_empty(self, tmp_path):
        assert layout_ranges.read(str(tmp_path / "nope.mp4")) == []

    def test_garbage_is_ignored_not_raised(self, tmp_path):
        clip = str(tmp_path / "c.mp4")
        with open(clip + ".layout.json", "w") as f:
            f.write("{not json")
        assert layout_ranges.read(clip) == []
        assert layout_ranges.normalise([{"start": "x"}, (3, 1, "split"), (1, 2, "SPLIT"), 7]) == [
            {"start": 1.0, "end": 2.0, "layout": "split"}]

    def test_write_failure_does_not_raise(self, tmp_path):
        layout_ranges.write(str(tmp_path / "missing_dir" / "c.mp4"), [(0, 1, "split")])


class TestSeamCaptions:
    def _events(self, tmp_path, split_ranges, **kw):
        out = tmp_path / "subs.ass"
        # Leading spaces as Whisper emits them: without them the merge step
        # glues the words into one token and one block.
        words = [(" one", 10.0, 10.4), (" two", 10.5, 10.9),   # single shot
                 (" three", 13.0, 13.4), (" four", 13.5, 13.9)]  # stacked
        ok = subtitles.generate_ass(_transcript(words), 10.0, 14.0, str(out),
                                    max_chars=8, split_ranges=split_ranges, **kw)
        assert ok
        return [l for l in out.read_text(encoding="utf-8-sig").splitlines()
                if l.startswith("Dialogue:")]

    def test_only_split_stretches_move_to_the_seam(self, tmp_path):
        # Ranges are in CLIP seconds, like the events themselves.
        ev = self._events(tmp_path, [(2.5, 4.0)])
        first_block = [e for e in ev if "one" in e]
        second_block = [e for e in ev if "three" in e]
        assert first_block and second_block
        assert all("{\\an5}" not in e for e in first_block)
        assert all(e.split(",,")[-1].startswith("{\\an5}") for e in second_block)

    def test_no_ranges_changes_nothing(self, tmp_path):
        assert all("\\an5" not in e for e in self._events(tmp_path, None))
        assert all("\\an5" not in e for e in self._events(tmp_path, []))

    def test_style_alignment_still_rules_elsewhere(self, tmp_path):
        out = tmp_path / "s.ass"
        subtitles.generate_ass(_transcript([(" hi", 0.0, 0.5)]), 0.0, 1.0, str(out),
                               alignment="top", split_ranges=[(5, 6)])
        header = out.read_text(encoding="utf-8-sig")
        assert ",8,10,10," in header  # Alignment 8 = top in the Style line


class TestCaptionerReadsSidecar:
    def test_auto_caption_uses_sidecar_when_not_told(self, tmp_path, monkeypatch):
        main = pytest.importorskip("main")
        clip = str(tmp_path / "x.mp4")
        open(clip, "wb").close()
        layout_ranges.write(clip, [(0, 3, "SPLIT")])
        seen = {}

        def fake_generate_ass(*a, **kw):
            seen["split_ranges"] = kw.get("split_ranges")
            return False  # stop before ffmpeg
        monkeypatch.setattr(subtitles, "generate_ass", fake_generate_ass)
        main.auto_caption_clip(clip, _transcript([(" a", 0.0, 0.5)]), 0.0, 3.0)
        assert seen["split_ranges"] == [(0.0, 3.0)]


class TestRemapAcrossCut:
    def test_ranges_follow_the_kept_segments(self):
        ranges = [(0, 10, "split"), (10, 20, "track"), (20, 30, "split")]
        # Keep 5-12 and 22-30 of the source: output is 0-7 then 7-15.
        segs = [{"start": 5, "end": 12}, {"start": 22, "end": 30}]
        got = layout_ranges.remap(ranges, segs)
        assert got == [
            {"start": 0.0, "end": 5.0, "layout": "split"},
            {"start": 5.0, "end": 7.0, "layout": "track"},
            {"start": 7.0, "end": 15.0, "layout": "split"},
        ]
        assert layout_ranges.split_ranges(got) == [(0.0, 5.0), (7.0, 15.0)]

    def test_no_input_ranges_means_none(self):
        assert layout_ranges.remap([], [{"start": 0, "end": 5}]) == []

    def test_fast_recut_carries_the_sidecar(self, tmp_path, monkeypatch):
        import shutil
        import recut
        src = tmp_path / "canon.mp4"
        src.write_bytes(b"x")
        layout_ranges.write(str(src), [
            (0, 4, "STACK", None, {"seam": 0.31, "caption_y": 0.48}), (4, 8, "TRACK"),
        ])
        # Stand in for ffmpeg: the "cut" is a copy.
        monkeypatch.setattr(recut, "run_cut_concat",
                            lambda i, segs, out, wd, runner=None: shutil.copy(i, out))
        served, clean = recut.perform_recut(
            input_path=str(src), segments=[{"start": 2, "end": 6}],
            output_dir=str(tmp_path), clean_name="c.mp4", reframe=False)
        ranges = layout_ranges.read(str(tmp_path / clean))
        assert ranges == [
            {"start": 0.0, "end": 2.0, "layout": "stack", "seam": 0.31, "caption_y": 0.48},
            {"start": 2.0, "end": 4.0, "layout": "track"}]
        assert layout_ranges.seam_of(ranges, 1) == 0.31
        assert layout_ranges.caption_anchors(ranges) == [(0.0, 2.0, 0.48)]


class TestLayoutEnvNone:
    def test_none_switches_the_picker_off_for_the_job(self):
        app_module = pytest.importorskip("app")
        assert app_module.layout_env(["none"]) == {"AUTO_LAYOUT": "0"}
        assert app_module.layout_env(["auto"]) == {"AUTO_LAYOUT": "1"}
        assert app_module.layout_env(["split"])["SPLIT_LAYOUT"] == "1"
