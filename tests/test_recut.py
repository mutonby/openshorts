"""Unit tests for recut.py — the clip editor's EDL engine.

Pure logic only: validation, range math, the piecewise transcript remap that
keeps captions correct on multi-segment clips, and the ffmpeg command shapes.
The heavy render paths are exercised through injected fakes, never real ffmpeg.
"""

import os

import pytest

import recut


def _seg(start, end):
    return {"start": start, "end": end}


TRANSCRIPT = {
    "language": "en",
    "segments": [
        {
            "start": 0.0, "end": 60.0, "text": "hello world again",
            "words": [
                {"word": "hello", "start": 12.0, "end": 12.5},
                {"word": "world", "start": 20.0, "end": 20.4},
                {"word": "again", "start": 50.0, "end": 50.5},
            ],
        },
    ],
}


class TestNormalizeSegments:
    def test_valid_segments_pass_through_in_order(self):
        segs = recut.normalize_segments([_seg(50, 60), _seg(10, 20)])
        # Order is the CLIP order — reordering source material is legal.
        assert segs == [_seg(50.0, 60.0), _seg(10.0, 20.0)]

    def test_clamps_to_source_duration(self):
        segs = recut.normalize_segments([_seg(-5, 20), _seg(90, 500)],
                                        source_duration=100)
        assert segs == [_seg(0.0, 20.0), _seg(90.0, 100.0)]

    def test_rejects_empty_and_non_list(self):
        for bad in ([], None, "nope", {}):
            with pytest.raises(recut.RecutError):
                recut.normalize_segments(bad)

    def test_rejects_bad_numbers(self):
        for bad in ([{"start": "x", "end": 5}], [{"end": 5}],
                    [{"start": float("nan"), "end": 5}]):
            with pytest.raises(recut.RecutError):
                recut.normalize_segments(bad)

    def test_rejects_too_short_segment(self):
        with pytest.raises(recut.RecutError):
            recut.normalize_segments([_seg(10, 10.2)])
        # A clamp can also make it too short — that must fail, not slip by.
        with pytest.raises(recut.RecutError):
            recut.normalize_segments([_seg(99.9, 150)], source_duration=100)

    def test_rejects_too_many_segments(self):
        segs = [_seg(i * 10, i * 10 + 5) for i in range(recut.MAX_SEGMENTS + 1)]
        with pytest.raises(recut.RecutError):
            recut.normalize_segments(segs)

    def test_rejects_total_over_cap(self):
        with pytest.raises(recut.RecutError):
            recut.normalize_segments([_seg(0, recut.MAX_TOTAL_SECONDS + 10)])


class TestRangeMath:
    def test_total_duration(self):
        assert recut.total_duration([_seg(10, 20), _seg(30, 35)]) == 15.0

    def test_within_range(self):
        assert recut.within_range([_seg(10, 20)], 10, 40)
        assert not recut.within_range([_seg(5, 20)], 10, 40)
        assert not recut.within_range([_seg(10, 41)], 10, 40)

    def test_within_range_tolerates_float_noise(self):
        assert recut.within_range([_seg(9.98, 40.02)], 10, 40)

    def test_rebase_segments_clamps_to_file_bounds(self):
        rebased = recut.rebase_segments([_seg(9.98, 40.02)], 10, 40)
        assert rebased == [_seg(0.0, 30.0)]

    def test_rebase_segments_shifts_by_range_start(self):
        rebased = recut.rebase_segments([_seg(15, 25), _seg(30, 35)], 10)
        assert rebased == [_seg(5.0, 15.0), _seg(20.0, 25.0)]


class TestTranscriptWords:
    def test_flattens_and_sorts(self):
        transcript = {
            "segments": [
                {"words": [{"word": "b", "start": 5.0, "end": 5.5}]},
                {"words": [{"word": "a", "start": 1.0, "end": 1.5}]},
            ],
        }
        words = recut.transcript_words(transcript)
        assert [w["w"] for w in words] == ["a", "b"]

    def test_survives_missing_and_broken_words(self):
        transcript = {
            "segments": [
                {"words": None},
                {},
                {"words": [{"word": "ok", "start": "1", "end": 2},
                           {"word": "bad", "start": None, "end": 2}]},
            ],
        }
        words = recut.transcript_words(transcript)
        assert [w["w"] for w in words] == ["ok"]

    def test_empty_transcript(self):
        assert recut.transcript_words(None) == []
        assert recut.transcript_words({}) == []


class TestVirtualTranscript:
    def test_two_segments_remap_onto_clip_timeline(self):
        v = recut.virtual_transcript(TRANSCRIPT, [_seg(10, 15), _seg(48, 52)])
        assert v["language"] == "en"
        assert len(v["segments"]) == 2
        # "hello" (12.0-12.5 in source) → 2.0-2.5 on the clip.
        first = v["segments"][0]
        assert first["words"] == [{"word": "hello", "start": 2.0, "end": 2.5}]
        assert (first["start"], first["end"]) == (0.0, 5.0)
        # "again" (50.0-50.5) → second segment starts at offset 5.0 → 7.0-7.5.
        second = v["segments"][1]
        assert second["words"] == [{"word": "again", "start": 7.0, "end": 7.5}]
        assert (second["start"], second["end"]) == (5.0, 9.0)

    def test_partial_overlap_is_clamped_to_the_segment(self):
        # Word 12.0-12.5, segment 12.3-20 → starts at 0, ends at 0.2.
        v = recut.virtual_transcript(TRANSCRIPT, [_seg(12.3, 20)])
        word = v["segments"][0]["words"][0]
        assert word["word"] == "hello"
        assert word["start"] == 0.0
        assert word["end"] == 0.2

    def test_words_outside_every_segment_are_dropped(self):
        v = recut.virtual_transcript(TRANSCRIPT, [_seg(30, 40)])
        assert v["segments"][0]["words"] == []
        assert v["segments"][0]["text"] == ""


class TestSnapSegments:
    def test_snaps_onto_word_boundaries(self):
        # 11.8 is near "hello"'s start (12.0); 20.6 near "world"'s end (20.4).
        snapped = recut.snap_segments([_seg(11.8, 20.6)], TRANSCRIPT, 60.0)
        start, end = snapped[0]["start"], snapped[0]["end"]
        assert 11.5 <= start <= 12.0
        assert 20.4 <= end <= 20.9

    def test_no_words_returns_input(self):
        segs = [_seg(1, 5)]
        assert recut.snap_segments(segs, {"segments": []}, 60.0) is segs


class TestFfmpegCommands:
    @pytest.fixture(autouse=True)
    def _stable_encode_args(self, monkeypatch):
        monkeypatch.setattr(recut, "video_encode_args", lambda tier: ["-c:v", "test"])
        monkeypatch.setattr(recut, "audio_encode_args", lambda: ["-c:a", "test"])

    def test_cut_commands_shape(self):
        commands = recut.cut_commands("in.mp4", [_seg(10, 20), _seg(30, 35)],
                                      ["p0.mp4", "p1.mp4"])
        assert commands[0] == ["ffmpeg", "-y", "-ss", "10", "-to", "20",
                               "-i", "in.mp4", "-c:v", "test", "-c:a", "test",
                               "p0.mp4"]
        assert commands[1][3] == "30" and commands[1][-1] == "p1.mp4"

    def test_concat_command_stream_copies(self):
        cmd = recut.concat_command("list.txt", "out.mp4")
        assert "-c" in cmd and cmd[cmd.index("-c") + 1] == "copy"

    def test_single_segment_cuts_straight_to_output(self, tmp_path):
        ran = []
        recut.run_cut_concat("in.mp4", [_seg(10, 20)],
                             str(tmp_path / "out.mp4"), str(tmp_path),
                             runner=ran.append)
        assert len(ran) == 1
        assert ran[0][-1] == str(tmp_path / "out.mp4")

    def test_multi_segment_concats_and_cleans_parts(self, tmp_path):
        ran = []

        def fake_run(cmd):
            ran.append(cmd)
            with open(cmd[-1], "wb") as f:
                f.write(b"x")

        out = str(tmp_path / "out.mp4")
        recut.run_cut_concat("in.mp4", [_seg(10, 20), _seg(30, 35)], out,
                             str(tmp_path), runner=fake_run)
        # Two cuts + one concat, and no part/list files left behind.
        assert len(ran) == 3
        assert ran[2][:4] == ["ffmpeg", "-y", "-f", "concat"]
        assert os.listdir(tmp_path) == ["out.mp4"]


class TestPerformRecut:
    @pytest.fixture(autouse=True)
    def _stable_encode_args(self, monkeypatch):
        monkeypatch.setattr(recut, "video_encode_args", lambda tier: [])
        monkeypatch.setattr(recut, "audio_encode_args", lambda: [])

    @staticmethod
    def _touching_runner(cmd):
        with open(cmd[-1], "wb") as f:
            f.write(b"x")

    def test_fast_path_no_reframe_no_captions(self, tmp_path):
        served, clean = recut.perform_recut(
            input_path="clip.mp4", segments=[_seg(0, 10)],
            output_dir=str(tmp_path), clean_name="t_clip_1.mp4",
            runner=self._touching_runner)
        assert served == clean
        assert served.startswith("recut_") and served.endswith("_t_clip_1.mp4")
        assert os.path.exists(tmp_path / served)
        # The temp work file is gone.
        assert all(not f.startswith("temp_") for f in os.listdir(tmp_path))

    def test_captions_burn_last_and_win_the_served_name(self, tmp_path):
        captioned = []

        def fake_captioner(path, transcript, start, end):
            captioned.append((os.path.basename(path), start, end))
            out = os.path.join(os.path.dirname(path),
                               f"subtitled_1_{os.path.basename(path)}")
            with open(out, "wb") as f:
                f.write(b"x")
            return out

        v_transcript = {"segments": [{"words": [{"word": "a", "start": 0, "end": 1}]}]}
        served, clean = recut.perform_recut(
            input_path="clip.mp4", segments=[_seg(0, 10)],
            output_dir=str(tmp_path), clean_name="t_clip_1.mp4",
            captions_transcript=v_transcript,
            runner=self._touching_runner, captioner=fake_captioner)
        assert served == f"subtitled_1_{clean}"
        # Captioned over the CLEAN recut, with the clip-relative window.
        assert captioned == [(clean, 0.0, 10.0)]
        # Both files remain: clean for re-styling, captioned for serving.
        assert os.path.exists(tmp_path / clean)
        assert os.path.exists(tmp_path / served)

    def test_source_path_reframes_and_watermarks(self, tmp_path):
        events = []

        def fake_renderer(work, out, output_format):
            events.append(("reframe", output_format))
            with open(out, "wb") as f:
                f.write(b"x")
            return True

        served, clean = recut.perform_recut(
            input_path="source.mp4", segments=[_seg(5, 15)],
            output_dir=str(tmp_path), clean_name="t_clip_1.mp4",
            reframe=True, output_format="vertical", watermark=True,
            runner=self._touching_runner, renderer=fake_renderer,
            watermarker=lambda path: events.append(("watermark",)))
        assert events == [("reframe", "vertical"), ("watermark",)]
        assert served == clean

    def test_renderer_failure_raises_and_cleans_up(self, tmp_path):
        with pytest.raises(RuntimeError):
            recut.perform_recut(
                input_path="source.mp4", segments=[_seg(5, 15)],
                output_dir=str(tmp_path), clean_name="t_clip_1.mp4",
                reframe=True, runner=self._touching_runner,
                renderer=lambda *a: False)
        assert all(not f.startswith("temp_") for f in os.listdir(tmp_path))
