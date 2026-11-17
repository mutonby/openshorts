"""pro_mode.py: what gets cut, how the shots are planned, how the pro file's
recipe maps back to the source, and the host-wide render slots.

The render itself (ffmpeg, mediapipe) is not run here: these tests own the
decisions around it."""

import os

import pytest

np = pytest.importorskip("numpy")
import pro_mode  # noqa: E402

FPS = 30.0


def _words(*spans):
    return [{"w": f"w{i}", "s": s, "e": e} for i, (s, e) in enumerate(spans)]


class TestMode:
    @pytest.mark.parametrize("value, expected", [
        (None, "suggest"), ("", "suggest"), ("AUTO", "auto"), (" off ", "off"),
        ("everything", "suggest")])
    def test_reads_the_environment(self, monkeypatch, value, expected):
        if value is None:
            monkeypatch.delenv("PRO_MODE", raising=False)
        else:
            monkeypatch.setenv("PRO_MODE", value)
        assert pro_mode.mode() == expected


class TestSuggestion:
    def test_track_share_counts_only_track(self):
        ranges = [{"start": 0, "end": 9, "layout": "track"},
                  {"start": 9, "end": 10, "layout": "general"}]
        assert pro_mode.track_share(ranges, 10) == pytest.approx(0.9)

    def test_no_recorded_layouts_is_unknown(self):
        assert pro_mode.track_share([], 10) is None

    def test_one_clear_face_in_most_frames(self):
        samples = [(1, 0.2)] * 8 + [None, None]
        assert pro_mode.faces_look_like_one_speaker(samples)

    def test_two_speakers_are_not_a_talking_head(self):
        assert not pro_mode.faces_look_like_one_speaker([(2, 0.2)] * 6 + [(1, 0.2)] * 2)

    def test_a_screen_with_no_face_is_not_suggested(self):
        assert not pro_mode.faces_look_like_one_speaker([None] * 8)

    def test_a_tiny_face_in_a_corner_is_not_the_subject(self):
        assert not pro_mode.faces_look_like_one_speaker([(1, 0.03)] * 8)

    def test_one_frame_is_not_enough_to_judge(self):
        assert not pro_mode.faces_look_like_one_speaker([(1, 0.2)])

    def test_an_unreadable_clip_is_just_not_flagged(self, tmp_path):
        assert pro_mode.suggested(str(tmp_path / "missing.mp4"), []) is False


class TestCuts:
    def test_a_pause_is_cut_with_air_on_both_sides(self):
        frames = pro_mode.keep_frames([(4.0, 6.0)], _words((0.2, 3.9), (6.1, 9.8)), 10.0, FPS)
        assert frames == [(0, round((4.0 + pro_mode.KEEP_BEFORE) * FPS)),
                          (round((6.0 - pro_mode.KEEP_AFTER) * FPS), 300)]

    def test_leading_silence_goes_up_to_the_first_word(self):
        frames = pro_mode.keep_frames([(0.0, 1.5)], _words((1.6, 9.0)), 10.0, FPS)
        assert frames == [(round((1.5 - pro_mode.KEEP_AFTER) * FPS), 300)]

    def test_nothing_silent_keeps_the_whole_clip(self):
        assert pro_mode.keep_frames([], _words((0.1, 9.9)), 10.0, FPS) == [(0, 300)]

    def test_never_cuts_a_clip_down_to_a_stub(self):
        # 6 s clip, 4 s of it silent: what is left is not a clip.
        assert pro_mode.keep_frames([(1.0, 5.0)], _words((0.1, 0.9)), 6.0, FPS) == [(0, 180)]

    def test_silences_come_from_the_audio_level(self):
        loud = np.full(300, -20.0, dtype=np.float32)
        loud[100:160] = -70.0     # 0.6 s of silence at a 10 ms hop
        loud[200:220] = -70.0     # 0.2 s: too short to be a pause worth cutting
        assert pro_mode.silences(loud) == [(pytest.approx(1.0), pytest.approx(1.6))]

    def test_times_move_onto_the_cut_timeline(self):
        convert = pro_mode.to_output_time([(0, 90), (150, 300)], FPS)
        assert convert(1.0) == pytest.approx(1.0)
        assert convert(4.0) == pytest.approx(3.0)   # inside the removed stretch
        assert convert(6.0) == pytest.approx(4.0)
        assert convert(99.0) == pytest.approx(8.0)


class TestSourceSegments:
    def test_a_pristine_clip_shifts_by_its_start(self):
        assert pro_mode.source_segments(
            [[0.3, 4.0], [5.0, 9.0]], [{"start": 100.0, "end": 110.0}]) == [
                {"start": 100.3, "end": 104.0}, {"start": 105.0, "end": 109.0}]

    def test_a_kept_stretch_over_a_previous_cut_is_split_there(self):
        # The clip is already two stretches of the source back to back.
        segments = [{"start": 10.0, "end": 14.0}, {"start": 30.0, "end": 36.0}]
        assert pro_mode.source_segments([[2.0, 6.0]], segments) == [
            {"start": 12.0, "end": 14.0}, {"start": 30.0, "end": 32.0}]

    def test_contiguous_pieces_stay_one_segment(self):
        assert pro_mode.source_segments(
            [[0.0, 2.0], [2.0, 4.0]], [{"start": 5.0, "end": 9.0}]) == [
                {"start": 5.0, "end": 9.0}]


class TestShots:
    def _faces(self, seconds, count=1, height=300.0):
        return [(count, 540.0, 600.0, height)] * int(seconds * pro_mode.TRACK_FPS)

    def test_every_cut_changes_the_scale(self):
        frames = [(0, 90), (150, 240), (300, 390)]
        shots, total = pro_mode.plan_shots(frames, FPS, [], self._faces(13), 1920)
        assert total == 270
        assert [s["f0"] for s in shots] == [0, 90, 180]
        scales = [s["z"] for s in shots]
        assert all(a != b for a, b in zip(scales, scales[1:]))

    def test_a_long_take_is_split_at_a_sentence_start(self):
        words = [{"w": "one.", "s": 2.6, "e": 3.0}, {"w": "Two", "s": 3.1, "e": 3.4}]
        shots, _ = pro_mode.plan_shots([(0, 240)], FPS, words, self._faces(8), 1920)
        assert [s["f0"] for s in shots] == [0, 93]

    def test_no_face_no_reframing(self):
        shots, _ = pro_mode.plan_shots([(0, 90)], FPS, [], [None] * 15, 1920)
        assert shots[0]["anchor"] is None
        assert pro_mode.zoom_at(shots[0], 45, FPS) == 1.0

    def test_two_faces_are_left_as_framed(self):
        shots, _ = pro_mode.plan_shots([(0, 90)], FPS, [], self._faces(3, count=2), 1920)
        assert shots[0]["anchor"] is None

    def test_a_stretch_the_render_did_not_track_is_left_alone(self):
        shots, _ = pro_mode.plan_shots([(0, 90)], FPS, [], self._faces(3), 1920,
                                       non_track=[(1.0, 2.0)])
        assert shots[0]["anchor"] is None

    def test_the_face_never_fills_more_than_its_share_of_the_frame(self):
        shots, _ = pro_mode.plan_shots([(0, 300)], FPS, [], self._faces(10, height=500.0), 1920)
        for shot in shots:
            for frame in (shot["f0"], shot["f1"] - 1):
                zoom = pro_mode.zoom_at(shot, frame, FPS)
                assert 500.0 * zoom <= (pro_mode.FACE_HEIGHT_MAX * 1920
                                        + pro_mode.PUSH_MAX * 500.0 + 1)

    def test_the_frame_stays_inside_the_picture_sideways(self):
        for cx in (0.0, 540.0, 1080.0):
            m = pro_mode.frame_matrix((cx, 200.0), 1.5, 1080, 1920)
            assert m[0][2] <= 0 and m[0][2] >= (1 - 1.5) * 1080

    def test_the_track_does_not_smooth_across_a_camera_change(self):
        faces = [(1, 200.0, 600.0, 300.0)] * 10 + [(1, 900.0, 600.0, 300.0)] * 10
        track = pro_mode.smooth_track(faces, 1080)
        assert pro_mode.track_at(track, 1.0)[0] == pytest.approx(200.0)
        assert pro_mode.track_at(track, 3.0)[0] == pytest.approx(900.0)
        # The last sample before the change does not drift towards the next.
        assert pro_mode.track_at(track, 1.9)[0] == pytest.approx(200.0)


class TestSlots:
    def test_a_taken_slot_is_busy_then_free_again(self, tmp_path):
        pytest.importorskip("fcntl")
        with pro_mode.slot(slots=1, lock_dir=str(tmp_path)):
            with pytest.raises(pro_mode.Busy):
                with pro_mode.slot(wait=0, slots=1, lock_dir=str(tmp_path)):
                    pass
        with pro_mode.slot(wait=0, slots=1, lock_dir=str(tmp_path)):
            pass

    def test_zero_slots_means_no_cap(self, tmp_path):
        with pro_mode.slot(slots=0, lock_dir=str(tmp_path)):
            with pro_mode.slot(slots=0, lock_dir=str(tmp_path)):
                pass


TRANSCRIPT = {"language": "en", "segments": [{
    "start": 0.0, "end": 60.0, "text": "hello world again",
    "words": [{"word": "hello", "start": 12.0, "end": 12.5},
              {"word": "world", "start": 20.0, "end": 20.4},
              {"word": "again", "start": 50.0, "end": 50.5}]}]}


class TestConvert:
    @pytest.fixture()
    def rendered(self, monkeypatch):
        calls = []

        def fake_run(clip_path, out_path, words=(), ranges=(), wait=0.0):
            calls.append({"clip": clip_path, "words": list(words), "wait": wait})
            with open(out_path, "wb") as f:
                f.write(b"pro")
            return {"keep": [[0.5, 10.0], [12.0, 29.0]], "duration": 26.5,
                    "shots": 2, "framed": 2}
        monkeypatch.setattr(pro_mode, "run", fake_run)
        return calls

    def test_the_pro_file_is_a_recut_with_its_recipe(self, tmp_path, rendered):
        base = tmp_path / "t_clip_1.mp4"
        base.write_bytes(b"clean")
        clip = {"start": 10.0, "end": 40.0, "caption_words": [{"word": " x"}],
                "layout_ranges": [{"start": 0.0, "end": 30.0, "layout": "track"}]}
        name, updates = pro_mode.convert(str(base), "t_clip_1.mp4", clip, TRANSCRIPT, wait=7)

        assert name.startswith("recut_") and name.endswith("_t_clip_1.mp4")
        assert os.path.exists(tmp_path / name)
        assert updates["mode"] == "pro"
        assert updates["recipe"] == {
            "v": 1, "canonical_range": {"start": 10.0, "end": 40.0},
            "segments": [{"start": 10.5, "end": 20.0}, {"start": 22.0, "end": 39.0}]}
        assert (updates["start"], updates["end"]) == (10.5, 39.0)
        # The layouts follow the cut, on the new timeline.
        assert updates["layout_ranges"] == [
            {"start": 0.0, "end": 9.5, "layout": "track"},
            {"start": 9.5, "end": 26.5, "layout": "track"}]
        assert updates["caption_words"] is None
        # The render got the clip's words on the clip's own timeline.
        assert rendered[0]["words"] == [{"w": "hello", "s": 2.0, "e": 2.5},
                                        {"w": "world", "s": 10.0, "e": 10.4}]
        assert rendered[0]["wait"] == 7

    def test_going_back_restores_what_the_conversion_replaced(self, tmp_path, rendered):
        base = tmp_path / "t_clip_1.mp4"
        base.write_bytes(b"clean")
        clip = {"start": 10.0, "end": 40.0, "caption_words": [{"word": " x"}]}
        original = dict(clip)
        _name, updates = pro_mode.convert(str(base), "t_clip_1.mp4", clip, TRANSCRIPT)
        assert updates["pro_before"]["base"] == "t_clip_1.mp4"
        clip.update(updates)
        clip.update(pro_mode.restore(clip))
        assert clip["mode"] == "normal" and clip["pro_before"] is None
        assert (clip["start"], clip["end"]) == (10.0, 40.0)
        assert clip["caption_words"] == original["caption_words"]
        assert not clip["recipe"]

    def test_a_clip_already_recut_keeps_its_own_cut_underneath(self, tmp_path, rendered):
        base = tmp_path / "recut_1_abc_t_clip_1.mp4"
        base.write_bytes(b"clean")
        recipe = {"v": 1, "framing": "track",
                  "canonical_range": {"start": 10.0, "end": 40.0},
                  "segments": [{"start": 10.0, "end": 20.0}, {"start": 25.0, "end": 45.0}]}
        clip = {"start": 10.0, "end": 45.0, "recipe": recipe}
        _name, updates = pro_mode.convert(str(base), "t_clip_1.mp4", clip, TRANSCRIPT)
        assert updates["recipe"]["segments"] == [
            {"start": 10.5, "end": 20.0}, {"start": 27.0, "end": 44.0}]
        assert updates["recipe"]["framing"] == "track"
        assert updates["recipe"]["canonical_range"] == {"start": 10.0, "end": 40.0}
        assert updates["pro_before"]["recipe"] == recipe
